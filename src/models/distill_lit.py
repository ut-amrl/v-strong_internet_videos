"""SAM → ResNet knowledge distillation module.

The teacher (frozen SAM image encoder) generates 64×64×256 feature maps.
The student (ResNet + projection neck) is trained end-to-end to reproduce them.

After training the student weights are saved and can be loaded by NanoSAMBackbone
in the main VStrongLit pipeline for fast traversability inference.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
)


# ──────────────────────────────────────────────────────────────────────────────
# Student variants
# ──────────────────────────────────────────────────────────────────────────────

_STUDENT_VARIANTS: dict[str, dict] = {
    "resnet18": {
        "builder": resnet18,
        "weights": ResNet18_Weights.IMAGENET1K_V1,
        "backbone_channels": 512,
    },
    "resnet34": {
        "builder": resnet34,
        "weights": ResNet34_Weights.IMAGENET1K_V1,
        "backbone_channels": 512,
    },
    "resnet50": {
        "builder": resnet50,
        "weights": ResNet50_Weights.IMAGENET1K_V2,
        "backbone_channels": 2048,
    },
    "resnet101": {
        "builder": resnet101,
        "weights": ResNet101_Weights.IMAGENET1K_V2,
        "backbone_channels": 2048,
    },
}

# SAM always outputs 256-channel embeddings at 64×64 for a 1024×1024 input.
_SAM_EMBED_DIM = 256


# ──────────────────────────────────────────────────────────────────────────────
# Student model: ResNet trunk + projection neck
# ──────────────────────────────────────────────────────────────────────────────

class ResNetStudentEncoder(nn.Module):
    """ResNet backbone with a 1×1 conv neck that maps to SAM's embedding space.

    Output: (B, 256, 64, 64) — same spatial/channel layout as SAM ViT-H.
    """

    def __init__(
        self,
        variant: str = "resnet18",
        img_size: int = 1024,
        pretrained: bool = True,
    ):
        super().__init__()
        if variant not in _STUDENT_VARIANTS:
            raise ValueError(
                f"Unknown student variant {variant!r}. "
                f"Valid options: {sorted(_STUDENT_VARIANTS)}"
            )

        info = _STUDENT_VARIANTS[variant]
        weights = info["weights"] if pretrained else None
        backbone_model = info["builder"](weights=weights)
        in_channels = info["backbone_channels"]

        # Strip classification head; keep convolutional feature extractor.
        self.stem = nn.Sequential(
            backbone_model.conv1,
            backbone_model.bn1,
            backbone_model.relu,
            backbone_model.maxpool,
        )
        self.layer1 = backbone_model.layer1
        self.layer2 = backbone_model.layer2
        self.layer3 = backbone_model.layer3
        self.layer4 = backbone_model.layer4

        # Projection neck: map backbone channels → SAM embed dim (256).
        self.neck = nn.Sequential(
            nn.Conv2d(in_channels, _SAM_EMBED_DIM, kernel_size=1, bias=False),
            nn.BatchNorm2d(_SAM_EMBED_DIM),
            nn.ReLU(inplace=True),
        )

        # Target spatial size that SAM produces for a 1024-px input.
        self.img_size = int(img_size)
        self.target_hw = img_size // 16  # 64 for 1024-px input.

        self.register_buffer(
            "pixel_mean", torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        )
        self.register_buffer(
            "pixel_std", torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        )

    # ------------------------------------------------------------------
    def _preprocess(self, resized_rgbs: list[np.ndarray]) -> torch.Tensor:
        """Stack a list of H×W×3 uint8 images into a padded (B,3,img_size,img_size) tensor."""
        device = self.pixel_mean.device
        tensors = []
        for rgb in resized_rgbs:
            t = (
                torch.from_numpy(rgb)
                .to(device=device)
                .permute(2, 0, 1)
                .to(torch.float32)
                / 255.0
            )
            t = (t - self.pixel_mean) / self.pixel_std
            padded = torch.zeros(
                (3, self.img_size, self.img_size), dtype=t.dtype, device=device
            )
            _, h, w = t.shape
            padded[:, :h, :w] = t
            tensors.append(padded)
        return torch.stack(tensors, dim=0)  # (B, 3, img_size, img_size)

    def forward_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Raw forward pass on a (B, 3, H, W) tensor already on device."""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # (B, backbone_channels, ~32, ~32) for 1024-px input
        x = self.neck(x)    # (B, 256, ~32, ~32)

        # Upsample to the exact target resolution (64×64).
        if x.shape[-2:] != (self.target_hw, self.target_hw):
            x = F.interpolate(
                x,
                size=(self.target_hw, self.target_hw),
                mode="bilinear",
                align_corners=False,
            )
        return x  # (B, 256, 64, 64)

    def forward(self, resized_rgbs: list[np.ndarray]) -> torch.Tensor:
        """Accept a list of H×W×3 uint8 numpy arrays, return (B, 256, 64, 64)."""
        batch = self._preprocess(resized_rgbs)
        return self.forward_tensor(batch)

    def export_backbone_state_dict(self) -> dict:
        """Return a state dict containing only the ResNet layers (without the neck).

        This translates our `stem.X` keys back to the standard torchvision `conv1`/`bn1`
        names so that `NanoSAMBackbone` can load them seamlessly.
        """
        sd = self.state_dict()
        out = {}
        for k, v in sd.items():
            if k.startswith("stem.0."):
                out[k.replace("stem.0.", "conv1.")] = v
            elif k.startswith("stem.1."):
                out[k.replace("stem.1.", "bn1.")] = v
            elif any(k.startswith(p) for p in ("layer1.", "layer2.", "layer3.", "layer4.")):
                out[k] = v
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Teacher wrapper: frozen SAM image encoder
# ──────────────────────────────────────────────────────────────────────────────

class SAMTeacherEncoder(nn.Module):
    """Thin wrapper around a frozen SAM image encoder."""

    def __init__(self, sam_checkpoint: str, sam_variant: str, img_size: int):
        super().__init__()
        sys.path.insert(
            0,
            str(Path(__file__).resolve().parents[2] / "third_party" / "segment-anything"),
        )
        from segment_anything import sam_model_registry

        sam = sam_model_registry[sam_variant](checkpoint=sam_checkpoint)
        # Store the full sam object as a registered submodule so that
        # pixel_mean / pixel_std buffers (used by sam.preprocess) are
        # moved to the correct device by Lightning alongside image_encoder.
        self._sam = sam
        self.img_size = int(img_size)

        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, resized_rgbs: list[np.ndarray]) -> torch.Tensor:
        device = next(self.parameters()).device
        inputs = []
        for rgb in resized_rgbs:
            t = (
                torch.from_numpy(rgb)
                .to(device=device)
                .permute(2, 0, 1)
                .to(torch.float32)
            )
            t = self._sam.preprocess(t)  # pixel_mean/std now on same device
            inputs.append(t)
        batch = torch.stack(inputs, dim=0)
        return self._sam.image_encoder(batch)  # (B, 256, 64, 64)


# ──────────────────────────────────────────────────────────────────────────────
# Lightning distillation module
# ──────────────────────────────────────────────────────────────────────────────

class DistillLit(pl.LightningModule):
    """Knowledge-distillation of a SAM teacher into a ResNet student.

    Loss = λ_mse * MSE(student, teacher) + λ_cos * (1 - cos_sim(student, teacher))

    Both terms operate on the whole (B,256,64,64) feature map.
    """

    def __init__(
        self,
        # Teacher
        sam_checkpoint: str,
        sam_variant: str = "vit_h",
        # Student
        student_variant: str = "resnet18",
        student_pretrained: bool = True,
        # Training
        img_size: int = 1024,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        lambda_mse: float = 1.0,
        lambda_cos: float = 0.5,
        # Logging
        log_images_every_n_steps: int = 500,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.img_size = int(img_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.lambda_mse = float(lambda_mse)
        self.lambda_cos = float(lambda_cos)
        self.log_images_every_n_steps = int(log_images_every_n_steps)

        # Teacher — frozen
        self.teacher = SAMTeacherEncoder(
            sam_checkpoint=sam_checkpoint,
            sam_variant=sam_variant,
            img_size=img_size,
        )
        # Student — trainable
        self.student = ResNetStudentEncoder(
            variant=student_variant,
            img_size=img_size,
            pretrained=student_pretrained,
        )

    # ------------------------------------------------------------------
    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.student.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def transfer_batch_to_device(self, batch, device, dataloader_idx=0):
        # Batches are plain dicts / lists of numpy arrays; handle manually.
        return batch

    # ------------------------------------------------------------------
    def _distill_loss(
        self, student_feat: torch.Tensor, teacher_feat: torch.Tensor
    ) -> torch.Tensor:
        """Pixel-wise MSE + (1 − cosine-similarity) on flattened spatial dims."""
        mse = F.mse_loss(student_feat, teacher_feat)
        # Cosine similarity per spatial location — shape (B, 64*64).
        b, c, h, w = student_feat.shape
        s_flat = student_feat.permute(0, 2, 3, 1).reshape(b * h * w, c)
        t_flat = teacher_feat.permute(0, 2, 3, 1).reshape(b * h * w, c)
        cos_sim = F.cosine_similarity(s_flat, t_flat, dim=1).mean()
        loss = self.lambda_mse * mse + self.lambda_cos * (1.0 - cos_sim)
        return loss, mse, cos_sim

    # ------------------------------------------------------------------
    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        # batch is a list of images (numpy uint8 H×W×3)
        resized_rgbs: list[np.ndarray] = batch

        with torch.no_grad():
            teacher_feat = self.teacher(resized_rgbs)  # (B, 256, 64, 64)

        student_feat = self.student(resized_rgbs)  # (B, 256, 64, 64)

        loss, mse, cos_sim = self._distill_loss(student_feat, teacher_feat)

        bs = len(resized_rgbs)
        self.log(f"{stage}/loss",    loss,    prog_bar=True, on_step=True, on_epoch=True, batch_size=bs)
        self.log(f"{stage}/mse",     mse,     on_step=False, on_epoch=True, batch_size=bs)
        self.log(f"{stage}/cos_sim", cos_sim, on_step=False, on_epoch=True, batch_size=bs)
        return loss

    def training_step(self, batch, batch_idx: int):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx: int):
        self._shared_step(batch, "val")
