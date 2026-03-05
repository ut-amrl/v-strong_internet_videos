"""Backbone adapters for V-STRONG."""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


_SIZE_ALIASES = {
    None: "small",
    "s": "small",
    "small": "small",
    "m": "medium",
    "med": "medium",
    "medium": "medium",
    "l": "large",
    "large": "large",
    "xl": "xlarge",
    "xlarge": "xlarge",
}

_BACKBONE_SPECS = {
    "sam": {
        "default_img_size": 1024,
        "sizes": {
            "small": "vit_b",
            "medium": "vit_l",
            "large": "vit_h",
        },
    },
    "nanosam": {
        "default_img_size": 1024,
        "sizes": {
            "small": "resnet18",
            "medium": "resnet34",
            "large": "resnet50",
            "xlarge": "resnet101",
        },
    },
    "dino": {
        "default_img_size": 224,
        "sizes": {
            "small": "dino_vits16",
            "medium": "dino_vitb16",
            "large": "dino_vitb8",
        },
    },
    "dinov2": {
        "default_img_size": 518,
        "sizes": {
            "small": "dinov2_vits14",
            "medium": "dinov2_vitb14",
            "large": "dinov2_vitl14",
        },
    },
}

_DINO_VARIANTS = {
    "dino_vits16": {"embed_dim": 384, "patch_size": 16},
    "dino_vits8": {"embed_dim": 384, "patch_size": 8},
    "dino_vitb16": {"embed_dim": 768, "patch_size": 16},
    "dino_vitb8": {"embed_dim": 768, "patch_size": 8},
}

_DINOV2_VARIANTS = {
    "dinov2_vits14": {"embed_dim": 384, "patch_size": 14},
    "dinov2_vitb14": {"embed_dim": 768, "patch_size": 14},
    "dinov2_vitl14": {"embed_dim": 1024, "patch_size": 14},
    "dinov2_vitg14": {"embed_dim": 1536, "patch_size": 14},
}

def _get_nanosam_variants() -> dict:
    # Import torchvision lazily so SAM-only workflows don't require torchvision (and
    # don't trip torchvision->onnx->transformers import chains in mismatched envs).
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

    return {
        "resnet18": {"builder": resnet18, "weights": ResNet18_Weights.IMAGENET1K_V1, "embed_dim": 512},
        "resnet34": {"builder": resnet34, "weights": ResNet34_Weights.IMAGENET1K_V1, "embed_dim": 512},
        "resnet50": {"builder": resnet50, "weights": ResNet50_Weights.IMAGENET1K_V2, "embed_dim": 2048},
        "resnet101": {"builder": resnet101, "weights": ResNet101_Weights.IMAGENET1K_V2, "embed_dim": 2048},
    }


def _normalize_backbone_type(backbone_type: str | None) -> str:
    value = str(backbone_type or "sam").strip().lower()
    if value not in _BACKBONE_SPECS:
        raise ValueError(f"Unsupported backbone_type={backbone_type!r}. Expected one of {sorted(_BACKBONE_SPECS)}")
    return value


def _normalize_backbone_size(backbone_size: str | None) -> str:
    value = str(backbone_size).strip().lower() if backbone_size is not None else None
    if value not in _SIZE_ALIASES:
        raise ValueError(f"Unsupported backbone_size={backbone_size!r}. Expected small, medium, or large.")
    return _SIZE_ALIASES[value]


def resolve_backbone_config(
    backbone_type: str = "sam",
    backbone_size: str | None = None,
    backbone_variant: str | None = None,
    img_size: int | None = None,
) -> dict:
    """Resolve user-facing backbone settings into a concrete variant."""

    resolved_type = _normalize_backbone_type(backbone_type)

    if backbone_variant is not None:
        variant_as_size = str(backbone_variant).strip().lower()
        if variant_as_size in _SIZE_ALIASES and backbone_size is None:
            backbone_size = variant_as_size
            backbone_variant = None

    resolved_size = _normalize_backbone_size(backbone_size)
    spec = _BACKBONE_SPECS[resolved_type]
    resolved_variant = str(backbone_variant or spec["sizes"][resolved_size]).strip()
    resolved_img_size = int(spec["default_img_size"] if img_size is None else img_size)

    return {
        "backbone_type": resolved_type,
        "backbone_size": resolved_size,
        "backbone_variant": resolved_variant,
        "img_size": resolved_img_size,
    }


class BaseBackbone(nn.Module, ABC):
    """Frozen foundation model that emits dense feature maps."""

    feature_dim: int
    img_size: int

    @abstractmethod
    def encode(self, resized_rgbs: list[np.ndarray]) -> torch.Tensor:
        """Encode resized RGB images into a dense feature map `(B, C, H, W)`."""

    @property
    def device(self) -> torch.device:
        param = next(self.parameters(), None)
        if param is not None:
            return param.device
        buffer = next(self.buffers(), None)
        if buffer is not None:
            return buffer.device
        return torch.device("cpu")


class SAMBackbone(BaseBackbone):
    """Frozen SAM image encoder."""

    def __init__(self, checkpoint: str, variant: str, img_size: int, trainable: bool = False):
        super().__init__()
        if not checkpoint:
            raise ValueError("SAM requires a backbone checkpoint path.")

        self.img_size = int(img_size)
        self.feature_dim = 256
        self.trainable = bool(trainable)

        sys.path.insert(
            0,
            str(Path(__file__).resolve().parents[2] / "third_party" / "segment-anything"),
        )
        from segment_anything import sam_model_registry

        self.sam = sam_model_registry[variant](checkpoint=checkpoint)
        for param in self.sam.parameters():
            param.requires_grad = self.trainable

    def encode(self, resized_rgbs: list[np.ndarray]) -> torch.Tensor:
        with torch.set_grad_enabled(self.trainable):
            inputs = []
            device = self.device
            for rgb in resized_rgbs:
                tensor = torch.from_numpy(rgb).to(device=device)
                tensor = tensor.permute(2, 0, 1).contiguous().to(torch.float32)
                tensor = self.sam.preprocess(tensor)
                inputs.append(tensor)
            batch = torch.stack(inputs, dim=0)
            return self.sam.image_encoder(batch)


def _extract_state_dict(payload) -> dict:
    if not isinstance(payload, dict):
        return payload

    for key in ("state_dict", "model", "model_state_dict"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return payload


def _normalize_state_dict_keys(state_dict: dict, known_prefixes: tuple[str, ...]) -> dict:
    if not isinstance(state_dict, dict):
        return state_dict

    for prefix in known_prefixes:
        if any(key.startswith(prefix) for key in state_dict):
            normalized = {}
            for key, value in state_dict.items():
                normalized[key[len(prefix):] if key.startswith(prefix) else key] = value
            return normalized
    return state_dict


def _load_optional_checkpoint(model: nn.Module, checkpoint_path: str | None):
    if checkpoint_path is None:
        return

    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise RuntimeError(f"backbone_checkpoint does not exist: {checkpoint_path}")

    payload = torch.load(str(ckpt), map_location="cpu")
    state_dict = _extract_state_dict(payload)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        missing_str = ", ".join(missing[:5]) if missing else "-"
        unexpected_str = ", ".join(unexpected[:5]) if unexpected else "-"
        raise RuntimeError(
            "Failed to load backbone checkpoint cleanly. "
            f"Missing keys: {missing_str}; unexpected keys: {unexpected_str}"
        )


class NanoSAMBackbone(BaseBackbone):
    """ResNet-based frozen encoder for NanoSAM-style training."""

    def __init__(self, variant: str, img_size: int, checkpoint: str | None = None, trainable: bool = False):
        super().__init__()
        variants = _get_nanosam_variants()
        if variant not in variants:
            raise ValueError(f"Unsupported NanoSAM variant={variant!r}. Expected one of {sorted(variants)}")

        info = variants[variant]
        self.img_size = int(img_size)
        self.feature_dim = int(info["embed_dim"])
        self.trainable = bool(trainable)
        # Distillation targets SAM's 64x64 grid at 1024 input size (stride 16).
        # Keep NanoSAM's output on the same spatial grid to avoid coarse 32x32 artifacts.
        self.target_hw = max(1, self.img_size // 16)

        if checkpoint is None:
            backbone_model = info["builder"](weights=info["weights"])
        else:
            ckpt_path = Path(checkpoint)
            if not ckpt_path.exists():
                import os
                cwd = os.getcwd()
                raise RuntimeError(f"backbone_checkpoint does not exist: {checkpoint}. CWD={cwd}, Absolute={ckpt_path.absolute()}")
            backbone_model = info["builder"](weights=None)
            payload = torch.load(str(ckpt_path), map_location="cpu")
            state_dict = _extract_state_dict(payload)
            state_dict = _normalize_state_dict_keys(
                state_dict,
                known_prefixes=("image_encoder.", "backbone.", "model.", "encoder."),
            )
            missing, unexpected = backbone_model.load_state_dict(state_dict, strict=False)
            # fc.* keys are expected to be absent — we never use the classification head.
            missing = [k for k in missing if not k.startswith("fc.")]
            if missing:
                missing_str = ", ".join(missing[:5])
                raise RuntimeError(f"Failed to load NanoSAM checkpoint cleanly. Missing keys: {missing_str}")
            if unexpected:
                print(
                    "Warning: NanoSAM checkpoint had unexpected keys; ignoring extras: "
                    + ", ".join(unexpected[:5])
                )

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

        for param in self.parameters():
            param.requires_grad = self.trainable

        self.register_buffer("pixel_mean", torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1))

    def encode(self, resized_rgbs: list[np.ndarray]) -> torch.Tensor:
        with torch.set_grad_enabled(self.trainable):
            inputs = []
            device = self.device
            for rgb in resized_rgbs:
                tensor = torch.from_numpy(rgb).to(device=device).permute(2, 0, 1).to(torch.float32) / 255.0
                tensor = (tensor - self.pixel_mean) / self.pixel_std
                inputs.append(tensor)

            batch = torch.stack(inputs, dim=0)
            x = self.stem(batch)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            if x.shape[-2:] != (self.target_hw, self.target_hw):
                x = F.interpolate(
                    x,
                    size=(self.target_hw, self.target_hw),
                    mode="bilinear",
                    align_corners=False,
                )
            return x


class DINOBackbone(BaseBackbone):
    """Frozen DINO v1 backbone."""

    def __init__(self, variant: str, img_size: int, checkpoint: str | None = None, trainable: bool = False):
        super().__init__()
        if variant not in _DINO_VARIANTS:
            raise ValueError(f"Unsupported DINO variant={variant!r}. Expected one of {sorted(_DINO_VARIANTS)}")

        info = _DINO_VARIANTS[variant]
        self.feature_dim = int(info["embed_dim"])
        self.patch_size = int(info["patch_size"])
        self.img_size = int(img_size)
        self.trainable = bool(trainable)
        if self.img_size % self.patch_size != 0:
            raise ValueError(f"img_size={self.img_size} must be divisible by patch_size={self.patch_size}")

        self.model = torch.hub.load("facebookresearch/dino:main", variant, verbose=False)
        _load_optional_checkpoint(self.model, checkpoint)
        for param in self.model.parameters():
            param.requires_grad = self.trainable

        self.register_buffer("pixel_mean", torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1))

    def encode(self, resized_rgbs: list[np.ndarray]) -> torch.Tensor:
        with torch.set_grad_enabled(self.trainable):
            inputs = []
            device = self.device
            for rgb in resized_rgbs:
                tensor = torch.from_numpy(rgb).to(device=device).permute(2, 0, 1).to(torch.float32) / 255.0
                tensor = (tensor - self.pixel_mean) / self.pixel_std
                inputs.append(tensor)

            batch = torch.stack(inputs, dim=0)
            features = self.model.get_intermediate_layers(batch, n=1)[0]
            patch_tokens = features[:, 1:]
            feat_h = self.img_size // self.patch_size
            feat_w = self.img_size // self.patch_size
            return patch_tokens.reshape(-1, feat_h, feat_w, self.feature_dim).permute(0, 3, 1, 2).contiguous()


class DINOv2Backbone(BaseBackbone):
    """Frozen DINOv2 backbone."""

    def __init__(self, variant: str, img_size: int, checkpoint: str | None = None, trainable: bool = False):
        super().__init__()
        if variant not in _DINOV2_VARIANTS:
            raise ValueError(f"Unsupported DINOv2 variant={variant!r}. Expected one of {sorted(_DINOV2_VARIANTS)}")

        info = _DINOV2_VARIANTS[variant]
        self.feature_dim = int(info["embed_dim"])
        self.patch_size = int(info["patch_size"])
        self.img_size = int(img_size)
        self.trainable = bool(trainable)
        if self.img_size % self.patch_size != 0:
            raise ValueError(f"img_size={self.img_size} must be divisible by patch_size={self.patch_size}")

        self.model = torch.hub.load("facebookresearch/dinov2", variant, verbose=False)
        _load_optional_checkpoint(self.model, checkpoint)
        for param in self.model.parameters():
            param.requires_grad = self.trainable

        self.register_buffer("pixel_mean", torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1))

    def encode(self, resized_rgbs: list[np.ndarray]) -> torch.Tensor:
        with torch.set_grad_enabled(self.trainable):
            inputs = []
            device = self.device
            for rgb in resized_rgbs:
                tensor = torch.from_numpy(rgb).to(device=device).permute(2, 0, 1).to(torch.float32) / 255.0
                tensor = (tensor - self.pixel_mean) / self.pixel_std
                inputs.append(tensor)

            batch = torch.stack(inputs, dim=0)
            return self.model.get_intermediate_layers(batch, n=1, reshape=True)[0]


def build_backbone(
    backbone_type: str = "sam",
    backbone_size: str | None = None,
    backbone_variant: str | None = None,
    backbone_checkpoint: str | None = None,
    img_size: int | None = None,
    trainable: bool = False,
) -> BaseBackbone:
    """Construct the configured frozen backbone."""

    resolved = resolve_backbone_config(
        backbone_type=backbone_type,
        backbone_size=backbone_size,
        backbone_variant=backbone_variant,
        img_size=img_size,
    )

    if resolved["backbone_type"] == "sam":
        return SAMBackbone(
            checkpoint=backbone_checkpoint or "",
            variant=resolved["backbone_variant"],
            img_size=resolved["img_size"],
            trainable=trainable,
        )
    if resolved["backbone_type"] == "nanosam":
        return NanoSAMBackbone(
            variant=resolved["backbone_variant"],
            img_size=resolved["img_size"],
            checkpoint=backbone_checkpoint,
            trainable=trainable,
        )
    if resolved["backbone_type"] == "dino":
        return DINOBackbone(
            variant=resolved["backbone_variant"],
            img_size=resolved["img_size"],
            checkpoint=backbone_checkpoint,
            trainable=trainable,
        )
    return DINOv2Backbone(
        variant=resolved["backbone_variant"],
        img_size=resolved["img_size"],
        checkpoint=backbone_checkpoint,
        trainable=trainable,
    )
