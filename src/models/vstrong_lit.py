import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "third_party" / "segment-anything"))
from segment_anything import sam_model_registry


@dataclass(frozen=True)
class VStrongViz:
    rgb: np.ndarray  # HxWx3 uint8 (resized)
    heatmap_bgr: np.ndarray  # HxWx3 uint8
    overlay_bgr: np.ndarray  # HxWx3 uint8


def supervised_contrastive_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Supervised contrastive loss (SupCon).

    z: (N, D) float
    labels: (N,) int
    """
    if z.ndim != 2:
        raise ValueError(f"z must be (N,D), got {tuple(z.shape)}")
    if labels.ndim != 1 or labels.shape[0] != z.shape[0]:
        raise ValueError("labels must be (N,) matching z")
    if z.shape[0] < 2:
        return torch.zeros((), device=z.device, dtype=z.dtype)

    z = F.normalize(z, dim=1)
    logits = (z @ z.t()) / float(temperature)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    n = z.shape[0]
    self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
    logits_mask = ~self_mask

    labels = labels.to(torch.long)
    pos_mask = (labels[:, None] == labels[None, :]) & logits_mask

    exp_logits = torch.exp(logits) * logits_mask.float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))

    pos_count = pos_mask.sum(dim=1)  # (N,)
    valid = pos_count > 0
    if valid.sum() == 0:
        return torch.zeros((), device=z.device, dtype=z.dtype)

    mean_log_prob_pos = (log_prob * pos_mask.float()).sum(dim=1) / pos_count.clamp_min(1)
    loss = -(mean_log_prob_pos[valid]).mean()
    return loss


def _points_to_grid(
    points_xy: torch.Tensor,
    map_h: int,
    map_w: int,
    sam_img_size: int = 1024,
) -> torch.Tensor:
    """
    Convert points in resized-image pixel coordinates (x,y) into grid_sample coords for a feature map.
    """
    pts = points_xy.to(torch.float32)
    pts[..., 0] = pts[..., 0].clamp(0, sam_img_size - 1)
    pts[..., 1] = pts[..., 1].clamp(0, sam_img_size - 1)

    x = pts[..., 0] * (map_w / float(sam_img_size))
    y = pts[..., 1] * (map_h / float(sam_img_size))

    x = x.clamp(0, map_w - 1)
    y = y.clamp(0, map_h - 1)

    x_norm = (x / float(map_w - 1)) * 2.0 - 1.0
    y_norm = (y / float(map_h - 1)) * 2.0 - 1.0
    return torch.stack([x_norm, y_norm], dim=-1)


def sample_from_feature_map(
    feat_map: torch.Tensor,
    points_xy: torch.Tensor,
    sam_img_size: int = 1024,
) -> torch.Tensor:
    """
    feat_map: (B, C, Hm, Wm)
    points_xy: (B, K, 2) in resized coords (x,y) where image was resized with longest side=1024
    returns: (B, K, C)
    """
    b, c, hm, wm = feat_map.shape
    grid = _points_to_grid(points_xy, map_h=hm, map_w=wm, sam_img_size=sam_img_size)  # (B,K,2)
    grid = grid.unsqueeze(2)  # (B,K,1,2)
    sampled = F.grid_sample(feat_map, grid, mode="bilinear", align_corners=True)  # (B,C,K,1)
    sampled = sampled.squeeze(-1).permute(0, 2, 1).contiguous()  # (B,K,C)
    return sampled


class VStrongLit(pl.LightningModule):
    """
    Minimal V-STRONG-style contrastive training:
      - Frozen SAM image encoder provides dense features.
      - 1x1 conv projection head learns a contrastive embedding.
      - Supervised contrastive loss clusters pos points and neg points.
      - Visualization: cosine similarity to a traversability reference vector.
    """

    def __init__(
        self,
        sam_checkpoint: str,
        sam_type: str = "vit_b",
        sam_img_size: int = 1024,
        embed_dim: int = 64,
        proj_hidden: int = 256,
        temperature: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        traversability_ema_alpha: float = 0.999,
        log_images_every_n_steps: int = 200,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.sam_img_size = int(sam_img_size)
        self.temperature = float(temperature)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.traversability_ema_alpha = float(traversability_ema_alpha)
        self.log_images_every_n_steps = int(log_images_every_n_steps)

        sam = sam_model_registry[sam_type](checkpoint=sam_checkpoint)
        sam.eval()
        for p in sam.parameters():
            p.requires_grad = False
        self.sam = sam

        # SAM image encoder output is 256 channels for v1 models (after the neck).
        in_ch = 256
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, int(proj_hidden), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(proj_hidden), int(embed_dim), kernel_size=1),
        )
        self.register_buffer("traversability_vector", torch.zeros(int(embed_dim), dtype=torch.float32))
        self.register_buffer("traversability_initialized", torch.tensor(False, dtype=torch.bool))

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.proj.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def transfer_batch_to_device(self, batch, device, dataloader_idx=0):
        # Keep batch as Python objects (np arrays), we handle device transfer manually.
        return batch

    def _encode_images(self, resized_rgbs: list[np.ndarray]) -> torch.Tensor:
        """
        resized_rgbs: list of HxWx3 uint8 RGB images already resized by ResizeLongestSide(1024)
        returns: (B, 256, 64, 64) SAM image embeddings
        """
        imgs = []
        for rgb in resized_rgbs:
            t = torch.from_numpy(rgb).to(self.device)
            t = t.permute(2, 0, 1).contiguous().to(torch.float32)  # 3xHxW in [0,255]
            t = self.sam.preprocess(t)  # normalize + pad to 1024 square
            imgs.append(t)

        inp = torch.stack(imgs, dim=0)
        with torch.no_grad():
            emb = self.sam.image_encoder(inp)
        return emb

    def _make_viz(
        self,
        resized_rgb: np.ndarray,
        z_map: torch.Tensor,  # (D, Hm, Wm)
        pos_points: torch.Tensor,  # (K,2) resized coords
    ) -> VStrongViz | None:
        if pos_points.numel() == 0:
            return None

        z_map = F.normalize(z_map, dim=0)
        pos_points_b = pos_points.unsqueeze(0)  # 1xKx2
        z_pos = sample_from_feature_map(z_map.unsqueeze(0), pos_points_b, sam_img_size=self.sam_img_size)[0]  # KxD
        z_pos = F.normalize(z_pos, dim=1)
        proto = z_pos.mean(dim=0)
        return self._make_viz_from_reference_vectors(
            resized_rgb=resized_rgb,
            z_map=z_map,
            pos_proto=proto,
            neg_proto=None,
        )

    def _score_map_from_reference_vectors(
        self,
        z_map: torch.Tensor,  # (D,Hm,Wm)
        pos_proto: torch.Tensor,  # (D,)
        neg_proto: torch.Tensor | None = None,  # (D,)
    ) -> torch.Tensor:
        z_map = F.normalize(z_map, dim=0)
        pos_proto = F.normalize(pos_proto, dim=0)
        score = (z_map.permute(1, 2, 0) * pos_proto).sum(dim=-1)  # Hm x Wm
        if neg_proto is not None and int(neg_proto.numel()) > 0:
            neg_proto = F.normalize(neg_proto, dim=0)
            score_neg = (z_map.permute(1, 2, 0) * neg_proto).sum(dim=-1)
            score = score - score_neg
        return score

    def _make_viz_from_score(
        self,
        resized_rgb: np.ndarray,
        score_hw: torch.Tensor,  # (Hm,Wm)
    ) -> VStrongViz:
        score = score_hw.unsqueeze(0).unsqueeze(0)  # 1x1xHm xWm

        # z_map lives on SAM's padded 1024x1024 canvas (feature map 64x64).
        # For visualization, upsample to the full padded canvas, then crop
        # to the unpadded resized image (top-left region).
        resized_h, resized_w = resized_rgb.shape[:2]
        score_full = F.interpolate(
            score,
            size=(self.sam_img_size, self.sam_img_size),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        score_up = score_full[:resized_h, :resized_w]
        score_np = score_up.detach().cpu().numpy()

        score01 = (score_np - score_np.min()) / max(score_np.max() - score_np.min(), 1e-6)
        heat_u8 = (score01 * 255.0).astype(np.uint8)
        heat_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

        rgb_bgr = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR)
        overlay = (0.6 * rgb_bgr + 0.4 * heat_bgr).astype(np.uint8)
        return VStrongViz(rgb=resized_rgb, heatmap_bgr=heat_bgr, overlay_bgr=overlay)

    def _make_viz_from_reference_vectors(
        self,
        resized_rgb: np.ndarray,
        z_map: torch.Tensor,  # (D,Hm,Wm)
        pos_proto: torch.Tensor,  # (D,)
        neg_proto: torch.Tensor | None = None,  # (D,)
    ) -> VStrongViz:
        score = self._score_map_from_reference_vectors(
            z_map=z_map,
            pos_proto=pos_proto,
            neg_proto=neg_proto,
        )
        return self._make_viz_from_score(resized_rgb=resized_rgb, score_hw=score)

    def _make_viz_with_traversability_vector(
        self,
        resized_rgb: np.ndarray,
        z_map: torch.Tensor,  # (D,Hm,Wm)
    ) -> VStrongViz:
        if (not bool(self.traversability_initialized.item())) or int(self.traversability_vector.numel()) == 0:
            raise RuntimeError("traversability vector is not initialized")
        return self._make_viz_from_reference_vectors(
            resized_rgb=resized_rgb,
            z_map=z_map,
            pos_proto=self.traversability_vector,
            neg_proto=None,
        )

    def _log_viz(self, tag: str, viz: VStrongViz):
        # TensorBoard
        for logger in (self.trainer.loggers or []):
            try:
                from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
            except Exception:
                TensorBoardLogger = None
                WandbLogger = None

            if TensorBoardLogger is not None and isinstance(logger, TensorBoardLogger):
                tb = logger.experiment
                img = torch.from_numpy(viz.overlay_bgr[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
                tb.add_image(tag, img, global_step=self.global_step)

            if WandbLogger is not None and isinstance(logger, WandbLogger):
                try:
                    import wandb
                    logger.experiment.log({tag: wandb.Image(viz.overlay_bgr[:, :, ::-1], caption=tag)}, step=self.global_step)
                except Exception:
                    pass

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        # batch is a list[VStrongSample]
        resized_rgbs = [s.resized_rgb for s in batch]
        pos_points = [torch.from_numpy(s.pos_points_resized).to(self.device) for s in batch]
        neg_points = [torch.from_numpy(s.neg_points_resized).to(self.device) for s in batch]

        emb = self._encode_images(resized_rgbs)  # Bx256x64x64
        z_map = self.proj(emb)  # BxD x64x64

        # Build fixed tensors (B,K,2) assuming dataset already returns fixed K per class.
        pos_xy = torch.stack(pos_points, dim=0)  # BxKx2
        neg_xy = torch.stack(neg_points, dim=0)  # BxKx2

        z_pos = sample_from_feature_map(z_map, pos_xy, sam_img_size=self.sam_img_size)  # BxKxD
        z_neg = sample_from_feature_map(z_map, neg_xy, sam_img_size=self.sam_img_size)  # BxKxD

        if stage == "train":
            with torch.no_grad():
                _, _, d = z_pos.shape
                pos_flat = z_pos.detach().reshape(-1, d)
                pos_flat = F.normalize(pos_flat, dim=1)
                pos_mean = F.normalize(pos_flat.mean(dim=0), dim=0)

                if bool(self.traversability_initialized.item()):
                    z_prev = F.normalize(self.traversability_vector.detach(), dim=0)
                    z_new = (self.traversability_ema_alpha * z_prev) + ((1.0 - self.traversability_ema_alpha) * pos_mean)
                else:
                    z_new = pos_mean
                z_new = F.normalize(z_new, dim=0)
                self.traversability_vector.copy_(z_new)
                self.traversability_initialized.fill_(True)

        b, k, d = z_pos.shape
        z_all = torch.cat([z_pos, z_neg], dim=1).reshape(b * (2 * k), d)
        labels = torch.cat(
            [
                torch.ones((b, k), device=self.device, dtype=torch.long),
                torch.zeros((b, k), device=self.device, dtype=torch.long),
            ],
            dim=1,
        ).reshape(-1)

        loss = supervised_contrastive_loss(z_all, labels, temperature=self.temperature)
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=len(batch))

        if stage == "train" and self.log_images_every_n_steps > 0:
            if (self.global_step % self.log_images_every_n_steps) == 0:
                viz = self._make_viz(
                    resized_rgb=batch[0].resized_rgb,
                    z_map=z_map[0].detach(),
                    pos_points=pos_xy[0].detach(),
                )
                if viz is not None:
                    self._log_viz("train/traversability_overlay", viz)

        return loss

    def training_step(self, batch, batch_idx: int):
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx: int):
        self._shared_step(batch, stage="val")
