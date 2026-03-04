import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .sam_mask_generator import sample_pos_neg_points


class ResizeLongestSide:
    """Resize image so its longest side equals target_length (matches SAM's transform)."""

    def __init__(self, target_length: int):
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        scale = self.target_length / max(h, w)
        new_h = int(h * scale + 0.5)
        new_w = int(w * scale + 0.5)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    def apply_coords(self, coords: np.ndarray, original_size: tuple) -> np.ndarray:
        old_h, old_w = original_size
        scale = self.target_length / max(old_h, old_w)
        return (coords * scale).astype(np.float32)


@dataclass(frozen=True)
class VStrongSample:
    frame_idx: int
    image_rgb: np.ndarray  # HxWx3 uint8 (original)
    resized_rgb: np.ndarray  # h'xw'x3 uint8 (ResizeLongestSide applied)
    pos_points_resized: np.ndarray  # (K,2) float32 in resized coords (x,y)
    neg_points_resized: np.ndarray  # (K,2) float32 in resized coords (x,y)
    original_size_hw: tuple[int, int]


def _parse_frame_idx(frame_path: Path) -> int:
    try:
        return int(frame_path.stem)
    except ValueError as e:
        raise RuntimeError(f"Expected integer frame filename like 000123.jpg, got {frame_path.name}") from e


def _sample_fixed_k(points_xy: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    points_xy = points_xy.astype(np.float32, copy=False)
    if k <= 0:
        return np.empty((0, 2), dtype=np.float32)
    if len(points_xy) == 0:
        return np.empty((0, 2), dtype=np.float32)
    if len(points_xy) >= k:
        idxs = rng.choice(len(points_xy), size=k, replace=False)
        return points_xy[idxs]
    idxs = rng.choice(len(points_xy), size=k, replace=True)
    return points_xy[idxs]


class VStrongDataset(Dataset):
    """
    Dataset for V-STRONG training.

    Consumes:
      - frames/*.jpg
      - pos_neg_points/*.npz with keys {pos_points, neg_points}
      - masks/*.png (optional, used for denser sampling)
      - dataset_index.json (optional; directory conventions work too)
    """

    def __init__(
        self,
        dataset_dir: str,
        split: str = "train",
        val_ratio: float = 0.1,
        seed: int = 0,
        img_size: int = 1024,
        points_per_class: int = 64,
        points_source: str = "mixed",  # saved|mask|mixed
        neg_top_frac: float = 0.3,
        sample_margin_px: int = 10,
        # Legacy alias
        sam_img_size: int | None = None,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.frames_dir = self.dataset_dir / "frames"
        self.points_dir = self.dataset_dir / "pos_neg_points"
        self.masks_dir = self.dataset_dir / "masks"
        self.seed = int(seed)
        self.points_per_class = int(points_per_class)
        self.points_source = str(points_source)
        self.neg_top_frac = float(neg_top_frac)
        self.sample_margin_px = int(sample_margin_px)

        # Legacy: accept sam_img_size as alias for img_size
        if sam_img_size is not None:
            img_size = sam_img_size

        self.transform = ResizeLongestSide(int(img_size))

        if self.points_source not in {"saved", "mask", "mixed"}:
            raise ValueError(f"Invalid points_source={self.points_source}, expected saved|mask|mixed")

        entries = self._load_entries()
        if len(entries) == 0:
            raise RuntimeError(f"No usable frames found in {self.frames_dir}")

        rng = np.random.default_rng(self.seed)
        rng.shuffle(entries)
        n_val = int(round(len(entries) * float(val_ratio)))
        n_val = min(max(n_val, 1), max(len(entries) - 1, 1)) if len(entries) > 1 else 0

        if split == "train":
            self.entries = entries[n_val:]
        elif split == "val":
            self.entries = entries[:n_val]
        else:
            raise ValueError(f"Invalid split={split}, expected train|val")

    def _load_entries(self) -> list[dict]:
        index_path = self.dataset_dir / "dataset_index.json"
        if index_path.exists():
            data = json.load(open(index_path, "r"))
            entries = []
            for e in data:
                frame_path = self.dataset_dir / e["frame_path"]
                frame_idx = int(e["frame_idx"])
                points_path = self.dataset_dir / e.get("pos_neg_points_path", f"pos_neg_points/{frame_idx:06d}.npz")
                mask_path = self.dataset_dir / e.get("mask_path", f"masks/{frame_idx:06d}.png")
                if not frame_path.exists():
                    continue
                if (not points_path.exists()) and (not mask_path.exists()):
                    continue
                # Keep even if points/mask missing; __getitem__ handles.
                entries.append({"frame_idx": frame_idx, "frame_path": frame_path, "points_path": points_path, "mask_path": mask_path})
            return entries

        # Fallback: scan frames directory
        entries = []
        for frame_path in sorted(self.frames_dir.glob("*.jpg")):
            frame_idx = _parse_frame_idx(frame_path)
            points_path = self.points_dir / f"{frame_idx:06d}.npz"
            mask_path = self.masks_dir / f"{frame_idx:06d}.png"
            if (not points_path.exists()) and (not mask_path.exists()):
                continue
            entries.append(
                {"frame_idx": frame_idx, "frame_path": frame_path, "points_path": points_path, "mask_path": mask_path}
            )
        return entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> VStrongSample:
        e = self.entries[idx]
        frame_idx = int(e["frame_idx"])
        frame_path: Path = e["frame_path"]
        points_path: Path = e["points_path"]
        mask_path: Path = e["mask_path"]

        img_bgr = cv2.imread(str(frame_path))
        if img_bgr is None:
            raise RuntimeError(f"Failed to read image: {frame_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        rng = np.random.default_rng(self.seed + frame_idx)

        saved_pos = np.empty((0, 2), dtype=np.float32)
        saved_neg = np.empty((0, 2), dtype=np.float32)
        if points_path.exists():
            pts = np.load(str(points_path))
            saved_pos = pts.get("pos_points", saved_pos).astype(np.float32, copy=False)
            saved_neg = pts.get("neg_points", saved_neg).astype(np.float32, copy=False)

        mask01 = None
        if mask_path.exists():
            mask_u8 = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_u8 is not None:
                mask01 = (mask_u8 > 0).astype(np.uint8)

        if self.points_source == "saved":
            pos_xy = _sample_fixed_k(saved_pos, self.points_per_class, rng)
            neg_xy = _sample_fixed_k(saved_neg, self.points_per_class, rng)
        elif self.points_source == "mask" and mask01 is not None:
            pos_xy, neg_xy = sample_pos_neg_points(
                mask01=mask01,
                num_pos=self.points_per_class,
                num_neg=self.points_per_class,
                neg_top_frac=self.neg_top_frac,
                sample_margin_px=self.sample_margin_px,
                rng=rng,
            )
            pos_xy = _sample_fixed_k(pos_xy, self.points_per_class, rng)
            neg_xy = _sample_fixed_k(neg_xy, self.points_per_class, rng)
        else:
            # mixed (or mask missing): prefer saved, fill from mask if needed
            pos_xy = saved_pos
            neg_xy = saved_neg
            if mask01 is not None:
                need_pos = max(0, self.points_per_class - len(pos_xy))
                need_neg = max(0, self.points_per_class - len(neg_xy))
                if need_pos > 0 or need_neg > 0:
                    extra_pos, extra_neg = sample_pos_neg_points(
                        mask01=mask01,
                        num_pos=max(need_pos, 0),
                        num_neg=max(need_neg, 0),
                        neg_top_frac=self.neg_top_frac,
                        sample_margin_px=self.sample_margin_px,
                        rng=rng,
                    )
                    if need_pos > 0:
                        pos_xy = np.concatenate([pos_xy, extra_pos], axis=0) if len(extra_pos) else pos_xy
                    if need_neg > 0:
                        neg_xy = np.concatenate([neg_xy, extra_neg], axis=0) if len(extra_neg) else neg_xy
            pos_xy = _sample_fixed_k(pos_xy, self.points_per_class, rng)
            neg_xy = _sample_fixed_k(neg_xy, self.points_per_class, rng)

        resized_rgb = self.transform.apply_image(img_rgb)
        pos_resized = self.transform.apply_coords(pos_xy, (h, w))
        neg_resized = self.transform.apply_coords(neg_xy, (h, w))

        return VStrongSample(
            frame_idx=frame_idx,
            image_rgb=img_rgb,
            resized_rgb=resized_rgb,
            pos_points_resized=pos_resized,
            neg_points_resized=neg_resized,
            original_size_hw=(h, w),
        )


def vstrong_collate(batch: list[VStrongSample]) -> list[VStrongSample]:
    return batch
