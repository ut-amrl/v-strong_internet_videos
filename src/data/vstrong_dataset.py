"""V-STRONG dataset: loads frames, masks, and pos/neg points for contrastive training.

Supports two layouts:
  1. **Flat** — ``dataset_dir/frames/*.jpg``, ``dataset_dir/masks/*.png``,
     ``dataset_dir/pos_neg_points/*.npz``
  2. **Chunked** — ``dataset_dir/index.jsonl`` pointing to per-chunk sub-dirs
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset

_third_party_sam = str(Path(__file__).resolve().parents[2] / "third_party" / "segment-anything")
if _third_party_sam not in sys.path:
    sys.path.insert(0, _third_party_sam)
from segment_anything.utils.transforms import ResizeLongestSide

from data.sam_mask_generator import sample_pos_neg_points


@dataclass
class VStrongSample:
    resized_rgb: np.ndarray       # HxWx3 uint8 (after ResizeLongestSide)
    pos_points_resized: np.ndarray  # (K,2) float32 in resized coords
    neg_points_resized: np.ndarray  # (K,2) float32 in resized coords
    frame_idx: int


def vstrong_collate(batch: list[VStrongSample]) -> list[VStrongSample]:
    """Identity collate — train loop expects list[VStrongSample]."""
    return batch


class VStrongDataset(Dataset):
    """Dataset for V-STRONG contrastive training.

    Parameters
    ----------
    dataset_dir : str
        Root dataset directory (flat or chunked layout).
    split : str
        ``"train"`` or ``"val"``.
    val_ratio : float
        Fraction of frames used for validation.
    seed : int
        Random seed for reproducible splits.
    sam_img_size : int
        Target size for ResizeLongestSide (default 1024).
    points_per_class : int
        Fixed number of pos/neg points per sample.
    points_source : str
        ``"saved"`` (use .npz), ``"mask"`` (resample from mask), or ``"mixed"``.
    neg_top_frac : float
        Fraction of image top for biased negative sampling.
    sample_margin_px : int
        Border margin for point sampling.
    """

    def __init__(
        self,
        dataset_dir: str,
        split: str = "train",
        val_ratio: float = 0.1,
        seed: int = 0,
        sam_img_size: int = 1024,
        points_per_class: int = 64,
        points_source: str = "mixed",
        neg_top_frac: float = 0.3,
        sample_margin_px: int = 10,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.sam_img_size = sam_img_size
        self.points_per_class = points_per_class
        self.points_source = points_source
        self.neg_top_frac = neg_top_frac
        self.sample_margin_px = sample_margin_px
        self.transform = ResizeLongestSide(sam_img_size)

        self.samples = self._discover_samples()

        # Split
        rng = np.random.default_rng(seed)
        n = len(self.samples)
        perm = rng.permutation(n)
        n_val = max(1, int(n * val_ratio)) if val_ratio > 0 else 0
        val_idxs = set(perm[:n_val].tolist())
        if split == "val":
            self.samples = [self.samples[i] for i in sorted(val_idxs)]
        else:
            self.samples = [self.samples[i] for i in range(n) if i not in val_idxs]

    def _discover_samples(self) -> list[dict]:
        """Discover all samples, supporting both flat and chunked layouts."""
        index_file = self.dataset_dir / "index.jsonl"
        if index_file.exists():
            return self._load_from_index(index_file)
        return self._load_flat()

    def _load_from_index(self, index_file: Path) -> list[dict]:
        """Load sample entries from a chunked index.jsonl."""
        samples = []
        with open(index_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                frame_path = self.dataset_dir / entry["frame_path"]
                mask_path = self.dataset_dir / entry.get("mask_path", "")
                points_path = self.dataset_dir / entry.get("points_path", "")
                if frame_path.exists():
                    samples.append({
                        "frame_path": frame_path,
                        "mask_path": mask_path if mask_path.exists() else None,
                        "points_path": points_path if points_path.exists() else None,
                        "frame_idx": entry.get("frame_idx", 0),
                    })
        return sorted(samples, key=lambda s: s["frame_idx"])

    def _load_flat(self) -> list[dict]:
        """Load samples from flat dataset layout."""
        frames_dir = self.dataset_dir / "frames"
        masks_dir = self.dataset_dir / "masks"
        points_dir = self.dataset_dir / "pos_neg_points"

        if not frames_dir.exists():
            raise RuntimeError(f"No frames/ dir found in {self.dataset_dir}")

        frame_paths = sorted(frames_dir.glob("*.jpg"))
        samples = []
        for fp in frame_paths:
            try:
                idx = int(fp.stem)
            except ValueError:
                continue
            mask_path = masks_dir / f"{idx:06d}.png"
            pts_path = points_dir / f"{idx:06d}.npz"
            samples.append({
                "frame_path": fp,
                "mask_path": mask_path if mask_path.exists() else None,
                "points_path": pts_path if pts_path.exists() else None,
                "frame_idx": idx,
            })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> VStrongSample:
        s = self.samples[idx]
        rng = np.random.default_rng(s["frame_idx"])

        # Load and resize image
        bgr = cv2.imread(str(s["frame_path"]))
        if bgr is None:
            raise RuntimeError(f"Cannot read frame: {s['frame_path']}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = rgb.shape[:2]
        resized_rgb = self.transform.apply_image(rgb)

        # Load saved points
        saved_pos = np.empty((0, 2), dtype=np.float32)
        saved_neg = np.empty((0, 2), dtype=np.float32)
        if s["points_path"] is not None:
            pts = np.load(str(s["points_path"]))
            saved_pos = pts.get("pos_points", saved_pos).astype(np.float32)
            saved_neg = pts.get("neg_points", saved_neg).astype(np.float32)

        # Load mask
        mask01 = None
        if s["mask_path"] is not None:
            mask_u8 = cv2.imread(str(s["mask_path"]), cv2.IMREAD_GRAYSCALE)
            if mask_u8 is not None:
                mask01 = (mask_u8 > 0).astype(np.uint8)

        # Get pos/neg points based on source strategy
        k = self.points_per_class
        if self.points_source == "saved":
            pos_xy = self._sample_fixed_k(saved_pos, k, rng)
            neg_xy = self._sample_fixed_k(saved_neg, k, rng)
        elif self.points_source == "mask" and mask01 is not None:
            pos_xy, neg_xy = sample_pos_neg_points(
                mask01=mask01,
                num_pos=k,
                num_neg=k,
                neg_top_frac=self.neg_top_frac,
                sample_margin_px=self.sample_margin_px,
                rng=rng,
            )
        else:  # mixed
            pos_xy = saved_pos.copy()
            neg_xy = saved_neg.copy()
            if mask01 is not None:
                if len(pos_xy) < k:
                    extra_pos, _ = sample_pos_neg_points(
                        mask01=mask01,
                        num_pos=k - len(pos_xy),
                        num_neg=0,
                        neg_top_frac=self.neg_top_frac,
                        sample_margin_px=self.sample_margin_px,
                        rng=rng,
                    )
                    if len(extra_pos) > 0:
                        pos_xy = np.concatenate([pos_xy, extra_pos], axis=0)
                if len(neg_xy) < k:
                    _, extra_neg = sample_pos_neg_points(
                        mask01=mask01,
                        num_pos=0,
                        num_neg=k - len(neg_xy),
                        neg_top_frac=self.neg_top_frac,
                        sample_margin_px=self.sample_margin_px,
                        rng=rng,
                    )
                    if len(extra_neg) > 0:
                        neg_xy = np.concatenate([neg_xy, extra_neg], axis=0)
            pos_xy = self._sample_fixed_k(pos_xy, k, rng)
            neg_xy = self._sample_fixed_k(neg_xy, k, rng)

        # Transform points to resized coords
        pos_resized = self.transform.apply_coords(pos_xy, (h_orig, w_orig)).astype(np.float32)
        neg_resized = self.transform.apply_coords(neg_xy, (h_orig, w_orig)).astype(np.float32)

        return VStrongSample(
            resized_rgb=resized_rgb,
            pos_points_resized=pos_resized,
            neg_points_resized=neg_resized,
            frame_idx=s["frame_idx"],
        )

    @staticmethod
    def _sample_fixed_k(points: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
        """Ensure exactly k points (with replacement if needed)."""
        if k <= 0:
            return np.empty((0, 2), dtype=np.float32)
        points = points.astype(np.float32, copy=False)
        n = len(points)
        if n == 0:
            return np.zeros((k, 2), dtype=np.float32)
        if n >= k:
            idx = rng.choice(n, size=k, replace=False)
        else:
            idx = rng.choice(n, size=k, replace=True)
        return points[idx]
