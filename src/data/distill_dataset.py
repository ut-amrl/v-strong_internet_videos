"""Image-only dataset for student–teacher distillation.

Returns a list of (resized HxWx3 uint8 numpy) arrays per batch.
No point labels are required — we only need the raw images.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset


class ResizeLongestSide:
    """Resize image so its longest side equals target_length."""

    def __init__(self, target_length: int):
        self.target_length = int(target_length)

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        scale = self.target_length / max(h, w)
        new_h = int(h * scale + 0.5)
        new_w = int(w * scale + 0.5)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


class DistillDataset(Dataset):
    """Loads all JPEG frames from ``dataset_dir/frames/`` and returns resized RGB arrays.

    Compatible with the V-STRONG dataset structure. Any frame that can be read
    is included — point labels are not required.

    Args:
        dataset_dir: Root directory of a V-STRONG dataset (must have a ``frames/`` sub-dir).
        split: ``"train"`` or ``"val"``.
        val_ratio: Fraction of frames held out for validation.
        seed: Shuffling seed for the train/val split.
        img_size: Longest-side resize target (typically 1024 to match SAM).
    """

    def __init__(
        self,
        dataset_dir: str,
        split: str = "train",
        val_ratio: float = 0.1,
        seed: int = 0,
        img_size: int = 1024,
    ):
        self.frames_dir = Path(dataset_dir) / "frames"
        if not self.frames_dir.exists():
            raise RuntimeError(f"Frames directory not found: {self.frames_dir}")

        self.transform = ResizeLongestSide(img_size)

        all_frames = sorted(self.frames_dir.glob("*.jpg"))
        if not all_frames:
            raise RuntimeError(f"No *.jpg frames found in {self.frames_dir}")

        rng = np.random.default_rng(seed)
        indices = np.arange(len(all_frames))
        rng.shuffle(indices)

        n_val = max(1, int(round(len(all_frames) * val_ratio)))
        val_indices = set(indices[:n_val].tolist())

        if split == "train":
            self.entries = [all_frames[i] for i in range(len(all_frames)) if i not in val_indices]
        elif split == "val":
            self.entries = [all_frames[i] for i in range(len(all_frames)) if i in val_indices]
        else:
            raise ValueError(f"Invalid split={split!r}. Expected 'train' or 'val'.")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> np.ndarray:
        path = self.entries[idx]
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            raise RuntimeError(f"Failed to read image: {path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return self.transform.apply_image(img_rgb)  # HxWx3 uint8


def distill_collate(batch: list[np.ndarray]) -> list[np.ndarray]:
    """Keep images as a plain Python list — models consume them directly."""
    return batch
