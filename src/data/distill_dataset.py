"""Image-only dataset for student–teacher distillation.

Returns a list of (resized HxWx3 uint8 numpy) arrays per batch.
No point labels are required — we only need the raw images.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset


class ResizeSquare:
    """Resize image to a square of target_length x target_length."""

    def __init__(self, target_length: int):
        self.target_length = int(target_length)

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, (self.target_length, self.target_length), interpolation=cv2.INTER_LINEAR)


class DistillDataset(Dataset):
    """Loads all JPEG frames from one or more dataset directories.

    Supports both flat layout (``dataset_dir/frames/*.jpg``) and chunked layout
    (``dataset_dir/index.jsonl``). Multiple directories can be passed as a list.

    Args:
        dataset_dir: Root directory (or list of directories) of V-STRONG datasets.
        split: ``"train"`` or ``"val"``.
        val_ratio: Fraction of frames held out for validation.
        seed: Shuffling seed for the train/val split.
        img_size: Longest-side resize target (typically 1024 to match SAM).
    """

    def __init__(
        self,
        dataset_dir: str | list[str],
        split: str = "train",
        val_ratio: float = 0.1,
        seed: int = 0,
        img_size: int = 1024,
    ):
        if isinstance(dataset_dir, (list, tuple)):
            dataset_dirs = [Path(d) for d in dataset_dir if str(d).strip()]
        else:
            dataset_dirs = [Path(str(dataset_dir))]
        if not dataset_dirs:
            raise RuntimeError("dataset_dir must be a non-empty path or list of paths.")

        self.transform = ResizeSquare(img_size)

        all_frames: list[Path] = []
        for root in dataset_dirs:
            index_file = root / "index.jsonl"
            if index_file.exists():
                all_frames.extend(self._collect_from_index(root, index_file))
            else:
                all_frames.extend(self._collect_flat(root))

        if not all_frames:
            raise RuntimeError(f"No *.jpg frames found in: {[str(d) for d in dataset_dirs]}")

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

    @staticmethod
    def _collect_from_index(root: Path, index_file: Path) -> list[Path]:
        """Collect frame paths from a chunked index.jsonl."""
        frames = []
        with open(index_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                frame_path = root / entry["frame_path"]
                if frame_path.exists():
                    frames.append(frame_path)
        return frames

    @staticmethod
    def _collect_flat(root: Path) -> list[Path]:
        """Collect frame paths from a flat frames/ directory."""
        frames_dir = root / "frames"
        if not frames_dir.exists():
            raise RuntimeError(f"No frames/ dir found in {root}")
        return sorted(frames_dir.glob("*.jpg"))

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
