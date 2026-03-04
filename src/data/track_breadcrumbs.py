"""Track breadcrumb points through video frames using chained LK optical flow.

This is ported from `origin/modular_net` (commit `c077017`), adapted to be used
as a reusable module by `src/data/generate_dataset.py`.

Algorithm (reverse tracking, chained LK):
  1. Process frames in reverse order (last → first).
  2. In each frame, seed ONE new point slightly below image center.
  3. Track ALL existing points from the current frame to the previous frame
     using Lucas–Kanade optical flow. The output coords from one frame become
     the input for the next frame (chained tracking).
  4. Remove points that drift near the image boundary.
  5. (Optional) cap each point's lifespan to `max_track_len`.

Outputs:
  - `breadcrumbs`: dict[int, np.ndarray] mapping frame_idx -> (N,2) float32 x,y
  - `save_breadcrumbs(...)` writes `{frame_idx:06d}_breadcrumbs.npz` files
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def get_sorted_frame_paths(frames_dir: str | Path) -> List[Path]:
    """Return sorted list of frame image paths (by filename integer)."""
    frames_dir = Path(frames_dir)
    exts = {".jpg", ".jpeg", ".png"}
    paths = sorted([p for p in frames_dir.iterdir() if p.suffix.lower() in exts])
    if len(paths) == 0:
        raise RuntimeError(f"No image files found in {frames_dir}")
    return paths


def _parse_frame_idx(frame_path: Path) -> int:
    try:
        return int(frame_path.stem)
    except ValueError as e:
        raise RuntimeError(
            f"Frame filename must be an integer like 000123.jpg, got: {frame_path.name}"
        ) from e


def track_breadcrumbs_chained(
    frame_paths: List[Path],
    center_offset_frac: float = 0.1,
    boundary_thresh: int = 30,
    max_track_len: int = 20,
) -> Dict[int, np.ndarray]:
    """Track breadcrumb points using frame-by-frame chained LK in reverse.

    Args:
        frame_paths: Sorted list of frame paths (chronological order).
        center_offset_frac: Fraction of image height below center for the seed point.
        boundary_thresh: Remove points within this many px of image edge.
        max_track_len: Max number of frames a single seeded point can remain active
                       (<= 0 means unlimited).

    Returns:
        Dict mapping frame index → np.ndarray of shape (M, 2) with (x, y).
    """
    lk_params = dict(
        winSize=(31, 31),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.01),
    )

    n_frames = len(frame_paths)
    if n_frames == 0:
        return {}

    # Read last frame to get dimensions
    last_frame = cv2.imread(str(frame_paths[-1]))
    if last_frame is None:
        raise RuntimeError(f"Cannot read frame: {frame_paths[-1]}")
    h, w = last_frame.shape[:2]

    # Seed point location: slightly below center
    seed_x = w / 2.0
    seed_y = (h / 2.0) + (h * float(center_offset_frac))

    # Start from last frame, work backward
    prev_gray = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)

    # Active points: list of (x, y) currently being tracked
    active_points = np.array([[seed_x, seed_y]], dtype=np.float32)
    # Per-point age measured as "number of frames this point has appeared in".
    active_ages = np.array([1], dtype=np.int32)

    breadcrumbs: Dict[int, np.ndarray] = {}
    breadcrumbs[_parse_frame_idx(frame_paths[-1])] = active_points.copy()

    it = range(1, n_frames)
    pbar = tqdm(it, desc="Tracking breadcrumbs (LK reverse)", unit="frame") if tqdm is not None else it
    for rev_step in pbar:
        fwd_path = frame_paths[n_frames - 1 - rev_step]
        fwd_idx = _parse_frame_idx(fwd_path)

        curr_frame = cv2.imread(str(fwd_path))
        if curr_frame is None:
            continue
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # Track existing points from prev frame → curr frame
        if len(active_points) > 0:
            pts_prev = active_points.reshape(-1, 1, 2)
            pts_next, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts_prev, None, **lk_params)
            status_ok = status.ravel() == 1
            tracked = pts_next[status_ok].reshape(-1, 2)
            tracked_ages = active_ages[status_ok] + 1
        else:
            tracked = np.empty((0, 2), dtype=np.float32)
            tracked_ages = np.empty((0,), dtype=np.int32)

        # Remove points near image boundary
        if len(tracked) > 0:
            keep = (
                (tracked[:, 0] >= boundary_thresh)
                & (tracked[:, 0] < w - boundary_thresh)
                & (tracked[:, 1] >= boundary_thresh)
                & (tracked[:, 1] < h - boundary_thresh)
            )
            tracked = tracked[keep]
            tracked_ages = tracked_ages[keep]

        # Enforce per-point max lifespan
        if max_track_len > 0 and len(tracked_ages) > 0:
            keep_age = tracked_ages <= int(max_track_len)
            tracked = tracked[keep_age]
            tracked_ages = tracked_ages[keep_age]

        # Add new seed point for this frame
        new_seed = np.array([[seed_x, seed_y]], dtype=np.float32)
        new_seed_age = np.array([1], dtype=np.int32)
        if len(tracked) > 0:
            active_points = np.vstack([tracked, new_seed])
            active_ages = np.concatenate([tracked_ages, new_seed_age])
        else:
            active_points = new_seed
            active_ages = new_seed_age

        breadcrumbs[fwd_idx] = active_points.copy()
        prev_gray = curr_gray

        if tqdm is not None and hasattr(pbar, "set_postfix_str"):
            pbar.set_postfix_str(f"active={len(active_points)}")

    return breadcrumbs


def save_breadcrumbs(breadcrumbs: Dict[int, np.ndarray], output_dir: str | Path) -> None:
    """Save breadcrumb points to `*_breadcrumbs.npz` files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, points in breadcrumbs.items():
        out_path = output_dir / f"{int(idx):06d}_breadcrumbs.npz"
        np.savez_compressed(str(out_path), points=points)


def load_breadcrumbs(breadcrumbs_dir: str | Path) -> Dict[int, np.ndarray]:
    """Load breadcrumbs from `*_breadcrumbs.npz` files."""
    breadcrumbs_dir = Path(breadcrumbs_dir)
    paths = sorted(breadcrumbs_dir.glob("*_breadcrumbs.npz"))
    if len(paths) == 0:
        raise RuntimeError(f"No breadcrumb files found in {breadcrumbs_dir}")

    breadcrumbs: Dict[int, np.ndarray] = {}
    for p in paths:
        stem = p.stem  # e.g. "000123_breadcrumbs"
        if not stem.endswith("_breadcrumbs"):
            continue
        idx_str = stem.rsplit("_breadcrumbs", 1)[0]
        try:
            frame_idx = int(idx_str)
        except ValueError as e:
            raise RuntimeError(f"Cannot parse frame index from {p.name}") from e

        data = np.load(str(p))
        if "points" not in data:
            raise RuntimeError(f"Missing 'points' key in {p}")
        points = data["points"].astype(np.float32, copy=False)
        if points.ndim != 2 or points.shape[1] != 2:
            raise RuntimeError(f"Invalid points shape in {p}: {points.shape}")
        breadcrumbs[frame_idx] = points

    if len(breadcrumbs) == 0:
        raise RuntimeError(f"No valid breadcrumb files found in {breadcrumbs_dir}")
    return breadcrumbs


def main() -> None:
    parser = argparse.ArgumentParser(description="Track breadcrumb points via chained LK (reverse).")
    parser.add_argument("--frames_dir", type=str, required=True, help="Directory of extracted frames.")
    parser.add_argument("--output", type=str, required=True, help="Output directory for breadcrumb .npz files.")
    parser.add_argument("--center_offset_frac", type=float, default=0.1)
    parser.add_argument("--boundary_thresh", type=int, default=30)
    parser.add_argument(
        "--max_track_len",
        type=int,
        default=20,
        help="Max frames one point can remain active (<=0 for unlimited).",
    )
    args = parser.parse_args()

    frame_paths = get_sorted_frame_paths(args.frames_dir)
    breadcrumbs = track_breadcrumbs_chained(
        frame_paths,
        center_offset_frac=args.center_offset_frac,
        boundary_thresh=args.boundary_thresh,
        max_track_len=args.max_track_len,
    )
    save_breadcrumbs(breadcrumbs, args.output)


if __name__ == "__main__":
    main()

