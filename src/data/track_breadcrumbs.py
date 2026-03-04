"""
Track breadcrumb points through video frames using frame-by-frame
chained Lucas-Kanade optical flow in REVERSE.

Algorithm:
  1. Process frames in reverse order (last → first).
  2. In each frame, seed ONE new point slightly below image center.
  3. Track ALL existing points from the current frame to the previous
     frame using LK optical flow. The OUTPUT coordinates from one frame
     become the INPUT for the next frame (chained tracking).
  4. Remove any point that falls within a threshold of the image boundary.
  5. The result: on each frame, the breadcrumb points mark the ground
     locations the car will traverse (has traversed from the perspective
     of later frames), forming a trail showing the car's path.

Usage:
    python src/data/track_breadcrumbs.py \
        --frames_dir data/output_2/frames \
        --output data/output_2/breadcrumbs \
        --center_offset_frac 0.1 \
        --boundary_thresh 30 \
        --max_track_len 20
"""

import argparse
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def get_sorted_frame_paths(frames_dir: str) -> List[Path]:
    """Return sorted list of frame image paths."""
    frames_dir = Path(frames_dir)
    exts = {".jpg", ".jpeg", ".png"}
    paths = sorted([p for p in frames_dir.iterdir() if p.suffix.lower() in exts])
    if len(paths) == 0:
        raise RuntimeError(f"No image files found in {frames_dir}")
    return paths


def track_breadcrumbs_chained(
    frame_paths: List[Path],
    center_offset_frac: float = 0.1,
    boundary_thresh: int = 30,
    max_track_len: int = 20,
) -> Dict[int, np.ndarray]:
    """
    Track breadcrumb points using frame-by-frame chained LK in reverse.

    Starting from the last frame, we seed one point per frame and track
    all existing points backward (last → first) using Lucas-Kanade.
    The output coordinates from frame N become the input for tracking
    into frame N-1.

    Args:
        frame_paths: Sorted list of frame paths (chronological order)
        center_offset_frac: Fraction of image height below center for
                            the seed point (e.g. 0.1 = 10% below center)
        boundary_thresh: Remove points within this many px of image edge
        max_track_len: Max number of frames a single seeded point can
                       remain active (<= 0 means unlimited)

    Returns:
        Dict mapping frame index → np.ndarray of shape (M, 2) with (x, y)
    """
    lk_params = dict(
        winSize=(31, 31),
        maxLevel=4,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            50,
            0.01,
        ),
    )

    n_frames = len(frame_paths)

    # Read last frame to get dimensions
    last_frame = cv2.imread(str(frame_paths[-1]))
    h, w = last_frame.shape[:2]

    # Seed point location: slightly below center
    seed_x = w / 2.0
    seed_y = (h / 2.0) + (h * center_offset_frac)

    print(f"Tracking breadcrumbs (chained LK, reverse) across {n_frames} frames")
    print(f"  Seed point: ({seed_x:.0f}, {seed_y:.0f})")
    print(f"  Boundary threshold: {boundary_thresh}px")
    if max_track_len > 0:
        print(f"  Max track length: {max_track_len} frames per point")
    else:
        print("  Max track length: unlimited")

    # Start from last frame, work backward
    prev_gray = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)

    # Active points: list of (x, y) currently being tracked
    # Start with one seed on the last frame
    active_points = np.array([[seed_x, seed_y]], dtype=np.float32)
    # Per-point age measured as "number of frames this point has appeared in".
    active_ages = np.array([1], dtype=np.int32)

    # Store breadcrumbs for each frame
    breadcrumbs: Dict[int, np.ndarray] = {}
    breadcrumbs[n_frames - 1] = active_points.copy()

    it = range(1, n_frames)
    pbar = tqdm(it, desc="Tracking breadcrumbs (LK reverse)", unit="frame") if tqdm is not None else it
    for rev_step in pbar:
        fwd_idx = n_frames - 1 - rev_step  # index in original order

        curr_frame = cv2.imread(str(frame_paths[fwd_idx]))
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # --- Track existing points from prev frame → curr frame ---
        if len(active_points) > 0:
            pts_prev = active_points.reshape(-1, 1, 2)
            pts_next, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, pts_prev, None, **lk_params
            )
            status_ok = status.ravel() == 1
            tracked = pts_next[status_ok].reshape(-1, 2)
            tracked_ages = active_ages[status_ok] + 1
        else:
            tracked = np.empty((0, 2), dtype=np.float32)
            tracked_ages = np.empty((0,), dtype=np.int32)

        # --- Remove points near image boundary ---
        if len(tracked) > 0:
            keep = (
                (tracked[:, 0] >= boundary_thresh)
                & (tracked[:, 0] < w - boundary_thresh)
                & (tracked[:, 1] >= boundary_thresh)
                & (tracked[:, 1] < h - boundary_thresh)
            )
            tracked = tracked[keep]
            tracked_ages = tracked_ages[keep]

        # --- Enforce per-point max lifespan ---
        if max_track_len > 0 and len(tracked_ages) > 0:
            keep_age = tracked_ages <= int(max_track_len)
            tracked = tracked[keep_age]
            tracked_ages = tracked_ages[keep_age]

        # --- Add new seed point for this frame ---
        new_seed = np.array([[seed_x, seed_y]], dtype=np.float32)
        new_seed_age = np.array([1], dtype=np.int32)
        if len(tracked) > 0:
            active_points = np.vstack([tracked, new_seed])
            active_ages = np.concatenate([tracked_ages, new_seed_age])
        else:
            active_points = new_seed
            active_ages = new_seed_age

        breadcrumbs[fwd_idx] = active_points.copy()

        if rev_step % 20 == 0 or fwd_idx == 0:
            msg = (
                f"  Frame {fwd_idx:4d} (rev step {rev_step}): "
                f"{len(active_points)} active points "
                f"({len(tracked)} tracked + 1 new)"
            )
            if tqdm is not None and hasattr(pbar, "write"):
                pbar.write(msg)
            else:
                print(msg)

        if tqdm is not None and hasattr(pbar, "set_postfix_str"):
            pbar.set_postfix_str(f"active={len(active_points)}")

        prev_gray = curr_gray

    total = sum(len(pts) for pts in breadcrumbs.values())
    print(f"  Total: {total} breadcrumb points across {len(breadcrumbs)} frames")
    return breadcrumbs


def save_breadcrumbs(breadcrumbs: Dict[int, np.ndarray], output_dir: str):
    """Save breadcrumb points to .npz files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, points in breadcrumbs.items():
        out_path = output_dir / f"{idx:06d}_breadcrumbs.npz"
        np.savez_compressed(str(out_path), points=points)

    print(f"Saved {len(breadcrumbs)} breadcrumb files to {output_dir}")


def load_breadcrumbs(breadcrumbs_dir: str) -> Dict[int, np.ndarray]:
    """
    Load breadcrumb points from a directory of `*_breadcrumbs.npz` files.

    Filenames are expected to be like `000123_breadcrumbs.npz` and contain
    an array under key `points` with shape (N, 2) as (x, y).
    """
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


def main():
    parser = argparse.ArgumentParser(
        description="Track breadcrumb points via chained LK (reverse)"
    )
    parser.add_argument(
        "--frames_dir", type=str, required=True,
        help="Directory of extracted frames",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory for breadcrumb .npz files",
    )
    parser.add_argument(
        "--center_offset_frac", type=float, default=0.1,
        help="Fraction of image height below center for seed (default: 0.1)",
    )
    parser.add_argument(
        "--boundary_thresh", type=int, default=30,
        help="Remove points within this many px of edge (default: 30)",
    )
    parser.add_argument(
        "--max_track_len", type=int, default=20,
        help="Max frames one point can remain active (<=0 for unlimited, default: 20)",
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
