"""Track breadcrumb points through video frames using CoTracker online predictor.

Drop-in alternative to ``track_breadcrumbs_chained`` (LK optical flow).
The public API mirrors the LK module:

    breadcrumbs = track_breadcrumbs_cotracker(frame_paths, ...)
    # -> Dict[int, np.ndarray]  mapping frame_idx -> (M, 2) float32 x,y

Algorithm (reverse tracking, CoTracker):
  1. Load all frames into a (1, T, 3, H, W) float32 tensor, reversed.
  2. Seed one query (t, seed_x, seed_y) per reversed-time frame,
     placed slightly below image centre (matching LK behaviour).
  3. Run CoTrackerOnlinePredictor over the reversed clip in overlapping
     windows of size ``model.step * 2``.
  4. For each original (forward-time) frame, gather all query points
     that are *visible*, apply boundary filtering and max-track-length
     pruning, and store the result as (M, 2) ndarray.

Outputs are fully compatible with ``save_breadcrumbs`` / ``load_breadcrumbs``
and the downstream SAM masking pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

# ---------------------------------------------------------------------------
# Ensure the co-tracker third-party package is importable
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]  # v-strong_internet_videos/
_COTRACKER_DIR = _REPO_ROOT / "third_party" / "co-tracker"
if str(_COTRACKER_DIR) not in sys.path:
    sys.path.insert(0, str(_COTRACKER_DIR))


def _load_frames_as_tensor(
    frame_paths: List[Path],
    device: torch.device,
) -> torch.Tensor:
    """Load frame images into a reversed (1, T, 3, H, W) float32 tensor.

    Frames are loaded in forward order, then the time axis is flipped
    so that index 0 corresponds to the *last* chronological frame.
    """
    frames = []
    for p in frame_paths:
        img = cv2.imread(str(p))
        if img is None:
            raise RuntimeError(f"Cannot read frame: {p}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(torch.from_numpy(img_rgb).permute(2, 0, 1).float())  # (3,H,W)

    video = torch.stack(frames, dim=0)        # (T, 3, H, W)
    video = torch.flip(video, dims=[0])       # reverse time
    return video.unsqueeze(0).to(device)       # (1, T, 3, H, W)


def track_breadcrumbs_cotracker(
    frame_paths: List[Path],
    center_offset_frac: float = 0.1,
    boundary_thresh: int = 30,
    max_track_len: int = 20,
    checkpoint: str = "./external/co-tracker/checkpoints/scaled_online.pth",
    window_len: int = 16,
    device: Optional[str] = None,
) -> Dict[int, np.ndarray]:
    """Track breadcrumb points using CoTracker (reverse, online mode).

    The interface and output format are identical to
    ``track_breadcrumbs_chained`` so the two can be swapped transparently.

    Args:
        frame_paths: Sorted list of frame paths (chronological order).
        center_offset_frac: Fraction of image height below centre for seed.
        boundary_thresh: Remove points within this many px of image edge.
        max_track_len: Max frames a single seeded point can remain active
                       (<=0 means unlimited).
        checkpoint: Path to CoTracker checkpoint file.
        window_len: CoTracker sliding-window length.
        device: ``'cuda'``, ``'cpu'``, etc.  Auto-detected if *None*.

    Returns:
        Dict mapping frame index -> np.ndarray of shape (M, 2) with (x, y).
    """
    from cotracker.predictor import CoTrackerOnlinePredictor

    n_frames = len(frame_paths)
    if n_frames == 0:
        return {}

    # ------------------------------------------------------------------
    # Device selection
    # ------------------------------------------------------------------
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    dev = torch.device(device)

    # ------------------------------------------------------------------
    # Read one frame to get dimensions & seed location
    # ------------------------------------------------------------------
    sample = cv2.imread(str(frame_paths[-1]))
    if sample is None:
        raise RuntimeError(f"Cannot read frame: {frame_paths[-1]}")
    h, w = sample.shape[:2]

    seed_x = w / 2.0
    seed_y = (h / 2.0) + (h * float(center_offset_frac))

    # ------------------------------------------------------------------
    # Build video tensor  (1, T, 3, H, W)  – reversed time
    # ------------------------------------------------------------------
    print("Loading frames into tensor for CoTracker …")
    video = _load_frames_as_tensor(frame_paths, dev)
    T = video.shape[1]  # == n_frames

    # ------------------------------------------------------------------
    # Build queries: one seed per reversed-time frame  (1, T, 3)
    # queries[:, i, :] = (t=i, seed_x, seed_y)
    # ------------------------------------------------------------------
    queries = torch.zeros(1, T, 3, device=dev, dtype=torch.float32)
    queries[0, :, 0] = torch.arange(T, device=dev, dtype=torch.float32)  # t
    queries[0, :, 1] = seed_x
    queries[0, :, 2] = seed_y

    # ------------------------------------------------------------------
    # Initialise CoTracker
    # ------------------------------------------------------------------
    print(f"Initialising CoTracker (checkpoint={checkpoint}, window_len={window_len})")
    model = CoTrackerOnlinePredictor(
        checkpoint=checkpoint,
        window_len=window_len,
    ).to(dev)
    model.eval()

    # First step: register queries
    model(video_chunk=video, is_first_step=True,
          grid_size=0, queries=queries, add_support_grid=True)

    # ------------------------------------------------------------------
    # Run inference in sliding windows
    # ------------------------------------------------------------------
    pred_tracks = None
    pred_visibility = None

    step = model.step
    it = range(0, T - step, step)
    pbar = (tqdm(it, desc="CoTracker inference", unit="window")
            if tqdm is not None else it)

    for ind in pbar:
        pred_tracks, pred_visibility = model(
            video_chunk=video[:, ind: ind + step * 2],
            grid_size=0, queries=queries, add_support_grid=True,
        )

    if pred_tracks is None:
        print("CoTracker produced no output – returning empty breadcrumbs.")
        del model
        torch.cuda.empty_cache()
        return {}

    # pred_tracks:     (1, T, N, 2)   – N = T (one query per frame)
    # pred_visibility: (1, T, N)      – bool
    tracks_np = pred_tracks[0].cpu().numpy()       # (T, N, 2)
    vis_np = pred_visibility[0].cpu().numpy()       # (T, N)

    # Free GPU memory
    del model, video, pred_tracks, pred_visibility
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Build per-frame breadcrumbs  (forward-time frame_idx -> (M, 2))
    # ------------------------------------------------------------------
    # Mapping between reversed-time index and forward-time frame index:
    #   reversed index r  <->  frame_paths[n_frames - 1 - r]
    from data.track_breadcrumbs import _parse_frame_idx

    breadcrumbs: Dict[int, np.ndarray] = {}
    N = tracks_np.shape[1]  # == T

    for rev_t in range(T):
        fwd_path = frame_paths[n_frames - 1 - rev_t]
        fwd_idx = _parse_frame_idx(fwd_path)

        # Gather all query points visible in this reversed-time frame
        visible = vis_np[rev_t]  # (N,) bool
        pts = tracks_np[rev_t]   # (N, 2)  x, y

        # Each query q was seeded at reversed-time q, so its "age" in this
        # frame is |rev_t - q|. We want at most max_track_len.
        ages = np.abs(np.arange(N) - rev_t)

        keep = visible.astype(bool)

        # Enforce max track length
        if max_track_len > 0:
            keep &= ages <= max_track_len

        pts = pts[keep]

        # Boundary filtering
        if len(pts) > 0:
            inside = (
                (pts[:, 0] >= boundary_thresh)
                & (pts[:, 0] < w - boundary_thresh)
                & (pts[:, 1] >= boundary_thresh)
                & (pts[:, 1] < h - boundary_thresh)
            )
            pts = pts[inside]

        breadcrumbs[fwd_idx] = pts.astype(np.float32)

    print(f"CoTracker breadcrumbs generated for {len(breadcrumbs)} frames.")
    return breadcrumbs
