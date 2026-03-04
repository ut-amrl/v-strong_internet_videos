"""
SAM Mask Generation & Positive/Negative Point Sampling.

Given breadcrumb points from LK optical-flow tracking, this script:
  1. Runs SAM on each frame using breadcrumbs as foreground prompts.
  2. Produces a binary traversability mask per frame.
  3. Samples positive points from inside the mask (traversable), with an
     optional safety margin via mask erosion.
  4. Samples negative points from outside the mask (non-traversable),
     restricted to the bottom portion of the frame to avoid sky, and
     with an optional safety margin via mask dilation.
  5. Saves masks, point samples, and overlay visualizations.

Usage:
    python src/data/sam_mask_generator.py \
        --frames_dir data/output_2/frames \
        --breadcrumbs_dir data/output_2/breadcrumbs \
        --output_dir data/output_2 \
        --checkpoint checkpoints/sam_vit_b_01ec64.pth
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

# ── SAM import ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "third_party" / "segment-anything"))
from segment_anything import sam_model_registry, SamPredictor

try:
    from .track_breadcrumbs import get_sorted_frame_paths, load_breadcrumbs
except Exception:  # pragma: no cover
    from track_breadcrumbs import get_sorted_frame_paths, load_breadcrumbs


# ────────────────────────────────────────────────────────────────────────
def _load_sam_predictor(checkpoint: str, sam_type: str) -> SamPredictor:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[sam_type](checkpoint=checkpoint)
    sam.to(device)
    sam.eval()
    predictor = SamPredictor(sam)
    print(f"  SAM loaded on {device}")
    return predictor


def _parse_frame_idx(frame_path: Path) -> int:
    try:
        return int(frame_path.stem)
    except ValueError:
        raise RuntimeError(
            f"Frame filename must be an integer like 000123.jpg, got: {frame_path.name}"
        )


def _select_frame_paths(
    frame_paths: list[Path],
    frame_start: int | None,
    frame_end: int | None,
    max_frames: int | None,
) -> list[Path]:
    selected: list[Path] = []
    for p in frame_paths:
        idx = _parse_frame_idx(p)
        if frame_start is not None and idx < frame_start:
            continue
        if frame_end is not None and idx >= frame_end:
            continue
        selected.append(p)

    if max_frames is not None:
        selected = selected[: int(max_frames)]
    return selected


def _clip_points_xy(points_xy: np.ndarray, w: int, h: int) -> np.ndarray:
    if len(points_xy) == 0:
        return points_xy.astype(np.float32, copy=False)
    pts = points_xy.astype(np.float32, copy=False)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    return pts


def _subsample_points(points_xy: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if max_points <= 0 or len(points_xy) <= max_points:
        return points_xy
    idxs = rng.choice(len(points_xy), size=max_points, replace=False)
    return points_xy[idxs]


def _morph_kernel(radius_px: int) -> np.ndarray | None:
    if radius_px <= 0:
        return None
    k = 2 * radius_px + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _sample_pixels(mask_bool: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    ys, xs = np.where(mask_bool)
    if len(ys) == 0 or k <= 0:
        return np.empty((0, 2), dtype=np.float32)
    n = min(k, len(ys))
    idxs = rng.choice(len(ys), size=n, replace=False)
    return np.stack([xs[idxs], ys[idxs]], axis=1).astype(np.float32)


def sample_pos_neg_points(
    mask01: np.ndarray,
    num_pos: int,
    num_neg: int,
    neg_top_frac: float,
    sample_margin_px: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample positives and negatives from a binary mask.

    - Positives are sampled from an eroded mask (margin away from boundary).
    - Negatives are sampled from outside a dilated mask (margin away from boundary),
      and restricted to the bottom (1 - neg_top_frac) of the image.
    - If margin-based candidates are empty, fall back to raw inside/outside.
    """
    h, w = mask01.shape[:2]
    mask_u8 = (mask01.astype(np.uint8) * 255)

    radius = int(max(0, sample_margin_px))
    kernel = _morph_kernel(radius)

    if kernel is not None:
        pos_u8 = cv2.erode(mask_u8, kernel, iterations=1)
        neg_block_u8 = cv2.dilate(mask_u8, kernel, iterations=1)
    else:
        pos_u8 = mask_u8
        neg_block_u8 = mask_u8

    # Positives
    pos_points = _sample_pixels(pos_u8 > 0, num_pos, rng)
    if len(pos_points) == 0 and kernel is not None:
        pos_points = _sample_pixels(mask_u8 > 0, num_pos, rng)

    # Negatives (exclude sky / top region)
    neg_top = int(h * float(neg_top_frac))
    neg_mask_margin = (neg_block_u8 == 0)
    neg_mask_margin[:neg_top, :] = False
    neg_points = _sample_pixels(neg_mask_margin, num_neg, rng)

    if len(neg_points) == 0 and kernel is not None:
        neg_mask_raw = (mask_u8 == 0)
        neg_mask_raw[:neg_top, :] = False
        neg_points = _sample_pixels(neg_mask_raw, num_neg, rng)

    return pos_points, neg_points


# ────────────────────────────────────────────────────────────────────────
def run_sam_pipeline(
    frames_dir: str,
    breadcrumbs_dir: str,
    output_dir: str,
    checkpoint: str,
    sam_type: str = "vit_b",
    prompt_max_points: int = 32,
    num_pos: int = 50,
    num_neg: int = 50,
    neg_top_frac: float = 0.3,  # exclude top 30% of frame for neg sampling
    sample_margin_px: int = 10,
    seed: int = 0,
    resume: bool = False,
    frame_start: int | None = None,
    frame_end: int | None = None,
    max_frames: int | None = None,
):
    """
    Full pipeline: saved breadcrumbs → SAM masks → pos/neg point sampling.
    """
    output_dir = Path(output_dir)
    masks_dir = output_dir / "masks"
    points_dir = output_dir / "pos_neg_points"
    viz_dir = output_dir / "sam_viz"
    masks_dir.mkdir(parents=True, exist_ok=True)
    points_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load frames ─────────────────────────────────────────────────
    frame_paths = get_sorted_frame_paths(frames_dir)
    frame_paths = _select_frame_paths(
        frame_paths,
        frame_start=frame_start,
        frame_end=frame_end,
        max_frames=max_frames,
    )
    n_frames = len(frame_paths)
    print(f"\n{'='*60}")
    print(f"  Frames : {n_frames}")
    print(f"  SAM    : {sam_type}  ({checkpoint})")
    print(f"  Prompts: max {prompt_max_points} breadcrumb points")
    print(f"  Pos/Neg: {num_pos} / {num_neg} per frame")
    print(f"  Margin : {sample_margin_px}px (erosion/dilation)")
    print(f"  Resume : {resume}")
    if frame_start is not None or frame_end is not None:
        print(f"  Range  : [{frame_start or 0}, {frame_end if frame_end is not None else '∞'}) by filename idx")
    if max_frames is not None:
        print(f"  Max    : {max_frames} frames")
    print(f"{'='*60}\n")

    # ── 2. Load breadcrumbs ───────────────────────────────────────────
    print("STEP 1/3: Loading breadcrumbs …")
    breadcrumbs = load_breadcrumbs(breadcrumbs_dir)

    # ── 3. Load SAM ────────────────────────────────────────────────────
    print("\nSTEP 2/3: Loading SAM …")
    predictor = _load_sam_predictor(checkpoint=checkpoint, sam_type=sam_type)

    # ── 4. Per-frame: SAM → mask → sample ──────────────────────────────
    print("\nSTEP 3/3: Generating masks & sampling points …")
    stats = {"total_pos": 0, "total_neg": 0, "frames_processed": 0}

    iterable = tqdm(frame_paths, desc="SAM masks + pos/neg", unit="frame") if tqdm is not None else frame_paths
    for frame_path in iterable:
        frame_idx = _parse_frame_idx(frame_path)
        out_mask_path = masks_dir / f"{frame_idx:06d}.png"
        out_points_path = points_dir / f"{frame_idx:06d}.npz"
        out_viz_path = viz_dir / f"{frame_idx:06d}.jpg"

        if resume and out_mask_path.exists() and out_points_path.exists():
            continue

        img_bgr = cv2.imread(str(frame_path))
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Get breadcrumb points for this frame
        bc_pts = breadcrumbs.get(frame_idx, np.empty((0, 2), dtype=np.float32))
        rng = np.random.default_rng(int(seed) + int(frame_idx))

        if len(bc_pts) < 1:
            mask01 = np.zeros((h, w), dtype=np.uint8)
            pos_points = np.empty((0, 2), dtype=np.float32)
            neg_points = np.empty((0, 2), dtype=np.float32)
        else:
            prompt_pts = _clip_points_xy(bc_pts[:, :2], w=w, h=h)
            prompt_pts = _subsample_points(prompt_pts, max_points=int(prompt_max_points), rng=rng)

            if len(prompt_pts) < 1:
                mask01 = np.zeros((h, w), dtype=np.uint8)
                pos_points = np.empty((0, 2), dtype=np.float32)
                neg_points = np.empty((0, 2), dtype=np.float32)
            else:
                predictor.set_image(img_rgb)

                point_coords = prompt_pts.astype(np.float32, copy=False)
                point_labels = np.ones(len(point_coords), dtype=np.int32)  # all foreground

                with torch.no_grad():
                    masks_out, scores, _ = predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        multimask_output=True,
                    )
                best_idx = int(np.argmax(scores))
                mask01 = masks_out[best_idx].astype(np.uint8)  # (H, W) 0/1

                pos_points, neg_points = sample_pos_neg_points(
                    mask01=mask01,
                    num_pos=int(num_pos),
                    num_neg=int(num_neg),
                    neg_top_frac=float(neg_top_frac),
                    sample_margin_px=int(sample_margin_px),
                    rng=rng,
                )

        # ── Save ────────────────────────────────────────────────────
        cv2.imwrite(str(out_mask_path), (mask01 * 255).astype(np.uint8, copy=False))
        np.savez_compressed(
            str(out_points_path),
            pos_points=pos_points,
            neg_points=neg_points,
        )

        # ── Visualization overlay ───────────────────────────────────
        viz = img_bgr.copy()
        # Semi-transparent mask overlay (blue tint for traversable)
        overlay = viz.copy()
        overlay[mask01 > 0] = (overlay[mask01 > 0] * 0.5 + np.array([200, 100, 0]) * 0.5).astype(np.uint8)
        viz = overlay

        # Draw breadcrumb points (white, small)
        for bx, by in bc_pts[:, :2]:
            cv2.circle(viz, (int(bx), int(by)), 4, (255, 255, 255), -1)

        # Draw positive points (green)
        for px, py in pos_points:
            cv2.circle(viz, (int(px), int(py)), 6, (0, 255, 0), -1)

        # Draw negative points (red)
        for nx, ny in neg_points:
            cv2.circle(viz, (int(nx), int(ny)), 6, (0, 0, 255), -1)

        cv2.putText(
            viz,
            f"Frame {frame_idx} | mask {int(mask01.sum())}px | +{len(pos_points)} -{len(neg_points)}",
            (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3,
        )
        cv2.imwrite(str(out_viz_path), viz)

        stats["total_pos"] += len(pos_points)
        stats["total_neg"] += len(neg_points)
        stats["frames_processed"] += 1

        if tqdm is not None and hasattr(iterable, "set_postfix_str"):
            iterable.set_postfix_str(f"frame={frame_idx} +{len(pos_points)} -{len(neg_points)}")

    print(f"\n{'='*60}")
    print(f"  Done! {stats['frames_processed']} frames processed")
    print(f"  Total pos points: {stats['total_pos']}")
    print(f"  Total neg points: {stats['total_neg']}")
    print(f"  Masks     → {masks_dir}")
    print(f"  Points    → {points_dir}")
    print(f"  Viz       → {viz_dir}")
    print(f"{'='*60}")


# ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="SAM mask generation & pos/neg point sampling"
    )
    parser.add_argument("--frames_dir", type=str, required=True)
    parser.add_argument("--breadcrumbs_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--checkpoint", type=str,
        default="checkpoints/sam_vit_b_01ec64.pth",
    )
    parser.add_argument("--sam_type", type=str, default="vit_b",
                        choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--prompt_max_points", type=int, default=32)
    parser.add_argument("--num_pos", type=int, default=50)
    parser.add_argument("--num_neg", type=int, default=50)
    parser.add_argument("--neg_top_frac", type=float, default=0.3,
                        help="Exclude top fraction of frame for neg sampling")
    parser.add_argument("--sample_margin_px", type=int, default=10,
                        help="Safety margin in px via erosion/dilation (default: 10)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base RNG seed for deterministic sampling (default: 0)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip frames whose mask+points already exist")
    parser.add_argument("--frame_start", type=int, default=None,
                        help="Start frame index (inclusive) by filename integer (e.g. 0)")
    parser.add_argument("--frame_end", type=int, default=None,
                        help="End frame index (exclusive) by filename integer (e.g. 50)")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Max number of frames to process after range filtering")
    args = parser.parse_args()

    run_sam_pipeline(
        frames_dir=args.frames_dir,
        breadcrumbs_dir=args.breadcrumbs_dir,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        sam_type=args.sam_type,
        prompt_max_points=args.prompt_max_points,
        num_pos=args.num_pos,
        num_neg=args.num_neg,
        neg_top_frac=args.neg_top_frac,
        sample_margin_px=args.sample_margin_px,
        seed=args.seed,
        resume=args.resume,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
