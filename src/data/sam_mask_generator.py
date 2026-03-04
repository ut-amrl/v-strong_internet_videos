"""SAM mask generation + positive/negative point sampling.

Two main workflows live here:

1) **Automatic mask generation** (legacy/simple):
   - ``generate_mask`` uses SAM automatic mask generation and selects the largest mask.

2) **Breadcrumb-prompted SAM** (ported from `origin/modular_net`):
   - ``run_sam_pipeline`` uses LK-tracked breadcrumb points as *foreground prompts* to
     ``SamPredictor`` to produce a traversability mask per frame, then samples pos/neg
     points from that mask.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

# Allow importing segment_anything from pip install or third_party/
_third_party_sam = str(Path(__file__).resolve().parents[2] / "third_party" / "segment-anything")
if _third_party_sam not in sys.path:
    sys.path.insert(0, _third_party_sam)


def generate_mask(
    image_rgb: np.ndarray,
    sam_checkpoint: str,
    sam_type: str = "vit_b",
    prompt_max_points: int = 32,
) -> np.ndarray:
    """Generate a binary traversability mask using SAM automatic mask generator.

    Uses the largest connected mask component from SAM's automatic segmentation
    as the traversable region.

    Parameters
    ----------
    image_rgb : np.ndarray
        HxWx3 uint8 RGB image.
    sam_checkpoint : str
        Path to SAM model checkpoint.
    sam_type : str
        SAM model type (vit_b, vit_l, vit_h).
    prompt_max_points : int
        Max points per side for automatic mask generation.

    Returns
    -------
    np.ndarray
        HxW uint8 binary mask (0 or 1).
    """
    import torch
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[sam_type](checkpoint=sam_checkpoint)
    sam.to(device)
    sam.eval()

    generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=prompt_max_points,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        min_mask_region_area=100,
    )

    masks = generator.generate(image_rgb)
    if not masks:
        # Fallback: bottom half of image as "traversable"
        h, w = image_rgb.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[h // 2 :, :] = 1
        return mask

    # Pick largest mask as traversable region
    largest = max(masks, key=lambda m: m["area"])
    return largest["segmentation"].astype(np.uint8)


def sample_pos_neg_points(
    mask01: np.ndarray,
    num_pos: int,
    num_neg: int,
    neg_top_frac: float = 0.3,
    sample_margin_px: int = 10,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample positive and negative (x, y) points from a binary mask.

    Ported semantics (from `origin/modular_net`):
    - Positives are sampled from an *eroded* mask (margin away from boundary).
    - Negatives are sampled from outside a *dilated* mask (margin away from boundary),
      and restricted to the bottom portion of the image by excluding the top
      `neg_top_frac` fraction (to avoid trivial sky negatives).

    Returns best-effort samples (can be fewer than requested if candidates are scarce).
    """
    if rng is None:
        rng = np.random.default_rng()

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
    pos_points = _sample_pixels(pos_u8 > 0, int(num_pos), rng)
    if len(pos_points) == 0 and kernel is not None:
        pos_points = _sample_pixels(mask_u8 > 0, int(num_pos), rng)

    # Negatives (exclude top region)
    neg_top = int(h * float(neg_top_frac))
    neg_mask_margin = (neg_block_u8 == 0)
    neg_mask_margin[:neg_top, :] = False
    neg_points = _sample_pixels(neg_mask_margin, int(num_neg), rng)

    if len(neg_points) == 0 and kernel is not None:
        neg_mask_raw = (mask_u8 == 0)
        neg_mask_raw[:neg_top, :] = False
        neg_points = _sample_pixels(neg_mask_raw, int(num_neg), rng)

    return pos_points, neg_points


def _morph_kernel(radius_px: int) -> np.ndarray | None:
    if radius_px <= 0:
        return None
    k = 2 * radius_px + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _sample_pixels(mask_bool: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    ys, xs = np.where(mask_bool)
    if len(ys) == 0 or k <= 0:
        return np.empty((0, 2), dtype=np.float32)
    n = min(int(k), len(ys))
    idxs = rng.choice(len(ys), size=n, replace=False)
    return np.stack([xs[idxs], ys[idxs]], axis=1).astype(np.float32)


# ────────────────────────────────────────────────────────────────────────
# Breadcrumb-prompted SAM pipeline (ported from origin/modular_net)


def _load_sam_predictor(checkpoint: str, sam_type: str):
    import torch
    from segment_anything import SamPredictor, sam_model_registry

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[sam_type](checkpoint=checkpoint)
    sam.to(device)
    sam.eval()
    return SamPredictor(sam), device


def _parse_frame_idx(frame_path: Path) -> int:
    try:
        return int(frame_path.stem)
    except ValueError as e:
        raise RuntimeError(
            f"Frame filename must be an integer like 000123.jpg, got: {frame_path.name}"
        ) from e


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
    idxs = rng.choice(len(points_xy), size=int(max_points), replace=False)
    return points_xy[idxs]


def run_sam_pipeline(
    *,
    frames_dir: str,
    breadcrumbs_dir: str,
    output_dir: str,
    checkpoint: str,
    sam_type: str = "vit_b",
    prompt_max_points: int = 32,
    num_pos: int = 50,
    num_neg: int = 50,
    neg_top_frac: float = 0.3,
    sample_margin_px: int = 10,
    seed: int = 0,
    resume: bool = False,
    frame_start: int | None = None,
    frame_end: int | None = None,
    max_frames: int | None = None,
    write_viz: bool = True,
) -> None:
    """Breadcrumb-prompted SAM masks + pos/neg sampling for all frames in `frames_dir`."""
    from data.track_breadcrumbs import get_sorted_frame_paths, load_breadcrumbs

    out_dir = Path(output_dir)
    masks_dir = out_dir / "masks"
    points_dir = out_dir / "pos_neg_points"
    viz_dir = out_dir / "sam_viz"
    masks_dir.mkdir(parents=True, exist_ok=True)
    points_dir.mkdir(parents=True, exist_ok=True)
    if write_viz:
        viz_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = get_sorted_frame_paths(frames_dir)
    frame_paths = _select_frame_paths(
        frame_paths,
        frame_start=frame_start,
        frame_end=frame_end,
        max_frames=max_frames,
    )

    breadcrumbs = load_breadcrumbs(breadcrumbs_dir)
    predictor, device = _load_sam_predictor(checkpoint=checkpoint, sam_type=sam_type)

    try:
        from tqdm import tqdm
    except Exception:  # pragma: no cover
        tqdm = None

    iterable = tqdm(frame_paths, desc="SAM masks + pos/neg", unit="frame") if tqdm is not None else frame_paths
    for frame_path in iterable:
        frame_idx = _parse_frame_idx(frame_path)
        out_mask_path = masks_dir / f"{frame_idx:06d}.png"
        out_points_path = points_dir / f"{frame_idx:06d}.npz"
        out_viz_path = viz_dir / f"{frame_idx:06d}.jpg"

        if resume and out_mask_path.exists() and out_points_path.exists():
            continue

        img_bgr = cv2.imread(str(frame_path))
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

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

        cv2.imwrite(str(out_mask_path), (mask01 * 255).astype(np.uint8, copy=False))
        np.savez_compressed(
            str(out_points_path),
            pos_points=pos_points,
            neg_points=neg_points,
        )

        if write_viz:
            viz = img_bgr.copy()
            overlay = viz.copy()
            overlay[mask01 > 0] = (
                overlay[mask01 > 0] * 0.5 + np.array([200, 100, 0]) * 0.5
            ).astype(np.uint8)
            viz = overlay

            for bx, by in bc_pts[:, :2]:
                cv2.circle(viz, (int(bx), int(by)), 4, (255, 255, 255), -1)
            for px, py in pos_points:
                cv2.circle(viz, (int(px), int(py)), 6, (0, 255, 0), -1)
            for nx, ny in neg_points:
                cv2.circle(viz, (int(nx), int(ny)), 6, (0, 0, 255), -1)
            cv2.putText(
                viz,
                f"Frame {frame_idx} | mask {int(mask01.sum())}px | +{len(pos_points)} -{len(neg_points)}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 255),
                3,
            )
            cv2.imwrite(str(out_viz_path), viz)

        if tqdm is not None and hasattr(iterable, "set_postfix_str"):
            iterable.set_postfix_str(f"dev={device}")
