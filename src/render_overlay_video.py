import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from data.sam_mask_generator import sample_pos_neg_points
from data.vstrong_dataset import ResizeSquare
from models.vstrong_lit import VStrongLit
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def _parse_frame_idx(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError as e:
        raise RuntimeError(f"Expected integer frame filename like 000123.jpg, got {path.name}") from e


def _load_effective_fps(dataset_dir: Path, default_fps: float) -> float:
    meta = dataset_dir / "metadata.json"
    if not meta.exists():
        return float(default_fps)
    try:
        data = json.load(open(meta, "r"))
        return float(data.get("effective_fps", default_fps))
    except Exception:
        return float(default_fps)


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


def _ensure_dataset(
    video_path: Path | None,
    dataset_dir: Path,
    fps: float,
    sam_ckpt: Path,
    sam_type: str,
    prompt_max_points: int,
    num_pos: int,
    num_neg: int,
    neg_top_frac: float,
    sample_margin_px: int,
    regenerate: bool,
):
    if dataset_dir.exists() and not regenerate:
        return

    if video_path is None:
        raise RuntimeError("video_path is required when generating a dataset")
    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    if dataset_dir.exists():
        # Be safe: only delete under data/ or /tmp
        ds_str = str(dataset_dir)
        if not (ds_str.startswith("data/") or ds_str.startswith("/tmp/")):
            raise RuntimeError(f"Refusing to delete dataset_dir outside data/ or /tmp: {dataset_dir}")
        import shutil

        shutil.rmtree(dataset_dir)

    cmd = [
        sys.executable,
        "src/data/generate_dataset.py",
        "--video",
        str(video_path),
        "--output",
        str(dataset_dir),
        "--fps",
        str(fps),
        "--checkpoint",
        str(sam_ckpt),
        "--sam_type",
        str(sam_type),
        "--prompt_max_points",
        str(prompt_max_points),
        "--num_pos",
        str(num_pos),
        "--num_neg",
        str(num_neg),
        "--neg_top_frac",
        str(neg_top_frac),
        "--sample_margin_px",
        str(sample_margin_px),
    ]
    subprocess.run(cmd, check=True)


def _draw_header(canvas: np.ndarray, title: str, left_label: str, right_label: str):
    h, w = canvas.shape[:2]
    bar_h = max(60, int(h * 0.08))
    out = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    out[bar_h:, :, :] = canvas

    # Title
    cv2.putText(
        out,
        title,
        (20, int(bar_h * 0.7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )

    # Side labels
    half_w = w // 2
    cv2.putText(
        out,
        left_label,
        (20, bar_h + 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        right_label,
        (half_w + 20, bar_h + 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    return out


@torch.no_grad()
def render_video(
    dataset_dir: Path,
    ckpt_path: Path,
    output_path: Path,
    title: str,
    frame_start: int | None,
    frame_end: int | None,
    max_frames: int | None,
    points_per_class: int,
    points_source: str,
    neg_top_frac: float,
    sample_margin_px: int,
    output_fps: float,
    output_height: int | None,
):
    frames_dir = dataset_dir / "frames"
    points_dir = dataset_dir / "pos_neg_points"
    masks_dir = dataset_dir / "masks"

    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if len(frame_paths) == 0:
        raise RuntimeError(f"No frames found in {frames_dir}")

    selected: list[Path] = []
    for p in frame_paths:
        idx = _parse_frame_idx(p)
        if frame_start is not None and idx < frame_start:
            continue
        if frame_end is not None and idx >= frame_end:
            continue
        if not p.exists():
            continue
        selected.append(p)
    if max_frames is not None:
        selected = selected[: int(max_frames)]
    if len(selected) == 0:
        raise RuntimeError("No frames selected for rendering.")

    model = VStrongLit.load_from_checkpoint(str(ckpt_path), strict=False)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    transform = ResizeSquare(int(getattr(model, "img_size", getattr(model, "sam_img_size", 1024))))

    first_rgb = cv2.imread(str(selected[0]))
    first_rgb = cv2.cvtColor(first_rgb, cv2.COLOR_BGR2RGB)
    resized_first = transform.apply_image(first_rgb)
    h0, w0 = resized_first.shape[:2]

    if output_height is not None:
        scale = float(output_height) / float(h0)
        w_out = int(round(w0 * scale))
        h_out = int(output_height)
    else:
        w_out = w0
        h_out = h0

    # side-by-side
    canvas_w = w_out * 2
    canvas_h = h_out

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, float(output_fps), (canvas_w, canvas_h + max(60, int(canvas_h * 0.08))))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")

    rng_base = 0
    wrote = 0
    skipped = 0
    iterable = tqdm(selected, desc="Rendering overlay video", unit="frame") if tqdm is not None else selected
    for i, frame_path in enumerate(iterable):
        frame_idx = _parse_frame_idx(frame_path)
        rng = np.random.default_rng(rng_base + frame_idx)

        bgr = cv2.imread(str(frame_path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized_rgb = transform.apply_image(rgb)

        pts_path = points_dir / f"{frame_idx:06d}.npz"
        mask_path = masks_dir / f"{frame_idx:06d}.png"

        saved_pos = np.empty((0, 2), dtype=np.float32)
        if pts_path.exists():
            pts = np.load(str(pts_path))
            saved_pos = pts.get("pos_points", saved_pos).astype(np.float32, copy=False)

        mask01 = None
        if mask_path.exists():
            mask_u8 = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_u8 is not None:
                mask01 = (mask_u8 > 0).astype(np.uint8)

        if points_source == "saved":
            pos_xy = _sample_fixed_k(saved_pos, points_per_class, rng)
        elif points_source == "mask" and mask01 is not None:
            pos_xy, _ = sample_pos_neg_points(
                mask01=mask01,
                num_pos=points_per_class,
                num_neg=points_per_class,
                neg_top_frac=neg_top_frac,
                sample_margin_px=sample_margin_px,
                rng=rng,
            )
            pos_xy = _sample_fixed_k(pos_xy, points_per_class, rng)
        else:
            pos_xy = saved_pos
            if len(pos_xy) < points_per_class and mask01 is not None:
                extra, _ = sample_pos_neg_points(
                    mask01=mask01,
                    num_pos=points_per_class - len(pos_xy),
                    num_neg=0,
                    neg_top_frac=neg_top_frac,
                    sample_margin_px=sample_margin_px,
                    rng=rng,
                )
                if len(extra):
                    pos_xy = np.concatenate([pos_xy, extra], axis=0)
            pos_xy = _sample_fixed_k(pos_xy, points_per_class, rng)

        # transform pos points to resized coords
        h, w = rgb.shape[:2]
        pos_resized = transform.apply_coords(pos_xy, (h, w)).astype(np.float32)

        emb = model._encode_images([resized_rgb]).to(device)
        z_map = model.proj(emb)[0]
        viz = model._make_viz(resized_rgb=resized_rgb, z_map=z_map, pos_points=torch.from_numpy(pos_resized).to(device))
        left_bgr = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR)
        if viz is None:
            right_bgr = left_bgr.copy()
            cv2.putText(
                right_bgr,
                "NO POS POINTS",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,
                (255, 255, 255),
                4,
                cv2.LINE_AA,
            )
            skipped += 1
        else:
            right_bgr = viz.overlay_bgr

        if (left_bgr.shape[0], left_bgr.shape[1]) != (h_out, w_out):
            left_bgr = cv2.resize(left_bgr, (w_out, h_out), interpolation=cv2.INTER_AREA)
        if (right_bgr.shape[0], right_bgr.shape[1]) != (h_out, w_out):
            right_bgr = cv2.resize(right_bgr, (w_out, h_out), interpolation=cv2.INTER_AREA)

        canvas = np.concatenate([left_bgr, right_bgr], axis=1)
        heading = f"{title} | frame {frame_idx:06d}"
        framed = _draw_header(canvas, heading, "RGB", "Traversability overlay")

        writer.write(framed)
        wrote += 1
        if tqdm is not None and hasattr(iterable, "set_postfix_str"):
            iterable.set_postfix_str(f"wrote={wrote} skipped={skipped}")
        elif i % 50 == 0 or i == len(selected) - 1:
            print(f"  Rendered {i + 1}/{len(selected)} frames")

    writer.release()
    if skipped:
        print(f"Note: {skipped} frames had no positive points; wrote RGB fallback for those.")
    print(f"Done. Wrote {wrote} frames to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Render side-by-side RGB + traversability overlay video.")
    parser.add_argument("--video", type=str, default=None, help="Input video path (optional if dataset_dir exists).")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Existing dataset dir; if absent, one is generated.")
    parser.add_argument("--output", type=str, required=True, help="Output mp4 path.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Trained VStrongLit checkpoint (e.g. last.ckpt).")

    parser.add_argument("--title", type=str, default="V-STRONG")
    parser.add_argument("--fps", type=float, default=5.0, help="FPS for frame extraction when generating dataset.")
    parser.add_argument("--output_fps", type=float, default=None, help="Output FPS (defaults to dataset effective_fps).")
    parser.add_argument("--output_height", type=int, default=None, help="Resize output height for both panels.")

    parser.add_argument("--frame_start", type=int, default=None)
    parser.add_argument("--frame_end", type=int, default=None)
    parser.add_argument("--max_frames", type=int, default=None)

    # Dataset generation params
    parser.add_argument("--regenerate", action="store_true", help="Force regenerate dataset_dir if it exists.")
    parser.add_argument("--sam_ckpt", type=str, default="checkpoints/sam_vit_b_01ec64.pth")
    parser.add_argument("--sam_type", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--prompt_max_points", type=int, default=32)
    parser.add_argument("--num_pos", type=int, default=50)
    parser.add_argument("--num_neg", type=int, default=50)
    parser.add_argument("--neg_top_frac", type=float, default=0.3)
    parser.add_argument("--sample_margin_px", type=int, default=10)

    # Inference points
    parser.add_argument("--points_per_class", type=int, default=64)
    parser.add_argument("--points_source", type=str, default="mixed", choices=["saved", "mask", "mixed"])
    args = parser.parse_args()

    video_path = Path(args.video) if args.video is not None else None
    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        raise RuntimeError(f"ckpt_path does not exist: {ckpt_path}")

    if args.dataset_dir is None:
        if video_path is None:
            raise RuntimeError("Provide --video or --dataset_dir")
        ts = time.strftime("%Y%m%d_%H%M%S")
        dataset_dir = Path("/tmp") / "vstrong_dataset" / f"{video_path.stem}_{ts}"
    else:
        dataset_dir = Path(args.dataset_dir)

    _ensure_dataset(
        video_path=video_path,
        dataset_dir=dataset_dir,
        fps=args.fps,
        sam_ckpt=Path(args.sam_ckpt),
        sam_type=args.sam_type,
        prompt_max_points=args.prompt_max_points,
        num_pos=args.num_pos,
        num_neg=args.num_neg,
        neg_top_frac=args.neg_top_frac,
        sample_margin_px=args.sample_margin_px,
        regenerate=args.regenerate,
    )

    out_fps = float(args.output_fps) if args.output_fps is not None else _load_effective_fps(dataset_dir, args.fps)

    render_video(
        dataset_dir=dataset_dir,
        ckpt_path=ckpt_path,
        output_path=Path(args.output),
        title=args.title,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        max_frames=args.max_frames,
        points_per_class=args.points_per_class,
        points_source=args.points_source,
        neg_top_frac=args.neg_top_frac,
        sample_margin_px=args.sample_margin_px,
        output_fps=out_fps,
        output_height=args.output_height,
    )


if __name__ == "__main__":
    main()
