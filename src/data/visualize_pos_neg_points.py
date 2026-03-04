"""
Visualize sampled positive/negative points (and optional mask overlay) on frames.

This is useful for QA without re-running SAM.

Usage:
    python src/data/visualize_pos_neg_points.py \
        --frames_dir data/output_2/frames \
        --points_dir data/output_2/pos_neg_points \
        --masks_dir data/output_2/masks \
        --output_dir data/output_2/pos_neg_viz \
        --frame_start 0 --frame_end 50
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def _parse_frame_idx(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError as e:
        raise RuntimeError(f"Expected integer filename like 000123.jpg, got {path.name}") from e


def visualize_pos_neg_points(
    frames_dir: str,
    points_dir: str,
    output_dir: str,
    masks_dir: str | None = None,
    frame_start: int | None = None,
    frame_end: int | None = None,
    max_frames: int | None = None,
    alpha: float = 0.5,
    pos_radius: int = 6,
    neg_radius: int = 6,
    bc_radius: int = 0,
    make_video: bool = False,
    video_fps: int = 10,
):
    frames_dir_p = Path(frames_dir)
    points_dir_p = Path(points_dir)
    output_dir_p = Path(output_dir)
    output_dir_p.mkdir(parents=True, exist_ok=True)
    masks_dir_p = Path(masks_dir) if masks_dir is not None else None

    frame_paths = sorted(frames_dir_p.glob("*.jpg"))
    if len(frame_paths) == 0:
        raise RuntimeError(f"No frames found in {frames_dir_p}")

    selected: list[Path] = []
    for p in frame_paths:
        idx = _parse_frame_idx(p)
        if frame_start is not None and idx < frame_start:
            continue
        if frame_end is not None and idx >= frame_end:
            continue
        if not (points_dir_p / f"{idx:06d}.npz").exists():
            continue
        selected.append(p)
    if max_frames is not None:
        selected = selected[: int(max_frames)]

    if len(selected) == 0:
        raise RuntimeError("No frames selected (check points_dir and range flags).")

    writer = None
    if make_video:
        sample = cv2.imread(str(selected[0]))
        h, w = sample.shape[:2]
        out_video = str(output_dir_p / "pos_neg_viz.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_video, fourcc, float(video_fps), (w, h))

    for i, frame_path in enumerate(selected):
        idx = _parse_frame_idx(frame_path)
        frame = cv2.imread(str(frame_path))
        h, w = frame.shape[:2]

        pts_path = points_dir_p / f"{idx:06d}.npz"
        pts = np.load(str(pts_path))
        pos_points = pts.get("pos_points", np.empty((0, 2), dtype=np.float32))
        neg_points = pts.get("neg_points", np.empty((0, 2), dtype=np.float32))

        viz = frame.copy()

        if masks_dir_p is not None:
            mask_path = masks_dir_p / f"{idx:06d}.png"
            if mask_path.exists():
                mask_u8 = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask_u8 is not None:
                    overlay = viz.copy()
                    overlay[mask_u8 > 0] = (overlay[mask_u8 > 0] * 0.5 + np.array([200, 100, 0]) * 0.5).astype(
                        np.uint8
                    )
                    viz = cv2.addWeighted(overlay, alpha, viz, 1.0 - alpha, 0)

        # Draw positives (green) and negatives (red)
        for x, y in pos_points:
            cv2.circle(viz, (int(x), int(y)), int(pos_radius), (0, 255, 0), -1)
        for x, y in neg_points:
            cv2.circle(viz, (int(x), int(y)), int(neg_radius), (0, 0, 255), -1)

        cv2.putText(
            viz,
            f"Frame {idx} | +{len(pos_points)} -{len(neg_points)}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255),
            3,
        )

        out_path = output_dir_p / f"{idx:06d}.jpg"
        cv2.imwrite(str(out_path), viz, [cv2.IMWRITE_JPEG_QUALITY, 95])

        if writer is not None:
            writer.write(viz)

        if i % 50 == 0 or i == len(selected) - 1:
            print(f"  Wrote {i + 1}/{len(selected)}: {out_path}")

    if writer is not None:
        writer.release()
        print(f"Video saved to {output_dir_p / 'pos_neg_viz.mp4'}")
    print(f"Done. Images saved to {output_dir_p}")


def main():
    parser = argparse.ArgumentParser(description="Visualize pos/neg points (and optional mask overlay).")
    parser.add_argument("--frames_dir", type=str, required=True)
    parser.add_argument("--points_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--masks_dir", type=str, default=None)
    parser.add_argument("--frame_start", type=int, default=None)
    parser.add_argument("--frame_end", type=int, default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--pos_radius", type=int, default=6)
    parser.add_argument("--neg_radius", type=int, default=6)
    parser.add_argument("--make_video", action="store_true")
    parser.add_argument("--video_fps", type=int, default=10)
    args = parser.parse_args()

    visualize_pos_neg_points(
        frames_dir=args.frames_dir,
        points_dir=args.points_dir,
        output_dir=args.output_dir,
        masks_dir=args.masks_dir,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        max_frames=args.max_frames,
        alpha=args.alpha,
        pos_radius=args.pos_radius,
        neg_radius=args.neg_radius,
        make_video=args.make_video,
        video_fps=args.video_fps,
    )


if __name__ == "__main__":
    main()

