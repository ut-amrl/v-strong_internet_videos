"""
Visualize breadcrumb points overlaid on extracted frames.

Usage:
    python src/data/visualize_breadcrumbs.py \
        --frames_dir data/output_2/frames \
        --breadcrumbs_dir data/output_2/breadcrumbs \
        --output data/output_2/viz \
        --num_samples 10
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def visualize_breadcrumbs(
    frames_dir: str,
    breadcrumbs_dir: str,
    output_dir: str,
    num_samples: int = 10,
    dot_radius: int = 5,
    dot_color: tuple = (0, 255, 0),  # Green in BGR
    alpha: float = 0.6,
):
    """
    Overlay breadcrumb points on frames and save visualization images.

    Args:
        frames_dir: Directory containing extracted frame images
        breadcrumbs_dir: Directory containing .npz breadcrumb files
        output_dir: Directory to save visualization images
        num_samples: Number of frames to visualize (evenly spaced)
        dot_radius: Radius of the breadcrumb dots
        dot_color: BGR color for dots
        alpha: Opacity of the overlay
    """
    frames_dir = Path(frames_dir)
    breadcrumbs_dir = Path(breadcrumbs_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all frame-breadcrumb pairs
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    bc_paths = sorted(breadcrumbs_dir.glob("*_breadcrumbs.npz"))

    if len(frame_paths) == 0:
        raise RuntimeError(f"No frame images found in {frames_dir}")
    if len(bc_paths) == 0:
        raise RuntimeError(f"No breadcrumb files found in {breadcrumbs_dir}")

    # Select evenly spaced samples
    n_total = min(len(frame_paths), len(bc_paths))
    if num_samples >= n_total:
        indices = list(range(n_total))
    else:
        indices = np.linspace(0, n_total - 1, num_samples, dtype=int).tolist()

    print(f"Visualizing {len(indices)} frames from {frames_dir}")

    for i, idx in enumerate(indices):
        frame_path = frame_paths[idx]
        bc_path = breadcrumbs_dir / f"{idx:06d}_breadcrumbs.npz"

        if not bc_path.exists():
            print(f"  Skipping frame {idx}: no breadcrumb file found")
            continue

        # Load frame and points
        frame = cv2.imread(str(frame_path))
        data = np.load(str(bc_path))
        points = data["points"]

        # Create overlay
        overlay = frame.copy()
        for pt in points:
            x, y = int(round(pt[0])), int(round(pt[1]))
            cv2.circle(overlay, (x, y), dot_radius, dot_color, -1)

        # Blend
        vis = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Add info text
        text = f"Frame {idx} | {len(points)} breadcrumb points"
        cv2.putText(vis, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        out_path = output_dir / f"viz_{idx:06d}.jpg"
        cv2.imwrite(str(out_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f"Saved {len(indices)} visualization images to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visualize breadcrumb points on frames")
    parser.add_argument("--frames_dir", type=str, required=True, help="Directory of extracted frames")
    parser.add_argument("--breadcrumbs_dir", type=str, required=True, help="Directory of breadcrumb .npz files")
    parser.add_argument("--output", type=str, required=True, help="Output directory for visualization images")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of frames to visualize (default: 10)")
    parser.add_argument("--dot_radius", type=int, default=5, help="Dot radius in pixels (default: 5)")
    args = parser.parse_args()

    visualize_breadcrumbs(
        args.frames_dir,
        args.breadcrumbs_dir,
        args.output,
        num_samples=args.num_samples,
        dot_radius=args.dot_radius,
    )


if __name__ == "__main__":
    main()
