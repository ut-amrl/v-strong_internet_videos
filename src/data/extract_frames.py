"""
Extract frames from video files at a configurable frame rate.

Usage:
    python src/data/extract_frames.py --video videos/output_2.mp4 --output data/output_2/frames --fps 5

Note:
    Use `--fps 0` (or any <= 0 value) to extract every frame.
"""

import argparse
import json
import os
from pathlib import Path

import cv2
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def extract_frames(video_path: str, output_dir: str, target_fps: float = 5.0) -> dict:
    """
    Extract frames from a video file at a target FPS.

    Args:
        video_path: Path to the input video file (.mp4, .webm, etc.)
        output_dir: Directory to save extracted frames
        target_fps: Target frames per second for extraction (default: 5)

    Returns:
        Metadata dict with video/extraction info
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    # Video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / original_fps

    # Calculate frame interval for target FPS
    # If target_fps <= 0, extract every frame.
    if target_fps <= 0:
        frame_interval = 1
    else:
        frame_interval = max(1, round(original_fps / target_fps))
    effective_fps = original_fps / frame_interval

    print(f"Video: {video_path.name}")
    print(f"  Original: {width}x{height} @ {original_fps:.2f} FPS, {total_frames} frames, {duration:.1f}s")
    print(f"  Extracting every {frame_interval} frames → ~{effective_fps:.2f} FPS")

    extracted_count = 0
    frame_idx = 0

    pbar = None
    if tqdm is not None and total_frames > 0:
        pbar = tqdm(total=total_frames, desc="Extracting frames", unit="frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            out_path = output_dir / f"{extracted_count:06d}.jpg"
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            extracted_count += 1

        frame_idx += 1
        if pbar is not None:
            pbar.update(1)
            if extracted_count % 50 == 0:
                pbar.set_postfix_str(f"saved={extracted_count}")

    cap.release()
    if pbar is not None:
        pbar.close()

    print(f"  Extracted {extracted_count} frames to {output_dir}")

    # Save metadata
    metadata = {
        "video_name": video_path.stem,
        "video_path": str(video_path),
        "original_fps": original_fps,
        "target_fps": target_fps,
        "effective_fps": effective_fps,
        "frame_interval": frame_interval,
        "width": width,
        "height": height,
        "total_original_frames": total_frames,
        "extracted_frames": extracted_count,
        "duration_seconds": duration,
    }

    metadata_path = output_dir.parent / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved to {metadata_path}")

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Extract frames from video at target FPS")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--output", type=str, required=True, help="Output directory for frames")
    parser.add_argument("--fps", type=float, default=5.0, help="Target FPS for extraction (default: 5)")
    args = parser.parse_args()

    extract_frames(args.video, args.output, args.fps)


if __name__ == "__main__":
    main()
