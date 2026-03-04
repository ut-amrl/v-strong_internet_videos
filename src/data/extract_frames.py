"""Extract frames from a video file as numbered JPEG images.

Supports two modes:
  - every frame  (target_fps <= 0)
  - sampled at a target FPS

Returns a metadata dict with 'effective_fps', 'total_extracted', etc.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2


def extract_frames(
    video_path: str,
    output_dir: str,
    target_fps: float = 0.0,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> dict:
    """Extract frames from *video_path* into *output_dir* as ``000000.jpg``, etc.

    Parameters
    ----------
    video_path : str
        Path to input video.
    output_dir : str
        Directory to write JPEG frames into.
    target_fps : float
        If > 0, sample frames at this FPS.  If <= 0, extract every frame.
    start_frame : int
        First video frame index to consider (inclusive).
    end_frame : int | None
        Last video frame index to consider (exclusive).  ``None`` means end of video.

    Returns
    -------
    dict
        Metadata including ``effective_fps``, ``total_extracted``, ``source_fps``,
        ``start_frame``, ``end_frame``.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if end_frame is None:
        end_frame = total_frames

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    every_frame = target_fps <= 0
    if every_frame:
        effective_fps = source_fps
        frame_interval = 1
    else:
        frame_interval = max(1, round(source_fps / target_fps))
        effective_fps = source_fps / frame_interval

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    written = 0
    frame_idx = start_frame
    out_idx = 0

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if (frame_idx - start_frame) % frame_interval == 0:
            fname = out / f"{out_idx:06d}.jpg"
            cv2.imwrite(str(fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            written += 1
            out_idx += 1

        frame_idx += 1

    cap.release()

    metadata = {
        "source_video": str(video_path),
        "source_fps": source_fps,
        "effective_fps": effective_fps,
        "frame_interval": frame_interval,
        "total_extracted": written,
        "start_frame": start_frame,
        "end_frame": end_frame,
    }

    meta_path = out / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata
