#!/usr/bin/env python3
"""Convert timestamp-based chunks to frame-index chunks.

Input YAML schema (minimal):
  video: videos/.../video.webm        # optional if --video is provided
  fps: 30                            # optional if fps can be read from video
  chunks:
    - ["00:00:05", "00:00:10"]
    - start: "00:01:00"
      end: "00:01:15"
      note: "optional"

Timestamp formats supported:
  - "HH:MM:SS" (required by the user request)
  - also allows fractional seconds: "HH:MM:SS.sss"
  - numeric values are interpreted as seconds

Conversion:
  - start_frame = floor(start_time_seconds * fps)
  - end_frame   = ceil(end_time_seconds   * fps)
  - output chunks use the half-open interval: [start, end)

Usage:
  python scripts/tools/timestamps_to_frames.py --input chunks_time.yaml
  python scripts/tools/timestamps_to_frames.py --input chunks_time.yaml --output chunks.yaml
  python scripts/tools/timestamps_to_frames.py --input chunks_time.yaml --video videos/.../video.webm --output -
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import yaml


_TS_RE = re.compile(
    r"^\s*(?P<h>\d+):(?P<m>[0-5]?\d):(?P<s>(?:[0-5]?\d)(?:\.\d+)?)\s*$"
)


@dataclass(frozen=True)
class _ChunkTime:
    start_s: Fraction
    end_s: Fraction
    note: str


def _parse_timestamp_to_seconds(value: object) -> Fraction:
    if isinstance(value, (int, float)):
        # Treat numeric inputs as seconds.
        return Fraction(str(value))

    if not isinstance(value, str):
        raise ValueError(f"timestamp must be a string like HH:MM:SS, got {type(value).__name__}")

    m = _TS_RE.match(value)
    if not m:
        raise ValueError(f"invalid timestamp {value!r}; expected HH:MM:SS or HH:MM:SS.sss")

    h = int(m.group("h"))
    mm = int(m.group("m"))
    ss = Fraction(m.group("s"))
    if ss >= 60:
        raise ValueError(f"invalid seconds field in timestamp {value!r}; must be < 60")

    total = Fraction(h * 3600 + mm * 60) + ss
    if total < 0:
        raise ValueError(f"timestamp must be non-negative, got {value!r}")
    return total


def _parse_chunks_time(doc: dict) -> list[_ChunkTime]:
    raw = doc.get("chunks")
    if not isinstance(raw, list) or not raw:
        raise ValueError("input YAML must contain a non-empty top-level 'chunks' list")

    out: list[_ChunkTime] = []
    for i, entry in enumerate(raw):
        if isinstance(entry, dict):
            start = entry.get("start")
            end = entry.get("end")
            note = entry.get("note", "")
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            start, end = entry[0], entry[1]
            note = entry[2] if len(entry) >= 3 else ""
        else:
            raise ValueError(f"chunks[{i}] must be a dict or [start,end], got {type(entry).__name__}")

        if start is None or end is None:
            raise ValueError(f"chunks[{i}] missing start/end")

        start_s = _parse_timestamp_to_seconds(start)
        end_s = _parse_timestamp_to_seconds(end)
        note_str = "" if note is None else str(note)
        if end_s <= start_s:
            raise ValueError(
                f"chunks[{i}] end must be > start, got start={start_s} end={end_s} (seconds)"
            )
        out.append(_ChunkTime(start_s=start_s, end_s=end_s, note=note_str))
    return out


def _ffprobe_fps(video_path: Path) -> Fraction | None:
    import shutil
    import subprocess

    if shutil.which("ffprobe") is None:
        return None

    # Try avg_frame_rate then r_frame_rate; both are rationals like "30000/1001".
    for field in ("avg_frame_rate", "r_frame_rate"):
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    f"stream={field}",
                    "-of",
                    "default=nw=1:nk=1",
                    str(video_path),
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            s = (result.stdout or "").strip()
            if not s or s == "N/A":
                continue
            if "/" in s:
                num, den = s.split("/", 1)
                fps = Fraction(int(num), int(den))
            else:
                fps = Fraction(s)
            if fps > 0:
                return fps
        except Exception:
            continue
    return None


def _opencv_fps(video_path: Path) -> Fraction | None:
    try:
        import cv2
    except Exception:
        return None

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 0:
            return None
        return Fraction(str(float(fps)))
    finally:
        cap.release()


def _resolve_doc_video_path(doc: dict, doc_base: Path) -> Path | None:
    video = doc.get("video") or doc.get("video_path")
    if not video:
        return None
    video_path = Path(str(video))
    if not video_path.is_absolute():
        video_path = doc_base / video_path
    return video_path


def _resolve_fps(doc: dict, cli_video: str | None, cli_fps: float | None, doc_base: Path) -> tuple[Fraction, Path | None]:
    # Explicit FPS wins.
    if cli_fps is not None:
        fps = Fraction(str(float(cli_fps)))
        if fps <= 0:
            raise ValueError(f"--fps must be > 0, got {cli_fps}")
        return fps, None

    if "fps" in doc and doc["fps"] is not None:
        fps = Fraction(str(doc["fps"]))
        if fps <= 0:
            raise ValueError(f"fps in YAML must be > 0, got {doc['fps']!r}")
        return fps, None

    # Otherwise infer from video.
    if cli_video:
        video_path = Path(str(cli_video))
    else:
        video_path = _resolve_doc_video_path(doc, doc_base=doc_base)
    if video_path is None:
        raise ValueError("provide fps via --fps or YAML 'fps', or provide a video via --video or YAML 'video'")

    if not video_path.exists():
        raise ValueError(f"video not found: {video_path}")

    fps = _ffprobe_fps(video_path) or _opencv_fps(video_path)
    if fps is None or fps <= 0:
        raise ValueError(f"could not determine fps for video: {video_path}")
    return fps, video_path


def _floor_fraction(x: Fraction) -> int:
    return x.numerator // x.denominator


def _ceil_fraction(x: Fraction) -> int:
    return (x.numerator + x.denominator - 1) // x.denominator


def _convert(chunks: list[_ChunkTime], fps: Fraction) -> list[dict]:
    out: list[dict] = []
    for i, c in enumerate(chunks):
        start_f = _floor_fraction(c.start_s * fps)
        end_f = _ceil_fraction(c.end_s * fps)
        if end_f <= start_f:
            raise ValueError(
                f"chunk[{i}] became empty after conversion: start_frame={start_f} end_frame={end_f}"
            )
        out.append({"start": int(start_f), "end": int(end_f), "note": c.note})
    return out


def _dump_yaml(doc: dict) -> str:
    return yaml.dump(doc, default_flow_style=False, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert YAML chunk timestamps to frame indices.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input YAML with timestamp chunks.")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help=(
            "Output YAML path. Default: write 'chunks.yaml' next to the video file "
            "(if known) or next to the input YAML. Use '-' for stdout."
        ),
    )
    parser.add_argument("--video", type=str, default=None, help="Video path to infer fps if not provided.")
    parser.add_argument("--fps", type=float, default=None, help="FPS override (skips video probing).")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    with open(in_path, "r") as f:
        loaded = yaml.safe_load(f)
    if loaded is None:
        doc: dict = {}
    elif isinstance(loaded, dict):
        doc = loaded
    elif isinstance(loaded, list):
        # Allow bare YAML lists as a convenience:
        #   - ["00:00:05", "00:00:10"]
        #   - ["00:01:00", "00:01:15"]
        doc = {"chunks": loaded}
    else:
        raise SystemExit("Input YAML must be a mapping (with 'chunks') or a bare list of [start,end] pairs.")

    chunks_time = _parse_chunks_time(doc)
    fps, video_path = _resolve_fps(doc, cli_video=args.video, cli_fps=args.fps, doc_base=in_path.parent)

    chunks_frames = _convert(chunks_time, fps=fps)
    out_doc = {
        "version": 1,
        "frame_indexing": {"base": 0, "interval": "[start,end)"},
        "source": {
            "video": str(video_path) if video_path is not None else str(_resolve_doc_video_path(doc, doc_base=in_path.parent) or ""),
            "fps": float(fps),
            "conversion": {"start": "floor(t*fps)", "end": "ceil(t*fps)"},
        },
        "chunks": chunks_frames,
    }

    rendered = _dump_yaml(out_doc)

    if args.output == "-":
        sys.stdout.write(rendered)
        return
    if args.output is None or args.output == "":
        base_dir = (video_path.parent if video_path is not None else in_path.parent)
        out_path = base_dir / "chunks.yaml"
    else:
        out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(rendered)
    print(f"Wrote {len(chunks_frames)} chunk(s) to {out_path}")


if __name__ == "__main__":
    main()
