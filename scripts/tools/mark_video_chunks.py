#!/usr/bin/env python3
"""Interactive chunk-marking tool for selecting frame ranges from videos.

Usage:
    python scripts/tools/mark_video_chunks.py                       # scan videos/ and pick
    python scripts/tools/mark_video_chunks.py --video videos/kerni_rc/video1/video1.webm
    python scripts/tools/mark_video_chunks.py --video ... --chunks_file out.yaml

Keyboard controls (shown on-screen):
    a / d         step -1 / +1 frame
    j / l         step -10 / +10 frames
    [ / ]         step -100 / +100 frames
    space         toggle play/pause
    s             set start = current frame
    e             set end = current frame
    n             add chunk [start, end) to list
    u             undo last chunk
    w             write chunks file
    q             quit (prompts if unsaved)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import yaml


VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".mkv", ".avi"}


# ──────────────────────────── chunks I/O ──────────────────────────────────


def load_chunks(path: Path) -> list[dict]:
    """Load chunks from a YAML file.  Returns list of dicts with start/end/note."""
    if not path.exists():
        return []
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if data is None:
        return []
    raw = data.get("chunks", [])
    chunks: list[dict] = []
    for entry in raw:
        if isinstance(entry, dict):
            chunks.append(
                {
                    "start": int(entry["start"]),
                    "end": int(entry["end"]),
                    "note": entry.get("note", ""),
                }
            )
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            chunks.append({"start": int(entry[0]), "end": int(entry[1]), "note": ""})
    return chunks


def save_chunks(path: Path, chunks: list[dict]) -> None:
    """Write chunks to YAML (Option B format from spec)."""
    sorted_chunks = sorted(chunks, key=lambda c: c["start"])
    doc = {
        "version": 1,
        "frame_indexing": {"base": 0, "interval": "[start,end)"},
        "chunks": [
            {"start": c["start"], "end": c["end"], "note": c.get("note", "")}
            for c in sorted_chunks
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(doc, f, default_flow_style=False, sort_keys=False)
    print(f"Saved {len(sorted_chunks)} chunk(s) to {path}")


# ──────────────────────────── video discovery ─────────────────────────────


def find_videos(root: Path) -> list[Path]:
    """Recursively find video files under *root*."""
    videos: list[Path] = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(root.rglob(f"*{ext}"))
    return sorted(set(videos))


def choose_video(root: Path) -> Path:
    """Interactive menu to pick a video from *root*."""
    videos = find_videos(root)
    if not videos:
        print(f"No video files found under {root}")
        sys.exit(1)
    if len(videos) == 1:
        print(f"Found 1 video: {videos[0]}")
        return videos[0]
    print("Available videos:")
    for i, v in enumerate(videos):
        print(f"  [{i}] {v.relative_to(root)}")
    while True:
        try:
            idx = int(input("Select video index: "))
            return videos[idx]
        except (ValueError, IndexError):
            print("Invalid selection, try again.")


# ──────────────────────────── frame count via ffprobe ─────────────────────


def _ffprobe_frame_count(video_path: Path) -> int | None:
    """Try to get frame count from container metadata (fast, no decoding).

    Falls back to OpenCV's estimate rather than running the slow -count_frames
    scan which requires decoding every frame.
    """
    import shutil
    import subprocess

    if shutil.which("ffprobe") is None:
        return None
    try:
        # Fast path: read nb_frames from stream metadata (no decoding)
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=nb_frames,r_frame_rate,duration",
                "-of", "csv=p=0",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        parts = result.stdout.strip().split(",")
        # parts: nb_frames, r_frame_rate, duration  (some may be "N/A")
        nb_frames_str = parts[0].strip() if parts else ""
        if nb_frames_str and nb_frames_str != "N/A":
            return int(nb_frames_str)

        # Fallback: derive from duration × fps (still fast, no decoding)
        if len(parts) >= 3:
            fps_str, dur_str = parts[1].strip(), parts[2].strip()
            if "/" in fps_str and dur_str not in ("", "N/A"):
                num, den = fps_str.split("/")
                fps = float(num) / float(den)
                duration = float(dur_str)
                return int(fps * duration)
    except Exception:
        pass
    return None


# ──────────────────────────── HUD drawing ─────────────────────────────────


def draw_hud(
    frame: "cv2.Mat",
    frame_idx: int,
    total_frames: int,
    fps: float,
    start_mark: int | None,
    end_mark: int | None,
    chunks: list[dict],
    playing: bool,
    unsaved: bool,
):
    """Draw an informational overlay on *frame* (mutates in-place)."""
    h, w = frame.shape[:2]

    def put(text: str, x: int, y: int, scale: float = 0.6, color=(255, 255, 255)):
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    timestamp = frame_idx / fps if fps > 0 else 0.0

    # top-left: frame info
    put(f"Frame: {frame_idx} / {total_frames - 1}", 10, 25)
    put(f"Time:  {timestamp:.2f}s", 10, 50)
    mode = "PLAYING" if playing else "PAUSED"
    put(mode, 10, 75, color=(0, 255, 0) if playing else (0, 200, 255))

    # top-right: markers
    rx = w - 300
    s_txt = str(start_mark) if start_mark is not None else "—"
    e_txt = str(end_mark) if end_mark is not None else "—"
    put(f"Start: {s_txt}", rx, 25, color=(0, 255, 0))
    put(f"End:   {e_txt}", rx, 50, color=(0, 100, 255))
    put(f"Chunks: {len(chunks)}", rx, 75)
    if unsaved:
        put("* UNSAVED *", rx, 100, color=(0, 0, 255))

    # bottom: chunk list (show last 5)
    y0 = h - 15
    visible = chunks[-5:] if len(chunks) > 5 else chunks
    for i, c in enumerate(reversed(visible)):
        note_str = f'  "{c["note"]}"' if c.get("note") else ""
        put(
            f'  #{len(chunks) - i}: [{c["start"]}, {c["end"]}){note_str}',
            10,
            y0 - i * 22,
            scale=0.5,
        )

    # bottom-right: key hints
    hints = [
        "a/d: -1/+1  j/l: -10/+10  [/]: -100/+100",
        "space: play  s: start  e: end  n: add  u: undo",
        "w: save  q: quit",
    ]
    for i, line in enumerate(hints):
        put(line, w - 480, h - 15 - (len(hints) - 1 - i) * 22, scale=0.45, color=(200, 200, 200))


# ──────────────────────────── main viewer loop ────────────────────────────


def run_viewer(video_path: Path, chunks_file: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: cannot open video {video_path}")
        sys.exit(1)

    total_frames_cv = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Try ffprobe for accurate count
    total_frames = _ffprobe_frame_count(video_path) or total_frames_cv
    print(f"Video: {video_path}")
    print(f"Frames: {total_frames} (cv2 reports {total_frames_cv})  FPS: {fps:.2f}")

    # Load existing chunks
    chunks = load_chunks(chunks_file)
    if chunks:
        print(f"Loaded {len(chunks)} existing chunk(s) from {chunks_file}")

    window_name = f"Chunk Marker - {video_path.name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Trackbar for seeking
    def on_trackbar(pos: int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

    cv2.createTrackbar("Frame", window_name, 0, max(total_frames - 1, 1), on_trackbar)

    frame_idx = 0
    start_mark: int | None = None
    end_mark: int | None = None
    playing = False
    unsaved = False
    dirty_since_save = len(chunks) > 0  # if we loaded chunks, they're already saved

    dirty_since_save = False  # reset: loaded state matches disk

    def seek(target: int):
        nonlocal frame_idx
        target = max(0, min(target, total_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        frame_idx = target
        cv2.setTrackbarPos("Frame", window_name, frame_idx)

    while True:
        if playing:
            ret, frame = cap.read()
            if not ret:
                playing = False
                seek(frame_idx)
                continue
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            cv2.setTrackbarPos("Frame", window_name, frame_idx)
        else:
            # Read current frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                # Try backing up
                if frame_idx > 0:
                    frame_idx -= 1
                    continue
                print("Error: cannot read any frames.")
                break

        display = frame.copy()
        draw_hud(display, frame_idx, total_frames, fps, start_mark, end_mark, chunks, playing, dirty_since_save)
        cv2.imshow(window_name, display)

        wait_ms = int(1000 / fps) if playing else 0
        key = cv2.waitKey(max(wait_ms, 1)) & 0xFF

        if key == ord("q"):
            if dirty_since_save:
                print("You have unsaved chunks. Press 'w' to save or 'q' again to quit without saving.")
                k2 = cv2.waitKey(0) & 0xFF
                if k2 == ord("w"):
                    save_chunks(chunks_file, chunks)
                    dirty_since_save = False
                elif k2 != ord("q"):
                    continue
            break

        elif key == ord(" "):
            playing = not playing
            if playing:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        # ─── stepping ───
        elif key == ord("d"):
            playing = False
            seek(frame_idx + 1)
        elif key == ord("a"):
            playing = False
            seek(frame_idx - 1)
        elif key == ord("l"):
            playing = False
            seek(frame_idx + 10)
        elif key == ord("j"):
            playing = False
            seek(frame_idx - 10)
        elif key == ord("]"):
            playing = False
            seek(frame_idx + 100)
        elif key == ord("["):
            playing = False
            seek(frame_idx - 100)

        # ─── markers ───
        elif key == ord("s"):
            start_mark = frame_idx
            print(f"Start set to {start_mark}")
        elif key == ord("e"):
            end_mark = frame_idx
            print(f"End set to {end_mark}")

        # ─── chunk management ───
        elif key == ord("n"):
            if start_mark is None or end_mark is None:
                print("Set both start (s) and end (e) before adding a chunk.")
            elif start_mark >= end_mark:
                print(f"Invalid chunk: start ({start_mark}) must be < end ({end_mark})")
            else:
                chunk = {"start": start_mark, "end": end_mark, "note": ""}
                chunks.append(chunk)
                dirty_since_save = True
                print(f"Added chunk [{start_mark}, {end_mark})  (total: {len(chunks)})")
                start_mark = None
                end_mark = None

        elif key == ord("u"):
            if chunks:
                removed = chunks.pop()
                dirty_since_save = True
                print(f'Undone chunk [{removed["start"]}, {removed["end"]})')
            else:
                print("No chunks to undo.")

        elif key == ord("w"):
            save_chunks(chunks_file, chunks)
            dirty_since_save = False

        # ─── trackbar sync (user dragged it) ───
        else:
            tb_pos = cv2.getTrackbarPos("Frame", window_name)
            if tb_pos != frame_idx and not playing:
                seek(tb_pos)

    cap.release()
    cv2.destroyAllWindows()


# ──────────────────────────── CLI ─────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Interactive tool to mark frame-range chunks in a video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to video file.  If omitted, scans --videos_root for candidates.",
    )
    parser.add_argument(
        "--videos_root",
        type=str,
        default="videos",
        help="Root directory to scan for videos (default: videos/).",
    )
    parser.add_argument(
        "--chunks_file",
        type=str,
        default=None,
        help="Path for chunks YAML.  Default: <video_dir>/chunks.yaml.",
    )
    args = parser.parse_args()

    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"Video not found: {video_path}")
            sys.exit(1)
    else:
        video_path = choose_video(Path(args.videos_root))

    if args.chunks_file:
        chunks_file = Path(args.chunks_file)
    else:
        chunks_file = video_path.parent / "chunks.yaml"

    print(f"Chunks file: {chunks_file}")
    run_viewer(video_path, chunks_file)


if __name__ == "__main__":
    main()
