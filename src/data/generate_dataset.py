"""Generate V-STRONG dataset from a video (single or chunked config mode).

Modes:
  - Single video:  ``python src/data/generate_dataset.py --video <path> --output <dir>``
  - Config mode:   ``python src/data/generate_dataset.py --config <yaml>``

Pipeline per video (or per chunk):
  1. Extract frames (at target FPS or every frame).
  2. Track breadcrumbs via chained LK optical flow (reverse).
  3. Use breadcrumbs as foreground prompts to SAM to create masks.
  4. Sample pos/neg points from masks.
  5. Write frames/, breadcrumbs/, masks/, pos_neg_points/, metadata.json.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

# Ensure src/ is on path so ``data.*`` and ``models.*`` resolve
_src_dir = str(Path(__file__).resolve().parents[1])
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from data.extract_frames import extract_frames
from data.sam_mask_generator import run_sam_pipeline
from data.track_breadcrumbs import get_sorted_frame_paths, save_breadcrumbs, track_breadcrumbs_chained


def generate_single_video(
    video_path: str,
    output_dir: str,
    fps: float = 5.0,
    every_frame: bool = False,
    start_frame: int = 0,
    end_frame: int | None = None,
    sam_checkpoint: str = "checkpoints/sam_vit_b_01ec64.pth",
    sam_type: str = "vit_b",
    prompt_max_points: int = 32,
    num_pos: int = 50,
    num_neg: int = 50,
    neg_top_frac: float = 0.3,
    sample_margin_px: int = 10,
    seed: int = 0,
    resume: bool = False,
    write_viz: bool = False,
    viz_every_n: int = 1,
    center_offset_frac: float = 0.1,
    boundary_thresh: int = 30,
    breadcrumb_max_track_len: int = 20,
    tracker_type: str = "lk",
    cotracker_checkpoint: str = "./external/co-tracker/checkpoints/scaled_online.pth",
    cotracker_window_len: int = 16,
    num_workers: int = 4,
) -> dict:
    """Generate dataset for a single video (or chunk).

    Returns metadata dict.
    """
    out = Path(output_dir)
    frames_dir = out / "frames"
    breadcrumbs_dir = out / "breadcrumbs"
    frames_dir.mkdir(parents=True, exist_ok=True)
    breadcrumbs_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract frames
    target_fps = 0.0 if every_frame else fps
    print(f"Extracting frames (fps={target_fps}, start={start_frame}, end={end_frame})")
    meta = extract_frames(
        video_path,
        str(frames_dir),
        target_fps=target_fps,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    print(f"  Extracted {meta['total_extracted']} frames (effective fps={meta['effective_fps']:.2f})")

    if meta["total_extracted"] <= 0:
        raise RuntimeError("No frames were extracted; cannot generate dataset.")

    # Step 2: Track breadcrumbs
    frame_paths = get_sorted_frame_paths(frames_dir)
    if tracker_type == "cotracker":
        print("Tracking breadcrumbs (CoTracker, reverse)")
        from data.track_breadcrumbs_cotracker import track_breadcrumbs_cotracker
        breadcrumbs = track_breadcrumbs_cotracker(
            frame_paths,
            center_offset_frac=center_offset_frac,
            boundary_thresh=boundary_thresh,
            max_track_len=breadcrumb_max_track_len,
            checkpoint=cotracker_checkpoint,
            window_len=cotracker_window_len,
        )
    else:
        print("Tracking breadcrumbs (chained LK, reverse)")
        breadcrumbs = track_breadcrumbs_chained(
            frame_paths,
            center_offset_frac=center_offset_frac,
            boundary_thresh=boundary_thresh,
            max_track_len=breadcrumb_max_track_len,
        )
    save_breadcrumbs(breadcrumbs, breadcrumbs_dir)

    # Step 3: Breadcrumb-prompted SAM + point sampling
    print("Generating SAM masks + pos/neg points (breadcrumb-prompted)")
    run_sam_pipeline(
        frames_dir=str(frames_dir),
        breadcrumbs_dir=str(breadcrumbs_dir),
        output_dir=str(out),
        checkpoint=sam_checkpoint,
        sam_type=sam_type,
        prompt_max_points=prompt_max_points,
        num_pos=num_pos,
        num_neg=num_neg,
        neg_top_frac=neg_top_frac,
        sample_margin_px=sample_margin_px,
        seed=seed,
        resume=resume,
        write_viz=write_viz,
        viz_every_n=viz_every_n,
        num_workers=num_workers,
    )

    # Write dataset metadata
    dataset_meta = {
        "source_video": str(video_path),
        "effective_fps": meta["effective_fps"],
        "source_fps": meta["source_fps"],
        "total_frames": meta["total_extracted"],
        "start_frame": start_frame,
        "end_frame": end_frame,
        "sam_type": sam_type,
        "sam_checkpoint": sam_checkpoint,
        "prompt_max_points": int(prompt_max_points),
        "num_pos": num_pos,
        "num_neg": num_neg,
        "neg_top_frac": float(neg_top_frac),
        "sample_margin_px": int(sample_margin_px),
        "breadcrumbs": {
            "tracker_type": tracker_type,
            "center_offset_frac": float(center_offset_frac),
            "boundary_thresh": int(boundary_thresh),
            "max_track_len": int(breadcrumb_max_track_len),
            "total_frames_with_breadcrumbs": int(len(breadcrumbs)),
        },
    }
    with open(out / "metadata.json", "w") as f:
        json.dump(dataset_meta, f, indent=2)

    print(f"Dataset written to {out} ({meta['total_extracted']} frames)")
    return dataset_meta


def generate_from_config(
    config_path: str,
    *,
    regenerate: bool = False,
    tracker_type_override: str | None = None,
    cotracker_checkpoint_override: str | None = None,
    cotracker_window_len_override: int | None = None,
) -> None:
    """Generate datasets from a YAML config file (chunk-aware).

    Config schema documented in .agent/task.md.
    """
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    output_root = Path(cfg["output_dataset_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    chunks_root = output_root / "chunks"
    chunks_root.mkdir(parents=True, exist_ok=True)

    extraction = cfg.get("extraction", {})
    every_frame = extraction.get("every_frame", False)
    fps = extraction.get("fps", 5)

    datagen = cfg.get("datagen", {})
    sam_ckpt = datagen.get("sam_ckpt", "checkpoints/sam_vit_b_01ec64.pth")
    sam_type = datagen.get("sam_type", "vit_b")
    prompt_max_points = datagen.get("prompt_max_points", 32)
    num_pos = datagen.get("num_pos", 50)
    num_neg = datagen.get("num_neg", 50)
    neg_top_frac = datagen.get("neg_top_frac", 0.3)
    sample_margin_px = datagen.get("sample_margin_px", 10)
    breadcrumb_max_track_len = datagen.get("breadcrumb_max_track_len", 20)
    center_offset_frac = datagen.get("center_offset_frac", 0.1)
    boundary_thresh = datagen.get("boundary_thresh", 30)
    seed = datagen.get("seed", 0)
    resume = datagen.get("resume", False)
    write_viz = datagen.get("write_viz", False)
    viz_every_n = datagen.get("viz_every_n", 1)
    tracker_type = tracker_type_override or datagen.get("tracker_type", "lk")
    cotracker_checkpoint = cotracker_checkpoint_override or datagen.get(
        "cotracker_checkpoint",
        "./external/co-tracker/checkpoints/scaled_online.pth",
    )
    cotracker_window_len = cotracker_window_len_override or datagen.get("cotracker_window_len", 16)
    num_workers = datagen.get("num_workers", 4)

    index_lines = []

    for video_entry in cfg.get("videos", []):
        video_id = video_entry["id"]
        video_dir = Path(video_entry.get("video_dir", ""))
        video_file = video_entry.get("video_file", "")
        video_path = video_dir / video_file if video_file else None

        # Allow direct video_path
        if video_path is None or not video_path.exists():
            vp = video_entry.get("video_path")
            if vp:
                video_path = Path(vp)

        if video_path is None or not video_path.exists():
            print(f"WARNING: Video not found for {video_id}, skipping")
            continue

        # Load chunks
        chunks_time_file = video_entry.get("chunks_time_file") or video_entry.get("chunks_time_yaml")
        if not chunks_time_file:
            default_time = video_dir / "chunks_time.yaml"
            if default_time.exists():
                chunks_time_file = default_time

        if chunks_time_file:
            chunks = _load_chunks_time(Path(str(chunks_time_file)), video_path=video_path)
        else:
            chunks_file = video_entry.get("chunks_file")
            if chunks_file:
                chunks_file = Path(chunks_file)
            else:
                chunks_file = video_dir / "chunks.yaml"

            chunks = _load_chunks(chunks_file)
        if not chunks:
            warn_src = chunks_time_file or str(video_entry.get("chunks_file") or (video_dir / "chunks.yaml"))
            print(f"WARNING: No chunks found for {video_id} at {warn_src}")
            # Process entire video as one chunk
            chunks = [{"start": 0, "end": None, "note": "full_video"}]

        for chunk in chunks:
            start = chunk["start"]
            end = chunk.get("end")
            end_str = f"{end:06d}" if end is not None else "end"
            chunk_id = f"{video_id}__{start:06d}_{end_str}"
            chunk_out = chunks_root / chunk_id

            if chunk_out.exists():
                if regenerate:
                    import shutil

                    print(f"Chunk {chunk_id} already exists; regenerating")
                    shutil.rmtree(chunk_out)
                else:
                    print(f"Chunk {chunk_id} already exists, skipping (use --regenerate to redo)")
                    _add_to_index(index_lines, chunk_id, chunk_out)
                    continue

            print(f"\n=== Processing chunk: {chunk_id} ===")
            generate_single_video(
                video_path=str(video_path),
                output_dir=str(chunk_out),
                fps=fps,
                every_frame=every_frame,
                start_frame=start,
                end_frame=end,
                sam_checkpoint=sam_ckpt,
                sam_type=sam_type,
                prompt_max_points=prompt_max_points,
                num_pos=num_pos,
                num_neg=num_neg,
                neg_top_frac=neg_top_frac,
                sample_margin_px=sample_margin_px,
                seed=seed,
                resume=resume,
                write_viz=write_viz,
                viz_every_n=viz_every_n,
                center_offset_frac=center_offset_frac,
                boundary_thresh=boundary_thresh,
                breadcrumb_max_track_len=breadcrumb_max_track_len,
                tracker_type=tracker_type,
                cotracker_checkpoint=cotracker_checkpoint,
                cotracker_window_len=cotracker_window_len,
                num_workers=num_workers,
            )
            _add_to_index(index_lines, chunk_id, chunk_out)

    # Write index.jsonl
    index_path = output_root / "index.jsonl"
    with open(index_path, "w") as f:
        for line in index_lines:
            f.write(json.dumps(line) + "\n")
    print(f"\nWrote index with {len(index_lines)} entries to {index_path}")

    # Write root metadata
    root_meta = {
        "name": cfg.get("name", ""),
        "config_file": str(config_path),
        "total_samples": len(index_lines),
    }
    with open(output_root / "metadata.json", "w") as f:
        json.dump(root_meta, f, indent=2)


def _load_chunks(chunks_file: Path) -> list[dict]:
    """Load chunk definitions from a YAML file."""
    if not chunks_file.exists():
        return []
    import yaml

    with open(chunks_file) as f:
        data = yaml.safe_load(f)
    if data is None:
        return []
    raw = data.get("chunks", [])
    chunks = []
    for entry in raw:
        if isinstance(entry, dict):
            chunks.append({
                "start": int(entry["start"]),
                "end": int(entry["end"]),
                "note": entry.get("note", ""),
            })
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            chunks.append({"start": int(entry[0]), "end": int(entry[1]), "note": ""})
    return chunks


_TS_RE = re.compile(r"^\s*(?P<h>\d+):(?P<m>[0-5]?\d):(?P<s>(?:[0-5]?\d)(?:\.\d+)?)\s*$")


def _parse_timestamp_to_seconds(value: object) -> Fraction:
    if isinstance(value, (int, float)):
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


def _ffprobe_fps(video_path: Path) -> Fraction | None:
    import shutil
    import subprocess

    if shutil.which("ffprobe") is None:
        return None
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


def _floor_fraction(x: Fraction) -> int:
    return x.numerator // x.denominator


def _ceil_fraction(x: Fraction) -> int:
    return (x.numerator + x.denominator - 1) // x.denominator


def _load_chunks_time(chunks_time_file: Path, *, video_path: Path) -> list[dict]:
    """Load timestamp-based chunks from YAML and convert them to frame ranges."""
    if not chunks_time_file.exists():
        return []
    import yaml

    with open(chunks_time_file) as f:
        loaded = yaml.safe_load(f)
    if loaded is None:
        doc: dict = {}
    elif isinstance(loaded, dict):
        doc = loaded
    elif isinstance(loaded, list):
        doc = {"chunks": loaded}
    else:
        raise ValueError("chunks_time YAML must be a mapping (with 'chunks') or a bare list of [start,end] pairs.")

    raw = doc.get("chunks")
    if not isinstance(raw, list) or not raw:
        return []

    # Determine fps
    if "fps" in doc and doc["fps"] is not None:
        fps = Fraction(str(doc["fps"]))
    else:
        if not video_path.exists():
            raise ValueError(f"video not found (needed to infer fps): {video_path}")
        fps = _ffprobe_fps(video_path) or _opencv_fps(video_path)
        if fps is None or fps <= 0:
            raise ValueError(f"could not determine fps for video: {video_path}")

    out: list[dict] = []
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
        if end_s <= start_s:
            raise ValueError(f"chunks[{i}] end must be > start, got start={start_s} end={end_s}")

        start_f = _floor_fraction(start_s * fps)
        end_f = _ceil_fraction(end_s * fps)
        if end_f <= start_f:
            raise ValueError(f"chunks[{i}] became empty after conversion: start_frame={start_f} end_frame={end_f}")
        out.append({"start": int(start_f), "end": int(end_f), "note": "" if note is None else str(note)})
    return out


def _add_to_index(index_lines: list[dict], chunk_id: str, chunk_dir: Path) -> None:
    """Add all frames in a chunk directory to the index."""
    frames_dir = chunk_dir / "frames"
    if not frames_dir.exists():
        return
    for fp in sorted(frames_dir.glob("*.jpg")):
        try:
            idx = int(fp.stem)
        except ValueError:
            continue
        rel_chunk = chunk_dir.name
        bc_rel = f"chunks/{rel_chunk}/breadcrumbs/{idx:06d}_breadcrumbs.npz"
        bc_abs = chunk_dir / "breadcrumbs" / f"{idx:06d}_breadcrumbs.npz"
        index_lines.append({
            "chunk_id": chunk_id,
            "frame_idx": idx,
            "frame_path": f"chunks/{rel_chunk}/frames/{fp.name}",
            "mask_path": f"chunks/{rel_chunk}/masks/{idx:06d}.png",
            "points_path": f"chunks/{rel_chunk}/pos_neg_points/{idx:06d}.npz",
            **({"breadcrumbs_path": bc_rel} if bc_abs.exists() else {}),
        })


def main():
    parser = argparse.ArgumentParser(description="Generate V-STRONG dataset from video(s).")

    # Single-video mode
    parser.add_argument("--video", type=str, default=None, help="Input video path.")
    parser.add_argument("--output", type=str, default=None, help="Output dataset directory.")

    # Config mode
    parser.add_argument("--config", type=str, default=None, help="YAML config for multi-video/chunk generation.")

    # Extraction
    parser.add_argument("--fps", type=float, default=5.0, help="Target FPS for frame extraction.")
    parser.add_argument("--every_frame", action="store_true", help="Extract every frame (ignore --fps).")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=None)

    # SAM / masking
    parser.add_argument("--checkpoint", type=str, default="checkpoints/sam_vit_b_01ec64.pth")
    parser.add_argument("--sam_type", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument(
        "--prompt_max_points",
        type=int,
        default=32,
        help="Max breadcrumb prompt points per frame for SAM (default: 32).",
    )
    parser.add_argument("--resume", action="store_true", help="Skip frames whose outputs already exist.")
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed for deterministic sampling.")
    parser.add_argument("--write_viz", action="store_true", help="Write visualization frames under sam_viz/.")
    parser.add_argument(
        "--viz_every_n",
        type=int,
        default=1,
        help="If --write_viz is set, write SAM visualizations only every Nth frame (default: 1 = all).",
    )

    # Point sampling
    parser.add_argument("--num_pos", type=int, default=50)
    parser.add_argument("--num_neg", type=int, default=50)
    parser.add_argument("--neg_top_frac", type=float, default=0.3)
    parser.add_argument("--sample_margin_px", type=int, default=10)

    # Breadcrumbs (LK optical flow)
    parser.add_argument("--breadcrumb_max_track_len", type=int, default=20)
    parser.add_argument("--center_offset_frac", type=float, default=0.1)
    parser.add_argument("--boundary_thresh", type=int, default=30)

    # Parallelism
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="Thread pool workers for post-processing (mask/points writes + viz). 0 = sequential.",
    )

    # Tracker selection
    parser.add_argument(
        "--tracker_type", type=str, default="lk",
        choices=["lk", "cotracker"],
        help="Breadcrumb tracking method: 'lk' (Lucas-Kanade) or 'cotracker'.",
    )
    parser.add_argument(
        "--cotracker_checkpoint", type=str,
        default="./external/co-tracker/checkpoints/scaled_online.pth",
        help="Path to CoTracker checkpoint (only used when --tracker_type=cotracker).",
    )
    parser.add_argument(
        "--cotracker_window_len", type=int, default=16,
        help="CoTracker sliding window length (only used when --tracker_type=cotracker).",
    )

    # Config mode options
    parser.add_argument("--regenerate", action="store_true", help="Regenerate existing chunks.")

    args = parser.parse_args()

    if args.config:
        generate_from_config(
            args.config,
            regenerate=args.regenerate,
            tracker_type_override=args.tracker_type,
            cotracker_checkpoint_override=args.cotracker_checkpoint,
            cotracker_window_len_override=args.cotracker_window_len,
        )
    elif args.video and args.output:
        generate_single_video(
            video_path=args.video,
            output_dir=args.output,
            fps=args.fps,
            every_frame=args.every_frame,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            sam_checkpoint=args.checkpoint,
            sam_type=args.sam_type,
            prompt_max_points=args.prompt_max_points,
            num_pos=args.num_pos,
            num_neg=args.num_neg,
            neg_top_frac=args.neg_top_frac,
            sample_margin_px=args.sample_margin_px,
            seed=args.seed,
            resume=args.resume,
            write_viz=args.write_viz,
            viz_every_n=args.viz_every_n,
            center_offset_frac=args.center_offset_frac,
            boundary_thresh=args.boundary_thresh,
            breadcrumb_max_track_len=args.breadcrumb_max_track_len,
            tracker_type=args.tracker_type,
            cotracker_checkpoint=args.cotracker_checkpoint,
            cotracker_window_len=args.cotracker_window_len,
            num_workers=args.num_workers,
        )
    else:
        parser.error("Provide --video + --output for single-video mode, or --config for config mode.")


if __name__ == "__main__":
    main()
