"""
End-to-end data generation pipeline:
  extract frames → track breadcrumbs → (optional) SAM mask + pos/neg sampling → assemble dataset.

Usage:
    python src/data/generate_dataset.py \
        --video videos/output_2.mp4 \
        --output data/output_2 \
        --fps 5 \
        --breadcrumb_max_track_len 20
"""

import argparse
import json
from pathlib import Path

from extract_frames import extract_frames
from track_breadcrumbs import get_sorted_frame_paths, save_breadcrumbs, track_breadcrumbs_chained


def generate_dataset(
    video_path: str,
    output_dir: str,
    fps: float = 5.0,
    center_offset_frac: float = 0.1,
    boundary_thresh: int = 30,
    breadcrumb_max_track_len: int = 20,
    skip_extract: bool = False,
    skip_sam: bool = False,
    resume: bool = False,
    checkpoint: str = "checkpoints/sam_vit_b_01ec64.pth",
    sam_type: str = "vit_b",
    prompt_max_points: int = 32,
    num_pos: int = 50,
    num_neg: int = 50,
    neg_top_frac: float = 0.3,
    sample_margin_px: int = 10,
    seed: int = 0,
    frame_start: int | None = None,
    frame_end: int | None = None,
    max_frames: int | None = None,
):
    """
    Run the full data generation pipeline.

    1. Extract frames from video at target FPS
    2. Track breadcrumb points in reverse via chained LK optical flow
    3. (Optional) Generate SAM masks + pos/neg points
    4. Assemble dataset index

    Args:
        video_path: Path to input video
        output_dir: Root output directory for this video's dataset
        fps: Target FPS for frame extraction
        center_offset_frac: Seed point y offset below center (fraction of height)
        boundary_thresh: Drop tracked points within this many px of the image edge
        breadcrumb_max_track_len: Max number of frames each breadcrumb can persist
                                  (<= 0 means unlimited)
        skip_extract: Skip frame extraction if frames already exist
        skip_sam: Skip SAM mask + point generation step
        resume: Skip SAM frames with existing outputs (masks + points)
        checkpoint: SAM checkpoint path
        sam_type: SAM model type (vit_b, vit_l, vit_h)
        prompt_max_points: Max breadcrumb points used as SAM prompts per frame
        num_pos, num_neg: Pos/neg points sampled per frame (best effort)
        neg_top_frac: Exclude top fraction of frame for negative sampling
        sample_margin_px: Margin in px via mask erosion/dilation for sampling
        seed: Base RNG seed for deterministic sampling
    """
    output_dir = Path(output_dir)
    frames_dir = output_dir / "frames"
    breadcrumbs_dir = output_dir / "breadcrumbs"

    # Step 1: Extract frames
    print("=" * 60)
    print("STEP 1: Extracting frames")
    print("=" * 60)
    if skip_extract:
        existing = sorted(frames_dir.glob("*.jpg"))
        if len(existing) == 0:
            raise RuntimeError(
                f"--skip_extract was set but no frames found in {frames_dir}"
            )
        print(f"Skipping extraction; using existing frames in {frames_dir} ({len(existing)} files)")
    else:
        extract_frames(video_path, str(frames_dir), fps)

    # Step 2: Track breadcrumbs
    print("\n" + "=" * 60)
    print("STEP 2: Tracking breadcrumb points (chained LK, reverse)")
    print("=" * 60)
    frame_paths = get_sorted_frame_paths(str(frames_dir))
    if frame_start is not None or frame_end is not None or max_frames is not None:
        frame_paths_filtered = []
        for p in frame_paths:
            try:
                idx = int(p.stem)
            except ValueError:
                idx = None
            if idx is None:
                continue
            if frame_start is not None and idx < frame_start:
                continue
            if frame_end is not None and idx >= frame_end:
                continue
            frame_paths_filtered.append(p)
        if max_frames is not None:
            frame_paths_filtered = frame_paths_filtered[: int(max_frames)]
        frame_paths = frame_paths_filtered

    breadcrumbs = track_breadcrumbs_chained(
        frame_paths,
        center_offset_frac=center_offset_frac,
        boundary_thresh=boundary_thresh,
        max_track_len=breadcrumb_max_track_len,
    )
    save_breadcrumbs(breadcrumbs, str(breadcrumbs_dir))

    # Step 3: SAM mask + pos/neg sampling (optional)
    if not skip_sam:
        print("\n" + "=" * 60)
        print("STEP 3: SAM masks + pos/neg point sampling")
        print("=" * 60)
        from sam_mask_generator import run_sam_pipeline

        run_sam_pipeline(
            frames_dir=str(frames_dir),
            breadcrumbs_dir=str(breadcrumbs_dir),
            output_dir=str(output_dir),
            checkpoint=checkpoint,
            sam_type=sam_type,
            prompt_max_points=prompt_max_points,
            num_pos=num_pos,
            num_neg=num_neg,
            neg_top_frac=neg_top_frac,
            sample_margin_px=sample_margin_px,
            seed=seed,
            resume=resume,
            frame_start=frame_start,
            frame_end=frame_end,
            max_frames=max_frames,
        )

    # Step 4: Create dataset index
    print("\n" + "=" * 60)
    print("STEP 4: Assembling dataset index")
    print("=" * 60)
    index_entries = []
    for fwd_idx in sorted(breadcrumbs.keys()):
        frame_file = f"{fwd_idx:06d}.jpg"
        bc_file = f"{fwd_idx:06d}_breadcrumbs.npz"
        n_points = len(breadcrumbs[fwd_idx])
        index_entries.append({
            "frame_idx": fwd_idx,
            "frame_path": f"frames/{frame_file}",
            "breadcrumbs_path": f"breadcrumbs/{bc_file}",
            "mask_path": f"masks/{fwd_idx:06d}.png",
            "pos_neg_points_path": f"pos_neg_points/{fwd_idx:06d}.npz",
            "num_points": n_points,
        })

    index_path = output_dir / "dataset_index.json"
    with open(index_path, "w") as f:
        json.dump(index_entries, f, indent=2)

    # Summary stats
    total_points = sum(e["num_points"] for e in index_entries)
    frames_with_points = sum(1 for e in index_entries if e["num_points"] > 0)

    print(f"\nDataset index saved to {index_path}")
    print(f"  Total frames: {len(index_entries)}")
    print(f"  Frames with breadcrumbs: {frames_with_points}")
    print(f"  Total breadcrumb points: {total_points}")
    print(f"  Avg points/frame: {total_points / max(len(index_entries), 1):.1f}")
    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(description="End-to-end data generation pipeline")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--output", type=str, required=True, help="Output directory for dataset")
    parser.add_argument("--fps", type=float, default=5.0, help="Target FPS for extraction (default: 5)")
    parser.add_argument("--every_frame", action="store_true",
                        help="Extract every frame (equivalent to --fps 0)")
    parser.add_argument("--center_offset_frac", type=float, default=0.1,
                        help="Seed point y offset below center (default: 0.1)")
    parser.add_argument("--boundary_thresh", type=int, default=30,
                        help="Drop points within this many px of edge (default: 30)")
    parser.add_argument("--breadcrumb_max_track_len", type=int, default=20,
                        help="Max frames each breadcrumb is kept (<=0 for unlimited, default: 20)")
    parser.add_argument("--skip_extract", action="store_true",
                        help="Skip frame extraction if frames already exist")

    parser.add_argument("--skip_sam", action="store_true",
                        help="Skip SAM mask + pos/neg point generation")
    parser.add_argument("--resume", action="store_true",
                        help="Skip SAM frames whose outputs already exist")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/sam_vit_b_01ec64.pth",
                        help="SAM checkpoint path")
    parser.add_argument("--sam_type", type=str, default="vit_b",
                        choices=["vit_b", "vit_l", "vit_h"],
                        help="SAM model type (default: vit_b)")
    parser.add_argument("--prompt_max_points", type=int, default=32,
                        help="Max breadcrumb points used as SAM prompts per frame (default: 32)")
    parser.add_argument("--num_pos", type=int, default=50,
                        help="Positive points sampled per frame (default: 50)")
    parser.add_argument("--num_neg", type=int, default=50,
                        help="Negative points sampled per frame (default: 50)")
    parser.add_argument("--neg_top_frac", type=float, default=0.3,
                        help="Exclude top fraction for negative sampling (default: 0.3)")
    parser.add_argument("--sample_margin_px", type=int, default=10,
                        help="Margin in px via erosion/dilation for sampling (default: 10)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base RNG seed for deterministic sampling (default: 0)")
    parser.add_argument("--frame_start", type=int, default=None,
                        help="Start frame index (inclusive) by filename integer (e.g. 0)")
    parser.add_argument("--frame_end", type=int, default=None,
                        help="End frame index (exclusive) by filename integer (e.g. 50)")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Max number of frames to process after range filtering")
    args = parser.parse_args()

    generate_dataset(
        args.video,
        args.output,
        fps=(0.0 if args.every_frame else args.fps),
        center_offset_frac=args.center_offset_frac,
        boundary_thresh=args.boundary_thresh,
        breadcrumb_max_track_len=args.breadcrumb_max_track_len,
        skip_extract=args.skip_extract,
        skip_sam=args.skip_sam,
        resume=args.resume,
        checkpoint=args.checkpoint,
        sam_type=args.sam_type,
        prompt_max_points=args.prompt_max_points,
        num_pos=args.num_pos,
        num_neg=args.num_neg,
        neg_top_frac=args.neg_top_frac,
        sample_margin_px=args.sample_margin_px,
        seed=args.seed,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
