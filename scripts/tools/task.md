# Tools + Minimal Workflow (Chunks → Dataset)

Use this repo in a config-only way:

1) Put `chunks_time.yaml` next to each video in its folder under `videos/`.
2) Write one `dataset_generation_config.yaml` listing `{video_dir, video_file, chunks_time_file}` for all videos (plus all datagen args).
3) Run `src/data/generate_dataset.py --config <config.yaml>` to generate the dataset chunk-by-chunk.

Reference spec: `.agent/task.md` (kept concise and up-to-date).
Starter config: `scripts/tools/dataset_generation_config.yaml`.

---

## Helper scripts in this folder

- `mark_video_chunks.py`: interactive *frame-based* chunk marking (writes `chunks.yaml` with frame indices). Optional / legacy if you prefer timestamps.
- `timestamps_to_frames.py`: optional debugging/inspection helper to convert `chunks_time.yaml` (timestamps) → `chunks.yaml` (frame indices). Not required: `src/data/generate_dataset.py` consumes `chunks_time.yaml` directly.
