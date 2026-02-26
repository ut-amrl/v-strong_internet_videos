# v-strong_internet_videos
Unofficial Implementation of V-STRONG for internet videos

```
bash scripts/run_full_pipeline.sh videos/output_2.mp4 data/output_2
```

```
 bash scripts/run_inference_only.sh videos/output_2.mp4 logs/checkpoints/vstrong/last.ckpt output.mp4
```

## Environment
All commands assume the conda env exists:
- `conda run -n env_isaaclab ...`
If you want live progress bars (tqdm), prefer `conda run --no-capture-output -n env_isaaclab ...`.

## 1) Generate dataset (video → frames → breadcrumbs → SAM masks → pos/neg)
Example:
```bash
conda run -n env_isaaclab python -u src/data/generate_dataset.py \
  --video videos/output_2.mp4 \
  --output data/output_2 \
  --fps 5 \
  --checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --sam_type vit_b
```

## 2) Train (contrastive)
```bash
conda run -n env_isaaclab python -u src/train_vstrong.py \
  --dataset_dir data/output_2 \
  --sam_checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --sam_type vit_b \
  --points_per_class 64 \
  --points_source mixed \
  --batch_size 1 \
  --num_workers 0 \
  --max_epochs 5 \
  --wandb_mode disabled
```
Checkpoints go to `logs/checkpoints/vstrong/` (including `logs/checkpoints/vstrong/last.ckpt`).

## 3) Inference (input video → output side-by-side MP4)
There are two modes:
- `scripts/run_inference_only.sh`: paper-style image-only inference (`video -> extract frames -> model inference with EMA traversability vector in checkpoint -> output video`)
- `scripts/run_inference_video.sh`: full preprocessing mode (`video -> dataset generation with SAM/points -> output video`)

### Inference-only (recommended for just model inference)
```bash
bash scripts/run_inference_only.sh \
  videos/output_2.mp4 \
  logs/checkpoints/vstrong/last.ckpt \
  logs/overlay_inference_only.mp4
```
Note: this requires a checkpoint trained with the current codebase (which saves the EMA traversability vector in the checkpoint).

### Full preprocessing + render
This runs full preprocessing for the input video and then renders a side-by-side video:
- left: RGB
- right: traversability overlay

Recommended:
```bash
bash scripts/run_inference_video.sh \
  videos/output_2.mp4 \
  logs/checkpoints/vstrong/last.ckpt \
  logs/overlay_side_by_side.mp4
```
By default, `scripts/run_inference_video.sh` extracts **every frame** from the input video.
To downsample instead, run with `EVERY_FRAME=0` and set `FPS`:
```bash
EVERY_FRAME=0 FPS=5 bash scripts/run_inference_video.sh videos/output_2.mp4 logs/checkpoints/vstrong/last.ckpt logs/overlay.mp4
```

You can also run the renderer directly if you already have a dataset directory:
```bash
conda run -n env_isaaclab python -u src/render_overlay_video.py \
  --dataset_dir data/output_2 \
  --ckpt_path logs/checkpoints/vstrong/last.ckpt \
  --output logs/overlay_side_by_side.mp4 \
  --title "V-STRONG" \
  --max_frames 200
```
