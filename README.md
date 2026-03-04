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
Training is now config-driven. The canonical entrypoint is:
```bash
conda run -n env_isaaclab python -u train.py --config configs/sam_small.yaml
```

Example configs are provided:
- `configs/sam_small.yaml`
- `configs/sam_small_unfrozen.yaml`
- `configs/nanosam_small.yaml`
- `configs/dino_small.yaml`
- `configs/dino_small_unfrozen.yaml`
- `configs/dinov2_large.yaml`

To pre-download local DINO-family checkpoints:
```bash
bash scripts/download_nanosam_weights.sh small
bash scripts/download_dino_weights.sh small
bash scripts/download_dinov2_weights.sh large
```

By default these save to:
- `checkpoints/nanosam_resnet18.pth`
- `checkpoints/dino_vits16.pth`
- `checkpoints/dinov2_vitl14.pth`

Backbone selection lives in the config under `model:`:
```yaml
model:
  backbone: sam      # sam | nanosam | dino | dinov2
  size: small        # small | medium | large
  checkpoint: checkpoints/sam_vit_b_01ec64.pth
  img_size: 1024
  freeze_backbone: true
```

The config file is the source of truth for:
- dataset path
- backbone family / size
- optimizer settings
- trainer settings
- logging and checkpoint paths

To finetune the encoder instead of freezing it, set:
```yaml
model:
  freeze_backbone: false

optimizer:
  backbone_lr: 0.0001
```

For `nanosam`, the current training path uses a ResNet-based frozen image encoder (`small -> resnet18`, `medium -> resnet34`, `large -> resnet50`) initialized from torchvision ImageNet weights or a local checkpoint.

Checkpoints go to the `paths.checkpoint_dir` configured in the YAML (for example `logs/checkpoints/vstrong/`).

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
