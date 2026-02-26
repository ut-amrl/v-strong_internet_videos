#!/usr/bin/env bash
set -euo pipefail

# Inference convenience script:
#   input video -> generate dataset -> render side-by-side overlay video
#
# Usage:
#   bash scripts/run_inference_video.sh <input_video> <ckpt_path> <output_mp4>
#
# Example:
#   bash scripts/run_inference_video.sh videos/output_2.mp4 logs/checkpoints/vstrong/last.ckpt logs/overlay.mp4

VIDEO_PATH="${1:?input video path required}"
CKPT_PATH="${2:?checkpoint path required}"
OUT_MP4="${3:?output mp4 path required}"

FPS="${FPS:-5}"
# By default, inference should run on all frames.
# Set to 0 to sample at `FPS` instead.
EVERY_FRAME="${EVERY_FRAME:-1}"
SAM_CKPT="${SAM_CKPT:-checkpoints/sam_vit_b_01ec64.pth}"
SAM_TYPE="${SAM_TYPE:-vit_b}"
PROMPT_MAX_POINTS="${PROMPT_MAX_POINTS:-32}"
NUM_POS="${NUM_POS:-50}"
NUM_NEG="${NUM_NEG:-50}"
NEG_TOP_FRAC="${NEG_TOP_FRAC:-0.3}"
SAMPLE_MARGIN_PX="${SAMPLE_MARGIN_PX:-10}"

POINTS_PER_CLASS="${POINTS_PER_CLASS:-64}"
POINTS_SOURCE="${POINTS_SOURCE:-mixed}"   # saved|mask|mixed

TITLE="${TITLE:-V-STRONG}"
OUTPUT_HEIGHT="${OUTPUT_HEIGHT:-}"        # optional
MAX_FRAMES="${MAX_FRAMES:-}"              # optional

TS="$(date +%Y%m%d_%H%M%S)"
STEM="$(basename "${VIDEO_PATH}")"
STEM="${STEM%.*}"
DATASET_DIR="${DATASET_DIR:-/tmp/vstrong_infer/${STEM}_${TS}}"

echo "VIDEO_PATH  = ${VIDEO_PATH}"
echo "CKPT_PATH   = ${CKPT_PATH}"
echo "OUT_MP4     = ${OUT_MP4}"
echo "DATASET_DIR = ${DATASET_DIR}"
echo

rm -rf "${DATASET_DIR}"

echo "==> 1) Generate dataset"
EXTRACT_FLAGS=()
if [[ "${EVERY_FRAME}" == "1" ]]; then
  EXTRACT_FLAGS+=(--every_frame)
fi

# `conda run` captures output by default; disable capture so tqdm renders live.
conda run --no-capture-output -n env_isaaclab python -u src/data/generate_dataset.py \
  --video "${VIDEO_PATH}" \
  --output "${DATASET_DIR}" \
  --fps "${FPS}" \
  "${EXTRACT_FLAGS[@]}" \
  --checkpoint "${SAM_CKPT}" \
  --sam_type "${SAM_TYPE}" \
  --prompt_max_points "${PROMPT_MAX_POINTS}" \
  --num_pos "${NUM_POS}" \
  --num_neg "${NUM_NEG}" \
  --neg_top_frac "${NEG_TOP_FRAC}" \
  --sample_margin_px "${SAMPLE_MARGIN_PX}"

echo "==> 2) Render side-by-side video"
ARGS=(conda run --no-capture-output -n env_isaaclab python -u src/render_overlay_video.py
  --dataset_dir "${DATASET_DIR}"
  --ckpt_path "${CKPT_PATH}"
  --output "${OUT_MP4}"
  --title "${TITLE}"
  --points_per_class "${POINTS_PER_CLASS}"
  --points_source "${POINTS_SOURCE}"
  --neg_top_frac "${NEG_TOP_FRAC}"
  --sample_margin_px "${SAMPLE_MARGIN_PX}"
)

if [[ -n "${OUTPUT_HEIGHT}" ]]; then
  ARGS+=(--output_height "${OUTPUT_HEIGHT}")
fi
if [[ -n "${MAX_FRAMES}" ]]; then
  ARGS+=(--max_frames "${MAX_FRAMES}")
fi

"${ARGS[@]}"

echo
echo "Done: ${OUT_MP4}"
