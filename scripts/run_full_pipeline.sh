#!/usr/bin/env bash
set -euo pipefail

# Full V-STRONG pipeline:
#   video -> frames -> breadcrumbs -> SAM masks -> pos/neg points -> train -> eval overlays
#
# Usage:
#   bash scripts/run_full_pipeline.sh videos/output_2.mp4 data/output_2
#
# Notes:
# - Uses `conda run -n env_isaaclab ...` (no interactive activation needed).
# - Deletes the target dataset directory before regenerating it.

VIDEO_PATH="${1:-videos/output_2.mp4}"
DATASET_DIR="${2:-data/output_2}"

FPS="${FPS:-5}"
SAM_CKPT="${SAM_CKPT:-checkpoints/sam_vit_b_01ec64.pth}"
SAM_TYPE="${SAM_TYPE:-vit_b}"
PROMPT_MAX_POINTS="${PROMPT_MAX_POINTS:-32}"
NUM_POS="${NUM_POS:-50}"
NUM_NEG="${NUM_NEG:-50}"
NEG_TOP_FRAC="${NEG_TOP_FRAC:-0.3}"
SAMPLE_MARGIN_PX="${SAMPLE_MARGIN_PX:-10}"

POINTS_PER_CLASS="${POINTS_PER_CLASS:-64}"
POINTS_SOURCE="${POINTS_SOURCE:-mixed}"   # saved|mask|mixed
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-5}"

CKPT_DIR="${CKPT_DIR:-logs/checkpoints/vstrong}"
EVAL_OUT_DIR="${EVAL_OUT_DIR:-logs/eval_vstrong/train20}"

echo "VIDEO_PATH     = ${VIDEO_PATH}"
echo "DATASET_DIR    = ${DATASET_DIR}"
echo "FPS            = ${FPS}"
echo "SAM_CKPT       = ${SAM_CKPT}"
echo "SAM_TYPE       = ${SAM_TYPE}"
echo "POINTS_SOURCE  = ${POINTS_SOURCE}"
echo "POINTS_PER_CLASS = ${POINTS_PER_CLASS}"
echo "CKPT_DIR       = ${CKPT_DIR}"
echo "EVAL_OUT_DIR   = ${EVAL_OUT_DIR}"
echo

if [[ "${DATASET_DIR}" != data/* ]]; then
  echo "Refusing to delete non-data/ path: ${DATASET_DIR}" >&2
  exit 2
fi

echo "==> Cleaning dataset dir: ${DATASET_DIR}"
rm -rf "${DATASET_DIR}"

echo "==> Generating dataset (frames + breadcrumbs + SAM + pos/neg)"
conda run -n env_isaaclab python -u src/data/generate_dataset.py \
  --video "${VIDEO_PATH}" \
  --output "${DATASET_DIR}" \
  --fps "${FPS}" \
  --checkpoint "${SAM_CKPT}" \
  --sam_type "${SAM_TYPE}" \
  --prompt_max_points "${PROMPT_MAX_POINTS}" \
  --num_pos "${NUM_POS}" \
  --num_neg "${NUM_NEG}" \
  --neg_top_frac "${NEG_TOP_FRAC}" \
  --sample_margin_px "${SAMPLE_MARGIN_PX}"

echo "==> Training V-STRONG"
conda run -n env_isaaclab python -u src/train_vstrong.py \
  --dataset_dir "${DATASET_DIR}" \
  --sam_checkpoint "${SAM_CKPT}" \
  --sam_type "${SAM_TYPE}" \
  --points_per_class "${POINTS_PER_CLASS}" \
  --points_source "${POINTS_SOURCE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --max_epochs "${MAX_EPOCHS}" \
  --checkpoint_dir "${CKPT_DIR}" \
  --wandb_mode disabled

LAST_CKPT="${CKPT_DIR}/last.ckpt"
if [[ ! -f "${LAST_CKPT}" ]]; then
  echo "Expected checkpoint not found: ${LAST_CKPT}" >&2
  exit 3
fi

echo "==> Eval on 20 train frames (writes overlays)"
conda run -n env_isaaclab python -u src/eval_vstrong.py \
  --dataset_dir "${DATASET_DIR}" \
  --ckpt_path "${LAST_CKPT}" \
  --output_dir "${EVAL_OUT_DIR}" \
  --split train \
  --num_frames 20 \
  --points_per_class "${POINTS_PER_CLASS}" \
  --points_source "${POINTS_SOURCE}" \
  --neg_top_frac "${NEG_TOP_FRAC}" \
  --sample_margin_px "${SAMPLE_MARGIN_PX}"

echo
echo "Done:"
echo "  Dataset:   ${DATASET_DIR}"
echo "  Checkpoint:${LAST_CKPT}"
echo "  Eval viz:  ${EVAL_OUT_DIR}"

