#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(cd "$(dirname "$0")/.." && pwd)/src"

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
BREADCRUMB_MAX_TRACK_LEN="${BREADCRUMB_MAX_TRACK_LEN:-20}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
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
TRAIN_ACCELERATOR="${TRAIN_ACCELERATOR:-gpu}"
TRAIN_DEVICES="${TRAIN_DEVICES:-1}"
TRAIN_STRATEGY="${TRAIN_STRATEGY:-auto}"
TRAIN_NUM_NODES="${TRAIN_NUM_NODES:-1}"

if [[ "${TRAIN_STRATEGY}" == "single_device" ]]; then
  echo "TRAIN_STRATEGY=single_device maps to CPU on this Lightning setup; overriding to auto for GPU."
  TRAIN_STRATEGY="auto"
fi

CKPT_DIR="${CKPT_DIR:-logs/checkpoints/vstrong}"
EVAL_OUT_DIR="${EVAL_OUT_DIR:-logs/eval_vstrong/train20}"

echo "VIDEO_PATH     = ${VIDEO_PATH}"
echo "DATASET_DIR    = ${DATASET_DIR}"
echo "FPS            = ${FPS}"
echo "BREADCRUMB_MAX_TRACK_LEN = ${BREADCRUMB_MAX_TRACK_LEN}"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "SAM_CKPT       = ${SAM_CKPT}"
echo "SAM_TYPE       = ${SAM_TYPE}"
echo "POINTS_SOURCE  = ${POINTS_SOURCE}"
echo "POINTS_PER_CLASS = ${POINTS_PER_CLASS}"
echo "TRAIN_ACCELERATOR = ${TRAIN_ACCELERATOR}"
echo "TRAIN_DEVICES   = ${TRAIN_DEVICES}"
echo "TRAIN_STRATEGY  = ${TRAIN_STRATEGY}"
echo "TRAIN_NUM_NODES = ${TRAIN_NUM_NODES}"
echo "CKPT_DIR       = ${CKPT_DIR}"
echo "EVAL_OUT_DIR   = ${EVAL_OUT_DIR}"
echo

if [[ "${DATASET_DIR}" != data/* ]]; then
  echo "Refusing to delete non-data/ path: ${DATASET_DIR}" >&2
  exit 2
fi

echo "==> Cleaning dataset dir: ${DATASET_DIR}"
rm -rf "${DATASET_DIR}"

export CUDA_VISIBLE_DEVICES

echo "==> Generating dataset (frames + SAM masks + pos/neg points)"
conda run --no-capture-output -n env_isaaclab python -u src/data/generate_dataset.py \
  --video "${VIDEO_PATH}" \
  --output "${DATASET_DIR}" \
  --fps "${FPS}" \
  --breadcrumb_max_track_len "${BREADCRUMB_MAX_TRACK_LEN}" \
  --checkpoint "${SAM_CKPT}" \
  --sam_type "${SAM_TYPE}" \
  --prompt_max_points "${PROMPT_MAX_POINTS}" \
  --num_pos "${NUM_POS}" \
  --num_neg "${NUM_NEG}" \
  --neg_top_frac "${NEG_TOP_FRAC}" \
  --sample_margin_px "${SAMPLE_MARGIN_PX}"

echo "==> Training V-STRONG"
conda run --no-capture-output -n env_isaaclab python -u src/train_vstrong.py \
  --dataset_dir "${DATASET_DIR}" \
  --sam_checkpoint "${SAM_CKPT}" \
  --sam_type "${SAM_TYPE}" \
  --points_per_class "${POINTS_PER_CLASS}" \
  --points_source "${POINTS_SOURCE}" \
  --accelerator "${TRAIN_ACCELERATOR}" \
  --devices "${TRAIN_DEVICES}" \
  --strategy "${TRAIN_STRATEGY}" \
  --num_nodes "${TRAIN_NUM_NODES}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --max_epochs "${MAX_EPOCHS}" \
  --checkpoint_dir "${CKPT_DIR}" \
  --wandb_mode disabled

# Pick the most recent "last*.ckpt" so we don't accidentally use stale last.ckpt
# when Lightning has emitted versioned files (e.g., last-v4.ckpt).
LAST_CKPT="$(ls -1t "${CKPT_DIR}"/last*.ckpt 2>/dev/null | head -n1 || true)"
if [[ -z "${LAST_CKPT}" || ! -f "${LAST_CKPT}" ]]; then
  echo "Expected checkpoint not found under ${CKPT_DIR}/last*.ckpt" >&2
  exit 3
fi

echo "==> Eval on 20 train frames (writes overlays)"
conda run --no-capture-output -n env_isaaclab python -u src/eval_vstrong.py \
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
