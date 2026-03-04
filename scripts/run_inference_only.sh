#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(cd "$(dirname "$0")/.." && pwd)/src"

# Image-only inference pipeline:
#   input video -> extract frames -> V-STRONG image-only overlay video
#
# Usage:
#   bash scripts/run_inference_only.sh <input_video> <ckpt_path> <output_mp4>

VIDEO_PATH="${1:?input video path required}"
CKPT_PATH="${2:?checkpoint path required}"
OUT_MP4="${3:?output mp4 path required}"

FPS="${FPS:-0}"                          # <=0 means every frame
OUTPUT_FPS="${OUTPUT_FPS:-}"             # optional override
MAX_FRAMES="${MAX_FRAMES:-}"             # optional limit
OUTPUT_HEIGHT="${OUTPUT_HEIGHT:-}"       # optional resize
TITLE="${TITLE:-V-STRONG Inference}"

# If caller passes ".../last.ckpt", prefer the newest last*.ckpt in that folder.
if [[ "$(basename "${CKPT_PATH}")" == "last.ckpt" ]]; then
  CKPT_DIR="$(dirname "${CKPT_PATH}")"
  NEWEST_LAST="$(ls -1t "${CKPT_DIR}"/last*.ckpt 2>/dev/null | head -n1 || true)"
  if [[ -n "${NEWEST_LAST}" ]]; then
    CKPT_PATH="${NEWEST_LAST}"
  fi
fi

echo "VIDEO_PATH = ${VIDEO_PATH}"
echo "CKPT_PATH  = ${CKPT_PATH}"
echo "OUT_MP4    = ${OUT_MP4}"
echo

ARGS=(conda run --no-capture-output -n env_isaaclab python -u src/infer_video_rgb_only.py
  --video "${VIDEO_PATH}"
  --ckpt_path "${CKPT_PATH}"
  --output "${OUT_MP4}"
  --title "${TITLE}"
  --fps "${FPS}"
)

if [[ -n "${OUTPUT_FPS}" ]]; then
  ARGS+=(--output_fps "${OUTPUT_FPS}")
fi
if [[ -n "${MAX_FRAMES}" ]]; then
  ARGS+=(--max_frames "${MAX_FRAMES}")
fi
if [[ -n "${OUTPUT_HEIGHT}" ]]; then
  ARGS+=(--output_height "${OUTPUT_HEIGHT}")
fi

"${ARGS[@]}"
