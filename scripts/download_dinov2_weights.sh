#!/usr/bin/env bash
set -euo pipefail

# Download and save DINOv2 weights locally.
#
# Usage:
#   bash scripts/download_dinov2_weights.sh [size] [output]
#
# Examples:
#   bash scripts/download_dinov2_weights.sh
#   bash scripts/download_dinov2_weights.sh medium
#   bash scripts/download_dinov2_weights.sh large checkpoints/dinov2_vitl14_custom.pth

SIZE="${1:-small}"
OUTPUT="${2:-}"
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"

ARGS=(conda run --no-capture-output -n "${CONDA_ENV}" python -u scripts/download_backbone_weights.py
  --backbone dinov2
  --size "${SIZE}"
)

if [[ -n "${OUTPUT}" ]]; then
  ARGS+=(--output "${OUTPUT}")
fi

"${ARGS[@]}"
