#!/usr/bin/env bash
set -euo pipefail

# Download and save NanoSAM-style ResNet encoder weights locally.
#
# Usage:
#   bash scripts/download_nanosam_weights.sh [size] [output]
#
# Examples:
#   bash scripts/download_nanosam_weights.sh
#   bash scripts/download_nanosam_weights.sh medium
#   bash scripts/download_nanosam_weights.sh large checkpoints/nanosam_resnet50_custom.pth

SIZE="${1:-small}"
OUTPUT="${2:-}"
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"

ARGS=(conda run --no-capture-output -n "${CONDA_ENV}" python -u scripts/download_backbone_weights.py
  --backbone nanosam
  --size "${SIZE}"
)

if [[ -n "${OUTPUT}" ]]; then
  ARGS+=(--output "${OUTPUT}")
fi

"${ARGS[@]}"
