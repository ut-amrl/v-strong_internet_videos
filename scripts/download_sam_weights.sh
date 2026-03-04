#!/usr/bin/env bash
set -euo pipefail

# Download SAM (Segment Anything Model) checkpoints.
#
# Usage:
#   bash scripts/download_sam_weights.sh [size] [output]
#
# Sizes map to SAM variants:
#   small  → vit_b  → sam_vit_b_01ec64.pth   (~375 MB)
#   medium → vit_l  → sam_vit_l_0b3195.pth   (~1.2 GB)
#   large  → vit_h  → sam_vit_h_4b8939.pth   (~2.4 GB)
#
# Examples:
#   bash scripts/download_sam_weights.sh             # ViT-B (small)
#   bash scripts/download_sam_weights.sh medium      # ViT-L
#   bash scripts/download_sam_weights.sh large       # ViT-H (needed for distillation)

SIZE="${1:-large}"
OUTPUT="${2:-}"
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"

ARGS=(conda run --no-capture-output -n "${CONDA_ENV}" python -u scripts/download_backbone_weights.py
  --backbone sam
  --size "${SIZE}"
)

if [[ -n "${OUTPUT}" ]]; then
  ARGS+=(--output "${OUTPUT}")
fi

"${ARGS[@]}"
