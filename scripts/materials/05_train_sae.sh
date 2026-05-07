#!/usr/bin/env bash
#
# 05_train_sae.sh -- Stage 5 of the materials port.
# Train Top-K SAE on cached CHGNet embeddings, with optional
# dead-feature resampling (Bricken et al. 2023).
#
# Usage:
#   bash scripts/materials/05_train_sae.sh
#   RESAMPLE_EVERY=0 bash scripts/materials/05_train_sae.sh    # disable resampling
#
# Environment variables (optional):
#   EMBEDDINGS_DIR     default: latest under runs/materials/embeddings/
#   HIDDEN_DIM         default: 1024
#   K                  default: 32
#   EPOCHS             default: 30
#   BATCH_SIZE         default: 256
#   RESAMPLE_EVERY     default: 1000 (0 disables; was 0 in the first run)
#   RESAMPLE_WINDOW    default: 0 (means "use RESAMPLE_EVERY")
#   RESAMPLE_THRESHOLD default: 0 (features with <= this fires are dead)
#   GPU                default: 0
#
# Recommended for the materials port:
#   RESAMPLE_EVERY=1000 RESAMPLE_THRESHOLD=0 bash scripts/materials/05_train_sae.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Defaults validated 2026-05-07 on 50K MP specimens with CHGNet 64-dim
# pooled embedding. 256/16 with 500-step resampling produced 4.3% dead
# features, 95.7% coverage, and acceptable reconstruction MSE (0.025 in
# normalized space). 1024/32 had 19% dead and severe mode collapse.
HIDDEN_DIM="${HIDDEN_DIM:-256}"
K="${K:-16}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-256}"
RESAMPLE_EVERY="${RESAMPLE_EVERY:-500}"
RESAMPLE_WINDOW="${RESAMPLE_WINDOW:-0}"
RESAMPLE_THRESHOLD="${RESAMPLE_THRESHOLD:-0}"
GPU="${GPU:-0}"

EXTRA=()
EXTRA+=(--hidden-dim "${HIDDEN_DIM}")
EXTRA+=(--k "${K}")
EXTRA+=(--epochs "${EPOCHS}")
EXTRA+=(--batch-size "${BATCH_SIZE}")
EXTRA+=(--resample-every "${RESAMPLE_EVERY}")
EXTRA+=(--resample-window "${RESAMPLE_WINDOW}")
EXTRA+=(--resample-threshold "${RESAMPLE_THRESHOLD}")
if [ -n "${EMBEDDINGS_DIR:-}" ]; then
    EXTRA+=(--embeddings-dir "${EMBEDDINGS_DIR}")
fi

echo "==> Materials port Stage 5: train SAE"
echo "    EMBEDDINGS_DIR     : ${EMBEDDINGS_DIR:-(latest)}"
echo "    HIDDEN_DIM         : ${HIDDEN_DIM}"
echo "    K                  : ${K}"
echo "    EPOCHS             : ${EPOCHS}"
echo "    BATCH_SIZE         : ${BATCH_SIZE}"
echo "    RESAMPLE_EVERY     : ${RESAMPLE_EVERY}"
echo "    RESAMPLE_WINDOW    : ${RESAMPLE_WINDOW}"
echo "    RESAMPLE_THRESHOLD : ${RESAMPLE_THRESHOLD}"
echo "    GPU                : ${GPU}"
echo

CUDA_VISIBLE_DEVICES="${GPU}" uv run python scripts/materials/05_train_sae.py "${EXTRA[@]}"
