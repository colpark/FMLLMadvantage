#!/usr/bin/env bash
#
# 05_train_sae.sh -- Stage 5 of the materials port.
# Train Top-K SAE on cached CHGNet embeddings.
#
# Usage:
#   bash scripts/materials/05_train_sae.sh
#
# Environment variables (optional):
#   EMBEDDINGS_DIR  default: latest under runs/materials/embeddings/
#   HIDDEN_DIM      default: 1024
#   K               default: 32
#   EPOCHS          default: 30
#   BATCH_SIZE      default: 256
#   GPU             default: 0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

HIDDEN_DIM="${HIDDEN_DIM:-1024}"
K="${K:-32}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-256}"
GPU="${GPU:-0}"

EXTRA=()
EXTRA+=(--hidden-dim "${HIDDEN_DIM}")
EXTRA+=(--k "${K}")
EXTRA+=(--epochs "${EPOCHS}")
EXTRA+=(--batch-size "${BATCH_SIZE}")
if [ -n "${EMBEDDINGS_DIR:-}" ]; then
    EXTRA+=(--embeddings-dir "${EMBEDDINGS_DIR}")
fi

echo "==> Materials port Stage 5: train SAE"
echo "    EMBEDDINGS_DIR : ${EMBEDDINGS_DIR:-(latest)}"
echo "    HIDDEN_DIM     : ${HIDDEN_DIM}"
echo "    K              : ${K}"
echo "    EPOCHS         : ${EPOCHS}"
echo "    BATCH_SIZE     : ${BATCH_SIZE}"
echo "    GPU            : ${GPU}"
echo

CUDA_VISIBLE_DEVICES="${GPU}" uv run python scripts/materials/05_train_sae.py "${EXTRA[@]}"
