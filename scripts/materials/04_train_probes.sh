#!/usr/bin/env bash
#
# 04_train_probes.sh -- Stage 4 of the materials port.
# Train probe bank on cached CHGNet embeddings.
#
# Usage:
#   bash scripts/materials/04_train_probes.sh
#
# Environment variables (optional):
#   EMBEDDINGS_DIR    default: latest under runs/materials/embeddings/
#   HIDDEN            default: 128
#   EPOCHS            default: 30
#   LR                default: 1e-3
#   BATCH_SIZE        default: 256
#   SPACE_GROUP_TOP_K default: 20
#   GPU               default: 0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

HIDDEN="${HIDDEN:-128}"
EPOCHS="${EPOCHS:-30}"
LR="${LR:-1e-3}"
BATCH_SIZE="${BATCH_SIZE:-256}"
SPACE_GROUP_TOP_K="${SPACE_GROUP_TOP_K:-20}"
GPU="${GPU:-0}"

EXTRA=()
EXTRA+=(--hidden "${HIDDEN}")
EXTRA+=(--epochs "${EPOCHS}")
EXTRA+=(--lr "${LR}")
EXTRA+=(--batch-size "${BATCH_SIZE}")
EXTRA+=(--space-group-top-k "${SPACE_GROUP_TOP_K}")
if [ -n "${EMBEDDINGS_DIR:-}" ]; then
    EXTRA+=(--embeddings-dir "${EMBEDDINGS_DIR}")
fi

echo "==> Materials port Stage 4: train probes"
echo "    EMBEDDINGS_DIR    : ${EMBEDDINGS_DIR:-(latest)}"
echo "    HIDDEN            : ${HIDDEN}"
echo "    EPOCHS            : ${EPOCHS}"
echo "    LR                : ${LR}"
echo "    BATCH_SIZE        : ${BATCH_SIZE}"
echo "    SPACE_GROUP_TOP_K : ${SPACE_GROUP_TOP_K}"
echo "    GPU               : ${GPU}"
echo

CUDA_VISIBLE_DEVICES="${GPU}" uv run python scripts/materials/04_train_probes.py "${EXTRA[@]}"
