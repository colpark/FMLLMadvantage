#!/usr/bin/env bash
#
# train_qwen_sae.sh
#
# Phase 15 Stage B: train a Top-K SAE on harvested Qwen residual
# activations. Picks the latest activations.npy under
# runs/qwen_activations/ unless overridden.
#
# Usage:
#   bash scripts/train_qwen_sae.sh
#
# Environment variables (optional):
#   ACTIVATIONS_DIR   default: latest under runs/qwen_activations/
#   HIDDEN_DIM        default: 16384 (~4.5x Qwen 7B residual dim)
#   K                 default: 64 (Top-K active features per row)
#   EPOCHS            default: 30
#   BATCH_SIZE        default: 128
#   GPU               default: 0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

HIDDEN_DIM="${HIDDEN_DIM:-16384}"
K="${K:-64}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-128}"
GPU="${GPU:-0}"

EXTRA=()
EXTRA+=(--hidden-dim "${HIDDEN_DIM}")
EXTRA+=(--k "${K}")
EXTRA+=(--epochs "${EPOCHS}")
EXTRA+=(--batch-size "${BATCH_SIZE}")
if [ -n "${ACTIVATIONS_DIR:-}" ]; then
    EXTRA+=(--activations-dir "${ACTIVATIONS_DIR}")
fi

echo "==> Phase 15 Stage B: SAE on Qwen activations"
echo "    Activations dir : ${ACTIVATIONS_DIR:-(latest under runs/qwen_activations/)}"
echo "    Hidden dim      : ${HIDDEN_DIM}"
echo "    Top-K           : ${K}"
echo "    Epochs          : ${EPOCHS}"
echo "    Batch size      : ${BATCH_SIZE}"
echo "    GPU             : ${GPU}"
echo

CUDA_VISIBLE_DEVICES="${GPU}" uv run python scripts/train_qwen_sae.py \
    "${EXTRA[@]}"
