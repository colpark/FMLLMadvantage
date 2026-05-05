#!/usr/bin/env bash
#
# train_sae.sh
#
# Phase 13 Stage 0: train a Top-K sparse autoencoder on the frozen
# supervised FM2's CLS embeddings. The trained SAE produces sparse
# (~32 active) latent codes per specimen that Stage 1 will label.
#
# Usage:
#   bash scripts/train_sae.sh
#
# Environment variables (optional):
#   HIDDEN_DIM        (default: 1024)
#   K                 (default: 32)
#   EPOCHS            (default: 30)
#   N_SPECIMENS       (default: 20000)
#   GPU               (default: 0)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

HIDDEN_DIM="${HIDDEN_DIM:-1024}"
K="${K:-32}"
EPOCHS="${EPOCHS:-30}"
N_SPECIMENS="${N_SPECIMENS:-20000}"
GPU="${GPU:-0}"

echo "==> SAE training on FM2 CLS"
echo "    Hidden dim    : ${HIDDEN_DIM}"
echo "    Top-K         : ${K}"
echo "    Epochs        : ${EPOCHS}"
echo "    Specimens     : ${N_SPECIMENS}"
echo "    GPU           : ${GPU}"
echo

CUDA_VISIBLE_DEVICES="${GPU}" uv run python scripts/train_sae.py \
    --hidden-dim "${HIDDEN_DIM}" \
    --k "${K}" \
    --epochs "${EPOCHS}" \
    --n-specimens "${N_SPECIMENS}"
