#!/usr/bin/env bash
#
# train_probe_bank.sh
#
# Phase 11 Stage 0: train the probe bank on top of frozen FM2
# features. Saves to checkpoints/probes/<run_id>/. The probes the
# Stage 1 dataset builder consumes are: n_atoms, motif, phase,
# coordination, peak_position.
#
# Usage:
#   bash scripts/train_probe_bank.sh
#
# Environment variables (optional):
#   N_SPECIMENS   (default: 10000)
#   EPOCHS        (default: 50)
#   LR            (default: 1.0e-3)
#   HIDDEN        (default: 128)
#   GPU           (default: 0)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

N_SPECIMENS="${N_SPECIMENS:-10000}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-1.0e-3}"
HIDDEN="${HIDDEN:-128}"
GPU="${GPU:-0}"

echo "==> Probe bank training"
echo "    Specimens : ${N_SPECIMENS}"
echo "    Epochs    : ${EPOCHS}"
echo "    LR        : ${LR}"
echo "    Hidden    : ${HIDDEN}"
echo "    GPU       : ${GPU}"
echo

CUDA_VISIBLE_DEVICES="${GPU}" uv run python scripts/train_probe_bank.py \
    --n-specimens "${N_SPECIMENS}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --hidden "${HIDDEN}"
