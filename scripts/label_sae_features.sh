#!/usr/bin/env bash
#
# label_sae_features.sh
#
# Phase 13 Stage 1: label every SAE feature using correlations with
# dataset attributes (motif, atom count, temperature, phase).
# Picks the latest SAE under checkpoints/sae/ unless overridden.
#
# Usage:
#   bash scripts/label_sae_features.sh
#
# Environment variables (optional):
#   N_SPECIMENS   (default: 10000)
#   TOP_N         (default: 50)
#   MIN_PURITY    (default: 0.70)
#   MIN_CORR      (default: 0.30)
#   GPU           (default: 0)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

N_SPECIMENS="${N_SPECIMENS:-10000}"
TOP_N="${TOP_N:-50}"
MIN_PURITY="${MIN_PURITY:-0.70}"
MIN_CORR="${MIN_CORR:-0.30}"
GPU="${GPU:-0}"

echo "==> SAE feature labelling"
echo "    Specimens : ${N_SPECIMENS}"
echo "    Top N     : ${TOP_N}"
echo "    Min purity: ${MIN_PURITY}"
echo "    Min corr  : ${MIN_CORR}"
echo "    GPU       : ${GPU}"
echo

CUDA_VISIBLE_DEVICES="${GPU}" uv run python scripts/label_sae_features.py \
    --n-specimens "${N_SPECIMENS}" \
    --top-n "${TOP_N}" \
    --min-purity "${MIN_PURITY}" \
    --min-corr "${MIN_CORR}"
