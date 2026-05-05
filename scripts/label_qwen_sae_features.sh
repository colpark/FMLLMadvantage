#!/usr/bin/env bash
#
# label_qwen_sae_features.sh
#
# Phase 15 Stage C: label every Qwen-SAE feature using verdict /
# correctness / motif / phase locks plus N/T correlations. Picks
# the latest SAE under checkpoints/qwen_sae/ and the latest
# activations under runs/qwen_activations/ unless overridden.
#
# Usage:
#   bash scripts/label_qwen_sae_features.sh
#
# Environment variables (optional):
#   SAE_DIR           default: latest under checkpoints/qwen_sae/
#   ACTIVATIONS_DIR   default: latest under runs/qwen_activations/
#   TOP_N             default: 50
#   MIN_PURITY        default: 0.70
#   MIN_CORR          default: 0.30
#   GPU               default: 0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

TOP_N="${TOP_N:-50}"
MIN_PURITY="${MIN_PURITY:-0.70}"
MIN_CORR="${MIN_CORR:-0.30}"
GPU="${GPU:-0}"

EXTRA=()
EXTRA+=(--top-n "${TOP_N}")
EXTRA+=(--min-purity "${MIN_PURITY}")
EXTRA+=(--min-corr "${MIN_CORR}")
if [ -n "${SAE_DIR:-}" ]; then
    EXTRA+=(--sae-dir "${SAE_DIR}")
fi
if [ -n "${ACTIVATIONS_DIR:-}" ]; then
    EXTRA+=(--activations-dir "${ACTIVATIONS_DIR}")
fi

echo "==> Phase 15 Stage C: label Qwen-SAE features"
echo "    SAE dir         : ${SAE_DIR:-(latest under checkpoints/qwen_sae/)}"
echo "    Activations dir : ${ACTIVATIONS_DIR:-(latest under runs/qwen_activations/)}"
echo "    Top N           : ${TOP_N}"
echo "    Min purity      : ${MIN_PURITY}"
echo "    Min corr        : ${MIN_CORR}"
echo "    GPU             : ${GPU}"
echo

CUDA_VISIBLE_DEVICES="${GPU}" uv run python scripts/label_qwen_sae_features.py \
    "${EXTRA[@]}"
