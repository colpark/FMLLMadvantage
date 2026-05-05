#!/usr/bin/env bash
#
# sae_causal_audit.sh
#
# Phase 14: causal audit of SAE features against FM2's energy head.
# For every feature in the trained SAE, run a knock-out and a
# knock-in intervention and measure the resulting change in FM2's
# predicted per-atom energy. Features that meet the configured
# effect-size threshold land in causal_filter.json; the LLM-facing
# baseline can then surface only the causally meaningful subset.
#
# Picks the latest SAE under checkpoints/sae/ and the latest labels
# under runs/sae_labels/ unless overridden.
#
# Usage:
#   bash scripts/sae_causal_audit.sh
#
# Environment variables (optional):
#   N_SPECIMENS         (default: 2000)
#   MIN_NORM_EFFECT     (default: 0.10)
#   MIN_ACTIVATION_RATE (default: 0.005)
#   FEATURE_SUBSET      (default: unset; comma-separated to audit a slice)
#   GPU                 (default: 0)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

N_SPECIMENS="${N_SPECIMENS:-2000}"
MIN_NORM_EFFECT="${MIN_NORM_EFFECT:-0.10}"
MIN_ACTIVATION_RATE="${MIN_ACTIVATION_RATE:-0.005}"
GPU="${GPU:-0}"

EXTRA=()
if [ -n "${FEATURE_SUBSET:-}" ]; then
    EXTRA+=(--feature-subset "${FEATURE_SUBSET}")
fi

echo "==> SAE causal audit (FM2 energy head)"
echo "    Specimens          : ${N_SPECIMENS}"
echo "    Min norm effect    : ${MIN_NORM_EFFECT}"
echo "    Min activation rate: ${MIN_ACTIVATION_RATE}"
echo "    Feature subset     : ${FEATURE_SUBSET:-(all)}"
echo "    GPU                : ${GPU}"
echo

CUDA_VISIBLE_DEVICES="${GPU}" uv run python scripts/sae_causal_audit.py \
    --n-specimens "${N_SPECIMENS}" \
    --min-norm-effect "${MIN_NORM_EFFECT}" \
    --min-activation-rate "${MIN_ACTIVATION_RATE}" \
    "${EXTRA[@]}"
