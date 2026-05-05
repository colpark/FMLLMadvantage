#!/usr/bin/env bash
#
# run_baseline_qwen_steered.sh
#
# Phase 15 Stage D: run Pipeline A with Qwen activation steering.
# Every Qwen forward pass during the OHVD loop has an
# ActivationSteerer hook attached at the SAE's training layer,
# adding ``coefficient * decoder_column[FEATURE_IDX]`` to the
# residual stream. The intent is the canonical Templeton et al. /
# Golden Gate Claude experiment: ablate a "wrong-PASS" feature
# (negative coefficient) and measure whether hallucination_rate
# drops.
#
# Output goes to runs/holdout/full_steered_<fid>_<coef>/<run_id>/
# so scripts/evaluate_baselines.sh auto-discovers it as a column.
#
# Usage:
#   FEATURE_IDX=8421 COEFFICIENT=-2.0 bash scripts/run_baseline_qwen_steered.sh
#   FEATURE_IDX=8421 COEFFICIENT=-2.0 \
#       SPECIMEN_IDS_FILE=runs/holdout_lock/ids.json \
#       bash scripts/run_baseline_qwen_steered.sh
#
# Required environment:
#   FEATURE_IDX        SAE feature index (read steering_candidates.yaml
#                      for principled choices)
#
# Optional environment:
#   COEFFICIENT        steering multiplier (default: -1.0; negative
#                      ablates, positive amplifies)
#   SAE_DIR            checkpoints/qwen_sae/<run-id>; default latest
#   LAYER_PATH         must match the SAE's training layer
#                      (default: model.layers.14)
#   SPECIMEN_IDS_FILE  default: unset; overrides START/COUNT
#   START              default: 0
#   COUNT              default: 200
#   MAX_STEPS          default: 16
#   ABLATION           default: V4
#   LLM_TEMP           default: 0.4
#   GPU                default: 0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [ -z "${FEATURE_IDX:-}" ]; then
    echo "ERROR: FEATURE_IDX must be set" >&2
    echo "       Read runs/qwen_sae_labels/<run-id>/steering_candidates.yaml" >&2
    echo "       to pick a candidate from the wrong_pass list." >&2
    exit 2
fi

COEFFICIENT="${COEFFICIENT:--1.0}"
LAYER_PATH="${LAYER_PATH:-model.layers.14}"
START="${START:-0}"
COUNT="${COUNT:-200}"
MAX_STEPS="${MAX_STEPS:-16}"
ABLATION="${ABLATION:-V4}"
LLM_TEMP="${LLM_TEMP:-0.4}"
GPU="${GPU:-0}"

EXTRA=()
EXTRA+=(--feature-idx "${FEATURE_IDX}")
EXTRA+=(--coefficient "${COEFFICIENT}")
EXTRA+=(--layer-path "${LAYER_PATH}")
EXTRA+=(--start "${START}")
EXTRA+=(--count "${COUNT}")
EXTRA+=(--max-steps "${MAX_STEPS}")
EXTRA+=(--ablation "${ABLATION}")
EXTRA+=(--llm-temperature "${LLM_TEMP}")
if [ -n "${SAE_DIR:-}" ]; then
    EXTRA+=(--sae-dir "${SAE_DIR}")
fi
if [ -n "${SPECIMEN_IDS_FILE:-}" ]; then
    EXTRA+=(--specimen-ids-file "${SPECIMEN_IDS_FILE}")
fi
if [ -n "${ADAPTER_PATH:-}" ]; then
    EXTRA+=(--adapter-path "${ADAPTER_PATH}")
fi

echo "==> Phase 15 Stage D: steered Pipeline A"
echo "    Feature idx     : ${FEATURE_IDX}"
echo "    Coefficient     : ${COEFFICIENT}"
echo "    Layer           : ${LAYER_PATH}"
echo "    SAE dir         : ${SAE_DIR:-(latest under checkpoints/qwen_sae/)}"
echo "    Specimens       : ${SPECIMEN_IDS_FILE:-[${START}, $((START + COUNT)))}"
echo "    GPU             : ${GPU}"
echo

CUDA_VISIBLE_DEVICES="${GPU}" uv run python scripts/run_baseline_qwen_steered.py \
    "${EXTRA[@]}"
