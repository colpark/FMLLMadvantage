#!/usr/bin/env bash
#
# run_baseline_full_probes.sh
#
# Phase 12: probe-augmented Pipeline A. Runs the canonical OHVD loop
# (FM tools + multi-source verifier) on each specimen with FM2-derived
# probe outputs injected into the user prompt as approximate hints.
# Writes to runs/holdout/full_probes/<run_id>/ so
# scripts/evaluate_baselines.sh auto-discovers it as a fifth baseline.
#
# Usage:
#   bash scripts/run_baseline_full_probes.sh
#   SPECIMEN_IDS_FILE=runs/holdout_lock/ids.json bash scripts/run_baseline_full_probes.sh
#   ADAPTER_PATH=checkpoints/cot-sft/<run-id>/adapter \
#       SPECIMEN_IDS_FILE=runs/holdout_lock/ids.json \
#       bash scripts/run_baseline_full_probes.sh
#
# Environment variables (optional):
#   START              (default: 0)
#   COUNT              (default: 200)
#   SPECIMEN_IDS_FILE  (default: unset; overrides START/COUNT)
#   OUT_ROOT           (default: runs/holdout)
#   GPU                (default: 0)
#   BASE_MODEL         (default: Qwen/Qwen2.5-7B-Instruct)
#   ADAPTER_PATH       (default: unset; stack a LoRA adapter on top
#                       of the base LLM. The Phase 11 cot_sft adapter
#                       is a candidate but format mismatch may apply.)
#   PROBE_BANK_DIR     (default: latest under checkpoints/probes/)
#   MAX_STEPS          (default: 16)
#   ABLATION           (default: V4)
#   LLM_TEMP           (default: 0.4)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

START="${START:-0}"
COUNT="${COUNT:-200}"
OUT_ROOT="${OUT_ROOT:-runs/holdout}"
GPU="${GPU:-0}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
MAX_STEPS="${MAX_STEPS:-16}"
ABLATION="${ABLATION:-V4}"
LLM_TEMP="${LLM_TEMP:-0.4}"

EXTRA=()
if [ -n "${SPECIMEN_IDS_FILE:-}" ]; then
    EXTRA+=(--specimen-ids-file "${SPECIMEN_IDS_FILE}")
fi
if [ -n "${ADAPTER_PATH:-}" ]; then
    EXTRA+=(--adapter-path "${ADAPTER_PATH}")
fi
if [ -n "${PROBE_BANK_DIR:-}" ]; then
    EXTRA+=(--probe-bank-dir "${PROBE_BANK_DIR}")
fi

echo "==> Phase 12: probe-augmented Pipeline A"
echo "    Start         : ${START}"
echo "    Count         : ${COUNT}"
echo "    Output root   : ${OUT_ROOT}"
echo "    Base model    : ${BASE_MODEL}"
echo "    Adapter path  : ${ADAPTER_PATH:-(none)}"
echo "    Max steps     : ${MAX_STEPS}"
echo "    Ablation      : ${ABLATION}"
echo "    LLM temp      : ${LLM_TEMP}"
echo "    GPU           : ${GPU}"
echo

CUDA_VISIBLE_DEVICES="${GPU}" uv run python scripts/run_baseline_full_probes.py \
    --start "${START}" \
    --count "${COUNT}" \
    --out "${OUT_ROOT}" \
    --base-model "${BASE_MODEL}" \
    --max-steps "${MAX_STEPS}" \
    --ablation "${ABLATION}" \
    --llm-temperature "${LLM_TEMP}" \
    "${EXTRA[@]}"
