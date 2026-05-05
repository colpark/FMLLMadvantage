#!/usr/bin/env bash
#
# run_baseline_cot.sh
#
# Phase 11.B: run the CoT-SFT baseline on a list of specimens.
# Single-shot inference: probes -> Qwen+adapter -> Final commit JSON
# parsed into a Trajectory. The output trajectories.jsonl is schema-
# compatible with the other Phase 8a baselines, so
# scripts/evaluate_baselines.sh auto-discovers it.
#
# Default writes to runs/holdout/cot_sft/<run_id>/, alongside the
# other held-out baselines, so the existing held-out comparison
# picks it up.
#
# Usage:
#   bash scripts/run_baseline_cot.sh
#   START=0 COUNT=200 bash scripts/run_baseline_cot.sh
#   SPECIMEN_IDS_FILE=runs/holdout_lock/ids.json bash scripts/run_baseline_cot.sh
#
# Environment variables (optional):
#   START              (default: 0)
#   COUNT              (default: 200)
#   SPECIMEN_IDS_FILE  (default: unset; overrides START/COUNT)
#   OUT_ROOT           (default: runs/holdout)
#   GPU                (default: 0)
#   BASE_MODEL         (default: Qwen/Qwen2.5-7B-Instruct)
#   ADAPTER_PATH       (default: latest under checkpoints/cot-sft/)
#   PROBE_BANK_DIR     (default: latest under checkpoints/probes/)
#   QUANTIZE           (default: 4bit; 'none' for bf16, '8bit' for
#                       8-bit. 4bit drops Qwen's GPU memory from
#                       ~14GB to ~4GB. Use 'none' if you have ample
#                       memory and want full-precision generation.)
#   MAX_NEW_TOKENS     (default: 256)
#   BATCH_SIZE         (default: 16; FM2 forward batch only)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

START="${START:-0}"
COUNT="${COUNT:-200}"
OUT_ROOT="${OUT_ROOT:-runs/holdout}"
GPU="${GPU:-0}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
QUANTIZE="${QUANTIZE:-4bit}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
BATCH_SIZE="${BATCH_SIZE:-16}"

EXTRA=()
EXTRA+=(--quantize "${QUANTIZE}")
EXTRA+=(--max-new-tokens "${MAX_NEW_TOKENS}")
EXTRA+=(--batch-size "${BATCH_SIZE}")
if [ -n "${SPECIMEN_IDS_FILE:-}" ]; then
    EXTRA+=(--specimen-ids-file "${SPECIMEN_IDS_FILE}")
fi
if [ -n "${ADAPTER_PATH:-}" ]; then
    EXTRA+=(--adapter-path "${ADAPTER_PATH}")
fi
if [ -n "${PROBE_BANK_DIR:-}" ]; then
    EXTRA+=(--probe-bank-dir "${PROBE_BANK_DIR}")
fi

echo "==> Phase 11.B CoT-SFT baseline"
echo "    Start         : ${START}"
echo "    Count         : ${COUNT}"
echo "    Output root   : ${OUT_ROOT}"
echo "    Base model    : ${BASE_MODEL}"
echo "    Quantize      : ${QUANTIZE}"
echo "    Max new tokens: ${MAX_NEW_TOKENS}"
echo "    Batch size    : ${BATCH_SIZE}"
echo "    GPU           : ${GPU}"
echo

CUDA_VISIBLE_DEVICES="${GPU}" uv run python scripts/run_baseline_cot.py \
    --start "${START}" \
    --count "${COUNT}" \
    --out "${OUT_ROOT}" \
    --base-model "${BASE_MODEL}" \
    "${EXTRA[@]}"
