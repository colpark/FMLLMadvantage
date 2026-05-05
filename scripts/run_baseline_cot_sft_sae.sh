#!/usr/bin/env bash
#
# run_baseline_cot_sft_sae.sh
#
# Phase 16 Stage 3: single-shot evaluation of the SAE-augmented
# CoT-SFT adapter on the held-out range. No verifier, no OHVD loop.
# Output to runs/holdout/cot_sft_sae/<run_id>/ so
# scripts/evaluate_baselines.sh auto-discovers it as a column.
#
# Usage:
#   bash scripts/run_baseline_cot_sft_sae.sh
#   SPECIMEN_IDS_FILE=runs/holdout_lock/ids.json \
#       bash scripts/run_baseline_cot_sft_sae.sh
#
# Environment variables (optional):
#   ADAPTER_PATH      (default: latest adapter under checkpoints/cot-sft-sae/)
#   PROBE_BANK_DIR    (default: latest under checkpoints/probes/)
#   SAE_DIR           (default: latest under checkpoints/sae/)
#   SAE_LABELS_PATH   (default: latest labels.json under runs/sae_labels/)
#   TOP_K_FEATURES    (default: 8)
#   BASE_MODEL        (default: Qwen/Qwen2.5-7B-Instruct)
#   QUANTIZE          (default: 4bit)
#   START / COUNT     (defaults: 0 / 200; override SPECIMEN_IDS_FILE)
#   GPU               (default: 0)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
QUANTIZE="${QUANTIZE:-4bit}"
START="${START:-0}"
COUNT="${COUNT:-200}"
TOP_K_FEATURES="${TOP_K_FEATURES:-8}"
GPU="${GPU:-0}"

EXTRA=()
EXTRA+=(--base-model "${BASE_MODEL}")
EXTRA+=(--quantize "${QUANTIZE}")
EXTRA+=(--start "${START}")
EXTRA+=(--count "${COUNT}")
EXTRA+=(--top-k-features "${TOP_K_FEATURES}")
if [ -n "${SPECIMEN_IDS_FILE:-}" ]; then
    EXTRA+=(--specimen-ids-file "${SPECIMEN_IDS_FILE}")
fi
if [ -n "${ADAPTER_PATH:-}" ]; then
    EXTRA+=(--adapter-path "${ADAPTER_PATH}")
fi
if [ -n "${PROBE_BANK_DIR:-}" ]; then
    EXTRA+=(--probe-bank-dir "${PROBE_BANK_DIR}")
fi
if [ -n "${SAE_DIR:-}" ]; then
    EXTRA+=(--sae-dir "${SAE_DIR}")
fi
if [ -n "${SAE_LABELS_PATH:-}" ]; then
    EXTRA+=(--sae-labels-path "${SAE_LABELS_PATH}")
fi

echo "==> Phase 16: cot_sft_sae single-shot baseline"
echo "    Base model      : ${BASE_MODEL}"
echo "    Quantize        : ${QUANTIZE}"
echo "    Specimens       : ${SPECIMEN_IDS_FILE:-[${START}, $((START + COUNT)))}"
echo "    Adapter         : ${ADAPTER_PATH:-(latest under checkpoints/cot-sft-sae/)}"
echo "    Probe bank      : ${PROBE_BANK_DIR:-(latest)}"
echo "    SAE             : ${SAE_DIR:-(latest)}"
echo "    SAE labels      : ${SAE_LABELS_PATH:-(latest)}"
echo "    Top-K features  : ${TOP_K_FEATURES}"
echo "    GPU             : ${GPU}"
echo

CUDA_VISIBLE_DEVICES="${GPU}" uv run python scripts/run_baseline_cot_sft_sae.py "${EXTRA[@]}"
