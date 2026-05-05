#!/usr/bin/env bash
#
# run_baseline_naked_vision.sh
#
# Fairer "naked LLM" baseline: gives the LLM the rasterized specimen
# image piped through a generic image-captioning model (BLIP base),
# with no FM tools, no probes, no SAE, and no verifier. Tests what
# a non-specialist AI system would actually achieve on this task --
# vs the original "zero-observation guess" naked baseline that hit
# 0.000 because it had no observations of any kind.
#
# Output writes to runs/holdout/naked_vision/<run_id>/ so
# scripts/evaluate_baselines.sh auto-discovers it as a new column.
#
# Usage:
#   bash scripts/run_baseline_naked_vision.sh
#   SPECIMEN_IDS_FILE=runs/holdout_lock/ids.json \
#       bash scripts/run_baseline_naked_vision.sh
#
# Environment variables (optional):
#   START               (default: 0)
#   COUNT               (default: 200)
#   SPECIMEN_IDS_FILE   (default: unset; overrides START/COUNT)
#   OUT_ROOT            (default: runs/holdout)
#   GPU                 (default: 0)
#   BASE_MODEL          (default: Qwen/Qwen2.5-7B-Instruct)
#   CAPTIONER_MODEL     (default: Salesforce/blip-image-captioning-base)
#   QUANTIZE            (default: 4bit)
#   MAX_NEW_TOKENS_COMMIT (default: 192)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

START="${START:-0}"
COUNT="${COUNT:-200}"
OUT_ROOT="${OUT_ROOT:-runs/holdout}"
GPU="${GPU:-0}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
CAPTIONER_MODEL="${CAPTIONER_MODEL:-Salesforce/blip-image-captioning-base}"
QUANTIZE="${QUANTIZE:-4bit}"
MAX_NEW_TOKENS_COMMIT="${MAX_NEW_TOKENS_COMMIT:-192}"

EXTRA=()
EXTRA+=(--quantize "${QUANTIZE}")
EXTRA+=(--max-new-tokens-commit "${MAX_NEW_TOKENS_COMMIT}")
if [ -n "${SPECIMEN_IDS_FILE:-}" ]; then
    EXTRA+=(--specimen-ids-file "${SPECIMEN_IDS_FILE}")
fi

echo "==> Naked-vision baseline (image + generic VLM caption)"
echo "    Captioner   : ${CAPTIONER_MODEL}"
echo "    Base model  : ${BASE_MODEL}"
echo "    Specimens   : [${START}, $((START + COUNT)))"
echo "    Quantize    : ${QUANTIZE}"
echo "    GPU         : ${GPU}"
echo

CUDA_VISIBLE_DEVICES="${GPU}" uv run python scripts/run_baseline_naked_vision.py \
    --start "${START}" \
    --count "${COUNT}" \
    --out "${OUT_ROOT}" \
    --base-model "${BASE_MODEL}" \
    --captioner-model "${CAPTIONER_MODEL}" \
    "${EXTRA[@]}"
