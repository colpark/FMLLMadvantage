#!/usr/bin/env bash
#
# train_cot_sft_with_sae.sh
#
# Phase 16 Stage 2: SFT-tune the LLM on the SAE-augmented synthetic
# CoT dataset. Identical to scripts/train_cot_sft.sh but points at
# runs/cot_datasets_sae/<latest>/records.jsonl by default and saves
# the adapter under checkpoints/cot-sft-sae/<run_id>/adapter/.
#
# Usage:
#   bash scripts/train_cot_sft_with_sae.sh
#
# Environment variables (optional):
#   DATASET       (default: latest records.jsonl under runs/cot_datasets_sae/)
#   BASE_MODEL    (default: Qwen/Qwen2.5-7B-Instruct)
#   EPOCHS        (default: 3)
#   LR            (default: 1.0e-4)
#   LORA_R        (default: 16)
#   LORA_ALPHA    (default: 32)
#   GRAD_ACCUM    (default: 16)
#   GPU           (default: 0)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-1.0e-4}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
GPU="${GPU:-0}"
OUT_ROOT="checkpoints/cot-sft-sae"

# Resolve dataset (latest records.jsonl under runs/cot_datasets_sae/ unless
# overridden). The trainer also has a default but we pass explicitly so
# the banner is unambiguous.
if [ -n "${DATASET:-}" ]; then
    RESOLVED_DATASET="${DATASET}"
else
    RESOLVED_DATASET="$(ls -td runs/cot_datasets_sae/*/records.jsonl 2>/dev/null | head -1 || true)"
    if [ -z "${RESOLVED_DATASET}" ]; then
        echo "ERROR: no records.jsonl under runs/cot_datasets_sae/." >&2
        echo "       Run scripts/build_cot_dataset_with_sae.sh first." >&2
        exit 2
    fi
fi

echo "==> Phase 16 Stage 2: SFT on SAE-augmented CoT dataset"
echo "    Dataset    : ${RESOLVED_DATASET}"
echo "    Out root   : ${OUT_ROOT}"
echo "    Base model : ${BASE_MODEL}"
echo "    Epochs     : ${EPOCHS}"
echo "    LR         : ${LR}"
echo "    LoRA r/α   : ${LORA_R}/${LORA_ALPHA}"
echo "    Grad accum : ${GRAD_ACCUM}"
echo "    GPU        : ${GPU}"
echo

CUDA_VISIBLE_DEVICES="${GPU}" uv run python scripts/train_cot_sft.py \
    --dataset "${RESOLVED_DATASET}" \
    --out "${OUT_ROOT}" \
    --base-model "${BASE_MODEL}" \
    --epochs "${EPOCHS}" \
    --learning-rate "${LR}" \
    --lora-r "${LORA_R}" \
    --lora-alpha "${LORA_ALPHA}" \
    --grad-accum "${GRAD_ACCUM}"
