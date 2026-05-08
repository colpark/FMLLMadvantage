#!/usr/bin/env bash
#
# 08_train_sft.sh -- Stage 8 of the materials port.
# SFT-tune the LLM on the materials SAE-augmented CoT records.
#
# Thin wrapper over scripts/train_cot_sft.py. Defaults the dataset
# to the latest records.jsonl under runs/materials/cot_datasets_sae/
# and saves the LoRA adapter under
# checkpoints/materials/cot-sft-sae/<run_id>/adapter/.
#
# Multi-GPU: set NUM_GPUS=N (>=2) to launch via torchrun. Effective
# batch size = PER_DEVICE_BS * GRAD_ACCUM * NUM_GPUS.
#
# Usage:
#   bash scripts/materials/08_train_sft.sh                       # 1 GPU
#   NUM_GPUS=4 PER_DEVICE_BS=2 GRAD_ACCUM=4 MAX_SEQ=1024 \
#       bash scripts/materials/08_train_sft.sh                   # 4xH100
#
# Environment variables (optional):
#   DATASET           default: latest records.jsonl under runs/materials/cot_datasets_sae/
#   BASE_MODEL        default: Qwen/Qwen2.5-7B-Instruct
#   EPOCHS            default: 3
#   LR                default: 1.0e-4
#   LORA_R            default: 16
#   LORA_ALPHA        default: 32
#   PER_DEVICE_BS     default: 1
#   GRAD_ACCUM        default: 16
#   MAX_SEQ           default: 2048
#   NUM_GPUS          default: 1
#   GPUS              default: 0,1,2,3 when NUM_GPUS>=4, else 0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-1.0e-4}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
PER_DEVICE_BS="${PER_DEVICE_BS:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
# v2 default: 2560 tokens to fit rich-CoT records (Step 1/1b/2/3/4/Final
# w/ representative-specimen grounding). v1 records also fit.
MAX_SEQ="${MAX_SEQ:-2560}"
NUM_GPUS="${NUM_GPUS:-1}"
OUT_ROOT="${OUT_ROOT:-checkpoints/materials/cot-sft-sae}"

if [ -z "${GPUS:-}" ]; then
    if [ "${NUM_GPUS}" -ge 4 ]; then
        GPUS="0,1,2,3"
    elif [ "${NUM_GPUS}" -eq 2 ]; then
        GPUS="0,1"
    else
        GPUS="0"
    fi
fi

# Resolve dataset path
DATASET_ROOT="${DATASET_ROOT:-runs/materials/cot_datasets_sae}"
if [ -n "${DATASET:-}" ]; then
    RESOLVED_DATASET="${DATASET}"
else
    RESOLVED_DATASET="$(ls -td ${DATASET_ROOT}/*/records.jsonl 2>/dev/null | head -1 || true)"
    if [ -z "${RESOLVED_DATASET}" ]; then
        echo "ERROR: no records.jsonl under ${DATASET_ROOT}/." >&2
        echo "       Run scripts/materials/07_build_cot.sh first." >&2
        exit 2
    fi
fi

EFFECTIVE_BATCH=$((PER_DEVICE_BS * GRAD_ACCUM * NUM_GPUS))
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)-mat-cot-sft}"

echo "==> Materials port Stage 8: SFT on SAE-augmented CoT dataset"
echo "    Run id         : ${RUN_ID}"
echo "    Dataset        : ${RESOLVED_DATASET}"
echo "    Out root       : ${OUT_ROOT}"
echo "    Base model     : ${BASE_MODEL}"
echo "    Epochs         : ${EPOCHS}"
echo "    LR             : ${LR}"
echo "    LoRA r/alpha   : ${LORA_R}/${LORA_ALPHA}"
echo "    Per-device bs  : ${PER_DEVICE_BS}"
echo "    Grad accum     : ${GRAD_ACCUM}"
echo "    Max seq length : ${MAX_SEQ}"
echo "    Num GPUs       : ${NUM_GPUS}"
echo "    GPUs           : ${GPUS}"
echo "    Effective batch: ${EFFECTIVE_BATCH}"
echo

PY_ARGS=(
    --dataset "${RESOLVED_DATASET}"
    --out "${OUT_ROOT}"
    --base-model "${BASE_MODEL}"
    --epochs "${EPOCHS}"
    --learning-rate "${LR}"
    --lora-r "${LORA_R}"
    --lora-alpha "${LORA_ALPHA}"
    --per-device-batch-size "${PER_DEVICE_BS}"
    --grad-accum "${GRAD_ACCUM}"
    --max-seq-length "${MAX_SEQ}"
    --run-id "${RUN_ID}"
)

if [ "${NUM_GPUS}" -gt 1 ]; then
    CUDA_VISIBLE_DEVICES="${GPUS}" uv run torchrun --nproc_per_node="${NUM_GPUS}" --standalone scripts/train_cot_sft.py "${PY_ARGS[@]}"
else
    CUDA_VISIBLE_DEVICES="${GPUS}" uv run python scripts/train_cot_sft.py "${PY_ARGS[@]}"
fi
