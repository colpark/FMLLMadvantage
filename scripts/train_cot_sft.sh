#!/usr/bin/env bash
#
# train_cot_sft.sh
#
# Phase 11 Stage 2: SFT-tune the orchestrator LLM on synthetic
# (probe outputs -> CoT -> ground-truth claim) records produced by
# scripts/build_cot_dataset.sh.
#
# Saves a LoRA adapter under checkpoints/cot-sft/<run_id>/adapter/.
# Use --adapter-path checkpoints/cot-sft/<run_id>/adapter when calling
# scripts/run_baseline.sh full to evaluate the trained behavior.
#
# Multi-GPU support: set NUM_GPUS=N (>=2) to launch via torchrun
# with N data-parallel ranks. Effective batch =
# PER_DEVICE_BS * GRAD_ACCUM * NUM_GPUS.
#
# Usage:
#   bash scripts/train_cot_sft.sh                                # 1 GPU
#   NUM_GPUS=4 PER_DEVICE_BS=2 GRAD_ACCUM=4 MAX_SEQ=1024 \
#       bash scripts/train_cot_sft.sh                            # 4 GPUs
#
# Environment variables (optional):
#   BASE_MODEL    (default: Qwen/Qwen2.5-7B-Instruct)
#   EPOCHS        (default: 3)
#   LR            (default: 1.0e-4)
#   LORA_R        (default: 16)
#   LORA_ALPHA    (default: 32)
#   PER_DEVICE_BS (default: 1; try 2-4 on H100 80GB)
#   GRAD_ACCUM    (default: 16; drop to 4 when scaling per-device)
#   MAX_SEQ       (default: 2048; drop to 1024 for CoT records)
#   NUM_GPUS      (default: 1; set to 2/4/... for DDP)
#   GPUS          (default: 0,1,2,3 when NUM_GPUS>=4 etc; override
#                  to a comma-separated list to pin a specific subset)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-1.0e-4}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
PER_DEVICE_BS="${PER_DEVICE_BS:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
MAX_SEQ="${MAX_SEQ:-2048}"
NUM_GPUS="${NUM_GPUS:-1}"

if [ -z "${GPUS:-}" ]; then
    if [ "${NUM_GPUS}" -ge 4 ]; then
        GPUS="0,1,2,3"
    elif [ "${NUM_GPUS}" -eq 2 ]; then
        GPUS="0,1"
    else
        GPUS="0"
    fi
fi

EFFECTIVE_BATCH=$((PER_DEVICE_BS * GRAD_ACCUM * NUM_GPUS))

# Generate the run_id ONCE in the shell so every torchrun rank uses
# the same value (avoids NCCL-pre-init coordination).
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)-cot-sft-stage2}"

echo "==> Phase 11 Stage 2 SFT"
echo "    Run id         : ${RUN_ID}"
echo "    Base model     : ${BASE_MODEL}"
echo "    Epochs         : ${EPOCHS}"
echo "    LR             : ${LR}"
echo "    LoRA r/α       : ${LORA_R}/${LORA_ALPHA}"
echo "    Per-device bs  : ${PER_DEVICE_BS}"
echo "    Grad accum     : ${GRAD_ACCUM}"
echo "    Max seq length : ${MAX_SEQ}"
echo "    Num GPUs       : ${NUM_GPUS}"
echo "    GPUs           : ${GPUS}"
echo "    Effective batch: ${EFFECTIVE_BATCH}"
echo

PY_ARGS=(
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
    CUDA_VISIBLE_DEVICES="${GPUS}" uv run torchrun \
        --nproc_per_node="${NUM_GPUS}" --standalone \
        scripts/train_cot_sft.py "${PY_ARGS[@]}"
else
    CUDA_VISIBLE_DEVICES="${GPUS}" uv run python \
        scripts/train_cot_sft.py "${PY_ARGS[@]}"
fi
