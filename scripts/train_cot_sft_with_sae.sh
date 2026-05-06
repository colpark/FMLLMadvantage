#!/usr/bin/env bash
#
# train_cot_sft_with_sae.sh
#
# Phase 16 Stage 2: SFT-tune the LLM on the SAE-augmented synthetic
# CoT dataset. Identical to scripts/train_cot_sft.sh but points at
# runs/cot_datasets_sae/<latest>/records.jsonl by default and saves
# the adapter under checkpoints/cot-sft-sae/<run_id>/adapter/.
#
# Multi-GPU support: set NUM_GPUS=N (>=2) to launch via torchrun
# with N data-parallel ranks. The trainer detects LOCAL_RANK /
# WORLD_SIZE and uses per-rank device_map; effective batch size is
# PER_DEVICE_BS * GRAD_ACCUM * NUM_GPUS.
#
# Usage:
#   bash scripts/train_cot_sft_with_sae.sh                       # 1 GPU, slow
#   NUM_GPUS=4 PER_DEVICE_BS=2 GRAD_ACCUM=4 MAX_SEQ=1024 \
#       bash scripts/train_cot_sft_with_sae.sh                   # ~6-10x faster
#
# Environment variables (optional):
#   DATASET           default: latest records.jsonl under runs/cot_datasets_sae/
#   BASE_MODEL        default: Qwen/Qwen2.5-7B-Instruct
#   EPOCHS            default: 3
#   LR                default: 1.0e-4
#   LORA_R            default: 16
#   LORA_ALPHA        default: 32
#   PER_DEVICE_BS     default: 1   (try 2-4 on H100 80GB)
#   GRAD_ACCUM        default: 16  (drop to 4 when scaling per-device)
#   MAX_SEQ           default: 2048 (drop to 1024; CoT records are <800 tokens)
#   NUM_GPUS          default: 1   (set to 2/4/... for DDP)
#   GPUS              default: 0,1,2,3 when NUM_GPUS>1, else 0
#                     (override to e.g. "0,2" to skip a GPU)
#
# Recommended fast configuration on a 4xH100 host:
#   NUM_GPUS=4 PER_DEVICE_BS=2 GRAD_ACCUM=4 MAX_SEQ=1024 \
#       bash scripts/train_cot_sft_with_sae.sh
#
#   That gives effective batch = 2 * 4 * 4 = 32 (2x current) and
#   ~6-10x throughput improvement from 4 GPUs + larger per-device
#   batch + halved sequence length.

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
OUT_ROOT="checkpoints/cot-sft-sae"

# Default GPUS list based on NUM_GPUS unless explicitly set.
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

EFFECTIVE_BATCH=$((PER_DEVICE_BS * GRAD_ACCUM * NUM_GPUS))

# Generate the run_id ONCE in the shell so every torchrun rank uses
# the same value -- avoids having to coordinate via NCCL inside the
# python process before the device is bound. Override with RUN_ID env
# var if you want a specific id (useful for resumable runs).
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)-cot-sft-stage2}"

echo "==> Phase 16 Stage 2: SFT on SAE-augmented CoT dataset"
echo "    Run id         : ${RUN_ID}"
echo "    Dataset        : ${RESOLVED_DATASET}"
echo "    Out root       : ${OUT_ROOT}"
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
    # Multi-GPU: launch via torchrun. Each rank loads its own model
    # copy on its own GPU; gradients synchronized via DDP.
    CUDA_VISIBLE_DEVICES="${GPUS}" uv run torchrun \
        --nproc_per_node="${NUM_GPUS}" --standalone \
        scripts/train_cot_sft.py "${PY_ARGS[@]}"
else
    CUDA_VISIBLE_DEVICES="${GPUS}" uv run python \
        scripts/train_cot_sft.py "${PY_ARGS[@]}"
fi
