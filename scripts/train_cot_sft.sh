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
# Usage:
#   bash scripts/train_cot_sft.sh
#
# Environment variables (optional):
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

echo "==> Phase 11 Stage 2 SFT"
echo "    Base model : ${BASE_MODEL}"
echo "    Epochs     : ${EPOCHS}"
echo "    LR         : ${LR}"
echo "    LoRA r/α   : ${LORA_R}/${LORA_ALPHA}"
echo "    Grad accum : ${GRAD_ACCUM}"
echo "    GPU        : ${GPU}"
echo

CUDA_VISIBLE_DEVICES="${GPU}" uv run python scripts/train_cot_sft.py \
    --base-model "${BASE_MODEL}" \
    --epochs "${EPOCHS}" \
    --learning-rate "${LR}" \
    --lora-r "${LORA_R}" \
    --lora-alpha "${LORA_ALPHA}" \
    --grad-accum "${GRAD_ACCUM}"
