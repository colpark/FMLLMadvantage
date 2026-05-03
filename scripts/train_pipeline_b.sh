#!/usr/bin/env bash
#
# train_pipeline_b.sh
#
# Train Pipeline B (SFT, DPO, or GRPO) on collected trajectories.
# The script picks the latest trajectories.jsonl under
# runs/trajectories/ unless overridden, then dispatches to
# scripts/train_pipeline_b.py with the chosen mode.
#
# Usage:
#   bash scripts/train_pipeline_b.sh sft
#   bash scripts/train_pipeline_b.sh dpo
#   bash scripts/train_pipeline_b.sh grpo
#
# Environment variables (optional):
#   TRAJECTORIES    (default: latest trajectories.jsonl under runs/trajectories/)
#   OUT_ROOT        (default: checkpoints/pipeline-b)
#   BASE_MODEL      (default: meta-llama/Llama-3.1-8B-Instruct)
#   LEARNING_RATE   (default: trainer-specific, see scripts/train_pipeline_b.py)
#   EPOCHS          (default: trainer-specific)
#   LORA_R          (default: 16)
#   LORA_ALPHA      (default: 32)
#   GPU             (default: 0)
#   ACCELERATE      (default: 0)   # set to 1 to launch via accelerate launch

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODE="${1:-}"
if [ -z "${MODE}" ] || ! [[ "${MODE}" =~ ^(sft|dpo|grpo)$ ]]; then
    echo "Usage: bash scripts/train_pipeline_b.sh {sft|dpo|grpo}" >&2
    exit 1
fi

if [ -z "${TRAJECTORIES:-}" ]; then
    LATEST=$(ls -td runs/trajectories/*/trajectories.jsonl 2>/dev/null | head -1 || true)
    if [ -z "${LATEST}" ]; then
        echo "No trajectories.jsonl found under runs/trajectories/." >&2
        echo "Run scripts/collect_trajectories.sh first." >&2
        exit 1
    fi
    TRAJECTORIES="${LATEST}"
fi

OUT_ROOT="${OUT_ROOT:-checkpoints/pipeline-b}"
BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
GPU="${GPU:-0}"
ACCELERATE="${ACCELERATE:-0}"

EXTRA=()
if [ -n "${LEARNING_RATE:-}" ]; then
    EXTRA+=(--learning-rate "${LEARNING_RATE}")
fi
if [ -n "${EPOCHS:-}" ]; then
    EXTRA+=(--epochs "${EPOCHS}")
fi
# SFT-only: pass --include-caveat to widen the training set with
# CAVEAT trajectories.
if [ "${INCLUDE_CAVEAT:-0}" -eq 1 ]; then
    EXTRA+=(--include-caveat)
fi

echo "==> Pipeline B trainer"
echo "    Mode            : ${MODE}"
echo "    Trajectories    : ${TRAJECTORIES}"
echo "    Output root     : ${OUT_ROOT}"
echo "    Base model      : ${BASE_MODEL}"
echo "    LoRA            : r=${LORA_R}, alpha=${LORA_ALPHA}"
echo "    GPU             : ${GPU}"
echo "    Accelerate      : ${ACCELERATE}"
echo "    Include CAVEAT  : ${INCLUDE_CAVEAT:-0}"
echo

if [ "${ACCELERATE}" -eq 1 ]; then
    CUDA_VISIBLE_DEVICES="${GPU}" uv run accelerate launch scripts/train_pipeline_b.py \
        --mode "${MODE}" \
        --trajectories "${TRAJECTORIES}" \
        --out "${OUT_ROOT}" \
        --base-model "${BASE_MODEL}" \
        --lora-r "${LORA_R}" \
        --lora-alpha "${LORA_ALPHA}" \
        "${EXTRA[@]}"
else
    CUDA_VISIBLE_DEVICES="${GPU}" uv run python scripts/train_pipeline_b.py \
        --mode "${MODE}" \
        --trajectories "${TRAJECTORIES}" \
        --out "${OUT_ROOT}" \
        --base-model "${BASE_MODEL}" \
        --lora-r "${LORA_R}" \
        --lora-alpha "${LORA_ALPHA}" \
        "${EXTRA[@]}"
fi
