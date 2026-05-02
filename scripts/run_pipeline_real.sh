#!/usr/bin/env bash
#
# run_pipeline_real.sh
#
# Run the OHVD loop with a real chat LLM (default: Llama 3.1 8B
# Instruct via transformers). The first invocation downloads ~16 GB
# of weights and caches them; subsequent runs reuse the cache.
#
# Usage:
#   bash scripts/run_pipeline_real.sh                    # specimen 42
#   bash scripts/run_pipeline_real.sh 7                  # specimen 7
#   bash scripts/run_pipeline_real.sh 7 train_30k        # different scale
#
# Environment variables (optional):
#   CONFIG          (default: configs/default.yaml)
#   H5_PATH         (default: data/synthetic_lj_v1/specimens.h5)
#   SPLITS_PATH     (default: data/synthetic_lj_v1/splits.yaml)
#   CHECKPOINT_ROOT (default: checkpoints)
#   LITERATURE_DB   (default: data/literature/clusters.json)
#   LLM_MODEL       (default: meta-llama/Llama-3.1-8B-Instruct)
#   LLM_TEMPERATURE (default: 0.2)
#   STEP_BUDGET     (default: 16)
#   ABLATION        (default: V4)
#   GPU             (default: 0)
#
# Authentication:
#   Llama 3.1 is gated on Hugging Face. Run `huggingface-cli login`
#   once before the first invocation. Alternatively swap LLM_MODEL
#   to a non-gated chat model.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

SPECIMEN_ID="${1:-42}"
TRAIN_SPLIT="${2:-train_50k}"

CONFIG="${CONFIG:-configs/default.yaml}"
H5_PATH="${H5_PATH:-data/synthetic_lj_v1/specimens.h5}"
SPLITS_PATH="${SPLITS_PATH:-data/synthetic_lj_v1/splits.yaml}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints}"
LITERATURE_DB="${LITERATURE_DB:-data/literature/clusters.json}"
LLM_MODEL="${LLM_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
LLM_TEMPERATURE="${LLM_TEMPERATURE:-0.2}"
STEP_BUDGET="${STEP_BUDGET:-16}"
ABLATION="${ABLATION:-V4}"
GPU="${GPU:-0}"

echo "==> Pipeline A (real LLM)"
echo "    Specimen ID    : ${SPECIMEN_ID}"
echo "    Train split    : ${TRAIN_SPLIT}"
echo "    LLM model      : ${LLM_MODEL}"
echo "    Temperature    : ${LLM_TEMPERATURE}"
echo "    Step budget    : ${STEP_BUDGET}"
echo "    Ablation       : ${ABLATION}"
echo "    GPU            : ${GPU}"
echo

CUDA_VISIBLE_DEVICES="${GPU}" uv run python scripts/run_pipeline.py \
    --specimen-id "${SPECIMEN_ID}" \
    --config "${CONFIG}" \
    --h5-path "${H5_PATH}" \
    --splits-path "${SPLITS_PATH}" \
    --checkpoint-root "${CHECKPOINT_ROOT}" \
    --train-split "${TRAIN_SPLIT}" \
    --literature-db "${LITERATURE_DB}" \
    --llm-model "${LLM_MODEL}" \
    --llm-temperature "${LLM_TEMPERATURE}" \
    --step-budget "${STEP_BUDGET}" \
    --ablation "${ABLATION}"

LATEST=$(ls -td runs/*pipeline-a-${SPECIMEN_ID}* 2>/dev/null | head -1 || true)
if [ -n "${LATEST}" ]; then
    echo
    echo "==> Latest trajectory: ${LATEST}"
    bash "${SCRIPT_DIR}/inspect_trajectory.sh" "${LATEST}"
fi
