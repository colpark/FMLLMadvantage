#!/usr/bin/env bash
#
# train_fm2_connector.sh
#
# Phase 9.A Stage 1: alignment training for the FM2 Q-Former
# connector. Freezes FM2 and the orchestrator LLM; trains only the
# connector to project FM2 hidden states into LLM-readable tokens
# such that the LLM can produce templated specimen descriptions.
#
# Run this AFTER scripts/run_fm2_probes.sh confirms the FM2
# representation has task-extra signal (Phase 9.0). If the probes
# come back near chance, skip this and consider self-supervised
# pretraining instead.
#
# Usage:
#   bash scripts/train_fm2_connector.sh
#
# Environment variables (optional):
#   N_SPECIMENS       (default: 2000)
#   EPOCHS            (default: 3)
#   BATCH_SIZE        (default: 8)
#   LR                (default: 1.0e-4)
#   GRAD_ACCUM        (default: 1)
#   N_QUERY           (default: 32)
#   GPU               (default: 0)
#   LLM_MODEL         (default: Qwen/Qwen2.5-7B-Instruct)
#   SHUFFLE_FEATURES  (default: 0; set to 1 to run the diagnostic
#                      ablation where FM features are shuffled within
#                      each batch. Output goes to a separate run-id
#                      with -shuffled in the slug.)
#   USE_SSL           (default: 0; set to 1 to train the connector
#                      on top of the Phase 10 SSL FM2 backbone.
#                      Requires that scripts/train_fm2_ssl.sh has run
#                      and produced a checkpoint under
#                      checkpoints/fm2_rdf_ssl/. Output goes to a
#                      separate run-id with -ssl in the slug.)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

N_SPECIMENS="${N_SPECIMENS:-2000}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1.0e-4}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
N_QUERY="${N_QUERY:-32}"
GPU="${GPU:-0}"
LLM_MODEL="${LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
SHUFFLE_FEATURES="${SHUFFLE_FEATURES:-0}"
USE_SSL="${USE_SSL:-0}"

EXTRA=()
if [ "${SHUFFLE_FEATURES}" -eq 1 ]; then
    EXTRA+=(--shuffle-features)
fi
if [ "${USE_SSL}" -eq 1 ]; then
    EXTRA+=(--use-ssl)
fi

echo "==> FM2 connector Stage 1 training"
echo "    LLM             : ${LLM_MODEL}"
echo "    Specimens       : ${N_SPECIMENS}"
echo "    Epochs          : ${EPOCHS}"
echo "    Batch size      : ${BATCH_SIZE}"
echo "    LR              : ${LR}"
echo "    Grad accum      : ${GRAD_ACCUM}"
echo "    n_query         : ${N_QUERY}"
echo "    GPU             : ${GPU}"
echo "    Shuffle features: ${SHUFFLE_FEATURES}"
echo "    Use SSL backbone: ${USE_SSL}"
echo

CUDA_VISIBLE_DEVICES="${GPU}" uv run python scripts/train_fm2_connector.py \
    --n-specimens "${N_SPECIMENS}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --grad-accum "${GRAD_ACCUM}" \
    --n-query "${N_QUERY}" \
    --llm-model "${LLM_MODEL}" \
    "${EXTRA[@]}"
