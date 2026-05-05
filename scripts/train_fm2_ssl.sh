#!/usr/bin/env bash
#
# train_fm2_ssl.sh
#
# Phase 10: pretrain a parallel FM2 backbone with masked-RDF
# reconstruction. The supervised FM2 stays where it is; this script
# trains a new backbone with the same hyperparameters but a self-
# supervised objective. After it finishes, run scripts/run_fm2_probes.sh
# with USE_SSL=1 to A/B compare the SSL vs supervised representation.
#
# Usage:
#   bash scripts/train_fm2_ssl.sh
#
# Environment variables (optional):
#   TRAIN_SPLIT       (default: train_50k)
#   EPOCHS            (default: 20)
#   BATCH_SIZE        (default: 64)
#   LR                (default: 1.0e-4)
#   MASK_RATIO        (default: 0.30)
#   GPU               (default: 0)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

TRAIN_SPLIT="${TRAIN_SPLIT:-train_50k}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-1.0e-4}"
MASK_RATIO="${MASK_RATIO:-0.30}"
GPU="${GPU:-0}"

echo "==> FM2 SSL pretraining (masked-RDF reconstruction)"
echo "    Train split   : ${TRAIN_SPLIT}"
echo "    Epochs        : ${EPOCHS}"
echo "    Batch size    : ${BATCH_SIZE}"
echo "    LR            : ${LR}"
echo "    Mask ratio    : ${MASK_RATIO}"
echo "    GPU           : ${GPU}"
echo

CUDA_VISIBLE_DEVICES="${GPU}" uv run python scripts/train_fm2_ssl.py \
    --train-split "${TRAIN_SPLIT}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --mask-ratio "${MASK_RATIO}"
