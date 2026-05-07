#!/usr/bin/env bash
#
# 10_benchmark_chgnet.sh -- Stage 4 sanity check.
# Forwards CHGNet on the held-out 200 and compares the energy MAE
# against published reference numbers (~0.030 eV/atom).
#
# This validates that the materials data pipeline + CHGNet
# wrapper are wired correctly before training probes / SAE / SFT
# on top of the embeddings. If energy MAE is materially above the
# published target, fix the data pipeline before proceeding.
#
# Usage:
#   bash scripts/materials/10_benchmark_chgnet.sh
#
# Environment variables (optional):
#   H5_PATH               default: data/materials_project_v1/specimens.h5
#   HOLDOUT_IDS_PATH      default: data/materials_project_v1/holdout_lock/ids.json
#   CHGNET_MODEL_NAME     default: 0.3.0
#   GPU                   default: 0
#   MAX_ATOMS             default: 80

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

H5_PATH="${H5_PATH:-data/materials_project_v1/specimens.h5}"
HOLDOUT_IDS_PATH="${HOLDOUT_IDS_PATH:-data/materials_project_v1/holdout_lock/ids.json}"
CHGNET_MODEL_NAME="${CHGNET_MODEL_NAME:-0.3.0}"
GPU="${GPU:-0}"
MAX_ATOMS="${MAX_ATOMS:-80}"

echo "==> Materials port Stage 4 sanity: CHGNet benchmark"
echo "    H5_PATH               : ${H5_PATH}"
echo "    HOLDOUT_IDS_PATH      : ${HOLDOUT_IDS_PATH}"
echo "    CHGNET_MODEL_NAME     : ${CHGNET_MODEL_NAME}"
echo "    GPU                   : ${GPU}"
echo "    MAX_ATOMS             : ${MAX_ATOMS}"
echo

CUDA_VISIBLE_DEVICES="${GPU}" uv run python scripts/materials/10_benchmark_chgnet.py --h5-path "${H5_PATH}" --holdout-ids-path "${HOLDOUT_IDS_PATH}" --chgnet-model-name "${CHGNET_MODEL_NAME}" --max-atoms "${MAX_ATOMS}"
