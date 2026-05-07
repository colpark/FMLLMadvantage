#!/usr/bin/env bash
#
# 03_encode.sh -- Stage 3 of the materials port.
# Forward CHGNet over all train specimens, cache pooled embeddings.
#
# Usage:
#   bash scripts/materials/03_encode.sh
#   INCLUDE_HOLDOUT=1 bash scripts/materials/03_encode.sh
#
# Environment variables (optional):
#   H5_PATH               default: data/materials_project_v1/specimens.h5
#   SPLITS_PATH           default: data/materials_project_v1/splits.yaml
#   OUT                   default: runs/materials/embeddings
#   INCLUDE_HOLDOUT       default: 0 (1 to also encode the held-out 200)
#   CHGNET_MODEL_NAME     default: 0.3.0
#   MAX_ATOMS             default: 80
#   N_MAX                 default: 0 (no cap)
#   GPU                   default: 0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

H5_PATH="${H5_PATH:-data/materials_project_v1/specimens.h5}"
SPLITS_PATH="${SPLITS_PATH:-data/materials_project_v1/splits.yaml}"
OUT="${OUT:-runs/materials/embeddings}"
INCLUDE_HOLDOUT="${INCLUDE_HOLDOUT:-0}"
CHGNET_MODEL_NAME="${CHGNET_MODEL_NAME:-0.3.0}"
MAX_ATOMS="${MAX_ATOMS:-80}"
N_MAX="${N_MAX:-0}"
GPU="${GPU:-0}"

EXTRA=()
EXTRA+=(--h5-path "${H5_PATH}")
EXTRA+=(--splits-path "${SPLITS_PATH}")
EXTRA+=(--out "${OUT}")
EXTRA+=(--chgnet-model-name "${CHGNET_MODEL_NAME}")
EXTRA+=(--max-atoms "${MAX_ATOMS}")
EXTRA+=(--n-max "${N_MAX}")
if [ "${INCLUDE_HOLDOUT}" = "1" ]; then
    EXTRA+=(--include-holdout)
fi

echo "==> Materials port Stage 3: encode"
echo "    H5_PATH               : ${H5_PATH}"
echo "    SPLITS_PATH           : ${SPLITS_PATH}"
echo "    OUT                   : ${OUT}"
echo "    INCLUDE_HOLDOUT       : ${INCLUDE_HOLDOUT}"
echo "    CHGNET_MODEL_NAME     : ${CHGNET_MODEL_NAME}"
echo "    MAX_ATOMS             : ${MAX_ATOMS}"
echo "    N_MAX                 : ${N_MAX}"
echo "    GPU                   : ${GPU}"
echo

CUDA_VISIBLE_DEVICES="${GPU}" uv run python scripts/materials/03_encode.py "${EXTRA[@]}"
