#!/usr/bin/env bash
#
# 02_lock_holdout.sh -- Stage 3 of the materials port.
# Pick 200 stratified held-out specimens; write splits.yaml + ids.json.
#
# Usage:
#   bash scripts/materials/02_lock_holdout.sh
#
# Environment variables (optional):
#   H5_PATH           default: data/materials_project_v1/specimens.h5
#   N_HOLDOUT         default: 200
#   STRATIFY_BY       default: crystal_system,is_metal
#   SEED              default: 0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

H5_PATH="${H5_PATH:-data/materials_project_v1/specimens.h5}"
N_HOLDOUT="${N_HOLDOUT:-200}"
STRATIFY_BY="${STRATIFY_BY:-crystal_system,is_metal}"
SEED="${SEED:-0}"

echo "==> Materials port Stage 3: lock holdout"
echo "    H5_PATH     : ${H5_PATH}"
echo "    N_HOLDOUT   : ${N_HOLDOUT}"
echo "    STRATIFY_BY : ${STRATIFY_BY}"
echo "    SEED        : ${SEED}"
echo

uv run python scripts/materials/02_lock_holdout.py --h5-path "${H5_PATH}" --n-holdout "${N_HOLDOUT}" --stratify-by "${STRATIFY_BY}" --seed "${SEED}"
