#!/usr/bin/env bash
#
# 01_build_mp_h5.sh -- Stage 2 of the materials port.
# Pack the raw JSON cache into a single HDF5 file.
#
# Usage:
#   bash scripts/materials/01_build_mp_h5.sh
#
# Environment variables (optional):
#   RAW_DIR     default: data/materials_project_v1/raw
#   H5_PATH     default: data/materials_project_v1/specimens.h5
#   MAX_ATOMS   default: 80
#   MIN_ATOMS   default: 1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RAW_DIR="${RAW_DIR:-data/materials_project_v1/raw}"
H5_PATH="${H5_PATH:-data/materials_project_v1/specimens.h5}"
MAX_ATOMS="${MAX_ATOMS:-80}"
MIN_ATOMS="${MIN_ATOMS:-1}"

echo "==> Materials port Stage 2: build HDF5"
echo "    RAW_DIR    : ${RAW_DIR}"
echo "    H5_PATH    : ${H5_PATH}"
echo "    MAX_ATOMS  : ${MAX_ATOMS}"
echo "    MIN_ATOMS  : ${MIN_ATOMS}"
echo

uv run python scripts/materials/01_build_mp_h5.py --raw-dir "${RAW_DIR}" --h5-path "${H5_PATH}" --max-atoms "${MAX_ATOMS}" --min-atoms "${MIN_ATOMS}"
