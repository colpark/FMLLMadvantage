#!/usr/bin/env bash
#
# 00_download_mp.sh -- Stage 1 of the materials port.
# Download Materials Project structures + properties.
#
# Prerequisite: export MP_API_KEY=<key> from
# https://next-gen.materialsproject.org/api
#
# Usage:
#   bash scripts/materials/00_download_mp.sh
#
# Environment variables (optional):
#   N_MAX               default: 50000
#   E_ABOVE_HULL_MAX    default: 0.5
#   RAW_DIR             default: data/materials_project_v1/raw
#   BATCH               default: 500

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [ -z "${MP_API_KEY:-}" ]; then
    echo "ERROR: MP_API_KEY env var not set." >&2
    echo "       Get a free key: https://next-gen.materialsproject.org/api" >&2
    exit 2
fi

N_MAX="${N_MAX:-50000}"
E_ABOVE_HULL_MAX="${E_ABOVE_HULL_MAX:-0.5}"
RAW_DIR="${RAW_DIR:-data/materials_project_v1/raw}"
BATCH="${BATCH:-500}"

echo "==> Materials port Stage 1: download Materials Project"
echo "    N_MAX            : ${N_MAX}"
echo "    E_ABOVE_HULL_MAX : ${E_ABOVE_HULL_MAX}"
echo "    RAW_DIR          : ${RAW_DIR}"
echo "    BATCH            : ${BATCH}"
echo

uv run python scripts/materials/00_download_mp.py --raw-dir "${RAW_DIR}" --n-max "${N_MAX}" --e-above-hull-max "${E_ABOVE_HULL_MAX}" --batch "${BATCH}"
