#!/usr/bin/env bash
#
# inspect_cot_records.sh -- check whether CoT-record ground_truth
# matches the current HDF5. Auto-discovers the latest records.jsonl
# under runs/materials/cot_datasets_sae/ and the default HDF5.
#
# Usage:
#   bash scripts/materials/inspect_cot_records.sh
#   RECORDS=runs/materials/cot_datasets_sae/<run>/records.jsonl \
#       bash scripts/materials/inspect_cot_records.sh
#
# Environment variables (optional):
#   RECORDS    Path to records.jsonl (default: latest under runs/materials/cot_datasets_sae/)
#   H5_PATH    Default: data/materials_project_v1/specimens.h5
#   N_SHOW     Number of disagreement examples per drifted axis (default: 10)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

H5_PATH="${H5_PATH:-data/materials_project_v1/specimens.h5}"
N_SHOW="${N_SHOW:-10}"

EXTRA=()
EXTRA+=(--h5-path "${H5_PATH}")
EXTRA+=(--n-show "${N_SHOW}")
if [ -n "${RECORDS:-}" ]; then
    EXTRA+=(--records-path "${RECORDS}")
fi

uv run python scripts/materials/inspect_cot_records.py "${EXTRA[@]}" "$@"
