#!/usr/bin/env bash
#
# rescore_records.sh -- rescore an old records JSONL against fresh HDF5.
#
# Use after rebuilding the HDF5 (e.g. fixed a field-name bug) when
# you have an old JSONL whose ground_truth values are stale. The
# rescored JSONL preserves the original LLM claims; only the
# ground_truth and correctness fields are refreshed.
#
# Usage:
#   bash scripts/materials/rescore_records.sh --input <jsonl>
#   INPUT=runs/materials/holdout/cot_sft_sae/<run>/records.jsonl \
#       bash scripts/materials/rescore_records.sh
#
# Environment variables (optional):
#   INPUT     Path to input JSONL. Required if no --input flag.
#   OUTPUT    Path to output JSONL. Default: <input>.rescored.jsonl.
#   H5_PATH   Default: data/materials_project_v1/specimens.h5

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

H5_PATH="${H5_PATH:-data/materials_project_v1/specimens.h5}"

EXTRA=()
EXTRA+=(--h5-path "${H5_PATH}")
if [ -n "${INPUT:-}" ]; then
    EXTRA+=(--input "${INPUT}")
fi
if [ -n "${OUTPUT:-}" ]; then
    EXTRA+=(--output "${OUTPUT}")
fi

uv run python scripts/materials/rescore_records.py "${EXTRA[@]}" "$@"
