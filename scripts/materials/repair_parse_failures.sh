#!/usr/bin/env bash
#
# repair_parse_failures.sh -- re-parse raw_text in an existing
# Stage 9 JSONL using the lenient Final-commit parser. Writes a
# repaired JSONL alongside the input. Skips records that already
# parsed; recomputes is_correct / per_axis_correct for newly
# recovered claims using the current HDF5 ground truth.
#
# Usage:
#   bash scripts/materials/repair_parse_failures.sh
#   INPUT=runs/materials/holdout/cot_sft_sae/<run>/records.jsonl \
#       bash scripts/materials/repair_parse_failures.sh
#
# Environment variables (optional):
#   INPUT     Path to records.jsonl (default: latest cot_sft_sae)
#   OUTPUT    Path to repaired JSONL (default: <input>.repaired.jsonl)
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

uv run python scripts/materials/repair_parse_failures.py "${EXTRA[@]}" "$@"
