#!/usr/bin/env bash
#
# 09c_score_hybrid.sh -- hybrid LLM+probe scoring (no inference).
#
# Reads the latest cot_sft_sae records.jsonl (prefers .repaired.jsonl
# when available), builds a hybrid claim per specimen using the LLM
# for regression axes (formation_energy / e_above_hull) and the probe
# rules for classification axes (is_stable / band_gap_class /
# space_group), then scores joint + per-axis accuracy.
#
# Quantifies "what if we use the LLM only where it adds value?".
#
# Usage:
#   bash scripts/materials/09c_score_hybrid.sh
#   INPUT=runs/materials/holdout/cot_sft_sae/<run>/records.repaired.jsonl \
#       bash scripts/materials/09c_score_hybrid.sh
#
# Environment variables (optional):
#   INPUT     records.jsonl to read (default: latest cot_sft_sae,
#             prefers .repaired.jsonl variant)
#   H5_PATH   default: data/materials_project_v1/specimens.h5

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

H5_PATH="${H5_PATH:-data/materials_project_v1/specimens.h5}"
INPUT_SUBDIR="${INPUT_SUBDIR:-cot_sft_sae}"
OUT_SUBDIR="${OUT_SUBDIR:-hybrid}"

EXTRA=()
EXTRA+=(--h5-path "${H5_PATH}")
EXTRA+=(--input-subdir "${INPUT_SUBDIR}")
EXTRA+=(--out-subdir "${OUT_SUBDIR}")
if [ -n "${INPUT:-}" ]; then
    EXTRA+=(--input "${INPUT}")
fi

uv run python scripts/materials/09c_score_hybrid.py "${EXTRA[@]}" "$@"
