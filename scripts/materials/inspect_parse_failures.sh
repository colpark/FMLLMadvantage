#!/usr/bin/env bash
#
# inspect_parse_failures.sh -- categorize Stage 9 parse failures.
#
# Reads the latest records.jsonl under runs/materials/holdout/cot_sft_sae/
# (override with RECORDS=...), classifies each parse failure as
# truncation / unclosed JSON / invalid syntax / etc., and reports
# representative examples plus an actionable diagnosis.
#
# Usage:
#   bash scripts/materials/inspect_parse_failures.sh
#   RECORDS=runs/materials/holdout/cot_sft_sae/<run>/records.jsonl \
#       bash scripts/materials/inspect_parse_failures.sh
#
# Environment variables (optional):
#   RECORDS    Path to records.jsonl (default: latest cot_sft_sae)
#   N_EXAMPLES Examples per category to surface (default: 3)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

N_EXAMPLES="${N_EXAMPLES:-3}"

EXTRA=()
EXTRA+=(--n-examples "${N_EXAMPLES}")
if [ -n "${RECORDS:-}" ]; then
    EXTRA+=(--records "${RECORDS}")
fi

uv run python scripts/materials/inspect_parse_failures.py "${EXTRA[@]}" "$@"
