#!/usr/bin/env bash
#
# compare_baselines.sh
#
# Side-by-side comparison of baseline evaluation reports.
#
# Usage:
#   bash scripts/compare_baselines.sh \
#       naked=runs/eval/<id-naked>/report.yaml \
#       no_verifier=runs/eval/<id-nv>/report.yaml \
#       full=runs/eval/<id-full>/report.yaml
#
# Or pass paths only and let the script use the parent directory name
# as the column label:
#
#   bash scripts/compare_baselines.sh \
#       runs/eval/<id-naked>/report.yaml \
#       runs/eval/<id-full>/report.yaml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [ "$#" -lt 2 ]; then
    echo "Usage: bash scripts/compare_baselines.sh KEY=PATH [KEY=PATH ...]" >&2
    exit 1
fi

REPORT_ARGS=()
for spec in "$@"; do
    REPORT_ARGS+=(--report "${spec}")
done

uv run python scripts/compare_baselines.py "${REPORT_ARGS[@]}"
