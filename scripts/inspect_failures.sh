#!/usr/bin/env bash
#
# inspect_failures.sh
#
# Summarize FAIL trajectories in a baseline run. Defaults to the
# latest trajectories.jsonl under runs/baselines/full/.
#
# Usage:
#   bash scripts/inspect_failures.sh
#   bash scripts/inspect_failures.sh path/to/trajectories.jsonl
#   bash scripts/inspect_failures.sh runs/baselines/no_verifier/<id>/trajectories.jsonl

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

H5_PATH="${H5_PATH:-data/synthetic_lj_v1/specimens.h5}"

TRAJ="${1:-}"
if [ -z "${TRAJ}" ]; then
    TRAJ=$(ls -td runs/baselines/full/*/trajectories.jsonl 2>/dev/null | head -1 || true)
    if [ -z "${TRAJ}" ]; then
        echo "No trajectories.jsonl under runs/baselines/full/." >&2
        echo "Pass an explicit path, e.g.: bash scripts/inspect_failures.sh runs/baselines/no_verifier/<id>/trajectories.jsonl" >&2
        exit 1
    fi
fi

uv run python scripts/inspect_failures.py \
    --trajectories "${TRAJ}" \
    --h5-path "${H5_PATH}"
