#!/usr/bin/env bash
#
# run_evaluation.sh
#
# Run the eight world-model evaluation tests. Default mode picks up
# the latest trajectories.jsonl under runs/trajectories/. Ablation
# mode runs over a lattice of trajectory files supplied via repeated
# --ablation flags.
#
# Usage:
#   bash scripts/run_evaluation.sh                  # latest trajectories
#   bash scripts/run_evaluation.sh path/to/x.jsonl  # explicit path
#   bash scripts/run_evaluation.sh ablation V0=path0 V1=path1 ...
#
# Environment variables (optional):
#   H5_PATH       (default: data/synthetic_lj_v1/specimens.h5)
#   OUT_ROOT      (default: runs/eval)
#   FAIL_ON_ERROR (default: 0; set to 1 to exit non-zero on FAIL)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

H5_PATH="${H5_PATH:-data/synthetic_lj_v1/specimens.h5}"
OUT_ROOT="${OUT_ROOT:-runs/eval}"
FAIL_ON_ERROR="${FAIL_ON_ERROR:-0}"

EXTRA=()
if [ "${FAIL_ON_ERROR}" -eq 1 ]; then
    EXTRA+=(--fail-on-error)
fi

MODE="${1:-}"
if [ "${MODE}" = "ablation" ]; then
    shift
    if [ "$#" -eq 0 ]; then
        echo "Usage: bash scripts/run_evaluation.sh ablation V0=path0 V1=path1 ..." >&2
        exit 1
    fi
    ABLATION_ARGS=()
    for spec in "$@"; do
        ABLATION_ARGS+=(--ablation "${spec}")
    done
    echo "==> Evaluation harness (ablation mode)"
    echo "    H5 path     : ${H5_PATH}"
    echo "    Output root : ${OUT_ROOT}"
    echo "    Ablations   : $*"
    echo
    uv run python scripts/run_evaluation.py \
        --h5-path "${H5_PATH}" \
        --out "${OUT_ROOT}" \
        "${ABLATION_ARGS[@]}" \
        "${EXTRA[@]}"
    exit $?
fi

if [ -n "${MODE}" ]; then
    TRAJ="${MODE}"
else
    TRAJ=$(ls -td runs/trajectories/*/trajectories.jsonl 2>/dev/null | head -1 || true)
    if [ -z "${TRAJ}" ]; then
        echo "No trajectories.jsonl found under runs/trajectories/." >&2
        echo "Run scripts/collect_trajectories.sh first." >&2
        exit 1
    fi
fi

echo "==> Evaluation harness"
echo "    Trajectories : ${TRAJ}"
echo "    H5 path      : ${H5_PATH}"
echo "    Output root  : ${OUT_ROOT}"
echo

uv run python scripts/run_evaluation.py \
    --trajectories "${TRAJ}" \
    --h5-path "${H5_PATH}" \
    --out "${OUT_ROOT}" \
    "${EXTRA[@]}"
