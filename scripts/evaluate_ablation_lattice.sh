#!/usr/bin/env bash
#
# evaluate_ablation_lattice.sh
#
# Evaluate the V0..V4 ablation lattice produced by
# run_ablation_lattice.sh. Picks the latest trajectories.jsonl under
# runs/ablations/<V>/full/, hands all of them to run_evaluation.py
# with --ablation flags, and writes a single report. The
# federated_factorability test consumes the lattice directly; the
# other seven tests run on the union of all ablations' trajectories.
#
# Usage:
#   bash scripts/evaluate_ablation_lattice.sh
#
# Environment variables (optional):
#   ABLATIONS     (default: "V0 V1 V2 V3 V4")
#   OUT_BASE      (default: runs/ablations)
#   H5_PATH       (default: data/synthetic_lj_v1/specimens.h5)
#   EVAL_OUT      (default: runs/eval)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

ABLATIONS="${ABLATIONS:-V0 V1 V2 V3 V4}"
OUT_BASE="${OUT_BASE:-runs/ablations}"
H5_PATH="${H5_PATH:-data/synthetic_lj_v1/specimens.h5}"
EVAL_OUT="${EVAL_OUT:-runs/eval}"

ABL_ARGS=()
FOUND=0
for V in ${ABLATIONS}; do
    LATEST=$(ls -td "${OUT_BASE}/${V}"/full/*/trajectories.jsonl 2>/dev/null | head -1 || true)
    if [ -z "${LATEST}" ]; then
        echo "==> Skipping ${V}: no trajectories under ${OUT_BASE}/${V}/full/"
        continue
    fi
    echo "==> ${V}: ${LATEST}"
    ABL_ARGS+=(--ablation "${V}=${LATEST}")
    FOUND=$((FOUND + 1))
done

if [ "${FOUND}" -lt 2 ]; then
    echo "Need >=2 ablations with trajectories; found ${FOUND}." >&2
    echo "Run scripts/run_ablation_lattice.sh first." >&2
    exit 1
fi

echo
echo "==> Running run_evaluation.py with ${FOUND} ablations"
echo

uv run python scripts/run_evaluation.py \
    --h5-path "${H5_PATH}" \
    --out "${EVAL_OUT}" \
    "${ABL_ARGS[@]}"
