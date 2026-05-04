#!/usr/bin/env bash
#
# evaluate_baselines.sh
#
# One-shot baseline evaluation:
#   1. Auto-discovers the latest trajectories.jsonl under each of
#      runs/baselines/{naked,no_verifier,full}/<run-id>/.
#   2. Runs scripts/run_evaluation.py against each.
#   3. Parses the report path from each evaluation's stdout.
#   4. Hands all reports to scripts/compare_baselines.py for the
#      side-by-side table.
#
# No run-ids needed. The script prints what it picked, evaluates,
# and compares in a single command.
#
# Usage:
#   bash scripts/evaluate_baselines.sh
#
# Environment variables (optional):
#   H5_PATH          (default: data/synthetic_lj_v1/specimens.h5)
#   OUT_ROOT         (default: runs/eval)
#   BASELINES_ROOT   (default: runs/baselines)
#   BASELINES        (default: "naked no_verifier full"; space-separated)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

H5_PATH="${H5_PATH:-data/synthetic_lj_v1/specimens.h5}"
OUT_ROOT="${OUT_ROOT:-runs/eval}"
BASELINES_ROOT="${BASELINES_ROOT:-runs/baselines}"
BASELINES="${BASELINES:-naked no_verifier full}"

REPORT_ARGS=()
EVAL_COUNT=0

for baseline in ${BASELINES}; do
    TRAJ=$(ls -td "${BASELINES_ROOT}/${baseline}"/*/trajectories.jsonl 2>/dev/null | head -1 || true)
    if [ -z "${TRAJ}" ]; then
        echo "==> Skipping ${baseline}: no trajectories.jsonl under ${BASELINES_ROOT}/${baseline}/"
        echo
        continue
    fi

    echo "==> Evaluating ${baseline}"
    echo "    Trajectories: ${TRAJ}"

    LOG_FILE=$(mktemp)
    uv run python scripts/run_evaluation.py \
        --trajectories "${TRAJ}" \
        --h5-path "${H5_PATH}" \
        --out "${OUT_ROOT}" 2>&1 | tee "${LOG_FILE}"

    # The eval CLI prints "==> Report: <path>" near the top of its output.
    REPORT=$(grep -oE "${OUT_ROOT}/[^ ]+/report\.yaml" "${LOG_FILE}" | tail -1 || true)
    rm -f "${LOG_FILE}"

    if [ -z "${REPORT}" ]; then
        echo "ERROR: could not parse report path for ${baseline}" >&2
        exit 1
    fi
    echo "==> ${baseline} report: ${REPORT}"
    echo
    REPORT_ARGS+=("${baseline}=${REPORT}")
    EVAL_COUNT=$((EVAL_COUNT + 1))
done

if [ "${EVAL_COUNT}" -lt 2 ]; then
    echo "Need at least 2 baselines with trajectories; found ${EVAL_COUNT}." >&2
    echo "Run scripts/run_baseline.sh {naked|no_verifier|full} first." >&2
    exit 1
fi

echo "==========================================================="
echo "Side-by-side comparison"
echo "==========================================================="
COMPARE_ARGS=()
for spec in "${REPORT_ARGS[@]}"; do
    COMPARE_ARGS+=(--report "${spec}")
done
uv run python scripts/compare_baselines.py "${COMPARE_ARGS[@]}"
