#!/usr/bin/env bash
#
# run_ablation_lattice.sh
#
# Run the full pipeline once per verifier ablation preset (V0..V4) on
# the same span of specimens. Each run lands under
# runs/ablations/<V>/full/<run-id>/. The federated_factorability test
# in Phase 7 needs >=2 ablations side-by-side to produce a non-skip
# result; running all five at once unlocks the full lattice.
#
# Sequential by default. To parallelize across the 4xH100 host, run
# different ablations in separate shells with different GPU= values.
#
# Usage:
#   bash scripts/run_ablation_lattice.sh
#
# Environment variables (optional):
#   START         (default: 0)
#   COUNT         (default: 200)
#   ABLATIONS     (default: "V0 V1 V2 V3 V4"; space-separated)
#   OUT_BASE      (default: runs/ablations)
#   GPU           (default: 0)
#   LLM_MODEL     (default: Qwen/Qwen2.5-7B-Instruct)
#   LLM_TEMP      (default: 0.4)
#   SKIP_EXISTING (default: 0; set to 1 to skip ablations that
#                  already have a trajectories.jsonl on disk)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

START="${START:-0}"
COUNT="${COUNT:-200}"
ABLATIONS="${ABLATIONS:-V0 V1 V2 V3 V4}"
OUT_BASE="${OUT_BASE:-runs/ablations}"
GPU="${GPU:-0}"
LLM_MODEL="${LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
LLM_TEMP="${LLM_TEMP:-0.4}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

echo "==> Ablation lattice"
echo "    Specimens : [${START}, $((START + COUNT)))"
echo "    Ablations : ${ABLATIONS}"
echo "    Output    : ${OUT_BASE}/<V>/full/<run-id>/"
echo "    GPU       : ${GPU}"
echo "    LLM       : ${LLM_MODEL} (T=${LLM_TEMP})"
echo

for V in ${ABLATIONS}; do
    if ! [[ "${V}" =~ ^V[0-4]$ ]]; then
        echo "ERROR: ablation key must match V0..V4, got ${V}" >&2
        exit 1
    fi

    if [ "${SKIP_EXISTING}" -eq 1 ]; then
        EXISTING=$(ls -td "${OUT_BASE}/${V}"/full/*/trajectories.jsonl 2>/dev/null | head -1 || true)
        if [ -n "${EXISTING}" ]; then
            echo "==> ${V}: skipping (existing ${EXISTING})"
            echo
            continue
        fi
    fi

    echo "================================================================="
    echo "==> Running ablation ${V}"
    echo "================================================================="

    START="${START}" \
        COUNT="${COUNT}" \
        OUT_ROOT="${OUT_BASE}/${V}" \
        GPU="${GPU}" \
        LLM_MODEL="${LLM_MODEL}" \
        LLM_TEMP="${LLM_TEMP}" \
        bash scripts/run_baseline.sh full --ablation "${V}"

    echo
    echo "==> ${V}: done"
    echo
done

echo "================================================================="
echo "==> Lattice complete"
echo "================================================================="
echo "Latest trajectories per ablation:"
for V in ${ABLATIONS}; do
    LATEST=$(ls -td "${OUT_BASE}/${V}"/full/*/trajectories.jsonl 2>/dev/null | head -1 || true)
    if [ -n "${LATEST}" ]; then
        echo "  ${V}: ${LATEST}"
    else
        echo "  ${V}: (none)"
    fi
done

echo
echo "Next: bash scripts/evaluate_ablation_lattice.sh"
