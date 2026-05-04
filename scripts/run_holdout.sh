#!/usr/bin/env bash
#
# run_holdout.sh
#
# Run the four-configuration held-out evaluation protocol with locked
# thresholds and a pinned specimen range:
#
#   1. Verify locked thresholds match code defaults (abort on drift).
#   2. Resolve held-out specimen IDs from configs/holdout_lock.yaml.
#   3. Run all three baselines (naked, no_verifier, full) on the held-
#      out IDs. Output goes under runs/holdout/<baseline>/.
#   4. Run an extra "full-strict" configuration with literature
#      compare_energy=True. Output under runs/holdout-strict/.
#   5. Evaluate every configuration and write a comparison.
#
# Usage:
#   bash scripts/run_holdout.sh
#   bash scripts/run_holdout.sh --skip-strict   # omit the strict-literature run
#
# Environment variables (optional):
#   GPU       (default: 0)
#   LLM_MODEL (default: Qwen/Qwen2.5-7B-Instruct)
#   LLM_TEMP  (default: 0.4)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

SKIP_STRICT=0
for arg in "$@"; do
    case "${arg}" in
        --skip-strict) SKIP_STRICT=1 ;;
    esac
done

GPU="${GPU:-0}"
LLM_MODEL="${LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
LLM_TEMP="${LLM_TEMP:-0.4}"

echo "==============================================================="
echo "Held-out evaluation protocol"
echo "==============================================================="
echo "GPU       : ${GPU}"
echo "LLM       : ${LLM_MODEL} (T=${LLM_TEMP})"
echo "Skip strict-literature run: ${SKIP_STRICT}"
echo

# 1. Verify thresholds.
echo "==> Step 1: verify locked thresholds"
uv run python scripts/verify_thresholds.py
echo

# 2. Resolve held-out IDs.
echo "==> Step 2: resolve held-out specimen IDs"
uv run python scripts/pick_holdout_ids.py
echo

IDS_FILE="${REPO_ROOT}/runs/holdout_lock/ids.json"
if [ ! -f "${IDS_FILE}" ]; then
    echo "ERROR: held-out IDs not produced at ${IDS_FILE}" >&2
    exit 1
fi
N_IDS=$(uv run python -c "import json; print(len(json.load(open('${IDS_FILE}'))))")
echo "==> ${N_IDS} held-out specimens locked at ${IDS_FILE}"
echo

# 3. Run the three baselines on held-out IDs.
for baseline in naked no_verifier full; do
    echo "==============================================================="
    echo "==> Step 3.${baseline}: held-out baseline ${baseline}"
    echo "==============================================================="
    SPECIMEN_IDS_FILE="${IDS_FILE}" \
        OUT_ROOT="runs/holdout" \
        GPU="${GPU}" \
        LLM_MODEL="${LLM_MODEL}" \
        LLM_TEMP="${LLM_TEMP}" \
        bash scripts/run_baseline.sh "${baseline}"
    echo
done

# 4. Optional strict-literature configuration.
if [ "${SKIP_STRICT}" -eq 0 ]; then
    echo "==============================================================="
    echo "==> Step 4: held-out full + literature_compare_energy=True"
    echo "==============================================================="
    SPECIMEN_IDS_FILE="${IDS_FILE}" \
        OUT_ROOT="runs/holdout-strict" \
        LITERATURE_COMPARE_ENERGY=1 \
        GPU="${GPU}" \
        LLM_MODEL="${LLM_MODEL}" \
        LLM_TEMP="${LLM_TEMP}" \
        bash scripts/run_baseline.sh full
    echo
fi

eval_one() {
    # eval_one <baseline_label> <trajectories.jsonl>
    # Runs scripts/run_evaluation.py on the given trajectories file,
    # parses the report path from its stdout, and echoes JUST the
    # report path on the final line. Tee'd output goes to stderr so
    # callers can capture the report path without log noise.
    local label="$1"
    local traj="$2"
    local logf
    logf=$(mktemp)
    uv run python scripts/run_evaluation.py \
        --trajectories "${traj}" \
        --h5-path "${H5_PATH:-data/synthetic_lj_v1/specimens.h5}" \
        --out runs/eval 2>&1 | tee "${logf}" >&2
    local rep
    rep=$(grep -oE "runs/eval/[^ ]+/report\.yaml" "${logf}" | tail -1)
    rm -f "${logf}"
    if [ -z "${rep}" ]; then
        echo "ERROR: could not parse report path for ${label}" >&2
        exit 1
    fi
    echo "${rep}"
}

# 5. Evaluate the three default-literature baselines and capture each
# report path explicitly. We do not rely on mtime ordering because
# multiple eval runs can produce ambiguous results.
echo "==============================================================="
echo "==> Step 5: evaluate held-out runs"
echo "==============================================================="
NAKED_TRAJ=$(ls -td runs/holdout/naked/*/trajectories.jsonl 2>/dev/null | head -1 || true)
NV_TRAJ=$(ls -td runs/holdout/no_verifier/*/trajectories.jsonl 2>/dev/null | head -1 || true)
FULL_TRAJ=$(ls -td runs/holdout/full/*/trajectories.jsonl 2>/dev/null | head -1 || true)
if [ -z "${NAKED_TRAJ}" ] || [ -z "${NV_TRAJ}" ] || [ -z "${FULL_TRAJ}" ]; then
    echo "ERROR: held-out trajectories missing under runs/holdout/" >&2
    exit 1
fi

echo "==> naked       : ${NAKED_TRAJ}"
NAKED_REPORT=$(eval_one naked "${NAKED_TRAJ}")
echo "    report      : ${NAKED_REPORT}"
echo

echo "==> no_verifier : ${NV_TRAJ}"
NV_REPORT=$(eval_one no_verifier "${NV_TRAJ}")
echo "    report      : ${NV_REPORT}"
echo

echo "==> full        : ${FULL_TRAJ}"
FULL_REPORT=$(eval_one full "${FULL_TRAJ}")
echo "    report      : ${FULL_REPORT}"
echo

echo "==> Default-literature comparison"
uv run python scripts/compare_baselines.py \
    --report "naked=${NAKED_REPORT}" \
    --report "no_verifier=${NV_REPORT}" \
    --report "full=${FULL_REPORT}"

if [ "${SKIP_STRICT}" -eq 0 ]; then
    echo "==============================================================="
    echo "==> Step 5b: evaluate strict-literature full run"
    echo "==============================================================="
    STRICT_TRAJ=$(ls -td runs/holdout-strict/full/*/trajectories.jsonl 2>/dev/null | head -1 || true)
    if [ -z "${STRICT_TRAJ}" ]; then
        echo "Skipping strict comparison: no trajectories under runs/holdout-strict/full/" >&2
    else
        echo "==> full (strict literature): ${STRICT_TRAJ}"
        STRICT_REPORT=$(eval_one full_strict "${STRICT_TRAJ}")
        echo "    report                  : ${STRICT_REPORT}"
        echo
        echo "==> Four-configuration comparison (default + strict)"
        uv run python scripts/compare_baselines.py \
            --report "naked=${NAKED_REPORT}" \
            --report "no_verifier=${NV_REPORT}" \
            --report "full_default=${FULL_REPORT}" \
            --report "full_strict=${STRICT_REPORT}"
    fi
fi

echo
echo "==============================================================="
echo "Held-out protocol complete"
echo "==============================================================="
echo "Reports under runs/eval/. The held-out comparison is the"
echo "publishable headline; the strict-literature comparison is the"
echo "footnote that documents the alternative configuration."
