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

# 5. Evaluate everything.
echo "==============================================================="
echo "==> Step 5: evaluate held-out runs"
echo "==============================================================="
BASELINES_ROOT=runs/holdout bash scripts/evaluate_baselines.sh
echo

if [ "${SKIP_STRICT}" -eq 0 ]; then
    echo "==============================================================="
    echo "==> Step 5b: evaluate strict-literature full run"
    echo "==============================================================="
    # Compare strict-literature full against the same naked + no_verifier.
    # Build a synthetic baseline tree pointing at strict's full and the
    # default holdout's naked + no_verifier.
    NAKED=$(ls -td runs/holdout/naked/*/trajectories.jsonl 2>/dev/null | head -1)
    NO_VERIFIER=$(ls -td runs/holdout/no_verifier/*/trajectories.jsonl 2>/dev/null | head -1)
    FULL_STRICT=$(ls -td runs/holdout-strict/full/*/trajectories.jsonl 2>/dev/null | head -1)
    if [ -n "${NAKED}" ] && [ -n "${NO_VERIFIER}" ] && [ -n "${FULL_STRICT}" ]; then
        echo "Evaluating strict-literature lattice:"
        echo "  naked         : ${NAKED}"
        echo "  no_verifier   : ${NO_VERIFIER}"
        echo "  full (strict) : ${FULL_STRICT}"

        # Use a shadow tree so evaluate_baselines.sh's auto-discovery
        # picks up strict's full instead of default's full.
        SHADOW="runs/holdout-strict-compare"
        mkdir -p "${SHADOW}/naked" "${SHADOW}/no_verifier" "${SHADOW}/full"
        ln -sfn "$(dirname "${NAKED}")"       "${SHADOW}/naked/run"
        ln -sfn "$(dirname "${NO_VERIFIER}")" "${SHADOW}/no_verifier/run"
        ln -sfn "$(dirname "${FULL_STRICT}")" "${SHADOW}/full/run"
        BASELINES_ROOT="${SHADOW}" bash scripts/evaluate_baselines.sh
    else
        echo "Skipping strict comparison: missing trajectories." >&2
    fi
fi

echo
echo "==============================================================="
echo "Held-out protocol complete"
echo "==============================================================="
echo "Reports under runs/eval/. The held-out comparison is the"
echo "publishable headline; the strict-literature comparison is the"
echo "footnote that documents the alternative configuration."
