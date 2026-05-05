#!/usr/bin/env bash
#
# run_full_sae_direct.sh
#
# Bypass the Phase 13 wrapper and call the probe-augmented Pipeline A
# python CLI directly with --sae-dir on the argv. This is the
# minimum-indirection path: no env-var hand-off, no shell wrapper,
# no line-continuation hazard. Use when run_phase13_full_sae.sh has
# silently dropped Stage 2.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

GPU="${GPU:-0}"
SPECIMEN_IDS_FILE="${SPECIMEN_IDS_FILE:-runs/holdout_lock/ids.json}"

# Resolve the latest SAE directory that actually has sae.pt (skips
# any aborted partials).
SAE_DIR_RESOLVED=""
for d in $(ls -td checkpoints/sae/*/ 2>/dev/null); do
    if [ -f "${d%/}/sae.pt" ]; then
        SAE_DIR_RESOLVED="${d%/}"
        break
    fi
done
if [ -z "${SAE_DIR_RESOLVED}" ]; then
    echo "ERROR: no usable SAE under checkpoints/sae/ (need sae.pt)." >&2
    echo "       Run scripts/train_sae.sh first." >&2
    exit 1
fi

SAE_LABELS_RESOLVED="$(ls -td runs/sae_labels/*/labels.json 2>/dev/null | head -1 || true)"
if [ -z "${SAE_LABELS_RESOLVED}" ] || [ ! -f "${SAE_LABELS_RESOLVED}" ]; then
    echo "ERROR: no labels.json under runs/sae_labels/." >&2
    echo "       Run scripts/label_sae_features.sh first." >&2
    exit 1
fi

if [ ! -f "${SPECIMEN_IDS_FILE}" ]; then
    echo "ERROR: SPECIMEN_IDS_FILE not found: ${SPECIMEN_IDS_FILE}" >&2
    exit 1
fi

echo "==> Direct full_sae invocation"
echo "    GPU              : ${GPU}"
echo "    SPECIMEN_IDS_FILE: ${SPECIMEN_IDS_FILE}"
echo "    SAE dir          : ${SAE_DIR_RESOLVED}"
echo "    SAE labels       : ${SAE_LABELS_RESOLVED}"
echo

CUDA_VISIBLE_DEVICES="${GPU}" uv run python scripts/run_baseline_full_probes.py --specimen-ids-file "${SPECIMEN_IDS_FILE}" --out runs/holdout --base-model Qwen/Qwen2.5-7B-Instruct --max-steps 16 --ablation V4 --llm-temperature 0.4 --sae-dir "${SAE_DIR_RESOLVED}" --sae-labels-path "${SAE_LABELS_RESOLVED}" --sae-top-k-prompt 8

echo
echo "==> Post-run verification"
LATEST_FS="$(ls -td runs/holdout/full_sae/*/ 2>/dev/null | head -1 || true)"
if [ -z "${LATEST_FS}" ]; then
    echo "STILL MISSING — runs/holdout/full_sae/ was not created." >&2
    echo "Check the python CLI output above for an error." >&2
    exit 2
fi
LATEST_FS="${LATEST_FS%/}"
echo "    full_sae run : ${LATEST_FS}"
echo "    trajectories : $(wc -l < "${LATEST_FS}/trajectories.jsonl" | tr -d ' ')"
