#!/usr/bin/env bash
#
# 09b_run_probe_head.sh -- probe-head baseline (no LLM).
# CHGNet's supervised heads alone, scored on the same held-out 200
# with the same joint-correctness criterion as Stage 9. Establishes
# the floor for what the LLM has to beat.
#
# Usage:
#   bash scripts/materials/09b_run_probe_head.sh
#
# Environment variables (optional):
#   PROBE_BANK_DIR    default: latest under checkpoints/materials/probes/
#   HOLDOUT_IDS_PATH  default: data/materials_project_v1/holdout_lock/ids.json
#   CHGNET_MODEL_NAME default: 0.3.0
#   MAX_ATOMS         default: 80
#   GPU               default: 0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CHGNET_MODEL_NAME="${CHGNET_MODEL_NAME:-0.3.0}"
MAX_ATOMS="${MAX_ATOMS:-80}"
GPU="${GPU:-0}"
HOLDOUT_IDS_PATH="${HOLDOUT_IDS_PATH:-data/materials_project_v1/holdout_lock/ids.json}"

EXTRA=()
EXTRA+=(--chgnet-model-name "${CHGNET_MODEL_NAME}")
EXTRA+=(--max-atoms "${MAX_ATOMS}")
EXTRA+=(--holdout-ids-path "${HOLDOUT_IDS_PATH}")
if [ -n "${PROBE_BANK_DIR:-}" ]; then
    EXTRA+=(--probe-bank-dir "${PROBE_BANK_DIR}")
fi

echo "==> Materials port Stage 9b: probe-head baseline (no LLM)"
echo "    PROBE_BANK_DIR    : ${PROBE_BANK_DIR:-(latest)}"
echo "    HOLDOUT_IDS_PATH  : ${HOLDOUT_IDS_PATH}"
echo "    CHGNET_MODEL_NAME : ${CHGNET_MODEL_NAME}"
echo "    MAX_ATOMS         : ${MAX_ATOMS}"
echo "    GPU               : ${GPU}"
echo

CUDA_VISIBLE_DEVICES="${GPU}" uv run python scripts/materials/09b_run_probe_head.py "${EXTRA[@]}"
