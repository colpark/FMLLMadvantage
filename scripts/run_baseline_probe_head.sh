#!/usr/bin/env bash
#
# run_baseline_probe_head.sh
#
# Phase 16 reference baseline: direct prediction from FM2 + probe
# bank, no LLM, no verifier. Output to runs/holdout/probe_head/ so
# scripts/evaluate_baselines.sh auto-discovers it as a column.
#
# This is the "FM downstream head" baseline that the
# (probes + SAE)-trained LLM is supposed to beat.
#
# Usage:
#   bash scripts/run_baseline_probe_head.sh
#   SPECIMEN_IDS_FILE=runs/holdout_lock/ids.json \
#       bash scripts/run_baseline_probe_head.sh
#
# Environment variables (optional):
#   PROBE_BANK_DIR      (default: latest under checkpoints/probes/)
#   START / COUNT       (defaults: 0 / 200)
#   SPECIMEN_IDS_FILE   (optional override)
#   SOLID_CENTROID_T    (default: 0.30)
#   LIQUID_CENTROID_T   (default: 0.80)
#   GPU                 (default: 0)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

START="${START:-0}"
COUNT="${COUNT:-200}"
SOLID_CENTROID_T="${SOLID_CENTROID_T:-0.30}"
LIQUID_CENTROID_T="${LIQUID_CENTROID_T:-0.80}"
GPU="${GPU:-0}"

EXTRA=()
EXTRA+=(--start "${START}")
EXTRA+=(--count "${COUNT}")
EXTRA+=(--solid-centroid-t "${SOLID_CENTROID_T}")
EXTRA+=(--liquid-centroid-t "${LIQUID_CENTROID_T}")
if [ -n "${SPECIMEN_IDS_FILE:-}" ]; then
    EXTRA+=(--specimen-ids-file "${SPECIMEN_IDS_FILE}")
fi
if [ -n "${PROBE_BANK_DIR:-}" ]; then
    EXTRA+=(--probe-bank-dir "${PROBE_BANK_DIR}")
fi

echo "==> Phase 16: probe_head (FM2 + probes, no LLM)"
echo "    Specimens        : ${SPECIMEN_IDS_FILE:-[${START}, $((START + COUNT)))}"
echo "    Probe bank       : ${PROBE_BANK_DIR:-(latest)}"
echo "    Solid centroid T : ${SOLID_CENTROID_T}"
echo "    Liquid centroid T: ${LIQUID_CENTROID_T}"
echo "    GPU              : ${GPU}"
echo

CUDA_VISIBLE_DEVICES="${GPU}" uv run python scripts/run_baseline_probe_head.py "${EXTRA[@]}"
