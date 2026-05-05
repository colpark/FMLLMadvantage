#!/usr/bin/env bash
#
# run_fm2_probes.sh
#
# Phase 9.0 probing study. Tests whether FM2's energy-supervised
# representation holds task-extra signal that the head suppressed.
#
# Decision rule for the next phase:
#   all probes >= 0.85   ⇒  proceed with FM2 connector (Phase 9.A)
#   mixed                ⇒  build connector but expect modest gains
#   all near chance      ⇒  representation collapsed to energy; skip
#                            connector and consider self-supervised
#                            pretraining instead
#
# Usage:
#   bash scripts/run_fm2_probes.sh
#
# Environment variables (optional):
#   TRAIN_SPLIT       (default: train_50k)
#   PROBE_SPLIT       (default: train_50k; switch to a held-out slice
#                      if available)
#   MAX_SPECIMENS     (default: 2000)
#   PROBE_ARCH        (default: mlp; alternative: linear)
#   EPOCHS            (default: 30)
#   GPU               (default: 0)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

TRAIN_SPLIT="${TRAIN_SPLIT:-train_50k}"
PROBE_SPLIT="${PROBE_SPLIT:-train_50k}"
MAX_SPECIMENS="${MAX_SPECIMENS:-2000}"
PROBE_ARCH="${PROBE_ARCH:-mlp}"
EPOCHS="${EPOCHS:-30}"
GPU="${GPU:-0}"

echo "==> FM2 probing study"
echo "    Train split   : ${TRAIN_SPLIT}"
echo "    Probe split   : ${PROBE_SPLIT}"
echo "    Max specimens : ${MAX_SPECIMENS}"
echo "    Probe arch    : ${PROBE_ARCH}"
echo "    Epochs        : ${EPOCHS}"
echo "    GPU           : ${GPU}"
echo

CUDA_VISIBLE_DEVICES="${GPU}" uv run python scripts/run_fm2_probes.py \
    --train-split "${TRAIN_SPLIT}" \
    --probe-split "${PROBE_SPLIT}" \
    --max-specimens "${MAX_SPECIMENS}" \
    --probe-arch "${PROBE_ARCH}" \
    --epochs "${EPOCHS}"
