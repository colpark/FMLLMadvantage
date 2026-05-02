#!/usr/bin/env bash
#
# run_pipeline_smoke.sh
#
# Mock-script smoke test for the OHVD loop. Loads the actual FMs and
# bridges from disk, runs each FM forward pass on the requested
# specimen, and feeds the bridged outputs into a scripted LLM
# response sequence (no LLM weights required). Useful for verifying
# Phase 5 wiring before launching the real Llama path.
#
# Usage:
#   bash scripts/run_pipeline_smoke.sh                    # specimen 42
#   bash scripts/run_pipeline_smoke.sh 7                  # specimen 7
#   bash scripts/run_pipeline_smoke.sh 7 train_30k        # different scale
#
# Environment variables (optional):
#   CONFIG          (default: configs/default.yaml)
#   H5_PATH         (default: data/synthetic_lj_v1/specimens.h5)
#   SPLITS_PATH     (default: data/synthetic_lj_v1/splits.yaml)
#   CHECKPOINT_ROOT (default: checkpoints)
#   LITERATURE_DB   (default: data/literature/clusters.json)
#   MOCK_SCRIPT     (default: scripts/mock_scripts/example.json)
#   ABLATION        (default: V4)
#   GPU             (default: 0)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

SPECIMEN_ID="${1:-42}"
TRAIN_SPLIT="${2:-train_50k}"

CONFIG="${CONFIG:-configs/default.yaml}"
H5_PATH="${H5_PATH:-data/synthetic_lj_v1/specimens.h5}"
SPLITS_PATH="${SPLITS_PATH:-data/synthetic_lj_v1/splits.yaml}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints}"
LITERATURE_DB="${LITERATURE_DB:-data/literature/clusters.json}"
MOCK_SCRIPT="${MOCK_SCRIPT:-scripts/mock_scripts/example.json}"
ABLATION="${ABLATION:-V4}"
GPU="${GPU:-0}"

echo "==> Pipeline A smoke (mock LLM)"
echo "    Specimen ID    : ${SPECIMEN_ID}"
echo "    Train split    : ${TRAIN_SPLIT}"
echo "    Mock script    : ${MOCK_SCRIPT}"
echo "    Ablation       : ${ABLATION}"
echo "    GPU            : ${GPU}"
echo

CUDA_VISIBLE_DEVICES="${GPU}" uv run python scripts/run_pipeline.py \
    --specimen-id "${SPECIMEN_ID}" \
    --config "${CONFIG}" \
    --h5-path "${H5_PATH}" \
    --splits-path "${SPLITS_PATH}" \
    --checkpoint-root "${CHECKPOINT_ROOT}" \
    --train-split "${TRAIN_SPLIT}" \
    --literature-db "${LITERATURE_DB}" \
    --ablation "${ABLATION}" \
    --mock-script "${MOCK_SCRIPT}"

# Find the most recent trajectory under runs/ matching this specimen.
LATEST=$(ls -td runs/*pipeline-a-${SPECIMEN_ID}* 2>/dev/null | head -1 || true)
if [ -n "${LATEST}" ]; then
    echo
    echo "==> Latest trajectory: ${LATEST}"
    bash "${SCRIPT_DIR}/inspect_trajectory.sh" "${LATEST}"
fi
