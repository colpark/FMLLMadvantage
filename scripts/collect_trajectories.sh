#!/usr/bin/env bash
#
# collect_trajectories.sh
#
# Run Pipeline A across a range of specimens and save trajectories
# as JSONL. By default uses the mock LLM (no auth, no LLM weights).
# Pass --real to switch to Llama 3.1 8B Instruct.
#
# Usage:
#   bash scripts/collect_trajectories.sh                     # mock, first 200 specimens
#   bash scripts/collect_trajectories.sh 0 1000              # mock, first 1000
#   bash scripts/collect_trajectories.sh 0 200 train_30k     # different scale
#   bash scripts/collect_trajectories.sh --real 0 1000       # real Llama
#
# Environment variables (optional):
#   CONFIG          (default: configs/default.yaml)
#   H5_PATH         (default: data/synthetic_lj_v1/specimens.h5)
#   SPLITS_PATH     (default: data/synthetic_lj_v1/splits.yaml)
#   CHECKPOINT_ROOT (default: checkpoints)
#   LITERATURE_DB   (default: data/literature/clusters.json)
#   MOCK_SCRIPT     (default: scripts/mock_scripts/example.json)
#   LLM_MODEL       (default: meta-llama/Llama-3.1-8B-Instruct)
#   ABLATION        (default: V4)
#   MAX_STEPS       (default: 16)
#   GPU             (default: 0)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

USE_REAL=0
if [ "${1:-}" = "--real" ]; then
    USE_REAL=1
    shift
fi

START="${1:-0}"
COUNT="${2:-200}"
TRAIN_SPLIT="${3:-train_50k}"

CONFIG="${CONFIG:-configs/default.yaml}"
H5_PATH="${H5_PATH:-data/synthetic_lj_v1/specimens.h5}"
SPLITS_PATH="${SPLITS_PATH:-data/synthetic_lj_v1/splits.yaml}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints}"
LITERATURE_DB="${LITERATURE_DB:-data/literature/clusters.json}"
MOCK_SCRIPT="${MOCK_SCRIPT:-scripts/mock_scripts/example.json}"
LLM_MODEL="${LLM_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
ABLATION="${ABLATION:-V4}"
MAX_STEPS="${MAX_STEPS:-16}"
GPU="${GPU:-0}"

OUT_ROOT="runs/trajectories"
mkdir -p "${OUT_ROOT}"

echo "==> Pipeline A trajectory collection"
if [ "${USE_REAL}" -eq 1 ]; then
    echo "    Mode           : real LLM (${LLM_MODEL})"
else
    echo "    Mode           : mock (${MOCK_SCRIPT})"
fi
echo "    Specimens      : [${START}, $((START + COUNT)))"
echo "    Train split    : ${TRAIN_SPLIT}"
echo "    Ablation       : ${ABLATION}"
echo "    Max steps      : ${MAX_STEPS}"
echo "    Output root    : ${OUT_ROOT}"
echo

EXTRA=()
if [ "${USE_REAL}" -eq 1 ]; then
    EXTRA+=(--llm-model "${LLM_MODEL}")
    if [ -n "${ADAPTER_PATH:-}" ]; then
        EXTRA+=(--adapter-path "${ADAPTER_PATH}")
        echo "    Adapter path   : ${ADAPTER_PATH}"
    fi
else
    EXTRA+=(--mock-script "${MOCK_SCRIPT}")
fi

CUDA_VISIBLE_DEVICES="${GPU}" uv run python scripts/collect_trajectories.py \
    --start "${START}" \
    --count "${COUNT}" \
    --config "${CONFIG}" \
    --h5-path "${H5_PATH}" \
    --splits-path "${SPLITS_PATH}" \
    --checkpoint-root "${CHECKPOINT_ROOT}" \
    --train-split "${TRAIN_SPLIT}" \
    --literature-db "${LITERATURE_DB}" \
    --out "${OUT_ROOT}" \
    --max-steps "${MAX_STEPS}" \
    --ablation "${ABLATION}" \
    "${EXTRA[@]}"
