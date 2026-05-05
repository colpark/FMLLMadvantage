#!/usr/bin/env bash
#
# harvest_qwen_activations.sh
#
# Phase 15 Stage A: harvest Qwen residual-stream activations from a
# prior baseline trajectories.jsonl. For each trajectory we
# reconstruct a minimal chat (system + user query + assistant
# final_claim), forward it through Qwen with a hook on a target
# transformer layer, and capture the residual stream at the last
# token. One vector per trajectory.
#
# Default source: latest under runs/holdout/full/. Override via
# TRAJECTORIES to point at a larger training-distribution run when
# scaling up SAE training.
#
# Usage:
#   bash scripts/harvest_qwen_activations.sh
#
# Environment variables (optional):
#   TRAJECTORIES   path to trajectories.jsonl (default: latest under
#                  runs/holdout/full/)
#   BASE_MODEL     default: Qwen/Qwen2.5-7B-Instruct
#   ADAPTER_PATH   default: unset
#   LAYER_PATH     default: model.layers.14 (middle of 28-layer Qwen 7B)
#   QUANTIZE       'none' | '4bit' (default: 4bit)
#   MAX_TOKENS     default: 2048
#   GPU            default: 0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
LAYER_PATH="${LAYER_PATH:-model.layers.14}"
QUANTIZE="${QUANTIZE:-4bit}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
GPU="${GPU:-0}"

EXTRA=()
EXTRA+=(--base-model "${BASE_MODEL}")
EXTRA+=(--layer-path "${LAYER_PATH}")
EXTRA+=(--quantize "${QUANTIZE}")
EXTRA+=(--max-tokens "${MAX_TOKENS}")
if [ -n "${TRAJECTORIES:-}" ]; then
    EXTRA+=(--trajectories "${TRAJECTORIES}")
fi
if [ -n "${ADAPTER_PATH:-}" ]; then
    EXTRA+=(--adapter-path "${ADAPTER_PATH}")
fi

echo "==> Phase 15 Stage A: harvest Qwen residual activations"
echo "    Base model    : ${BASE_MODEL}"
echo "    Adapter       : ${ADAPTER_PATH:-(none)}"
echo "    Layer         : ${LAYER_PATH}"
echo "    Quantize      : ${QUANTIZE}"
echo "    Max tokens    : ${MAX_TOKENS}"
echo "    Trajectories  : ${TRAJECTORIES:-(latest under runs/holdout/full/)}"
echo "    GPU           : ${GPU}"
echo

CUDA_VISIBLE_DEVICES="${GPU}" uv run python scripts/harvest_qwen_activations.py \
    "${EXTRA[@]}"
