#!/usr/bin/env bash
#
# build_cot_dataset.sh
#
# Phase 11 Stage 1: emit a JSONL of (probe outputs, synthetic CoT,
# ground truth) records ready for the Stage 2 SFT trainer.
#
# Usage:
#   bash scripts/build_cot_dataset.sh
#
# Environment variables (optional):
#   N_SPECIMENS   (default: 10000)
#   GPU           (default: 0)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

N_SPECIMENS="${N_SPECIMENS:-10000}"
GPU="${GPU:-0}"

echo "==> CoT dataset builder"
echo "    Specimens : ${N_SPECIMENS}"
echo "    GPU       : ${GPU}"
echo

CUDA_VISIBLE_DEVICES="${GPU}" uv run python scripts/build_cot_dataset.py \
    --n-specimens "${N_SPECIMENS}"
