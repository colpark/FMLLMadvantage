#!/usr/bin/env bash
#
# 06b_diagnose_sae.sh -- diagnostic for the trained SAE.
# Reports dead-feature count, activation distribution, mask overlap,
# and reconstruction-error distribution. Decides whether the SAE
# needs an upgrade (resampling, JumpReLU, Gated SAE) before stage 7.
#
# Usage:
#   bash scripts/materials/06b_diagnose_sae.sh
#
# Environment variables (optional):
#   SAE_DIR          default: latest under checkpoints/materials/sae/
#   EMBEDDINGS_DIR   default: from SAE manifest
#   GPU              default: 0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

GPU="${GPU:-0}"

EXTRA=()
if [ -n "${SAE_DIR:-}" ]; then
    EXTRA+=(--sae-dir "${SAE_DIR}")
fi
if [ -n "${EMBEDDINGS_DIR:-}" ]; then
    EXTRA+=(--embeddings-dir "${EMBEDDINGS_DIR}")
fi

echo "==> Materials port Stage 6b: SAE diagnostic"
echo "    SAE_DIR        : ${SAE_DIR:-(latest)}"
echo "    EMBEDDINGS_DIR : ${EMBEDDINGS_DIR:-(from SAE manifest)}"
echo "    GPU            : ${GPU}"
echo

CUDA_VISIBLE_DEVICES="${GPU}" uv run python scripts/materials/06b_diagnose_sae.py "${EXTRA[@]}"
