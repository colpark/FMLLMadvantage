#!/usr/bin/env bash
#
# 06_label_sae.sh -- Stage 6 of the materials port.
# Label SAE features by correlation with materials attributes.
#
# Usage:
#   bash scripts/materials/06_label_sae.sh
#
# Environment variables (optional):
#   SAE_DIR           default: latest under checkpoints/materials/sae/
#   EMBEDDINGS_DIR    default: the one used by SAE training
#   TOP_N             default: 50
#   MIN_PURITY        default: 0.70
#   MIN_CORR          default: 0.30
#   GPU               default: 0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TOP_N="${TOP_N:-50}"
MIN_PURITY="${MIN_PURITY:-0.70}"
MIN_CORR="${MIN_CORR:-0.30}"
GPU="${GPU:-0}"

EXTRA=()
EXTRA+=(--top-n "${TOP_N}")
EXTRA+=(--min-purity "${MIN_PURITY}")
EXTRA+=(--min-corr "${MIN_CORR}")
if [ -n "${SAE_DIR:-}" ]; then
    EXTRA+=(--sae-dir "${SAE_DIR}")
fi
if [ -n "${EMBEDDINGS_DIR:-}" ]; then
    EXTRA+=(--embeddings-dir "${EMBEDDINGS_DIR}")
fi

echo "==> Materials port Stage 6: label SAE features"
echo "    SAE_DIR          : ${SAE_DIR:-(latest)}"
echo "    EMBEDDINGS_DIR   : ${EMBEDDINGS_DIR:-(from SAE manifest)}"
echo "    TOP_N            : ${TOP_N}"
echo "    MIN_PURITY       : ${MIN_PURITY}"
echo "    MIN_CORR         : ${MIN_CORR}"
echo "    GPU              : ${GPU}"
echo

CUDA_VISIBLE_DEVICES="${GPU}" uv run python scripts/materials/06_label_sae.py "${EXTRA[@]}"
