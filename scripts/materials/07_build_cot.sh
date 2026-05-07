#!/usr/bin/env bash
#
# 07_build_cot.sh -- Stage 7 of the materials port.
# Build the SAE-augmented synthetic CoT dataset (chat records:
# system / user / assistant) by combining cached CHGNet embeddings
# + materials probe bank + materials SAE + materials labels +
# materials ground truth.
#
# Usage:
#   bash scripts/materials/07_build_cot.sh
#   N_SPECIMENS=20000 bash scripts/materials/07_build_cot.sh
#
# Environment variables (optional):
#   EMBEDDINGS_DIR   default: latest under runs/materials/embeddings/
#   PROBE_BANK_DIR   default: latest under checkpoints/materials/probes/
#   SAE_DIR          default: latest under checkpoints/materials/sae/
#   SAE_LABELS_PATH  default: latest labels.json under runs/materials/sae_labels/
#   N_SPECIMENS      default: 10000
#   TOP_K_FEATURES   default: 8
#   GPU              default: 0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

N_SPECIMENS="${N_SPECIMENS:-10000}"
TOP_K_FEATURES="${TOP_K_FEATURES:-8}"
INCLUDE_SAE="${INCLUDE_SAE:-1}"
GPU="${GPU:-0}"
# Default output root depends on whether SAE features are included,
# so the no-SAE ablation lands in a separate dir for downstream
# stages to find via their own latest-discovery.
if [ "${INCLUDE_SAE}" = "0" ] || [ "${INCLUDE_SAE}" = "false" ]; then
    DEFAULT_OUT="runs/materials/cot_datasets_no_sae"
else
    DEFAULT_OUT="runs/materials/cot_datasets_sae"
fi
OUT_ROOT="${OUT_ROOT:-${DEFAULT_OUT}}"

EXTRA=()
EXTRA+=(--n-specimens "${N_SPECIMENS}")
EXTRA+=(--top-k-features "${TOP_K_FEATURES}")
EXTRA+=(--out "${OUT_ROOT}")
if [ "${INCLUDE_SAE}" = "0" ] || [ "${INCLUDE_SAE}" = "false" ]; then
    EXTRA+=(--no-include-sae)
else
    EXTRA+=(--include-sae)
fi
if [ -n "${EMBEDDINGS_DIR:-}" ]; then
    EXTRA+=(--embeddings-dir "${EMBEDDINGS_DIR}")
fi
if [ -n "${PROBE_BANK_DIR:-}" ]; then
    EXTRA+=(--probe-bank-dir "${PROBE_BANK_DIR}")
fi
if [ -n "${SAE_DIR:-}" ]; then
    EXTRA+=(--sae-dir "${SAE_DIR}")
fi
if [ -n "${SAE_LABELS_PATH:-}" ]; then
    EXTRA+=(--sae-labels-path "${SAE_LABELS_PATH}")
fi

echo "==> Materials port Stage 7: build CoT records"
echo "    EMBEDDINGS_DIR  : ${EMBEDDINGS_DIR:-(latest)}"
echo "    PROBE_BANK_DIR  : ${PROBE_BANK_DIR:-(latest)}"
echo "    SAE_DIR         : ${SAE_DIR:-(latest)}"
echo "    SAE_LABELS_PATH : ${SAE_LABELS_PATH:-(latest)}"
echo "    N_SPECIMENS     : ${N_SPECIMENS}"
echo "    TOP_K_FEATURES  : ${TOP_K_FEATURES}"
echo "    INCLUDE_SAE     : ${INCLUDE_SAE}"
echo "    OUT_ROOT        : ${OUT_ROOT}"
echo "    GPU             : ${GPU}"
echo

CUDA_VISIBLE_DEVICES="${GPU}" uv run python scripts/materials/07_build_cot.py "${EXTRA[@]}"
