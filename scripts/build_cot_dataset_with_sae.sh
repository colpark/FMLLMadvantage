#!/usr/bin/env bash
#
# build_cot_dataset_with_sae.sh
#
# Phase 16 Stage 1: build the SAE-augmented synthetic CoT dataset.
# Each record is a chat (system, user, assistant) where the user
# message contains both PROBES and SAE_FEATURES, and the assistant
# rendering of the CoT references both. Ground truth comes from
# HDF5 metadata.
#
# Picks the latest probe bank, latest SAE, and latest SAE labels
# unless explicitly overridden.
#
# Usage:
#   bash scripts/build_cot_dataset_with_sae.sh
#
# Environment variables (optional):
#   N_SPECIMENS       (default: 10000)
#   TOP_K_FEATURES    (default: 8)
#   PROBE_BANK_DIR    (default: latest under checkpoints/probes/)
#   SAE_DIR           (default: latest under checkpoints/sae/)
#   SAE_LABELS_PATH   (default: latest labels.json under runs/sae_labels/)
#   GPU               (default: 0)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

N_SPECIMENS="${N_SPECIMENS:-10000}"
TOP_K_FEATURES="${TOP_K_FEATURES:-8}"
GPU="${GPU:-0}"

EXTRA=()
EXTRA+=(--n-specimens "${N_SPECIMENS}")
EXTRA+=(--top-k-features "${TOP_K_FEATURES}")
if [ -n "${PROBE_BANK_DIR:-}" ]; then
    EXTRA+=(--probe-bank-dir "${PROBE_BANK_DIR}")
fi
if [ -n "${SAE_DIR:-}" ]; then
    EXTRA+=(--sae-dir "${SAE_DIR}")
fi
if [ -n "${SAE_LABELS_PATH:-}" ]; then
    EXTRA+=(--sae-labels-path "${SAE_LABELS_PATH}")
fi

echo "==> Phase 16 Stage 1: build SAE-augmented CoT dataset"
echo "    N_SPECIMENS    : ${N_SPECIMENS}"
echo "    TOP_K_FEATURES : ${TOP_K_FEATURES}"
echo "    PROBE_BANK_DIR : ${PROBE_BANK_DIR:-(latest)}"
echo "    SAE_DIR        : ${SAE_DIR:-(latest)}"
echo "    SAE_LABELS_PATH: ${SAE_LABELS_PATH:-(latest)}"
echo "    GPU            : ${GPU}"
echo

CUDA_VISIBLE_DEVICES="${GPU}" uv run python scripts/build_cot_dataset_with_sae.py "${EXTRA[@]}"
