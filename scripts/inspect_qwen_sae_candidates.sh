#!/usr/bin/env bash
#
# inspect_qwen_sae_candidates.sh
#
# Pretty-print the latest Phase 15 Stage C steering_candidates.yaml
# and (by default) tabulate the wrong-PASS distribution by
# (motif, phase) from the source 'full' baseline trajectories. The
# pattern in the SAE locks should match the pattern in the wrong
# commits; this is the sanity check before Stage D steering.
#
# Usage:
#   bash scripts/inspect_qwen_sae_candidates.sh
#   TOP_K=12 bash scripts/inspect_qwen_sae_candidates.sh
#   bash scripts/inspect_qwen_sae_candidates.sh --no-diagnose
#
# Environment variables (optional):
#   CANDIDATES        explicit path to steering_candidates.yaml
#   TRAJECTORIES      explicit path to full/.../trajectories.jsonl
#   H5_PATH           default: data/synthetic_lj_v1/specimens.h5
#   TOP_K             default: 8

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

H5_PATH="${H5_PATH:-data/synthetic_lj_v1/specimens.h5}"
TOP_K="${TOP_K:-8}"

EXTRA=()
EXTRA+=(--top-k "${TOP_K}")
EXTRA+=(--h5-path "${H5_PATH}")
if [ -n "${CANDIDATES:-}" ]; then
    EXTRA+=(--candidates "${CANDIDATES}")
fi
if [ -n "${TRAJECTORIES:-}" ]; then
    EXTRA+=(--trajectories "${TRAJECTORIES}")
fi

uv run python scripts/inspect_qwen_sae_candidates.py "${EXTRA[@]}" "$@"
