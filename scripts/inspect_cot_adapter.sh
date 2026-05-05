#!/usr/bin/env bash
#
# inspect_cot_adapter.sh
#
# Spot-check the Phase 11 Stage 2 adapter. Picks the latest adapter
# under checkpoints/cot-sft/ and the latest probe bank under
# checkpoints/probes/, then generates CoTs on a handful of specimens
# with both the actual probe outputs and a "zero FM" control.
#
# Usage:
#   bash scripts/inspect_cot_adapter.sh
#   bash scripts/inspect_cot_adapter.sh -n 10
#   bash scripts/inspect_cot_adapter.sh --specimen-ids 0,42,1024
#
# Environment variables (optional):
#   GPU             (default: 0)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

GPU="${GPU:-0}"

CUDA_VISIBLE_DEVICES="${GPU}" uv run python scripts/inspect_cot_adapter.py "$@"
