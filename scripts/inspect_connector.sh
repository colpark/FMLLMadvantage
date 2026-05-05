#!/usr/bin/env bash
#
# inspect_connector.sh
#
# Diagnostic A: spot-check whether a trained FM2 connector is genuinely
# conditioning on FM2 features or just decorating the LLM prompt.
# Picks the latest connector.pt under runs/connectors/ and prints
# (truth, real-FM generation, zero-FM generation) for a handful of
# specimens.
#
# Reading guide:
#   - real-FM matches truth on N/motif/T → connector is real.
#   - zero-FM produces a generic description → confirms the conditioning
#     is via the connector tokens.
#   - both look identical → connector is decorative; rerun with the
#     shuffle ablation to confirm.
#
# Usage:
#   bash scripts/inspect_connector.sh                     # 5 specimens, IDs [0..4]
#   bash scripts/inspect_connector.sh -n 10               # 10 specimens
#   bash scripts/inspect_connector.sh --specimen-ids 0,42,1024
#
# Environment variables (optional):
#   GPU             (default: 0)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

GPU="${GPU:-0}"

CUDA_VISIBLE_DEVICES="${GPU}" uv run python scripts/inspect_connector.py "$@"
