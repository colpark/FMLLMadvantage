#!/usr/bin/env bash
#
# compare_baselines.sh -- side-by-side baseline comparison.
# Reads the latest probe_head + cot_sft_sae JSONLs and prints
# joint accuracy, per-axis accuracy, and the LLM contribution
# (cot_sft_sae - probe_head) as a delta table.
#
# Usage:
#   bash scripts/materials/compare_baselines.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

uv run python scripts/materials/compare_baselines.py "$@"
