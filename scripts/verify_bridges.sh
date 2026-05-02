#!/usr/bin/env bash
#
# verify_bridges.sh
#
# Bash wrapper around verify_bridges.py. Runs the bridge wiring smoke
# test against one or more training scales and prints a one-line per-FM
# summary plus the location of the artifacts.
#
# Usage:
#   bash scripts/verify_bridges.sh                       # all three scales
#   bash scripts/verify_bridges.sh train_50k             # one scale
#   bash scripts/verify_bridges.sh train_10k train_30k   # two scales
#
# Environment variables (optional):
#   CHECKPOINT_ROOT (default: checkpoints)
#   OUT_ROOT        (default: runs/bridge-verify)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints}"
OUT_ROOT="${OUT_ROOT:-runs/bridge-verify}"

if [ "$#" -eq 0 ]; then
    SCALES=(train_10k train_30k train_50k)
else
    SCALES=("$@")
fi

echo "==> Bridge verification sweep"
echo "    Checkpoint root: ${CHECKPOINT_ROOT}"
echo "    Output root    : ${OUT_ROOT}"
echo "    Scales         : ${SCALES[*]}"
echo

OK=0
FAIL=0
for SCALE in "${SCALES[@]}"; do
    echo "=== scale ${SCALE} ==="
    set +e
    uv run python scripts/verify_bridges.py \
        --checkpoint-root "${CHECKPOINT_ROOT}" \
        --scale "${SCALE}" \
        --out "${OUT_ROOT}"
    STATUS=$?
    set -e
    if [ "${STATUS}" -eq 0 ]; then
        OK=$((OK + 1))
    else
        FAIL=$((FAIL + 1))
    fi
    echo
done

echo "==> Summary: OK=${OK}  FAIL=${FAIL}  (across ${#SCALES[@]} scales)"
[ "${FAIL}" -eq 0 ]
