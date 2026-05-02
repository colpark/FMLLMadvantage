#!/usr/bin/env bash
#
# train_fm_sweep.sh
#
# Trains FM1, FM2, FM3 in parallel across GPUs 0, 1, 2 for one or more
# training scales. The script powers the E5 FM-quality sweep: it
# iterates over the requested scales (default: train_10k, train_30k,
# train_50k) and waits for all three FMs to finish each scale before
# moving to the next.
#
# Usage:
#   bash scripts/train_fm_sweep.sh                       # all three scales
#   bash scripts/train_fm_sweep.sh train_50k             # one scale
#   bash scripts/train_fm_sweep.sh train_10k train_50k   # two scales
#
# Environment variables (optional):
#   CONFIG       (default: configs/default.yaml)
#   H5_PATH      (default: data/synthetic_lj_v1/specimens.h5)
#   SPLITS_PATH  (default: data/synthetic_lj_v1/splits.yaml)
#   LOG_DIR      (default: runs)
#   FM1_GPU      (default: 0)
#   FM2_GPU      (default: 1)
#   FM3_GPU      (default: 2)
#   EPOCHS       (default: empty -> use config's epochs field)
#
# Example with overrides:
#   EPOCHS=20 LOG_DIR=runs/sweep1 bash scripts/train_fm_sweep.sh train_10k

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-configs/default.yaml}"
H5_PATH="${H5_PATH:-data/synthetic_lj_v1/specimens.h5}"
SPLITS_PATH="${SPLITS_PATH:-data/synthetic_lj_v1/splits.yaml}"
LOG_DIR="${LOG_DIR:-runs}"
FM1_GPU="${FM1_GPU:-0}"
FM2_GPU="${FM2_GPU:-1}"
FM3_GPU="${FM3_GPU:-2}"
EPOCHS="${EPOCHS:-}"

if [ "$#" -eq 0 ]; then
    SCALES=(train_10k train_30k train_50k)
else
    SCALES=("$@")
fi

mkdir -p "${LOG_DIR}"

# Compose extra flags shared across the three FMs.
EXTRA_ARGS=(
    --config "${CONFIG}"
    --h5-path "${H5_PATH}"
    --splits-path "${SPLITS_PATH}"
)
if [ -n "${EPOCHS}" ]; then
    EXTRA_ARGS+=(--epochs "${EPOCHS}")
fi

run_one() {
    local fm="$1"
    local gpu="$2"
    local scale="$3"
    local log="${LOG_DIR}/${fm}-${scale}.log"
    CUDA_VISIBLE_DEVICES="${gpu}" uv run python scripts/train_fm.py \
        --fm "${fm}" \
        --train-split "${scale}" \
        "${EXTRA_ARGS[@]}" \
        > "${log}" 2>&1
}

echo "==> FM training sweep"
echo "    Scales        : ${SCALES[*]}"
echo "    Config        : ${CONFIG}"
echo "    HDF5          : ${H5_PATH}"
echo "    Splits        : ${SPLITS_PATH}"
echo "    Log directory : ${LOG_DIR}"
echo "    GPU layout    : fm1=${FM1_GPU} fm2=${FM2_GPU} fm3=${FM3_GPU}"
echo "    Epoch override: ${EPOCHS:-from config}"
echo

for SCALE in "${SCALES[@]}"; do
    printf '=== Scale %-12s start  %s\n' "${SCALE}" "$(date '+%Y-%m-%d %H:%M:%S')"
    run_one fm1 "${FM1_GPU}" "${SCALE}" &
    PID1=$!
    run_one fm2 "${FM2_GPU}" "${SCALE}" &
    PID2=$!
    run_one fm3 "${FM3_GPU}" "${SCALE}" &
    PID3=$!

    # Wait for each PID and collect a per-FM exit code so a single failure
    # does not abort the others mid-stream.
    set +e
    wait "${PID1}"; STATUS1=$?
    wait "${PID2}"; STATUS2=$?
    wait "${PID3}"; STATUS3=$?
    set -e

    printf '=== Scale %-12s done   %s  fm1=%d fm2=%d fm3=%d\n' \
        "${SCALE}" "$(date '+%Y-%m-%d %H:%M:%S')" \
        "${STATUS1}" "${STATUS2}" "${STATUS3}"

    if [ "${STATUS1}" -ne 0 ] || [ "${STATUS2}" -ne 0 ] || [ "${STATUS3}" -ne 0 ]; then
        echo "ERROR: at least one FM failed at scale ${SCALE}. See ${LOG_DIR}/*-${SCALE}.log."
        exit 1
    fi
done

echo
echo "==> All scales finished."
echo "    Logs       : ${LOG_DIR}/<fm>-<scale>.log"
echo "    Checkpoints: checkpoints/<fm_name>/<scale>/<run_id>/"
