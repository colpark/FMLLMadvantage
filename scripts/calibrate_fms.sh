#!/usr/bin/env bash
#
# calibrate_fms.sh
#
# Stage 3 of the post-training pipeline: split-conformal calibration
# for FM1, FM2, FM3 across one or more training scales. The script
# locates the latest model.pt under
# checkpoints/<fm_dir>/<scale>/<run_id>/ for each (fm, scale) pair and
# invokes scripts/train_fm.py --calibrate-only against it. Output is
# calibration.json next to the checkpoint.
#
# Usage:
#   bash scripts/calibrate_fms.sh                       # all three scales
#   bash scripts/calibrate_fms.sh train_50k             # one scale
#   bash scripts/calibrate_fms.sh train_10k train_30k   # two scales
#
# Environment variables (optional):
#   CONFIG          (default: configs/default.yaml)
#   H5_PATH         (default: data/synthetic_lj_v1/specimens.h5)
#   SPLITS_PATH     (default: data/synthetic_lj_v1/splits.yaml)
#   CHECKPOINT_ROOT (default: checkpoints)
#   GPU             (default: 0)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-configs/default.yaml}"
H5_PATH="${H5_PATH:-data/synthetic_lj_v1/specimens.h5}"
SPLITS_PATH="${SPLITS_PATH:-data/synthetic_lj_v1/splits.yaml}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints}"
GPU="${GPU:-0}"

if [ "$#" -eq 0 ]; then
    SCALES=(train_10k train_30k train_50k)
else
    SCALES=("$@")
fi

FMS=(fm1 fm2 fm3)
declare -A FM_TO_DIR=(
    ["fm1"]="fm1_image"
    ["fm2"]="fm2_rdf"
    ["fm3"]="fm3_traj"
)

echo "==> Conformal calibration sweep"
echo "    Scales         : ${SCALES[*]}"
echo "    Config         : ${CONFIG}"
echo "    HDF5           : ${H5_PATH}"
echo "    Splits         : ${SPLITS_PATH}"
echo "    Checkpoint root: ${CHECKPOINT_ROOT}"
echo "    GPU            : ${GPU}"
echo

OK=0
FAIL=0
MISSING=0

for SCALE in "${SCALES[@]}"; do
    for FM in "${FMS[@]}"; do
        FMDIR="${FM_TO_DIR[$FM]}"
        # Pick the latest run directory under <fm_dir>/<scale>/.
        CKPT=$(ls -t "${CHECKPOINT_ROOT}/${FMDIR}/${SCALE}"/*/model.pt 2>/dev/null | head -1 || true)
        if [ -z "${CKPT}" ]; then
            printf '%-3s @ %-12s : MISSING (no checkpoint under %s/%s/%s)\n' \
                "${FM}" "${SCALE}" "${CHECKPOINT_ROOT}" "${FMDIR}" "${SCALE}"
            MISSING=$((MISSING + 1))
            continue
        fi

        printf '%-3s @ %-12s : calibrating  %s\n' "${FM}" "${SCALE}" "${CKPT}"
        set +e
        CUDA_VISIBLE_DEVICES="${GPU}" uv run python scripts/train_fm.py \
            --fm "${FM}" \
            --calibrate-only \
            --checkpoint "${CKPT}" \
            --config "${CONFIG}" \
            --h5-path "${H5_PATH}" \
            --splits-path "${SPLITS_PATH}"
        STATUS=$?
        set -e
        if [ "${STATUS}" -eq 0 ]; then
            CAL_JSON="$(dirname "${CKPT}")/calibration.json"
            if [ -f "${CAL_JSON}" ]; then
                printf '%-3s @ %-12s : OK           %s\n' "${FM}" "${SCALE}" "${CAL_JSON}"
                OK=$((OK + 1))
            else
                printf '%-3s @ %-12s : NO_OUTPUT    (script succeeded but calibration.json missing)\n' \
                    "${FM}" "${SCALE}"
                FAIL=$((FAIL + 1))
            fi
        else
            printf '%-3s @ %-12s : FAILED (exit %d)\n' "${FM}" "${SCALE}" "${STATUS}"
            FAIL=$((FAIL + 1))
        fi
    done
done

echo
echo "==> Summary: OK=${OK}  FAIL=${FAIL}  MISSING=${MISSING}"

if [ "${FAIL}" -gt 0 ] || [ "${MISSING}" -gt 0 ]; then
    exit 1
fi
exit 0
