#!/usr/bin/env bash
#
# verify_sweep.sh
#
# Sanity-check the output of train_fm_sweep.sh:
#   1. Confirm each per-FM log ends with "Training done" + "Probe report".
#   2. List every manifest and probe_report under checkpoints/.
#   3. Print probe satisfaction scores across all scales for at-a-glance
#      comparison.
#
# Usage:
#   bash scripts/verify_sweep.sh                       # all three scales
#   bash scripts/verify_sweep.sh train_50k             # one scale
#   bash scripts/verify_sweep.sh train_10k train_50k   # two scales
#
# Environment variables (optional):
#   LOG_DIR         (default: runs)
#   CHECKPOINT_ROOT (default: checkpoints)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

LOG_DIR="${LOG_DIR:-runs}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints}"

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

echo "=========================================================="
echo "  Step 1. Per-log completion check"
echo "=========================================================="
ALL_OK=1
for FM in "${FMS[@]}"; do
    for SCALE in "${SCALES[@]}"; do
        LOG="${LOG_DIR}/${FM}-${SCALE}.log"
        if [ ! -f "${LOG}" ]; then
            printf '  %-3s %-10s : MISSING LOG (%s)\n' "${FM}" "${SCALE}" "${LOG}"
            ALL_OK=0
            continue
        fi
        DONE=$(grep -E "Training done in" "${LOG}" | tail -1 || true)
        PROBE=$(grep -E "Probe report saved" "${LOG}" | tail -1 || true)
        if [ -z "${DONE}" ] || [ -z "${PROBE}" ]; then
            printf '  %-3s %-10s : INCOMPLETE\n' "${FM}" "${SCALE}"
            ALL_OK=0
        else
            DONE_TIME=$(echo "${DONE}" | grep -oE "[0-9]+\.[0-9]+s" | head -1)
            printf '  %-3s %-10s : OK  (%s)\n' "${FM}" "${SCALE}" "${DONE_TIME:-done}"
        fi
    done
done
echo

echo "=========================================================="
echo "  Step 2. Artifacts on disk"
echo "=========================================================="
N_MANIFESTS=$(find "${CHECKPOINT_ROOT}" -name 'manifest.yaml' 2>/dev/null | wc -l | tr -d ' ')
N_PROBES=$(find "${CHECKPOINT_ROOT}" -name 'probe_report.yaml' 2>/dev/null | wc -l | tr -d ' ')
N_MODELS=$(find "${CHECKPOINT_ROOT}" -name 'model.pt' 2>/dev/null | wc -l | tr -d ' ')
N_CALIBS=$(find "${CHECKPOINT_ROOT}" -name 'calibration.json' 2>/dev/null | wc -l | tr -d ' ')
echo "  model.pt           : ${N_MODELS}"
echo "  manifest.yaml      : ${N_MANIFESTS}"
echo "  probe_report.yaml  : ${N_PROBES}"
echo "  calibration.json   : ${N_CALIBS}"
echo
echo "  Manifests:"
find "${CHECKPOINT_ROOT}" -name 'manifest.yaml' 2>/dev/null | sort | sed 's/^/    /'
echo

echo "=========================================================="
echo "  Step 3. Probe satisfaction scores by scale"
echo "=========================================================="
for FM in "${FMS[@]}"; do
    FMDIR="${FM_TO_DIR[$FM]}"
    echo "  === ${FMDIR} ==="
    for SCALE in "${SCALES[@]}"; do
        RPT=$(ls -t "${CHECKPOINT_ROOT}/${FMDIR}/${SCALE}"/*/probe_report.yaml 2>/dev/null | head -1)
        if [ -z "${RPT}" ]; then
            printf '    %-12s : missing\n' "${SCALE}"
            continue
        fi
        SCORES=$(uv run python - <<PYEOF
import yaml
with open("${RPT}") as f:
    r = yaml.safe_load(f)
parts = []
for res in r["results"]:
    short = res["constraint_name"].split("_")[0]
    flag = "ok" if res["passes_threshold"] else "FAIL"
    parts.append(f"{short}={res['satisfaction_score']:.2f}/{flag}")
print(" ".join(parts))
PYEOF
)
        printf '    %-12s : %s\n' "${SCALE}" "${SCORES}"
    done
done
echo

echo "=========================================================="
echo "  Step 4. Final-epoch val metrics from training logs"
echo "=========================================================="
for FM in "${FMS[@]}"; do
    for SCALE in "${SCALES[@]}"; do
        LOG="${LOG_DIR}/${FM}-${SCALE}.log"
        if [ ! -f "${LOG}" ]; then
            printf '  %-3s @ %-12s : (no log)\n' "${FM}" "${SCALE}"
            continue
        fi
        # Last "val:" line in the log carries the final-epoch metrics.
        LAST_VAL=$(grep -E '\bval:' "${LOG}" 2>/dev/null | tail -1 | sed -E 's/^.*val:[[:space:]]*/val: /')
        if [ -z "${LAST_VAL}" ]; then
            printf '  %-3s @ %-12s : (no val line)\n' "${FM}" "${SCALE}"
        else
            printf '  %-3s @ %-12s : %s\n' "${FM}" "${SCALE}" "${LAST_VAL}"
        fi
    done
done
echo

if [ "${ALL_OK}" -eq 1 ]; then
    echo "=========================================================="
    echo "  Verification: PASS"
    echo "=========================================================="
    exit 0
else
    echo "=========================================================="
    echo "  Verification: SOME RUNS DID NOT COMPLETE"
    echo "=========================================================="
    exit 1
fi
