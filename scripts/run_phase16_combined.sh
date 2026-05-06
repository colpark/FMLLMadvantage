#!/usr/bin/env bash
#
# run_phase16_combined.sh
#
# Single bash driver for the two adapter+verifier experiments that
# combine the Phase 16 cot_sft_sae adapter with Pipeline A.
#
# Two experiments launched in parallel on separate GPUs:
#
#   E1 = full + adapter
#        Uses the *vanilla* OHVD loop (DEFAULT_SYSTEM_PROMPT, plain
#        user query, no probes/SAE in the prompt) with the
#        cot_sft_sae adapter loaded on top of Qwen.
#        Tests the *format-mismatched* combination -- the adapter
#        was trained on probes+SAE prompts, not on raw OHVD
#        prompts. Predicts whether the OHVD loop tolerates an
#        adapter that has never seen its turn-by-turn input format.
#        Output: runs/holdout_combined/full/<run_id>/
#        Column name (after merge): full_with_sft_sae_adapter
#
#   E2 = full_sae + adapter
#        Uses run_baseline_full_probes.py which injects probes +
#        labelled SAE features into the user message DURING the OHVD
#        loop. The adapter sees prompts very close to its training
#        distribution.
#        Tests the *format-aligned* combination -- the cleaner
#        question of whether training-time supervised CoT-SFT
#        and inference-time multi-source verification are
#        genuinely additive.
#        Output: runs/holdout_adapter_sae/full_sae/<run_id>/
#        Column name (after merge): full_sae_with_sft_adapter
#
# Distinct output roots prevent collision with the existing
# runs/holdout/full/ and runs/holdout/full_sae/ baselines and
# protect against the resume logic skipping all 200 specimens
# because they were already done in another run.
#
# Usage:
#   bash scripts/run_phase16_combined.sh              # launch both runs
#   bash scripts/run_phase16_combined.sh evaluate     # after they finish:
#                                                       merge into runs/holdout/
#                                                       and run the side-by-side
#   bash scripts/run_phase16_combined.sh status       # check progress
#
# Environment variables (optional):
#   GPU_E1               (default: 0)
#   GPU_E2               (default: 1)
#   PROGRESS_EVERY       (default: 5)
#   ADAPTER_PATH         (default: latest under checkpoints/cot-sft-sae/)
#   SAE_DIR              (default: latest under checkpoints/sae/)
#   SAE_LABELS_PATH      (default: latest labels.json under runs/sae_labels/)
#   SPECIMEN_IDS_FILE    (default: runs/holdout_lock/ids.json)
#   OUT_E1               (default: runs/holdout_combined)
#   OUT_E2               (default: runs/holdout_adapter_sae)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

GPU_E1="${GPU_E1:-0}"
GPU_E2="${GPU_E2:-1}"
PROGRESS_EVERY="${PROGRESS_EVERY:-5}"
SPECIMEN_IDS_FILE="${SPECIMEN_IDS_FILE:-runs/holdout_lock/ids.json}"
OUT_E1="${OUT_E1:-runs/holdout_combined}"
OUT_E2="${OUT_E2:-runs/holdout_adapter_sae}"

LOG_E1="/tmp/e1_full_with_adapter.log"
LOG_E2="/tmp/e2_full_sae_with_adapter.log"

MODE="${1:-launch}"

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

resolve_adapter() {
    if [ -n "${ADAPTER_PATH:-}" ] && [ -d "${ADAPTER_PATH}" ]; then
        echo "${ADAPTER_PATH}"
        return
    fi
    local cand
    cand="$(ls -td checkpoints/cot-sft-sae/*/adapter 2>/dev/null | head -1 || true)"
    if [ -z "${cand}" ] || [ ! -d "${cand}" ]; then
        echo "" ; return
    fi
    echo "${cand}"
}

resolve_sae() {
    if [ -n "${SAE_DIR:-}" ] && [ -d "${SAE_DIR}" ]; then
        echo "${SAE_DIR}" ; return
    fi
    local cand
    cand="$(ls -td checkpoints/sae/*/ 2>/dev/null | head -1 || true)"
    [ -z "${cand}" ] && { echo "" ; return ; }
    echo "${cand%/}"
}

resolve_labels() {
    if [ -n "${SAE_LABELS_PATH:-}" ] && [ -f "${SAE_LABELS_PATH}" ]; then
        echo "${SAE_LABELS_PATH}" ; return
    fi
    local cand
    cand="$(ls -td runs/sae_labels/*/labels.json 2>/dev/null | head -1 || true)"
    [ -z "${cand}" ] && { echo "" ; return ; }
    echo "${cand}"
}

verify_prereqs() {
    local adapter="$1" sae="$2" labels="$3"
    if [ -z "${adapter}" ]; then
        echo "ERROR: no Phase 16 adapter under checkpoints/cot-sft-sae/." >&2
        echo "       Run scripts/train_cot_sft_with_sae.sh first." >&2
        return 2
    fi
    if [ -z "${sae}" ]; then
        echo "ERROR: no SAE under checkpoints/sae/." >&2
        return 2
    fi
    if [ -z "${labels}" ]; then
        echo "ERROR: no labels.json under runs/sae_labels/." >&2
        return 2
    fi
    if [ ! -f "${SPECIMEN_IDS_FILE}" ]; then
        echo "ERROR: SPECIMEN_IDS_FILE not found: ${SPECIMEN_IDS_FILE}" >&2
        return 2
    fi
    if ! ls -d checkpoints/probes/*/ > /dev/null 2>&1; then
        echo "ERROR: no probe bank under checkpoints/probes/." >&2
        return 2
    fi
    if ! ls -d checkpoints/fm2_rdf/*/*/ > /dev/null 2>&1; then
        echo "ERROR: no FM2 checkpoint under checkpoints/fm2_rdf/." >&2
        return 2
    fi
    return 0
}

count_lines() {
    local f="$1"
    [ -f "${f}" ] || { echo 0 ; return ; }
    wc -l < "${f}" | tr -d ' '
}

latest_traj_under() {
    # Find the latest trajectories.jsonl under <root>/<subdir>/.
    local root="$1" sub="$2"
    ls -td "${root}/${sub}/"*/trajectories.jsonl 2>/dev/null | head -1 || true
}

print_header() {
    echo "==========================================================="
    echo "$1"
    echo "==========================================================="
}

# ----------------------------------------------------------------------
# Mode: status
# ----------------------------------------------------------------------

if [ "${MODE}" = "status" ]; then
    print_header "Phase 16 combined runs -- status"

    echo "--- Processes ---"
    ps -ef | grep -E "run_baseline\.py|run_baseline_full_probes\.py" | grep -v grep || echo "  (none)"
    echo

    echo "--- E1 (full + adapter) -- ${OUT_E1} ---"
    e1_traj="$(latest_traj_under "${OUT_E1}" full)"
    if [ -n "${e1_traj}" ]; then
        echo "  trajectories: ${e1_traj}"
        echo "  lines       : $(count_lines "${e1_traj}") / 200"
        echo "  last log    : $(tail -3 "${LOG_E1}" 2>/dev/null | head -1)"
    else
        echo "  (no trajectories yet at ${OUT_E1}/full/)"
    fi
    echo

    echo "--- E2 (full_sae + adapter) -- ${OUT_E2} ---"
    e2_traj="$(latest_traj_under "${OUT_E2}" full_sae)"
    if [ -n "${e2_traj}" ]; then
        echo "  trajectories: ${e2_traj}"
        echo "  lines       : $(count_lines "${e2_traj}") / 200"
        echo "  last log    : $(tail -3 "${LOG_E2}" 2>/dev/null | head -1)"
    else
        echo "  (no trajectories yet at ${OUT_E2}/full_sae/)"
    fi
    exit 0
fi

# ----------------------------------------------------------------------
# Mode: evaluate (merge into runs/holdout/ and run side-by-side)
# ----------------------------------------------------------------------

if [ "${MODE}" = "evaluate" ]; then
    print_header "Phase 16 combined runs -- merge + evaluate"

    e1_traj="$(latest_traj_under "${OUT_E1}" full)"
    e2_traj="$(latest_traj_under "${OUT_E2}" full_sae)"

    e1_ok=0
    e2_ok=0
    if [ -n "${e1_traj}" ] && [ "$(count_lines "${e1_traj}")" -ge 200 ]; then
        e1_ok=1
    fi
    if [ -n "${e2_traj}" ] && [ "$(count_lines "${e2_traj}")" -ge 200 ]; then
        e2_ok=1
    fi

    echo "E1 status: $([ ${e1_ok} = 1 ] && echo 'COMPLETE' || echo "INCOMPLETE ($(count_lines "${e1_traj:-/dev/null}")/200)")"
    echo "E2 status: $([ ${e2_ok} = 1 ] && echo 'COMPLETE' || echo "INCOMPLETE ($(count_lines "${e2_traj:-/dev/null}")/200)")"
    echo

    if [ "${e1_ok}" = "1" ]; then
        e1_run_dir="$(dirname "${e1_traj}")"
        e1_dest="runs/holdout/full_with_sft_sae_adapter/$(basename "${e1_run_dir}")"
        if [ ! -d "${e1_dest}" ]; then
            echo "Copying E1 -> ${e1_dest}"
            mkdir -p "$(dirname "${e1_dest}")"
            cp -r "${e1_run_dir}" "${e1_dest}"
        else
            echo "E1 already merged at ${e1_dest}"
        fi
    fi

    if [ "${e2_ok}" = "1" ]; then
        e2_run_dir="$(dirname "${e2_traj}")"
        e2_dest="runs/holdout/full_sae_with_sft_adapter/$(basename "${e2_run_dir}")"
        if [ ! -d "${e2_dest}" ]; then
            echo "Copying E2 -> ${e2_dest}"
            mkdir -p "$(dirname "${e2_dest}")"
            cp -r "${e2_run_dir}" "${e2_dest}"
        else
            echo "E2 already merged at ${e2_dest}"
        fi
    fi

    echo
    print_header "Side-by-side comparison (BASELINES_ROOT=runs/holdout)"
    BASELINES_ROOT=runs/holdout bash scripts/evaluate_baselines.sh
    exit 0
fi

# ----------------------------------------------------------------------
# Mode: launch (default)
# ----------------------------------------------------------------------

print_header "Phase 16 combined runs -- launching"

ADAPTER="$(resolve_adapter)"
SAE="$(resolve_sae)"
LABELS="$(resolve_labels)"

verify_prereqs "${ADAPTER}" "${SAE}" "${LABELS}"

echo "Adapter         : ${ADAPTER}"
echo "SAE dir         : ${SAE}"
echo "SAE labels      : ${LABELS}"
echo "SPECIMEN_IDS_FILE: ${SPECIMEN_IDS_FILE}"
echo "OUT_E1          : ${OUT_E1}"
echo "OUT_E2          : ${OUT_E2}"
echo "GPU_E1 / GPU_E2 : ${GPU_E1} / ${GPU_E2}"
echo

# Kill any zombie processes from prior attempts on the same scripts
echo "Killing any prior run_baseline.py / run_baseline_full_probes.py processes..."
pkill -f "scripts/run_baseline\.py" 2>/dev/null || true
pkill -f "scripts/run_baseline_full_probes\.py" 2>/dev/null || true
sleep 2

# Launch E1: full + adapter (vanilla OHVD loop, no probes/SAE in prompt)
echo
echo "==> Launching E1 (full + adapter) on GPU ${GPU_E1}"
ADAPTER_PATH="${ADAPTER}" SPECIMEN_IDS_FILE="${SPECIMEN_IDS_FILE}" OUT="${OUT_E1}" PROGRESS_EVERY="${PROGRESS_EVERY}" GPU="${GPU_E1}" nohup bash scripts/run_baseline.sh full > "${LOG_E1}" 2>&1 &
E1_PID=$!
echo "    PID: ${E1_PID}   log: ${LOG_E1}"

# Brief pause so the two processes' python startup banners don't
# interleave in their own logs (cosmetic, not functional).
sleep 3

# Launch E2: full_sae + adapter (probe-augmented Pipeline A with SAE
# in prompt, format-aligned with the adapter's training distribution)
echo
echo "==> Launching E2 (full_sae + adapter) on GPU ${GPU_E2}"
SAE_DIR="${SAE}" SAE_LABELS_PATH="${LABELS}" ADAPTER_PATH="${ADAPTER}" SPECIMEN_IDS_FILE="${SPECIMEN_IDS_FILE}" OUT_ROOT="${OUT_E2}" GPU="${GPU_E2}" nohup bash scripts/run_baseline_full_probes.sh > "${LOG_E2}" 2>&1 &
E2_PID=$!
echo "    PID: ${E2_PID}   log: ${LOG_E2}"

echo
print_header "Both runs launched. Monitor with:"
echo "  tail -f ${LOG_E1}"
echo "  tail -f ${LOG_E2}"
echo
echo "  bash scripts/run_phase16_combined.sh status"
echo
echo "When both complete (~30-50 min each), merge + evaluate with:"
echo "  bash scripts/run_phase16_combined.sh evaluate"
echo
echo "Or wait for both PIDs in foreground:"
echo "  wait ${E1_PID} ${E2_PID}"
