#!/usr/bin/env bash
#
# run_phase13_full_sae.sh
#
# One-shot wrapper that drives the entire Phase 13 pipeline:
#   Stage 0  train_sae.sh                  -> checkpoints/sae/<run_id>/sae.pt
#   Stage 1  label_sae_features.sh         -> runs/sae_labels/<run_id>/labels.json
#   Stage 2  run_baseline_full_probes.sh   -> runs/holdout/full_sae/<run_id>/
#                                              (with SAE_DIR exported)
#
# Each stage is skipped if its output already exists, so this script
# is safe to rerun. The third stage is the one that's been silently
# falling back to full_probes/ when SAE_DIR was forgotten -- this
# wrapper exports it explicitly and refuses to call the runner if the
# SAE artefacts are not present.
#
# Usage:
#   bash scripts/run_phase13_full_sae.sh
#   SPECIMEN_IDS_FILE=runs/holdout_lock/ids.json \
#       bash scripts/run_phase13_full_sae.sh
#   FORCE_RETRAIN_SAE=1 bash scripts/run_phase13_full_sae.sh
#   FORCE_RELABEL=1     bash scripts/run_phase13_full_sae.sh
#
# Environment variables (optional):
#   SPECIMEN_IDS_FILE   propagated to Stage 2 (default: unset)
#   FORCE_RETRAIN_SAE   set to 1 to retrain even if checkpoints/sae/ exists
#   FORCE_RELABEL       set to 1 to relabel even if runs/sae_labels/ exists
#   GPU                 default 0; propagated to all stages

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

GPU="${GPU:-0}"
export GPU

echo "==================================================================="
echo "Phase 13 pipeline: SAE -> labels -> full_sae baseline"
echo "==================================================================="
echo "  Repo root        : ${REPO_ROOT}"
echo "  GPU              : ${GPU}"
echo "  SPECIMEN_IDS_FILE: ${SPECIMEN_IDS_FILE:-(unset)}"
echo "  FORCE_RETRAIN_SAE: ${FORCE_RETRAIN_SAE:-0}"
echo "  FORCE_RELABEL    : ${FORCE_RELABEL:-0}"
echo

# ---------------------------------------------------------------------
# Prerequisite check: the SAE pipeline depends on a trained FM2 and
# (for Stage 2) a trained probe bank. Both should already exist from
# Phases 2 and 11. Fail loudly if not.
# ---------------------------------------------------------------------
echo "==> Checking prerequisites"

FM2_CKPT_PARENT="checkpoints/fm2_rdf"
if ! compgen -G "${FM2_CKPT_PARENT}/*/*/model.pt" > /dev/null 2>&1; then
    echo "ERROR: no FM2 checkpoint under ${FM2_CKPT_PARENT}/<train_split>/<run_id>/" >&2
    echo "       Phase 2 (FM training) must complete before Phase 13." >&2
    exit 2
fi
echo "    FM2          : OK ($(ls -d ${FM2_CKPT_PARENT}/*/*/ | tail -1))"

PROBE_BANK_PARENT="checkpoints/probes"
if ! compgen -G "${PROBE_BANK_PARENT}/*" > /dev/null 2>&1; then
    echo "ERROR: no probe bank under ${PROBE_BANK_PARENT}/" >&2
    echo "       Run scripts/train_probe_bank.sh first." >&2
    exit 2
fi
echo "    Probe bank   : OK ($(ls -td ${PROBE_BANK_PARENT}/*/ | head -1))"
echo

# ---------------------------------------------------------------------
# Stage 0 -- train the SAE if missing.
# ---------------------------------------------------------------------
echo "==> Stage 0: SAE training"
if [ "${FORCE_RETRAIN_SAE:-0}" = "1" ] || ! compgen -G "checkpoints/sae/*/sae.pt" > /dev/null 2>&1; then
    echo "    Training new SAE..."
    bash scripts/train_sae.sh
else
    SAE_LATEST="$(ls -td checkpoints/sae/*/ | head -1)"
    echo "    SAE already exists at ${SAE_LATEST} (FORCE_RETRAIN_SAE=1 to override)."
fi

SAE_DIR_RESOLVED="$(ls -td checkpoints/sae/*/ 2>/dev/null | head -1)"
if [ -z "${SAE_DIR_RESOLVED}" ] || [ ! -f "${SAE_DIR_RESOLVED%/}/sae.pt" ]; then
    echo "ERROR: Stage 0 produced no usable SAE under checkpoints/sae/" >&2
    exit 3
fi
SAE_DIR_RESOLVED="${SAE_DIR_RESOLVED%/}"
echo "    Resolved SAE : ${SAE_DIR_RESOLVED}"
echo

# ---------------------------------------------------------------------
# Stage 1 -- label SAE features if missing.
# ---------------------------------------------------------------------
echo "==> Stage 1: SAE feature labelling"
if [ "${FORCE_RELABEL:-0}" = "1" ] || ! compgen -G "runs/sae_labels/*/labels.json" > /dev/null 2>&1; then
    echo "    Labelling features..."
    bash scripts/label_sae_features.sh
else
    LABELS_LATEST="$(ls -td runs/sae_labels/*/ | head -1)"
    echo "    Labels already exist at ${LABELS_LATEST} (FORCE_RELABEL=1 to override)."
fi

SAE_LABELS_RESOLVED="$(ls -td runs/sae_labels/*/labels.json 2>/dev/null | head -1)"
if [ -z "${SAE_LABELS_RESOLVED}" ] || [ ! -f "${SAE_LABELS_RESOLVED}" ]; then
    echo "ERROR: Stage 1 produced no labels.json under runs/sae_labels/" >&2
    exit 4
fi
echo "    Resolved labels: ${SAE_LABELS_RESOLVED}"
echo

# ---------------------------------------------------------------------
# Stage 2 -- run probe-augmented Pipeline A WITH the SAE flag set.
# This is the step that has been silently producing full_probes/
# instead of full_sae/ when SAE_DIR is forgotten.
# ---------------------------------------------------------------------
echo "==> Stage 2: full_sae baseline run (Pipeline A + probes + SAE features)"

# Ensure no stale full_sae run from a prior aborted attempt is half
# present. We don't delete it; we just report it so the resume logic
# inside the runner picks it up cleanly.
if [ -d "runs/holdout/full_sae" ]; then
    EXISTING_RUNS="$(ls -d runs/holdout/full_sae/*/ 2>/dev/null | wc -l | tr -d ' ')"
    if [ "${EXISTING_RUNS}" != "0" ]; then
        echo "    Note: ${EXISTING_RUNS} existing full_sae run(s) detected;"
        echo "          the runner's resume logic will pick the latest."
    fi
fi

export SAE_DIR="${SAE_DIR_RESOLVED}"
export SAE_LABELS_PATH="${SAE_LABELS_RESOLVED}"
echo "    Exporting SAE_DIR        = ${SAE_DIR}"
echo "    Exporting SAE_LABELS_PATH= ${SAE_LABELS_PATH}"

bash scripts/run_baseline_full_probes.sh

# ---------------------------------------------------------------------
# Verify Stage 2 actually wrote into full_sae/, not full_probes/.
# ---------------------------------------------------------------------
echo
echo "==> Post-run verification"
if ! compgen -G "runs/holdout/full_sae/*/trajectories.jsonl" > /dev/null 2>&1; then
    echo "ERROR: no trajectories.jsonl under runs/holdout/full_sae/." >&2
    echo "       This usually means SAE_DIR was lost between Stage 2's caller" >&2
    echo "       and the python runner. Check runs/holdout/full_probes/ for a" >&2
    echo "       freshly-modified directory -- if it is fresher than full_sae/," >&2
    echo "       SAE_DIR did not propagate." >&2
    exit 5
fi
LATEST_FULL_SAE="$(ls -td runs/holdout/full_sae/*/ | head -1)"
N_TRAJ="$(wc -l < "${LATEST_FULL_SAE}/trajectories.jsonl" | tr -d ' ')"
echo "    full_sae run : ${LATEST_FULL_SAE}"
echo "    trajectories : ${N_TRAJ}"

# Spot-check the metadata of the first trajectory to confirm the SAE
# branch was taken (sae_dir is recorded in metadata when SAE was used).
uv run python - <<PY
import json
from pathlib import Path
traj = Path("${LATEST_FULL_SAE}") / "trajectories.jsonl"
with traj.open() as f:
    first = json.loads(f.readline())
md = first.get("metadata", {}) or {}
print(f"    metadata.sae_dir       : {md.get('sae_dir')}")
print(f"    metadata.probe_bank_dir: {md.get('probe_bank_dir')}")
PY

echo
echo "==> Done. Re-run scripts/evaluate_baselines.sh to update the"
echo "    side-by-side comparison; full_sae will appear as a column."
