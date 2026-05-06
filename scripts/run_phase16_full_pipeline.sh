#!/usr/bin/env bash
#
# run_phase16_full_pipeline.sh
#
# End-to-end Phase 16 driver: retrain the SAE-augmented CoT-SFT
# adapter with sufficient sequence budget, run the probe-head
# reference baseline, run cot_sft_sae single-shot inference, and
# regenerate the side-by-side comparison.
#
# Bundles the whole sequence so a single nohup invocation runs
# everything end-to-end without copy-pasting individual commands.
#
# Stages (each is skipped if its output already exists, unless
# FRESH=1):
#
#   0. Optionally clean prior broken artefacts (FRESH=1).
#   1. Build SAE-augmented CoT dataset (skipped if one exists, or
#      forced with REBUILD_DATASET=1).
#   2. SFT train with MAX_SEQ=1536 on 4 GPUs (DDP). Single adapter
#      output thanks to the broadcast-run-id fix in commit 9f6e5b8.
#   3. Run probe_head reference baseline (no LLM, no verifier).
#   4. Run cot_sft_sae single-shot inference (bf16, no verifier).
#   5. Run evaluate_baselines.sh to refresh the side-by-side.
#
# Usage:
#   bash scripts/run_phase16_full_pipeline.sh
#   FRESH=1 bash scripts/run_phase16_full_pipeline.sh        # nuke prior outputs
#   REBUILD_DATASET=1 bash scripts/run_phase16_full_pipeline.sh
#
# Suggested launch (detaches from terminal so container recycle
# does not kill it; resume support inside the inference and
# training stages keeps partial work):
#
#   nohup bash scripts/run_phase16_full_pipeline.sh > /tmp/phase16.log 2>&1 &
#   tail -f /tmp/phase16.log
#
# Environment variables (optional):
#   FRESH              1 => rm -rf prior cot-sft-sae checkpoints,
#                          cot_datasets_sae runs (when REBUILD_DATASET=1),
#                          and runs/holdout/cot_sft_sae partials. (default 0)
#   REBUILD_DATASET    1 => rerun build_cot_dataset_with_sae even if
#                          a records.jsonl exists. (default 0)
#   N_SPECIMENS        Stage 1 dataset size. (default 10000)
#   TOP_K_FEATURES     Stage 1 / Stage 4 SAE features per row. (default 8)
#   NUM_GPUS           Stage 2 DDP rank count. (default 4)
#   PER_DEVICE_BS      Stage 2 per-device batch size. (default 2)
#   GRAD_ACCUM         Stage 2 grad accum. (default 4; effective batch
#                      = PER_DEVICE_BS * GRAD_ACCUM * NUM_GPUS = 32)
#   MAX_SEQ            Stage 2 max sequence length. (default 1536;
#                      MUST be > longest record token count or the
#                      Final commit gets truncated and inference fails)
#   EPOCHS             Stage 2 training epochs. (default 3)
#   QUANTIZE           Stage 4 inference quantization. (default 'none'
#                      = bf16; matches training precision)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

FRESH="${FRESH:-0}"
REBUILD_DATASET="${REBUILD_DATASET:-0}"
N_SPECIMENS="${N_SPECIMENS:-10000}"
TOP_K_FEATURES="${TOP_K_FEATURES:-8}"
NUM_GPUS="${NUM_GPUS:-4}"
PER_DEVICE_BS="${PER_DEVICE_BS:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_SEQ="${MAX_SEQ:-1536}"
EPOCHS="${EPOCHS:-3}"
QUANTIZE="${QUANTIZE:-none}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-768}"
SPECIMEN_IDS_FILE_DEFAULT="runs/holdout_lock/ids.json"

LOG_PREFIX="==> Phase 16 pipeline:"

echo "${LOG_PREFIX} starting"
echo "    FRESH            : ${FRESH}"
echo "    REBUILD_DATASET  : ${REBUILD_DATASET}"
echo "    N_SPECIMENS      : ${N_SPECIMENS}"
echo "    TOP_K_FEATURES   : ${TOP_K_FEATURES}"
echo "    NUM_GPUS         : ${NUM_GPUS}"
echo "    PER_DEVICE_BS    : ${PER_DEVICE_BS}"
echo "    GRAD_ACCUM       : ${GRAD_ACCUM}"
echo "    MAX_SEQ          : ${MAX_SEQ}"
echo "    EPOCHS           : ${EPOCHS}"
echo "    QUANTIZE         : ${QUANTIZE}"
echo

# -----------------------------------------------------------------------
# Stage 0: prerequisite checks + optional cleanup
# -----------------------------------------------------------------------

echo "${LOG_PREFIX} Stage 0 -- prerequisite checks"

if ! ls -d checkpoints/probes/*/ > /dev/null 2>&1; then
    echo "ERROR: no probe bank under checkpoints/probes/." >&2
    echo "       Run scripts/train_probe_bank.sh first." >&2
    exit 2
fi
echo "    probe bank   : OK ($(ls -td checkpoints/probes/*/ | head -1))"

if ! compgen -G "checkpoints/sae/*/sae.pt" > /dev/null 2>&1; then
    echo "ERROR: no SAE under checkpoints/sae/." >&2
    echo "       Run scripts/train_sae.sh first." >&2
    exit 2
fi
echo "    SAE          : OK ($(ls -td checkpoints/sae/*/ | head -1))"

if ! compgen -G "runs/sae_labels/*/labels.json" > /dev/null 2>&1; then
    echo "ERROR: no SAE labels under runs/sae_labels/." >&2
    echo "       Run scripts/label_sae_features.sh first." >&2
    exit 2
fi
echo "    SAE labels   : OK ($(ls -td runs/sae_labels/*/labels.json | head -1))"

if [ "${FRESH}" = "1" ]; then
    echo "${LOG_PREFIX} FRESH=1 -- cleaning prior Phase 16 artefacts"
    rm -rf checkpoints/cot-sft-sae 2>/dev/null || true
    rm -rf runs/holdout/cot_sft_sae 2>/dev/null || true
    if [ "${REBUILD_DATASET}" = "1" ]; then
        rm -rf runs/cot_datasets_sae 2>/dev/null || true
    fi
fi

echo

# -----------------------------------------------------------------------
# Stage 1: build SAE-augmented CoT dataset (idempotent)
# -----------------------------------------------------------------------

echo "${LOG_PREFIX} Stage 1 -- build SAE-augmented CoT dataset"

if [ "${REBUILD_DATASET}" != "1" ] && compgen -G "runs/cot_datasets_sae/*/records.jsonl" > /dev/null 2>&1; then
    EXISTING_DATASET="$(ls -td runs/cot_datasets_sae/*/records.jsonl | head -1)"
    echo "    Reusing existing dataset: ${EXISTING_DATASET}"
else
    echo "    Building fresh dataset..."
    N_SPECIMENS="${N_SPECIMENS}" TOP_K_FEATURES="${TOP_K_FEATURES}" bash scripts/build_cot_dataset_with_sae.sh
fi

DATASET_PATH="$(ls -td runs/cot_datasets_sae/*/records.jsonl | head -1)"
echo "    Dataset      : ${DATASET_PATH}"

echo

# -----------------------------------------------------------------------
# Stage 2: SFT train (multi-GPU DDP)
# -----------------------------------------------------------------------

echo "${LOG_PREFIX} Stage 2 -- SFT training (DDP across ${NUM_GPUS} GPUs)"

if compgen -G "checkpoints/cot-sft-sae/*/adapter/adapter_model.safetensors" > /dev/null 2>&1 \
   || compgen -G "checkpoints/cot-sft-sae/*/adapter/adapter_model.bin" > /dev/null 2>&1; then
    EXISTING_ADAPTER="$(ls -td checkpoints/cot-sft-sae/*/adapter | head -1)"
    echo "    Adapter exists at ${EXISTING_ADAPTER} -- skipping retraining."
    echo "    Set FRESH=1 to retrain."
else
    echo "    Training new adapter..."
    NUM_GPUS="${NUM_GPUS}" PER_DEVICE_BS="${PER_DEVICE_BS}" GRAD_ACCUM="${GRAD_ACCUM}" MAX_SEQ="${MAX_SEQ}" EPOCHS="${EPOCHS}" bash scripts/train_cot_sft_with_sae.sh
fi

ADAPTER_PATH="$(ls -td checkpoints/cot-sft-sae/*/adapter | head -1)"
echo "    Adapter      : ${ADAPTER_PATH}"

echo

# -----------------------------------------------------------------------
# Stage 3: probe_head reference baseline (no LLM, fast)
# -----------------------------------------------------------------------

echo "${LOG_PREFIX} Stage 3 -- probe_head reference baseline"

if compgen -G "runs/holdout/probe_head/*/trajectories.jsonl" > /dev/null 2>&1; then
    EXISTING_PH="$(ls -td runs/holdout/probe_head/*/trajectories.jsonl | head -1)"
    EXISTING_PH_LINES="$(wc -l < "${EXISTING_PH}" | tr -d ' ')"
    if [ "${EXISTING_PH_LINES}" -ge 200 ]; then
        echo "    probe_head exists with ${EXISTING_PH_LINES} trajectories -- skipping."
    else
        echo "    probe_head partial (${EXISTING_PH_LINES} lines), rerunning."
        SPECIMEN_IDS_FILE="${SPECIMEN_IDS_FILE_DEFAULT}" bash scripts/run_baseline_probe_head.sh
    fi
else
    SPECIMEN_IDS_FILE="${SPECIMEN_IDS_FILE_DEFAULT}" bash scripts/run_baseline_probe_head.sh
fi

echo

# -----------------------------------------------------------------------
# Stage 4: cot_sft_sae single-shot inference (bf16)
# -----------------------------------------------------------------------

echo "${LOG_PREFIX} Stage 4 -- cot_sft_sae single-shot inference"

SPECIMEN_IDS_FILE="${SPECIMEN_IDS_FILE_DEFAULT}" QUANTIZE="${QUANTIZE}" TOP_K_FEATURES="${TOP_K_FEATURES}" MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" LOG_EVERY=5 bash scripts/run_baseline_cot_sft_sae.sh

echo

# -----------------------------------------------------------------------
# Stage 5: evaluate side-by-side
# -----------------------------------------------------------------------

echo "${LOG_PREFIX} Stage 5 -- side-by-side comparison"

BASELINES_ROOT=runs/holdout bash scripts/evaluate_baselines.sh

echo
echo "${LOG_PREFIX} done."
