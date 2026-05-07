#!/usr/bin/env bash
#
# run_sae_ablation.sh -- end-to-end SAE ablation runner.
#
# Builds no-SAE CoT records, trains a no-SAE LoRA adapter, runs
# inference, repairs parse failures, scores the hybrid, and runs
# compare_baselines against all four/five operating points
# (probe_head, cot_sft_sae, cot_sft_no_sae, hybrid, hybrid_no_sae).
#
# Designed to run unattended in the background:
#
#   nohup bash scripts/materials/run_sae_ablation.sh \
#       > sae_ablation.log 2>&1 &
#   tail -f sae_ablation.log
#
# Or in a tmux session:
#
#   tmux new -s sae_ablation
#   bash scripts/materials/run_sae_ablation.sh
#   # Ctrl-b d  to detach
#
# Total wall-clock on 4xH100 80GB: ~35-50 min
#   * Stage 7 (build no-SAE records, 10K)        : ~5 min
#   * Stage 8 (train LoRA, 3 epochs, 4 GPUs)     : ~25 min
#   * Stage 9 (inference, batched bs=16)         : ~3 min
#   * Repair + 9c hybrid + compare               : ~30 sec
#
# Environment variables (optional):
#   N_SPECIMENS    default: 10000 (passed to Stage 7)
#   NUM_GPUS       default: 4
#   PER_DEVICE_BS  default: 2
#   GRAD_ACCUM     default: 4
#   MAX_SEQ        default: 1024
#   EPOCHS         default: 3
#   LORA_R         default: 16

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

N_SPECIMENS="${N_SPECIMENS:-10000}"
NUM_GPUS="${NUM_GPUS:-4}"
PER_DEVICE_BS="${PER_DEVICE_BS:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_SEQ="${MAX_SEQ:-1024}"
EPOCHS="${EPOCHS:-3}"
LORA_R="${LORA_R:-16}"

echo "============================================================"
echo "==> SAE ablation runner"
echo "============================================================"
echo "    N_SPECIMENS    : ${N_SPECIMENS}"
echo "    NUM_GPUS       : ${NUM_GPUS}"
echo "    PER_DEVICE_BS  : ${PER_DEVICE_BS}"
echo "    GRAD_ACCUM     : ${GRAD_ACCUM}"
echo "    MAX_SEQ        : ${MAX_SEQ}"
echo "    EPOCHS         : ${EPOCHS}"
echo "    LORA_R         : ${LORA_R}"
echo "    Started        : $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo

# ----------------------------------------------------------------
# Stage 7: build no-SAE CoT records
# ----------------------------------------------------------------
echo "============================================================"
echo "[1/5] Building no-SAE CoT records"
echo "============================================================"
INCLUDE_SAE=0 N_SPECIMENS="${N_SPECIMENS}" bash scripts/materials/07_build_cot.sh
echo "    [done] $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo

# ----------------------------------------------------------------
# Stage 8: train no-SAE LoRA adapter
# ----------------------------------------------------------------
echo "============================================================"
echo "[2/5] Training no-SAE LoRA adapter"
echo "============================================================"
DATASET_ROOT="runs/materials/cot_datasets_no_sae" \
    OUT_ROOT="checkpoints/materials/cot-sft-no-sae" \
    NUM_GPUS="${NUM_GPUS}" \
    PER_DEVICE_BS="${PER_DEVICE_BS}" \
    GRAD_ACCUM="${GRAD_ACCUM}" \
    MAX_SEQ="${MAX_SEQ}" \
    EPOCHS="${EPOCHS}" \
    LORA_R="${LORA_R}" \
    bash scripts/materials/08_train_sft.sh
echo "    [done] $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo

# ----------------------------------------------------------------
# Resolve the freshly-trained adapter path so Stage 9 picks it up.
# ----------------------------------------------------------------
ADAPTER_PATH="$(ls -td checkpoints/materials/cot-sft-no-sae/*/adapter 2>/dev/null | head -1 || true)"
if [ -z "${ADAPTER_PATH}" ]; then
    echo "ERROR: no adapter under checkpoints/materials/cot-sft-no-sae/" >&2
    exit 2
fi
echo "    no-SAE adapter : ${ADAPTER_PATH}"
echo

# ----------------------------------------------------------------
# Stage 9: single-shot inference with the no-SAE adapter
# ----------------------------------------------------------------
echo "============================================================"
echo "[3/5] Single-shot inference (no-SAE adapter)"
echo "============================================================"
ADAPTER_PATH="${ADAPTER_PATH}" \
    INCLUDE_SAE=0 \
    OUT_SUBDIR="cot_sft_no_sae" \
    bash scripts/materials/09_run_singleshot.sh
echo "    [done] $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo

# ----------------------------------------------------------------
# Repair parse failures on the no-SAE JSONL
# ----------------------------------------------------------------
echo "============================================================"
echo "[4/5] Repairing parse failures (lenient parser)"
echo "============================================================"
NO_SAE_RAW="$(ls -td runs/materials/holdout/cot_sft_no_sae/*/records.jsonl 2>/dev/null | head -1 || true)"
if [ -z "${NO_SAE_RAW}" ]; then
    echo "ERROR: no no-SAE records.jsonl found." >&2
    exit 2
fi
INPUT="${NO_SAE_RAW}" bash scripts/materials/repair_parse_failures.sh
echo "    [done] $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo

# ----------------------------------------------------------------
# Stage 9c: hybrid scoring on the no-SAE LLM claims
# ----------------------------------------------------------------
echo "============================================================"
echo "[5/5] Hybrid scoring (no-SAE LLM regression + probe classification)"
echo "============================================================"
INPUT_SUBDIR="cot_sft_no_sae" \
    OUT_SUBDIR="hybrid_no_sae" \
    bash scripts/materials/09c_score_hybrid.sh
echo "    [done] $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo

# ----------------------------------------------------------------
# Final compare across all variants
# ----------------------------------------------------------------
echo "============================================================"
echo "==> FINAL COMPARISON"
echo "============================================================"
bash scripts/materials/compare_baselines.sh
echo
echo "============================================================"
echo "==> SAE ablation finished: $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo "============================================================"
