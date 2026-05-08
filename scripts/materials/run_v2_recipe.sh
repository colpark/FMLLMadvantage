#!/usr/bin/env bash
#
# run_v2_recipe.sh -- end-to-end v2 recipe runner.
#
# Three coordinated improvements over v1:
#
#   1. Token budget: MAX_SEQ=2560, MAX_NEW_TOKENS=1536 so rich CoTs
#      fit at training and inference. (v1 was 1024 / 768; rich CoTs
#      are ~1500 tokens of assistant content.)
#
#   2. Richer SAE info: tighter labelling thresholds (purity>=0.90,
#      |corr|>=0.55), correlations on top-N activators only (sharper
#      signal), top-5 representative training specimens per feature
#      (Si, Ge, C-diamond -- chemistry the LLM has priors for),
#      activation quantiles for calibrated firing strength.
#
#   3. Rich CoT: 5-step reasoning trace with composition sanity
#      check, probe-consistency review with confidence, and
#      counterfactual / probe-tension analysis. Each step builds
#      on the prior.
#
# Designed for unattended background execution on 4xH100 80GB:
#
#   nohup bash scripts/materials/run_v2_recipe.sh \
#       > v2_recipe.log 2>&1 &
#   tail -f v2_recipe.log
#
# Or in tmux:
#
#   tmux new -s v2_recipe
#   bash scripts/materials/run_v2_recipe.sh
#   # Ctrl-b d to detach
#
# Total wall-clock on 4xH100 80GB: ~60-90 min
#   * Stage 6 relabel SAE (4 GPUs idle here, single-process)  : ~1 min
#   * Stage 7 build rich CoT records (10K)                    : ~5 min
#   * Stage 8 train LoRA at MAX_SEQ=2560 (4 GPUs DDP)         : ~45-75 min
#   * Stage 9 batched inference (BS=16, MAX_NEW=1536)         : ~5 min
#   * Repair + 9c hybrid + compare                            : ~30 sec
#
# Environment variables (optional):
#   N_SPECIMENS    default: 10000
#   NUM_GPUS       default: 4
#   PER_DEVICE_BS  default: 2
#   GRAD_ACCUM     default: 4
#   MAX_SEQ        default: 2560
#   MAX_NEW_TOKENS default: 1536
#   EPOCHS         default: 3
#   LORA_R         default: 16
#   BATCH_SIZE     default: 16  (Stage 9 batched generate)
#   MIN_PURITY     default: 0.90
#   MIN_CORR       default: 0.55

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

N_SPECIMENS="${N_SPECIMENS:-10000}"
NUM_GPUS="${NUM_GPUS:-4}"
PER_DEVICE_BS="${PER_DEVICE_BS:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_SEQ="${MAX_SEQ:-2560}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1536}"
EPOCHS="${EPOCHS:-3}"
LORA_R="${LORA_R:-16}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MIN_PURITY="${MIN_PURITY:-0.90}"
MIN_CORR="${MIN_CORR:-0.55}"

echo "============================================================"
echo "==> v2 recipe runner (rich SAE labels + rich CoT + bigger tokens)"
echo "============================================================"
echo "    N_SPECIMENS    : ${N_SPECIMENS}"
echo "    NUM_GPUS       : ${NUM_GPUS}"
echo "    PER_DEVICE_BS  : ${PER_DEVICE_BS}"
echo "    GRAD_ACCUM     : ${GRAD_ACCUM}"
echo "    Effective BS   : $((PER_DEVICE_BS * GRAD_ACCUM * NUM_GPUS))"
echo "    MAX_SEQ        : ${MAX_SEQ}"
echo "    MAX_NEW_TOKENS : ${MAX_NEW_TOKENS}"
echo "    EPOCHS         : ${EPOCHS}"
echo "    LORA_R         : ${LORA_R}"
echo "    BATCH_SIZE     : ${BATCH_SIZE}"
echo "    MIN_PURITY     : ${MIN_PURITY}"
echo "    MIN_CORR       : ${MIN_CORR}"
echo "    Started        : $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo

# ----------------------------------------------------------------
# Stage 6: re-label SAE with v2 thresholds + rich metadata.
# ----------------------------------------------------------------
echo "============================================================"
echo "[1/6] Re-labelling SAE features (v2: rich + tighter thresholds)"
echo "============================================================"
uv run python scripts/materials/06_label_sae.py \
    --min-purity "${MIN_PURITY}" \
    --min-corr "${MIN_CORR}" \
    --corr-on-top-n \
    --top-specimens-keep 5
echo "    [done] $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo

# ----------------------------------------------------------------
# Stage 7: build rich CoT records.
# ----------------------------------------------------------------
echo "============================================================"
echo "[2/6] Building rich CoT records (Step 1/1b/2/3/4/Final)"
echo "============================================================"
RICH_COT=1 \
    INCLUDE_SAE=1 \
    N_SPECIMENS="${N_SPECIMENS}" \
    OUT_ROOT="runs/materials/cot_datasets_v2_rich" \
    bash scripts/materials/07_build_cot.sh
echo "    [done] $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo

# ----------------------------------------------------------------
# Stage 8: train LoRA on rich records, MAX_SEQ=2560, 4 H100 DDP.
# ----------------------------------------------------------------
echo "============================================================"
echo "[3/6] Training LoRA at MAX_SEQ=${MAX_SEQ} on 4 H100s (DDP)"
echo "============================================================"
DATASET_ROOT="runs/materials/cot_datasets_v2_rich" \
    OUT_ROOT="checkpoints/materials/cot-sft-v2-rich" \
    NUM_GPUS="${NUM_GPUS}" \
    PER_DEVICE_BS="${PER_DEVICE_BS}" \
    GRAD_ACCUM="${GRAD_ACCUM}" \
    MAX_SEQ="${MAX_SEQ}" \
    EPOCHS="${EPOCHS}" \
    LORA_R="${LORA_R}" \
    bash scripts/materials/08_train_sft.sh
echo "    [done] $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo

# Resolve the freshly-trained adapter path so Stage 9 picks it up.
ADAPTER_PATH="$(ls -td checkpoints/materials/cot-sft-v2-rich/*/adapter 2>/dev/null | head -1 || true)"
if [ -z "${ADAPTER_PATH}" ]; then
    echo "ERROR: no adapter under checkpoints/materials/cot-sft-v2-rich/" >&2
    exit 2
fi
echo "    v2 adapter     : ${ADAPTER_PATH}"
echo

# ----------------------------------------------------------------
# Stage 9: single-shot inference, MAX_NEW_TOKENS=1536, BS=16.
# ----------------------------------------------------------------
echo "============================================================"
echo "[4/6] Single-shot inference (v2 adapter, MAX_NEW=${MAX_NEW_TOKENS})"
echo "============================================================"
ADAPTER_PATH="${ADAPTER_PATH}" \
    INCLUDE_SAE=1 \
    OUT_SUBDIR="cot_sft_v2_rich" \
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    bash scripts/materials/09_run_singleshot.sh
echo "    [done] $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo

# ----------------------------------------------------------------
# Repair parse failures using the lenient parser.
# ----------------------------------------------------------------
echo "============================================================"
echo "[5/6] Repairing parse failures (lenient parser)"
echo "============================================================"
V2_RAW="$(ls -td runs/materials/holdout/cot_sft_v2_rich/*/records.jsonl 2>/dev/null | head -1 || true)"
if [ -z "${V2_RAW}" ]; then
    echo "ERROR: no v2 records.jsonl under cot_sft_v2_rich/" >&2
    exit 2
fi
INPUT="${V2_RAW}" bash scripts/materials/repair_parse_failures.sh
echo "    [done] $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo

# ----------------------------------------------------------------
# Stage 9c: hybrid scoring on the v2 LLM claims.
# ----------------------------------------------------------------
echo "============================================================"
echo "[6/6] Hybrid scoring (v2 LLM regression + probe classification)"
echo "============================================================"
INPUT_SUBDIR="cot_sft_v2_rich" \
    OUT_SUBDIR="hybrid_v2_rich" \
    bash scripts/materials/09c_score_hybrid.sh
echo "    [done] $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo

# ----------------------------------------------------------------
# Final compare across all variants the script can find.
# ----------------------------------------------------------------
echo "============================================================"
echo "==> FINAL COMPARISON"
echo "============================================================"
bash scripts/materials/compare_baselines.sh \
    --cot-sft-sae-jsonl "$(ls -td runs/materials/holdout/cot_sft_v2_rich/*/records.repaired.jsonl 2>/dev/null | head -1)" \
    --hybrid-jsonl "$(ls -td runs/materials/holdout/hybrid_v2_rich/*/records.jsonl 2>/dev/null | head -1)"
echo
echo "============================================================"
echo "==> v2 recipe finished: $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo "============================================================"
echo
echo "Key artifacts:"
echo "  rich labels   : $(ls -td runs/materials/sae_labels/*/labels_rich.json 2>/dev/null | head -1)"
echo "  rich records  : $(ls -td runs/materials/cot_datasets_v2_rich/*/records.jsonl 2>/dev/null | head -1)"
echo "  v2 adapter    : ${ADAPTER_PATH}"
echo "  v2 inference  : $(ls -td runs/materials/holdout/cot_sft_v2_rich/*/records.repaired.jsonl 2>/dev/null | head -1)"
echo "  v2 hybrid     : $(ls -td runs/materials/holdout/hybrid_v2_rich/*/records.jsonl 2>/dev/null | head -1)"
