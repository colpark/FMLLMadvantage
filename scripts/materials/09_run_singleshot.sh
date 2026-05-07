#!/usr/bin/env bash
#
# 09_run_singleshot.sh -- Stage 9 of the materials port.
# Single-shot LLM inference on the held-out 200 specimens. Forwards
# CHGNet live (no embeddings cache dependency), runs the materials
# probes + SAE, generates one CoT per specimen via the SFT-tuned
# LoRA + Qwen base, and parses + scores the "Final commit:" JSON.
#
# Usage:
#   bash scripts/materials/09_run_singleshot.sh
#   QUANTIZE=8bit bash scripts/materials/09_run_singleshot.sh   # higher fidelity
#
# Environment variables (optional):
#   ADAPTER_PATH      default: latest under checkpoints/materials/cot-sft-sae/
#   PROBE_BANK_DIR    default: latest under checkpoints/materials/probes/
#   SAE_DIR           default: latest under checkpoints/materials/sae/
#   SAE_LABELS_PATH   default: latest labels.json under runs/materials/sae_labels/
#   HOLDOUT_IDS_PATH  default: data/materials_project_v1/holdout_lock/ids.json
#   CHGNET_MODEL_NAME default: 0.3.0
#   BASE_MODEL        default: Qwen/Qwen2.5-7B-Instruct
#   MAX_NEW_TOKENS    default: 768
#   QUANTIZE          default: 4bit ('none' | '4bit' | '8bit')
#   BATCH_SIZE        default: 16 (LLM batched generate; H100 80GB fits 32+)
#   TOP_K_FEATURES    default: 8
#   MAX_ATOMS         default: 80
#   GPU               default: 0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CHGNET_MODEL_NAME="${CHGNET_MODEL_NAME:-0.3.0}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-768}"
QUANTIZE="${QUANTIZE:-4bit}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TOP_K_FEATURES="${TOP_K_FEATURES:-8}"
MAX_ATOMS="${MAX_ATOMS:-80}"
INCLUDE_SAE="${INCLUDE_SAE:-1}"
GPU="${GPU:-0}"
HOLDOUT_IDS_PATH="${HOLDOUT_IDS_PATH:-data/materials_project_v1/holdout_lock/ids.json}"
if [ "${INCLUDE_SAE}" = "0" ] || [ "${INCLUDE_SAE}" = "false" ]; then
    DEFAULT_OUT_SUBDIR="cot_sft_no_sae"
else
    DEFAULT_OUT_SUBDIR="cot_sft_sae"
fi
OUT_SUBDIR="${OUT_SUBDIR:-${DEFAULT_OUT_SUBDIR}}"

EXTRA=()
EXTRA+=(--chgnet-model-name "${CHGNET_MODEL_NAME}")
EXTRA+=(--base-model "${BASE_MODEL}")
EXTRA+=(--max-new-tokens "${MAX_NEW_TOKENS}")
EXTRA+=(--quantize "${QUANTIZE}")
EXTRA+=(--batch-size "${BATCH_SIZE}")
EXTRA+=(--top-k-features "${TOP_K_FEATURES}")
EXTRA+=(--max-atoms "${MAX_ATOMS}")
EXTRA+=(--out-subdir "${OUT_SUBDIR}")
EXTRA+=(--holdout-ids-path "${HOLDOUT_IDS_PATH}")
if [ "${INCLUDE_SAE}" = "0" ] || [ "${INCLUDE_SAE}" = "false" ]; then
    EXTRA+=(--no-include-sae)
else
    EXTRA+=(--include-sae)
fi
if [ -n "${ADAPTER_PATH:-}" ]; then
    EXTRA+=(--adapter-path "${ADAPTER_PATH}")
fi
if [ -n "${PROBE_BANK_DIR:-}" ]; then
    EXTRA+=(--probe-bank-dir "${PROBE_BANK_DIR}")
fi
if [ -n "${SAE_DIR:-}" ]; then
    EXTRA+=(--sae-dir "${SAE_DIR}")
fi
if [ -n "${SAE_LABELS_PATH:-}" ]; then
    EXTRA+=(--sae-labels-path "${SAE_LABELS_PATH}")
fi

echo "==> Materials port Stage 9: single-shot inference"
echo "    ADAPTER_PATH      : ${ADAPTER_PATH:-(latest)}"
echo "    PROBE_BANK_DIR    : ${PROBE_BANK_DIR:-(latest)}"
echo "    SAE_DIR           : ${SAE_DIR:-(latest)}"
echo "    SAE_LABELS_PATH   : ${SAE_LABELS_PATH:-(latest)}"
echo "    HOLDOUT_IDS_PATH  : ${HOLDOUT_IDS_PATH}"
echo "    CHGNET_MODEL_NAME : ${CHGNET_MODEL_NAME}"
echo "    BASE_MODEL        : ${BASE_MODEL}"
echo "    MAX_NEW_TOKENS    : ${MAX_NEW_TOKENS}"
echo "    QUANTIZE          : ${QUANTIZE}"
echo "    BATCH_SIZE        : ${BATCH_SIZE}"
echo "    INCLUDE_SAE       : ${INCLUDE_SAE}"
echo "    OUT_SUBDIR        : ${OUT_SUBDIR}"
echo "    TOP_K_FEATURES    : ${TOP_K_FEATURES}"
echo "    MAX_ATOMS         : ${MAX_ATOMS}"
echo "    GPU               : ${GPU}"
echo

CUDA_VISIBLE_DEVICES="${GPU}" uv run python scripts/materials/09_run_singleshot.py "${EXTRA[@]}"
