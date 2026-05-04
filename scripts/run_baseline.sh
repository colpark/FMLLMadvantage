#!/usr/bin/env bash
#
# run_baseline.sh
#
# Run one Phase 8a baseline on a range of specimens. Three modes:
#   naked       (B0): one-shot LLM commit, no FM tools, no verifier
#   no_verifier (B2): OHVD loop with NoOpVerifier; FMs + bridges intact
#   full        (B3): canonical Pipeline A
#
# Usage:
#   bash scripts/run_baseline.sh naked
#   bash scripts/run_baseline.sh no_verifier --count 200
#   bash scripts/run_baseline.sh full --start 0 --count 200
#
# Environment variables (optional):
#   START         (default: 0)
#   COUNT         (default: 200)
#   H5_PATH       (default: data/synthetic_lj_v1/specimens.h5)
#   OUT_ROOT      (default: runs/baselines)
#   LLM_MODEL     (default: Qwen/Qwen2.5-7B-Instruct; open-weights model
#                  consistent with Pipeline B SFT. Set to a Llama variant
#                  only if your HF account has access to that gated repo.)
#   LLM_TEMP      (default: 0.4)
#   GPU           (default: 0)
#   MOCK_SCRIPT   (default: unset; pass to use a JSON-list mock LLM)
#   ADAPTER_PATH  (default: unset; pass for Pipeline B inference)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

BASELINE="${1:-}"
shift || true
if [ -z "${BASELINE}" ] || ! [[ "${BASELINE}" =~ ^(naked|no_verifier|full)$ ]]; then
    echo "Usage: bash scripts/run_baseline.sh {naked|no_verifier|full} [extra args]" >&2
    exit 1
fi

START="${START:-0}"
COUNT="${COUNT:-200}"
H5_PATH="${H5_PATH:-data/synthetic_lj_v1/specimens.h5}"
OUT_ROOT="${OUT_ROOT:-runs/baselines}"
LLM_MODEL="${LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
LLM_TEMP="${LLM_TEMP:-0.4}"
GPU="${GPU:-0}"

EXTRA=()
if [ -n "${MOCK_SCRIPT:-}" ]; then
    EXTRA+=(--mock-script "${MOCK_SCRIPT}")
fi
if [ -n "${ADAPTER_PATH:-}" ]; then
    EXTRA+=(--adapter-path "${ADAPTER_PATH}")
fi

echo "==> Baseline runner"
echo "    Baseline   : ${BASELINE}"
echo "    Specimens  : [${START}, $((START + COUNT)))"
echo "    H5 path    : ${H5_PATH}"
echo "    Output root: ${OUT_ROOT}"
echo "    LLM        : ${LLM_MODEL} (T=${LLM_TEMP})"
echo "    GPU        : ${GPU}"
echo

CUDA_VISIBLE_DEVICES="${GPU}" uv run python scripts/run_baseline.py \
    --baseline "${BASELINE}" \
    --start "${START}" \
    --count "${COUNT}" \
    --h5-path "${H5_PATH}" \
    --out "${OUT_ROOT}" \
    --llm-model "${LLM_MODEL}" \
    --llm-temperature "${LLM_TEMP}" \
    "${EXTRA[@]}" \
    "$@"
