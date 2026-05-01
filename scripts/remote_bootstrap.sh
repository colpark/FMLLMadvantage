#!/usr/bin/env bash
#
# remote_bootstrap.sh
#
# Idempotent bootstrap for the FMLLMadvantage repo on a CUDA host. Run
# this once after pulling the repo onto the remote 4xH100 server. Re-
# running the script does nothing harmful, so the user can re-execute
# after a `git pull` to pick up dependency changes.
#
# What the script does:
#   1. Confirms uv is installed. Installs it via the official installer
#      if missing.
#   2. Pins Python 3.11 through uv.
#   3. Runs `uv sync --extra dev` to materialize the dev environment.
#   4. Runs a small Python verification block that prints the PyTorch
#      version, CUDA visibility, every visible GPU, and a tiny matmul
#      on each device.
#
# Usage:
#   bash scripts/remote_bootstrap.sh
#
# Exit codes:
#   0 on success.
#   non-zero if uv fails to install, dependencies fail to sync, or the
#   GPU verification block reports no CUDA devices.

set -euo pipefail

# Resolve the repo root regardless of where the user invoked the script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "==> FMLLMadvantage bootstrap"
echo "==> Repo root: ${REPO_ROOT}"
echo

# ---------------------------------------------------------------------------
# Step 1. uv
# ---------------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    echo "==> uv not found, installing via the official installer"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # The installer drops uv at $HOME/.local/bin. Make it visible for the
    # rest of this script invocation.
    export PATH="${HOME}/.local/bin:${PATH}"
    if ! command -v uv >/dev/null 2>&1; then
        echo "ERROR: uv installer ran but uv is still not on PATH." >&2
        echo "Add \$HOME/.local/bin to your PATH and re-run this script." >&2
        exit 1
    fi
fi
echo "==> uv version: $(uv --version)"
echo

# ---------------------------------------------------------------------------
# Step 2. Python 3.11
# ---------------------------------------------------------------------------
echo "==> Ensuring Python 3.11 is available through uv"
uv python install 3.11
uv python pin 3.11 >/dev/null
echo "==> Python pin: $(cat .python-version)"
echo

# ---------------------------------------------------------------------------
# Step 3. Sync dependencies
# ---------------------------------------------------------------------------
echo "==> Installing dependencies via uv sync --extra dev"
echo "    (first run takes several minutes while torch downloads)"
uv sync --extra dev
echo "==> uv sync completed"
echo

# ---------------------------------------------------------------------------
# Step 4. GPU verification
# ---------------------------------------------------------------------------
echo "==> Verifying CUDA, PyTorch, and all GPUs"
uv run python - <<'PY'
import sys

import torch

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA built      : {torch.version.cuda}")
print(f"CUDA available  : {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("ERROR: PyTorch reports no CUDA devices. Check the NVIDIA driver,")
    print("CUDA runtime, and that you installed torch from the cu124 index.")
    sys.exit(1)

device_count = torch.cuda.device_count()
print(f"Device count    : {device_count}")
for i in range(device_count):
    name = torch.cuda.get_device_name(i)
    cap = torch.cuda.get_device_capability(i)
    mem_gb = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
    print(f"  GPU {i}: {name}  (sm_{cap[0]}{cap[1]}, {mem_gb:.1f} GB)")
    x = torch.randn(1024, 1024, device=f"cuda:{i}")
    y = x @ x
    torch.cuda.synchronize(i)
    print(f"    matmul on cuda:{i} OK, |y| sum = {y.sum().item():.3e}")

if device_count != 4:
    print(f"WARNING: expected 4 H100s, found {device_count}.")
PY
echo

echo "==> Bootstrap complete."
echo "==> Next steps live in docs/remote-setup.md."
