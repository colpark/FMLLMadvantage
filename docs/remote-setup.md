# Remote Setup Guide

This guide walks through bringing up the FMLLMadvantage repository on
the remote 4xH100 server. Code authoring happens on the local laptop.
Everything that touches GPUs, large datasets, or trained models runs on
the remote.

## Prerequisites

The remote host must satisfy the following before bootstrap.

- Linux x86_64.
- NVIDIA driver 590 or newer. The reference machine reports
  `590.48.01` against CUDA 13.1.
- 4 NVIDIA H100 80GB GPUs visible to the kernel and accessible by the
  user account. `nvidia-smi` should list all four.
- Network access to GitHub and to `pypi.org` plus
  `download.pytorch.org`. The torch wheels for cu124 come from the
  PyTorch index.
- At least 200 GB of free disk space under the repo for datasets,
  checkpoints, and run logs.
- `git`, `curl`, and `bash` available on the system.

The bootstrap script installs `uv` and Python 3.11 itself, so the user
does not need to provision them ahead of time.

## Step-by-step bootstrap

### 1. Pull the repo

```
git clone https://github.com/colpark/FMLLMadvantage.git
cd FMLLMadvantage
```

### 2. Run the bootstrap script

```
bash scripts/remote_bootstrap.sh
```

The script is idempotent. Re-run after `git pull` to pick up updated
dependencies.

### 3. Confirm the verification block

The bootstrap prints a verification block at the end. Confirm the
output looks like the following.

```
PyTorch version : 2.5.x+cu124
CUDA built      : 12.4
CUDA available  : True
Device count    : 4
  GPU 0: NVIDIA H100 80GB HBM3  (sm_90, 79.x GB)
    matmul on cuda:0 OK, |y| sum = ...
  GPU 1: NVIDIA H100 80GB HBM3  (sm_90, 79.x GB)
    matmul on cuda:1 OK, |y| sum = ...
  GPU 2: NVIDIA H100 80GB HBM3  (sm_90, 79.x GB)
    matmul on cuda:2 OK, |y| sum = ...
  GPU 3: NVIDIA H100 80GB HBM3  (sm_90, 79.x GB)
    matmul on cuda:3 OK, |y| sum = ...
```

### 4. Run the local-only test suite

```
uv run pytest -m "not gpu" -v
```

The utility tests run without a GPU and serve as a smoke test for the
Python environment.

## Expected output and what to do if it differs

| Symptom | Likely cause | Action |
|---|---|---|
| `uv: command not found` after install | `$HOME/.local/bin` not on PATH | `export PATH="$HOME/.local/bin:$PATH"` then re-run |
| `CUDA available  : False` | Torch installed without CUDA wheel | Confirm `pyproject.toml` retains the cu124 index, then run `uv sync --reinstall-package torch` |
| `Device count    : 0` | NVIDIA driver missing or `CUDA_VISIBLE_DEVICES=""` | `nvidia-smi` to confirm the driver, then `unset CUDA_VISIBLE_DEVICES` |
| `Device count    : 1` (or fewer than 4) | Other processes hold the rest, or `CUDA_VISIBLE_DEVICES` restricts | Check `nvidia-smi`, then unset or widen `CUDA_VISIBLE_DEVICES` |
| `uv sync` fails resolving torch | Index unreachable | Confirm outbound HTTPS to `download.pytorch.org` |

## Selecting GPUs at run time

All training and inference scripts respect `CUDA_VISIBLE_DEVICES`. To
run a single-GPU script on GPU 0:

```
CUDA_VISIBLE_DEVICES=0 uv run python scripts/<some_script>.py
```

To run a script across all four GPUs through `accelerate`:

```
uv run accelerate launch --num_processes 4 scripts/<some_script>.py
```

Per-FM training scripts use one GPU each. The recommended layout for
parallel FM training places FM1 on GPU 0, FM2 on GPU 1, and FM3 on
GPU 2, leaving GPU 3 free for evaluation and ad-hoc work.

## Updating dependencies

After a `git pull` that touches `pyproject.toml` or `uv.lock`, re-run
the bootstrap or run `uv sync --extra dev` directly. Both commands are
safe to re-run.

## Troubleshooting

### PyTorch wheel mismatch

If the H100 reports compute capability `sm_90` but PyTorch raises
`NVIDIA H100 with CUDA capability sm_90 is not compatible with the
current PyTorch installation`, the CPU wheel made it through the
resolver. Force a reinstall:

```
uv sync --reinstall-package torch --extra dev
```

If the issue persists, edit `pyproject.toml` and confirm that the
`[[tool.uv.index]]` block names `pytorch-cu124` and that
`[tool.uv.sources]` routes `torch` to that index for Linux.

### Multi-GPU visibility under containers

Inside a Docker or Slurm container, the runtime must expose all four
GPUs via `--gpus all` (Docker) or the equivalent `srun --gres=gpu:4`
flag. Confirm with `nvidia-smi` inside the container before running
the bootstrap.

### Torch reports the wrong CUDA version

PyTorch built for cu124 prints `CUDA built : 12.4` and runs against
the system driver through forward compatibility. The driver version
(13.1) does not have to match the build target. Mismatches surface
only when the driver is older than the build target.
