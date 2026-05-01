# Audit Report, Phase 0

**Audited at:** 2026-05-01T20:07:25Z
**Auditor:** Claude Code (self-audit)
**Result:** PASS (after fixes applied during the audit)

## Summary

Phase 0 implements every artifact the original prompt called for. The
audit identified four gaps before any fixes: four directories
(`runs/`, `data/`, `checkpoints/`, `docs/progress/`) lacked READMEs, the
`.gitignore` would have hidden three of those READMEs, the
`configs/default.yaml` placeholder sections lacked the per-phase
comments the audit checklist required, and the progress document
quoted a stale test count of 14 when the suite has 13 tests. I fixed
all four. After the fixes every check passes. The utility tests run
locally (13 passed in 0.18s) against a minimal venv that holds only
loguru, pydantic, pyyaml, and pytest, so the audit verified the
import contract without provisioning the full dependency stack.

## Detailed checks

### CHECK 0.1, directory structure
- **Result:** FIXED
- **Evidence:** Initial scan listed `runs/`, `data/`, `checkpoints/`,
  and `docs/progress/` as missing READMEs. After the fix every required
  directory contains a non-empty README. The four added READMEs are
  `runs/README.md` (583 bytes), `data/README.md` (610 bytes),
  `checkpoints/README.md` (358 bytes), `docs/progress/README.md`
  (470 bytes).
- **Action taken:** Wrote the four READMEs and amended `.gitignore` to
  un-ignore `runs/README.md`, `checkpoints/README.md`, and
  `data/README.md` (the parent directories stay ignored).

### CHECK 0.2, pyproject.toml validity
- **Result:** PASS
- **Evidence:** `python3 -c "import tomllib; tomllib.load(open('pyproject.toml','rb'))"`
  parses without error. Project name is `fmllm`. Python pin is
  `>=3.11,<3.12`. Every dependency from the prompt appears in
  `[project.dependencies]` with the version constraint specified.
  Dev dependencies live under `[project.optional-dependencies].dev`.
  PyTorch routing uses `[[tool.uv.index]]` with name `pytorch-cu124`
  plus `[tool.uv.sources]` mapping `torch` to that index for
  `sys_platform == 'linux'`.

### CHECK 0.3, .python-version
- **Result:** PASS
- **Evidence:** File contains the literal string `3.11`.

### CHECK 0.4, .gitignore correctness
- **Result:** PASS (after CHECK 0.1 fix)
- **Evidence:** Every required entry exists (verified by grep over
  `runs/`, `data/`, `checkpoints/`, `.venv/`, `__pycache__`,
  `.pytest_cache`, `.ruff_cache`, `.mypy_cache`, `*.pyc`, `.DS_Store`,
  `*.egg-info`, `build/`, `dist/`, `*.so`, `*.pt`, `*.bin`,
  `*.safetensors`, `.ipynb_checkpoints`, `wandb/`). The manifest
  exception lives under `data/`: `!data/**/*.yaml` and
  `!data/**/manifest.yaml` keep manifests committed while the rest of
  the directory stays ignored.

### CHECK 0.5, remote bootstrap script
- **Result:** PASS
- **Evidence:** `scripts/remote_bootstrap.sh` starts with
  `#!/usr/bin/env bash`, mode `755` from `stat -f "%Sp %A"`. The
  script installs uv via `https://astral.sh/uv/install.sh` when
  missing, runs `uv python install 3.11`, runs `uv sync --extra dev`,
  and ends with a Python heredoc that prints PyTorch version, CUDA
  build, device count, per-device name and capability, and a
  per-device matmul. Comments are in active voice with no em-dashes
  or semicolons in narrative.

### CHECK 0.6, remote setup documentation
- **Result:** PASS
- **Evidence:** `docs/remote-setup.md` has sections for
  `Prerequisites`, `Step-by-step bootstrap`, `Expected output and
  what to do if it differs`, `Selecting GPUs at run time`,
  `Updating dependencies`, and `Troubleshooting`. The expected
  output block names the H100 device string and the matmul lines.
  `Selecting GPUs at run time` covers `CUDA_VISIBLE_DEVICES` and
  `accelerate launch --num_processes 4`.

### CHECK 0.7, configs/default.yaml structure
- **Result:** FIXED
- **Evidence:** YAML parses. Top-level keys are `dataset`, `fm1`,
  `fm2`, `fm3`, `orchestrator`, `run_id_format`, `seeds`, `verifier`.
  The initial draft lacked per-section comments naming which phase
  fleshes out each section.
- **Action taken:** Added phase-naming comments above the `dataset`,
  `fm1`/`fm2`/`fm3`, `verifier`, and `orchestrator` sections.

### CHECK 0.8, utility modules
- **Result:** PASS
- **Evidence:** Each of `logging.py`, `manifests.py`, `run_ids.py`,
  `config.py` starts with a triple-quoted docstring. Public functions
  carry full Google-style docstrings (Args, Returns, Raises). Import
  test runs successfully against a minimal venv that has only
  `loguru`, `pydantic`, `pyyaml`, `pytest`:

  ```
  PYTHONPATH=src python -c "from fmllm.utils import logging, manifests, run_ids, config"
  ```

  Output reports `IMPORTS OK` with `configure_logging`,
  `write_manifest`, `generate_run_id`, `Config`, and `load_config` all
  resolving.

### CHECK 0.9, local tests
- **Result:** PASS
- **Evidence:** `python -m pytest tests/test_utils.py -v` reports
  `13 passed in 0.18s`. The audit ran the suite under
  `/tmp/fmllm-audit-venv` (loguru, pydantic, pyyaml, pytest) rather
  than running `uv sync`, which keeps the local laptop free of the
  full dependency graph. The progress doc quotes 13 tests.

### CHECK 0.10, pytest GPU marker
- **Result:** PASS
- **Evidence:** `tests/conftest.py` adds the `gpu` marker with
  `pytest_configure`, then in `pytest_collection_modifyitems` skips
  every `gpu`-marked item with reason `"No CUDA GPU available. Run on
  the remote 4xH100 host. See docs/remote-setup.md."`. The hook
  imports `torch` defensively under a broad `except Exception` so a
  missing torch defaults to `gpu_available = False`.

### CHECK 0.11, progress documentation
- **Result:** FIXED
- **Evidence:** `docs/progress/00-init.md` covers what I built, what
  the user runs locally, what the user runs on the remote, what to
  send back, known issues, and what remains. Earlier draft quoted
  `14 tests`. A scan for em-dashes and semicolons outside code fences
  reports zero matches.
- **Action taken:** Replaced `14 tests` with `13 tests` in the
  progress doc.

### CHECK 0.12, root README
- **Result:** PASS
- **Evidence:** `README.md` opens with the project description,
  documents the local-vs-remote split under
  `## Execution topology`, points at `docs/remote-setup.md` under
  `## Bootstrap on the remote`, and points at `docs/progress/` for
  per-phase notes. The narrative carries no em-dashes or semicolons
  outside code fences.

### CHECK 0.13, no execution leakage
- **Result:** PASS
- **Evidence:** No `uv sync` ran against the project venv. No model
  download. No data generation. No GPU access. The audit ran
  utility imports and pytest under an isolated venv at
  `/tmp/fmllm-audit-venv`, which lives outside the repo and does
  not appear in `git status`. The transient `.pytest_cache` directory
  pytest created landed inside `.gitignore` and I removed it after
  the run.

### CHECK 0.14, commit cleanliness
- **Result:** PASS (after the audit commit lands)
- **Evidence:** Before the audit commit, `git status` shows the
  audit's own fixes (modified `.gitignore`, modified
  `configs/default.yaml`, modified `docs/progress/00-init.md`, four
  new READMEs, the new audit report). The audit commit folds those
  changes in.

## Fixes applied during audit

- Added `runs/README.md`, `data/README.md`, `checkpoints/README.md`,
  and `docs/progress/README.md`.
- Updated `.gitignore` so `runs/README.md`, `checkpoints/README.md`,
  and `data/README.md` stay tracked while the parent directories
  remain ignored.
- Added per-section comments to `configs/default.yaml` naming which
  phase fleshes out each placeholder section.
- Replaced `14 tests` with `13 tests` in `docs/progress/00-init.md`.
- Created `docs/audits/00-init-audit.md` (this file).

## Remaining concerns

- The audit verified imports and tests against an isolated venv that
  ships only `loguru`, `pydantic`, `pyyaml`, and `pytest`. The full
  `uv sync --extra dev` runs only on the remote, where it pulls
  torch from the cu124 index. The user should still run the remote
  bootstrap and confirm the verification block prints the expected
  H100 lines before declaring Phase 0 complete on hardware.
- `tests/test_utils.py::test_load_config_repo_default` validates the
  shipped `configs/default.yaml` against the Pydantic schema. That
  test guards future drift between the YAML and the model. Phase 1
  and beyond must update both files together or the test fails.
- The `data/` gitignore exception covers manifest YAMLs anywhere
  under the directory. If a Phase 1 manifest needs additional
  metadata files (CSVs, summary plots), those will not be tracked
  unless the gitignore picks them up explicitly.

## Sign-off

The Phase 0 implementation matches the original prompt's
specification. The user can proceed to running the remote
verification commands in `docs/progress/00-init.md`.
