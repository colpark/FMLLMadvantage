# Phase 0: Repository Initialization

## What I built

- Repository scaffolding with one directory per pipeline component.
  Every directory carries a README that names its purpose and lists
  its files.
- `pyproject.toml` declaring the full dependency stack pinned to
  Python 3.11. PyTorch routes through the cu124 wheel index for
  Linux hosts via `[[tool.uv.index]]` plus `[tool.uv.sources]`. Other
  platforms fall back to the default PyPI wheel so local laptop
  development works for non-GPU code.
- `.python-version` pinning Python 3.11.
- `.gitignore` excluding generated artifacts (runs, checkpoints,
  most of `data/`) while preserving manifest YAMLs and the
  `data/literature/` subtree for the literature database that
  arrives in Phase 4.
- `scripts/remote_bootstrap.sh`, an idempotent bootstrap for the
  remote 4xH100 host. Installs uv if missing, pins Python 3.11,
  runs `uv sync --extra dev`, and verifies all four GPUs through a
  small PyTorch script.
- `docs/remote-setup.md` with prerequisites, exact bootstrap
  commands, expected output at every step, troubleshooting notes,
  and `CUDA_VISIBLE_DEVICES` guidance.
- `src/fmllm/utils/` with four modules and full docstrings:
  - `logging.py` configures loguru sinks for stdout (INFO+) and
    `<run_dir>/run.log` (DEBUG+).
  - `manifests.py` writes manifest YAMLs with a fixed schema
    covering script identity, timestamp, git commit, host
    platform, tracked package versions, inputs, config, and extras.
  - `run_ids.py` generates `YYYYMMDD-HHMMSS-<slug>` run identifiers
    with deterministic slug sanitization.
  - `config.py` defines the project-wide Pydantic config schema and
    a YAML loader that rejects unknown keys.
- `tests/conftest.py` registering the `gpu` marker. The hook skips
  GPU-marked tests automatically when CUDA is unavailable, with a
  message pointing at `docs/remote-setup.md`.
- `tests/test_utils.py` exercising every utility module (run-ID
  generation, config defaults, config round-trip, config rejection
  of unknown keys, repo-default config validation, manifest content
  and parent-directory creation, loguru file logging).
- `configs/default.yaml` stub. The schema validates against
  `Config` so the YAML cannot drift from the Pydantic model without
  one of `tests/test_utils.py::test_load_config_repo_default` or
  `test_load_config_rejects_unknown_keys` failing.

## Audit fixes (post-initial)

A self-audit (see `docs/audits/00-init-audit.md`) caught four gaps and
applied the fixes here:

- Added READMEs to `runs/`, `data/`, `checkpoints/`, and
  `docs/progress/`. Updated `.gitignore` so the first three READMEs
  stay tracked even though the directories themselves stay ignored.
- Added phase-naming comments to each placeholder section in
  `configs/default.yaml`.
- Corrected the test count from 14 to 13 to match the actual
  `tests/test_utils.py` suite.

## What I did not do

- No `uv sync` ran locally. No GPU touched. No remote command issued.
- No data generated, no models trained.
- No git push. The local repo carries one commit on the default
  branch and waits for the user to push to GitHub.

## What the user runs to verify Phase 0

### Local laptop (no GPU)

Confirm the utility tests pass without a GPU:

```
cd FMLLMadvantage
uv sync --extra dev
uv run pytest -m "not gpu" -v
```

Expected outcome:
- `uv sync --extra dev` resolves the lock and installs the dev
  dependencies. On non-Linux hosts torch installs as the CPU build,
  which is fine for the utility tests.
- `pytest` collects 13 tests under `tests/test_utils.py` and reports
  every test as passed.

### Remote 4xH100 host

Pull the repo and run the bootstrap:

```
git clone https://github.com/colpark/FMLLMadvantage.git
cd FMLLMadvantage
bash scripts/remote_bootstrap.sh
```

Expected outcome (matches the block in `docs/remote-setup.md`):

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

Then run the same pytest invocation on the remote:

```
uv run pytest -m "not gpu" -v
```

Same expectation as on the local machine.

## What to send back

- Full stdout from `bash scripts/remote_bootstrap.sh`.
- Full stdout from `uv run pytest -m "not gpu" -v` on the remote.
- Any warnings or version-pin conflicts uv emitted during sync.
- Any deviation from the expected GPU verification block (different
  device count, different model name, different CUDA build).

## Known issues to flag

- The PyTorch wheel routing assumes a Linux remote. macOS and Windows
  fall back to the default PyPI wheel, which is the CPU build.
- The literature database for Phase 4 does not exist yet.
  `configs/default.yaml` carries the path as a placeholder.
- The bootstrap script reports a warning (not an error) if it sees
  fewer than 4 GPUs. That accommodates partial-availability hosts
  for development. The full pipeline assumes 4 GPUs.

## What remains for Phase 1

- Implement the LJ Hamiltonian, MD integrator, structure generators,
  and observables under `src/fmllm/physics/`.
- Implement the synthetic dataset generator, HDF5 dataset, and split
  logic under `src/fmllm/data/`.
- Write physics tests covering energy conservation, RDF
  normalization, permutation invariance, and rasterizer accuracy.
- Document the dataset format under `docs/data-format.md`.
