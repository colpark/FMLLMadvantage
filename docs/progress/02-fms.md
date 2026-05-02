# Phase 2: The Three Foundation Models

## What I built

### FM common utilities (`src/fmllm/fms/common.py`)

- `make_dataloaders(...)` opens the dataset, splits the training pool
  into train / val / calibration with a deterministic seed, and
  returns three PyTorch DataLoaders plus the split-IDs dict.
- `make_optimizer_and_schedule(...)` builds AdamW with separate
  weight-decay / no-decay parameter groups, plus a per-step linear
  warmup -> half-cosine schedule.
- `save_checkpoint` / `load_checkpoint` round-trip model + optimizer
  + scheduler state.
- `split_conformal_quantile(scores, alpha)` applies the standard
  finite-sample correction.
- `write_conformal_calibration` / `read_conformal_calibration`
  round-trip a JSON file with the per-FM thresholds.
- `per_atom_potential_energy(positions, mask, k_conf)` evaluates the
  LJ pair sum with pair-level masking plus the harmonic contribution
  from real atoms only. Used as the FM2 training target.
- `kinetic_energies_masked(velocities, mask)` returns per-(atom,
  frame) KE plus the broadcast mask.

### FM1: image Vision Transformer (`src/fmllm/fms/fm1_image/`)

- `model.py:FM1ImageViT`. Conv-based patch embedding (8x8 patches on
  64x64 input) feeds a Transformer encoder; learned object queries
  cross-attend to the encoded patches via a Transformer decoder. The
  CLS token feeds a count head, each query feeds position and
  confidence heads.
- `train.py`. Sums count cross-entropy, Hungarian-matched L2 position
  loss, BCE objectness loss, and a soft box-constraint penalty.
  AdamW + warmup-cosine schedule, optional mixed precision, validates
  every epoch, saves best by val total loss.
- `conformal.py`. Per-pair L2 position errors on the calibration
  subset, threshold per `conformal_alpha_levels`.
- Default config produces **10.71M parameters** (target 10-30M).
- Translation equivariance: exact for translations by integer
  multiples of `patch_size`; approximate otherwise (documented in
  `model.py` and the README).

### FM2: RDF transformer (`src/fmllm/fms/fm2_rdf/`)

- `model.py:FM2RDFTransformer`. Bin embedding + learned positional
  embedding feed a Transformer encoder with a CLS token. The CLS
  output passes through an MLP energy head.
- `train.py`. Huber loss against per-atom potential energy plus a
  soft non-negativity penalty against the LJ energy floor.
- `conformal.py`. Absolute residuals on the calibration subset.
- Default config produces **7.57M parameters** (target 5-15M).
- Permutation invariance: automatic (input g(r) is permutation
  invariant by construction).
- Extensive scaling: by output design (model predicts per-atom
  energy).

### FM3: trajectory transformer (`src/fmllm/fms/fm3_traj/`)

- `model.py:FM3TrajTransformer`. A small MLP encodes
  `(x, y, vx, vy)` per atom per frame. Per-frame masked-mean and
  masked-max pool over real atoms (permutation invariant). A learned
  temporal positional embedding plus a CLS token feed a Transformer
  encoder. The head emits `(log alpha, log beta)`, then `softplus`
  enforces positivity.
- `train.py`. Negative log-likelihood of the empirical KE
  distribution under `Gamma(alpha, beta)` plus a soft equipartition
  penalty pulling `alpha * beta` toward observed mean KE.
- `conformal.py`. Per-specimen mean NLL on the calibration subset.
- Default config produces **12.78M parameters** (target 10-25M).
- Permutation invariance: by construction (verified by test).
- Equipartition prior: soft training penalty.

### Unified CLI (`scripts/train_fm.py`)

Typer-based CLI with `--fm {fm1, fm2, fm3}` plus `--config`,
`--h5-path`, `--splits-path`, `--device`, `--epochs`,
`--calibrate-only`, `--checkpoint`. Reuse the same script for
training and conformal calibration after training completes.

### Configuration

- Replaced `FMConfig` with `FM1Config`, `FM2Config`, `FM3Config`
  inheriting common training fields and adding per-architecture
  hyperparameters. The Pydantic schema rejects unknown keys, so the
  YAML and the schema move together.
- `configs/default.yaml` includes architectures, training
  hyperparameters, and `conformal_alpha_levels` per FM.

### Tests (`tests/test_fms.py`)

- FM1: forward-pass shape (4D and 3D inputs), Hungarian matching
  correctness on a hand-crafted case, box-constraint loss sign
  (positive outside the box, zero inside), and total-loss
  composition.
- FM2: forward-pass shape, validation that wrong bin-count input
  errors out, non-negativity loss only fires below floor, and the
  per-atom potential energy target is finite for clusters of
  different `N`.
- FM3: forward-pass shape, atom-permutation invariance (different
  ordering produces identical output to within `1e-5`), Gamma NLL
  matches `torch.distributions.Gamma`, equipartition loss responds
  to mismatch.
- Common: `split_conformal_quantile` behavior on a known
  distribution, calibration JSON round-trip.

All 66 tests pass locally on CPU in roughly 1 second.

## What the user runs to verify Phase 2

### Local laptop (no GPU)

```
git pull
uv sync --extra dev
uv run pytest -m "not gpu" -v
```

Expect 66 passing tests.

### Remote 4xH100 host

#### Step 1. Smoke-train each FM for 1 epoch

Each FM takes a small subset and runs for one epoch to confirm the
training loop works end to end:

```
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_fm.py \
    --fm fm1 --config configs/default.yaml --epochs 1 \
    --h5-path data/synthetic_lj_v1/specimens.h5 \
    --splits-path data/synthetic_lj_v1/splits.yaml

CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_fm.py \
    --fm fm2 --config configs/default.yaml --epochs 1 \
    --h5-path data/synthetic_lj_v1/specimens.h5 \
    --splits-path data/synthetic_lj_v1/splits.yaml

CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_fm.py \
    --fm fm3 --config configs/default.yaml --epochs 1 \
    --h5-path data/synthetic_lj_v1/specimens.h5 \
    --splits-path data/synthetic_lj_v1/splits.yaml
```

Expect each smoke run to complete inside a few minutes and write a
checkpoint plus manifest under
`checkpoints/<fm_name>/<run_id>/`.

#### Step 2. Train all three FMs to convergence in parallel

Launch the three trainings on GPUs 0, 1, 2 simultaneously:

```
( CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_fm.py --fm fm1 \
        --config configs/default.yaml \
        --h5-path data/synthetic_lj_v1/specimens.h5 \
        --splits-path data/synthetic_lj_v1/splits.yaml \
        > runs/fm1.log 2>&1 ) &
( CUDA_VISIBLE_DEVICES=1 uv run python scripts/train_fm.py --fm fm2 \
        --config configs/default.yaml \
        --h5-path data/synthetic_lj_v1/specimens.h5 \
        --splits-path data/synthetic_lj_v1/splits.yaml \
        > runs/fm2.log 2>&1 ) &
( CUDA_VISIBLE_DEVICES=2 uv run python scripts/train_fm.py --fm fm3 \
        --config configs/default.yaml \
        --h5-path data/synthetic_lj_v1/specimens.h5 \
        --splits-path data/synthetic_lj_v1/splits.yaml \
        > runs/fm3.log 2>&1 ) &
wait
```

Expected runtime per FM on a single H100 with default config and
50K specimens:

| FM | Per-epoch time | 50 epochs | Notes |
|----|----------------|-----------|-------|
| FM1 image ViT | 1.5 to 3 minutes | 1.5 to 3 hours | 10.71M params, batch 64 |
| FM2 RDF transformer | 30 to 60 seconds | 30 to 60 minutes | 7.57M params, batch 128 |
| FM3 trajectory transformer | 1 to 2 minutes | 1 to 2 hours | 12.78M params, batch 32 |

The exact runtime depends on H100 memory bandwidth and dataloader
overhead. Three FMs in parallel finish inside a single 4xH100 day.

#### Step 3. Conformal calibration

After each FM converges, run the calibration step against the same
checkpoint:

```
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_fm.py \
    --fm fm1 --calibrate-only \
    --checkpoint checkpoints/fm1_image/<run_id>/model.pt \
    --config configs/default.yaml \
    --h5-path data/synthetic_lj_v1/specimens.h5 \
    --splits-path data/synthetic_lj_v1/splits.yaml

# Repeat for fm2 and fm3.
```

The script writes `calibration.json` next to each checkpoint with
thresholds at every alpha in `conformal_alpha_levels`.

## Target validation metrics

Targets come from the pre-registered evaluation framework. Treat as
proximate goals; they are not pass / fail thresholds.

- **FM1**: validation atom-count accuracy above 95% on the in-
  distribution split. Median position L2 error per matched pair below
  one pixel (`pixel_size_lj = 0.15`, so target below 0.15 LJ).
- **FM2**: validation MAE on per-atom potential energy below 0.10 LJ
  per atom on the in-distribution split.
- **FM3**: validation Gamma NLL within 0.5 of the empirical-fit NLL
  baseline (the model prediction sits close to the empirical fit on
  in-distribution specimens).

Conformal coverage on the calibration set comes for free from the
split-conformal procedure (coverage approximately matches `1 - alpha`
by construction).

## What to send back

- `runs/fm1.log`, `runs/fm2.log`, `runs/fm3.log`.
- `checkpoints/fm1_image/<run_id>/manifest.yaml` (and analogous for
  fm2, fm3). The manifest records training history, parameter count,
  best validation metrics, and split sizes.
- The three `calibration.json` files after the conformal step.
- The exact command lines used and any deviations from the recipe
  above. Useful for reproducibility if metrics drift.

## Known issues to flag

- The `enable_nested_tensor is True ...` UserWarning from PyTorch's
  `nn.TransformerEncoder` is benign. It fires because we use
  `norm_first=True` in our encoder layers.
- The trainer assumes the dataset HDF5 file has the layout the Phase 1
  generator writes. Older datasets need regeneration.
- Mixed precision is `True` by default. If CUDA versions on the
  remote disagree with torch's AMP support, set `mixed_precision:
  false` per FM in the config and re-run.
- The FM1 Hungarian matcher uses scipy on CPU per batch. For batch
  sizes above 256 the matcher dominates step time. Reduce
  `batch_size` if profiling shows the dataloader stalls.

## What remains for Phase 3

- Implement the language-anchored and structure-preserving bridges
  under `src/fmllm/bridges/` plus a shared base class.
- Define the typed JSON schemas the structure-preserving bridge emits
  per FM.
- Write the bridge tests (round-trip serialization, parseable
  captions on a sample specimen).
