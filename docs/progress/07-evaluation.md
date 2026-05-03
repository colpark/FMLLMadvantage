# Phase 7: World-Model Evaluation

## What I built

The eight world-model evaluation tests live in
`src/fmllm/evaluation/`. Each test exposes a single `measure(...)`
function returning a typed `TestResult`. The CLI runner aggregates
all eight into an `EvaluationReport` serialized to YAML.

### `fmllm.evaluation`

- `schema.py` - `TestResult` and `EvaluationReport` Pydantic models,
  plus `make_skipped(...)` for degenerate-input cases and
  `threshold_check(...)` for `ge`/`le`/`eq` direction-aware
  comparisons.
- `utils.py` - shared helpers:
  - `extract_observations(traj)` - latest bridged output per FM.
  - `extract_final_claim(traj)` - the trajectory's commit claim.
  - `trajectory_action_signature(traj)` - coarse string sequence
    summarizing the actions taken (`call_fm:fm1_image`,
    `hypothesize`, `commit`).
  - `trajectory_outcome(traj)` - aggregate decision string.
  - `load_ground_truth(h5_path, specimen_ids)` - reads `atom_counts`,
    `temperatures`, and motif name from the dataset HDF5.
  - `physical_equivalence_class(truth)` - `(N, motif)` tuple,
    ignoring temperature.
  - `edit_distance(a, b)` - Levenshtein over action signatures.
  - `claim_distance(a, b)` - weighted distance over typed claims.

Layer 1 (trajectory-level):

- `trajectory_compression.py` - within-class median pairwise edit
  distance across action signatures and claims.
  Threshold: action `<= 2.0`, claim `<= 1.0`.
- `trajectory_distinction.py` - across-class median pairwise
  distance over up to N random pairs from different classes.
  Threshold: action `>= 1.0`.
- `step_recoverability.py` - fraction of commit/hypothesis steps
  whose typed claim agrees with FM observations within tolerance
  (atom-count off-by-2, temperature 25% rel, energy 0.5 abs).
  Threshold: `>= 0.70`.

Layer 2 (prediction-level):

- `prediction_compression.py` - mean within-class claim distance,
  averaged across classes with at least two members.
  Threshold: `<= 1.0`.
- `prediction_distinction.py` - median across-class claim distance
  over sampled pairs. Threshold: `>= 2.0`.
- `goal_competence.py` - failure rate across four goals (size,
  temperature, motif, size-and-motif). Threshold: `<= 0.50`.

Cross-layer:

- `federated_factorability.py` - over the V0..V4 ablation lattice,
  measures monotonicity of pass-rate gains (fraction of adjacent
  ablations where the rate improves) and step factor (largest single
  gain divided by total V0->V4 gain). Score is the unweighted
  average of monotonicity and `1 - step_factor`. Test passes when
  score `>= 0.45` AND monotonicity `>= 0.75` AND step factor
  `<= 0.85`.
- `calibrated_uncertainty.py` - reads conformal verdicts from every
  step's verifier output, groups per-FM `flag` entries, computes
  empirical clean rate (`flag == "ok"` over total) per FM, and
  reports the mean absolute gap to the claimed coverage `1 - alpha
  = 0.90`. Threshold: `<= 0.10`.

### Tests (`tests/test_evaluation.py`)

- Schema helpers: `threshold_check` directions, `make_skipped`,
  `claim_distance`, `edit_distance`, `physical_equivalence_class`.
- Layer 1: trajectory compression passes on clones, skips when no
  class has pairs; trajectory distinction separates distant
  classes; step recoverability passes when claim matches FM
  signals, skips when no observation steps.
- Layer 2: prediction compression clusters within class; prediction
  distinction separates distant classes; goal competence counts
  per-goal correctly.
- Cross-layer: federated factorability skips with one ablation,
  passes on a monotone V0..V4 lattice, fails on a brittle lattice
  where all gain concentrates in one transition; calibrated
  uncertainty reads per-FM flags from conformal verdicts and
  reports zero gap when 18/20 are clean, skips without conformal.

### CLIs

- `scripts/run_evaluation.py` - typer-based runner. Two modes:
  `--trajectories <path>` for a single set, or repeated
  `--ablation KEY=PATH` for a lattice (federated factorability
  needs `>= 2` ablations). Loads ground truth only for specimens
  observed across all input files, runs every test, prints a
  result table, writes `report.yaml` plus `manifest.yaml`.
- `scripts/run_evaluation.sh` - bash wrapper. With no argument
  picks the latest `runs/trajectories/*/trajectories.jsonl`. With
  a path argument runs against that file. With `ablation V0=...
  V1=...` runs lattice mode.

## What the user runs to verify Phase 7

### Local laptop (no GPU)

```
git pull
uv sync --extra dev
uv run pytest -m "not gpu" -v
```

Phase 7 contributes new tests under `tests/test_evaluation.py`.
They construct trajectories by hand and exercise every `measure`
function, including pass, fail, and skip paths.

### Remote 4xH100 host

After Phase 6 has produced trajectories under `runs/trajectories/`:

```
ssh remote
cd ~/FMLLMadvantage
git pull && uv sync --extra dev

# single-ablation run on the latest collection
bash scripts/run_evaluation.sh

# ablation lattice once V0..V4 collections exist
bash scripts/run_evaluation.sh ablation \
    V0=runs/trajectories/<id-V0>/trajectories.jsonl \
    V1=runs/trajectories/<id-V1>/trajectories.jsonl \
    V2=runs/trajectories/<id-V2>/trajectories.jsonl \
    V3=runs/trajectories/<id-V3>/trajectories.jsonl \
    V4=runs/trajectories/<id-V4>/trajectories.jsonl
```

The runner prints a table:

```
test                          layer          metric         threshold   status     samples
trajectory_compression        trajectory     0.5000         le 2.000    PASS       42
trajectory_distinction        trajectory     2.5000         ge 1.000    PASS       18
step_recoverability           trajectory     0.8200         ge 0.700    PASS       210
prediction_compression        prediction     0.7500         le 1.000    PASS       28
prediction_distinction        prediction     3.4000         ge 2.000    PASS       20
goal_competence               prediction     0.3300         le 0.500    PASS       72
federated_factorability       cross_layer    n/a            ge 0.450    SKIP       0
calibrated_uncertainty        cross_layer    0.0500         le 0.100    PASS       240
AGGREGATE: pass=7, fail=0, skip=1, all-pass=yes
```

The `report.yaml` and `manifest.yaml` land in
`runs/eval/<run_id>/`.

## Where Phase 7 fits

Phase 7 closes the loop on the OHVD pipeline. Phases 0–4 built the
testbed, the FMs, the bridges, and the verifier. Phase 5 wired the
orchestrator. Phase 6 fine-tuned the LLM on verified trajectories.
Phase 7 measures whether the resulting system has world-model
properties: trajectories collapse onto canonical sequences within
each physical-equivalence class, predictions cluster tightly
within classes and separate across them, individual steps
recoverable from FM observations, capability degrades gracefully
under ablation, uncertainty bands cover ground truth at the
claimed level. Phase 8 will run the experiment matrix and use
these reports as evidence.
