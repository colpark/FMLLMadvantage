# `fmllm.evaluation` — eight world-model tests

This subpackage operationalizes the eight world-model evaluation
checks called for by the spec, organized into two layers and a
cross-layer pair.

| Layer        | Test                          | Module                              | Direction | Default threshold |
|--------------|-------------------------------|-------------------------------------|-----------|-------------------|
| trajectory   | `trajectory_compression`      | `trajectory_compression.py`         | `<=`      | action 2.0, claim 1.0 |
| trajectory   | `trajectory_distinction`      | `trajectory_distinction.py`         | `>=`      | action 1.0        |
| trajectory   | `step_recoverability`         | `step_recoverability.py`            | `>=`      | 0.70              |
| prediction   | `prediction_compression`      | `prediction_compression.py`         | `<=`      | 1.0               |
| prediction   | `prediction_distinction`      | `prediction_distinction.py`         | `>=`      | 2.0               |
| prediction   | `goal_competence`             | `goal_competence.py`                | `<=`      | 0.50 failure rate |
| cross_layer  | `federated_factorability`     | `federated_factorability.py`        | `>=`      | 0.45 score        |
| cross_layer  | `calibrated_uncertainty`      | `calibrated_uncertainty.py`         | `<=`      | 0.10 gap          |

Each module exposes a single function, `measure(...)`, returning a
`TestResult` (see `schema.py`). Tests gracefully skip rather than
fail when input is degenerate (only one equivalence class present,
no conformal verdicts in the trajectories, etc).

## Inputs

`measure(...)` reads:

* `trajectories: list[Trajectory]` — JSONL output of
  `scripts/collect_trajectories.py`. Each trajectory carries its
  `specimen_id`, full `steps` list (including bridged FM outputs and
  per-step verdicts), and `final_verdict.aggregate_decision`.
* `truth: dict[int, dict[str, Any]]` — per-specimen ground truth
  loaded from `data/synthetic_lj_v1/specimens.h5` via
  `utils.load_ground_truth`. Keys are `n`, `t`, `motif`.
* `trajectories_by_ablation: dict[str, list[Trajectory]]` (only
  `federated_factorability`) — trajectories collected under each
  ablation preset `V0..V4`.

## Output: `EvaluationReport`

`EvaluationReport` aggregates the eight `TestResult` objects under
three buckets (`trajectory_results`, `prediction_results`,
`cross_layer_results`) plus a single `aggregate_pass` flag. The CLI
runner serializes the report as `runs/eval/<run_id>/report.yaml`
alongside a manifest.

## Pre-registered thresholds

Every threshold is registered in code as the default argument of
`measure(...)`. Override per call when running an ablation by
passing `threshold=...`, `claim_threshold=...`, etc.

The thresholds are calibrated to the small synthetic testbed: a 2D
LJ system with three motifs and atom counts in `[5, 25]`, so claim
distances rarely exceed 5–10 units. For larger benchmarks rebuild
thresholds against the new distribution before relying on
pass/fail flags.

## Running

```bash
# single ablation
bash scripts/run_evaluation.sh                                  # latest trajectories
bash scripts/run_evaluation.sh path/to/trajectories.jsonl       # explicit

# ablation lattice (federated factorability needs >=2 ablations)
bash scripts/run_evaluation.sh ablation V0=runs/.../V0.jsonl V4=runs/.../V4.jsonl
```

The harness writes:

```
runs/eval/<run_id>/
├── report.yaml      # EvaluationReport
└── manifest.yaml    # config + summary counts
```

## Skip semantics

`make_skipped(...)` constructs a `TestResult` with `skipped=True`,
`passes=False`, and a populated `skip_reason`. The aggregate pass
flag in `EvaluationReport` ignores skipped tests (they neither help
nor block a pass), but a report where every test is skipped still
fails because nothing was measured.

## Extending

To add a new test:

1. Drop `<my_test>.py` next to the existing modules. Implement
   `measure(...) -> TestResult`. Use `make_skipped(...)` and
   `threshold_check(...)` from `schema.py`.
2. Re-export it from `__init__.py`.
3. Register the call in `scripts/run_evaluation.py::_run_all_tests`,
   placing it under the right layer bucket.
4. Cover the new module under `tests/test_evaluation.py` with a
   pass, a fail, and a skip case.
