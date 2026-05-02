# scripts/

CLI entry points and operational scripts. The user invokes everything
in this directory directly; nothing here imports from another script.

## Files

- `remote_bootstrap.sh` - idempotent bootstrap for the remote 4xH100
  host. Installs uv if missing, pins Python 3.11, syncs dependencies,
  and verifies all four GPUs.
- `train_fm.py` - unified CLI for FM1, FM2, FM3 training and conformal
  calibration. Use `--fm fm1` (or fm2 / fm3) and pass paths to the
  HDF5 dataset and splits YAML. See `docs/progress/02-fms.md` for the
  parallel-launch recipe across GPUs 0, 1, 2.
- `train_fm_sweep.sh` - convenience driver that trains FM1, FM2, FM3
  in parallel across GPUs 0, 1, 2 for one or more training scales.
  Defaults to the full E5 sweep (`train_10k`, `train_30k`,
  `train_50k`). Pass scale names as positional arguments to override.
- `verify_sweep.sh` - checks the output of `train_fm_sweep.sh`. Per
  scale: confirms each per-FM log finished, lists checkpoint
  artifacts on disk, prints probe satisfaction scores side-by-side
  for at-a-glance comparison across scales. Exits non-zero if any
  run did not complete.
- `calibrate_fms.sh` - Stage 3 conformal calibration. Locates the
  latest `model.pt` under `checkpoints/<fm>/<scale>/<run_id>/` for
  each (fm, scale) pair and runs `train_fm.py --calibrate-only`.
  Writes `calibration.json` next to each checkpoint.
- `save_data_samples.py` - saves an inspectable subset of the
  synthetic dataset. Per specimen: rasterized image PNG, RDF plot,
  initial-position scatter, trajectory overlay, summary YAML. Plus
  a multi-panel grid of images and RDFs. Stratify by atom count or
  pass explicit `--indices` to pick exact specimens. CPU-only.
- `verify_bridges.py` - Phase 3 wiring smoke. Loads `FMContext` for
  the latest run-id under each `checkpoints/<fm>/<scale>/` and
  emits a synthetic-input bridged output (both flavors) plus a
  context snapshot per FM. Confirms metadata + probe report +
  calibration compose into working bridges. CPU-only.
- `verify_bridges.sh` - bash wrapper for `verify_bridges.py` that
  loops over one or more training scales (defaults to all three).
- `build_literature_db.py` - regenerates
  `data/literature/clusters.json` from canonical structures via the
  project's LJ Hamiltonian. Deterministic. CPU-only.
- `run_pipeline.py` - Phase 5 end-to-end CLI. Loads each FM model
  from the latest checkpoint at the requested training scale,
  builds bridges and the verifier, and runs the OHVD loop on one
  specimen. Default LLM is Llama 3.1 8B Instruct via
  `transformers`. Supports `--mock-script` for smoke tests without
  LLM weights.
- `run_pipeline_smoke.sh` - bash wrapper around `run_pipeline.py`
  with the mock LLM. Pass specimen ID and (optional) train_split
  as positional args (defaults: 42, train_50k). Tail-calls
  `inspect_trajectory.sh` on the resulting run.
- `run_pipeline_real.sh` - bash wrapper around `run_pipeline.py`
  with the real chat LLM (Llama 3.1 8B Instruct by default; swap
  via `LLM_MODEL`). First run downloads ~16 GB of weights.
- `inspect_trajectory.sh` - pretty-prints a saved
  `trajectory.json`. With no arg it picks the latest pipeline-A
  run; with one arg accepts either a run directory or an explicit
  file path.
- `mock_scripts/` - hard-coded LLM response sequences for
  `--mock-script` smoke runs.
- `collect_trajectories.py` - Phase 6 trajectory collector. Runs
  Pipeline A across a range of specimens and writes JSONL plus a
  summary.
- `collect_trajectories.sh` - bash wrapper around
  `collect_trajectories.py`. First arg `--real` switches from the
  mock LLM to Llama 3.1 8B (or any `LLM_MODEL`). Positional args
  are start, count, train_split.
- `train_pipeline_b.py` - Phase 6 unified Pipeline B trainer.
  `--mode {sft, dpo, grpo}` dispatches to the matching trainer
  with LoRA adapters.
- `train_pipeline_b.sh` - bash wrapper that picks the latest
  `trajectories.jsonl` under `runs/trajectories/` unless
  `TRAJECTORIES` env overrides.

Subsequent phases will add:
- `run_evaluation.py` - the eight world-model tests (Phase 7).
- `exp_e1_composition_curve.py` - E1 composition sweep (Phase 8).
- `exp_e2_train_vs_inference.py` - E2 Pipeline A vs B (Phase 8).
- `exp_e3_bridge_anchor.py` - E3 bridge comparison (Phase 8).
- `exp_e4_verifier_ablation.py` - E4 verifier ablation (Phase 8).
- `exp_e5_fm_quality_sweep.py` - E5 FM quality sweep (Phase 8).
- `reproduce_all.sh` - end-to-end reproducibility harness (Phase 9).
