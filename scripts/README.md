# scripts/

CLI entry points and operational scripts. The user invokes everything
in this directory directly; nothing here imports from another script.

## Files

- `remote_bootstrap.sh` - idempotent bootstrap for the remote 4xH100
  host. Installs uv if missing, pins Python 3.11, syncs dependencies,
  and verifies all four GPUs.

Subsequent phases will add:
- `train_fm.py` - unified training CLI for FM1, FM2, FM3 (Phase 2).
- `run_pipeline.py` - end-to-end orchestration loop (Phase 5).
- `train_pipeline_b.py` - Pipeline B RL fine-tuning (Phase 6).
- `run_evaluation.py` - the eight world-model tests (Phase 7).
- `exp_e1_composition_curve.py` - E1 composition sweep (Phase 8).
- `exp_e2_train_vs_inference.py` - E2 Pipeline A vs B (Phase 8).
- `exp_e3_bridge_anchor.py` - E3 bridge comparison (Phase 8).
- `reproduce_all.sh` - end-to-end reproducibility harness (Phase 9).
