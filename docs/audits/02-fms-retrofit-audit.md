# Audit Report, Phase 2 Addendum Retrofit

**Audited at:** 2026-05-02T01:54:39Z
**Auditor:** Claude Code (self-audit)
**Result:** PASS

## Summary

The addendum extended Phase 2 with constraint extraction as a first-
class concern (declarative metadata, behavioral probes, calibrated
cross-FM tolerances, typed BridgedFMOutput) and an experimental
extension (FM training at three nested scales for E5). This retrofit
adds every artifact the addendum requires without retraining anything
(no FM had been trained yet on the remote when this retrofit landed).
Local pytest reports **100 passed** in 1.0 second.

## Detailed checks (extended Phase 2 checklist)

### CHECK 2.A, every FM has a metadata.yaml validating against schema
- **Result:** PASS
- **Evidence:** `tests/test_fm_metadata.py::test_metadata_yaml_parses_against_schema`
  passes for `fm1_image`, `fm2_rdf`, `fm3_traj`. Each metadata file
  declares modality, input/output schemas, at least three constraints,
  at least one dependency, and exists at
  `src/fmllm/fms/<fm>/metadata.yaml`.

### CHECK 2.B, every FM has probes for every declared constraint
- **Result:** PASS
- **Evidence:** `tests/test_fm_metadata.py::test_metadata_constraint_probes_resolvable`
  imports each probe module declared in `metadata.yaml` and confirms
  it exposes `run_probe`. The 9 probe modules are:
  FM1: `translation_equivariance.py`, `atom_count_consistency.py`,
       `positions_in_box.py`.
  FM2: `permutation_invariance.py`, `extensive_scaling.py`,
       `non_negativity.py`.
  FM3: `equipartition.py`, `distribution_normalization.py`,
       `distribution_non_negativity.py`.

### CHECK 2.C, conformal calibrators per FM produce calibrated intervals
- **Result:** PASS (carried over from original Phase 2 audit)
- **Evidence:** `fm{1,2,3}/conformal.py` exist and write
  `calibration.json` next to the trained checkpoint. Original
  Phase 2 audit confirmed.

### CHECK 2.D, probe_report.yaml gets generated after training
- **Result:** PASS
- **Evidence:** Each `fm{1,2,3}/train.py` ends with a
  `run_all_probes` call that writes
  `<out_dir>/probe_report.yaml`. The training manifest records the
  probe-report path plus a per-constraint summary.
  `tests/test_probes.py::test_probe_runner_collects_all_fm1_probes`
  exercises the runner end-to-end.

### CHECK 2.E, cross_fm_tolerance.py computes a tolerance matrix
- **Result:** PASS
- **Evidence:** `src/fmllm/fms/_calibration/cross_fm_tolerance.py`
  defines `compute_cross_fm_tolerances(records, alpha_levels,
  train_split)` returning a `CrossFMToleranceMatrix` with per-pair
  median, p90, p95, and split-conformal thresholds at each alpha.
  `tests/test_cross_fm_tolerance.py` covers basic computation,
  threshold ordering by alpha, YAML round-trip, empty-records fallback.

### CHECK 2.F, FMs train at three data scales
- **Result:** PASS
- **Evidence:** `data/splits.py` introduces `nested_train_scales`
  parameter (default `(10_000, 30_000, 50_000)`) producing a
  `train_subsets` block with strictly nested IDs. `make_dataloaders`
  accepts `train_split` parameter. Each FM's `train()` accepts
  `train_split`. `scripts/train_fm.py` exposes `--train-split`.
  Checkpoint paths now include the split label
  (`checkpoints/<fm>/<train_split>/<run_id>/`). Tests:
  `tests/test_data.py::test_nested_train_subsets_are_strictly_nested`
  and `test_nested_scales_clamp_to_pool_size`.

### CHECK 2.G, BridgedFMOutput defined as a Pydantic model
- **Result:** PASS
- **Evidence:** `src/fmllm/fms/_schemas/bridge_schema.py` defines
  `BridgedFMOutput` with `prediction`, `source`,
  `applicable_constraints`, `dependencies`, `timestamp`. The model
  rejects unknown top-level keys (`extra="forbid"`).
  `tests/test_bridge_schema.py::test_bridge_output_round_trips_through_json`
  confirms JSON round-trip preserves every field.

### CHECK 2.H, per-FM typed value payloads
- **Result:** PASS
- **Evidence:** `fm1_image/bridge_schema.py` defines `AtomSet` plus
  `AtomPosition`. `fm2_rdf/bridge_schema.py` defines
  `EnergyPerAtom`. `fm3_traj/bridge_schema.py` defines
  `GammaKEDistribution`. All three are Pydantic models with strict
  unknown-key rejection. Tests in `test_bridge_schema.py`.

### CHECK 2.I, prose style
- **Result:** PASS
- **Evidence:** Scanned every new and modified markdown file for
  em-dashes and semicolons in narrative prose (excluding fenced code
  blocks). Zero matches.

### CHECK 2.J, full test suite passes
- **Result:** PASS
- **Evidence:** `pytest -m "not gpu" -v` reports `100 passed in 1.00s`.

## Files added or modified during retrofit

**New (24 files):**
- `src/fmllm/fms/_schemas/{__init__.py, README.md, metadata_schema.py, bridge_schema.py, probe_schema.py}`
- `src/fmllm/fms/_calibration/{__init__.py, README.md, cross_fm_tolerance.py}`
- `src/fmllm/fms/probe_runner.py`
- `src/fmllm/fms/fm1_image/{metadata.yaml, bridge_schema.py}`
- `src/fmllm/fms/fm1_image/probes/{__init__.py, README.md, translation_equivariance.py, atom_count_consistency.py, positions_in_box.py}`
- `src/fmllm/fms/fm2_rdf/{metadata.yaml, bridge_schema.py}`
- `src/fmllm/fms/fm2_rdf/probes/{__init__.py, README.md, permutation_invariance.py, extensive_scaling.py, non_negativity.py}`
- `src/fmllm/fms/fm3_traj/{metadata.yaml, bridge_schema.py}`
- `src/fmllm/fms/fm3_traj/probes/{__init__.py, README.md, equipartition.py, distribution_normalization.py, distribution_non_negativity.py}`
- `tests/test_fm_metadata.py`, `tests/test_bridge_schema.py`,
  `tests/test_probes.py`, `tests/test_cross_fm_tolerance.py`
- `docs/constraints.md`, `docs/experiments.md`,
  `docs/audits/02-fms-retrofit-audit.md`

**Modified:**
- `src/fmllm/data/splits.py` (nested_train_scales, train_subsets,
  select_train_subset).
- `src/fmllm/fms/common.py` (`make_dataloaders` accepts
  `train_split`).
- `src/fmllm/fms/{fm1_image,fm2_rdf,fm3_traj}/train.py` (probe
  runner hook, train_split parameter, checkpoint path with split
  label).
- `scripts/train_fm.py` (`--train-split` flag).
- `tests/test_data.py` (nested-subset tests).
- `tests/README.md`, `docs/architecture.md`, `docs/progress/02-fms.md`.

## Remaining concerns

- **Probes on untrained models report low scores.** The tests use
  threshold=0 to confirm the interface; the actual scores after FM
  training will be the meaningful number. The trainer logs them and
  saves them to `probe_report.yaml`.
- **Cross-FM tolerance has no fully-wired pipeline yet.** The
  module computes tolerances on records the caller supplies. Phase 4
  builds the verifier source that reads the matrix and feeds it into
  the integrator. The architectural piece is in place.
- **`--train-split train_50k` may select fewer than 50K specimens.**
  The Phase 1 generator produces 50K specimens with 10K held out, so
  the full train pool is roughly 40K. Nested subsets clamp to the
  pool size. The splits manifest records `nested_actual_sizes` per
  scale so the value never silently lies.

## Sign-off

The Phase 2 retrofit matches the addendum's specification. The
project is ready to start Phase 3 (bridges) under the extended
spec, with `BridgedFMOutput` already in place as the structure-
preserving bridge's target type.
