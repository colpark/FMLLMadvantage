# fmllm.fms._schemas

Shared Pydantic schemas for the three-layer constraint extraction
pipeline. All FMs and downstream consumers (bridges, verifier,
evaluation harness) pull these classes from here, so the contract
stays single-sourced.

## Files

- `metadata_schema.py` - `FMMetadata`, `InputSchema`, `OutputSchema`,
  `ConstraintDeclaration`, `DependencyDeclaration`, plus
  `load_fm_metadata(path)` for the per-FM `metadata.yaml`.
- `probe_schema.py` - `ProbeResult`, `ProbeReport`, plus
  `save_probe_report` / `load_probe_report` for the per-FM
  `probe_report.yaml` written after training.
- `bridge_schema.py` - `BridgedFMOutput`, `Prediction`, `Source`,
  `Uncertainty`, `ApplicableConstraint`, `BridgedDependency`. The
  structure-preserving bridge produces objects of this type. JSON
  round-trip preserves all fields.

## Layering

The three layers map cleanly to schemas:

- **Layer 1 (declarative)**: `metadata.yaml` validates against
  `FMMetadata`.
- **Layer 2 (calibrated)**: conformal calibration files
  (`calibration.json`) sit alongside checkpoints. The Pydantic
  schema for these stays light (calibration is a flat dict).
- **Layer 3 (behavioral)**: probes emit `ProbeResult`, aggregated
  into `ProbeReport`, saved as `probe_report.yaml`.

The bridge composes all three layers into a `BridgedFMOutput` per
prediction at runtime.
