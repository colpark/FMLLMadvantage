# fmllm.verifier

Multi-source verifier with rule, literature, cross-FM, simulator, and
conformal sources. The integrator runs each enabled source against a
list of `BridgedFMOutput` objects plus the LLM's `PhysicalStateClaim`
and aggregates the per-source verdicts into a structured
`VerifierVerdict`.

## Files

- `schema.py` - `PhysicalStateClaim`, `SourceDecision`,
  `SourceVerdict`, `Hint`, `VerifierVerdict`, `SourcesConfig` (with
  E4 ablation presets V0..V4).
- `integrator.py` - `Verifier` class, `build_default_verifier`
  factory. Accepts a runtime `sources_config` for E4 ablation.
- `sources/rule_library.py` - registry of per-constraint check
  functions (`@register_check`). Default checks cover the seven
  constraints declared across FM1/FM2/FM3 metadata.
- `sources/literature.py` - lookup against
  `data/literature/clusters.json`.
- `sources/cross_fm.py` - reads dependency edges from bridged
  outputs and applies a calibrated `CrossFMToleranceMatrix` (or a
  hard-coded fallback).
- `sources/simulator.py` - short MD rollout from the claim's
  positions and temperature; compares per-atom mean KE to the
  claim and to FM3's derived temperature.
- `sources/conformal.py` - reads `source.in_distribution` and the
  calibrated band on `Prediction.uncertainty`.

## Usage

```python
from fmllm.verifier import (
    PhysicalStateClaim, SourcesConfig, build_default_verifier,
)
from fmllm.bridges import load_fm_context, make_structure_bridge

# 1. Build the verifier once.
verifier = build_default_verifier(
    literature_db_path="data/literature/clusters.json",
    tolerance_matrix_path="data/literature/cross_fm_tolerance_train_50k.yaml",
)

# 2. Bridge each FM's forward-pass result.
bridged = [
    make_structure_bridge(load_fm_context(fm_name="fm1_image", checkpoint_dir=...)).emit(...),
    make_structure_bridge(load_fm_context(fm_name="fm2_rdf", checkpoint_dir=...)).emit(...),
    make_structure_bridge(load_fm_context(fm_name="fm3_traj", checkpoint_dir=...)).emit(...),
]

# 3. Verify the LLM's typed claim.
claim = PhysicalStateClaim(n_atoms=7, motif="triangular_disk", temperature=0.5)
verdict = verifier.verify(bridged, claim)

# 4. Inspect.
print(verdict.aggregate_decision)         # "pass" / "caveat" / "fail" / "skip"
for sv in verdict.source_verdicts:
    print(sv.source_name, sv.decision, sv.message)
print(verdict.hint.flagged_sources, verdict.hint.suggested_revisions)
```

## E4 ablation

Pass a `SourcesConfig` to `verify` to override the default:

```python
from fmllm.verifier import SourcesConfig

verdict_v3 = verifier.verify(bridged, claim, sources_config=SourcesConfig.for_ablation("V3"))
```

Disabled sources contribute a `SKIP` verdict so the trace stays the
same shape across V0..V4.
