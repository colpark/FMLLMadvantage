# Constraint Extraction

This project commits to one architectural principle: constraints live
with the foundation models, not in an external knowledge base. Each
FM ships its own declared constraints, calibrated reliability bounds,
and behavioral probes. The bridge layer composes these into typed
output objects. The verifier integrates constraints from FM bundles
rather than supplying them.

## Three layers

### Layer 1, declarative metadata

Each FM ships a `metadata.yaml` next to its model code that declares
the FM's modality, input and output schemas, the physics constraints
its predictions should respect, and the dependency edges between its
prediction and other causal variables in the system.

The schema lives in `src/fmllm/fms/_schemas/metadata_schema.py`. Top-
level fields:

- `name`, `version`, `modality`
- `input_schema` (shape, dtype, normalization description)
- `output_schema` (output type, semantic name, units, value range)
- `physics_constraints` (list of `ConstraintDeclaration`)
- `dependencies` (list of `DependencyDeclaration`)
- `references` (free-form list of papers or sources)

A `ConstraintDeclaration` carries `name`, `type` (`hard` or `soft`),
`description`, `expected_satisfaction` in [0, 1], and `probe` (a
dotted Python path to the probe module).

A `DependencyDeclaration` carries `target_variable`, `relationship`
(`derives`, `scales_with`, `implies`, `requires`), `confidence` in
[0, 1], and a free-form `description`.

### Layer 2, learned calibration

After training, two calibrations attach to each FM.

The first is per-FM: a split-conformal calibrator wraps the trained
model with calibrated confidence intervals on the validation set.
Phase 2 records these in `calibration.json` next to the checkpoint.
The verifier's `conformal` source reads them to flag low-confidence
predictions.

The second is cross-FM: after all three FMs at a given training
scale finish, `cross_fm_tolerance.compute_cross_fm_tolerances`
measures pairwise empirical agreement on shared causal variables and
saves a `CrossFMToleranceMatrix`. The verifier's `cross_fm` source
reads the matrix to decide whether two FMs disagree more than the
calibration anticipates.

### Layer 3, behavioral probes

For every constraint declared in `metadata.yaml`, a probe module
under `fmllm/fms/<fm>/probes/` exposes
`run_probe(model, items, device, config) -> ProbeResult`. Each
`ProbeResult` carries `constraint_name`, `satisfaction_score` in
[0, 1], `num_test_cases`, `metric` (a short label), `passes_threshold`,
`threshold`, and `details` (a free-form dict).

After training, the trainer calls `fmllm.fms.probe_runner.run_all_probes`
on the validation items, collects every probe's result into a
`ProbeReport`, and saves `probe_report.yaml` next to the checkpoint.

## Per-FM examples

### FM1 (image -> atom positions)

```yaml
physics_constraints:
  - name: translation_equivariance
    type: hard
    expected_satisfaction: 0.90
    probe: fmllm.fms.fm1_image.probes.translation_equivariance

  - name: atom_count_consistency
    type: hard
    expected_satisfaction: 0.95
    probe: fmllm.fms.fm1_image.probes.atom_count_consistency

  - name: positions_in_box
    type: hard
    expected_satisfaction: 0.99
    probe: fmllm.fms.fm1_image.probes.positions_in_box

dependencies:
  - target_variable: atom_count
    relationship: derives
  - target_variable: positions_in_box
    relationship: requires
```

### FM2 (RDF -> energy)

```yaml
physics_constraints:
  - name: permutation_invariance     # automatic; probe verifies
  - name: extensive_scaling          # by output design (per-atom output)
  - name: non_negativity             # bounded by LJ energy floor

dependencies:
  - target_variable: atom_count
    relationship: scales_with
  - target_variable: temperature
    relationship: derives
```

### FM3 (trajectory -> Gamma KE moments)

```yaml
physics_constraints:
  - name: equipartition              # alpha * beta = mean KE
  - name: distribution_normalization # Gamma integrates to 1
  - name: distribution_non_negativity  # alpha, beta > 0

dependencies:
  - target_variable: temperature
    relationship: derives             # T = alpha * beta in 2D
```

## Writing a new probe

A probe is a Python module declaring a single function:

```python
def run_probe(*, model, items, device, config) -> ProbeResult:
    threshold = float(config.get("threshold", ...))
    n_samples = int(config.get("n_samples", ...))
    # 1. select n_samples items
    # 2. run the model and compute the constraint metric
    # 3. return ProbeResult(constraint_name=..., satisfaction_score=...,
    #                       num_test_cases=..., metric=...,
    #                       passes_threshold=score >= threshold,
    #                       threshold=threshold, details={...})
```

Three rules for new probes:

1. The function tolerates an empty `items` list and returns a
   `ProbeResult` with `num_test_cases = 0`. The runner falls back to
   a degraded score rather than crashing.
2. The probe never modifies the model. Always use `model.eval()` plus
   `with torch.no_grad():`.
3. The probe stores compact summary statistics (counts, means,
   thresholds) in `details`. Raw tensors do not belong in the YAML
   probe report.

## BridgedFMOutput, field by field

The structure-preserving bridge composes Layer 1 metadata, Layer 2
calibration, Layer 3 probe scores, plus the raw FM output into a
typed object the verifier and the LLM both consume:

```python
class BridgedFMOutput:
    prediction: Prediction          # quantity, value, units, uncertainty
    source: Source                  # fm_name, fm_version, in_distribution flag
    applicable_constraints: list[ApplicableConstraint]
    dependencies: list[BridgedDependency]
    timestamp: str
```

- `prediction.value` carries a per-FM typed payload defined in
  `fms/<fm>/bridge_schema.py` (`AtomSet`, `EnergyPerAtom`,
  `GammaKEDistribution`).
- `prediction.uncertainty` reads from the conformal calibrator.
- `source.in_distribution` reads from the conformal in-distribution
  flag for the input.
- `applicable_constraints` reads from the FM's probe report. Each
  entry carries `constraint_name`, `type`, `satisfied_in_training`,
  `satisfaction_score`.
- `dependencies` reads from `metadata.yaml.dependencies`, with the
  bridge populating `derived_value` per query (for instance, FM1's
  derived `atom_count` reads off the count head's argmax).
- `timestamp` records when the bridge produced the object.

JSON round-trip preserves every field. `tests/test_bridge_schema.py`
exercises this.

## How the verifier consumes bridged outputs

Phase 4 wires the verifier sources to the bridged schema. The
contract:

- `verifier/sources/rule_library.py` dispatches to per-constraint
  check functions registered against `applicable_constraints[i].constraint_name`.
- `verifier/sources/cross_fm.py` reads `dependencies` to discover
  shared variables across FMs and applies the calibrated tolerance
  matrix.
- `verifier/sources/conformal.py` reads `source.in_distribution` and
  the calibrated uncertainty to flag low-confidence predictions.
- `verifier/sources/literature.py` and `verifier/sources/simulator.py`
  stay external (literature database, MD rollouts) since they do not
  live with any single FM.
- The integrator accepts a runtime `sources_config` so individual
  sources can be activated or disabled per call. This supports the
  E4 verifier-ablation experiment.
