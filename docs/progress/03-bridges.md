# Phase 3: Bridges

## What I built

The bridges transport raw FM output into LLM- and verifier-consumable
artifacts. Two flavors share one abstract base. Each consumes an
`FMContext` (metadata + probe report + calibration) loaded once per
checkpoint and wraps an unbounded number of forward-pass results.

### `fmllm.bridges`

- `base.py` - `BaseBridge` ABC, `FMContext` dataclass, helpers
  `assemble_applicable_constraints` and `assemble_dependencies`
  shared by both flavors. The context exposes a
  `calibration_threshold(alpha)` lookup so per-FM bridges read the
  conformal thresholds without coupling to the calibration JSON
  schema directly.
- `compose.py` - `load_fm_context(fm_name, checkpoint_dir)` finds
  `metadata.yaml` (in source) plus `probe_report.yaml` and
  `calibration.json` (in checkpoint dir) and returns a populated
  `FMContext`. Falls back gracefully when probe report or
  calibration is missing.
- `structure_preserving.py` - `StructurePreservingBridge` (abstract)
  plus `FM1StructureBridge`, `FM2StructureBridge`,
  `FM3StructureBridge`. Factory `make_structure_bridge(context)`
  dispatches by `context.fm_name`. Output is a `BridgedFMOutput`
  Pydantic object; JSON round-trip preserves every field.
- `language_anchored.py` - `LanguageAnchoredBridge` (abstract) plus
  three FM-specific subclasses, `make_language_bridge` factory, and
  `parse_caption(caption, fm_name)` reverse parser used by tests
  and downstream consumers. Output is a string.

### Per-FM specifics

| FM | Structure bridge value | Uncertainty | Derived dependencies |
|---|---|---|---|
| FM1 | `AtomSet { n_atoms_pred, positions: [AtomPosition], raw_count_logits, raw_query_count }` | `(0, q_alpha)` per-atom radius | `atom_count` from count head |
| FM2 | `EnergyPerAtom { value_lj }` | `(E - q_alpha, E + q_alpha)` symmetric band | (none derived; verifier consults FM1/FM3) |
| FM3 | `GammaKEDistribution { alpha, beta, mean, variance, implied_temperature_lj }` | `None`; verifier checks per-specimen NLL against `q_alpha` instead | `temperature` from `alpha * beta` |

The structure-preserving bridge also populates:

- `Source` with `fm_name`, `fm_version`, `in_distribution` flag, and
  caller-supplied `raw_input_provenance` (for example
  `{"specimen_id": 42}`).
- `applicable_constraints` cross-referenced from the probe report
  (satisfaction score, passes-threshold flag) and the metadata
  (`hard`/`soft` type).
- `dependencies` materialized from `metadata.dependencies` with
  runtime-derived values where available.
- `timestamp` in ISO 8601 UTC.

The language-anchored bridge adds a constraint summary tail and an
optional out-of-distribution flag to the body of every caption, so
the LLM sees the same content as the structure-preserving bridge in
prose form.

### Tests (`tests/test_bridges.py`, 18 tests)

- `FMContext` loads from disk with full artifacts and falls back
  gracefully when missing.
- Three `make_structure_bridge` smoke tests confirm the three
  per-FM payloads (`AtomSet`, `EnergyPerAtom`, `GammaKEDistribution`)
  populate correctly with the right uncertainty shape.
- `BridgedFMOutput` JSON round-trip across all three FMs.
- `make_structure_bridge` and `make_language_bridge` reject unknown
  FM names; `parse_caption` rejects unknown FM names.
- Three language-bridge round-trip tests: parse extracts the same
  numerical content the bridge wrote.
- The language bridge embeds the constraint summary and the OOD flag
  in the caption.
- Bridge omits uncertainty cleanly when calibration is absent.

All 118 local tests pass in 1.0 second.

## What the user runs to verify Phase 3

### Local laptop (no GPU)

```
git pull
uv sync --extra dev
uv run pytest -m "not gpu" -v
```

Expect 118 passing tests (100 from before plus 18 new bridge tests).

### Remote 4xH100 host

The bridges are CPU-only Python. They have no remote-only
verification step beyond running the same pytest invocation against
the live environment and confirming the new tests pass.

A practical end-to-end smoke test requires loading a trained FM and
its calibration from disk, running a forward pass, and feeding the
raw output through the bridge. We will add that smoke test in
Phase 4 alongside the verifier integration.

## Known caveats

- **FM2 emits no derived dependencies.** The metadata declares
  `atom_count` and `temperature` as scaled-with / derived, but FM2's
  scalar energy doesn't directly recover either. The bridge leaves
  `derived_value` as `None`. The verifier's cross-FM source
  consults FM1's `atom_count` and FM3's `temperature` to fill the
  cross-FM agreement check.
- **FM3 uncertainty is not a numerical interval.** The conformal
  threshold for FM3 is the per-specimen mean Gamma NLL. The bridge
  surfaces it via `context.calibration_threshold` rather than
  populating `Prediction.uncertainty` directly. The verifier reads
  the threshold and computes the actual band check on demand.
- **The language-anchored FM3 caption uses a heuristic temperature
  band.** It scales `beta * sqrt(q_alpha / alpha)` as a rough
  asymptotic standard error of the Gamma mean. This is a caption-
  level approximation; the structure-preserving bridge does not
  expose any temperature band.

## What remains for Phase 4

- Implement the multi-source verifier under `src/fmllm/verifier/`.
- The integrator must accept a runtime `sources_config` for E4
  ablation (the architectural enabler from the addendum).
- Wire each source to read from `BridgedFMOutput` consistently:
  - `rule_library` dispatches per-constraint check functions on
    `applicable_constraints`.
  - `cross_fm` reads `dependencies` and consults the calibrated
    tolerance matrix.
  - `conformal` reads `source.in_distribution` and `Prediction.uncertainty`.
  - `simulator` runs MD rollouts from the LLM's proposed state.
  - `literature` does cluster-database lookup (external).
- Tests cover hand-crafted bridged objects under varying
  source-ablation configurations.
