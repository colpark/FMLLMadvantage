# fmllm.bridges

Bridges that transport raw FM output into LLM-consumable artifacts.

Two flavors share one abstract base. Each consumes an `FMContext`
(metadata + probe report + calibration) loaded once per checkpoint
and wraps an unbounded number of forward-pass results.

## Files

- `base.py` - `BaseBridge` ABC, `FMContext` dataclass, helpers
  `assemble_applicable_constraints` and `assemble_dependencies`
  shared between the two flavors.
- `compose.py` - `load_fm_context(fm_name, checkpoint_dir)` finds the
  three artifacts (`metadata.yaml`, `probe_report.yaml`,
  `calibration.json`) and returns an `FMContext`. Falls back to empty
  defaults for missing probe report or calibration.
- `structure_preserving.py` - `StructurePreservingBridge` (abstract)
  plus `FM1StructureBridge`, `FM2StructureBridge`,
  `FM3StructureBridge`. Factory `make_structure_bridge(context)`
  picks the right subclass by `context.fm_name`. Output:
  `BridgedFMOutput` Pydantic object, JSON round-trip safe.
- `language_anchored.py` - `LanguageAnchoredBridge` (abstract) plus
  three FM-specific subclasses. Factory `make_language_bridge`. Output:
  natural-language caption string. `parse_caption(caption, fm_name)`
  recovers the numerical values for round-trip checks.

## Typical use

```python
from fmllm.bridges import load_fm_context, make_structure_bridge, make_language_bridge

ctx = load_fm_context(
    fm_name="fm1_image",
    checkpoint_dir="checkpoints/fm1_image/train_50k/<run_id>/",
)

struct_bridge = make_structure_bridge(ctx)
lang_bridge = make_language_bridge(ctx)

raw = model(image)        # dict of tensors
bridged = struct_bridge.emit(raw, input_provenance={"specimen_id": 42})
caption = lang_bridge.emit(raw, input_provenance={"specimen_id": 42})
```

`bridged` is a `BridgedFMOutput`; `caption` is a string. The two
carry the same content in different shapes.

## Per-FM value payloads

The structure-preserving bridge populates `Prediction.value` with
typed payloads defined in `fmllm/fms/<fm>/bridge_schema.py`:

- FM1: `AtomSet { n_atoms_pred, positions: [AtomPosition], raw_count_logits, raw_query_count }`.
- FM2: `EnergyPerAtom { value_lj }`.
- FM3: `GammaKEDistribution { alpha, beta, mean, variance, implied_temperature_lj }`.

## Calibrated uncertainty

| FM | Uncertainty shape | Source |
|----|-------------------|--------|
| FM1 | `(0, q_alpha)` per-atom radius | `calibration.json` thresholds keyed by alpha |
| FM2 | `(E - q_alpha, E + q_alpha)` symmetric band | same |
| FM3 | `None`; the verifier checks per-specimen NLL against the threshold instead | same |

When `calibration.json` is missing the bridge still emits a
prediction; the `Prediction.uncertainty` field stays `None`.

## Dependency materialization

The bridge materializes the dependency edges declared in
`metadata.yaml` with runtime-derived values where available. For
example, FM1 fills `derived_value` for `atom_count` from its count
head; FM2 leaves the `atom_count` and `temperature` derived values
empty so the verifier consults FM1 / FM3 for them.
