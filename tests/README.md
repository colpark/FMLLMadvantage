# tests/

Pytest suite for the project. The default invocation runs every test
that does not require a GPU.

## Layout

- `conftest.py` - registers the `gpu` marker and skips GPU tests when
  CUDA is unavailable.
- `test_utils.py` - exercises the helpers under `src/fmllm/utils/`.
- `test_physics.py` - LJ potential, MD energy conservation, RDF /
  pair-histogram normalization, RDF permutation invariance,
  rasterizer accuracy, structure generators, MB velocities.
- `test_data.py` - splits assignment, splits YAML round trip, the
  HDF5-backed `LJSpecimenDataset` reader.
- `test_fms.py` - forward-pass shapes and physics-constraint losses for
  FM1 / FM2 / FM3, plus conformal-quantile and calibration-file tests.
- `test_fm_metadata.py` - per-FM `metadata.yaml` validates against the
  schema and every declared probe path imports.
- `test_bridge_schema.py` - `BridgedFMOutput` JSON round-trip plus the
  per-FM typed value payloads.
- `test_probes.py` - every probe runs end-to-end on a tiny model and
  produces a valid `ProbeResult`. The probe runner produces a
  consistent `ProbeReport` for FM1.
- `test_cross_fm_tolerance.py` - pairwise tolerance computation on
  synthetic records and YAML round-trip.
- `test_bridges.py` - structure-preserving and language-anchored
  bridges. Verifies BridgedFMOutput JSON round-trip, per-FM payload
  correctness, calibration fall-back behavior, factory dispatch by
  FM name, and caption parser round-trip.
- `test_verifier.py` - five verifier sources (rule library,
  literature, cross-FM, simulator, conformal) on hand-crafted
  bridged objects, integrator aggregation rules, E4 ablation
  presets V0/V1/V4, and VerifierVerdict JSON round-trip.

## Running

Local (no GPU):

```
uv run pytest -m "not gpu" -v
```

Remote (with all four H100s visible):

```
uv run pytest -v
```

## Markers

- `gpu` - the test requires a CUDA device. Local runs skip these
  automatically with a pointer to `docs/remote-setup.md`.
