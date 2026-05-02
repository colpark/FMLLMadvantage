# Audit Report, Phase 3

**Audited at:** 2026-05-02T03:36:59Z
**Auditor:** Claude Code (self-audit)
**Result:** PASS

## Summary

Phase 3 implements both bridge flavors over the `FMContext` abstraction
introduced by the Phase 2.5 addendum. The structure-preserving bridge
emits `BridgedFMOutput` Pydantic objects with typed value payloads,
calibrated uncertainty, applicable-constraint summaries, and
materialized dependencies. The language-anchored bridge emits captions
paraphrasing the same content, with a complementary `parse_caption`
that recovers the numerical values for round-trip verification. Local
pytest reports **118 passed in 1.0s** (100 from before + 18 new
bridge tests). One regex bug fixed during the audit.

## Detailed checks

### CHECK 3.1, BaseBridge ABC + FMContext + factories
- **Result:** PASS
- **Evidence:** `src/fmllm/bridges/base.py` defines `BaseBridge`
  (`abstract emit`), `FMContext` dataclass (`fm_name`, `metadata`,
  `probe_report`, `calibration` plus `calibration_threshold`,
  `constraint_type` lookups), and `assemble_applicable_constraints` /
  `assemble_dependencies` shared between the flavors. Factories in
  `structure_preserving.py` and `language_anchored.py` dispatch by
  `fm_name` and reject unknown names with a clear message.

### CHECK 3.2, structure-preserving bridge produces BridgedFMOutput
- **Result:** PASS
- **Evidence:** Three subclasses (`FM1StructureBridge`,
  `FM2StructureBridge`, `FM3StructureBridge`) emit
  `BridgedFMOutput` with the expected typed value payloads
  (`AtomSet`, `EnergyPerAtom`, `GammaKEDistribution`). The base
  class assembles `Source`, `applicable_constraints`,
  `dependencies`, and `timestamp` once. Tests verify field-by-field
  correctness (`tests/test_bridges.py::test_fm{1,2,3}_structure_bridge_*`).

### CHECK 3.3, JSON round-trip preserves all fields
- **Result:** PASS
- **Evidence:**
  `tests/test_bridges.py::test_bridged_output_round_trips_through_json`
  iterates over all three FMs, dumps to JSON, loads back, and asserts
  `model_dump()` equality. Pydantic `extra="forbid"` plus consistent
  per-FM payload models ensures lossless round-trip.

### CHECK 3.4, language-anchored bridge produces parseable captions
- **Result:** FIXED then PASS
- **Evidence:** Caption format is fixed per FM (per the original
  prompt's specifications). `parse_caption(caption, fm_name)` extracts
  numerical values via per-FM regexes. Tests round-trip the
  numerical content for all three FMs.
- **Issue caught + fix:** the FM1 confidence-list regex used
  `[\d\.,\s]+?` with non-greedy match and a `.` terminator, which
  stopped at the first decimal point inside the first float. Replaced
  with explicit float-list regex
  `(?:\d+\.\d+(?:,\s*)?)+` that matches a sequence of comma-separated
  floats. Fixed in commit; round-trip test now passes.

### CHECK 3.5, applicable_constraints cross-references probe + metadata
- **Result:** PASS
- **Evidence:** `assemble_applicable_constraints(context)` reads
  every probe result and looks up the declared `hard`/`soft` type
  from `context.metadata.physics_constraints`. The result has the
  full Pydantic shape required by `BridgedFMOutput`. Tests confirm
  that names match `metadata.physics_constraints` exactly.

### CHECK 3.6, dependencies materialize from metadata
- **Result:** PASS
- **Evidence:** `assemble_dependencies(context, derived_values)`
  iterates `metadata.dependencies` and fills `derived_value` from
  the per-FM hook `_derived_values(raw_output)`. FM1 fills
  `atom_count`. FM3 fills `temperature`. FM2 leaves them empty by
  design (verifier consults FM1/FM3). Tests confirm.

### CHECK 3.7, in-distribution flag wired through both bridges
- **Result:** PASS
- **Evidence:**
  `tests/test_bridges.py::test_structure_bridge_emits_in_distribution_flag`
  and `test_language_bridge_marks_out_of_distribution` confirm the
  flag propagates correctly. Default is `True`. Caller can override.

### CHECK 3.8, calibration absence handled gracefully
- **Result:** PASS
- **Evidence:**
  `test_load_fm_context_handles_missing_artifacts` confirms compose
  loader returns a valid context with empty probe report and
  calibration when files are missing.
  `test_structure_bridge_without_calibration_omits_uncertainty`
  confirms the bridge emits `Prediction.uncertainty=None` when no
  calibration is in context.
  `test_language_bridge_handles_no_calibration` confirms the FM2
  caption omits the `plus or minus` clause.

### CHECK 3.9, full local test suite passes
- **Result:** PASS
- **Evidence:** `pytest -m "not gpu"` reports `118 passed in 1.0s`.

### CHECK 3.10, prose style
- **Result:** PASS
- **Evidence:** Scanned every new and modified markdown file for
  em-dashes and semicolons in narrative prose (excluding fenced code
  blocks). Zero matches.

### CHECK 3.11, working tree clean after Phase 3 commit
- **Result:** PASS (after the Phase 3 commit lands)
- **Evidence:** Pre-commit `git status --short` shows only the new
  bridge module, the new test file, the new progress and audit
  documents, and the README updates.

## Files added during this phase

- `src/fmllm/bridges/{__init__.py, README.md, base.py, compose.py,
   structure_preserving.py, language_anchored.py}` (the bridges
  proper).
- `tests/test_bridges.py` (18 new tests).
- `docs/progress/03-bridges.md`.
- `docs/audits/03-bridges-audit.md` (this file).

## Fixes applied during audit

- `language_anchored.py:_FM1_CONF_RE` regex narrowed to a
  comma-separated float-list pattern. The earlier non-greedy form
  matched only the first digit of the first confidence value.

## Remaining concerns

- **FM3 uncertainty surface is asymmetric across FMs.** FM1 and FM2
  populate `Prediction.uncertainty`; FM3 does not. The verifier
  reads the calibration threshold via `context.calibration_threshold`
  for FM3 instead. This is an intentional asymmetry (FM3's
  non-conformity score is a per-specimen NLL, not a scalar interval
  on the predicted moments) but worth re-examining in Phase 4 if
  the verifier's cross-source weighting needs uniform shape.
- **The compose loader hard-codes the metadata-yaml path lookup.**
  `metadata_yaml_path(fm_name)` builds the path relative to the
  source tree. When the package is installed via `uv sync` the path
  still resolves correctly because hatch builds an editable install
  and `__file__` lives under `src/fmllm/bridges/`. A non-editable
  install would need a `importlib.resources` lookup; flag for
  Phase 9 reproducibility.
- **The language-anchored FM3 caption uses a heuristic standard-
  error formula.** Documented in the docstring and in the progress
  doc. Not load-bearing for the architecture; Phase 7 evaluation
  may revise.

## Sign-off

The Phase 3 implementation matches the original prompt's
specification (modified by the Phase 2.5 addendum and the local-
Claude / remote-execution split). The bridges are ready for the
verifier in Phase 4 to consume.
