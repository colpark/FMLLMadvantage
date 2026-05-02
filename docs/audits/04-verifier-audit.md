# Audit Report, Phase 4

**Audited at:** 2026-05-02T03:52:38Z
**Auditor:** Claude Code (self-audit)
**Result:** PASS

## Summary

Phase 4 implements the multi-source verifier as specified by the
original prompt and extended by the Phase 2.5 addendum. All five
sources read `BridgedFMOutput` objects; the integrator accepts a
runtime `sources_config` that disables individual sources for the E4
ablation; the literature source consults a curated `clusters.json`
generated from canonical structures via the project's LJ Hamiltonian.
Local pytest reports **142 passed in 1.1s** (118 from before + 24
new verifier tests).

## Detailed checks (extended Phase 4 checklist)

### CHECK 4.1, verifier sources read BridgedFMOutput
- **Result:** PASS
- **Evidence:** Each source's `check(bridged_outputs, claim)`
  signature accepts the typed bridge output. Tests construct
  `BridgedFMOutput` via the structure-preserving bridge and pass
  them in. No source reaches into raw FM model outputs.

### CHECK 4.2, rule_library dispatches per-constraint check functions
- **Result:** PASS
- **Evidence:** `RuleLibrarySource.check` iterates each
  `applicable_constraint` in every bridged output and dispatches via
  the `_REGISTRY` mapping populated by `@register_check` decorators.
  Default checks cover the seven constraint names declared across
  FM1/FM2/FM3 metadata. Aggregates hard fails to FAIL, soft fails
  to CAVEAT.

### CHECK 4.3, cross_fm reads dependency edges and tolerances
- **Result:** PASS
- **Evidence:** `CrossFMSource._collect_per_variable` reads
  `bridged_output.dependencies` and `claim.{n_atoms, temperature,
  per_atom_potential_energy}` into a per-variable dict. The
  pairwise comparison reads from `tolerance_matrix.pairwise` when a
  matrix is supplied; falls back to hard-coded defaults otherwise.
  Test `test_cross_fm_with_calibrated_tolerance` confirms the
  matrix path works end-to-end.

### CHECK 4.4, conformal source reads in_distribution flags
- **Result:** PASS
- **Evidence:** `ConformalSource.check` iterates
  `bridged_output.source.in_distribution` and `Prediction.uncertainty`.
  Returns CAVEAT when an FM is OOD or when the claim's value sits
  outside the calibrated band. Never escalates to FAIL by design.

### CHECK 4.5, integrator accepts a sources-config field at runtime
- **Result:** PASS
- **Evidence:** `Verifier.verify(bridged, claim, sources_config=...)`
  threads the per-call config through. Disabled sources contribute
  a SKIP verdict. `SourcesConfig.for_ablation("V0".."V4")` provides
  the addendum's E4 presets. Tests
  `test_integrator_disabled_sources_skip_under_v0`,
  `test_integrator_v1_only_runs_rule_library`, and
  `test_integrator_runs_all_sources_under_v4` cover ablation.

### CHECK 4.6, integration tests cover ablation cases
- **Result:** PASS
- **Evidence:** `tests/test_verifier.py` exercises hand-crafted
  bridged objects under V0, V1, and V4 plus aggregate-decision
  rules. The aggregate ordering FAIL > CAVEAT > PASS > SKIP is
  verified by `test_integrator_aggregates_fail_over_caveat`.

### CHECK 4.7, literature database exists and validates
- **Result:** PASS
- **Evidence:** `data/literature/clusters.json` shipped with 12
  entries from `scripts/build_literature_db.py`. Schema documented
  in `data/literature/README.md`. `test_literature_passes_for_canonical_cluster`
  confirms a well-formed FM2 prediction matches the reference.

### CHECK 4.8, simulator source runs MD from a claim
- **Result:** PASS
- **Evidence:** `SimulatorSource.check` initializes
  Maxwell-Boltzmann velocities at the claim's temperature, runs
  `run_md` for `n_steps`, computes per-atom mean KE, and compares
  to the claim. Tests confirm the simulator passes on a clean HCP-7
  cluster and skips when positions or temperature are missing.

### CHECK 4.9, prose style
- **Result:** PASS
- **Evidence:** Scanned every new and modified markdown file for
  em-dashes and semicolons in narrative prose (excluding fenced
  code blocks). Zero matches.

### CHECK 4.10, full test suite passes
- **Result:** PASS
- **Evidence:** `pytest -m "not gpu"` reports `142 passed in 1.11s`.

### CHECK 4.11, verdict JSON round-trip
- **Result:** PASS
- **Evidence:** `test_verifier_verdict_json_round_trip` confirms a
  full `VerifierVerdict` (including nested `SourceVerdict` list,
  `Hint`, and `SourcesConfig`) survives JSON dump and load with
  zero diff. Phase 5's trajectory storage relies on this.

### CHECK 4.12, working tree clean after commit
- **Result:** PASS (after the Phase 4 commit lands)

## Files added during this phase

- `src/fmllm/verifier/{schema.py, integrator.py}`.
- `src/fmllm/verifier/sources/{__init__.py, rule_library.py,
   literature.py, cross_fm.py, simulator.py, conformal.py}`.
- `src/fmllm/verifier/README.md` updated.
- `data/literature/{clusters.json, README.md}`.
- `scripts/build_literature_db.py`.
- `tests/test_verifier.py` (24 new tests).
- `docs/progress/04-verifier.md`.
- `docs/audits/04-verifier-audit.md` (this file).

## Fixes applied during audit

None. All checks passed on first inspection. The literature DB
generation ran cleanly and produced 12 entries on first invocation.

## Remaining concerns

- **Cross-FM tolerance matrix is not yet generated.** The verifier
  works without one (falls back to defaults), but Phase 8's E4
  should run `fmllm.fms._calibration.cross_fm_tolerance` over the
  calibration set first to ship a calibrated matrix. The
  architectural slot is ready.
- **Hint suggestions are templated, not generative.** Each flagged
  source maps to a fixed string. Phase 6's RL fine-tuning may
  generate richer hints, but the current strings are clear enough
  for the LLM to act on.
- **The simulator source is single-specimen NVE.** It does not
  thermostat or run a control comparison; the temperature
  comparison treats the claim's T as ground truth and the MD's
  per-atom mean KE as the response. For Phase 8 we may want a
  control rollout from a known reference to compare deltas.
- **Position-set agreement (LLM claim vs FM1) is not yet wired.**
  The cross-FM source does not compare claim.positions to FM1.
  This is a Phase 5 follow-up once the orchestrator finalizes the
  claim's position field shape.

## Sign-off

The Phase 4 implementation matches the original prompt's
specification and the addendum's extensions. Phase 5 (orchestrator)
is ready to start.
