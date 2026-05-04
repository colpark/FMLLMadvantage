# Audit Report, Phase 8a

**Audited at:** 2026-05-03T00:00:00Z
**Auditor:** Claude Code (self-audit)
**Result:** PASS

## Summary

Phase 8a adds three pieces of comparative evaluation infrastructure:
the `NoOpVerifier` for the B2 baseline, the `run_naked_baseline`
function for the B0 baseline, and a ground-truth accuracy metric in
`fmllm.evaluation.accuracy`. A unified CLI
(`scripts/run_baseline.py`) dispatches the three modes; a comparison
CLI (`scripts/compare_baselines.py`) prints a side-by-side metric
table across baselines and writes a YAML comparison file.

The schema-extension is minimal and additive: `EvaluationReport`
gains an optional `accuracy_results` bucket. Existing reports
without the field still parse because the field has a default
factory.

## Detailed checks

### CHECK 8.1, NoOpVerifier matches Verifier interface
- **Result:** PASS
- **Evidence:** `NoOpVerifier.verify(bridged_outputs, claim,
  *, sources_config=None)` matches `Verifier.verify` exactly. The
  loop only calls `.verify(...)` on its `verifier` attribute
  (`fmllm/orchestrator/loop.py:238`), so duck typing is sufficient.
  `available_sources()` returns `["noop"]` for diagnostics. The
  default `SourcesConfig` has every source disabled to make the
  behavior explicit.

### CHECK 8.2, Naked baseline produces compatible Trajectory
- **Result:** PASS
- **Evidence:** `run_naked_baseline(...)` builds a `Trajectory`
  with `metadata={"baseline": "naked"}`, one `Step` of type
  `FINAL` on a parseable commit, `final_verdict=None` (no verifier
  in B0), and `termination=COMMITTED`. Non-commit responses
  produce `termination=PARSE_FAILURE` with one `ERROR` step.
  Three tests cover commit, parse-failure, and "hypothesize is not
  a commit" paths.

### CHECK 8.3, Goal-accuracy metric math
- **Result:** PASS
- **Evidence:** Six tests verify per-field accuracy, compound
  accuracy, hallucination rate, calibrated abstention rate, the
  no-commit skip path, and `n_total` accounting for unparseable
  trajectories. The hallucination test uses two PASS verdicts (one
  correct, one wrong) and asserts `0.5`; the abstention test uses
  two wrong commits (one CAVEAT, one PASS) and asserts `0.5`.

### CHECK 8.4, EvaluationReport extension is backward compatible
- **Result:** PASS
- **Evidence:** `EvaluationReport.accuracy_results` defaults to
  `Field(default_factory=list)`. Reports written before this phase
  will load with an empty `accuracy_results` list. The runner
  (`scripts/run_evaluation.py::_run_all_tests`) returns a 4-tuple
  now, and the `aggregate_pass` flag includes accuracy results in
  the all-pass calculation.

### CHECK 8.5, run_baseline.py wires three modes correctly
- **Result:** PASS
- **Evidence:**
  - `naked` mode: skips FM loading and dataset reads entirely.
    Loops over specimen IDs and calls `run_naked_baseline` per
    specimen. Writes `trajectories.jsonl`, `summary.yaml`, and
    `manifest.yaml`.
  - `no_verifier` mode: loads FMs and dataset, instantiates
    `NoOpVerifier()`, calls `collect_trajectories(...)` with the
    same machinery as full. The literature DB is not loaded.
  - `full` mode: loads FMs, dataset, and the real verifier with
    literature DB; identical to the existing
    `scripts/collect_trajectories.py`.

### CHECK 8.6, run_baseline.sh wrapper handles the three modes
- **Result:** PASS
- **Evidence:** First positional argument is the baseline name
  (validated against `naked|no_verifier|full`). Subsequent args
  are forwarded to the python CLI. Honors `START`, `COUNT`,
  `H5_PATH`, `OUT_ROOT`, `LLM_MODEL`, `LLM_TEMP`, `GPU`,
  `MOCK_SCRIPT`, `ADAPTER_PATH` env vars.

### CHECK 8.7, compare_baselines.py reads per-baseline reports
- **Result:** PASS
- **Evidence:** `_parse_report_arg(...)` accepts `KEY=PATH` or bare
  `PATH` (default key = parent directory name). Disambiguates
  duplicate keys with a `_2` suffix. `_flatten_results(...)` walks
  the four buckets and indexes each `TestResult` by `test_name`.
  The output table prints metric and status side-by-side per
  baseline; the YAML keeps full details for follow-up analysis.

### CHECK 8.8, headline accuracy line in comparison output
- **Result:** PASS
- **Evidence:** The compare CLI prints a final
  `HEADLINE: naked=X | no_verifier=Y | full=Z` line keyed on
  `goal_accuracy.metric_value`. This is the one-number-per-baseline
  summary the user requested in the Phase 8a discussion.

### CHECK 8.9, no test imports torch directly
- **Result:** PASS (consistent with prior phases)
- **Evidence:** `tests/test_baselines.py` builds trajectories from
  Pydantic models and uses `MockLLM` directly. Importing
  `fmllm.baselines` does pull `fmllm.orchestrator` indirectly
  (which lazy-resolves torch at runner instantiation time), but no
  test actually constructs FM runners. Local pytest without torch
  fails at the import barrier in the same way it does for
  `tests/test_evaluation.py`; remote pytest passes.

### CHECK 8.10, syntax compiles for every changed file
- **Result:** PASS
- **Evidence:** `python -c "import ast; ast.parse(open(...).read())"`
  on all 10 changed files returns OK.

## Scope boundary

Phase 8a covers B0 + B2 + B3 only. Three things are out of scope and
land in 8b if pursued:

1. **B1 — FMs without bridges.** Would need a parallel runner that
   exposes raw FM outputs to the LLM as JSON tensors. Architecture
   intrusion.
2. **B4 — Pipeline B inference.** Already supported via
   `--adapter-path` on the existing full baseline. No new code.
3. **B5 — external LLM (GPT-4 / Claude).** Needs an API client.
   Skip if no keys.

The factorability ablation (V0..V4) is unblocked by Phase 8a's
`run_baseline.py --baseline full --ablation V<i>` invocations and
the existing `compare_baselines.py` already reads ablation reports.

## Issues caught during the audit

None. The schema extension was reviewed for backward compatibility,
the `NoOpVerifier`'s duck typing was verified against the loop's
actual usage, and the accuracy-metric math was checked by
hand-computing the four cases in the test file.
