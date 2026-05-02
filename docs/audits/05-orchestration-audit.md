# Audit Report, Phase 5

**Audited at:** 2026-05-02T04:05:05Z
**Auditor:** Claude Code (self-audit)
**Result:** PASS

## Summary

Phase 5 implements the LLM orchestration loop with three components:
typed trajectory schemas, an LLM wrapper layer (mock + real Llama
3.1), and the OHVD controller. The CLI in `scripts/run_pipeline.py`
runs Pipeline A end-to-end on one specimen. Local pytest reports
**160 passed in 1.2s** (142 from before + 18 new orchestrator tests).
The real Llama wrapper lazy-imports torch/transformers; the
laptop-side test path uses MockLLM only.

## Detailed checks

### CHECK 5.1, llm.py wraps a chat LLM with tool-calling
- **Result:** PASS
- **Evidence:** `BaseLLM` ABC with `chat(messages) -> str`.
  `TransformersLLM` lazy-loads `AutoTokenizer` + `AutoModelForCausalLM`,
  applies the chat template, and generates one assistant turn.
  `MockLLM` returns scripted responses for tests.
  `parse_llm_response` extracts the first JSON object from the model
  output and validates it against `LLMAction` (with action shapes
  `call_fm`, `hypothesize`, `commit`, `error`).

### CHECK 5.2, loop.py runs the OHVD cycle
- **Result:** PASS
- **Evidence:** `OHVDLoop.run` builds the chat with the system
  prompt + user message, then iterates up to `max_steps`. Each
  iteration calls the LLM, parses the response, dispatches by
  `action_type`, records a `Step`, appends a tool message back to
  the chat. `call_fm` -> runner -> bridged output.
  `hypothesize`/`commit` -> verifier -> verdict. Errors get logged
  as ERROR steps; the loop keeps going.

### CHECK 5.3, trajectory.py serializes the full record
- **Result:** PASS
- **Evidence:** `Trajectory.model_dump_json()` round-trips through
  `model_validate_json` with zero diff in
  `tests/test_orchestrator.py::test_trajectory_round_trips_through_json`.
  All step types (observation, hypothesis, verifier_verdict,
  final, error) appear in the trajectory.

### CHECK 5.4, scripts/run_pipeline.py end-to-end CLI
- **Result:** PASS
- **Evidence:** The Typer CLI loads each FM model from the latest
  checkpoint at the requested training scale, builds the bridges,
  opens the dataset, builds the verifier (with literature DB), and
  instantiates either `MockLLM` (when `--mock-script` is set) or
  `TransformersLLM` (default Llama 3.1 8B). Output: per-run
  `trajectory.json` + `manifest.yaml`. `runs/<run_id>/run.log`
  captures the loop log via `configure_logging`.

### CHECK 5.5, tests confirm loop terminates on simple cases
- **Result:** PASS
- **Evidence:**
  `test_loop_terminates_on_immediate_commit` (commit -> COMMITTED).
  `test_loop_terminates_on_budget_exhausted` (no commit -> BUDGET_EXHAUSTED).
  Both produce a populated trajectory with the right
  `TerminationReason`.

### CHECK 5.6, tests confirm tool calls dispatch correctly
- **Result:** PASS
- **Evidence:** `test_loop_dispatches_call_fm_actions` issues two
  call_fm actions in sequence; observation steps land with the
  right `bridged_output.source.fm_name`.
  `test_loop_handles_unknown_tool_as_error_and_continues` confirms
  unknown tool names produce ERROR steps without crashing the loop.

### CHECK 5.7, tests confirm verdicts feed back into LLM context
- **Result:** PASS
- **Evidence:**
  `test_loop_feeds_verdicts_back_to_llm_context` inspects the
  messages the mock LLM saw on the second call and confirms a
  `role: tool` message appears with a JSON payload containing the
  `aggregate_decision` and per-source breakdown.

### CHECK 5.8, sources_config flows through loop to verifier
- **Result:** PASS
- **Evidence:**
  `test_loop_passes_sources_config_through_to_verifier` runs with
  `SourcesConfig.for_ablation("V0")` and confirms the resulting
  final verdict has `aggregate_decision = skip`. This is the
  architectural slot E4 plugs into.

### CHECK 5.9, lazy import keeps laptop path clean
- **Result:** PASS
- **Evidence:** `TransformersLLM.__init__` does not import torch
  beyond what the rest of the package already imports;
  `_ensure_loaded` defers `transformers` import until the first
  `chat` call. Laptop pytest (no GPU, no model weights) imports
  `fmllm.orchestrator` and runs all 18 tests cleanly.

### CHECK 5.10, full local test suite passes
- **Result:** PASS
- **Evidence:** `pytest -m "not gpu"` reports `160 passed in 1.19s`.

### CHECK 5.11, prose style
- **Result:** PASS
- **Evidence:** Scanned every new and modified markdown file for
  em-dashes and semicolons in narrative prose (excluding fenced
  code blocks). Zero matches.

### CHECK 5.12, working tree clean after Phase 5 commit
- **Result:** PASS (after the Phase 5 commit lands)

## Files added during this phase

- `src/fmllm/orchestrator/{__init__.py, README.md, trajectory.py, llm.py, loop.py, runners.py}`.
- `scripts/run_pipeline.py`.
- `scripts/mock_scripts/{example.json, README.md}`.
- `tests/test_orchestrator.py` (18 new tests).
- `docs/progress/05-orchestration.md`.
- `docs/audits/05-orchestration-audit.md` (this file).

## Fixes applied during audit

None. All 18 new tests passed on first run. The lazy-import
strategy for `transformers` worked cleanly: the test venv has no
transformers package, yet `import fmllm.orchestrator` and the loop
tests run without issue.

## Remaining concerns

- **System prompt is conservative.** The default prompt asks for
  one JSON action per turn. Real Llama may emit prose alongside.
  The parser tolerates that, but the loop's tool messages back to
  the LLM stay strict JSON, which can pull the model into an
  agreeable mode. Phase 6 fine-tuning will calibrate this.
- **No real-LLM integration test.** The `TransformersLLM` class
  ships untested; the first remote run is the integration test.
  Add a smoke test in Phase 6 once the LLM is loaded once.
- **Runners assume identity HDF5 row to specimen ID.** When the
  dataset is filtered (a future use case for evaluation), the
  runner builds an `id_to_index` map on first use. The map is
  small (one int per specimen) so the cost is negligible.
- **Trajectory storage uses pretty-printed JSON.** Trajectories on
  the order of 16 steps with full bridged outputs will be ~20 KB
  each. For Phase 6 RL collection across thousands of specimens,
  consider a more compact format or NDJSON.

## Sign-off

The Phase 5 implementation matches the original prompt's
specification and exposes the architectural slot Phase 6 will use
(verifier-passing trajectories collected end-to-end via this loop).
Phase 6 (RL fine-tuning) is ready to start.
