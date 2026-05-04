# Phase 8a: Baseline Comparison

## What I built

Phase 8a turns the Phase 7 measurement apparatus into a comparative
study. Three baselines run on the same 200 specimens:

* **B0 — naked**: one LLM call, no FM tools, no verifier. Isolates
  whether grounding in FMs matters at all.
* **B2 — no_verifier**: standard OHVD loop with `NoOpVerifier`
  always returning PASS. The LLM still calls FMs and sees bridged
  outputs but loses every verifier signal. Isolates whether the
  multi-source verifier loop teaches the LLM anything.
* **B3 — full**: canonical Pipeline A (existing).

Each baseline produces trajectories with the same `Trajectory`
schema, so the eight world-model tests and the new ground-truth
accuracy metric score them uniformly.

### `fmllm.baselines`

- `noop_verifier.py` — `NoOpVerifier` is a structural
  `Verifier`-compatible class. `verify(...)` returns aggregate PASS
  with one stub `noop` source verdict. Drops in via duck typing
  because `OHVDLoop` only calls the `verify(...)` method.
- `naked.py` — `run_naked_baseline(llm, query, specimen_id)` makes
  one LLM call with `NAKED_SYSTEM_PROMPT` (which describes the
  testbed distribution but provides no observations). Builds a
  one-step `Trajectory` with `final_verdict=None` on a parseable
  commit, or `termination=PARSE_FAILURE` otherwise. Anything other
  than a `commit` action (`hypothesize`, `call_fm`, error) is
  treated as a parse failure.

### `fmllm.evaluation.accuracy`

New ground-truth accuracy metric, separate from the eight world-
model tests:

- **N-accuracy**: `|claimed - true| <= 2`.
- **T-accuracy**: `|claimed - true| / true <= 25%` rel.
- **Motif-accuracy**: exact string match.
- **Compound accuracy**: all three above.
- **Commit rate**: committed / total trajectories.
- **Hallucination rate**: PASS-and-wrong / PASS-committed.
- **Calibrated abstention rate**: CAVEAT-on-wrong / total-wrong
  (higher is better — the verifier flagged the wrong commits).

`EvaluationReport` gains an `accuracy_results: list[TestResult]`
bucket, populated by `scripts/run_evaluation.py`.

### Tests (`tests/test_baselines.py`)

Eleven new tests:
- `NoOpVerifier` returns aggregate PASS with one stub source.
- `NoOpVerifier` ignores `sources_config`.
- `run_naked_baseline` commit path: builds one-step trajectory,
  no verdict, baseline metadata set.
- Naked baseline parse failure on non-JSON.
- Naked baseline rejects `hypothesize` (one-shot only).
- Accuracy: perfect compound on perfect claims.
- Accuracy: per-field math when one claim has wrong motif and
  another has T off by 50%.
- Accuracy: hallucination rate = 0.5 with one PASS-correct and one
  PASS-wrong commit.
- Accuracy: calibrated abstention rate = 0.5 with one CAVEAT-wrong
  and one PASS-wrong commit.
- Accuracy: skipped when no trajectory committed.
- Accuracy: `n_total` correctly counts unparseable trajectories.

### CLIs

- `scripts/run_baseline.py` — single typer CLI with
  `--baseline {naked, no_verifier, full}`. Naked skips FM loading
  and the dataset entirely (one LLM call per specimen). The other
  two share `collect_trajectories(...)` from Phase 6.
- `scripts/run_baseline.sh` — bash wrapper. Honors `START`,
  `COUNT`, `H5_PATH`, `LLM_MODEL`, `LLM_TEMP`, `GPU`,
  `MOCK_SCRIPT`, `ADAPTER_PATH` environment overrides.
- `scripts/compare_baselines.py` — loads multiple
  `runs/eval/*/report.yaml`, prints a side-by-side metric table
  per test, and writes `runs/comparisons/<run_id>/comparison.yaml`.
- `scripts/compare_baselines.sh` — wrapper.

## What the user runs to verify Phase 8a

### Local laptop (no GPU)

```
git pull
uv sync --extra dev
uv run pytest -m "not gpu" -v
```

Phase 8a contributes 11 new tests under `tests/test_baselines.py`.

### Remote 4xH100 host

```
ssh remote
cd ~/FMLLMadvantage
git pull && uv sync --extra dev

# 1. Run all three baselines on the same 200 specimens (start=0, count=200)
START=0 COUNT=200 bash scripts/run_baseline.sh naked
START=0 COUNT=200 bash scripts/run_baseline.sh no_verifier
START=0 COUNT=200 bash scripts/run_baseline.sh full

# 2. Evaluate each (record the run ids)
bash scripts/run_evaluation.sh runs/baselines/naked/<run-id>/trajectories.jsonl
bash scripts/run_evaluation.sh runs/baselines/no_verifier/<run-id>/trajectories.jsonl
bash scripts/run_evaluation.sh runs/baselines/full/<run-id>/trajectories.jsonl

# 3. Compare side-by-side
bash scripts/compare_baselines.sh \
    naked=runs/eval/<id-naked>/report.yaml \
    no_verifier=runs/eval/<id-nv>/report.yaml \
    full=runs/eval/<id-full>/report.yaml
```

Expected table shape:

```
test                          naked         no_verifier   full
trajectory_compression        ?  PASS/FAIL  ?  PASS/FAIL  ?  PASS/FAIL
trajectory_distinction        ?             ?             ?
step_recoverability           SKIP          ?             ?
prediction_compression        ?             ?             ?
prediction_distinction        ?             ?             ?
goal_competence               ?             ?             ?
federated_factorability       SKIP          SKIP          SKIP
calibrated_uncertainty        SKIP          ?             ?
goal_accuracy                 ?             ?             ?
HEADLINE: naked=? | no_verifier=? | full=?
```

`step_recoverability` skips on naked because there are no FM
observations to recover from. `calibrated_uncertainty` skips on
naked because there are no conformal verdicts.
`federated_factorability` always skips at this stage (needs the
ablation lattice from Phase 7's E4 experiment).

## What this phase resolves

The Phase 7 audit closed with an open question: "where's the
baseline?". Phase 8a answers it. Pipeline A is now measured against
two natural strawmen on the same dataset, with the same metrics,
under the same harness. The expected differentiation is in
**calibrated abstention** and **hallucination rate** — naked LLMs
have no way to flag their own wrong answers.

## What's left for Phase 8b (optional)

* B1 — FMs without bridges (raw tensor outputs).
* B4 — Pipeline B (already trained; just needs `--adapter-path` on
  the existing full baseline).
* B5 — external API LLM (GPT-4 / Claude). Needs an API client
  module. Skip if no keys.
* The E4 ablation lattice (V0..V4) so `federated_factorability`
  produces a non-skip result.
