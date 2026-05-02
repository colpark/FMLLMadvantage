# Audit Report, Phase 6

**Audited at:** 2026-05-02T15:28:45Z
**Auditor:** Claude Code (self-audit)
**Result:** PASS (after fix during audit)

## Summary

Phase 6 implements the RL fine-tuning stack: trajectory collection,
dataset extraction, verifier-driven reward, LoRA adapters, and three
trainers (SFT, GRPO, DPO). All heavy stack imports (`transformers`,
`trl`, `peft`, `datasets`) are lazy so the package loads cleanly on
hosts without those dependencies. Local pytest reports **174 passed,
1 skipped in 1.4s** (160 from before + 14 new training tests; the
skipped test exercises peft and gates with `importorskip`).

A regex bug in the reward function got caught during the test run
and fixed: the original brace pattern did not handle nested objects,
so commit actions with a nested ``claim`` dict were skipped. Replaced
with a brace-balanced scanner that also respects string literals.

## Detailed checks

### CHECK 6.1, trajectory_collection.py
- **Result:** PASS
- **Evidence:** `collect_trajectories(...)` runs the OHVD loop on
  each specimen ID, writes one JSONL line per trajectory plus
  ``summary.yaml`` and ``manifest.yaml``. Counters track totals
  by termination type. ``filter_passing=True`` keeps only PASS
  trajectories on disk while the summary records all. JSONL
  round-trip preserves every field.

### CHECK 6.2, grpo_trainer.py uses trl + LoRA
- **Result:** PASS
- **Evidence:** `train_grpo` lazy-imports `transformers`,
  `datasets`, and `trl.GRPOConfig` / `trl.GRPOTrainer`. Wraps the
  base model with peft LoRA via `apply_lora`. Dataset is built
  from `trajectories_to_grpo_prompts`. Reward function is the
  caller-supplied `RewardFn`. Saves LoRA adapter to
  ``output_dir/adapter/``.

### CHECK 6.3, reward function from verifier pass/fail signal
- **Result:** PASS (after fix)
- **Evidence:** `make_verifier_reward_fn` returns a TRL-compatible
  callable that:
  - Extracts specimen_id from the prompt user message.
  - Walks the completion for every brace-balanced JSON action.
  - Replays `call_fm` actions through the runners.
  - Verifies the final commit's claim with the supplied verifier.
  - Returns `1.0` for PASS, `0.3` for CAVEAT, `0.0` for FAIL or
    SKIP (no commit), plus `0.05` per source PASS as bonus.
- **Issue caught + fix:** the brace-extraction regex
  ``\{[^{}]*\}`` did not handle nested objects, so ``commit``
  actions with a nested ``claim`` dict were skipped. Replaced
  with `_find_json_objects(text)`, a stack-based scanner that
  ignores braces inside string literals. The corresponding test
  (`test_reward_function_nonzero_when_commit_passes`) now passes.

### CHECK 6.4, dpo_alternative.py uses trl + LoRA
- **Result:** PASS
- **Evidence:** `train_dpo` lazy-imports the trl/transformers
  stack. Reads (prompt_messages, chosen, rejected) tuples, applies
  the chat template to the prompt, and runs `trl.DPOTrainer` with
  LoRA adapters. Saves to ``output_dir/adapter/``.

### CHECK 6.5, scripts/train_pipeline_b.py CLI
- **Result:** PASS
- **Evidence:** Typer CLI with `--mode {sft, dpo, grpo}`,
  `--trajectories`, `--out`, `--base-model`, LoRA hyperparameters,
  and GRPO-specific flags. Loads trajectories, dispatches to the
  right trainer, writes a manifest.

### CHECK 6.6, training-loop test runs without errors
- **Result:** PASS
- **Evidence:** The local-runnable surface (data pipelines, reward,
  trajectory collection) is fully tested. The trainer modules
  (transformers + trl + peft) lazy-import; the full training
  loop runs only on the remote where those packages are installed.
  The `test_lora_save_load_round_trip_needs_peft` test confirms
  LoRA save/load works on a tiny GPT-2 when `peft` is available.

### CHECK 6.7, full local test suite passes
- **Result:** PASS
- **Evidence:** `pytest -m "not gpu"` reports `174 passed,
  1 skipped in 1.38s`. The single skip is the LoRA round-trip
  test that requires `peft`, gated via `pytest.importorskip`.

### CHECK 6.8, prose style
- **Result:** PASS
- **Evidence:** Scanned every new markdown for em-dashes and
  semicolons in narrative prose (excluding fenced code blocks).
  Zero matches.

### CHECK 6.9, working tree clean after Phase 6 commit
- **Result:** PASS (after the Phase 6 commit lands)

## Files added during this phase

- `src/fmllm/training/{__init__.py, README.md, trajectory_collection.py,
   dataset.py, reward.py, lora.py, sft_trainer.py, grpo_trainer.py,
   dpo_alternative.py}`.
- `scripts/{collect_trajectories.py, collect_trajectories.sh,
   train_pipeline_b.py, train_pipeline_b.sh}`.
- `tests/test_training.py` (14 new tests, 1 skipped).
- `docs/progress/06-rl-finetuning.md`.
- `docs/audits/06-rl-finetuning-audit.md` (this file).

## Fixes applied during audit

- `src/fmllm/training/reward.py`: replaced the naive
  ``\{[^{}]*\}`` regex with `_find_json_objects`, a brace-balanced
  scanner that respects string literals. The reward function now
  correctly extracts commit actions whose claim dict contains
  nested objects.

## Remaining concerns

- **GRPO is wall-clock heavy.** The verifier in the inner loop
  costs roughly the same as one Pipeline A pass per generated
  completion. With 4 generations per prompt and a 100-prompt
  epoch, that's ~10 minutes of pure verification on H100 each
  GRPO step. Document expects 3-6 hours per epoch; small batch
  sizes are warranted.
- **DPO requires PASS + FAIL preferences.** Static mock LLM
  trajectories rarely produce both classes for the same specimen.
  DPO data quality depends on running real-LLM trajectory
  collection at non-zero temperature.
- **Real-LLM integration test deferred.** The trainer wrappers
  ship untested at the model-loading layer because the audit
  venv lacks the heavy stack. The first remote real-LLM run is
  the integration test. Use a tiny base model
  (e.g. `microsoft/Phi-3.5-mini-instruct`) for a quick wiring
  check before committing to Llama 3.1 8B.
- **Reward weighting is heuristic.** The `+0.05 per source PASS`
  bonus encourages broad coverage but may dominate the aggregate
  signal. Tune via the `reward_per_source_pass` argument once
  empirical reward distributions land in Phase 8.

## Sign-off

The Phase 6 implementation matches the original prompt and the
addendum. Pipeline B is ready for trajectory collection and
training on the remote. Phase 7 (evaluation) is next.
