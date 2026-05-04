# Held-out evaluation protocol

This document describes the audit-grade evaluation protocol introduced after the Phase 8a self-review. The Phase 8a numbers reported in `docs/project-summary.html` came from iterative debugging on the same 200 specimens, with thresholds adjusted as we observed results. That is appropriate for development but cannot be reported as a fair held-out estimate. This protocol fixes that.

## What this protocol guarantees

1. **Thresholds are locked.** Every threshold is captured in `configs/evaluation_thresholds_locked.yaml`. A drift checker (`scripts/verify_thresholds.py`) refuses to evaluate when code defaults disagree.
2. **The specimen set is locked.** `configs/holdout_lock.yaml` pins a selection rule against the dataset's `splits.yaml` `holdout` partition. The dev set [0, 200) is explicitly excluded; the script verifies non-overlap.
3. **The literature configuration is reported both ways.** The Phase 8a fix that disabled `literature.compare_energy` lifted the headline accuracy from 0.33 to 0.72. The protocol runs the held-out evaluation in both configurations so the alternative is on the record.

## Files

| Path | Role |
|---|---|
| `configs/evaluation_thresholds_locked.yaml` | Frozen threshold snapshot tied to a commit. Every test's default arguments are listed here. |
| `configs/holdout_lock.yaml` | Frozen specimen-selection rule. Names the `splits.holdout` cell, the count, and the dev-set boundary. |
| `scripts/verify_thresholds.py` | Reads the lock, imports each test module, asserts `inspect.signature(measure)` defaults match. Exits non-zero on drift. |
| `scripts/pick_holdout_ids.py` | Reads the lock, opens `splits.yaml`, writes `runs/holdout_lock/ids.json` plus a summary YAML. |
| `scripts/run_holdout.sh` | One-shot runner: verifies thresholds, resolves IDs, runs four configurations, evaluates each. |

## What "held out" means here

Two senses of held out apply:

1. **Held out from the orchestrator.** The Phase 8a dev set was specimens [0, 200). The held-out specimens have never been collected by Pipeline A. This is the primary definition; what we want to know is how the LLM + verifier behave on unseen specimens.
2. **Held out from FM training.** The dataset's `splits.yaml` carries an explicit `holdout` partition that is excluded from the FM train splits.

The current `configs/holdout_lock.yaml` (version 2) uses sense (1) only. The current testbed was generated with `num_holdout=0`, so every specimen is in the FM training pool. Until the testbed is regenerated, the held-out claim is narrower: these specimens are fresh to the orchestrator and verifier, but the FMs have seen them as supervised examples during training.

The picker supports two selection sources via `selection.source`:

* `contiguous_range`: picks `[start, start + count)`. Used by version 2 of the lock.
* `splits.holdout`: pulls from `splits.yaml`'s holdout cells. Use after regenerating the testbed with `num_holdout > 0`.

## Commands to run

On the remote 4xH100 host, after pulling the latest commit:

```bash
git pull

# Sanity check: thresholds in code match the lock
uv run python scripts/verify_thresholds.py

# Sanity check: the held-out selection resolves
uv run python scripts/pick_holdout_ids.py

# Run the full protocol (about 75 minutes wall clock for 200 specimens
# across four configurations on one GPU)
bash scripts/run_holdout.sh

# Or skip the strict-literature footnote if you only want the headline
bash scripts/run_holdout.sh --skip-strict
```

The script prints, in order:

1. Threshold verification (one OK line per threshold).
2. The held-out ID resolution (count and first/last IDs).
3. Four baseline runs: `naked`, `no_verifier`, `full` (default), `full` (strict literature).
4. Two evaluation rollups:
   - Default-literature comparison `naked / no_verifier / full` &mdash; the headline.
   - Strict-literature comparison `naked / no_verifier / full(strict)` &mdash; the footnote.

## Reading the output

Every evaluation prints the same nine-row table plus the verdict-stratified breakdown. The headline numbers to compare:

| Metric | What to look for |
|---|---|
| `goal_accuracy` | Direct comparison across baselines on the held-out set. The publishable number. |
| `hallucination_rate` | Architecturally meaningful only on `full`; both baselines lack abstention. |
| `calibrated_abstention_rate` | Should be 0 for naked and no_verifier by construction. Non-zero for full means the architecture is doing what it claims. |
| `verdict P/C/F` | Reveals whether `full` is rubber-stamping (high PASS), constantly worried (high CAVEAT), or balanced. |

The strict-literature comparison shows what happens with the legacy verifier configuration. If `goal_accuracy(full-strict)` is much lower than `goal_accuracy(full)`, that is the cost of the literature source's ground-state vs finite-T mismatch on this testbed.

## When to re-lock

Generate a new lock (bumping the version number, regenerating the held-out IDs) when:

- A test's default threshold legitimately changes &mdash; for example, a new evaluation campaign at a different problem scale.
- A new evaluation test is added that needs its own threshold registered.
- The dataset is regenerated with different seeds.

In every case, mark the prior held-out result as superseded. Do not edit the locks in place to make a failing run pass.

## What this protocol does not fix

- The thresholds are still chosen by humans who have seen development-set data. Locking only prevents further drift.
- The held-out set is one specific draw of 200 specimens. Confidence intervals on the metrics require many such draws or a much larger held-out set.
- The architecture's behavior on a different LLM, a different testbed, or a different verifier configuration may differ. Phase 8b explicitly tests two of these (Pipeline B adapter and a frontier-model baseline).
