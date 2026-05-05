# Audit Report, Phase 11 (Stage 2 scaffolding)

**Audited at:** 2026-05-04T23:50:00Z
**Auditor:** Claude Code (self-audit)
**Result:** PASS for the Stage 0 + Stage 1 + Stage 2 scaffolding plus
the Phase 11.B held-out evaluation. Phase 11 closes with a documented
negative result on goal accuracy (0.467 vs Pipeline A's 0.695); Stage
3 (rejection sampling) and Stage 4 (verifier-reward RL) remain
intentionally deferred.

## Summary

Phase 11 ships the scaffolding to bootstrap a CoT-over-probes
reasoning pattern via supervised fine-tuning. The chain is:
probe-bank training (Stage 0) → synthetic CoT dataset (Stage 1) →
LoRA SFT on Qwen (Stage 2). Each stage produces an artifact the
next consumes, all on disk under
`checkpoints/probes/`, `runs/cot_datasets/`, and
`checkpoints/cot-sft/` respectively. Phase 6's existing
`train_sft` does the actual training; the new code is the probe
bank + the synthetic-CoT generator + three CLIs.

## Detailed checks

### CHECK 11.1, ProbeBank round-trips correctly
- **Result:** PASS
- **Evidence:** `tests/test_synthetic_cot.py::test_probe_bank_save_load_round_trip`
  saves a toy bank with one regression and one classification
  probe, reloads it, and asserts that both probes produce
  bit-identical outputs on the same input.

### CHECK 11.2, ProbeBank.evaluate returns the documented shape
- **Result:** PASS
- **Evidence:** `tests/test_synthetic_cot.py::test_probe_bank_evaluate_shape`
  exercises a 4-row evaluation. Each row is a dict keyed by probe
  name; each inner dict has `prediction`, `confidence`, `kind`,
  and (for classification) `class_probs`.

### CHECK 11.3, `expected_coordination` is physically faithful
- **Result:** PASS
- **Evidence:** Three tests cover the three motifs.
  - Ring: always 2.0 regardless of N (every atom has exactly 2
    neighbors by construction).
  - Linear: end-corrected mean = 2(N-1)/N. Confirmed at N=2
    (1.0) and N=10 (1.8).
  - Triangular_disk: monotonically increasing in N. The 11-atom
    expected ~3.34 matches the empirical observations from the
    Phase 9 inspector logs.

### CHECK 11.4, `generate_cot` is deterministic
- **Result:** PASS
- **Evidence:** `test_generate_cot_is_deterministic` runs the same
  inputs twice and asserts byte-identical output text plus
  matching `consistent` flags. This is required because the
  dataset builder emits one record per specimen; non-determinism
  would give different SFT supervision across runs.

### CHECK 11.5, CoT mentions probes by name in Step 1
- **Result:** PASS
- **Evidence:** `test_generate_cot_mentions_each_probe` asserts
  each of `atom-count probe`, `motif probe`, `phase probe`,
  `coordination`, `RDF first peak` appears in the rendered text.
  This is what teaches the LLM that probes are inputs.

### CHECK 11.6, final commit comes from ground truth, not probes
- **Result:** PASS
- **Evidence:** `test_generate_cot_commits_ground_truth_verbatim`
  feeds a wildly wrong probe value (n_atoms = 25.0 against truth
  = 11) and asserts (a) the rendered CoT references the wrong
  value of 25.0, and (b) the final_claim still has n_atoms = 11.
  This is the architectural invariant of Phase 11: probes are
  evidence, truth is the answer, the LLM learns to bridge them.

### CHECK 11.7, inconsistent-branch resolution mentions tie-breaking
- **Result:** PASS
- **Evidence:** `test_generate_cot_inconsistent_branch` constructs
  a probe set where motif and coordination disagree, asserts
  `cot.consistent is False`, and confirms the rendered text
  contains "Defer to the highest-confidence probe", which is the
  rule the LLM should learn for handling disagreement.

### CHECK 11.8, build_sft_record matches Phase 6's train_sft contract
- **Result:** PASS
- **Evidence:** `test_build_sft_record_shape` asserts the record
  has top-level `messages` of length 3 with roles
  `[system, user, assistant]`. The trainer `train_sft` consumes
  exactly this shape (read from `src/fmllm/training/sft_trainer.py`).

### CHECK 11.9, scripts compose without state collisions
- **Result:** PASS
- **Evidence:** Each CLI writes to a distinct `runs/.../<run_id>/`
  or `checkpoints/.../<run_id>/` path. Run IDs follow the project
  convention (YYYYMMDD-HHMMSS-slug). `build_cot_dataset.py`
  defaults to "latest probe bank under `checkpoints/probes/`",
  and `train_cot_sft.py` defaults to "latest records.jsonl under
  `runs/cot_datasets/`", so the three stages chain without manual
  path bookkeeping.

### CHECK 11.10, syntax compiles for every new file
- **Result:** PASS
- **Evidence:** `python -c "import ast; ast.parse(open(...).read())"`
  on all eight new files returns OK.

## Scope boundary

Phase 11 ships Stage 2 (supervised bootstrap) only. Three things are
intentionally out of scope:

1. **Stage 3 -- STaR-style rejection sampling.** Sample many CoTs
   from the Stage-2-trained model, keep the ones whose final
   commit passes the verifier, SFT a second round on the kept
   set. Standard self-improvement loop. Builds on the verifier
   infrastructure from Phase 4 plus the Stage 2 adapter. ~1 week
   of work.

2. **Stage 4 -- GRPO with verifier reward.** Direct RL on the
   verifier signal using the Phase 6 GRPO trainer. The reward
   function already exists; the new piece is the prompt format
   that includes probe outputs.

3. **Inference-time integration with the held-out audit.** Phase
   11.B would update `scripts/run_baseline.py` to compute probes
   and inject them into the user prompt when an `ADAPTER_PATH`
   from `checkpoints/cot-sft/` is provided. Without this step the
   trained adapter cannot be evaluated end-to-end against the
   held-out specimens.

These are the natural next phases. Each is conditional on Stage 2
producing a meaningful effect, which is itself conditional on
running the SFT trainer on the remote.

## Phase 11.B held-out result (added post-run)

The trained adapter from Stage 2 was evaluated on the locked
held-out range. Output landed under ``runs/holdout/cot_sft/`` so
the existing held-out evaluator picked it up alongside the Phase
8a baselines.

| Metric | naked | cot_sft | no_verifier | full |
|---|---|---|---|---|
| goal_accuracy | 0.000 | **0.467** | 0.540 | **0.695** |
| commit_rate | 1.000 | 0.975 | 1.000 | 1.000 |
| hallucination_rate | n/a | n/a | 0.460 | 0.255 |
| calibrated_abstention | 0 | 0 | 0 | 0.590 |

The architectural reading (detailed in
``docs/progress/11-cot-sft.md``):

- ``cot_sft`` beats ``naked`` by +47 points -- trained CoT
  reasoning is real and generalizes to unseen specimens.
- ``cot_sft`` is below ``no_verifier`` by 7 points -- the probe
  bank is a lossy summary of the bridged FM tool messages the
  full Pipeline gives the LLM.
- ``cot_sft`` is below ``full`` by 23 points -- the verifier
  provides iterative revision, cross-source consistency, and
  calibrated abstention that single-shot CoT-SFT cannot match.

Phase 11 therefore closes as the third converging negative
result on this testbed, alongside Phase 9 and Phase 10.

## Reproduction

```bash
# Local
uv run pytest tests/test_synthetic_cot.py -v   # 11 tests, no GPU

# Remote
bash scripts/train_probe_bank.sh                                # Stage 0, ~5-10 min
bash scripts/build_cot_dataset.sh                               # Stage 1, ~5-15 min
bash scripts/train_cot_sft.sh                                   # Stage 2, ~2-4 hours
SPECIMEN_IDS_FILE=runs/holdout_lock/ids.json \
    bash scripts/run_baseline_cot.sh                            # Phase 11.B, ~30-50 min
BASELINES_ROOT=runs/holdout bash scripts/evaluate_baselines.sh  # four-way table
```
