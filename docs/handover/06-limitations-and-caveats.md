# 06 — Limitations and Caveats

This document is the honest bound on every claim made in
`05-architectural-findings.md`. Read it before generalizing
anything.

## What our setup actually was

| Dimension | What we used | What real-world systems often use |
|---|---|---|
| LLM | Qwen 2.5 7B Instruct, **frozen**, **4-bit nf4 quantized** | frontier reasoning models (o1, Claude Opus, GPT-4-class), full precision, sometimes fine-tuned end-to-end |
| Task type | discretized 3-tuple classification | open-ended generation, scientific dialogue, multi-step planning |
| World | closed synthetic (LJ clusters, generative model known) | open with domain shift, observational noise, novel categories |
| Specimen volume | 200 held-out, 50K train | millions of training samples, larger held-out |
| FM size | 6-depth transformers (small) | order-of-magnitude larger production FMs |
| SAE training data (Phase 13/14) | 20K activations | 1B+ in Templeton et al. |
| SAE training data (Phase 15) | **200 activations** | 1B+ |
| Verifier sources | 5 hand-coded oracles with perfect ground truth | imperfect, noisy, costly, sometimes unavailable |
| Held-out generation | same generative process as train | typically requires distribution shift |

Every one of these is a confound for at least one of our negative
results.

## What each negative result is bounded by

### Phase 9 (connector tokens)

- **Frozen LLM.** Soft-token connectors typically need joint
  fine-tuning of the LLM to the connector projection. We froze
  the LLM, so the LLM had no prior on what the connector tokens
  meant.
- **Templated alignment objective.** The connector training
  rewarded matching templated text, not solving the task. This is
  the classical alignment-vs-task-correctness mismatch.
- *We can claim:* "with a frozen LLM and a templated alignment
  loss, connector tokens did not transfer specimen identity."
- *We cannot claim:* "connector tokens never help."

### Phase 10 (SSL backbone)

- **Reconstruction objective.** Masked-RDF reconstruction rewards
  bin-bin correlations; downstream tasks reward features that
  predict labels. The two are not the same.
- *We can claim:* "masked-RDF SSL gave a poorer downstream
  representation than supervised energy regression in our
  testbed."
- *We cannot claim:* "SSL pretraining never helps." Different SSL
  objectives (contrastive, energy-based, generative) might.

### Phase 11 (CoT-SFT)

- **Stages 3-4 unbuilt.** STaR rejection sampling on
  verifier-PASS trajectories and GRPO with verifier reward are the
  natural complements. We stopped at Stage 2 (synthetic CoT
  bootstrap).
- **Probe-derived CoT only.** The CoT chains were templated from
  five probe outputs, not from the bridged FM tool messages. The
  trained adapter never saw the full evidence the standard
  pipeline gives.
- *We can claim:* "Stage 2 CoT-SFT on probe-derived chains
  underperformed the no-adapter pipeline by 7 points."
- *We cannot claim:* "trained CoT on FM tool use never helps." Or:
  "frontier reasoning model CoT doesn't help here."

### Phase 13 (SAE labels in prompt)

- **20K rows for FM2 SAE.** Adequate for a 320-d input but small
  by SAE standards.
- **Eight features in prompt.** Tunable; a different `top_k`
  might surface different evidence.
- **Frozen LLM.** Same caveat as Phase 9.
- *We can claim:* "SAE labels added to Pipeline A's user message
  underperformed vanilla Pipeline A by 11 points."
- *We cannot claim:* "auto-discovered representation features
  never help LLM reasoning."

### Phase 14 (causal SAE filter)

- **Causal target was FM2's energy head.** Filtering by causal
  effect on energy is not the same as filtering by causal effect
  on the LLM's downstream classification. Two different
  objectives.
- *We can claim:* "filtering SAE features by FM2-energy-head
  causal effect did not reduce the prompt-pollution penalty."
- *We cannot claim:* "causal validation never helps." A different
  causal target (e.g. LLM's classification accuracy under
  intervention) might.

### Phase 15 (SAE on Qwen + steering)

- **200 activations.** Far below Templeton scale. Many features
  are dead at this volume.
- **Single feature, single coefficient.** Only `fid=6844` at
  `coef=-2.0` was actually run. The coefficient sweep and the
  `fid=11357` / `fid=16320` candidates were planned but not
  executed at handover.
- **Last-token harvest only.** One activation per chat. A
  multi-token harvest (all assistant tokens) would 50-200x the
  data.
- **Correlation-based labels.** As we showed, correlation labels
  describe activation, not function.
- *We can claim:* "ablating `fid=6844` at `coef=-2.0` increased
  hallucination on this task. The likely mechanism is that the
  feature is a useful tri-disk-solid detector mislabelled by
  correlation."
- *We cannot claim:* "SAE steering of LLMs doesn't work on
  LLM-FM tool use." Or: "Templeton-style monosemanticity is
  illusory in this regime." Both are open at production scale.

## Known data hygiene issues at handover

1. **`runs/holdout/full_probes/<run_id>/trajectories.jsonl` has
   400 lines for 200 specimens.** Resume logic appended a second
   pass on top of the first. The headline `0.562` is computed
   over duplicates. The qualitative finding (probe injection
   hurts vs `full`) is robust to this; the specific number is
   not. Either:
   ```bash
   tail -n 200 <traj>.jsonl > tmp && mv tmp <traj>.jsonl
   ```
   or rerun `bash scripts/run_baseline_full_probes.sh` cleanly
   from an empty `runs/holdout/full_probes/` directory.

2. **`naked_vision = 0.000` not validated.** The result is
   consistent with "BLIP captions add no useful structural
   information so the LLM emits a default JSON for every
   specimen." The diagnostic to confirm — count unique
   final_claims, check whether they're literally identical —
   was sketched but never run. If it turns out the LLM produces
   varied (but uniformly wrong) commits, the interpretation
   shifts slightly.

3. **`hallucination_rate` discrepancy.** Our diagnostic
   `inspect_qwen_sae_candidates.py` counts 34 wrong-PASS in the
   `full` baseline; the headline reports `hallucination_rate =
   0.255 = 25 wrong-PASS / 98 PASS`. The 9-specimen difference
   is likely a stricter correctness check in our diagnostic
   (exact `n_atoms` match) vs the goal-accuracy scorer (possibly
   ±1 tolerance). Not investigated at handover. The qualitative
   conclusions are unchanged but precise numerical comparisons
   between the two should be done with care.

4. **Phase 15 Stage A's metadata count and the source
   trajectories' wrong-PASS count don't perfectly align.** The
   harvester reports 7 wrong-PASS candidates with `n_top=11`,
   but the source has only 8 wrong-PASS-in-tri-disk-solid
   specimens in the held-out 200. The mismatch is small and
   doesn't change conclusions but flag it if you're trying to
   reproduce specific feature lists.

## Reproducibility caveats

- **`full = 0.695` requires `literature_compare_energy=False`.** This
  is the post-Phase-8a configuration fix. With strict literature
  mode the headline was `0.471` — *below* `cot_sft = 0.467`. The
  literature DB references are ground-state; our data is finite-T;
  strict comparison flags every PASS as CAVEAT-by-energy. Document
  this in any paper writeup.

- **The `c723eaa` commit is the lock point for held-out
  thresholds.** If thresholds change after this commit (verifier
  source thresholds, conformal alpha, etc.), the headline numbers
  change accordingly. Re-run the held-out evaluator to confirm
  any changes.

- **Run IDs are timestamp-based (UTC).** Multiple runs of the
  same baseline produce multiple run-id directories.
  `evaluate_baselines.sh` picks the latest by mtime. If you want
  a specific run, set `BASELINES=...` and pass run-ids.

## Rounding edge cases

- `goal_accuracy` is truncated to 4 decimal places in the
  comparison report. The arithmetic decompositions in
  `04-experiments-and-results.md` use the underlying integer
  counts where possible.
- The `prediction_compression` and `goal_competence` tests
  produce values that don't sum / decompose cleanly across
  baselines — they are diagnostic-only metrics.

## Versions pinned at handover

- `transformers` warns that `torch_dtype` is deprecated in favor
  of `dtype`. Our scripts still use `torch_dtype` because it
  works on the pinned version; if you upgrade, the warning may
  become an error.
- `bitsandbytes` `BitsAndBytesConfig(load_in_4bit=True, ...)` —
  test on a single specimen after any version bump.
- `peft` `PeftModel.from_pretrained(... is_trainable=False)` is
  used in `TransformersLLM`. If a future `peft` removes that
  argument or changes its semantics, fix at the `_ensure_loaded`
  call site.

## What I'd do differently if redoing this project

For another agent picking this up:

1. **Build the resume-aware baseline runner from day one.** Lots
   of compute was wasted to non-resume runs killed by container
   recycles. The resume pattern in
   `scripts/run_baseline_qwen_steered.py` and
   `scripts/run_baseline_full_probes.py` is the template.

2. **Lock the held-out range and goal-accuracy logic in one
   commit early.** We had drift between my diagnostic's
   correctness check and the scorer's. Single source of truth
   would have prevented confusion.

3. **Multi-token activation harvest in Phase 15 from the
   beginning.** 200 rows is too few for an SAE; capturing all
   assistant tokens per chat is essentially free.

4. **Run the `inspect_qwen_sae_candidates` diagnostic *before*
   Stage D**, not after. Half a day of compute would have been
   saved by knowing wrong-PASS commits are concentrated in
   `(ring, *)` and `(tri_disk, liquid-like)` rather than where
   the SAE locked.

5. **Don't rely on a 200-row SAE for steering claims.** The
   pipeline is fine for a smoke test; the conclusions need
   Templeton-scale data before they're publishable as anything
   other than "in the small-data regime."

6. **Document the literature-source configuration in the
   paper-eligible numbers explicitly.** `full = 0.695` is not
   absolute; it's "post-fix mode 0.695, strict mode 0.471."
   Both need to be in any external writeup.
