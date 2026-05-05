# 09 — Handover Checklist

A first-day playbook for someone picking this project up cold.

## Day 0 — read these documents in this order

1. `README.md` (5 min) — what this project is.
2. `01-project-and-testbed.md` (15 min) — task and dataset.
3. `05-architectural-findings.md` (15 min) — what we concluded.
4. `06-limitations-and-caveats.md` (10 min) — what bounds those
   conclusions.
5. `07-open-questions-and-bets.md` (10 min) — what to do next.

If something doesn't match what you see in the codebase or
results, prefer the codebase / results — these docs were written
once and may have drifted.

## Day 1 — verify the development environment

On the remote 4xH100 host:

```bash
ssh remote
cd ~/workspace/Temporary/dpark1/scratch/FMLLM/FMLLMadvantage
git pull
git log -5 --oneline                         # confirm at handover commit

# Set up uv per the rules in 03-running-the-code.md
export UV_LINK_MODE=copy
export UV_PROJECT_ENVIRONMENT="$HOME/.cache/fmllm-venv"
uv sync --extra dev

# Run the CPU test suite
uv run pytest tests/ -v
```

All tests should pass. If they don't, that's the first thing to
fix.

## Day 1 — verify the held-out evaluator runs

```bash
BASELINES_ROOT=runs/holdout bash scripts/evaluate_baselines.sh
```

Compare the `HEADLINE` line against the table in
`04-experiments-and-results.md`. If numbers match, the artefacts
on disk are intact.

## Day 1 — read one trajectory by hand

```bash
LATEST=$(ls -td runs/holdout/full/*/ | head -1)
head -1 "${LATEST}/trajectories.jsonl" | uv run python -m json.tool
```

This shows the structure of a single specimen's OHVD record:
the original query, the steps (observation/hypothesis/verifier
verdict turns), the final claim, the final verdict, the
metadata. Understanding this format is essential for any
downstream work.

## Days 2-3 — choose your direction

Based on `07-open-questions-and-bets.md`, pick one of:

- **Cleanup work**: clean rerun of `full_probes`,
  `naked_vision` diagnostic. Half a day each. Tightens existing
  numbers.
- **Phase 15 sweep**: positive-coefficient steering on
  `fid=6844`, plus `fid=11357` and `fid=16320` ablation. ~1 day
  of GPU. Firms up or flips the Phase 15 negative.
- **Phase 15 scale**: multi-token harvest + larger trajectory
  source. ~1 week. Cheapest path to make Phase 15 production-
  quality.
- **CoT Stage 3-4**: rejection sampling + GRPO. ~2 weeks. Closes
  the 23-point gap if it can be closed.
- **One of the three larger bets**: scale interpretability,
  causal abstraction, testbed reframe. Multi-month commitments.

If unsure, do the Phase 15 sweep first — cheapest, most
diagnostic.

## Operational rules to remember

These come from accumulated experience on this codebase:

1. **Never run `pytest`, `uv sync`, or scripts locally.** The
   user runs everything on remote. Local is for editing,
   syntax-checking, committing, pushing. (Memory note: `~/.claude/projects/-Users-davidpark-Documents-Claude-FMLLM/memory/feedback_no_local_testing.md`.)

2. **NFS hardlinks fail.** Always have `UV_LINK_MODE=copy`
   exported. Move venvs off NFS to `$HOME/.cache/`.

3. **Container recycles.** Long jobs need resume support and
   nohup. Both are documented in `03-running-the-code.md`.

4. **Resume + duplication.** Always `wc -l` after a resumed run
   to confirm the JSONL has the expected line count. The
   `full_probes` 400-line bug is the cautionary tale.

5. **Validate diagnostics before steering.** Phase 15's `fid=6844`
   ablation cost compute that would have been better spent if we
   had run `inspect_qwen_sae_candidates.sh` first to see that
   wrong-PASS isn't where the SAE locked.

6. **Read the literature config in any verifier-using number.**
   `full = 0.695` requires `literature_compare_energy=False`.
   Strict mode gives `0.471`. Document this in any external
   writeup.

## Things to flag to the user

When you start, the user may want to know:

- **What state are the runs/ artefacts in?** — `ls runs/holdout/`
  shows what baselines are computed and current.
- **Is the comparison table reproducible from the existing
  artefacts?** — yes, via `evaluate_baselines.sh` (Day 1 step
  above).
- **Are there pending cleanups?** — yes, `full_probes` 400-line
  duplication, `naked_vision` diagnostic, Phase 15 coefficient
  sweep.
- **What's the recommended next step?** — depends on time budget.
  See `07-open-questions-and-bets.md`.

## Communication style with the user (from past sessions)

The user prefers:

- **Direct technical answers**, not hedged or marketing-style
  phrasing.
- **Concrete commands and file paths**, not abstract
  descriptions.
- **Honest reading of negative results** — when something didn't
  work, say so cleanly with the mechanism.
- **Stop and ask before destructive operations** (force pushes,
  deletes, etc.).
- **Auto mode is often active**: minimize interruptions, prefer
  action over planning, expect course corrections.

The user has been thorough in this project — they will catch
glossing-over and ask hard questions. Don't paper over
limitations or extrapolate beyond what the data supports.

## What "done" looks like for the next phase

If you take the Phase 15 sweep:
- Three additional `full_steered_*` columns in the side-by-side.
- A clear narrative on whether any feature/coefficient
  combination beats `full = 0.695`.
- Updated `04-experiments-and-results.md` and
  `05-architectural-findings.md`.

If you take the Phase 15 scale-up:
- 50K+ activation rows in `runs/qwen_activations/<new>/`.
- New SAE checkpoint with denser feature space.
- Re-run Stages C and D; report whether new candidates emerge.

If you take Bet 3 (testbed reframe):
- New testbed dataset + scorer.
- Re-run of all six negative-result axes on the new task.
- Comparative analysis of which conclusions transfer.

Whatever you do, **update this handover folder** before signing
off. Future agents will read it the same way you read it.

## When to write back to the user

- After every milestone (one phase complete).
- When you hit a blocker that needs decision input.
- When numbers change in ways the paper would care about.
- Never just to acknowledge — the user prefers terse, content-
  rich updates.

## Getting unstuck

If something is unclear:

1. Read the original `docs/progress/<phase>.md` for the
   chronological narrative.
2. Read `docs/audits/<phase>-audit.md` for the pass/fail check
   list and reproduction protocol.
3. Read the source code for the module in question — it's
   reasonably commented.
4. Ask the user. They have context this folder doesn't capture.

Welcome to the project.
