# FMLLMadvantage — Handover Documentation

This folder is the self-contained pickup point for any future agent
or collaborator on this project. Read it before touching code.

## What is this project?

A research investigation into **how an LLM should orchestrate
reasoning across multiple small foundation models (FMs)**. We
built a sandbox testbed (synthetic Lennard-Jones clusters, three
specialist FMs, a Qwen-class LLM as orchestrator) and ran a
systematic series of architectural variants to find out which
**information pathway** between FM and LLM works best.

## The headline finding

A typed-output contract plus multi-source verifier (`full = 0.695`
goal accuracy on held-out specimens) is the architectural ceiling
on this testbed. Five separate attempts to enrich the
representation pathway between FM and LLM — connector tokens,
self-supervised pretraining of the FM, CoT supervised fine-tuning,
SAE feature injection into the prompt, and SAE activation steering
of the LLM — all landed below `full`. The convergence is the
result.

The negatives are bounded by our setup (frozen 7B-class LLM,
narrow categorical task, small SAE training data, closed-world
synthetic data). They do not refute the methods themselves; they
say *under these conditions on this task, none of these
representation-level pathways improved on a typed-output +
verification baseline*.

## How to read this folder

| File | Purpose | Read if you... |
|---|---|---|
| `01-project-and-testbed.md` | Goal, task, dataset, conventions | are starting from zero |
| `02-architecture-and-components.md` | Code map, components, data flow | need to find or modify code |
| `03-running-the-code.md` | Setup, workflows, infrastructure gotchas | are about to execute anything |
| `04-experiments-and-results.md` | Every phase, with hypothesis, method, headline | want to know what was tried |
| `05-architectural-findings.md` | The conclusions and the mechanisms | want the takeaways |
| `06-limitations-and-caveats.md` | What our results can and can't claim | are about to generalize |
| `07-open-questions-and-bets.md` | Three concrete directions for next research | are choosing what to build next |
| `08-references.md` | External literature and internal docs | want citations |
| `09-handover-checklist.md` | First-day playbook | are picking this up cold |
| `10-literature-by-phase.md` | Phase-by-phase mapping of related literature | want to know how each step relates to prior work |

The progress and audit documents in `docs/progress/` and
`docs/audits/` contain the per-phase narrative. The handover
folder summarizes them; if you want the original chronological
record, those are the source.

## Repository state at handover

- Branch: `main`
- Latest commit: see `git log -1`
- Remote: <https://github.com/colpark/FMLLMadvantage>
- Python: 3.11
- LLM: Qwen 2.5 7B Instruct (default), 4-bit nf4 via bitsandbytes
- FMs: trained checkpoints under `checkpoints/fm1_*/`,
  `checkpoints/fm2_rdf/`, `checkpoints/fm3_*/`
- Held-out specimens: `[40000, 40200)`, locked at commit `c723eaa`
  via `runs/holdout_lock/ids.json`

## Operating discipline

The user runs every test, training, and baseline on a remote 4xH100
host. **Do not run code locally.** Edit, syntax-check, commit, push
— the user pulls and runs on the remote. A persistent memory note
captures this: `~/.claude/projects/-Users-davidpark-Documents-Claude-FMLLM/memory/feedback_no_local_testing.md`.

The shared remote container periodically recycles, killing
long-running processes. Long jobs should support resume; the
steered baseline already does. New runners that loop over many
specimens should follow the same pattern.

## Status of work-in-progress items at handover

- `runs/holdout/full_probes/` contains a duplicated 400-line
  trajectories.jsonl from a resume that appended without
  de-duplicating. The headline 0.562 is computed over the
  duplicates. Either trim with `tail -n 200` or rerun cleanly
  before reporting that number.
- Phase 15 Stages C & D ran on **200 rows** of harvested
  activations. This is a smoke test, not a production-quality
  SAE. The negatives there are not refutations of the method;
  they are evidence within the small-data regime. See
  `06-limitations-and-caveats.md`.
- The `naked_vision = 0.000` result is consistent with "BLIP
  captions carry no useful structural information for this task"
  but the diagnostic to confirm "all 200 commits are identical
  default JSON" was not run.

## Most useful starting commands on the remote

```bash
ssh remote
cd ~/workspace/Temporary/dpark1/scratch/FMLLM/FMLLMadvantage
git pull
ls runs/holdout/                                      # see all baselines
BASELINES_ROOT=runs/holdout bash scripts/evaluate_baselines.sh
```

That regenerates the side-by-side comparison from whatever
trajectory artefacts are on disk. Read its output alongside
`05-architectural-findings.md`.
