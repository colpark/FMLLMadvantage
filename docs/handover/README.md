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

## The headline findings

**Two architectural pathways achieve near-ceiling goal accuracy
on the held-out range:**

1. **Typed-output contract + multi-source verifier**
   (`full = 0.695`, Phase 8a) — the standard recipe. LLM
   orchestrates FM tools as bridged-JSON evidence, multi-source
   verifier gates with PASS/CAVEAT/FAIL, OHVD loop iterates.

2. **Supervised CoT-SFT over rich representation evidence,
   no verifier** (`cot_sft_sae = 0.650`, Phase 16) — single
   forward, no orchestration loop. LoRA-tuned LLM trained on
   synthetic CoTs that include both probe outputs and SAE-
   derived feature labels, with the final commit anchored to
   ground truth. Within 4.5 points of the verifier ceiling
   without any inference-time orchestration.

**The unifying architectural principle:** representation
features are useful as *training-time supervision* for the LLM
(Phase 16 positive: +18.3 over probes-only CoT-SFT, +54 over
the FM head alone) but harmful as *inference-time prompt
injection* (Phases 13-15 negatives: -11 to -3 vs `full`). The
distinction between training and inference is the load-bearer.

The findings are bounded by our setup (frozen 7B-class LLM,
narrow categorical task, closed-world synthetic data). They
say: *under these conditions on this task, the verifier and
the SAE-augmented CoT-SFT recipe each achieve near-ceiling
performance via different mechanisms; combining them is a
natural follow-up.*

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
