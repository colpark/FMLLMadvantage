# 01 — Project and Testbed

## Research question

> When an LLM is composed with multiple specialist foundation
> models, what is the right information pathway between them?
> Should the LLM see typed JSON outputs, soft connector tokens,
> SAE-derived feature labels, raw activations under steering, or
> trained chain-of-thought traces over probe outputs?

The project tests these alternatives empirically against a
reference architecture (typed JSON outputs + multi-source
verifier) on a controlled testbed.

## Why this question

A frontier-LLM-orchestrated multi-FM system is the dominant
applied-AI pattern in 2025-2026: a general LLM picks tools, calls
specialist models, reasons over their outputs, and commits. The
*architectural* question — what should those tool outputs look like
to maximize the LLM's reasoning capacity — is under-tested. Most
production systems use ad hoc JSON; most research papers use
ad hoc soft tokens or end-to-end joint training. We tested both
and several adjacent variants.

## The sandbox testbed

A synthetic 2D Lennard-Jones cluster identification task. Each
specimen is a configuration of `n_atoms` particles in some motif
(`ring`, `linear`, `triangular_disk`) at some temperature `T`.
Three independent FMs see three independent views of the same
specimen:

| FM | Input | Output | Architecture |
|---|---|---|---|
| FM1 | rasterized image | per-atom energy + image-CLS embedding | small image transformer |
| FM2 | radial distribution function `g(r)` (200 bins) | per-atom energy + RDF-CLS embedding (320-d) | 1D transformer |
| FM3 | trajectory (positions over time) | per-atom energy + trajectory-CLS embedding | 1D transformer |

The LLM (Qwen 2.5 7B Instruct, frozen) is the orchestrator. Its
job is to identify the specimen as `(motif, n_atoms, temperature)`
by calling the FMs as tools and reasoning over their outputs.

## Why this testbed has the properties it has

- **Multi-modal evidence** — three FMs see different views of the
  same physical object. Cross-FM consistency is a real signal.
- **Ground truth** — the dataset is generated with known
  `(motif, n_atoms, T)` labels, so goal accuracy is computable
  exactly.
- **Multiple verifier sources** — physical rules (e.g. ring atoms
  have coordination 2.0), literature references for known
  ground-state energies, a thermal-equilibrium simulator, a
  cross-FM agreement check, and a conformal-prediction coverage
  test all exist.
- **Tractable on one H100** — full training of the FMs is hours,
  not days.
- **Closed world** — train and test draw from the same generative
  process. No domain shift in the testbed itself.

## Output schema

Every commit is a `PhysicalStateClaim`:

```json
{"motif": "triangular_disk", "n_atoms": 11, "temperature": 0.20}
```

`motif ∈ {ring, linear, triangular_disk}`, `n_atoms ∈ [2, 30]`,
`temperature ∈ [0.05, 1.00]`. The verifier returns one of:
`PASS`, `CAVEAT`, `FAIL` (committed) or no commit (parse failure
or budget exhausted).

## Goal accuracy metric

A claim is "correct" iff `motif` matches exactly, `n_atoms`
matches exactly, and `|T_pred - T_gt| <= 0.10` LJ units. The
goal-accuracy score is `right_claims / total_specimens` over the
held-out range.

Verdict-stratified breakdown reports:

- `commit_rate = committed / total` (200 = 1.000 means all
  specimens received a commit, none were budget-exhausted).
- `hallucination_rate = wrong-PASS / total-PASS` (PASS commits
  the verifier OK'd that turn out wrong).
- `calibrated_abstention = wrong-CAVEAT / (wrong-PASS + wrong-CAVEAT)`
  (of all wrong commits, what fraction did the verifier catch).

## Held-out range

Specimens `[40000, 40200)`, locked under
`runs/holdout_lock/ids.json` as of commit `c723eaa`. All Phase 8a
onwards baselines evaluate on this range with locked thresholds.

The training split for FM training is `train_50k`
(specimens 0..49999); SAE labelling and probe-bank training also
draw from there. The literature DB is held-out from training in
the sense that its references were not used in any FM loss
function.

## Phase 8a reference baseline

`full = 0.695` on the held-out range with literature-source
configuration `compare_energy=False` (the post-fix mode that
properly handles finite-temperature data against ground-state
literature references). With strict literature mode the headline
was `0.471`. **The 0.695 number depends on the literature config;
the paper writeup must state this.**

## Conventions used throughout the codebase

- Run IDs follow `YYYYMMDD-HHMMSS-slug` in UTC.
- All artefacts live under `runs/<purpose>/<run_id>/` or
  `checkpoints/<component>/<run_id>/`.
- Manifests (`manifest.yaml`) are written next to every artefact
  and record the inputs, config, and extras for reproducibility.
- Trajectories are stored as JSONL (one
  `Trajectory.model_dump_json()` per line).
- The "verdict P/C/N" notation in side-by-side tables means
  `PASS / CAVEAT / NULL` counts where NULL covers FAIL +
  no-commit.

## Why the test outputs are sometimes confusing

Two things to know:

1. **`naked` and `naked_vision` columns show `goal_accuracy = 0.000`
   with `verdict P/C/N = 0/0/200`.** The verdict is NULL because
   neither baseline runs the verifier — there's no PASS/CAVEAT
   gate. The `goal_accuracy` is computed against ground truth,
   not against the verdict, so 0 means none of the 200 commits
   matched. For `naked` this makes sense (zero observations).
   For `naked_vision` it means BLIP's caption carries no useful
   structural information.

2. **`full_probes` headline is computed over 400 lines** because
   resume appended a second pass on top of the first instead of
   deduplicating. The qualitative finding (probe injection
   underperforms `full`) holds, but the specific 0.562 number is
   over duplicates. Re-run cleanly before reporting it.

## Where to look first

- `src/fmllm/orchestrator/loop.py` — the OHVD loop.
- `src/fmllm/verifier/` — the multi-source verifier and its
  ablation knobs.
- `src/fmllm/fms/` — the three FMs and their bridges.
- `scripts/run_baseline.py` — the canonical Pipeline A entry
  point (drives `naked`, `no_verifier`, `full`).
- `scripts/evaluate_baselines.sh` — the side-by-side runner; it
  auto-discovers any `runs/holdout/*/<run_id>/trajectories.jsonl`
  and evaluates each as a column.

If you're looking at a specific phase result, see
`docs/progress/<phase>.md` for the original narrative.
