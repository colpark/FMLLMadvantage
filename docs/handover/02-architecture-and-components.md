# 02 — Architecture and Components

This document maps the architecture to the codebase. Every code
location below is relative to the repo root.

## High-level data flow (Pipeline A, the `full` baseline)

```
Specimen (HDF5)
   |
   v
+--------------------------------------------------------+
|  FMs (three independent; the orchestrator picks calls) |
|                                                        |
|   FM1 (image)          FM2 (RDF)         FM3 (traj.)   |
|       \                   |                  /         |
|        \                  v                 /          |
|         +-> BridgedFMOutput (typed JSON contracts)     |
+--------------------------------------------------------+
                       |
                       v
+---------------------------------------------------------+
|  OHVD loop (orchestrator/loop.py)                       |
|  Observe -> Hypothesize -> Verify -> Decide, max 16 turns
|                                                         |
|       LLM (Qwen 2.5 7B)  <---- chat() with tool messages |
|                                                         |
|       Verifier (multi-source) -> PASS / CAVEAT / FAIL   |
+---------------------------------------------------------+
                       |
                       v
                  Trajectory (JSONL)
                       |
                       v
            Eight world-model evaluations
            + goal_accuracy scorer
```

## Foundation models

### FM1 — image
- Code: `src/fmllm/fms/fm1_image/`
- Architecture: small image transformer over a rasterized
  cluster image (~ViT-tiny class).
- Output: per-atom energy + image-CLS embedding.
- Bridge: `BridgedFMOutput` with `motif_logits`, `n_atoms_estimate`,
  energy point estimate + uncertainty band.

### FM2 — RDF
- Code: `src/fmllm/fms/fm2_rdf/`
- Architecture: 1D transformer over `g(r)` of length 200.
  CLS token aggregates; `embed_dim=320`, `depth=6`, `heads=8`.
- Trained on `train_50k` for energy regression.
- Output: per-atom energy from the energy_head MLP (320 -> 320 -> 1).
- Probes (Phase 11): `n_atoms`, `motif`, `phase`, `coordination`,
  `peak_position`, all small MLPs over the CLS embedding.
- SAE (Phase 13): TopKSAE on the CLS, `hidden_dim=1024`, `k=32`.

### FM3 — trajectory
- Code: `src/fmllm/fms/fm3_traj/` (similar layout)
- Architecture: 1D transformer over a temporal positions tensor.
- Output: per-atom energy + trajectory CLS embedding.

### Bridge schemas

The contract every FM emits to the LLM lives in
`src/fmllm/fms/_schemas.py` as `BridgedFMOutput`. Fields:

- `tool_name` (string)
- `prediction` (typed dict — atom-count int, motif str, energy float)
- `confidence` (calibrated 0..1)
- `dependency` (which other tools this depends on; used by
  cross-FM consistency)
- `applicable_constraint` (which physical constraints this
  output is bound by — e.g. extensive scaling)

The LLM consumes this as a tool message in the chat sequence; the
verifier consumes the same structure to assert consistency.

## Orchestrator

### OHVD loop
- Code: `src/fmllm/orchestrator/loop.py`
- Per-step states: observation, hypothesis, verifier_verdict,
  final, error.
- Termination: `committed` (LLM committed and verifier returned
  PASS/CAVEAT), `budget_exhausted` (16 steps without commit),
  `parse_failure` (LLM emitted unparseable text), `llm_error`
  (HF stack errored).
- The DEFAULT_SYSTEM_PROMPT is exported from
  `fmllm.orchestrator.__init__`.

### LLM wrapper
- Code: `src/fmllm/orchestrator/llm.py`
- `BaseLLM` interface with `chat(messages) -> str`.
- `MockLLM` for offline testing with scripted responses.
- `TransformersLLM` for HF AutoModelForCausalLM. Supports
  optional LoRA adapter via PEFT, optional 4-bit quantization
  via bitsandbytes.

### Steered LLM wrapper (Phase 15)
- Code: `src/fmllm/representation/steered_llm.py`
- Wraps any chat-style LLM and attaches an `ActivationSteerer`
  hook on every `chat()` call.

## Verifier

### Sources
- Code: `src/fmllm/verifier/`
- Five sources, each returning one of `PASS / CAVEAT / FAIL` plus
  rationale text:
  1. `rule_library` — handcrafted physical rules (extensive
     scaling, motif consistency, coordination bounds).
  2. `literature` — known-cluster reference DB at
     `data/literature/clusters.json`. Caveat-on-mismatch logic
     can be enabled or disabled (`literature_compare_energy`).
  3. `cross_fm` — checks the three FMs agree where they overlap
     (e.g. on energy estimate within tolerance).
  4. `simulator` — runs a short MC simulation at the proposed
     temperature and compares emergent properties.
  5. `conformal` — checks the predicted value lies inside the
     calibrated conformal interval.

### Aggregation
- `aggregate_decision` is the worst case across enabled sources
  (PASS only if all PASS; CAVEAT if any CAVEAT, no FAIL; FAIL if
  any FAIL).
- `SourcesConfig.for_ablation("V4")` enables all five.

## Evaluation

### Goal-accuracy scorer
- Code: `src/fmllm/evaluation/`
- Runs the eight world-model tests (compression, distinction,
  recoverability, etc.) plus the goal-accuracy headline.
- Verdict-stratified breakdown produces `commit_rate`,
  `hallucination_rate`, `calibrated_abstention`, and the
  P/C/N count tuple.

### Side-by-side comparison
- Driver: `scripts/evaluate_baselines.sh`
- Auto-discovers any subdir of `BASELINES_ROOT` (default
  `runs/holdout`) that contains a `*/trajectories.jsonl`. Each
  becomes a column in the comparison table.
- Output: `runs/comparisons/<run_id>/comparison.yaml`.

## Per-phase artefacts (representation experiments)

| Phase | Artefact | Location |
|---|---|---|
| 9 (connector) | connector training run | `checkpoints/connector_layerC/` |
| 10 (SSL) | SSL-pretrained FM2 | `checkpoints/fm2_ssl_layerD/` |
| 11 (CoT-SFT) | probe bank | `checkpoints/probes/` |
| 11 | synthetic CoT records | `runs/cot_datasets/` |
| 11 | trained LoRA adapter | `checkpoints/cot-sft/` |
| 13 (FM2 SAE) | TopKSAE on FM2 CLS | `checkpoints/sae/` |
| 13 | feature labels | `runs/sae_labels/` |
| 14 (causal audit) | per-feature CausalEffect | `runs/sae_causal/` |
| 15 (Qwen SAE A) | harvested activations | `runs/qwen_activations/` |
| 15 (Qwen SAE B) | TopKSAE on Qwen residual | `checkpoints/qwen_sae/` |
| 15 (Qwen SAE C) | feature labels + steering candidates | `runs/qwen_sae_labels/` |
| 15 (Qwen SAE D) | steered Pipeline A run | `runs/holdout/full_steered_<fid>_<coef>/` |

## Key code modules to know

| Module | Purpose |
|---|---|
| `fmllm.fms.common` | Checkpoint load/save, common types |
| `fmllm.fms._schemas` | `BridgedFMOutput` typed contract |
| `fmllm.orchestrator.loop` | OHVD loop |
| `fmllm.orchestrator.llm` | LLM wrappers |
| `fmllm.verifier` | Verifier and source implementations |
| `fmllm.training.trajectory_collection` | The collect_trajectories helper that drives baselines (now resume-aware) |
| `fmllm.training.synthetic_cot` | Phase 11 synthetic CoT dataset builder |
| `fmllm.training.probe_bank` | Phase 11 probe bank |
| `fmllm.representation.sae` | TopKSAE module |
| `fmllm.representation.labels` | Phase 13 SAE feature labelling |
| `fmllm.representation.causal` | Phase 14 causal interventions |
| `fmllm.representation.llm_sae` | Phase 15 hooks (Harvester, Steerer) |
| `fmllm.representation.llm_labels` | Phase 15 Stage C labelling |
| `fmllm.representation.steered_llm` | Phase 15 Stage D LLM wrapper |
| `fmllm.evaluation` | Goal accuracy + 8 world-model tests |

## Test layout

CPU-only tests in `tests/`:

- `test_*` for each subsystem.
- Phase 13: `tests/test_sae.py`
- Phase 14: `tests/test_sae_causal.py`
- Phase 15 hooks: `tests/test_llm_sae.py`
- Phase 15 labelling: `tests/test_llm_labels.py`
- Phase 15 steered wrapper: `tests/test_steered_llm.py`

Run with `uv run pytest tests/ -v` (on remote, never local).

## Configuration

- `configs/default.yaml` — FM hyperparameters, dataset paths.
- `pyproject.toml` — Python deps; `uv` is the package manager.

## Where the side-by-side numbers live

- `runs/holdout/<baseline>/<run_id>/trajectories.jsonl` — primary
  artefact (per-specimen claim + verdict).
- `runs/holdout/<baseline>/<run_id>/summary.yaml` — counters.
- `runs/holdout/<baseline>/<run_id>/manifest.yaml` — provenance.
- `runs/eval/<run_id>/report.yaml` — eight world-model tests for
  one baseline.
- `runs/comparisons/<run_id>/comparison.yaml` — multi-baseline
  summary.

When in doubt about a number, trace it back through the manifest
chain to the trajectories that produced it.
