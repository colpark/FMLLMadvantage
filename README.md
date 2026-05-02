# FMLLMadvantage

Compositional World Models on a synthetic 2D Lennard-Jones testbed.

The repo implements a pipeline where three small foundation models trained
on different observation modalities (image, radial distribution function,
molecular-dynamics trajectory) reconcile through structure-preserving
bridges and a multi-source verifier. A Llama 3.1 orchestrator drives an
Observe-Hypothesize-Verify-Decide loop that produces verified scientific
chains of thought.

## Status

Phase 4 complete. Phase 0 set up the repository, dependency stack, and
remote bootstrap. Phase 1 added the LJ Hamiltonian, MD integrator,
synthetic-dataset generator CLI, HDF5 dataset reader, and held-out
split machinery. Phase 2 implements the three foundation models (image
ViT, RDF transformer, trajectory transformer) plus their training
loops and split-conformal calibrators. The Phase 2.5 addendum retrofit
elevated constraint extraction to first-class status (per-FM metadata,
behavioral probes, cross-FM tolerance, BridgedFMOutput schema) and
extended training to three nested data scales for the E5 quality
sweep. Phase 3 implements both bridge flavors. Phase 4 implements the
multi-source verifier with rule-library, literature, cross-FM,
simulator, and conformal sources, plus the integrator with the
runtime `sources_config` slot the E4 ablation experiment requires.
The literature database lives at `data/literature/clusters.json`.

Per-phase notes live under `docs/progress/`.

## Layout

- `src/fmllm/` houses the package code grouped by component.
- `configs/` holds YAML configurations validated by Pydantic models.
- `scripts/` contains CLI entry points and the remote bootstrap.
- `tests/` runs locally for code that does not touch GPUs.
- `docs/` holds architecture notes, data formats, the remote-setup guide,
  and the per-phase progress documents.
- `runs/`, `data/`, `checkpoints/` exist on the remote and stay out of git.

## Execution topology

Code authoring happens locally. All training, dataset generation, and
LLM inference happens on the remote 4xH100 host. The remote pulls the
repo from GitHub, runs `scripts/remote_bootstrap.sh`, and then drives
training and experiments through the scripts under `scripts/`.

## Bootstrap on the remote

The user pulls the repo onto the CUDA host and runs:

```
bash scripts/remote_bootstrap.sh
```

Read `docs/remote-setup.md` for the full guide.

## Local development

Local development runs everything that does not require a GPU. The
utility modules, bridges, verifier rule library, and most evaluation
code run on a laptop. Set up the local dev environment with:

```
uv sync --extra dev
```

Then run the local tests:

```
uv run pytest -m "not gpu"
```
