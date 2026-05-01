# src/fmllm/

The core Python package. Every component of the pipeline lives in a
subpackage so the dependency graph stays explicit and tests can target
one component at a time.

## Subpackages

- `data/` - synthetic Lennard-Jones data generation, HDF5 I/O, splits.
- `fms/` - the three foundation models: image (FM1), RDF (FM2),
  trajectory (FM3).
- `bridges/` - structure-preserving and language-anchored bridges from
  FM outputs to LLM-consumable artifacts.
- `verifier/` - the multi-source verifier with rule, literature,
  cross-FM, simulator, and conformal sources.
- `orchestrator/` - LLM wrapper plus the Observe-Hypothesize-Verify-
  Decide loop and trajectory data structures.
- `training/` - RL fine-tuning utilities (Pipeline B).
- `evaluation/` - the eight world-model tests.
- `physics/` - LJ Hamiltonian, MD integrator, observables.
- `utils/` - shared infrastructure: logging, manifests, run IDs, config.

## Imports

The top-level `fmllm` namespace stays thin. Reach into the subpackage
you need directly, for example `from fmllm.utils import load_config` or
`from fmllm.physics import lj_potential`.
