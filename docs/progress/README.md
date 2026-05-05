# docs/progress/

One markdown document per pipeline phase. Each document records:

- What I built during the phase.
- What the user must run on the remote to verify the phase.
- What logs and artifacts the user should send back.
- Known issues to flag.
- What remains for the next phase.

## Files

- `00-init.md` - Phase 0, repository initialization.
- `01-testbed.md` - Phase 1, synthetic Lennard-Jones testbed.
- `02-fms.md` - Phase 2, the three foundation models.
- `03-bridges.md` - Phase 3, structure-preserving and language-anchored bridges.
- `04-verifier.md` - Phase 4, multi-source verifier with E4 ablation.
- `05-orchestration.md` - Phase 5, OHVD loop and trajectory storage.
- `06-rl-finetuning.md` - Phase 6, RL fine-tuning (Pipeline B).
- `07-evaluation.md` - Phase 7, eight world-model evaluation tests.
- `08-baselines.md` - Phase 8a, baseline comparison (B0 + B2 + B3) and goal-accuracy metric.
- `09-connector.md` - Phase 9 scaffolding, FM2 representation connector (Layer C) plus the probing study. Closed with a documented negative result.
- `10-ssl-fm2.md` - Phase 10 scaffolding, self-supervised FM2 backbone with masked-RDF reconstruction (Layer D).
