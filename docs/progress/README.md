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

Subsequent phases (Phase 8 and Phase 9) add one progress document
each as those phases land.
