# docs/

Project documentation.

## Files

- `architecture.md` - the pipeline diagram and component contracts.
  Phase-specific sections grow as components land.
- `constraints.md` - the three-layer constraint extraction pipeline,
  the metadata schema with per-FM examples, the probe interface, the
  `BridgedFMOutput` schema field by field, and how the verifier
  consumes constraint information from bridged outputs.
- `experiments.md` - the five primary experiments (E1, E2, E3, E4,
  E5) with their hypotheses, manipulations, measurements, and pass
  criteria.
- `data-format.md` - the on-disk dataset and trajectory format.
- `remote-setup.md` - prerequisites and step-by-step bootstrap for the
  remote 4xH100 host.
- `progress/` - one markdown document per phase summarizing what was
  built, what the user should verify, and what remains.
- `audits/` - one self-audit report per phase with pass / fail / fixed
  per check.

## Conventions

- Active voice across all narrative.
- No em-dashes or semicolons in narrative text.
- Code blocks use fenced syntax with the language tag where it helps
  the reader.
