# notebooks/

Exploratory analysis notebooks. Notebooks live here for ad-hoc
investigation and figure prototyping. Production code never lives in a
notebook. Move any logic worth keeping into `src/fmllm/`.

## Conventions

- Restart and run-all before committing.
- Strip outputs that hold heavy binary data.
- Convert finished analyses into scripts under `scripts/` or modules
  under `src/fmllm/` once they stabilize.
