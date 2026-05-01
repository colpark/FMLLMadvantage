# fmllm.utils

Shared infrastructure that every other component reuses.

## Files

- `logging.py` - configures loguru sinks for stdout (INFO+) and a
  per-run log file (DEBUG+).
- `manifests.py` - writes artifact manifests with a fixed schema.
- `run_ids.py` - generates `YYYYMMDD-HHMMSS-<slug>` run identifiers.
- `config.py` - Pydantic config schema and YAML loader.
- `__init__.py` - re-exports the public helpers from the four modules.

These modules import only standard-library, pyyaml, pydantic, and
loguru, so they run on any environment without a GPU.
