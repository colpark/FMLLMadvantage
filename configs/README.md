# configs/

YAML configurations for every pipeline stage. Code reads these files
through `fmllm.utils.config.load_config`, which validates them against
the Pydantic schema in `src/fmllm/utils/config.py`. Misconfigurations
fail loudly at load time rather than mid-run.

## Files

- `default.yaml` - the project-wide default configuration. Phase-specific
  configurations override only the keys they need to change.
