"""Shared utilities for the FMLLMadvantage pipeline.

This subpackage groups infrastructure that every other component reuses.
The four current modules cover logging configuration, manifest writing,
run identifier generation, and Pydantic-validated configuration loading.

Modules:
    logging: configures loguru sinks for stdout and per-run log files.
    manifests: writes artifact manifests with a fixed schema.
    run_ids: generates unique run identifiers of the form
        YYYYMMDD-HHMMSS-<slug>.
    config: defines the Pydantic config schema and a YAML loader.
"""

from fmllm.utils.config import Config, load_config
from fmllm.utils.logging import configure_logging
from fmllm.utils.manifests import write_manifest
from fmllm.utils.run_ids import generate_run_id

__all__ = [
    "Config",
    "configure_logging",
    "generate_run_id",
    "load_config",
    "write_manifest",
]
