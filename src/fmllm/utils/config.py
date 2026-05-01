"""Pydantic models and loader for the global YAML configuration.

This module defines a single :class:`Config` schema that every script
validates its YAML against. Misconfigurations fail loudly at load time
rather than mid-run.

The schema starts as a stub. Each phase of the project extends a
section, so downstream changes only edit the relevant submodel here
plus the corresponding entry in ``configs/default.yaml``.

Produces:
    Validated :class:`Config` instances loaded from disk.

Depends on:
    pydantic, pyyaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _StrictModel(BaseModel):
    """Base model that rejects unknown keys to catch typos early."""

    model_config = ConfigDict(extra="forbid")


class SeedsConfig(_StrictModel):
    """Random seeds for the major libraries."""

    numpy: int = 0
    torch: int = 0
    python: int = 0


class DatasetConfig(_StrictModel):
    """Synthetic Lennard-Jones dataset parameters.

    The defaults match the Phase 1 plan. We will refine the schema as
    the data generator lands.
    """

    root: str = "data/synthetic_lj_v1"
    num_specimens: int = 50000
    num_holdout: int = 10000
    n_choices: list[int] = Field(
        default_factory=lambda: [5, 7, 9, 11, 13, 17, 19, 21, 25, 30],
    )
    n_in_distribution: list[int] = Field(
        default_factory=lambda: [5, 7, 9, 11, 13],
    )
    n_ood: list[int] = Field(
        default_factory=lambda: [17, 19, 21, 25, 30],
    )
    t_min: float = 0.1
    t_max: float = 2.0
    t_in_distribution_max: float = 1.0
    image_size: int = 64
    rdf_bins: int = 200
    md_steps_per_specimen: int = 100


class FMConfig(_StrictModel):
    """Per-foundation-model training configuration."""

    name: str
    checkpoint_root: str = "checkpoints"
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-2


class VerifierConfig(_StrictModel):
    """Verifier configuration. Phase 4 fleshes this out."""

    literature_db: str = "data/literature/clusters.json"
    conformal_alpha_levels: list[float] = Field(
        default_factory=lambda: [0.10, 0.20],
    )


class OrchestratorConfig(_StrictModel):
    """LLM orchestration configuration. Phase 5 fleshes this out."""

    llm_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    step_budget: int = 16
    temperature: float = 0.2


class Config(_StrictModel):
    """Top-level configuration for the project."""

    run_id_format: str = "%Y%m%d-%H%M%S-{slug}"
    seeds: SeedsConfig = Field(default_factory=SeedsConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    fm1: FMConfig = Field(default_factory=lambda: FMConfig(name="fm1_image"))
    fm2: FMConfig = Field(default_factory=lambda: FMConfig(name="fm2_rdf"))
    fm3: FMConfig = Field(default_factory=lambda: FMConfig(name="fm3_traj"))
    verifier: VerifierConfig = Field(default_factory=VerifierConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config from ``path``.

    Args:
        path: Filesystem path to the YAML file.

    Returns:
        The validated :class:`Config` instance.

    Raises:
        FileNotFoundError: If the path does not exist.
        pydantic.ValidationError: If the YAML fails schema validation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open("r") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    return Config.model_validate(raw)
