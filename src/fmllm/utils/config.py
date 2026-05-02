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
    """Synthetic Lennard-Jones dataset parameters."""

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

    # MD integrator and equilibration.
    md_dt: float = 0.005
    md_equilibration_steps: int = 400
    md_thermostat_every: int = 20
    confinement_k: float = 0.05

    # Image rasterization.
    image_pixel_size_lj: float = 0.15
    image_blur_radius_lj: float = 0.20
    image_noise_std: float = 0.0

    # RDF binning and pair-distance window.
    rdf_r_max: float = 6.0

    # Generator runtime.
    perturbation_std: float = 0.02
    generator_batch_size: int = 256
    generator_master_seed: int = 1234
    holdout_fraction: float | None = None  # None means use num_holdout


class FMConfig(_StrictModel):
    """Common training fields for every foundation model."""

    name: str
    checkpoint_root: str = "checkpoints"
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-2
    grad_clip: float = 1.0
    warmup_epochs: int = 2
    num_workers: int = 4
    mixed_precision: bool = True
    val_fraction: float = 0.10
    calib_fraction: float = 0.10
    conformal_alpha_levels: list[float] = Field(default_factory=lambda: [0.10, 0.20])


class FM1Config(FMConfig):
    """FM1 image Vision Transformer hyperparameters."""

    name: str = "fm1_image"
    image_size: int = 64
    patch_size: int = 8
    embed_dim: int = 256
    encoder_depth: int = 8
    decoder_depth: int = 4
    num_heads: int = 8
    mlp_ratio: float = 4.0
    num_queries: int = 32
    max_n_atoms: int = 30
    box_half_width_lj: float = 4.8
    count_weight: float = 1.0
    position_weight: float = 5.0
    confidence_weight: float = 1.0
    box_constraint_weight: float = 0.1


class FM2Config(FMConfig):
    """FM2 RDF Transformer hyperparameters."""

    name: str = "fm2_rdf"
    rdf_bins: int = 200
    embed_dim: int = 320
    depth: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    energy_floor: float = -3.0
    nonneg_weight: float = 0.1
    huber_delta: float = 0.5


class FM3Config(FMConfig):
    """FM3 trajectory Transformer hyperparameters."""

    name: str = "fm3_traj"
    n_steps_input: int = 100
    max_n_atoms: int = 30
    embed_dim: int = 320
    depth: int = 10
    num_heads: int = 8
    mlp_ratio: float = 4.0
    equipartition_weight: float = 0.1
    nll_clip: float = 50.0


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
    fm1: FM1Config = Field(default_factory=FM1Config)
    fm2: FM2Config = Field(default_factory=FM2Config)
    fm3: FM3Config = Field(default_factory=FM3Config)
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
