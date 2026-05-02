"""Typed value payload for FM3 bridged outputs.

FM3 emits the moments of a Gamma distribution fitting the per-atom
kinetic-energy distribution. The bridge wraps them in
:class:`GammaKEDistribution`, which records ``alpha``, ``beta``, the
implied mean (``alpha * beta``, equal to temperature in 2D with unit
mass) and the implied variance (``alpha * beta ** 2``).

Produces:
    :class:`GammaKEDistribution` instances.

Depends on:
    pydantic.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GammaKEDistribution(_StrictModel):
    """Gamma moments for the kinetic-energy distribution per atom."""

    alpha: float = Field(gt=0.0)
    beta: float = Field(gt=0.0)
    mean: float = Field(ge=0.0)
    variance: float = Field(ge=0.0)
    implied_temperature_lj: float = Field(ge=0.0)


__all__ = ["GammaKEDistribution"]
