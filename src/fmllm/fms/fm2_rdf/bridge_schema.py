"""Typed value payload for FM2 bridged outputs.

FM2 emits a single scalar: per-atom potential energy in LJ units.
The bridge wraps it in :class:`EnergyPerAtom`.

Produces:
    :class:`EnergyPerAtom` instances.

Depends on:
    pydantic.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EnergyPerAtom(_StrictModel):
    """Coarse-grained per-atom potential energy in LJ units."""

    value_lj: float


__all__ = ["EnergyPerAtom"]
