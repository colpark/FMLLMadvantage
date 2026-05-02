"""Behavioral probes for FM3."""

from fmllm.fms.fm3_traj.probes import (
    distribution_non_negativity,
    distribution_normalization,
    equipartition,
)

__all__ = [
    "distribution_non_negativity",
    "distribution_normalization",
    "equipartition",
]
