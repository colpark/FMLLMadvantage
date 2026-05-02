"""Behavioral probes for FM1.

Each probe exposes ``run_probe(model, items, device, config) -> ProbeResult``
and tests one declared constraint from FM1's ``metadata.yaml``.
"""

from fmllm.fms.fm1_image.probes import (
    atom_count_consistency,
    positions_in_box,
    translation_equivariance,
)

__all__ = [
    "atom_count_consistency",
    "positions_in_box",
    "translation_equivariance",
]
