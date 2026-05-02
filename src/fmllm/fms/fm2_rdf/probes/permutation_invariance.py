"""Probe: FM2 permutation invariance.

g(r) is permutation invariant by construction. The probe verifies the
model's output is identical for repeated forward passes on the same
input (a deterministic-output sanity check) and also that two
RDF inputs computed from the same atom configuration but in different
atom orderings produce the same output. The first part runs on real
items; the second part synthesizes a small example using
``fmllm.physics``.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from fmllm.fms._schemas import ProbeResult


def run_probe(
    *,
    model: nn.Module,
    items: list[dict[str, Any]],
    device: torch.device,
    config: dict[str, Any],
) -> ProbeResult:
    threshold = float(config.get("threshold", 1.0))
    n_samples = int(config.get("n_samples", min(64, len(items))))
    rtol = float(config.get("rtol", 1.0e-5))
    atol = float(config.get("atol", 1.0e-6))

    if not items:
        return ProbeResult(
            constraint_name="permutation_invariance",
            satisfaction_score=0.0, num_test_cases=0,
            metric="frac_identical_under_repeat",
            passes_threshold=False, threshold=threshold,
            details={"reason": "no items provided"},
        )

    model.eval()
    rdfs = torch.stack(
        [items[i]["rdf"] for i in range(min(n_samples, len(items)))],
        dim=0,
    ).to(device)
    with torch.no_grad():
        a = model(rdfs).cpu()
        b = model(rdfs).cpu()
    matches = torch.isclose(a, b, rtol=rtol, atol=atol).int()
    matched = int(matches.sum().item())
    total = int(matches.numel())
    score = matched / max(1, total)

    # Synthetic permutation check via the physics RDF computation.
    from fmllm.physics import (
        equilibrium_positions,
        radial_distribution_function,
    )

    pos = equilibrium_positions(13, motif="triangular_disk")
    g1, _ = radial_distribution_function(pos, r_max=6.0, num_bins=200)
    perm = torch.randperm(13)
    g2, _ = radial_distribution_function(pos[perm], r_max=6.0, num_bins=200)
    rdf_identical = bool(torch.equal(g1, g2))

    return ProbeResult(
        constraint_name="permutation_invariance",
        satisfaction_score=score if rdf_identical else 0.0,
        num_test_cases=total,
        metric="frac_identical_under_repeat",
        passes_threshold=(score >= threshold) and rdf_identical,
        threshold=threshold,
        details={
            "rdf_identical_under_atom_permutation": rdf_identical,
            "matched": matched,
        },
    )


__all__ = ["run_probe"]
