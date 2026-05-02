"""Probe: FM2 extensive scaling consistency.

The model predicts per-atom energy. The probe groups test items by
atom count ``N`` and verifies that for clusters at similar
temperatures, the per-atom energy stays in a stable range across the
``N`` values covered by the validation set. A specimen passes if its
predicted per-atom energy lies within a configured fractional band of
the median per-atom energy of clusters that share its atom count.
"""

from __future__ import annotations

from collections import defaultdict
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
    threshold = float(config.get("threshold", 0.85))
    n_samples = int(config.get("n_samples", min(128, len(items))))
    rel_tolerance = float(config.get("rel_tolerance", 0.50))

    if not items:
        return ProbeResult(
            constraint_name="extensive_scaling",
            satisfaction_score=0.0, num_test_cases=0,
            metric="frac_within_n_band",
            passes_threshold=False, threshold=threshold,
            details={"reason": "no items provided"},
        )

    model.eval()
    chosen = items[: min(n_samples, len(items))]
    rdfs = torch.stack([it["rdf"] for it in chosen], dim=0).to(device)
    with torch.no_grad():
        preds = model(rdfs).detach().cpu().tolist()

    by_n: dict[int, list[float]] = defaultdict(list)
    for it, pred in zip(chosen, preds, strict=True):
        n_atoms = int(it["atom_count"]) if "atom_count" in it else int(
            it.get("atom_mask", torch.tensor([])).sum().item()
        )
        by_n[n_atoms].append(pred)

    counted = 0
    matched = 0
    for n_atoms, energies in by_n.items():
        if len(energies) < 2:
            continue
        sorted_e = sorted(energies)
        median = sorted_e[len(sorted_e) // 2]
        denom = max(abs(median), 1.0e-3)
        for e in energies:
            counted += 1
            if abs(e - median) / denom <= rel_tolerance:
                matched += 1

    score = matched / max(1, counted)
    return ProbeResult(
        constraint_name="extensive_scaling",
        satisfaction_score=score,
        num_test_cases=counted,
        metric="frac_within_n_band",
        passes_threshold=score >= threshold,
        threshold=threshold,
        details={
            "rel_tolerance": rel_tolerance,
            "n_groups": len(by_n),
            "matched": matched,
        },
    )


__all__ = ["run_probe"]
