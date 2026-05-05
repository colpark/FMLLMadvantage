"""Synthetic chain-of-thought generator for Phase 11 SFT bootstrapping.

Given probe outputs and ground truth for a specimen, produce a
templated reasoning chain that:

  1. States each probe's output in plain language.
  2. Cross-checks the probes against each other using physical
     consistency rules (expected coordination from N + motif).
  3. Resolves disagreements by deferring to the highest-confidence
     probe.
  4. Commits a final claim using ground truth.

The point of training the LLM on these chains is *not* to teach it
to memorize the templates. It is to teach the LLM:

  - Probe outputs are inputs the LLM should explicitly reference.
  - The final commit is the ground-truth answer, not the probe
    consensus -- the LLM must learn to reconcile probes with truth.
  - The reasoning structure (read evidence -> cross-check ->
    resolve -> commit) is the form scientific reasoning should take.

Determinism: the same (probe outputs, ground truth) input must always
produce the same CoT. Tests assert this.

Depends on:
    Stdlib only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Physical-consistency helpers
# ---------------------------------------------------------------------------


def expected_coordination(n_atoms: int, motif: str) -> float:
    """Approximate mean first-shell coordination for a 2D cluster.

    These are heuristics calibrated to the testbed:

        triangular_disk: hexagonal-like packing; interior atoms have 6
            neighbors, boundary atoms have fewer. For N in [5, 30] the
            empirical mean ranges from ~3.0 to ~4.5.
        ring: each atom has exactly 2 neighbors by construction.
        linear: end atoms have 1 neighbor, interior atoms have 2;
            mean is approximately ``2 * (N - 1) / N`` for N >= 2.
    """
    n = max(int(n_atoms), 1)
    motif = motif.lower().replace(" ", "_")
    if motif == "triangular_disk":
        # Interpolated linear fit: ~3.0 at N=5, ~4.4 at N=30.
        return 3.0 + 0.056 * (n - 5)
    if motif == "ring":
        return 2.0
    if motif == "linear":
        if n <= 1:
            return 0.0
        return 2.0 * (n - 1) / n
    return 3.0  # unknown motif: middling default


def coordination_consistent(
    observed: float,
    n_atoms: int,
    motif: str,
    tolerance: float = 0.6,
) -> tuple[bool, float, float]:
    """Return ``(is_consistent, expected, abs_difference)``."""
    expected = expected_coordination(n_atoms, motif)
    diff = abs(observed - expected)
    return (diff <= tolerance, expected, diff)


# ---------------------------------------------------------------------------
# Probe-output access helpers
# ---------------------------------------------------------------------------


def _read(probe_outputs: dict[str, dict[str, Any]], name: str) -> dict[str, Any]:
    """Return ``probe_outputs[name]`` or a sentinel dict if missing."""
    return probe_outputs.get(name) or {"prediction": None, "confidence": 0.0}


def _confidence(value: dict[str, Any]) -> float:
    try:
        return float(value.get("confidence", 0.0))
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Synthetic CoT
# ---------------------------------------------------------------------------


@dataclass
class SyntheticCoT:
    """Output of :func:`generate_cot`. Carries the rendered text plus
    structured fields useful for tests and downstream analysis."""

    text: str
    consistent: bool
    expected_coordination: float
    coordination_difference: float
    final_claim: dict[str, Any]


_SYSTEM_PROMPT = (
    "You are a scientific reasoner working with a 2D Lennard-Jones "
    "cluster testbed. You receive probe outputs derived from a "
    "frozen foundation model and must reason explicitly about the "
    "evidence before committing a typed claim about the specimen's "
    "atom count, motif, and temperature."
)


def _format_sae_feature_dict(
    sae_features: list[tuple[str, float]] | None,
) -> str:
    """Render a list of (label, activation) tuples as a JSON-like string."""
    if not sae_features:
        return ""
    items = [f'"{lab}": {float(act):.2f}' for lab, act in sae_features]
    return "{" + ", ".join(items) + "}"


def _user_message(
    probe_outputs: dict[str, dict[str, Any]],
    sae_features: list[tuple[str, float]] | None = None,
) -> str:
    """The user message the LLM sees at training and inference time."""
    payload = {
        name: {
            "prediction": value.get("prediction"),
            "confidence": round(float(value.get("confidence", 0.0)), 3),
        }
        for name, value in probe_outputs.items()
    }
    parts: list[str] = [
        "PROBES (each derived from a frozen FM head on this specimen): "
        f"{json.dumps(payload, sort_keys=True)}",
    ]
    sae_payload = _format_sae_feature_dict(sae_features)
    if sae_payload:
        parts.append("")
        parts.append(
            "SAE_FEATURES (top-k labelled directions in the FM "
            "representation that activated on this specimen): "
            f"{sae_payload}"
        )
    parts.append("")
    parts.append(
        "Reason through the evidence step by step, cross-check the "
        "probes against each other, then commit a final JSON claim of "
        "the form {\"n_atoms\": int, \"motif\": str, \"temperature\": float}."
    )
    return "\n".join(parts)


def generate_cot(
    *,
    probe_outputs: dict[str, dict[str, Any]],
    ground_truth: dict[str, Any],
    sae_features: list[tuple[str, float]] | None = None,
) -> SyntheticCoT:
    """Render a deterministic templated reasoning chain.

    Args:
        probe_outputs: Output of :meth:`fmllm.training.probe_bank.ProbeBank.evaluate`
            for one specimen. Expected probe names: ``n_atoms``,
            ``motif``, ``phase``, ``coordination``, ``peak_position``.
            Missing probes are tolerated; the chain skips them.
        ground_truth: Dict with ``n``, ``motif``, ``t``.
    """
    n_pred = _read(probe_outputs, "n_atoms")
    motif_pred = _read(probe_outputs, "motif")
    phase_pred = _read(probe_outputs, "phase")
    coord_pred = _read(probe_outputs, "coordination")
    peak_pred = _read(probe_outputs, "peak_position")

    n_int_guess = int(round(float(n_pred.get("prediction") or 0.0)))
    motif_guess = str(motif_pred.get("prediction") or "triangular_disk")
    coord_value = float(coord_pred.get("prediction") or 0.0)
    consistent, expected_coord, diff = coordination_consistent(
        observed=coord_value, n_atoms=n_int_guess, motif=motif_guess,
    )

    lines: list[str] = []
    lines.append("Step 1 - Read the probes:")
    if n_pred["prediction"] is not None:
        lines.append(
            f"  - atom-count probe: N ≈ "
            f"{float(n_pred['prediction']):.1f} "
            f"(confidence {_confidence(n_pred):.2f})"
        )
    if motif_pred["prediction"] is not None:
        lines.append(
            f"  - motif probe     : {motif_pred['prediction']} "
            f"(confidence {_confidence(motif_pred):.2f})"
        )
    if phase_pred["prediction"] is not None:
        lines.append(
            f"  - phase probe     : {phase_pred['prediction']} "
            f"(confidence {_confidence(phase_pred):.2f})"
        )
    if coord_pred["prediction"] is not None:
        lines.append(
            f"  - coordination    : {coord_value:.2f} neighbors per atom"
        )
    if peak_pred["prediction"] is not None:
        lines.append(
            f"  - RDF first peak  : "
            f"{float(peak_pred['prediction']):.2f} LJ units"
        )

    if sae_features:
        lines.append("")
        lines.append(
            "Step 1b - Read the SAE-derived features (auto-discovered "
            "directions in the FM representation, with correlation labels):"
        )
        for lab, act in sae_features[:8]:        # cap rendered count
            lines.append(f"  - {lab} (activation {float(act):.2f})")

    lines.append("")
    lines.append("Step 2 - Cross-check coordination against the structural guess:")
    lines.append(
        f"  A {n_int_guess}-atom {motif_guess} cluster should have mean "
        f"coordination ≈ {expected_coord:.2f}. Observed "
        f"{coord_value:.2f} (difference {diff:.2f}). The probes are "
        f"{'consistent' if consistent else 'inconsistent'} on structure."
    )

    lines.append("")
    lines.append("Step 3 - Resolution:")
    if consistent:
        lines.append(
            "  All probes agree on a coherent structural picture. The "
            "phase probe disambiguates the temperature regime, and the "
            "RDF peak position confirms the LJ length scale."
        )
    else:
        # Identify the most-confident probe among (n_atoms, motif, phase).
        candidates = {
            "atom-count": _confidence(n_pred),
            "motif": _confidence(motif_pred),
            "phase": _confidence(phase_pred),
        }
        leader = max(candidates.items(), key=lambda kv: kv[1])
        lines.append(
            f"  The structural cross-check fails. Defer to the "
            f"highest-confidence probe ({leader[0]}, conf {leader[1]:.2f}) "
            f"and reconcile the others against it before committing."
        )

    final_claim = {
        "n_atoms": int(ground_truth["n"]),
        "motif": str(ground_truth["motif"]),
        "temperature": float(ground_truth["t"]),
    }
    lines.append("")
    lines.append(
        f"Final commit: {json.dumps(final_claim, sort_keys=True)}"
    )

    text = "\n".join(lines)
    return SyntheticCoT(
        text=text,
        consistent=consistent,
        expected_coordination=expected_coord,
        coordination_difference=diff,
        final_claim=final_claim,
    )


# ---------------------------------------------------------------------------
# SFT record builder
# ---------------------------------------------------------------------------


def build_sft_record(
    *,
    probe_outputs: dict[str, dict[str, Any]],
    ground_truth: dict[str, Any],
    specimen_id: int,
    sae_features: list[tuple[str, float]] | None = None,
) -> dict[str, Any]:
    """Assemble the (system, user, assistant) chat record consumed by
    Phase 6's SFT trainer.

    The trainer expects a top-level ``messages`` list. Additional
    fields are kept for traceability but ignored by training.

    When ``sae_features`` is provided, both the user message (input
    seen at training and inference) and the assistant CoT chain
    reference the SAE feature labels and activations explicitly.
    """
    cot = generate_cot(
        probe_outputs=probe_outputs,
        ground_truth=ground_truth,
        sae_features=sae_features,
    )
    return {
        "specimen_id": int(specimen_id),
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _user_message(probe_outputs, sae_features)},
            {"role": "assistant", "content": cot.text},
        ],
        "ground_truth": cot.final_claim,
        "cot_consistent": cot.consistent,
        "expected_coordination": cot.expected_coordination,
        "sae_features_count": (len(sae_features) if sae_features else 0),
    }


__all__ = [
    "SyntheticCoT",
    "build_sft_record",
    "coordination_consistent",
    "expected_coordination",
    "generate_cot",
]
