"""Labelling for SAE features over LLM activations (Phase 15 Stage C).

Mirrors :mod:`fmllm.representation.labels` (which labels FM2 SAE
features by motif/N/T/phase) but adds the two task-specific axes
that matter on the LLM side:

  * ``verdict`` -- ``pass`` / ``caveat`` / ``fail`` / ``null`` from
    the multi-source verifier. A feature that fires only on
    CAVEAT rows is a candidate for steering toward calibration.
  * ``is_correct`` -- whether the trajectory's final claim matches
    ground truth. A feature that fires only on *wrong* PASS rows
    is the canonical ablation target: clamp it down at inference
    and hallucination rate should drop.

The same correlation-based recipe applies: pick the top-N activating
rows for each feature, check whether they concentrate on a single
category at >= ``min_purity``, and tag continuous attributes whose
Pearson correlation with the activation exceeds ``min_corr``.

Depends on:
    numpy.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np


@dataclass
class LLMFeatureLabel:
    """Structured label record for one Qwen-SAE feature."""

    feature_idx: int
    label: str

    # Task-side locks (specific to LLM features).
    verdict_top: str | None = None
    verdict_purity: float | None = None
    correct_top: bool | None = None
    correct_purity: float | None = None

    # Specimen-side locks (mirror Phase 13's labels).
    motif_top: str | None = None
    motif_purity: float | None = None
    phase_top: str | None = None
    phase_purity: float | None = None
    n_atoms_corr: float | None = None
    temperature_corr: float | None = None

    n_top_activators: int = 0
    activation_mean_top: float = 0.0
    tags: list[str] = field(default_factory=list)


def _categorical_lock(
    values: list, min_purity: float,
) -> tuple[object | None, float]:
    """Return the dominant category if it exceeds ``min_purity``."""
    if not values:
        return None, 0.0
    counts: Counter = Counter(values)
    most, n = counts.most_common(1)[0]
    purity = n / len(values)
    if purity >= min_purity:
        return most, purity
    return None, purity


def _continuous_corr(
    activations: np.ndarray, attribute: np.ndarray,
) -> float:
    if activations.shape[0] < 3:
        return 0.0
    if activations.std() < 1.0e-8 or attribute.std() < 1.0e-8:
        return 0.0
    return float(np.corrcoef(activations, attribute)[0, 1])


def label_llm_feature(
    *,
    feature_idx: int,
    feature_activations: np.ndarray,    # (N_rows,)
    verdicts: np.ndarray,               # (N_rows,) of str
    is_correct: np.ndarray,             # (N_rows,) of bool
    motifs: np.ndarray,                 # (N_rows,) of str
    phases: np.ndarray,                 # (N_rows,) of str
    atom_counts: np.ndarray,            # (N_rows,) of float
    temperatures: np.ndarray,           # (N_rows,) of float
    top_n: int = 50,
    min_purity: float = 0.70,
    min_corr: float = 0.30,
) -> LLMFeatureLabel:
    """Build a label for one LLM-SAE feature.

    The label reports:

      * Whether the feature locks onto a verdict bucket (e.g.
        ``"verdict=caveat (purity 0.80)"``).
      * Whether it locks onto correctness (``"correct=False ..."``).
      * Specimen-side locks: motif, phase, N-correlation, T-correlation.

    Returns a fallback ``"feature-N (rare)"`` label when the feature
    activates on fewer than ``max(5, top_n // 5)`` rows.
    """
    n_rows = int(feature_activations.shape[0])
    if n_rows == 0:
        return LLMFeatureLabel(feature_idx=feature_idx, label=f"f{feature_idx}")

    nonzero = feature_activations > 1.0e-6
    n_nonzero = int(nonzero.sum())
    if n_nonzero < max(5, top_n // 5):
        return LLMFeatureLabel(
            feature_idx=feature_idx,
            label=f"f{feature_idx} (rare)",
            n_top_activators=n_nonzero,
        )

    take = min(top_n, n_nonzero)
    top_idx = np.argsort(feature_activations)[::-1][:take]

    top_verdicts = [str(v) for v in verdicts[top_idx].tolist()]
    top_correct = [bool(c) for c in is_correct[top_idx].tolist()]
    top_motifs = [str(m) for m in motifs[top_idx].tolist()]
    top_phases = [str(p) for p in phases[top_idx].tolist()]
    top_acts = feature_activations[top_idx]

    verdict_top, verdict_purity = _categorical_lock(top_verdicts, min_purity)
    correct_top, correct_purity = _categorical_lock(top_correct, min_purity)
    motif_top, motif_purity = _categorical_lock(top_motifs, min_purity)
    phase_top, phase_purity = _categorical_lock(top_phases, min_purity)

    n_corr = _continuous_corr(
        feature_activations, atom_counts.astype(np.float64),
    )
    t_corr = _continuous_corr(
        feature_activations, temperatures.astype(np.float64),
    )

    tags: list[str] = []
    if verdict_top is not None:
        tags.append(f"verdict={verdict_top}")
    if correct_top is not None:
        tags.append(f"correct={correct_top}")
    if motif_top is not None:
        tags.append(f"motif={motif_top}")
    if phase_top is not None and phase_top != "":
        tags.append(f"phase={phase_top}")
    if abs(n_corr) >= min_corr:
        direction = "large" if n_corr > 0 else "small"
        tags.append(f"N-{direction}(r={n_corr:+.2f})")
    if abs(t_corr) >= min_corr:
        direction = "hot" if t_corr > 0 else "cold"
        tags.append(f"T-{direction}(r={t_corr:+.2f})")

    label = (
        f"f{feature_idx}: " + " + ".join(tags)
        if tags
        else f"f{feature_idx}: unlabelled (no significant pattern)"
    )

    return LLMFeatureLabel(
        feature_idx=feature_idx,
        label=label,
        verdict_top=verdict_top if isinstance(verdict_top, str) else None,
        verdict_purity=(
            verdict_purity if isinstance(verdict_top, str) else None
        ),
        correct_top=correct_top if isinstance(correct_top, bool) else None,
        correct_purity=(
            correct_purity if isinstance(correct_top, bool) else None
        ),
        motif_top=motif_top if isinstance(motif_top, str) else None,
        motif_purity=motif_purity if isinstance(motif_top, str) else None,
        phase_top=phase_top if isinstance(phase_top, str) else None,
        phase_purity=phase_purity if isinstance(phase_top, str) else None,
        n_atoms_corr=n_corr,
        temperature_corr=t_corr,
        n_top_activators=int(take),
        activation_mean_top=float(top_acts.mean()),
        tags=tags,
    )


def rank_features_for_steering(
    labels: list[LLMFeatureLabel],
    *,
    target_axis: str = "correct",
    target_value: object = False,
    min_purity: float = 0.70,
) -> list[LLMFeatureLabel]:
    """Return labels whose top-activator distribution locks onto a target.

    Used to surface candidate features for Stage D steering. Default
    looks for features that fire on *wrong* commits (axis=``correct``,
    value=``False``); these are the canonical "down-clamp" targets.
    Pass ``target_axis="verdict"`` and ``target_value="caveat"`` to
    surface features that fire on uncertain commits, etc.
    """
    out: list[LLMFeatureLabel] = []
    for lab in labels:
        if target_axis == "correct":
            if lab.correct_top is None or lab.correct_purity is None:
                continue
            if lab.correct_top != bool(target_value):
                continue
            if lab.correct_purity < min_purity:
                continue
            out.append(lab)
        elif target_axis == "verdict":
            if lab.verdict_top is None or lab.verdict_purity is None:
                continue
            if lab.verdict_top != str(target_value):
                continue
            if lab.verdict_purity < min_purity:
                continue
            out.append(lab)
        else:
            raise ValueError(f"unknown target_axis {target_axis!r}")
    # Highest purity first; break ties with more top-activators.
    out.sort(
        key=lambda l: (
            -(l.correct_purity or 0.0)
            if target_axis == "correct"
            else -(l.verdict_purity or 0.0),
            -l.n_top_activators,
        )
    )
    return out


__all__ = [
    "LLMFeatureLabel",
    "label_llm_feature",
    "rank_features_for_steering",
]
