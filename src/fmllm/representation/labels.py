"""Correlation-based labelling of SAE features.

For each SAE feature i, find the specimens that activate it most
strongly, then describe what those specimens have in common using
the dataset's structured ground-truth fields (motif, atom count,
temperature, phase). The result is a human-readable label per
feature.

Two label kinds:

* **Categorical lock** -- when a feature's top activators concentrate
  on one motif (or one phase, or one N bucket) at >= ``min_purity``,
  we tag the feature with that category. Example label:
  ``"motif=triangular_disk (purity 0.94, n=50)"``.

* **Continuous descriptor** -- when activations correlate with N or T
  with |Pearson r| >= ``min_corr``, we tag the feature with the
  direction. Example label: ``"N-large (r=+0.78)"``.

A feature can carry multiple tags concatenated with ``+``. Features
that meet neither bar get a fallback label of the form
``"feature-<idx>"``. The labelling stays deterministic given the
input statistics so the labels are reproducible and auditable.

Depends on:
    numpy.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np


@dataclass
class FeatureLabel:
    """One feature's structured label record."""

    feature_idx: int
    label: str
    motif_top: str | None = None
    motif_purity: float | None = None
    n_atoms_corr: float | None = None
    temperature_corr: float | None = None
    phase_top: str | None = None
    phase_purity: float | None = None
    n_top_activators: int = 0
    activation_mean_top: float = 0.0
    tags: list[str] = field(default_factory=list)


def _categorical_lock(
    values: list[str], min_purity: float,
) -> tuple[str | None, float]:
    counts = Counter(values)
    if not counts:
        return None, 0.0
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


def label_feature(
    *,
    feature_idx: int,
    feature_activations: np.ndarray,    # (N_specimens,)
    motifs: np.ndarray,                 # (N_specimens,) of str
    atom_counts: np.ndarray,            # (N_specimens,) of float
    temperatures: np.ndarray,           # (N_specimens,) of float
    phases: np.ndarray,                 # (N_specimens,) of str
    top_n: int = 50,
    min_purity: float = 0.70,
    min_corr: float = 0.30,
) -> FeatureLabel:
    """Build a label for one SAE feature.

    Args:
        feature_idx: Index in the SAE codebook.
        feature_activations: Activation of this feature on each of the
            N labelling specimens.
        motifs, atom_counts, temperatures, phases: Per-specimen
            ground-truth attributes.
        top_n: Number of highest-activating specimens to summarize.
        min_purity: Threshold for a categorical lock. 0.70 means at
            least 70% of the top activators share the same category.
        min_corr: Threshold for a continuous descriptor (Pearson r).

    Returns:
        A :class:`FeatureLabel` with the structured fields plus a
        rendered ``label`` string.
    """
    n_specimens = feature_activations.shape[0]
    if n_specimens == 0:
        return FeatureLabel(feature_idx=feature_idx, label=f"feature-{feature_idx}")

    # If the feature is essentially never active, skip rich labelling.
    nonzero = feature_activations > 1.0e-6
    n_nonzero = int(nonzero.sum())
    if n_nonzero < max(5, top_n // 5):
        return FeatureLabel(
            feature_idx=feature_idx, label=f"feature-{feature_idx} (rare)",
            n_top_activators=n_nonzero,
        )

    take = min(top_n, n_nonzero)
    top_idx = np.argsort(feature_activations)[::-1][:take]
    top_motifs = motifs[top_idx].tolist()
    top_phases = phases[top_idx].tolist()
    top_n_atoms = atom_counts[top_idx]
    top_temps = temperatures[top_idx]
    top_acts = feature_activations[top_idx]

    motif_top, motif_purity = _categorical_lock(top_motifs, min_purity)
    phase_top, phase_purity = _categorical_lock(top_phases, min_purity)
    n_corr = _continuous_corr(feature_activations, atom_counts.astype(np.float64))
    t_corr = _continuous_corr(feature_activations, temperatures.astype(np.float64))

    tags: list[str] = []
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

    if tags:
        label = f"f{feature_idx}: " + " + ".join(tags)
    else:
        label = f"f{feature_idx}: unlabelled (no significant pattern)"

    return FeatureLabel(
        feature_idx=feature_idx,
        label=label,
        motif_top=motif_top,
        motif_purity=motif_purity if motif_top is not None else None,
        n_atoms_corr=n_corr,
        temperature_corr=t_corr,
        phase_top=phase_top,
        phase_purity=phase_purity if phase_top is not None else None,
        n_top_activators=int(take),
        activation_mean_top=float(top_acts.mean()),
        tags=tags,
    )


__all__ = ["FeatureLabel", "label_feature"]
