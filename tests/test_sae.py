"""Tests for the Phase 13 sparse autoencoder + label-by-correlation.

CPU-only. Cover:

  - TopKSAE forward shape: input (B, in_dim) -> recon (B, in_dim) and
    sparse latent (B, hidden_dim) with at most ``k`` nonzeros per row.
  - Decoder renormalization keeps unit-norm columns.
  - label_feature returns rich tags when patterns exist and falls
    back to "feature-N" when activations are random.
  - Categorical lock fires only when purity exceeds the threshold.
  - Continuous descriptor fires only when |Pearson r| exceeds the
    threshold.
"""

from __future__ import annotations

import numpy as np
import torch

from fmllm.representation.labels import label_feature
from fmllm.representation.sae import TopKSAE


# ---------------------------------------------------------------------------
# SAE
# ---------------------------------------------------------------------------


def test_topk_sae_forward_shape():
    sae = TopKSAE(in_dim=32, hidden_dim=128, k=8)
    x = torch.randn(5, 32)
    recon, z = sae(x)
    assert recon.shape == (5, 32)
    assert z.shape == (5, 128)


def test_topk_sae_enforces_sparsity():
    sae = TopKSAE(in_dim=32, hidden_dim=128, k=8)
    x = torch.randn(7, 32)
    _, z = sae(x)
    nonzero_per_row = (z > 0).sum(dim=-1)
    # k = 8 with positive activations after ReLU; at most k can be
    # nonzero per row.
    assert (nonzero_per_row <= 8).all()


def test_topk_sae_decoder_renorm_keeps_unit_columns():
    sae = TopKSAE(in_dim=16, hidden_dim=64, k=4, normalize_decoder=True)
    # Perturb decoder weights, then renormalize.
    sae.decoder.weight.data.normal_()
    sae._renormalize_decoder()
    norms = sae.decoder.weight.data.norm(dim=0)
    # Each column should have unit norm to within float tolerance.
    assert torch.allclose(norms, torch.ones_like(norms), atol=1.0e-5)


def test_topk_sae_full_capacity_when_k_equals_hidden():
    """When k == hidden_dim, no zero-out happens; the latent is just
    relu(encoder(x))."""
    sae = TopKSAE(in_dim=8, hidden_dim=8, k=8)
    x = torch.randn(3, 8)
    _, z = sae(x)
    assert z.shape == (3, 8)
    # ReLU output is non-negative.
    assert (z >= 0).all()


# ---------------------------------------------------------------------------
# Label by correlation
# ---------------------------------------------------------------------------


def _fake_attributes(n: int = 200, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    motifs = rng.choice(["triangular_disk", "ring", "linear"], size=n, p=[0.7, 0.2, 0.1])
    atom_counts = rng.integers(5, 30, size=n).astype(np.float32)
    temperatures = rng.uniform(0.1, 2.0, size=n).astype(np.float32)
    phases = np.where(
        temperatures < 0.30, "solid-like",
        np.where(temperatures < 1.0, "liquid-like", "gas-like"),
    )
    return {
        "motifs": motifs.astype(str),
        "atom_counts": atom_counts,
        "temperatures": temperatures,
        "phases": phases.astype(str),
    }


def test_label_feature_locks_motif_when_top_activators_share_motif():
    attrs = _fake_attributes(n=200)
    # Build a feature that activates only on the first 50 specimens,
    # which we force to be all "ring".
    attrs["motifs"][:50] = "ring"
    activations = np.zeros(200)
    activations[:50] = np.linspace(2.0, 1.0, 50)
    rec = label_feature(
        feature_idx=42,
        feature_activations=activations,
        motifs=attrs["motifs"],
        atom_counts=attrs["atom_counts"],
        temperatures=attrs["temperatures"],
        phases=attrs["phases"],
        top_n=50,
    )
    assert rec.motif_top == "ring"
    assert rec.motif_purity is not None and rec.motif_purity >= 0.9
    assert "motif=ring" in rec.label


def test_label_feature_locks_phase_when_top_activators_are_solid():
    attrs = _fake_attributes(n=200)
    # Force the top activators to be cold specimens.
    attrs["temperatures"][:50] = np.linspace(0.05, 0.25, 50)
    attrs["phases"][:50] = "solid-like"
    activations = np.zeros(200)
    activations[:50] = np.linspace(2.0, 1.0, 50)
    rec = label_feature(
        feature_idx=7,
        feature_activations=activations,
        motifs=attrs["motifs"],
        atom_counts=attrs["atom_counts"],
        temperatures=attrs["temperatures"],
        phases=attrs["phases"],
        top_n=50,
    )
    assert rec.phase_top == "solid-like"
    assert "phase=solid-like" in rec.label


def test_label_feature_continuous_descriptor_for_temperature_correlation():
    # Build an activation pattern strongly correlated with temperature.
    n = 200
    rng = np.random.default_rng(1)
    temperatures = np.linspace(0.1, 2.0, n)
    activations = (
        temperatures + 0.05 * rng.standard_normal(n)
    ).astype(np.float32)
    activations = np.clip(activations, 0.0, None)
    motifs = np.full(n, "triangular_disk")
    phases = np.where(
        temperatures < 0.30, "solid-like",
        np.where(temperatures < 1.0, "liquid-like", "gas-like"),
    )
    atom_counts = np.full(n, 12.0)
    rec = label_feature(
        feature_idx=11,
        feature_activations=activations,
        motifs=motifs,
        atom_counts=atom_counts,
        temperatures=temperatures.astype(np.float32),
        phases=phases.astype(str),
        top_n=50,
        min_purity=0.99,   # avoid the categorical lock
        min_corr=0.30,
    )
    assert rec.temperature_corr is not None
    assert rec.temperature_corr > 0.30
    assert "T-hot" in rec.label or "T-cold" in rec.label


def test_label_feature_falls_back_when_no_pattern():
    n = 200
    rng = np.random.default_rng(2)
    activations = rng.standard_normal(n).clip(min=0.0).astype(np.float32)
    motifs = rng.choice(["triangular_disk", "ring", "linear"], size=n, p=[0.4, 0.3, 0.3])
    rec = label_feature(
        feature_idx=99,
        feature_activations=activations,
        motifs=motifs.astype(str),
        atom_counts=rng.integers(5, 30, size=n).astype(np.float32),
        temperatures=rng.uniform(0.1, 2.0, size=n).astype(np.float32),
        phases=np.full(n, "liquid-like"),
        top_n=50,
        min_purity=0.95,
        min_corr=0.95,
    )
    # No clear pattern, so the rendered label should fall back to
    # the "unlabelled" sentinel rather than tagging anything.
    assert "unlabelled" in rec.label or "feature-99" in rec.label
