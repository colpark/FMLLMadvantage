"""Tests for Phase 14 causal interventions on SAE features.

CPU-only. Cover:

  - Intervention.apply zeros the right column for KNOCK_OUT and sets
    the right value for KNOCK_IN / CLAMP, leaving other columns
    untouched.
  - cls_through_sae round-trips with no intervention to the SAE's
    own reconstruction.
  - audit_feature returns a record with all the fields populated and
    consistent.
  - When the energy head is wired to read exactly feature i (via a
    constructed SAE with one-hot decoder columns and a head that
    selects column i of the CLS), knock_out_effect on feature i is
    large and on a different feature j is essentially zero. This is
    the discriminative test the audit must satisfy.
  - filter_features_by_causal_effect respects both the norm-effect
    threshold and the activation-rate gate.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from fmllm.representation.causal import (
    CausalEffect,
    Intervention,
    InterventionKind,
    audit_feature,
    cls_through_sae,
    filter_features_by_causal_effect,
    normalize_cls,
    predict_energy,
)
from fmllm.representation.sae import TopKSAE


# ---------------------------------------------------------------------------
# Intervention
# ---------------------------------------------------------------------------


def test_intervention_knock_out_zeros_only_target_column():
    z = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = Intervention(InterventionKind.KNOCK_OUT, feature_idx=1).apply(z)
    assert torch.equal(out[:, 0], z[:, 0])
    assert torch.equal(out[:, 1], torch.zeros(2))
    assert torch.equal(out[:, 2], z[:, 2])
    # Original is not mutated.
    assert z[0, 1].item() == 2.0


def test_intervention_knock_in_sets_value():
    z = torch.zeros(3, 4)
    out = Intervention(
        InterventionKind.KNOCK_IN, feature_idx=2, value=7.5,
    ).apply(z)
    assert torch.equal(out[:, 2], torch.tensor([7.5, 7.5, 7.5]))
    assert out[:, 0].abs().sum().item() == 0.0


def test_intervention_clamp_overrides_existing():
    z = torch.tensor([[0.0, 9.9], [3.0, 0.5]])
    out = Intervention(
        InterventionKind.CLAMP, feature_idx=1, value=-1.0,
    ).apply(z)
    assert torch.equal(out[:, 1], torch.tensor([-1.0, -1.0]))


# ---------------------------------------------------------------------------
# cls_through_sae
# ---------------------------------------------------------------------------


def test_cls_through_sae_no_intervention_matches_recon():
    torch.manual_seed(0)
    sae = TopKSAE(in_dim=8, hidden_dim=16, k=4, normalize_decoder=False)
    cls_norm = torch.randn(3, 8)
    direct_recon, _ = sae(cls_norm)
    via_helper = cls_through_sae(sae=sae, cls_norm=cls_norm, intervention=None)
    assert torch.allclose(direct_recon, via_helper, atol=1.0e-7)


# ---------------------------------------------------------------------------
# audit_feature
# ---------------------------------------------------------------------------


def _make_id_sae_and_head(
    in_dim: int = 4, hidden_dim: int = 4, k: int = 4, target_idx: int = 0,
) -> tuple[TopKSAE, nn.Module]:
    """Construct an SAE whose latent z[i] equals input dim i, and a
    head that selects exactly one CLS dim. Used to force a known
    causal structure for the discriminative test below.
    """
    sae = TopKSAE(in_dim=in_dim, hidden_dim=hidden_dim, k=k, normalize_decoder=False)
    # Encoder: identity, no bias, no pre_bias.
    sae.pre_bias.data.zero_()
    sae.encoder.weight.data = torch.eye(hidden_dim, in_dim)
    sae.encoder.bias.data.zero_()
    # Decoder: identity, no bias.
    sae.decoder.weight.data = torch.eye(in_dim, hidden_dim)
    sae.decoder.bias.data.zero_()
    sae.eval()

    # Energy head: linear projection that reads only column ``target_idx``
    # of the (denormalized) CLS. With cls_mean=0 and cls_std=1 this means
    # energy is determined entirely by latent feature ``target_idx``.
    head = nn.Linear(in_dim, 1, bias=False)
    head.weight.data = torch.zeros(1, in_dim)
    head.weight.data[0, target_idx] = 1.0
    head.eval()
    return sae, head


def test_audit_feature_record_shape_and_consistency():
    sae, head = _make_id_sae_and_head(target_idx=0)
    cls_orig = torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.7, 0.3, -0.1, 0.5]])
    mean = torch.zeros(4)
    std = torch.ones(4)
    rec = audit_feature(
        sae=sae, energy_head=head,
        cls_original=cls_orig, cls_mean=mean, cls_std=std,
        feature_idx=0, label="test",
    )
    assert isinstance(rec, CausalEffect)
    assert rec.feature_idx == 0
    assert rec.label == "test"
    assert rec.n_specimens == 2
    # Original energy = first column of cls (because head reads col 0).
    assert abs(rec.energy_original_mean - (1.0 + 0.7) / 2.0) < 1.0e-6
    # With identity SAE, recon == original.
    assert abs(rec.energy_recon_mean - rec.energy_original_mean) < 1.0e-6
    # Knock-out feature 0 zeros col 0, so knock-out energy = 0.
    assert abs(rec.energy_knock_out_mean - 0.0) < 1.0e-6


def test_audit_feature_targeted_intervention_dominates():
    """When the head reads only feature target, knock-out on target
    must show a large effect; knock-out on a different feature must
    show essentially zero effect."""
    target = 1
    sae, head = _make_id_sae_and_head(in_dim=4, hidden_dim=4, k=4, target_idx=target)
    torch.manual_seed(0)
    cls_orig = torch.randn(64, 4)              # spread of energy values
    mean = torch.zeros(4)
    std = torch.ones(4)

    rec_on = audit_feature(
        sae=sae, energy_head=head,
        cls_original=cls_orig, cls_mean=mean, cls_std=std,
        feature_idx=target, label="target",
    )
    rec_off = audit_feature(
        sae=sae, energy_head=head,
        cls_original=cls_orig, cls_mean=mean, cls_std=std,
        feature_idx=(target + 1) % 4, label="off-target",
    )

    # The targeted feature must dominate the off-target one by
    # orders of magnitude on the normalized score.
    assert rec_on.knock_out_effect_norm > 0.5
    assert rec_off.knock_out_effect_norm < 1.0e-5


def test_audit_feature_uses_99th_percentile_when_value_unspecified():
    sae, head = _make_id_sae_and_head(target_idx=0)
    cls_orig = torch.tensor([[1.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]])
    mean = torch.zeros(4)
    std = torch.ones(4)
    rec = audit_feature(
        sae=sae, energy_head=head,
        cls_original=cls_orig, cls_mean=mean, cls_std=std,
        feature_idx=0, label="test",
    )
    # Encoded values are 1 and 3 (identity SAE), so the 99th
    # percentile is ~3.0 -- recorded under extra.knock_in_value.
    assert rec.extra["knock_in_value"] >= 2.5


# ---------------------------------------------------------------------------
# filter_features_by_causal_effect
# ---------------------------------------------------------------------------


def _mk(idx: int, ko: float, ki: float, act: float) -> CausalEffect:
    return CausalEffect(
        feature_idx=idx, label=f"f{idx}", n_specimens=10,
        activation_rate=act,
        energy_original_mean=0.0, energy_recon_mean=0.0,
        energy_knock_out_mean=0.0, energy_knock_in_mean=0.0,
        knock_out_effect=0.0, knock_in_effect=0.0,
        knock_out_effect_norm=ko, knock_in_effect_norm=ki,
    )


def test_filter_passes_features_above_threshold():
    effects = [
        _mk(0, ko=0.5, ki=0.0, act=0.5),     # passes (ko)
        _mk(1, ko=0.0, ki=0.6, act=0.5),     # passes (ki)
        _mk(2, ko=0.05, ki=0.05, act=0.5),   # both below threshold
    ]
    out = filter_features_by_causal_effect(effects=effects, min_norm_effect=0.10)
    assert out == [0, 1]


def test_filter_drops_dead_features():
    effects = [
        _mk(0, ko=0.9, ki=0.9, act=0.0001),   # dead even though high-effect
        _mk(1, ko=0.5, ki=0.0, act=0.10),     # alive and effective
    ]
    out = filter_features_by_causal_effect(
        effects=effects, min_norm_effect=0.10,
        require_activation=True, min_activation_rate=0.005,
    )
    assert out == [1]


# ---------------------------------------------------------------------------
# normalize_cls / predict_energy as sanity sanity tests
# ---------------------------------------------------------------------------


def test_normalize_then_denormalize_roundtrips():
    from fmllm.representation.causal import denormalize_cls

    cls = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mean = torch.tensor([0.5, 1.5, 2.5])
    std = torch.tensor([0.5, 0.5, 0.5])
    n = normalize_cls(cls, mean, std)
    d = denormalize_cls(n, mean, std)
    assert torch.allclose(d, cls, atol=1.0e-7)


def test_predict_energy_flattens_trailing_singleton():
    head = nn.Linear(3, 1, bias=False)
    head.weight.data = torch.tensor([[1.0, 0.0, 0.0]])
    cls = torch.tensor([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    e = predict_energy(energy_head=head, cls=cls)
    assert e.shape == (2,)
    assert torch.allclose(e, torch.tensor([2.0, 3.0]), atol=1.0e-6)
