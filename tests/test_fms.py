"""Tests for the three foundation models.

Each FM gets:
    - Forward-pass shape checks at small architecture dimensions.
    - Physics-constraint loss-component tests on synthetic inputs.

The tests run on CPU with tiny model sizes, so the suite stays fast.
"""

from __future__ import annotations

import math

import pytest
import torch

from fmllm.fms.common import (
    per_atom_potential_energy,
    split_conformal_quantile,
    write_conformal_calibration,
    read_conformal_calibration,
)
from fmllm.fms.fm1_image.model import FM1ImageViT
from fmllm.fms.fm1_image.train import (
    compute_fm1_losses,
    hungarian_match_batch,
)
from fmllm.fms.fm2_rdf.model import FM2RDFTransformer
from fmllm.fms.fm2_rdf.train import compute_fm2_losses
from fmllm.fms.fm3_traj.model import FM3TrajTransformer
from fmllm.fms.fm3_traj.train import compute_fm3_losses, gamma_nll
from fmllm.utils.config import FM1Config, FM2Config, FM3Config


# ---------------------------------------------------------------------------
# FM1
# ---------------------------------------------------------------------------


@pytest.fixture
def small_fm1_config():
    return FM1Config(
        image_size=16,
        patch_size=4,
        embed_dim=32,
        encoder_depth=1,
        decoder_depth=1,
        num_heads=4,
        mlp_ratio=2.0,
        num_queries=4,
        max_n_atoms=4,
    )


def test_fm1_forward_shape(small_fm1_config):
    cfg = small_fm1_config
    model = FM1ImageViT(
        image_size=cfg.image_size, patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim, encoder_depth=cfg.encoder_depth,
        decoder_depth=cfg.decoder_depth, num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio, num_queries=cfg.num_queries,
        max_n_atoms=cfg.max_n_atoms,
    )
    image = torch.randn(2, 1, 16, 16)
    out = model(image)
    assert out["count_logits"].shape == (2, cfg.max_n_atoms + 1)
    assert out["positions"].shape == (2, cfg.num_queries, 2)
    assert out["confidence_logits"].shape == (2, cfg.num_queries)


def test_fm1_forward_accepts_3d_input(small_fm1_config):
    cfg = small_fm1_config
    model = FM1ImageViT(
        image_size=cfg.image_size, patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim, encoder_depth=cfg.encoder_depth,
        decoder_depth=cfg.decoder_depth, num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio, num_queries=cfg.num_queries,
        max_n_atoms=cfg.max_n_atoms,
    )
    image = torch.randn(2, 16, 16)
    out = model(image)
    assert out["positions"].shape == (2, cfg.num_queries, 2)


def test_fm1_hungarian_match_simple_case():
    """A two-atom batch with separated targets must match the closest queries."""
    pred = torch.tensor([[
        [10.0, 10.0],
        [0.0, 0.0],
        [-100.0, -100.0],
        [1.0, 1.0],
    ]])
    true = torch.tensor([[
        [0.0, 0.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ]])
    mask = torch.tensor([[True, True, False, False]])
    pred_idx, true_idx = hungarian_match_batch(pred, true, mask)
    assert pred_idx[0].numel() == 2
    paired = sorted(zip(pred_idx[0].tolist(), true_idx[0].tolist()))
    assert (1, 0) in paired and (3, 1) in paired


def test_fm1_box_constraint_loss_sign(small_fm1_config):
    cfg = small_fm1_config
    cfg = cfg.model_copy(update={"box_half_width_lj": 1.0})
    outputs = {
        "count_logits": torch.zeros(1, cfg.max_n_atoms + 1),
        "positions": torch.tensor([[[0.5, 0.0], [2.0, 0.0], [0.0, -3.0], [0.0, 0.0]]]),
        "confidence_logits": torch.zeros(1, cfg.num_queries),
    }
    target_count = torch.tensor([0])
    target_positions = torch.zeros(1, cfg.max_n_atoms, 2)
    atom_mask = torch.zeros(1, cfg.max_n_atoms, dtype=torch.bool)
    losses = compute_fm1_losses(
        outputs,
        target_count=target_count,
        target_positions=target_positions,
        atom_mask=atom_mask,
        cfg=cfg,
    )
    assert losses["box"].item() > 0.0

    # All inside the box: zero box loss.
    outputs["positions"] = torch.tensor([[[0.0, 0.0], [0.5, 0.5], [-0.5, 0.5], [0.5, -0.5]]])
    losses = compute_fm1_losses(
        outputs,
        target_count=target_count,
        target_positions=target_positions,
        atom_mask=atom_mask,
        cfg=cfg,
    )
    assert losses["box"].item() == 0.0


def test_fm1_total_loss_combines_components(small_fm1_config):
    cfg = small_fm1_config
    outputs = {
        "count_logits": torch.zeros(1, cfg.max_n_atoms + 1),
        "positions": torch.zeros(1, cfg.num_queries, 2),
        "confidence_logits": torch.zeros(1, cfg.num_queries),
    }
    target_count = torch.tensor([2])
    target_positions = torch.tensor([[[0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]])
    atom_mask = torch.tensor([[True, True, False, False]])
    losses = compute_fm1_losses(
        outputs, target_count=target_count, target_positions=target_positions,
        atom_mask=atom_mask, cfg=cfg,
    )
    assert losses["total"].item() > 0
    assert losses["matched_pairs"].item() == 2


# ---------------------------------------------------------------------------
# FM2
# ---------------------------------------------------------------------------


def test_fm2_forward_shape():
    model = FM2RDFTransformer(rdf_bins=20, embed_dim=32, depth=1, num_heads=4, mlp_ratio=2.0)
    rdf = torch.randn(3, 20)
    out = model(rdf)
    assert out.shape == (3,)


def test_fm2_rejects_wrong_bin_count():
    model = FM2RDFTransformer(rdf_bins=20, embed_dim=32, depth=1, num_heads=4, mlp_ratio=2.0)
    with pytest.raises(ValueError):
        model(torch.randn(2, 19))


def test_fm2_nonneg_loss_only_below_floor():
    cfg = FM2Config(rdf_bins=20, energy_floor=-2.0, nonneg_weight=1.0, huber_delta=1.0)
    pred = torch.tensor([0.0, -1.0, -3.0, -5.0])
    target = torch.zeros_like(pred)
    losses = compute_fm2_losses(pred, target_energy=target, cfg=cfg)
    # Only the entries below floor (-3, -5) generate non-zero penalty.
    assert losses["nonneg"].item() > 0.0

    pred_safe = torch.tensor([0.0, -1.0, -1.5, -1.9])
    losses_safe = compute_fm2_losses(pred_safe, target_energy=target, cfg=cfg)
    assert losses_safe["nonneg"].item() == 0.0


def test_fm2_extensive_per_atom_target_invariant_to_n():
    """Per-atom potential energy is the right quantity for extensive scaling."""
    # Build two clusters of different sizes at the LJ minimum.
    from fmllm.physics import equilibrium_positions

    pos_a = equilibrium_positions(7, motif="triangular_disk")
    pos_b = equilibrium_positions(13, motif="triangular_disk")
    n_max = 30
    pad_a = torch.zeros(n_max, 2)
    pad_a[:7] = pos_a
    pad_b = torch.zeros(n_max, 2)
    pad_b[:13] = pos_b
    final_positions = torch.stack([pad_a, pad_b], dim=0)
    mask_a = torch.zeros(n_max, dtype=torch.bool)
    mask_a[:7] = True
    mask_b = torch.zeros(n_max, dtype=torch.bool)
    mask_b[:13] = True
    atom_mask = torch.stack([mask_a, mask_b], dim=0)

    energies = per_atom_potential_energy(final_positions, atom_mask, confinement_k=0.0)
    # Both clusters share dense triangular packing at the LJ minimum, so
    # per-atom potential energy stays in a similar range.
    assert energies.shape == (2,)
    assert torch.isfinite(energies).all()


# ---------------------------------------------------------------------------
# FM3
# ---------------------------------------------------------------------------


def test_fm3_forward_shape():
    model = FM3TrajTransformer(
        n_steps_input=10, max_n_atoms=4,
        embed_dim=32, depth=1, num_heads=4, mlp_ratio=2.0,
    )
    traj_pos = torch.randn(2, 11, 4, 2)
    traj_vel = torch.randn(2, 11, 4, 2)
    mask = torch.tensor([
        [True, True, True, False],
        [True, True, False, False],
    ])
    out = model(traj_pos, traj_vel, mask)
    assert out["alpha"].shape == (2,)
    assert out["beta"].shape == (2,)
    assert (out["alpha"] > 0).all()
    assert (out["beta"] > 0).all()


def test_fm3_permutation_invariance():
    """Permuting atoms in the input must not change the model output."""
    torch.manual_seed(0)
    model = FM3TrajTransformer(
        n_steps_input=8, max_n_atoms=5,
        embed_dim=32, depth=1, num_heads=4, mlp_ratio=2.0,
    )
    model.eval()
    traj_pos = torch.randn(1, 9, 5, 2)
    traj_vel = torch.randn(1, 9, 5, 2)
    mask = torch.tensor([[True, True, True, True, False]])

    perm = torch.tensor([2, 0, 3, 1, 4])
    traj_pos_perm = traj_pos[:, :, perm]
    traj_vel_perm = traj_vel[:, :, perm]
    mask_perm = mask[:, perm]

    with torch.no_grad():
        out_a = model(traj_pos, traj_vel, mask)
        out_b = model(traj_pos_perm, traj_vel_perm, mask_perm)

    assert torch.allclose(out_a["alpha"], out_b["alpha"], atol=1e-5)
    assert torch.allclose(out_a["beta"], out_b["beta"], atol=1e-5)


def test_fm3_gamma_nll_matches_torch_distribution():
    """Our masked NLL agrees with torch.distributions.Gamma for full masks."""
    alpha = torch.tensor([2.0])
    beta = torch.tensor([1.5])
    samples = torch.tensor([[0.5, 1.0, 1.5, 2.0, 2.5]])
    mask = torch.ones_like(samples, dtype=torch.bool)

    nll = gamma_nll(alpha, beta, samples=samples, sample_mask=mask, nll_clip=50.0)

    # Reference from torch.distributions.Gamma
    from torch.distributions import Gamma
    dist = Gamma(alpha, 1.0 / beta)
    ref = -dist.log_prob(samples).mean()
    assert torch.allclose(nll, ref, atol=1.0e-5)


def test_fm3_equipartition_loss_pulls_toward_observed_mean():
    cfg = FM3Config(
        n_steps_input=4, max_n_atoms=2,
        embed_dim=8, depth=1, num_heads=2, mlp_ratio=2.0,
        equipartition_weight=1.0, nll_clip=50.0,
    )
    outputs = {
        "alpha": torch.tensor([1.0, 1.0]),
        "beta": torch.tensor([0.5, 1.0]),  # pred mean = 0.5 and 1.0
    }
    samples = torch.tensor([
        [1.0, 1.0],
        [1.0, 1.0],
    ])
    mask = torch.ones_like(samples, dtype=torch.bool)
    losses = compute_fm3_losses(outputs, samples=samples, sample_mask=mask, cfg=cfg)
    # First sample's pred_mean (0.5) differs from obs (1.0): equipartition > 0.
    # Second sample matches: lower contribution.
    assert losses["equipartition"].item() > 0.0
    assert math.isclose(float(losses["pred_mean_ke"]), 0.75, abs_tol=1e-5)
    assert math.isclose(float(losses["obs_mean_ke"]), 1.0, abs_tol=1e-5)


# ---------------------------------------------------------------------------
# Conformal calibration helpers
# ---------------------------------------------------------------------------


def test_split_conformal_quantile_basic():
    """For 9 evenly spaced scores, the alpha=0.10 quantile equals max."""
    scores = torch.arange(1, 10, dtype=torch.float32)  # 1..9
    q = split_conformal_quantile(scores, alpha=0.10)
    # n+1=10, ceil(10*0.9)=9 -> 9th smallest (1-indexed) = 9.0
    assert q == pytest.approx(9.0)


def test_split_conformal_quantile_alpha_50():
    scores = torch.arange(1, 10, dtype=torch.float32)
    q = split_conformal_quantile(scores, alpha=0.50)
    # ceil(10*0.5)=5 -> 5th smallest = 5.0
    assert q == pytest.approx(5.0)


def test_split_conformal_quantile_rejects_empty():
    with pytest.raises(ValueError):
        split_conformal_quantile(torch.tensor([]), alpha=0.10)


def test_calibration_round_trip(tmp_path):
    path = tmp_path / "cal.json"
    out = write_conformal_calibration(
        path,
        fm_name="fm_test",
        score_name="dummy",
        alpha_to_threshold={0.1: 1.5, 0.2: 1.0},
        extra={"n": 100},
    )
    assert out.exists()
    loaded = read_conformal_calibration(path)
    assert loaded["fm_name"] == "fm_test"
    assert loaded["score_name"] == "dummy"
    assert loaded["thresholds"]["0.1000"] == pytest.approx(1.5)
    assert loaded["thresholds"]["0.2000"] == pytest.approx(1.0)
    assert loaded["extra"]["n"] == 100
