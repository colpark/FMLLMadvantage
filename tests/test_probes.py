"""Tests that every probe runs end-to-end on a tiny model.

We are not asserting that an untrained model satisfies any particular
constraint. The tests confirm the probe interface works (returns a
valid ProbeResult, populates the metric and threshold fields, handles
empty input gracefully) so the runner can never silently swallow an
exception during training.
"""

from __future__ import annotations

import torch

from fmllm.fms._schemas import ProbeResult
from fmllm.fms.fm1_image.model import FM1ImageViT
from fmllm.fms.fm1_image.probes import (
    atom_count_consistency,
    positions_in_box,
    translation_equivariance,
)
from fmllm.fms.fm2_rdf.model import FM2RDFTransformer
from fmllm.fms.fm2_rdf.probes import (
    extensive_scaling,
    non_negativity,
    permutation_invariance,
)
from fmllm.fms.fm3_traj.model import FM3TrajTransformer
from fmllm.fms.fm3_traj.probes import (
    distribution_non_negativity,
    distribution_normalization,
    equipartition,
)


def _device():
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# FM1 probes
# ---------------------------------------------------------------------------


def _build_fm1_tiny():
    return FM1ImageViT(
        image_size=16, patch_size=4, embed_dim=32,
        encoder_depth=1, decoder_depth=1, num_heads=4,
        mlp_ratio=2.0, num_queries=4, max_n_atoms=4,
    )


def _fm1_items(n=4):
    return [
        {"image": torch.randn(1, 16, 16)} for _ in range(n)
    ]


def test_fm1_translation_equivariance_runs():
    model = _build_fm1_tiny()
    result = translation_equivariance.run_probe(
        model=model, items=_fm1_items(),
        device=_device(),
        config={"shift_pixels": 4, "pixel_size_lj": 0.15, "pixel_tolerance": 1.0,
                "n_samples": 4, "threshold": 0.0},
    )
    assert isinstance(result, ProbeResult)
    assert result.constraint_name == "translation_equivariance"
    assert 0.0 <= result.satisfaction_score <= 1.0
    assert result.metric == "frac_matched_within_tolerance"


def test_fm1_atom_count_consistency_runs():
    model = _build_fm1_tiny()
    result = atom_count_consistency.run_probe(
        model=model, items=_fm1_items(), device=_device(),
        config={"threshold": 0.0, "n_samples": 4},
    )
    assert isinstance(result, ProbeResult)
    assert 0.0 <= result.satisfaction_score <= 1.0


def test_fm1_positions_in_box_runs():
    model = _build_fm1_tiny()
    result = positions_in_box.run_probe(
        model=model, items=_fm1_items(), device=_device(),
        config={"threshold": 0.0, "n_samples": 4, "box_half_width_lj": 1.2},
    )
    assert isinstance(result, ProbeResult)
    assert 0.0 <= result.satisfaction_score <= 1.0


def test_fm1_probes_handle_empty_items():
    model = _build_fm1_tiny()
    for module in (translation_equivariance, atom_count_consistency, positions_in_box):
        result = module.run_probe(
            model=model, items=[], device=_device(),
            config={"threshold": 0.5},
        )
        assert isinstance(result, ProbeResult)
        assert result.num_test_cases == 0


# ---------------------------------------------------------------------------
# FM2 probes
# ---------------------------------------------------------------------------


def _build_fm2_tiny():
    return FM2RDFTransformer(
        rdf_bins=20, embed_dim=32, depth=1, num_heads=4, mlp_ratio=2.0,
    )


def _fm2_items(n=8):
    items = []
    for k in range(n):
        n_atoms = 5 if k % 2 == 0 else 7
        atom_mask = torch.zeros(30, dtype=torch.bool)
        atom_mask[:n_atoms] = True
        items.append({
            "rdf": torch.randn(20),
            "atom_count": n_atoms,
            "atom_mask": atom_mask,
        })
    return items


def test_fm2_permutation_invariance_runs():
    model = _build_fm2_tiny()
    result = permutation_invariance.run_probe(
        model=model, items=_fm2_items(), device=_device(),
        config={"threshold": 0.99, "n_samples": 4},
    )
    assert isinstance(result, ProbeResult)
    # Determinism + RDF invariance both hold for tiny models in eval mode.
    assert result.satisfaction_score == 1.0
    assert result.passes_threshold


def test_fm2_extensive_scaling_runs():
    model = _build_fm2_tiny()
    result = extensive_scaling.run_probe(
        model=model, items=_fm2_items(), device=_device(),
        config={"threshold": 0.0, "n_samples": 8, "rel_tolerance": 0.5},
    )
    assert isinstance(result, ProbeResult)
    assert 0.0 <= result.satisfaction_score <= 1.0


def test_fm2_non_negativity_runs():
    model = _build_fm2_tiny()
    result = non_negativity.run_probe(
        model=model, items=_fm2_items(), device=_device(),
        config={"threshold": 0.0, "n_samples": 8, "energy_floor": -1000.0},
    )
    assert isinstance(result, ProbeResult)
    assert result.satisfaction_score == 1.0  # floor very low


# ---------------------------------------------------------------------------
# FM3 probes
# ---------------------------------------------------------------------------


def _build_fm3_tiny():
    return FM3TrajTransformer(
        n_steps_input=10, max_n_atoms=4,
        embed_dim=32, depth=1, num_heads=4, mlp_ratio=2.0,
    )


def _fm3_items(n=4):
    items = []
    for _ in range(n):
        atom_mask = torch.tensor([True, True, True, False])
        items.append({
            "traj_positions": torch.randn(11, 4, 2),
            "traj_velocities": torch.randn(11, 4, 2),
            "atom_mask": atom_mask,
        })
    return items


def test_fm3_equipartition_runs():
    model = _build_fm3_tiny()
    result = equipartition.run_probe(
        model=model, items=_fm3_items(), device=_device(),
        config={"threshold": 0.0, "n_samples": 4, "rel_tolerance": 1.0},
    )
    assert isinstance(result, ProbeResult)
    assert 0.0 <= result.satisfaction_score <= 1.0


def test_fm3_distribution_normalization_runs():
    model = _build_fm3_tiny()
    result = distribution_normalization.run_probe(
        model=model, items=_fm3_items(), device=_device(),
        config={"threshold": 0.5, "n_samples": 4, "grid_points": 1024,
                "rel_tolerance": 0.20},
    )
    assert isinstance(result, ProbeResult)


def test_fm3_distribution_non_negativity_runs():
    model = _build_fm3_tiny()
    result = distribution_non_negativity.run_probe(
        model=model, items=_fm3_items(), device=_device(),
        config={"threshold": 1.0, "n_samples": 4},
    )
    assert isinstance(result, ProbeResult)
    # softplus + epsilon guarantees positivity.
    assert result.satisfaction_score == 1.0
    assert result.passes_threshold


# ---------------------------------------------------------------------------
# Probe runner end-to-end
# ---------------------------------------------------------------------------


def test_probe_runner_collects_all_fm1_probes(tmp_path):
    """The runner imports every metadata-declared probe and produces a report."""
    from fmllm.fms.probe_runner import run_all_probes
    from fmllm.fms._schemas import save_probe_report, load_probe_report

    model = _build_fm1_tiny()
    report = run_all_probes(
        "fm1_image", model=model, items=_fm1_items(8),
        device=_device(),
        config_overrides={
            "translation_equivariance": {"threshold": 0.0, "n_samples": 4},
            "atom_count_consistency": {"threshold": 0.0, "n_samples": 4},
            "positions_in_box": {"threshold": 0.0, "n_samples": 4,
                                 "box_half_width_lj": 1.5},
        },
    )
    assert report.fm_name == "fm1_image"
    assert len(report.results) == 3
    names = {r.constraint_name for r in report.results}
    assert names == {
        "translation_equivariance", "atom_count_consistency", "positions_in_box",
    }

    out = tmp_path / "probe_report.yaml"
    save_probe_report(report, out)
    rehydrated = load_probe_report(out)
    assert rehydrated.model_dump() == report.model_dump()
