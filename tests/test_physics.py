"""Tests for the physics module.

The four required tests from the Phase 1 spec live here:

    - Energy conservation in pure-LJ MD over 1000 steps, drift below 1e-3.
    - Pair distance histogram normalization sums to ``N(N-1)``.
    - Permutation invariance of g(r).
    - Rasterizer produces images that recover atom positions within
      sub-pixel error.

The file also exercises smaller invariants (LJ minimum, force signs,
structure-generator counts, MB velocity temperature target).

None of these tests require a GPU.
"""

from __future__ import annotations

import math

import pytest
import torch

from fmllm.physics import (
    R_MIN,
    equilibrium_positions,
    kinetic_energies_per_atom,
    lj_pair_energy_and_forces,
    maxwell_boltzmann_velocities,
    pair_distance_histogram,
    pairwise_distances,
    radial_distribution_function,
    rasterize_positions,
    regular_polygon,
    run_md,
    temperature_from_velocities,
    triangular_lattice_disk,
)


# ---------------------------------------------------------------------------
# LJ potential properties
# ---------------------------------------------------------------------------


def test_lj_minimum_at_two_to_one_sixth():
    """U_LJ has its minimum at r = 2**(1/6) with depth -1."""
    positions = torch.tensor([[0.0, 0.0], [R_MIN, 0.0]])
    energy, forces = lj_pair_energy_and_forces(positions)
    assert torch.allclose(energy, torch.tensor(-1.0), atol=1e-5)
    assert torch.allclose(forces, torch.zeros_like(forces), atol=1e-5)


def test_lj_repulsive_short_range():
    """At r < r_min, atoms repel each other."""
    positions = torch.tensor([[0.0, 0.0], [0.9, 0.0]])
    _, forces = lj_pair_energy_and_forces(positions)
    assert forces[0, 0] < 0  # atom 0 pushed in -x
    assert forces[1, 0] > 0  # atom 1 pushed in +x


def test_lj_attractive_long_range():
    """At r > r_min, atoms attract each other."""
    positions = torch.tensor([[0.0, 0.0], [1.5, 0.0]])
    _, forces = lj_pair_energy_and_forces(positions)
    assert forces[0, 0] > 0
    assert forces[1, 0] < 0


def test_lj_batched_broadcast():
    """The LJ function broadcasts over leading batch dimensions."""
    pos = torch.randn(4, 7, 2)
    energy, forces = lj_pair_energy_and_forces(pos)
    assert energy.shape == (4,)
    assert forces.shape == (4, 7, 2)


# ---------------------------------------------------------------------------
# Energy conservation
# ---------------------------------------------------------------------------


def test_energy_conservation_pure_lj_1000_steps():
    """Pure LJ MD conserves total energy over 1000 steps below 1e-3."""
    positions = equilibrium_positions(7, motif="triangular_disk")
    velocities = maxwell_boltzmann_velocities(
        n_atoms=7,
        temperature=0.05,
        dim=2,
        generator=torch.Generator(device="cpu").manual_seed(0),
    )

    forces_fn = lj_pair_energy_and_forces

    def total_energy(pos, vel):
        pe, _ = forces_fn(pos)
        ke = 0.5 * (vel * vel).sum()
        return pe + ke

    e0 = total_energy(positions, velocities)
    traj = run_md(
        positions, velocities, forces_fn,
        dt=0.002, n_steps=1000, record_every=200,
    )
    pos_traj = traj["positions"]
    vel_traj = traj["velocities"]
    drifts = []
    for i in range(pos_traj.shape[0]):
        e = total_energy(pos_traj[i], vel_traj[i])
        drifts.append(abs((e - e0) / e0).item())
    assert max(drifts) < 1.0e-3, f"max drift {max(drifts)} exceeds 1e-3"


# ---------------------------------------------------------------------------
# RDF / pair-distance histogram
# ---------------------------------------------------------------------------


def test_pair_histogram_sum_equals_n_squared_minus_n():
    """Sum over bins equals N(N-1) when r_max covers the cluster diameter."""
    n_atoms = 13
    positions = equilibrium_positions(n_atoms, motif="triangular_disk")
    diameter = pairwise_distances(positions).max().item()
    r_max = diameter * 1.5 + 1.0
    hist, _ = pair_distance_histogram(positions, r_max=r_max, num_bins=200)
    assert int(hist.sum().item()) == n_atoms * (n_atoms - 1)


def test_rdf_permutation_invariance():
    """g(r) is invariant under random permutation of atoms."""
    torch.manual_seed(0)
    n_atoms = 13
    positions = equilibrium_positions(n_atoms, motif="triangular_disk")
    perm = torch.randperm(n_atoms)
    pos_permuted = positions[perm]
    g1, _ = radial_distribution_function(positions, r_max=5.0, num_bins=200)
    g2, _ = radial_distribution_function(pos_permuted, r_max=5.0, num_bins=200)
    assert torch.equal(g1, g2)


def test_pair_histogram_drops_distances_above_rmax():
    """Distances at or above r_max land outside the histogram."""
    positions = torch.tensor([[0.0, 0.0], [3.0, 0.0]])
    hist, _ = pair_distance_histogram(positions, r_max=2.0, num_bins=10)
    assert int(hist.sum().item()) == 0


# ---------------------------------------------------------------------------
# Rasterizer
# ---------------------------------------------------------------------------


def test_rasterizer_single_atom_at_origin():
    """A single atom at origin lights up the center pixel."""
    image = rasterize_positions(
        torch.tensor([[0.0, 0.0]]),
        image_size=64,
        pixel_size_lj=0.1,
        blur_radius_lj=0.15,
        noise_std=0.0,
    )
    max_idx = int(image.argmax().item())
    row, col = divmod(max_idx, 64)
    assert abs(row - 32) <= 1
    assert abs(col - 32) <= 1


def test_rasterizer_offset_atom_sub_pixel_recovery():
    """A single atom at (1.0, -0.5) shows up at the expected pixel."""
    pixel_size = 0.1
    image_size = 64
    positions = torch.tensor([[1.0, -0.5]])
    image = rasterize_positions(
        positions,
        image_size=image_size,
        pixel_size_lj=pixel_size,
        blur_radius_lj=0.15,
        noise_std=0.0,
    )
    expected_col = int(round(1.0 / pixel_size + image_size / 2))  # 42
    expected_row = int(round(-(-0.5) / pixel_size + image_size / 2))  # 37
    max_idx = int(image.argmax().item())
    row, col = divmod(max_idx, image_size)
    assert abs(row - expected_row) <= 1
    assert abs(col - expected_col) <= 1


def test_rasterizer_permutation_invariance():
    """The image is invariant under permutation of input positions."""
    torch.manual_seed(0)
    pos = torch.randn(10, 2) * 0.5
    img1 = rasterize_positions(
        pos, image_size=32, pixel_size_lj=0.1, blur_radius_lj=0.15, noise_std=0.0,
    )
    img2 = rasterize_positions(
        pos[torch.randperm(10)], image_size=32, pixel_size_lj=0.1, blur_radius_lj=0.15,
        noise_std=0.0,
    )
    assert torch.allclose(img1, img2, atol=1e-6)


def test_rasterizer_blob_centroid_within_subpixel():
    """For each atom, the local intensity peak's center of mass recovers
    its position to better than one pixel.
    """
    pixel_size = 0.1
    image_size = 64
    blur = 0.1
    positions = torch.tensor([
        [0.5, 0.3],
        [-1.2, 0.7],
        [0.0, -1.5],
    ])
    image = rasterize_positions(
        positions,
        image_size=image_size,
        pixel_size_lj=pixel_size,
        blur_radius_lj=blur,
        noise_std=0.0,
    )

    # For each atom, compute the centroid of the image's intensity in a
    # small window around the projected pixel and confirm the recovered
    # LJ position is within one pixel.
    rows = torch.arange(image_size, dtype=torch.float32)
    cols = torch.arange(image_size, dtype=torch.float32)
    rr, cc = torch.meshgrid(rows, cols, indexing="ij")

    for atom in positions:
        col_target = float(atom[0]) / pixel_size + image_size / 2.0
        row_target = -float(atom[1]) / pixel_size + image_size / 2.0
        # 5-pixel window around the target
        r0 = max(0, int(row_target) - 4)
        r1 = min(image_size, int(row_target) + 5)
        c0 = max(0, int(col_target) - 4)
        c1 = min(image_size, int(col_target) + 5)
        patch = image[r0:r1, c0:c1]
        rr_patch = rr[r0:r1, c0:c1]
        cc_patch = cc[r0:r1, c0:c1]
        weight = patch.sum()
        cy = (rr_patch * patch).sum() / weight
        cx = (cc_patch * patch).sum() / weight
        recovered_x = (cx - image_size / 2.0) * pixel_size
        recovered_y = (image_size / 2.0 - cy) * pixel_size
        err = math.hypot(
            float(recovered_x) - float(atom[0]),
            float(recovered_y) - float(atom[1]),
        )
        assert err < pixel_size, (
            f"atom {atom.tolist()} recovered as ({recovered_x:.3f},"
            f" {recovered_y:.3f}), error {err:.4f} >= pixel_size {pixel_size}"
        )


# ---------------------------------------------------------------------------
# Velocities and structures
# ---------------------------------------------------------------------------


def test_maxwell_boltzmann_target_temperature():
    """The MB sampler with rescaling produces the requested temperature."""
    target = 0.7
    velocities = maxwell_boltzmann_velocities(
        n_atoms=20, temperature=target, dim=2,
        generator=torch.Generator(device="cpu").manual_seed(0),
        rescale_to_target=True,
    )
    t = float(temperature_from_velocities(velocities))
    assert math.isclose(t, target, rel_tol=1e-5)


def test_maxwell_boltzmann_zero_com():
    """Removing COM drift gives near-zero net momentum."""
    velocities = maxwell_boltzmann_velocities(
        n_atoms=15, temperature=1.0, dim=2,
        generator=torch.Generator(device="cpu").manual_seed(0),
        remove_com=True,
    )
    com_v = velocities.mean(dim=0)
    assert com_v.abs().max() < 1e-5


def test_kinetic_energies_per_atom_shape():
    """``kinetic_energies_per_atom`` returns one value per atom per frame."""
    velocities = torch.randn(5, 7, 2)
    ke = kinetic_energies_per_atom(velocities)
    assert ke.shape == (5, 7)


@pytest.mark.parametrize("n_atoms", [5, 7, 9, 11, 13, 17, 19, 21, 25, 30])
def test_triangular_lattice_disk_count(n_atoms):
    """The triangular-disk generator returns exactly ``n_atoms`` positions."""
    pos = triangular_lattice_disk(n_atoms)
    assert pos.shape == (n_atoms, 2)
    # No duplicate positions.
    distances = pairwise_distances(pos)
    eye = torch.eye(n_atoms, dtype=torch.bool)
    off_diag = distances[~eye]
    assert off_diag.min() > 0.5  # nearest neighbors at ~r_min ~= 1.122


def test_regular_polygon_nearest_neighbor_spacing():
    """A regular polygon with spacing ``r_min`` has the requested spacing."""
    pos = regular_polygon(7, spacing=R_MIN)
    distances = pairwise_distances(pos)
    eye = torch.eye(7, dtype=torch.bool)
    nearest = distances.masked_fill(eye, float("inf")).min(dim=-1).values
    assert torch.allclose(nearest, torch.full_like(nearest, R_MIN), atol=1e-5)
