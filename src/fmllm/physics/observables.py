"""Physical observables: pair distances, RDF, kinetic energies, rasterizer.

This module computes the per-specimen observables the dataset records
and the foundation models consume.

Pair distance histogram and radial distribution function (RDF):
    The pair distance histogram counts ordered pairs ``(i, j)`` with
    ``i != j`` whose distance falls in each bin. The total over all
    bins (with ``r_max`` exceeding the cluster diameter) equals
    ``N * (N - 1)``. The RDF normalizes that histogram by the area of
    each bin annulus and by the cluster's number density, producing a
    unitless function that approaches 1 at large ``r`` for an ideal
    homogeneous medium.

Kinetic energy distribution:
    ``kinetic_energies_per_atom`` returns one kinetic energy per atom
    per recorded frame. ``temperature_from_velocities`` extracts the
    instantaneous temperature.

Rasterizer:
    ``rasterize_positions`` renders 2D atom positions to a grayscale
    image by placing a Gaussian blob at each atom and adding optional
    Gaussian noise. The image domain is centered on the origin and
    runs from ``-image_size * pixel_size / 2`` to
    ``+image_size * pixel_size / 2`` along each axis. Image row indices
    grow downward, so a positive ``y`` LJ coordinate maps to a small
    row index.

Produces:
    Tensors with shapes documented per function.

Depends on:
    torch.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def pairwise_distances(positions: Tensor) -> Tensor:
    """Return all-pairs Euclidean distances.

    Args:
        positions: Tensor of shape ``(..., N, D)``.

    Returns:
        Tensor of shape ``(..., N, N)`` with zeros on the diagonal.
    """
    diff = positions.unsqueeze(-2) - positions.unsqueeze(-3)
    return diff.norm(dim=-1)


def pair_distance_histogram(
    positions: Tensor,
    *,
    r_max: float,
    num_bins: int = 200,
) -> tuple[Tensor, Tensor]:
    """Histogram of ordered-pair distances ``(i, j)`` with ``i != j``.

    The histogram counts each unordered pair twice. With ``r_max``
    larger than the cluster diameter, ``hist.sum() == N * (N - 1)``.

    Args:
        positions: Tensor of shape ``(N, D)``.
        r_max: Upper edge of the histogram domain. The lower edge is
            zero. Pair distances that fall outside ``[0, r_max]`` get
            dropped.
        num_bins: Number of equal-width bins.

    Returns:
        ``(hist, edges)`` where ``hist`` has shape ``(num_bins,)`` and
        ``edges`` has shape ``(num_bins + 1,)``.
    """
    if positions.dim() != 2:
        raise ValueError(
            f"pair_distance_histogram expects (N, D) positions, got shape {tuple(positions.shape)}"
        )
    if r_max <= 0:
        raise ValueError(f"r_max must be positive, got {r_max}")
    if num_bins < 1:
        raise ValueError(f"num_bins must be >= 1, got {num_bins}")

    n_atoms = positions.shape[0]
    distances = pairwise_distances(positions)
    eye = torch.eye(n_atoms, dtype=torch.bool, device=positions.device)
    pair_distances = distances[~eye]
    pair_distances = pair_distances[pair_distances < r_max]

    edges = torch.linspace(0.0, r_max, num_bins + 1, device=positions.device)
    hist = torch.histc(pair_distances, bins=num_bins, min=0.0, max=r_max)
    return hist, edges


def radial_distribution_function(
    positions: Tensor,
    *,
    r_max: float,
    num_bins: int = 200,
    cell_area: float | None = None,
) -> tuple[Tensor, Tensor]:
    """Compute the 2D radial distribution function ``g(r)``.

    The function divides the pair distance histogram by the area of
    each annulus ``2 * pi * r_center * dr``, then normalizes by the
    number density ``N / cell_area`` and by ``N`` so ``g(r) -> 1`` at
    large ``r`` for an ideal homogeneous medium.

    For finite clusters there is no natural simulation cell. The
    default ``cell_area`` uses ``pi * r_max ** 2``, which gives a
    well-defined, dimensionless quantity that the FM consumes
    consistently across specimens.

    Args:
        positions: Tensor of shape ``(N, D)``.
        r_max: Upper edge of the histogram domain.
        num_bins: Number of equal-width bins.
        cell_area: Optional override for the normalizing area.

    Returns:
        ``(g, edges)`` where ``g`` has shape ``(num_bins,)`` and
        ``edges`` has shape ``(num_bins + 1,)``.
    """
    hist, edges = pair_distance_histogram(positions, r_max=r_max, num_bins=num_bins)
    n_atoms = positions.shape[0]
    if n_atoms < 2:
        return torch.zeros_like(hist, dtype=torch.float32), edges

    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    dr = edges[1] - edges[0]
    area = cell_area if cell_area is not None else math.pi * r_max * r_max
    rho = n_atoms / area
    annulus_area = 2.0 * math.pi * bin_centers * dr
    # Per-atom average pair count in each bin: hist / N. Divide by the
    # ideal-gas expected count rho * annulus_area.
    g = hist.float() / max(n_atoms, 1)
    g = g / (rho * annulus_area).clamp_min(1e-12)
    return g, edges


def kinetic_energies_per_atom(
    velocities: Tensor,
    *,
    mass: float = 1.0,
) -> Tensor:
    """Return per-atom kinetic energies.

    Args:
        velocities: Tensor of shape ``(..., N, D)``.
        mass: Per-atom mass. Defaults to 1 in reduced units.

    Returns:
        Tensor of shape ``(..., N)``.
    """
    return 0.5 * mass * (velocities * velocities).sum(dim=-1)


def temperature_from_velocities(
    velocities: Tensor,
    *,
    mass: float = 1.0,
    dim: int | None = None,
    com_correction: bool = True,
) -> Tensor:
    """Return the instantaneous temperature implied by the velocities.

    Equipartition gives ``KE = 0.5 * dof * T``. The function reports
    ``T = 2 * KE / dof`` where ``dof = D * (N - int(com_correction))``.

    Args:
        velocities: Tensor of shape ``(..., N, D)``.
        mass: Per-atom mass.
        dim: Override for ``D``. Defaults to the last dim of ``velocities``.
        com_correction: When True, subtract one ``D``-degree-of-freedom
            block to account for zero net momentum. Use False for
            general velocity samples that have not been COM-corrected.

    Returns:
        Tensor of shape ``(...,)``.
    """
    n_atoms = velocities.shape[-2]
    spatial_dim = dim if dim is not None else velocities.shape[-1]
    dof = spatial_dim * (n_atoms - (1 if com_correction else 0))
    if dof <= 0:
        raise ValueError(f"non-positive dof {dof} for shape {tuple(velocities.shape)}")
    ke = 0.5 * mass * (velocities * velocities).sum(dim=(-1, -2))
    return 2.0 * ke / dof


def rasterize_positions(
    positions: Tensor,
    *,
    image_size: int,
    pixel_size_lj: float,
    blur_radius_lj: float,
    noise_std: float = 0.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Render 2D atom positions to a grayscale image with Gaussian blobs.

    Each atom contributes a 2D Gaussian centered at its projected pixel
    coordinates. The Gaussian normalizes to integrate to 1 in the
    continuous limit, which gives a continuum-consistent intensity that
    approximates ``N`` total before noise.

    Coordinate convention:
        ``image[row, col]`` corresponds to LJ position
        ``(x = (col - W/2) * pixel_size_lj, y = (H/2 - row) * pixel_size_lj)``.
        That is, increasing ``row`` moves down in the image and matches
        decreasing ``y`` in LJ space, the standard image-display
        convention. Atoms outside the imaging box still contribute to
        pixels through the Gaussian tail, which is intentional.

    Args:
        positions: Tensor of shape ``(N, 2)``.
        image_size: Side length of the square image in pixels.
        pixel_size_lj: Width of one pixel in LJ units.
        blur_radius_lj: Standard deviation of the Gaussian blob in LJ
            units. The pixel-space sigma is ``blur_radius_lj / pixel_size_lj``.
        noise_std: Optional standard deviation of additive Gaussian
            white noise. ``0`` disables noise.
        generator: Optional torch generator for the noise sample.

    Returns:
        Tensor of shape ``(image_size, image_size)``.
    """
    if positions.dim() != 2 or positions.shape[-1] != 2:
        raise ValueError(
            f"rasterize_positions expects (N, 2) positions, got shape {tuple(positions.shape)}"
        )
    if image_size < 1:
        raise ValueError(f"image_size must be >= 1, got {image_size}")
    if pixel_size_lj <= 0:
        raise ValueError(f"pixel_size_lj must be positive, got {pixel_size_lj}")
    if blur_radius_lj <= 0:
        raise ValueError(f"blur_radius_lj must be positive, got {blur_radius_lj}")
    if noise_std < 0:
        raise ValueError(f"noise_std must be non-negative, got {noise_std}")

    h = w = image_size
    device = positions.device
    dtype = positions.dtype if positions.is_floating_point() else torch.float32
    positions = positions.to(dtype)

    sigma_pixels = blur_radius_lj / pixel_size_lj

    # Atom positions in pixel coordinates.
    cols = positions[:, 0] / pixel_size_lj + w / 2.0
    rows = -positions[:, 1] / pixel_size_lj + h / 2.0

    rr, cc = torch.meshgrid(
        torch.arange(h, dtype=dtype, device=device),
        torch.arange(w, dtype=dtype, device=device),
        indexing="ij",
    )
    # Distances from each pixel center to each atom in pixel space.
    dr = rr.unsqueeze(0) - rows.view(-1, 1, 1)  # (N, H, W)
    dc = cc.unsqueeze(0) - cols.view(-1, 1, 1)  # (N, H, W)
    d2 = dr * dr + dc * dc

    norm = 1.0 / (2.0 * math.pi * sigma_pixels * sigma_pixels)
    blobs = norm * torch.exp(-0.5 * d2 / (sigma_pixels * sigma_pixels))
    image = blobs.sum(dim=0)

    if noise_std > 0:
        noise = torch.randn(
            (h, w), generator=generator, device=device, dtype=dtype,
        ) * noise_std
        image = image + noise

    return image


__all__ = [
    "kinetic_energies_per_atom",
    "pair_distance_histogram",
    "pairwise_distances",
    "radial_distribution_function",
    "rasterize_positions",
    "temperature_from_velocities",
]
