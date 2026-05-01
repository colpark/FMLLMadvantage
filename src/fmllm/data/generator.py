"""CLI script that produces the synthetic Lennard-Jones dataset.

The generator samples a population of specimens, draws ``(N, T, motif)``
per specimen from deterministic per-ID seeds, runs MD equilibration on
batches grouped by ``N``, then records observables and writes a single
HDF5 file plus a manifest YAML and a splits YAML to the output
directory.

Usage:
    uv run python -m fmllm.data.generator \\
        --config configs/default.yaml \\
        --out data/synthetic_lj_v1

Runtime expectation:
    Roughly 30 minutes on a single H100 for 50,000 specimens at default
    settings. The generator batches specimens by ``N`` so the GPU sees
    one large MD batch per ``N`` group at a time.

Produces:
    ``<out>/specimens.h5`` with the full dataset.
    ``<out>/manifest.yaml`` with the generation parameters.
    ``<out>/splits.yaml`` with held-out partitioning.

Depends on:
    torch, numpy, h5py, typer, loguru, pyyaml.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import typer
from loguru import logger

from fmllm.data.splits import assign_splits, save_splits_yaml
from fmllm.physics.lj_potential import total_energy_and_forces
from fmllm.physics.md import (
    equilibrate,
    maxwell_boltzmann_velocities,
    run_md,
)
from fmllm.physics.observables import (
    radial_distribution_function,
    rasterize_positions,
)
from fmllm.physics.structures import (
    VALID_MOTIFS_FOR_N,
    equilibrium_positions,
    valid_motifs,
)
from fmllm.utils.config import Config, load_config
from fmllm.utils.logging import configure_logging
from fmllm.utils.manifests import write_manifest
from fmllm.utils.run_ids import generate_run_id


# Stable mapping motif name -> small integer ID stored on disk.
MOTIF_NAMES: tuple[str, ...] = ("triangular_disk", "ring", "linear")
MOTIF_NAME_TO_ID: dict[str, int] = {name: i for i, name in enumerate(MOTIF_NAMES)}


# ---------------------------------------------------------------------------
# Specimen specs
# ---------------------------------------------------------------------------


def sample_specimen_specs(
    *,
    num_specimens: int,
    n_choices: list[int],
    t_min: float,
    t_max: float,
    master_seed: int,
) -> list[dict[str, object]]:
    """Sample ``(id, N, T, motif)`` per specimen with deterministic seeds.

    Each specimen ID gets its own RNG seeded by
    ``master_seed * num_specimens + id`` so the population stays
    reproducible across machines and partial regenerations.
    """
    specs: list[dict[str, object]] = []
    log_t_min = math.log(t_min)
    log_t_max = math.log(t_max)
    for i in range(num_specimens):
        rng = np.random.default_rng(master_seed * num_specimens + i)
        n_atoms = int(rng.choice(n_choices))
        temperature = float(math.exp(rng.uniform(log_t_min, log_t_max)))
        motifs = valid_motifs(n_atoms)
        motif = motifs[int(rng.integers(len(motifs)))]
        specs.append({
            "id": i,
            "n_atoms": n_atoms,
            "temperature": temperature,
            "motif": motif,
            "seed": int(master_seed * num_specimens + i),
        })
    return specs


# ---------------------------------------------------------------------------
# Batched simulation
# ---------------------------------------------------------------------------


def _initialize_batch(
    specs: list[dict[str, object]],
    *,
    n_atoms: int,
    perturbation_std: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build initial positions, velocities, and per-batch temperatures.

    All entries in the batch share ``n_atoms`` so the leading shape is
    ``(B, n_atoms, 2)``.
    """
    batch = len(specs)
    positions = torch.zeros((batch, n_atoms, 2), dtype=dtype, device=device)
    velocities = torch.zeros((batch, n_atoms, 2), dtype=dtype, device=device)
    temperatures = torch.zeros((batch,), dtype=dtype, device=device)

    for b, spec in enumerate(specs):
        per_seed = int(spec["seed"])  # type: ignore[arg-type]
        gen = torch.Generator(device="cpu").manual_seed(per_seed)
        eq = equilibrium_positions(n_atoms, motif=str(spec["motif"]))
        if perturbation_std > 0:
            eq = eq + perturbation_std * torch.randn(eq.shape, generator=gen)
        positions[b] = eq.to(device=device, dtype=dtype)
        v = maxwell_boltzmann_velocities(
            n_atoms,
            temperature=float(spec["temperature"]),  # type: ignore[arg-type]
            dim=2,
            device=device,
            dtype=dtype,
            generator=torch.Generator(device=device).manual_seed(per_seed + 1),
        )
        velocities[b] = v
        temperatures[b] = float(spec["temperature"])  # type: ignore[arg-type]

    return positions, velocities, temperatures


def _build_forces_fn(k_conf: float):
    def forces_fn(positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return total_energy_and_forces(positions, k_conf=k_conf)
    return forces_fn


def _generate_group(
    specs: list[dict[str, object]],
    *,
    cfg: Config,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, np.ndarray]:
    """Run equilibration plus trajectory recording for one ``(N, batch)``.

    Returns a dict of numpy arrays keyed by HDF5 dataset name. Each
    array has leading dim equal to ``len(specs)``.
    """
    n_atoms = int(specs[0]["n_atoms"])  # type: ignore[arg-type]
    assert all(int(s["n_atoms"]) == n_atoms for s in specs)  # type: ignore[arg-type]

    ds = cfg.dataset
    forces_fn = _build_forces_fn(ds.confinement_k)

    positions, velocities, temperatures = _initialize_batch(
        specs,
        n_atoms=n_atoms,
        perturbation_std=ds.perturbation_std,
        device=device,
        dtype=dtype,
    )
    eq_positions = positions.clone()

    # Equilibrate with a velocity-rescaling thermostat per specimen.
    # The thermostat treats the batch as independent simulations.
    for step in range(1, ds.md_equilibration_steps + 1):
        _, accelerations = forces_fn(positions)
        v_half = velocities + 0.5 * ds.md_dt * accelerations
        positions = positions + ds.md_dt * v_half
        _, accelerations = forces_fn(positions)
        velocities = v_half + 0.5 * ds.md_dt * accelerations
        if ds.md_thermostat_every > 0 and step % ds.md_thermostat_every == 0:
            ke = 0.5 * (velocities * velocities).sum(dim=(-1, -2), keepdim=True)
            free_dof = 2 * (n_atoms - 1)
            ke_target = (0.5 * free_dof * temperatures).view(-1, 1, 1)
            scale = torch.where(
                ke > 0,
                (ke_target / ke.clamp_min(1e-30)) ** 0.5,
                torch.ones_like(ke),
            )
            velocities = velocities * scale

    # Record the trajectory snippet.
    traj = run_md(
        positions,
        velocities,
        forces_fn,
        dt=ds.md_dt,
        n_steps=ds.md_steps_per_specimen,
        record_every=1,
        record_initial=True,
    )
    traj_positions = traj["positions"]   # (T+1, B, N, 2)
    traj_velocities = traj["velocities"]  # (T+1, B, N, 2)

    # Rearrange so leading dim is batch.
    traj_positions = traj_positions.permute(1, 0, 2, 3).contiguous()
    traj_velocities = traj_velocities.permute(1, 0, 2, 3).contiguous()

    # Compute observables on the trajectory's final frame.
    final_positions = traj_positions[:, -1]  # (B, N, 2)
    images = []
    rdfs = []
    for b in range(len(specs)):
        per_seed = int(specs[b]["seed"])  # type: ignore[arg-type]
        gen = torch.Generator(device="cpu").manual_seed(per_seed + 2)
        img = rasterize_positions(
            final_positions[b].cpu(),
            image_size=ds.image_size,
            pixel_size_lj=ds.image_pixel_size_lj,
            blur_radius_lj=ds.image_blur_radius_lj,
            noise_std=ds.image_noise_std,
            generator=gen if ds.image_noise_std > 0 else None,
        )
        images.append(img.numpy())
        g, _ = radial_distribution_function(
            final_positions[b].cpu(),
            r_max=ds.rdf_r_max,
            num_bins=ds.rdf_bins,
        )
        rdfs.append(g.numpy())

    return {
        "images": np.stack(images, axis=0).astype(np.float32),
        "rdfs": np.stack(rdfs, axis=0).astype(np.float32),
        "traj_positions": traj_positions.cpu().numpy().astype(np.float32),
        "traj_velocities": traj_velocities.cpu().numpy().astype(np.float32),
        "equilibrium_positions": eq_positions.cpu().numpy().astype(np.float32),
        "temperatures": temperatures.cpu().numpy().astype(np.float32),
    }


# ---------------------------------------------------------------------------
# HDF5 writer
# ---------------------------------------------------------------------------


def _allocate_hdf5(
    path: Path,
    *,
    num_specimens: int,
    max_n_atoms: int,
    image_size: int,
    rdf_bins: int,
    n_steps: int,
    cfg: Config,
) -> h5py.File:
    """Create the HDF5 file with empty fixed-shape datasets."""
    path.parent.mkdir(parents=True, exist_ok=True)
    f = h5py.File(path, "w")
    f.attrs["max_n_atoms"] = max_n_atoms
    f.attrs["image_size"] = image_size
    f.attrs["image_pixel_size_lj"] = cfg.dataset.image_pixel_size_lj
    f.attrs["image_blur_radius_lj"] = cfg.dataset.image_blur_radius_lj
    f.attrs["image_noise_std"] = cfg.dataset.image_noise_std
    f.attrs["rdf_bins"] = rdf_bins
    f.attrs["rdf_r_max"] = cfg.dataset.rdf_r_max
    f.attrs["md_dt"] = cfg.dataset.md_dt
    f.attrs["md_steps_per_specimen"] = n_steps
    f.attrs["md_equilibration_steps"] = cfg.dataset.md_equilibration_steps
    f.attrs["confinement_k"] = cfg.dataset.confinement_k
    f.attrs["motif_names"] = np.asarray(MOTIF_NAMES, dtype="S")

    f.create_dataset("images", shape=(num_specimens, image_size, image_size), dtype="f4")
    f.create_dataset("rdfs", shape=(num_specimens, rdf_bins), dtype="f4")
    f.create_dataset(
        "traj_positions",
        shape=(num_specimens, n_steps + 1, max_n_atoms, 2),
        dtype="f4",
    )
    f.create_dataset(
        "traj_velocities",
        shape=(num_specimens, n_steps + 1, max_n_atoms, 2),
        dtype="f4",
    )
    f.create_dataset(
        "equilibrium_positions",
        shape=(num_specimens, max_n_atoms, 2),
        dtype="f4",
    )
    f.create_dataset("atom_counts", shape=(num_specimens,), dtype="i4")
    f.create_dataset("temperatures", shape=(num_specimens,), dtype="f4")
    f.create_dataset("motif_ids", shape=(num_specimens,), dtype="i2")
    f.create_dataset("seeds", shape=(num_specimens,), dtype="i8")
    return f


def _write_group_to_hdf5(
    h5: h5py.File,
    *,
    specs: list[dict[str, object]],
    arrays: dict[str, np.ndarray],
    max_n_atoms: int,
) -> None:
    """Write a batch of specimens to their target rows in the HDF5 file."""
    for b, spec in enumerate(specs):
        row = int(spec["id"])  # type: ignore[arg-type]
        n = int(spec["n_atoms"])  # type: ignore[arg-type]

        h5["images"][row] = arrays["images"][b]
        h5["rdfs"][row] = arrays["rdfs"][b]

        traj_pos = np.zeros(
            (arrays["traj_positions"].shape[1], max_n_atoms, 2), dtype=np.float32,
        )
        traj_vel = np.zeros_like(traj_pos)
        traj_pos[:, :n, :] = arrays["traj_positions"][b]
        traj_vel[:, :n, :] = arrays["traj_velocities"][b]
        h5["traj_positions"][row] = traj_pos
        h5["traj_velocities"][row] = traj_vel

        eq = np.zeros((max_n_atoms, 2), dtype=np.float32)
        eq[:n] = arrays["equilibrium_positions"][b]
        h5["equilibrium_positions"][row] = eq

        h5["atom_counts"][row] = n
        h5["temperatures"][row] = float(arrays["temperatures"][b])
        h5["motif_ids"][row] = MOTIF_NAME_TO_ID[str(spec["motif"])]
        h5["seeds"][row] = int(spec["seed"])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device(device_arg)


def _grouped_batches(
    specs: list[dict[str, object]],
    batch_size: int,
) -> list[list[dict[str, object]]]:
    """Group specimens by ``n_atoms`` then chunk into ``batch_size`` blocks."""
    by_n: dict[int, list[dict[str, object]]] = defaultdict(list)
    for s in specs:
        by_n[int(s["n_atoms"])].append(s)  # type: ignore[arg-type]
    batches: list[list[dict[str, object]]] = []
    for n in sorted(by_n):
        group = by_n[n]
        for start in range(0, len(group), batch_size):
            batches.append(group[start : start + batch_size])
    return batches


app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def main(
    config: Path = typer.Option(
        Path("configs/default.yaml"),
        "--config", "-c",
        help="Path to the YAML config file.",
    ),
    out: Path = typer.Option(
        Path("data/synthetic_lj_v1"),
        "--out", "-o",
        help="Output directory for the dataset.",
    ),
    num_specimens: int | None = typer.Option(
        None, "--num-specimens", "-n",
        help="Override the number of specimens to generate. Useful for smoke tests.",
    ),
    device: str = typer.Option(
        "auto", "--device", "-d",
        help="Compute device. 'auto' picks cuda when available, else cpu.",
    ),
    batch_size: int | None = typer.Option(
        None, "--batch-size", "-b",
        help="Override generator_batch_size from the config.",
    ),
    smoke_test: bool = typer.Option(
        False, "--smoke-test",
        help="Run a tiny subset (200 specimens) for a quick end-to-end check.",
    ),
) -> None:
    """Generate the synthetic Lennard-Jones dataset."""
    cfg = load_config(config)

    if smoke_test:
        num_specimens = 200
    if num_specimens is None:
        num_specimens = cfg.dataset.num_specimens
    if batch_size is None:
        batch_size = cfg.dataset.generator_batch_size

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = generate_run_id("data-generate")
    run_dir = Path("runs") / run_id
    configure_logging(run_dir)

    logger.info(
        "Starting dataset generation: num_specimens={}, device={}, batch_size={}, out={}",
        num_specimens, device, batch_size, out_dir,
    )

    dev = _select_device(device)
    dtype = torch.float32

    specs = sample_specimen_specs(
        num_specimens=num_specimens,
        n_choices=cfg.dataset.n_choices,
        t_min=cfg.dataset.t_min,
        t_max=cfg.dataset.t_max,
        master_seed=cfg.dataset.generator_master_seed,
    )

    n_counts: dict[int, int] = defaultdict(int)
    for s in specs:
        n_counts[int(s["n_atoms"])] += 1  # type: ignore[arg-type]
    logger.info("Specimen-N distribution: {}", dict(sorted(n_counts.items())))

    max_n_atoms = max(cfg.dataset.n_choices)
    h5_path = out_dir / "specimens.h5"
    logger.info("Allocating HDF5 store at {}", h5_path)
    h5 = _allocate_hdf5(
        h5_path,
        num_specimens=num_specimens,
        max_n_atoms=max_n_atoms,
        image_size=cfg.dataset.image_size,
        rdf_bins=cfg.dataset.rdf_bins,
        n_steps=cfg.dataset.md_steps_per_specimen,
        cfg=cfg,
    )

    batches = _grouped_batches(specs, batch_size)
    logger.info("Processing {} batches", len(batches))

    t_start = time.time()
    try:
        for bi, batch_specs in enumerate(batches):
            arrays = _generate_group(batch_specs, cfg=cfg, device=dev, dtype=dtype)
            _write_group_to_hdf5(
                h5,
                specs=batch_specs,
                arrays=arrays,
                max_n_atoms=max_n_atoms,
            )
            if (bi + 1) % 10 == 0 or bi == 0 or bi == len(batches) - 1:
                elapsed = time.time() - t_start
                logger.info(
                    "Processed batch {}/{} (N={}, |B|={}). Elapsed {:.1f}s.",
                    bi + 1, len(batches), batch_specs[0]["n_atoms"],
                    len(batch_specs), elapsed,
                )
    finally:
        h5.close()

    # Write the splits YAML.
    atom_counts = [int(s["n_atoms"]) for s in specs]  # type: ignore[arg-type]
    temperatures = [float(s["temperature"]) for s in specs]  # type: ignore[arg-type]
    splits = assign_splits(
        specimen_ids=[int(s["id"]) for s in specs],  # type: ignore[arg-type]
        atom_counts=atom_counts,
        temperatures=temperatures,
        n_in_distribution=cfg.dataset.n_in_distribution,
        t_in_distribution_max=cfg.dataset.t_in_distribution_max,
        num_holdout=min(cfg.dataset.num_holdout, num_specimens),
        seed=cfg.dataset.generator_master_seed,
    )
    splits_path = out_dir / "splits.yaml"
    save_splits_yaml(splits, splits_path)
    logger.info("Wrote splits YAML to {}", splits_path)

    # Manifest.
    manifest_path = out_dir / "manifest.yaml"
    write_manifest(
        manifest_path,
        script="fmllm.data.generator",
        inputs={"config": str(config)},
        config={
            "dataset": cfg.dataset.model_dump(),
            "run_id": run_id,
            "device": str(dev),
            "batch_size": batch_size,
            "num_specimens": num_specimens,
        },
        extra={
            "num_batches": len(batches),
            "n_distribution": dict(sorted(n_counts.items())),
            "elapsed_seconds": time.time() - t_start,
            "h5_path": str(h5_path),
            "splits_path": str(splits_path),
        },
    )
    logger.info("Wrote manifest to {}", manifest_path)
    logger.info("Done in {:.1f}s.", time.time() - t_start)


if __name__ == "__main__":
    app()
