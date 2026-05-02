"""CLI: save inspectable samples from the synthetic LJ dataset.

The script picks a small set of specimens from the HDF5 store the
Phase 1 generator produces and writes per-specimen artifacts the user
can eyeball: a rasterized-image PNG, an RDF plot, a positions
scatter, a trajectory overlay, and a summary YAML. It also produces
two grid figures (images and RDFs) so a single PDF or image lets the
reader scan many specimens at once.

Usage:
    uv run python scripts/save_data_samples.py \\
        --h5-path data/synthetic_lj_v1/specimens.h5 \\
        --out runs/data-samples \\
        --n-samples 16 \\
        --stratify-by-n

Sampling strategies (mutually exclusive):
    --indices 0,42,1234    explicit specimen IDs
    --stratify-by-n        one specimen per atom-count value, then fill
    --random               uniform random over specimens (default)

The script does not require a GPU and runs on the laptop as well as
the remote (it sets matplotlib to the Agg backend so it works under
SSH).

Depends on:
    h5py, numpy, matplotlib, pyyaml, typer.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import typer  # noqa: E402
import yaml  # noqa: E402

# Ensure the package is importable when running from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fmllm.utils.manifests import write_manifest  # noqa: E402
from fmllm.utils.run_ids import generate_run_id  # noqa: E402

app = typer.Typer(add_completion=False, no_args_is_help=True)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _parse_indices(spec: str | None) -> list[int] | None:
    if not spec:
        return None
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    return [int(p) for p in parts]


def _select_indices(
    f: h5py.File,
    *,
    n_samples: int,
    indices: list[int] | None,
    stratify_by_n: bool,
    seed: int,
) -> list[int]:
    total = int(f["atom_counts"].shape[0])
    if indices is not None:
        for i in indices:
            if not (0 <= i < total):
                raise typer.BadParameter(f"index {i} out of range [0, {total})")
        return indices

    rng = np.random.default_rng(seed)
    if stratify_by_n:
        atom_counts = np.asarray(f["atom_counts"]).astype(np.int64)
        by_n: dict[int, list[int]] = defaultdict(list)
        for idx, n in enumerate(atom_counts):
            by_n[int(n)].append(idx)
        n_values = sorted(by_n)
        chosen: list[int] = []
        # Round 1: one per N value.
        for n in n_values:
            chosen.append(int(rng.choice(by_n[n])))
            if len(chosen) >= n_samples:
                break
        # Round 2: top up uniformly across the population.
        if len(chosen) < n_samples:
            remaining = n_samples - len(chosen)
            extra = rng.choice(total, size=min(remaining, total), replace=False)
            for x in extra:
                if int(x) not in chosen:
                    chosen.append(int(x))
                if len(chosen) >= n_samples:
                    break
        return chosen[:n_samples]

    # Pure random.
    sel = rng.choice(total, size=min(n_samples, total), replace=False)
    return sorted(int(x) for x in sel)


# ---------------------------------------------------------------------------
# Per-specimen plots
# ---------------------------------------------------------------------------


def _plot_image(image: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.imshow(image, cmap="gray", origin="upper")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_rdf(rdf: np.ndarray, r_max: float, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    r = np.linspace(0.0, r_max, rdf.shape[0] + 1)[:-1] + (r_max / rdf.shape[0]) / 2.0
    ax.plot(r, rdf, color="tab:blue", linewidth=1.4)
    ax.set_xlabel(r"$r$ (LJ units)")
    ax.set_ylabel(r"$g(r)$")
    ax.set_xlim(0, r_max)
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_positions(
    positions: np.ndarray,
    n_atoms: int,
    path: Path,
    title: str,
    box_half_width: float,
) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    real = positions[:n_atoms]
    ax.scatter(real[:, 0], real[:, 1], s=80, color="tab:orange",
               edgecolor="black", linewidth=0.7, zorder=3)
    ax.set_xlim(-box_half_width, box_half_width)
    ax.set_ylim(-box_half_width, box_half_width)
    ax.set_aspect("equal")
    ax.axhline(0, color="k", linewidth=0.3, alpha=0.3)
    ax.axvline(0, color="k", linewidth=0.3, alpha=0.3)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x (LJ units)")
    ax.set_ylabel("y (LJ units)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_trajectory(
    traj_positions: np.ndarray,
    n_atoms: int,
    path: Path,
    title: str,
    box_half_width: float,
) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    real = traj_positions[:, :n_atoms]
    n_frames = real.shape[0]

    # Per-atom faint trajectory lines.
    for i in range(n_atoms):
        ax.plot(real[:, i, 0], real[:, i, 1],
                color="tab:gray", alpha=0.35, linewidth=0.5)

    # Initial frame in light blue.
    ax.scatter(real[0, :, 0], real[0, :, 1], s=40,
               color="tab:blue", edgecolor="black", linewidth=0.5,
               label=f"frame 0", zorder=3)
    # Final frame in orange.
    ax.scatter(real[-1, :, 0], real[-1, :, 1], s=40,
               color="tab:orange", edgecolor="black", linewidth=0.5,
               label=f"frame {n_frames - 1}", zorder=3)

    ax.set_xlim(-box_half_width, box_half_width)
    ax.set_ylim(-box_half_width, box_half_width)
    ax.set_aspect("equal")
    ax.axhline(0, color="k", linewidth=0.3, alpha=0.3)
    ax.axvline(0, color="k", linewidth=0.3, alpha=0.3)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x (LJ units)")
    ax.set_ylabel("y (LJ units)")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Grid figures
# ---------------------------------------------------------------------------


def _grid_images(
    samples: list[dict],
    path: Path,
) -> None:
    n = len(samples)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.4 * rows))
    axes = np.atleast_2d(axes)
    for k, s in enumerate(samples):
        ax = axes[k // cols, k % cols]
        ax.imshow(s["image"], cmap="gray", origin="upper")
        ax.set_title(
            f"id={s['id']} N={s['n_atoms']} T={s['temperature']:.2f}",
            fontsize=8,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    for k in range(n, rows * cols):
        axes[k // cols, k % cols].axis("off")
    fig.suptitle(f"Image samples (n={n})", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _grid_rdfs(
    samples: list[dict],
    r_max: float,
    path: Path,
) -> None:
    n = len(samples)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.2 * rows))
    axes = np.atleast_2d(axes)
    for k, s in enumerate(samples):
        ax = axes[k // cols, k % cols]
        rdf = s["rdf"]
        r = np.linspace(0.0, r_max, rdf.shape[0] + 1)[:-1] + (r_max / rdf.shape[0]) / 2.0
        ax.plot(r, rdf, linewidth=1.0, color="tab:blue")
        ax.set_title(
            f"id={s['id']} N={s['n_atoms']} T={s['temperature']:.2f}",
            fontsize=8,
        )
        ax.set_xlim(0, r_max)
        ax.set_xlabel("r")
        ax.grid(alpha=0.3)
    for k in range(n, rows * cols):
        axes[k // cols, k % cols].axis("off")
    fig.suptitle(f"RDF samples (n={n})", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    h5_path: Path = typer.Option(
        Path("data/synthetic_lj_v1/specimens.h5"), "--h5-path",
        help="Path to the dataset HDF5 file.",
    ),
    out: Path = typer.Option(
        Path("runs/data-samples"), "--out", "-o",
        help="Output directory. The script creates a run-id subdirectory inside.",
    ),
    n_samples: int = typer.Option(
        16, "--n-samples", "-n",
        help="Number of specimens to save.",
    ),
    indices: str = typer.Option(
        None, "--indices",
        help="Comma-separated list of specimen IDs. Overrides --n-samples and sampling flags.",
    ),
    stratify_by_n: bool = typer.Option(
        False, "--stratify-by-n",
        help="Pick one specimen per atom-count value, then fill the remainder uniformly.",
    ),
    seed: int = typer.Option(0, "--seed", help="Random seed for sampling."),
    box_half_width: float = typer.Option(
        4.8, "--box-half-width",
        help="Half-width of the imaging box for position / trajectory plots.",
    ),
) -> None:
    """Save inspectable samples from the synthetic LJ dataset."""
    if not h5_path.exists():
        raise typer.BadParameter(f"HDF5 file not found: {h5_path}")

    run_id = generate_run_id("data-samples")
    out_dir = out / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"==> Reading {h5_path}")
    typer.echo(f"==> Output  {out_dir}")

    selected_indices = _parse_indices(indices)

    samples: list[dict] = []
    with h5py.File(h5_path, "r") as f:
        r_max = float(f.attrs.get("rdf_r_max", 6.0))
        max_n_atoms = int(f.attrs.get("max_n_atoms", 30))
        sample_idx = _select_indices(
            f,
            n_samples=n_samples,
            indices=selected_indices,
            stratify_by_n=stratify_by_n,
            seed=seed,
        )

        atom_counts = np.asarray(f["atom_counts"])
        temperatures = np.asarray(f["temperatures"])
        motif_ids = np.asarray(f["motif_ids"])
        motif_names = (
            [s.decode() if isinstance(s, bytes) else str(s)
             for s in f.attrs.get("motif_names", [])]
            if "motif_names" in f.attrs else []
        )

        for idx in sample_idx:
            sid = int(idx)
            n_atoms = int(atom_counts[sid])
            t = float(temperatures[sid])
            motif_id = int(motif_ids[sid])
            motif_name = motif_names[motif_id] if motif_id < len(motif_names) else str(motif_id)

            image = np.asarray(f["images"][sid])
            rdf = np.asarray(f["rdfs"][sid])
            traj_pos = np.asarray(f["traj_positions"][sid])
            eq_pos = np.asarray(f["equilibrium_positions"][sid])

            samples.append({
                "id": sid,
                "n_atoms": n_atoms,
                "temperature": t,
                "motif": motif_name,
                "image": image,
                "rdf": rdf,
                "traj_pos": traj_pos,
                "eq_pos": eq_pos,
            })

    typer.echo(f"==> Selected {len(samples)} specimens: "
               f"{[s['id'] for s in samples]}")

    # Per-specimen plots and summary YAML.
    index_entries: list[dict] = []
    for s in samples:
        spec_dir = out_dir / f"specimen_{s['id']:06d}"
        spec_dir.mkdir(parents=True, exist_ok=True)
        title = (f"id={s['id']} N={s['n_atoms']} T={s['temperature']:.2f} "
                 f"motif={s['motif']}")

        _plot_image(s["image"], spec_dir / "image.png", title)
        _plot_rdf(s["rdf"], r_max, spec_dir / "rdf.png", title)
        _plot_positions(
            s["eq_pos"], s["n_atoms"], spec_dir / "positions_initial.png",
            "Initial " + title, box_half_width,
        )
        _plot_trajectory(
            s["traj_pos"], s["n_atoms"], spec_dir / "trajectory.png",
            title, box_half_width,
        )

        summary = {
            "specimen_id": int(s["id"]),
            "n_atoms": int(s["n_atoms"]),
            "temperature_lj": float(s["temperature"]),
            "motif": s["motif"],
            "image_min": float(s["image"].min()),
            "image_max": float(s["image"].max()),
            "image_mean": float(s["image"].mean()),
            "rdf_sum": float(s["rdf"].sum()),
            "rdf_argmax_bin": int(np.argmax(s["rdf"])),
            "trajectory_n_frames": int(s["traj_pos"].shape[0]),
            "max_n_atoms_padding": int(max_n_atoms),
        }
        with (spec_dir / "summary.yaml").open("w") as fy:
            yaml.safe_dump(summary, fy, sort_keys=False)
        index_entries.append({
            "specimen_id": int(s["id"]),
            "n_atoms": int(s["n_atoms"]),
            "temperature_lj": float(s["temperature"]),
            "motif": s["motif"],
            "directory": spec_dir.name,
        })

    # Grid figures.
    if samples:
        _grid_images(samples, out_dir / "grid_images.png")
        _grid_rdfs(samples, r_max, out_dir / "grid_rdfs.png")

    # Index + manifest.
    with (out_dir / "index.yaml").open("w") as f:
        yaml.safe_dump(
            {"run_id": run_id, "h5_path": str(h5_path),
             "n_samples": len(samples), "specimens": index_entries},
            f, sort_keys=False,
        )

    write_manifest(
        out_dir / "manifest.yaml",
        script="scripts.save_data_samples",
        inputs={"h5_path": str(h5_path)},
        config={
            "n_samples": int(n_samples),
            "indices": selected_indices,
            "stratify_by_n": bool(stratify_by_n),
            "seed": int(seed),
            "box_half_width": float(box_half_width),
        },
        extra={
            "run_id": run_id,
            "selected_specimen_ids": [s["id"] for s in samples],
            "rdf_r_max": r_max,
            "max_n_atoms": max_n_atoms,
        },
    )

    typer.echo(f"==> Done. {len(samples)} specimens written to {out_dir}")


if __name__ == "__main__":
    app()
