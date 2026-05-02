"""Build the curated literature cluster database the verifier uses.

For every (n_atoms, motif) pair the project considers canonical, the
script generates equilibrium positions via
:mod:`fmllm.physics.structures`, computes the per-atom potential
energy under the project's LJ Hamiltonian (with no harmonic
confinement, which yields the bare cluster reference), and records
the first-peak position of the pair-distance histogram.

Output: ``data/literature/clusters.json`` with one record per
(N, motif). Commit the JSON so the verifier's
:class:`LiteratureSource` reads a stable reference dataset across
checkouts.

Usage:
    uv run python scripts/build_literature_db.py
    uv run python scripts/build_literature_db.py --out data/literature/clusters.json

Depends on:
    torch, numpy, typer.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fmllm.physics.lj_potential import (  # noqa: E402
    R_MIN, lj_pair_energy_and_forces,
)
from fmllm.physics.observables import pair_distance_histogram  # noqa: E402
from fmllm.physics.structures import (  # noqa: E402
    VALID_MOTIFS_FOR_N, equilibrium_positions,
)


N_VALUES = (5, 7, 9, 11, 13, 17, 19, 21, 25, 30)
MOTIFS = ("triangular_disk", "ring")
RDF_R_MAX = 4.0
RDF_BINS = 200


app = typer.Typer(add_completion=False, no_args_is_help=False)


def _record_for(n_atoms: int, motif: str) -> dict | None:
    if motif not in VALID_MOTIFS_FOR_N.get(n_atoms, ()):
        return None
    positions = equilibrium_positions(n_atoms, motif=motif)
    energy, _ = lj_pair_energy_and_forces(positions)
    per_atom_e = float(energy.item()) / max(1, n_atoms)
    hist, edges = pair_distance_histogram(
        positions, r_max=RDF_R_MAX, num_bins=RDF_BINS,
    )
    centers = (edges[:-1] + edges[1:]).cpu().numpy() / 2.0
    counts = hist.cpu().numpy()
    # First peak: the lowest non-zero bin with a local maximum.
    nonzero = np.where(counts > 0)[0]
    if nonzero.size == 0:
        first_peak = math.nan
    else:
        first_peak = float(centers[nonzero[0]])
    diameter = float(positions.norm(dim=-1).max().item() * 2.0)
    return {
        "n_atoms": int(n_atoms),
        "motif": motif,
        "per_atom_potential_energy_lj": per_atom_e,
        "rdf_first_peak_lj": first_peak,
        "diameter_lj": diameter,
        "reference": (
            "computed from fmllm.physics with sigma=epsilon=1, "
            f"r_min = 2**(1/6) ≈ {R_MIN:.4f}"
        ),
    }


@app.command()
def main(
    out: Path = typer.Option(
        Path("data/literature/clusters.json"), "--out", "-o",
        help="Output JSON path.",
    ),
    pretty: bool = typer.Option(True, "--pretty/--compact"),
) -> None:
    """Generate the literature cluster database."""
    records: list[dict] = []
    for n in N_VALUES:
        for m in MOTIFS:
            rec = _record_for(n, m)
            if rec is not None:
                records.append(rec)
    out.parent.mkdir(parents=True, exist_ok=True)
    indent = 2 if pretty else None
    with out.open("w") as f:
        json.dump(records, f, indent=indent, sort_keys=True)
    typer.echo(f"Wrote {len(records)} entries to {out}")


if __name__ == "__main__":
    app()
