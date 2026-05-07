"""Stage 1: download Materials Project structures + properties.

Queries the new Materials Project API for all materials matching
``e_above_hull <= E_ABOVE_HULL_MAX``. Caches each material as
``data/materials_project_v1/raw/<material_id>.json.gz``.

Resumable: re-running skips materials whose JSON is already on disk.

Output schema per file (gzipped JSON):

    material_id : "mp-149"
    formula_pretty : "Si"
    elements : ["Si"]
    nsites : int
    volume : float
    density : float
    formation_energy_per_atom : float
    energy_above_hull : float
    band_gap : float
    is_metal : bool
    total_magnetization : float | None
    symmetry:
      crystal_system : str
      space_group_symbol : str
      space_group_number : int
    structure : (pymatgen Structure as_dict() output)

Usage:

    export MP_API_KEY=<key>
    bash scripts/materials/00_download_mp.sh

Depends on:
    typer, mp-api, pymatgen.
"""

from __future__ import annotations

import gzip
import json
import os
import sys
import time
from pathlib import Path

import typer


app = typer.Typer(add_completion=False, no_args_is_help=False)


@app.command()
def main(
    raw_dir: Path = typer.Option(
        Path("data/materials_project_v1/raw"), "--raw-dir",
    ),
    n_max: int = typer.Option(
        50000, "--n-max",
        help="Cap on number of materials to fetch.",
    ),
    e_above_hull_max: float = typer.Option(
        0.5, "--e-above-hull-max",
        help="Stability filter (eV/atom). Default 0.5 keeps stable + "
             "metastable while excluding implausible structures.",
    ),
    batch: int = typer.Option(
        500, "--batch",
        help="Materials per API page.",
    ),
    progress_every: int = typer.Option(50, "--progress-every"),
    api_key_env: str = typer.Option("MP_API_KEY", "--api-key-env"),
) -> None:
    """Download Materials Project structures + properties."""
    api_key = os.environ.get(api_key_env)
    if not api_key:
        typer.echo(
            f"ERROR: ${api_key_env} env var not set. Get a free key at "
            f"https://next-gen.materialsproject.org/api and export it.",
            err=True,
        )
        sys.exit(2)

    from mp_api.client import MPRester  # noqa: PLC0415

    raw_dir.mkdir(parents=True, exist_ok=True)

    typer.echo("==> Materials Project download")
    typer.echo(f"    raw_dir          : {raw_dir}")
    typer.echo(f"    n_max            : {n_max}")
    typer.echo(f"    e_above_hull_max : {e_above_hull_max}")
    typer.echo(f"    batch            : {batch}")
    typer.echo("")

    # Discovery query: pull a thin summary first to know the candidate set.
    fields = [
        "material_id",
        "formula_pretty",
        "elements",
        "nsites",
        "volume",
        "density",
        "formation_energy_per_atom",
        "energy_above_hull",
        "band_gap",
        "is_metal",
        "is_magnetic",
        "total_magnetization",
        "symmetry",
        "structure",
    ]

    typer.echo("==> Querying Materials Project ...")
    t0 = time.time()
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(
            energy_above_hull=(0.0, e_above_hull_max),
            fields=fields,
            num_chunks=None,
            chunk_size=batch,
        )
    typer.echo(f"    received {len(docs)} candidates in {time.time() - t0:.1f}s")

    if len(docs) > n_max:
        typer.echo(f"    capping at n_max={n_max}")
        docs = docs[:n_max]

    typer.echo("")
    typer.echo("==> Writing per-material JSON")

    n_written = 0
    n_skipped = 0
    for i, doc in enumerate(docs):
        try:
            material_id = doc.material_id
        except AttributeError:
            material_id = doc.get("material_id")
        if not material_id:
            continue

        out_path = raw_dir / f"{material_id}.json.gz"
        if out_path.exists():
            n_skipped += 1
            if (i + 1) % progress_every == 0:
                typer.echo(
                    f"    {i + 1:>6}/{len(docs)} written={n_written} skipped={n_skipped}"
                )
            continue

        record = _doc_to_dict(doc)
        with gzip.open(out_path, "wt") as f:
            json.dump(record, f, separators=(",", ":"), default=str)
        n_written += 1

        if (i + 1) % progress_every == 0:
            typer.echo(
                f"    {i + 1:>6}/{len(docs)} written={n_written} skipped={n_skipped}"
            )

    typer.echo("")
    typer.echo(
        f"==> Done: {n_written} new, {n_skipped} already cached "
        f"(total {len(docs)} candidates)."
    )


def _doc_to_dict(doc: object) -> dict:
    """Convert an mp-api SummaryDoc into a plain JSON-serializable dict.

    The SummaryDoc has both attribute and dict-like access modes
    depending on the mp-api version; we handle both.
    """
    def _g(name: str, default=None):
        if hasattr(doc, name):
            return getattr(doc, name)
        if isinstance(doc, dict):
            return doc.get(name, default)
        return default

    structure = _g("structure")
    structure_dict = (
        structure.as_dict() if hasattr(structure, "as_dict") else structure
    )

    symmetry = _g("symmetry")
    if symmetry is not None and hasattr(symmetry, "dict"):
        sym_dict = symmetry.dict() if callable(symmetry.dict) else dict(symmetry.dict)
    elif hasattr(symmetry, "__dict__"):
        sym_dict = {
            k: getattr(symmetry, k) for k in (
                "crystal_system", "symbol", "number", "point_group",
                "symprec", "version",
            )
            if hasattr(symmetry, k)
        }
    else:
        sym_dict = symmetry

    # Normalize symmetry field names: mp-api's pydantic model uses
    # 'number' / 'symbol', but downstream code (01_build_mp_h5) reads
    # 'space_group_number' / 'space_group_symbol'. Forward-compatible
    # rename with both keys present.
    if isinstance(sym_dict, dict):
        if "space_group_number" not in sym_dict and "number" in sym_dict:
            sym_dict["space_group_number"] = sym_dict["number"]
        if "space_group_symbol" not in sym_dict and "symbol" in sym_dict:
            sym_dict["space_group_symbol"] = sym_dict["symbol"]

    return {
        "material_id": str(_g("material_id", "")),
        "formula_pretty": _g("formula_pretty", ""),
        "elements": [str(e) for e in (_g("elements") or [])],
        "nsites": int(_g("nsites", 0) or 0),
        "volume": float(_g("volume", 0.0) or 0.0),
        "density": float(_g("density", 0.0) or 0.0),
        "formation_energy_per_atom": (
            float(_g("formation_energy_per_atom", 0.0) or 0.0)
        ),
        "energy_above_hull": float(_g("energy_above_hull", 0.0) or 0.0),
        "band_gap": float(_g("band_gap", 0.0) or 0.0),
        "is_metal": bool(_g("is_metal", False) or False),
        "is_magnetic": bool(_g("is_magnetic", False) or False),
        "total_magnetization": _g("total_magnetization"),
        "symmetry": sym_dict,
        "structure": structure_dict,
    }


if __name__ == "__main__":
    app()
