"""CLI: resolve configs/holdout_lock.yaml into a concrete specimen ID list.

Reads the lock file, opens splits.yaml at the locked path, picks the
specimens by the locked selection rule, and writes the resulting ID
list to ``runs/holdout_lock/ids.json``. Subsequent baselines pass
this file via ``--specimen-ids-file`` so every configuration sees
exactly the same specimens.

The script also writes a summary YAML next to the JSON for human
inspection: which cell each ID came from, which dev-set IDs were
excluded, the final count.

Usage:
    uv run python scripts/pick_holdout_ids.py
    uv run python scripts/pick_holdout_ids.py --lock-file configs/holdout_lock.yaml \\
        --out runs/holdout_lock

Depends on:
    typer, pyyaml.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


app = typer.Typer(add_completion=False, no_args_is_help=False)


def _load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


@app.command()
def main(
    lock_file: Path = typer.Option(
        Path("configs/holdout_lock.yaml"), "--lock-file",
    ),
    out: Path = typer.Option(Path("runs/holdout_lock"), "--out"),
) -> None:
    """Resolve the held-out specimen IDs from splits.yaml."""
    if not lock_file.exists():
        typer.echo(f"ERROR: lock file not found: {lock_file}")
        raise typer.Exit(code=1)
    lock = _load_yaml(lock_file)

    splits_path = Path(lock.get("splits_path", "data/synthetic_lj_v1/splits.yaml"))
    if not splits_path.exists():
        typer.echo(f"ERROR: splits file not found: {splits_path}")
        raise typer.Exit(code=1)
    splits = _load_yaml(splits_path)

    sel = lock.get("selection") or {}
    source = sel.get("source")
    count = int(sel.get("count", 200))

    chosen: list[int] = []
    chosen_provenance: list[dict] = []

    if source == "splits.holdout":
        cell = sel.get("cell", "in_distribution")
        fallback_cells = list(sel.get("fallback_cells") or [])

        holdout = splits.get("holdout") or {}
        if not isinstance(holdout, dict):
            typer.echo("ERROR: splits.yaml `holdout` is not a dict of cells")
            raise typer.Exit(code=1)

        cells_in_order = [cell] + fallback_cells
        for c in cells_in_order:
            ids = sorted(int(x) for x in (holdout.get(c) or []))
            for sid in ids:
                if len(chosen) >= count:
                    break
                chosen.append(sid)
                chosen_provenance.append({"specimen_id": sid, "cell": c})
            if len(chosen) >= count:
                break

        if not chosen:
            cell_sizes = {
                k: len(v) for k, v in holdout.items() if isinstance(v, list)
            }
            typer.echo("")
            typer.echo("ERROR: splits.yaml's holdout partition is empty.")
            typer.echo(f"  splits.yaml keys      : {sorted(splits.keys())}")
            typer.echo(f"  holdout cell sizes    : {cell_sizes}")
            typer.echo(f"  meta.num_holdout      : "
                       f"{(splits.get('meta') or {}).get('num_holdout')}")
            typer.echo("")
            typer.echo(
                "Switch configs/holdout_lock.yaml to a contiguous range. "
                "Example:"
            )
            typer.echo("")
            typer.echo("  selection:")
            typer.echo("    source: contiguous_range")
            typer.echo("    start: 40000")
            typer.echo("    count: 200")
            typer.echo("")
            typer.echo(
                "Pick a range disjoint from the dev set [0, 200) and "
                "from any prior baseline collection."
            )
            raise typer.Exit(code=1)

    elif source == "contiguous_range":
        rng_start = int(sel.get("start", 0))
        for sid in range(rng_start, rng_start + count):
            chosen.append(sid)
            chosen_provenance.append({"specimen_id": sid, "cell": "contiguous"})

    else:
        typer.echo(
            f"ERROR: unsupported selection.source {source!r}; "
            "expected splits.holdout or contiguous_range"
        )
        raise typer.Exit(code=1)

    if len(chosen) < count:
        typer.echo(
            f"WARNING: requested {count} specimens, only {len(chosen)} resolved"
        )

    # Defensive check: do not include any dev-set ID.
    dev = lock.get("dev_set") or {}
    dev_start = int(dev.get("start", 0))
    dev_count = int(dev.get("count", 0))
    dev_ids = set(range(dev_start, dev_start + dev_count))
    overlap = [sid for sid in chosen if sid in dev_ids]
    if overlap:
        typer.echo(
            f"ERROR: held-out selection overlaps the dev set "
            f"[{dev_start}, {dev_start + dev_count}): {overlap[:5]}..."
        )
        raise typer.Exit(code=1)

    out.mkdir(parents=True, exist_ok=True)
    ids_path = out / "ids.json"
    with ids_path.open("w") as f:
        json.dump(chosen, f)
    summary_path = out / "summary.yaml"
    with summary_path.open("w") as f:
        yaml.safe_dump(
            {
                "lock_file": str(lock_file),
                "lock_version": lock.get("version"),
                "locked_at_commit": lock.get("locked_at_commit"),
                "splits_path": str(splits_path),
                "n_chosen": len(chosen),
                "first_5": chosen[:5],
                "last_5": chosen[-5:],
                "dev_overlap_count": 0,
                "provenance_by_cell": _aggregate_by_cell(chosen_provenance),
            },
            f,
            sort_keys=False,
        )

    typer.echo(f"==> Held-out IDs written: {ids_path} ({len(chosen)} specimens)")
    typer.echo(f"==> Summary             : {summary_path}")
    typer.echo(f"==> First 5: {chosen[:5]}")
    typer.echo(f"==> Last  5: {chosen[-5:]}")


def _aggregate_by_cell(prov: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for entry in prov:
        c = entry["cell"]
        out[c] = out.get(c, 0) + 1
    return out


if __name__ == "__main__":
    app()
