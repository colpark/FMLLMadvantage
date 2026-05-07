"""Stage 3: lock the held-out 200-specimen split for the materials port.

Stratified sampling so each major (crystal_system, is_metal) bucket
is represented in the held-out 200. Writes:

    data/materials_project_v1/splits.yaml         # train / holdout assignments
    data/materials_project_v1/holdout_lock/ids.json # the 200 held-out ids

The held-out 200 are excluded from probe training, SAE training,
CoT-SFT training, and labelling. Mirrors the LJ
``runs/holdout_lock/ids.json`` convention.

Usage:

    bash scripts/materials/02_lock_holdout.sh

Depends on:
    typer, h5py, numpy, pyyaml.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import typer
import yaml


app = typer.Typer(add_completion=False, no_args_is_help=False)


@app.command()
def main(
    h5_path: Path = typer.Option(
        Path("data/materials_project_v1/specimens.h5"), "--h5-path",
    ),
    splits_path: Path = typer.Option(
        Path("data/materials_project_v1/splits.yaml"), "--splits-path",
    ),
    holdout_ids_path: Path = typer.Option(
        Path("data/materials_project_v1/holdout_lock/ids.json"),
        "--holdout-ids-path",
    ),
    n_holdout: int = typer.Option(200, "--n-holdout"),
    stratify_by: str = typer.Option(
        "crystal_system,is_metal", "--stratify-by",
        help="Comma-separated keys to stratify the held-out sample by. "
             "Default: crystal_system,is_metal.",
    ),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    """Pick the held-out 200 with stratification and write splits."""
    if not h5_path.exists():
        typer.echo(f"ERROR: {h5_path} not found. Run stage 2 first.", err=True)
        sys.exit(2)

    rng = np.random.default_rng(seed)
    keys = [k.strip() for k in stratify_by.split(",") if k.strip()]

    typer.echo("==> Locking materials held-out split")
    typer.echo(f"    h5_path     : {h5_path}")
    typer.echo(f"    n_holdout   : {n_holdout}")
    typer.echo(f"    stratify_by : {keys}")
    typer.echo(f"    seed        : {seed}")
    typer.echo("")

    with h5py.File(h5_path, "r") as h5:
        n = int(h5.attrs.get("n_specimens", h5["material_id"].shape[0]))
        material_ids = [
            (m.decode() if isinstance(m, bytes) else str(m))
            for m in h5["material_id"][:]
        ]
        crystal_systems_attr = h5.attrs.get("crystal_systems")
        if crystal_systems_attr is not None:
            crystal_systems = [
                s.decode() if isinstance(s, bytes) else str(s)
                for s in crystal_systems_attr
            ]
        else:
            crystal_systems = [
                "triclinic", "monoclinic", "orthorhombic",
                "tetragonal", "trigonal", "hexagonal", "cubic",
            ]

        cs_ids = h5["crystal_system_id"][:]
        is_metal = h5["is_metal"][:]

        def bucket_key(idx: int) -> tuple:
            parts = []
            for k in keys:
                if k == "crystal_system":
                    cs = int(cs_ids[idx])
                    parts.append(
                        crystal_systems[cs] if 0 <= cs < len(crystal_systems) else "?"
                    )
                elif k == "is_metal":
                    parts.append("metal" if bool(is_metal[idx]) else "nonmetal")
                else:
                    parts.append("?")
            return tuple(parts)

        # Bucket every specimen.
        buckets: dict[tuple, list[int]] = defaultdict(list)
        for i in range(n):
            buckets[bucket_key(i)].append(i)

        typer.echo(f"==> Buckets: {len(buckets)}")
        bucket_counts = sorted(
            [(k, len(v)) for k, v in buckets.items()],
            key=lambda kv: -kv[1],
        )
        for k, c in bucket_counts:
            typer.echo(f"    {k}: {c}")
        typer.echo("")

        # Proportional allocation across buckets, capped at bucket size,
        # filled out by random sampling from the largest bucket if we
        # are short.
        target_per_bucket = {
            k: max(1, int(round(n_holdout * len(v) / max(n, 1))))
            for k, v in buckets.items()
        }
        # Adjust totals to exactly n_holdout.
        cur = sum(target_per_bucket.values())
        # Iterate adjusting one at a time.
        sorted_keys = [k for k, _ in bucket_counts]
        i = 0
        while cur != n_holdout and sorted_keys:
            key = sorted_keys[i % len(sorted_keys)]
            if cur < n_holdout and target_per_bucket[key] < len(buckets[key]):
                target_per_bucket[key] += 1
                cur += 1
            elif cur > n_holdout and target_per_bucket[key] > 0:
                target_per_bucket[key] -= 1
                cur -= 1
            i += 1
            if i > 100000:
                break

        # Sample.
        holdout_indices: list[int] = []
        for k, v in buckets.items():
            t = min(target_per_bucket.get(k, 0), len(v))
            if t == 0:
                continue
            chosen = rng.choice(np.asarray(v), size=t, replace=False)
            holdout_indices.extend(int(x) for x in chosen)
        holdout_indices.sort()
        holdout_set = set(holdout_indices)
        train_indices = [i for i in range(n) if i not in holdout_set]

        typer.echo(
            f"==> n_holdout={len(holdout_indices)} (target {n_holdout}); "
            f"n_train={len(train_indices)}"
        )

        # Write outputs.
        holdout_ids_path.parent.mkdir(parents=True, exist_ok=True)
        with holdout_ids_path.open("w") as f:
            json.dump(holdout_indices, f)

        splits_path.parent.mkdir(parents=True, exist_ok=True)
        with splits_path.open("w") as f:
            yaml.safe_dump(
                {
                    "train": train_indices,
                    "holdout": holdout_indices,
                    "stratify_by": keys,
                    "seed": seed,
                    "n_total": n,
                    "n_train": len(train_indices),
                    "n_holdout": len(holdout_indices),
                    "bucket_counts": {",".join(k): c for k, c in bucket_counts},
                    "target_per_bucket": {
                        ",".join(k): t for k, t in target_per_bucket.items()
                    },
                },
                f,
                sort_keys=False,
            )

    typer.echo(f"==> Wrote: {holdout_ids_path}")
    typer.echo(f"==> Wrote: {splits_path}")


if __name__ == "__main__":
    app()
