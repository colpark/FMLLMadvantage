"""Stage 7: build SAE-augmented synthetic CoT records for materials.

Mirrors ``scripts/build_cot_dataset_with_sae.py`` from the LJ
pipeline but uses materials' cached CHGNet embeddings, materials'
probe bank, materials' SAE + labels, and materials' synthetic-CoT
generator.

Each output record is the chat tuple (system, user, assistant)
where:

  * the user message contains both PROBES and SAE_FEATURES,
  * the assistant message is the deterministic Step-1 / 1b / 2 / 3
    / Final-commit chain produced by
    ``fmllm.materials.synthetic_cot.build_sft_record``,
  * the final commit JSON comes from materials ground truth, not
    probe consensus.

Output:

    runs/materials/cot_datasets_sae/<run_id>/records.jsonl
    runs/materials/cot_datasets_sae/<run_id>/manifest.yaml

Usage:

    bash scripts/materials/07_build_cot.sh
    uv run python scripts/materials/07_build_cot.py --n-specimens 10000

Depends on:
    typer, h5py, numpy, torch, pyyaml.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import torch
import typer
import yaml


app = typer.Typer(add_completion=False, no_args_is_help=False)


def _generate_run_id(slug: str = "cot-sae-dataset") -> str:
    return f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{slug}"


def _latest_dir(parent: Path) -> Path | None:
    if not parent.exists():
        return None
    cands = sorted(
        parent.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True,
    )
    return next((c for c in cands if c.is_dir()), None)


def _latest_labels(parent: Path) -> Path | None:
    if not parent.exists():
        return None
    cands = sorted(
        parent.glob("*/labels.json"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    return cands[0] if cands else None


def _load_sae(
    sae_path: Path, device: str,
) -> tuple[object, torch.Tensor, torch.Tensor]:
    from fmllm.representation.sae import TopKSAE  # noqa: PLC0415

    payload = torch.load(sae_path, map_location=device, weights_only=False)
    sae = TopKSAE(
        in_dim=int(payload["in_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        k=int(payload["k"]),
    ).to(device)
    sae.load_state_dict(payload["state_dict"], strict=True)
    sae.eval()
    cls_mean = torch.from_numpy(
        np.asarray(payload["cls_mean"], dtype=np.float32),
    ).to(device).flatten()
    cls_std = torch.from_numpy(
        np.asarray(payload["cls_std"], dtype=np.float32),
    ).to(device).flatten()
    return sae, cls_mean, cls_std


def _load_labels(labels_path: Path) -> dict[int, str]:
    with labels_path.open("r") as f:
        raw = json.load(f)
    return {int(k): str(v) for k, v in raw.items()}


def _top_k_sae_features(
    z: torch.Tensor, labels: dict[int, str], top_k: int,
) -> list[list[tuple[str, float]]]:
    out: list[list[tuple[str, float]]] = []
    z_np = z.detach().cpu().numpy()
    for row in z_np:
        nz = np.nonzero(row)[0]
        if nz.size == 0:
            out.append([])
            continue
        nz_acts = row[nz]
        order = np.argsort(nz_acts)[::-1][:top_k]
        top_idx = nz[order]
        top_act = nz_acts[order]
        out.append([
            (labels.get(int(i), f"f{int(i)}"), float(a))
            for i, a in zip(top_idx, top_act, strict=True)
        ])
    return out


@app.command()
def main(
    h5_path: Path = typer.Option(
        Path("data/materials_project_v1/specimens.h5"), "--h5-path",
    ),
    embeddings_dir: Path | None = typer.Option(
        None, "--embeddings-dir",
        help="Cached CHGNet embeddings dir. Default: latest under "
             "runs/materials/embeddings/.",
    ),
    probe_bank_dir: Path | None = typer.Option(
        None, "--probe-bank-dir",
        help="Probe bank dir. Default: latest under "
             "checkpoints/materials/probes/.",
    ),
    sae_dir: Path | None = typer.Option(
        None, "--sae-dir",
        help="Trained SAE dir. Default: latest under "
             "checkpoints/materials/sae/.",
    ),
    sae_labels_path: Path | None = typer.Option(
        None, "--sae-labels-path",
        help="labels.json for the SAE. Default: latest under "
             "runs/materials/sae_labels/.",
    ),
    top_k_features: int = typer.Option(
        8, "--top-k-features",
        help="How many top-active SAE features to surface per specimen.",
    ),
    n_specimens: int = typer.Option(
        10000, "--n-specimens",
        help="Number of training specimens to emit records for. The "
             "training pool is the embeddings cache (which excludes "
             "the held-out 200 by default).",
    ),
    out: Path = typer.Option(
        Path("runs/materials/cot_datasets_sae"), "--out", "-o",
    ),
    batch_size: int = typer.Option(256, "--batch-size"),
    seed: int = typer.Option(0, "--seed"),
    device: str = typer.Option("auto", "--device"),
) -> None:
    """Build the materials SAE-augmented synthetic SFT dataset."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(seed)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
    from fmllm.materials.ground_truth import truth_dict  # noqa: PLC0415
    from fmllm.materials.synthetic_cot import build_sft_record  # noqa: PLC0415
    from fmllm.training.probe_bank import ProbeBank  # noqa: PLC0415

    if embeddings_dir is None:
        embeddings_dir = _latest_dir(Path("runs/materials/embeddings"))
        if embeddings_dir is None:
            raise typer.BadParameter(
                "no embeddings under runs/materials/embeddings/. Run "
                "scripts/materials/03_encode.sh first."
            )
    if probe_bank_dir is None:
        probe_bank_dir = _latest_dir(Path("checkpoints/materials/probes"))
        if probe_bank_dir is None:
            raise typer.BadParameter(
                "no probe bank under checkpoints/materials/probes/."
            )
    if sae_dir is None:
        sae_dir = _latest_dir(Path("checkpoints/materials/sae"))
        if sae_dir is None:
            raise typer.BadParameter(
                "no SAE under checkpoints/materials/sae/."
            )
    sae_path = sae_dir / "sae.pt"
    if not sae_path.exists():
        raise typer.BadParameter(f"missing {sae_path}")
    if sae_labels_path is None:
        sae_labels_path = _latest_labels(Path("runs/materials/sae_labels"))
    if sae_labels_path is None or not sae_labels_path.exists():
        raise typer.BadParameter(
            "no labels.json under runs/materials/sae_labels/. Run "
            "scripts/materials/06_label_sae.sh first."
        )

    emb_path = embeddings_dir / "embeddings.npy"
    sid_path = embeddings_dir / "specimen_ids.npy"
    if not emb_path.exists() or not sid_path.exists():
        raise typer.BadParameter(f"missing embeddings under {embeddings_dir}")
    embeddings = np.load(emb_path).astype(np.float32)
    specimen_ids = np.load(sid_path).astype(np.int64)

    # Subsample the training pool deterministically.
    n_pool = int(specimen_ids.shape[0])
    n_take = min(max(n_specimens, 1), n_pool)
    perm = np.random.permutation(n_pool)[:n_take]
    take_emb = embeddings[perm]
    take_sids = specimen_ids[perm]

    run_id = _generate_run_id()
    out_dir = out / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    typer.echo("==> Materials port Stage 7: build CoT records")
    typer.echo(f"    embeddings_dir : {embeddings_dir}")
    typer.echo(f"    probe_bank_dir : {probe_bank_dir}")
    typer.echo(f"    sae_dir        : {sae_dir}")
    typer.echo(f"    sae_labels     : {sae_labels_path}")
    typer.echo(f"    n_pool         : {n_pool}")
    typer.echo(f"    n_take         : {n_take}")
    typer.echo(f"    top_k_features : {top_k_features}")
    typer.echo(f"    out_dir        : {out_dir}")
    typer.echo("")

    bank = ProbeBank.load(probe_bank_dir, device=device).eval()
    typer.echo(f"    probes loaded  : {bank.names()}")

    sae, cls_mean, cls_std = _load_sae(sae_path, device=device)
    typer.echo(
        f"    SAE config     : in_dim={sae.in_dim} "
        f"hidden_dim={sae.hidden_dim} k={sae.k}"
    )
    labels = _load_labels(sae_labels_path)

    jsonl_path = out_dir / "records.jsonl"
    n_written = 0
    n_consistent = 0
    n_with_sae = 0
    with h5py.File(h5_path, "r") as h5, jsonl_path.open("w") as out_f:
        for start in range(0, n_take, batch_size):
            batch_emb_np = take_emb[start : start + batch_size]
            batch_sids = take_sids[start : start + batch_size]
            x = torch.from_numpy(batch_emb_np).to(device)
            with torch.no_grad():
                x_norm = (x - cls_mean) / cls_std.clamp_min(1.0e-6)
                z = sae.encode(x_norm)
            probe_outputs_batch = bank.evaluate(x)
            sae_features_batch = _top_k_sae_features(
                z, labels=labels, top_k=top_k_features,
            )
            for sid, probe_out, sae_feat in zip(
                batch_sids, probe_outputs_batch, sae_features_batch,
                strict=True,
            ):
                truth = truth_dict(h5, int(sid))
                record = build_sft_record(
                    probe_outputs=probe_out,
                    ground_truth=truth,
                    specimen_id=int(sid),
                    sae_features=sae_feat,
                )
                if record["cot_consistent"]:
                    n_consistent += 1
                if record["sae_features_count"] > 0:
                    n_with_sae += 1
                out_f.write(json.dumps(record) + "\n")
                n_written += 1
            if (start // batch_size) % 10 == 0:
                typer.echo(
                    f"    wrote {n_written}/{n_take} records "
                    f"(consistent={n_consistent} with-SAE={n_with_sae})"
                )

    typer.echo(f"==> JSONL written : {jsonl_path} ({n_written} records)")
    typer.echo(
        f"    cot-consistent : {n_consistent} "
        f"({100.0 * n_consistent / max(n_written, 1):.1f}%)"
    )
    typer.echo(
        f"    with SAE feats : {n_with_sae} "
        f"({100.0 * n_with_sae / max(n_written, 1):.1f}%)"
    )

    with (out_dir / "manifest.yaml").open("w") as f:
        yaml.safe_dump(
            {
                "run_id": run_id,
                "completed_utc": datetime.now(UTC).isoformat(),
                "h5_path": str(h5_path),
                "embeddings_dir": str(embeddings_dir),
                "probe_bank_dir": str(probe_bank_dir),
                "sae_dir": str(sae_dir),
                "sae_labels_path": str(sae_labels_path),
                "n_pool": n_pool,
                "n_records": n_written,
                "n_consistent": n_consistent,
                "n_with_sae_features": n_with_sae,
                "top_k_features": top_k_features,
                "batch_size": batch_size,
                "seed": seed,
            },
            f,
            sort_keys=False,
        )


if __name__ == "__main__":
    app()
