"""CLI: pretrain FM2 with masked-RDF reconstruction (Phase 10 / Layer D).

The Phase 9 negative result showed that the supervised FM2 backbone
holds only selective task-extra signal and that a Layer C connector
trained on top of it does not transfer specimen identity to the LLM.
This script trains a parallel FM2 backbone with a self-supervised
objective: randomly mask a fraction of g(r) bins and predict the
masked values from context.

Key design points:

* **Same hyperparameters as supervised FM2.** ``cfg.fm2`` provides
  rdf_bins, embed_dim, depth, etc. The SSL backbone is structurally
  identical to ``FM2RDFTransformer`` aside from a learnable
  mask_token and a per-bin reconstruction head, so probes and the
  Q-Former connector consume it without changes.
* **Loss only on masked positions.** Standard masked-modeling
  recipe: zero loss on bins the encoder saw, MSE on bins it didn't.
* **No supervised signal.** Energy labels are not consulted. This
  is the whole point: tests whether the representation gets richer
  when freed from the per-atom-energy bottleneck.

Output:

    checkpoints/fm2_rdf_ssl/<train_split>/<run_id>/model.pt
    checkpoints/fm2_rdf_ssl/<train_split>/<run_id>/manifest.yaml
    checkpoints/fm2_rdf_ssl/<train_split>/<run_id>/training.yaml

Usage:

    bash scripts/train_fm2_ssl.sh
    uv run python scripts/train_fm2_ssl.py --epochs 20 --mask-ratio 0.30

Depends on:
    typer, torch, h5py, pyyaml.
"""

from __future__ import annotations

import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import typer
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fmllm.fms.common import save_checkpoint  # noqa: E402
from fmllm.fms.fm2_rdf_ssl.model import build_fm2_ssl_model  # noqa: E402
from fmllm.utils.config import load_config  # noqa: E402
from fmllm.utils.manifests import write_manifest  # noqa: E402
from fmllm.utils.run_ids import generate_run_id  # noqa: E402


app = typer.Typer(add_completion=False, no_args_is_help=False)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class _RDFDataset(torch.utils.data.Dataset):
    """In-memory RDFs for a fixed list of specimen IDs.

    50K float32 RDFs of length 200 ≈ 40 MB; comfortably fits in RAM.
    """

    def __init__(self, *, h5_path: Path, specimen_ids: list[int]) -> None:
        super().__init__()
        import h5py  # noqa: PLC0415

        self.specimen_ids = list(specimen_ids)
        with h5py.File(h5_path, "r") as f:
            self.rdfs = np.stack(
                [np.asarray(f["rdfs"][i]) for i in self.specimen_ids],
                axis=0,
            ).astype(np.float32)

    def __len__(self) -> int:
        return len(self.specimen_ids)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.rdfs[idx])


def _generate_mask(
    *,
    batch: int,
    bins: int,
    mask_ratio: float,
    device: torch.device | str,
    seed: int | None = None,
) -> torch.Tensor:
    """Independent random mask per row. ``mask[i, j] = True`` means
    bin ``j`` of row ``i`` is hidden from the encoder."""
    if seed is not None:
        torch.manual_seed(seed)
    n_mask = max(1, int(round(mask_ratio * bins)))
    # noise[i, j] in [0, 1); the lowest n_mask noise values per row
    # become masked.
    noise = torch.rand(batch, bins, device=device)
    threshold = torch.kthvalue(noise, n_mask, dim=-1, keepdim=True).values
    return noise <= threshold


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _load_split_ids(splits_path: Path, split_name: str) -> list[int]:
    with splits_path.open("r") as f:
        splits = yaml.safe_load(f)
    if split_name == "train":
        return [int(x) for x in splits.get("train", [])]
    subsets = splits.get("train_subsets") or {}
    if split_name in subsets:
        return [int(x) for x in subsets[split_name]]
    raise typer.BadParameter(f"unknown split {split_name!r}")


@app.command()
def main(
    h5_path: Path = typer.Option(
        Path("data/synthetic_lj_v1/specimens.h5"), "--h5-path",
    ),
    splits_path: Path = typer.Option(
        Path("data/synthetic_lj_v1/splits.yaml"), "--splits-path",
    ),
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    train_split: str = typer.Option("train_50k", "--train-split"),
    out: Path = typer.Option(Path("checkpoints/fm2_rdf_ssl"), "--out", "-o"),
    epochs: int = typer.Option(20, "--epochs"),
    batch_size: int = typer.Option(64, "--batch-size"),
    lr: float = typer.Option(1.0e-4, "--lr"),
    weight_decay: float = typer.Option(1.0e-2, "--weight-decay"),
    mask_ratio: float = typer.Option(0.30, "--mask-ratio"),
    grad_accum: int = typer.Option(1, "--grad-accum"),
    seed: int = typer.Option(0, "--seed"),
    log_every: int = typer.Option(50, "--log-every"),
    device: str = typer.Option("auto", "--device"),
) -> None:
    """Pretrain FM2 with masked-RDF reconstruction."""
    cfg = load_config(config)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)

    run_id = generate_run_id(f"fm2-ssl-{train_split}")
    out_dir = out / train_split / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"==> Run id    : {run_id}")
    typer.echo(f"==> Output    : {out_dir}")
    typer.echo(f"==> Train split: {train_split}")
    typer.echo(f"==> Mask ratio : {mask_ratio}")
    typer.echo(f"==> Device     : {device}")

    model = build_fm2_ssl_model(cfg.fm2).to(device)
    if device == "cuda":
        # bf16 for speed; weights stay in float32 internally where
        # PyTorch keeps them, but compute uses bf16 via autocast.
        pass
    typer.echo(
        f"==> Model      : FM2SSLTransformer "
        f"(rdf_bins={cfg.fm2.rdf_bins}, embed_dim={cfg.fm2.embed_dim}, "
        f"depth={cfg.fm2.depth})"
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    typer.echo(f"    trainable parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95),
    )

    specimen_ids = _load_split_ids(splits_path, train_split)
    if not specimen_ids:
        raise typer.BadParameter(f"split {train_split!r} is empty")
    typer.echo(f"==> Specimens  : {len(specimen_ids):,}")

    dataset = _RDFDataset(h5_path=h5_path, specimen_ids=specimen_ids)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, drop_last=False,
    )

    scaler = (
        torch.amp.GradScaler("cuda")
        if device == "cuda" and torch.cuda.is_bf16_supported() is False
        else None
    )
    use_amp = device == "cuda"

    typer.echo("")
    typer.echo("==> Pretraining")
    typer.echo("-" * 64)
    history: list[dict[str, float]] = []
    step = 0
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(epochs):
        for batch_idx, rdfs in enumerate(loader):
            rdfs = rdfs.to(device, non_blocking=True).float()
            B, bins = rdfs.shape
            mask = _generate_mask(
                batch=B, bins=bins, mask_ratio=mask_ratio,
                device=rdfs.device,
            )

            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    pred = model(rdfs, mask)
                    loss = torch.nn.functional.mse_loss(
                        pred[mask], rdfs[mask],
                    ) / max(grad_accum, 1)
            else:
                pred = model(rdfs, mask)
                loss = torch.nn.functional.mse_loss(
                    pred[mask], rdfs[mask],
                ) / max(grad_accum, 1)

            if scaler is not None:
                scaler.scale(loss).backward()
                if (step + 1) % max(grad_accum, 1) == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                loss.backward()
                if (step + 1) % max(grad_accum, 1) == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            step += 1
            if step == 1 or step % log_every == 0:
                lv = float(loss.item() * max(grad_accum, 1))
                history.append({"step": step, "epoch": epoch, "loss": lv})
                typer.echo(
                    f"  epoch={epoch:>3} step={step:>6} "
                    f"batch={batch_idx:>5}/{len(loader)} "
                    f"loss={lv:.6f}  elapsed={time.time() - t0:.1f}s"
                )

    typer.echo("-" * 64)

    # Save the checkpoint in the same payload shape used by the
    # supervised FM2 trainer so probes and the connector script can
    # use ``load_checkpoint`` unchanged.
    save_path = out_dir / "model.pt"
    save_checkpoint(
        save_path,
        model=model,
        optimizer=None,
        scheduler=None,
        epoch=epochs,
        extra={
            "kind": "fm2_rdf_ssl",
            "mask_ratio": mask_ratio,
            "lr": lr,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "n_specimens": len(specimen_ids),
        },
    )
    typer.echo(f"==> Saved      : {save_path}")

    with (out_dir / "training.yaml").open("w") as f:
        yaml.safe_dump(
            {
                "run_id": run_id,
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "train_split": train_split,
                "mask_ratio": mask_ratio,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "weight_decay": weight_decay,
                "history": history,
                "final_loss": history[-1]["loss"] if history else None,
                "wall_clock_seconds": time.time() - t0,
            },
            f,
            sort_keys=False,
        )

    write_manifest(
        out_dir / "manifest.yaml",
        script="scripts.train_fm2_ssl",
        inputs={
            "h5_path": str(h5_path),
            "splits_path": str(splits_path),
            "train_split": train_split,
        },
        config={
            "rdf_bins": cfg.fm2.rdf_bins,
            "embed_dim": cfg.fm2.embed_dim,
            "depth": cfg.fm2.depth,
            "num_heads": cfg.fm2.num_heads,
            "mlp_ratio": cfg.fm2.mlp_ratio,
            "dropout": float(getattr(cfg.fm2, "dropout", 0.0)),
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "mask_ratio": mask_ratio,
            "weight_decay": weight_decay,
            "grad_accum": grad_accum,
            "seed": seed,
        },
        extra={
            "objective": "masked-rdf-reconstruction",
            "n_train_specimens": len(specimen_ids),
            "trainable_parameters": int(n_params),
        },
    )


if __name__ == "__main__":
    app()
