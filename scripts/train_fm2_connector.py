"""CLI: Stage 1 alignment training for the FM2 connector.

Trains the Q-Former + projection (Layer C, Phase 9.A) so that its
output tokens, prepended into the orchestrator LLM's input embedding
stream, let the LLM produce specimen-faithful text descriptions.

Architecture:

    rdf ──► FM2.encode (frozen) ──► (B, 201, 320)
                                         │
                                         ▼
                                    FM2Connector ──► (B, n_query, llm_dim)
                                         │
                                         ▼
              prepend to LLM input embeds, before tokenized prompt+text
                                         │
                                         ▼
                                  Qwen 2.5 7B (frozen)
                                         │
                                         ▼
                                LM loss on text-portion tokens

Only the connector receives gradient updates. The FM2 backbone stays
frozen (we don't want to disturb the energy-head calibration). The
LLM stays frozen (Stage 1 is alignment, not task tuning; Stage 2
optionally LoRA-fine-tunes the LLM).

Output:
    runs/connectors/<run_id>/connector.pt
    runs/connectors/<run_id>/manifest.yaml
    runs/connectors/<run_id>/training.yaml

Usage:
    uv run python scripts/train_fm2_connector.py
    uv run python scripts/train_fm2_connector.py --epochs 5 --batch-size 16

Depends on:
    typer, torch, transformers (lazy), h5py, pyyaml.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import typer
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fmllm.connectors import FM2Connector  # noqa: E402
from fmllm.connectors.text_annotations import (  # noqa: E402
    annotate_specimen_from_h5,
)
from fmllm.fms.common import load_checkpoint  # noqa: E402
from fmllm.fms.fm2_rdf.model import build_fm2_model  # noqa: E402
from fmllm.utils.config import load_config  # noqa: E402
from fmllm.utils.manifests import write_manifest  # noqa: E402
from fmllm.utils.run_ids import generate_run_id  # noqa: E402


app = typer.Typer(add_completion=False, no_args_is_help=False)


# ---------------------------------------------------------------------------
# Dataset: pairs (rdf, text annotation, ground-truth metadata)
# ---------------------------------------------------------------------------


class _PairsDataset(torch.utils.data.Dataset):
    """In-memory cache of (rdf tensor, text annotation, specimen_id).

    The corpus is a few thousand specimens; loading everything once
    avoids repeated HDF5 random-access during training.
    """

    def __init__(
        self,
        *,
        h5_path: Path,
        specimen_ids: list[int],
    ) -> None:
        super().__init__()
        import h5py  # noqa: PLC0415

        self.specimen_ids = list(specimen_ids)
        with h5py.File(h5_path, "r") as f:
            self.rdfs = np.stack(
                [np.asarray(f["rdfs"][i]) for i in self.specimen_ids],
                axis=0,
            ).astype(np.float32)
        self.texts: list[str] = []
        for sid in self.specimen_ids:
            ann = annotate_specimen_from_h5(h5_path, sid, use_positions=True)
            self.texts.append(ann.text)

    def __len__(self) -> int:
        return len(self.specimen_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "specimen_id": self.specimen_ids[idx],
            "rdf": torch.from_numpy(self.rdfs[idx]),
            "text": self.texts[idx],
        }


# ---------------------------------------------------------------------------
# Loss: LM loss on the text portion of an input embed sequence with
# connector tokens prepended.
# ---------------------------------------------------------------------------


def _build_inputs(
    *,
    connector_tokens: torch.Tensor,           # (B, Q, D)
    prompt_ids: torch.Tensor,                 # (B, Lp) shared across batch
    target_ids: torch.Tensor,                 # (B, Lt)
    pad_id: int,
    embedder: torch.nn.Module,                # LLM input embedding layer
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the LLM input embedding sequence and the corresponding
    label sequence for cross-entropy.

    Layout (per row):
        [connector tokens, prompt tokens, target tokens]
    Labels:
        [-100 for connector, -100 for prompt, target_ids for target]
    """
    B = connector_tokens.shape[0]
    Q = connector_tokens.shape[1]
    Lp = prompt_ids.shape[1]
    Lt = target_ids.shape[1]
    device = connector_tokens.device

    prompt_emb = embedder(prompt_ids)              # (B, Lp, D)
    target_emb = embedder(target_ids)              # (B, Lt, D)
    inputs_embeds = torch.cat(
        [connector_tokens, prompt_emb, target_emb], dim=1,
    )                                              # (B, Q+Lp+Lt, D)

    # Labels: -100 (ignore) for connector + prompt, target_ids for target.
    ignore = torch.full((B, Q + Lp), -100, dtype=torch.long, device=device)
    labels = torch.cat([ignore, target_ids.clone()], dim=1)
    # Optionally mask pad targets.
    labels = labels.masked_fill(labels == pad_id, -100)
    return inputs_embeds, labels


def _tokenize_targets(
    *,
    tokenizer,
    texts: list[str],
    max_target_length: int,
) -> torch.Tensor:
    """Tokenize the assistant text portion. Uses left-padding so that
    every row's target ends with EOS at a known position; for LM loss
    we mask pad in :func:`_build_inputs`."""
    enc = tokenizer(
        [t + tokenizer.eos_token for t in texts],
        padding="max_length",
        truncation=True,
        max_length=max_target_length,
        return_tensors="pt",
    )
    return enc["input_ids"]


def _build_chat_prompt(tokenizer) -> torch.Tensor:
    """Construct the shared per-batch prompt that follows the connector
    tokens. Encoded once and broadcast across the batch."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a concise scientific assistant describing a single "
                "Lennard-Jones cluster specimen given the FM2 evidence "
                "tokens prepended to your input. Reply with one or two "
                "factual sentences."
            ),
        },
        {
            "role": "user",
            "content": (
                "Describe this specimen briefly. Mention atom count, "
                "motif, temperature regime, and any notable RDF features."
            ),
        },
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    return enc["input_ids"]                         # (1, Lp)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


@app.command()
def main(
    h5_path: Path = typer.Option(
        Path("data/synthetic_lj_v1/specimens.h5"), "--h5-path",
    ),
    splits_path: Path = typer.Option(
        Path("data/synthetic_lj_v1/splits.yaml"), "--splits-path",
    ),
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    checkpoint_root: Path = typer.Option(
        Path("checkpoints"), "--checkpoint-root",
    ),
    train_split: str = typer.Option("train_50k", "--train-split"),
    n_specimens: int = typer.Option(
        2000, "--n-specimens",
        help="Number of training specimens (drawn from the start of the "
             "split). 2K is enough for the alignment objective.",
    ),
    llm_model: str = typer.Option(
        "Qwen/Qwen2.5-7B-Instruct", "--llm-model",
    ),
    n_query: int = typer.Option(32, "--n-query"),
    n_layers: int = typer.Option(2, "--n-layers"),
    n_heads: int = typer.Option(8, "--n-heads"),
    epochs: int = typer.Option(3, "--epochs"),
    batch_size: int = typer.Option(8, "--batch-size"),
    lr: float = typer.Option(1.0e-4, "--lr"),
    weight_decay: float = typer.Option(0.0, "--weight-decay"),
    grad_accum: int = typer.Option(1, "--grad-accum"),
    max_target_length: int = typer.Option(96, "--max-target-length"),
    seed: int = typer.Option(0, "--seed"),
    out: Path = typer.Option(Path("runs/connectors"), "--out", "-o"),
    device: str = typer.Option("auto", "--device"),
    log_every: int = typer.Option(20, "--log-every"),
) -> None:
    """Train the FM2 connector (Stage 1 alignment)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    cfg = load_config(config)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)

    run_id = generate_run_id("fm2-connector-stage1")
    out_dir = out / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"==> Run id    : {run_id}")
    typer.echo(f"==> Output    : {out_dir}")

    # FM2 (frozen) ----------------------------------------------------------
    ckpt_dir = sorted(
        (checkpoint_root / "fm2_rdf" / train_split).glob("*"),
        key=lambda p: p.name,
        reverse=True,
    )
    if not ckpt_dir:
        raise typer.BadParameter(
            f"no FM2 checkpoint under "
            f"{checkpoint_root}/fm2_rdf/{train_split}/"
        )
    fm2 = build_fm2_model(cfg.fm2).to(device)
    load_checkpoint(ckpt_dir[0] / "model.pt", model=fm2, map_location=device)
    fm2.eval()
    for p in fm2.parameters():
        p.requires_grad = False
    typer.echo(f"==> FM2 ckpt  : {ckpt_dir[0]}")

    # LLM (frozen) ----------------------------------------------------------
    typer.echo(f"==> Loading LLM: {llm_model}")
    tokenizer = AutoTokenizer.from_pretrained(llm_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    llm = AutoModelForCausalLM.from_pretrained(
        llm_model,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device,
    )
    llm.eval()
    for p in llm.parameters():
        p.requires_grad = False
    embedder = llm.get_input_embeddings()
    llm_dim = int(embedder.embedding_dim)
    typer.echo(f"==> LLM hidden dim: {llm_dim}")

    # Connector (trainable) ------------------------------------------------
    connector = FM2Connector(
        fm_dim=getattr(fm2, "encoder", None) and fm2.cls_token.shape[-1] or 320,
        llm_dim=llm_dim,
        n_query=n_query,
        n_layers=n_layers,
        n_heads=n_heads,
    ).to(device)
    if device == "cuda":
        connector = connector.to(torch.bfloat16)
    typer.echo(
        f"==> Connector     : Q-Former (n_query={n_query}, n_layers={n_layers}), "
        f"params={connector.num_parameters():,}"
    )

    optimizer = torch.optim.AdamW(
        connector.parameters(), lr=lr, weight_decay=weight_decay,
    )

    # Dataset ---------------------------------------------------------------
    with splits_path.open("r") as f:
        splits = yaml.safe_load(f)
    pool: list[int] = []
    if train_split in (splits.get("train_subsets") or {}):
        pool = list(splits["train_subsets"][train_split])
    else:
        pool = list(splits.get("train", []))
    if not pool:
        raise typer.BadParameter(f"empty pool for split {train_split!r}")
    pool = pool[: max(n_specimens, 1)]
    typer.echo(f"==> Training specimens: {len(pool)} from {train_split}")

    dataset = _PairsDataset(h5_path=h5_path, specimen_ids=pool)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0,
        collate_fn=lambda batch: {
            "specimen_id": [b["specimen_id"] for b in batch],
            "rdf": torch.stack([b["rdf"] for b in batch], dim=0),
            "text": [b["text"] for b in batch],
        },
    )

    prompt_ids = _build_chat_prompt(tokenizer).to(device)   # (1, Lp)
    typer.echo(f"==> Prompt length : {prompt_ids.shape[1]} tokens")

    # Train loop ------------------------------------------------------------
    typer.echo("")
    typer.echo("==> Training")
    typer.echo("-" * 64)
    history: list[dict[str, float]] = []
    step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(epochs):
        for batch_idx, batch in enumerate(loader):
            rdfs = batch["rdf"].to(device).float()
            target_ids = _tokenize_targets(
                tokenizer=tokenizer, texts=batch["text"],
                max_target_length=max_target_length,
            ).to(device)
            prompt_ids_b = prompt_ids.expand(rdfs.shape[0], -1)

            with torch.no_grad():
                fm_features = fm2.encode(rdfs)
                if device == "cuda":
                    fm_features = fm_features.to(torch.bfloat16)

            connector_tokens = connector(fm_features)
            inputs_embeds, labels = _build_inputs(
                connector_tokens=connector_tokens,
                prompt_ids=prompt_ids_b,
                target_ids=target_ids,
                pad_id=tokenizer.pad_token_id,
                embedder=embedder,
            )
            outputs = llm(
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=False,
            )
            loss = outputs.loss / max(grad_accum, 1)
            loss.backward()

            if (step + 1) % max(grad_accum, 1) == 0:
                torch.nn.utils.clip_grad_norm_(connector.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            step += 1
            if step % log_every == 0 or step == 1:
                lv = float(loss.item() * max(grad_accum, 1))
                history.append(
                    {"step": step, "epoch": epoch, "loss": lv}
                )
                typer.echo(
                    f"  epoch={epoch} step={step:>5} "
                    f"batch={batch_idx:>4}/{len(loader)} loss={lv:.4f}"
                )
    typer.echo("-" * 64)

    # Save the connector ---------------------------------------------------
    ckpt_path = out_dir / "connector.pt"
    torch.save(
        {
            "state_dict": {
                k: v.detach().cpu()
                for k, v in connector.state_dict().items()
            },
            "fm_dim": connector.fm_dim,
            "llm_dim": connector.llm_dim,
            "n_query": connector.n_query,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "fm2_checkpoint": str(ckpt_dir[0]),
            "llm_model": llm_model,
            "stage": 1,
        },
        ckpt_path,
    )
    typer.echo(f"==> Saved connector: {ckpt_path}")

    with (out_dir / "training.yaml").open("w") as f:
        yaml.safe_dump(
            {
                "run_id": run_id,
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "history": history,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
            },
            f,
            sort_keys=False,
        )

    write_manifest(
        out_dir / "manifest.yaml",
        script="scripts.train_fm2_connector",
        inputs={
            "h5_path": str(h5_path),
            "splits_path": str(splits_path),
            "checkpoint_root": str(checkpoint_root),
            "train_split": train_split,
            "fm2_checkpoint": str(ckpt_dir[0]),
            "llm_model": llm_model,
        },
        config={
            "n_query": n_query,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "grad_accum": grad_accum,
            "max_target_length": max_target_length,
            "seed": seed,
        },
        extra={
            "n_train_specimens": len(pool),
            "final_loss": history[-1]["loss"] if history else None,
            "connector_params": connector.num_parameters(),
        },
    )


if __name__ == "__main__":
    app()
