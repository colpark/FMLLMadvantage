"""CLI: spot-check a trained FM2 connector.

Diagnostic A from the Phase 9.A audit. Loads the saved connector,
runs Qwen generation on a handful of specimens with the connector
tokens prepended, and prints (ground-truth annotation, generated
description) side-by-side. Also runs each specimen a second time
with the FM features zeroed, so the user can see what the LLM
produces from prompt prior alone.

Reading guide:

    GROUND TRUTH       — what the templated annotation says about
                          the specimen.
    GENERATED (real FM)— what Qwen produces with the connector
                          conditioned on the actual FM2 features.
    GENERATED (zero FM)— what Qwen produces with the connector
                          fed an all-zero feature tensor. Acts as
                          a prior-only baseline.

If the real-FM generation matches ground truth on specimen-specific
facts (atom count, motif, temperature) and the zero-FM generation
collapses to a generic description, the connector is genuinely
conditioning on FM2's representation. If both look identical, the
connector is decorative.

Usage:
    bash scripts/inspect_connector.sh
    uv run python scripts/inspect_connector.py --n-specimens 8

Depends on:
    typer, torch, transformers (lazy), h5py.
"""

from __future__ import annotations

import sys
from pathlib import Path

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


app = typer.Typer(add_completion=False, no_args_is_help=False)


def _build_chat_prompt(tokenizer) -> torch.Tensor:
    """Reproduce the same prompt the trainer used (so the LLM sees the
    same scaffolding the connector aligned against)."""
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
    return enc["input_ids"]


def _generate(
    *,
    llm,
    tokenizer,
    inputs_embeds: torch.Tensor,
    max_new_tokens: int,
) -> str:
    attn = torch.ones(
        inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device,
    )
    with torch.no_grad():
        out = llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
    # When inputs_embeds is used, transformers returns only the newly
    # generated token IDs (the embeds have no corresponding IDs).
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


@app.command()
def main(
    connector_path: Path | None = typer.Option(
        None, "--connector",
        help="Path to connector.pt. Default: latest under runs/connectors/.",
    ),
    h5_path: Path = typer.Option(
        Path("data/synthetic_lj_v1/specimens.h5"), "--h5-path",
    ),
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    n_specimens: int = typer.Option(
        5, "--n-specimens", "-n",
        help="How many specimens to inspect.",
    ),
    specimen_ids: str = typer.Option(
        "", "--specimen-ids",
        help="Optional comma-separated list of explicit IDs. Overrides "
             "--n-specimens when supplied.",
    ),
    max_new_tokens: int = typer.Option(64, "--max-new-tokens"),
    device: str = typer.Option("auto", "--device"),
) -> None:
    """Run generation diagnostics on a trained FM2 connector."""
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    cfg = load_config(config)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Locate the connector.
    if connector_path is None:
        candidates = sorted(
            Path("runs/connectors").glob("*/connector.pt"),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        if not candidates:
            raise typer.BadParameter(
                "no connector.pt under runs/connectors/. "
                "Run scripts/train_fm2_connector.sh first."
            )
        connector_path = candidates[0]
    typer.echo(f"==> Connector : {connector_path}")

    payload = torch.load(connector_path, map_location=device, weights_only=False)
    fm2_ckpt = Path(payload["fm2_checkpoint"]) / "model.pt"
    llm_model_name = payload["llm_model"]

    # FM2 (frozen).
    fm2 = build_fm2_model(cfg.fm2).to(device)
    load_checkpoint(fm2_ckpt, model=fm2, map_location=device)
    fm2.eval()
    for p in fm2.parameters():
        p.requires_grad = False
    typer.echo(f"==> FM2 ckpt  : {fm2_ckpt}")

    # LLM (frozen).
    typer.echo(f"==> Loading LLM: {llm_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    llm = AutoModelForCausalLM.from_pretrained(
        llm_model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device,
    )
    llm.eval()
    embedder = llm.get_input_embeddings()

    # Connector (frozen, loaded from disk).
    connector = FM2Connector(
        fm_dim=payload["fm_dim"],
        llm_dim=payload["llm_dim"],
        n_query=payload["n_query"],
        n_layers=payload.get("n_layers", 2),
        n_heads=payload.get("n_heads", 8),
    ).to(device)
    if device == "cuda":
        connector = connector.to(torch.bfloat16)
    connector.load_state_dict(payload["state_dict"], strict=True)
    connector.eval()

    # Prompt.
    prompt_ids = _build_chat_prompt(tokenizer).to(device)
    prompt_emb = embedder(prompt_ids)

    # Resolve which specimens to inspect.
    if specimen_ids.strip():
        ids = [int(x.strip()) for x in specimen_ids.split(",") if x.strip()]
    else:
        ids = list(range(n_specimens))

    typer.echo(f"==> Inspecting {len(ids)} specimens: {ids[:8]}"
               f"{'...' if len(ids) > 8 else ''}")
    typer.echo("")

    import h5py  # noqa: PLC0415

    with h5py.File(h5_path, "r") as f:
        rdfs_np = np.stack(
            [np.asarray(f["rdfs"][i]) for i in ids], axis=0,
        ).astype(np.float32)

    rdfs = torch.from_numpy(rdfs_np).to(device).float()

    with torch.no_grad():
        features = fm2.encode(rdfs)                      # (B, T, fm_dim)
        if device == "cuda":
            features = features.to(torch.bfloat16)
        zero_features = torch.zeros_like(features)
        tokens_real = connector(features)                # (B, Q, D)
        tokens_zero = connector(zero_features)

    for i, sid in enumerate(ids):
        ann = annotate_specimen_from_h5(h5_path, sid, use_positions=True)

        emb_real = torch.cat(
            [tokens_real[i:i + 1], prompt_emb], dim=1,
        )
        emb_zero = torch.cat(
            [tokens_zero[i:i + 1], prompt_emb], dim=1,
        )
        gen_real = _generate(
            llm=llm, tokenizer=tokenizer,
            inputs_embeds=emb_real, max_new_tokens=max_new_tokens,
        )
        gen_zero = _generate(
            llm=llm, tokenizer=tokenizer,
            inputs_embeds=emb_zero, max_new_tokens=max_new_tokens,
        )

        typer.echo(f"--- specimen {sid} ---")
        typer.echo(f"  GROUND TRUTH       : {ann.text}")
        typer.echo(f"  GENERATED (real FM): {gen_real}")
        typer.echo(f"  GENERATED (zero FM): {gen_zero}")
        typer.echo("")


if __name__ == "__main__":
    app()
