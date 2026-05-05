"""CLI: spot-check a CoT-SFT adapter (Phase 11 Stage 2 deliverable).

Loads the latest adapter under ``checkpoints/cot-sft/``, computes
probe outputs for a handful of specimens, builds the user message
the trainer used, generates from Qwen with the adapter applied, and
prints (truth, generated CoT) side-by-side. Also runs each specimen
with all-zero FM features so the user can compare conditioned vs
prior-only generations.

Reading guide:

    GROUND TRUTH       — N, motif, T from the dataset HDF5.
    GENERATED (real)   — Qwen+adapter conditioned on the actual
                          probe outputs.
    GENERATED (zeroed) — Qwen+adapter conditioned on probe outputs
                          built from zero FM features (the connector
                          / probe bank still runs but on a degenerate
                          input). This is the "did training make the
                          model attend to the probe payload" diagnostic.

Usage:
    bash scripts/inspect_cot_adapter.sh
    uv run python scripts/inspect_cot_adapter.py --n-specimens 6

Depends on:
    typer, torch, transformers, peft (lazy), h5py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fmllm.connectors.text_annotations import annotate_specimen_from_h5  # noqa: E402
from fmllm.fms.common import load_checkpoint  # noqa: E402
from fmllm.fms.fm2_rdf.model import build_fm2_model  # noqa: E402
from fmllm.training.probe_bank import ProbeBank  # noqa: E402
from fmllm.training.synthetic_cot import _user_message  # noqa: E402
from fmllm.utils.config import load_config  # noqa: E402


app = typer.Typer(add_completion=False, no_args_is_help=False)


def _latest(parent: Path, pattern: str = "*") -> Path | None:
    cands = sorted(
        parent.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True,
    )
    return cands[0] if cands else None


def _latest_completed_dir(parent: Path) -> Path | None:
    cands = sorted(
        parent.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True,
    )
    return next((c for c in cands if c.is_dir()), None)


def _latest_fm2_ckpt(checkpoint_root: Path, train_split: str) -> Path:
    parent = checkpoint_root / "fm2_rdf" / train_split
    cands = sorted(parent.glob("*"), key=lambda p: p.name, reverse=True)
    cands = [c for c in cands if (c / "model.pt").exists()]
    if not cands:
        raise typer.BadParameter(f"no fm2_rdf checkpoint under {parent}")
    return cands[0]


@app.command()
def main(
    adapter_path: Path | None = typer.Option(
        None, "--adapter",
        help="Path to the LoRA adapter directory. Default: latest under "
             "checkpoints/cot-sft/.",
    ),
    probe_bank_dir: Path | None = typer.Option(
        None, "--probe-bank-dir",
        help="Path to the probe bank directory. Default: latest under "
             "checkpoints/probes/.",
    ),
    h5_path: Path = typer.Option(
        Path("data/synthetic_lj_v1/specimens.h5"), "--h5-path",
    ),
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    base_model: str = typer.Option(
        "Qwen/Qwen2.5-7B-Instruct", "--base-model",
    ),
    n_specimens: int = typer.Option(6, "--n-specimens", "-n"),
    specimen_ids: str = typer.Option(
        "", "--specimen-ids",
        help="Optional comma-separated list. Overrides --n-specimens.",
    ),
    max_new_tokens: int = typer.Option(256, "--max-new-tokens"),
    device: str = typer.Option("auto", "--device"),
) -> None:
    """Generate CoTs from the trained Phase 11 adapter and print them."""
    from peft import PeftModel  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    cfg = load_config(config)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if adapter_path is None:
        cot_root = Path("checkpoints/cot-sft")
        run_dir = _latest_completed_dir(cot_root)
        if run_dir is None or not (run_dir / "adapter").exists():
            raise typer.BadParameter(
                "no adapter under checkpoints/cot-sft/. Run "
                "scripts/train_cot_sft.sh first."
            )
        adapter_path = run_dir / "adapter"
    typer.echo(f"==> Adapter   : {adapter_path}")

    if probe_bank_dir is None:
        probe_bank_dir = _latest_completed_dir(Path("checkpoints/probes"))
        if probe_bank_dir is None:
            raise typer.BadParameter(
                "no probe bank under checkpoints/probes/. Run "
                "scripts/train_probe_bank.sh first."
            )
    typer.echo(f"==> Probe bank: {probe_bank_dir}")

    fm2_ckpt = _latest_fm2_ckpt(Path("checkpoints"), "train_50k")
    typer.echo(f"==> FM2 ckpt  : {fm2_ckpt}")

    fm2 = build_fm2_model(cfg.fm2).to(device)
    load_checkpoint(fm2_ckpt / "model.pt", model=fm2, map_location=device)
    fm2.eval()
    for p in fm2.parameters():
        p.requires_grad = False

    bank = ProbeBank.load(probe_bank_dir, device=device).eval()

    typer.echo(f"==> Loading LLM: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base_llm = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device,
    )
    llm = PeftModel.from_pretrained(base_llm, str(adapter_path), is_trainable=False)
    llm.eval()

    if specimen_ids.strip():
        ids = [int(x.strip()) for x in specimen_ids.split(",") if x.strip()]
    else:
        ids = list(range(n_specimens))
    typer.echo(f"==> Specimens : {ids}")
    typer.echo("")

    with h5py.File(h5_path, "r") as h5:
        rdfs_np = np.stack(
            [np.asarray(h5["rdfs"][i]) for i in ids], axis=0,
        ).astype(np.float32)
    rdfs = torch.from_numpy(rdfs_np).to(device).float()

    with torch.no_grad():
        hidden_real = fm2.encode(rdfs)
        cls_real = hidden_real[:, 0, :]
        cls_zero = torch.zeros_like(cls_real)
        probes_real = bank.evaluate(cls_real)
        probes_zero = bank.evaluate(cls_zero)

    for i, sid in enumerate(ids):
        ann = annotate_specimen_from_h5(h5_path, sid, use_positions=True)
        truth = {
            "n_atoms": ann.n_atoms,
            "motif": ann.motif,
            "temperature": round(ann.temperature, 2),
        }

        for label, probe_out in (
            ("real", probes_real[i]),
            ("zero", probes_zero[i]),
        ):
            user = _user_message(probe_out)
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a scientific reasoner working with a 2D "
                        "Lennard-Jones cluster testbed. You receive probe "
                        "outputs derived from a frozen foundation model "
                        "and must reason explicitly about the evidence "
                        "before committing a typed claim about the "
                        "specimen's atom count, motif, and temperature."
                    ),
                },
                {"role": "user", "content": user},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = llm.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gen_ids = out[0, inputs["input_ids"].shape[1] :]
            gen = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            if label == "real":
                typer.echo(f"--- specimen {sid} ---")
                typer.echo(f"  GROUND TRUTH       : {json.dumps(truth)}")
                # Compact probe summary for context.
                summary = {
                    name: {
                        "p": probe_out[name].get("prediction"),
                        "c": round(float(probe_out[name].get("confidence", 0.0)), 2),
                    }
                    for name in bank.names()
                }
                typer.echo(f"  PROBES (real)      : {json.dumps(summary)}")
                typer.echo(f"  GENERATED (real)   :")
                for line in gen.splitlines():
                    typer.echo(f"      {line}")
            else:
                typer.echo(f"  GENERATED (zero FM):")
                for line in gen.splitlines():
                    typer.echo(f"      {line}")
                typer.echo("")


if __name__ == "__main__":
    app()
