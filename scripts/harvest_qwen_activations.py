"""CLI: harvest Qwen residual-stream activations for Phase 15 SAE training.

Phase 15 Stage A. For each trajectory in a prior baseline run,
reconstruct a minimal chat ``[system, user, assistant=final_claim]``,
forward it through Qwen with a hook on a target transformer layer,
and capture the residual-stream activation at the last token (where
Qwen would have finished emitting its commit JSON). One activation
vector per trajectory plus per-specimen metadata.

Why a minimal chat rather than the full OHVD trace: the goal is to
collect activations that represent Qwen's *commit state* in a
controlled, comparable way across specimens. Reconstructing the full
chat (with all interleaved tool messages) would inflate token counts
unevenly and mix many decision points into one sample. The minimal
chat is uniform across trajectories and isolates the commit position.

Output:

    runs/qwen_activations/<run_id>/activations.npy   # (N, hidden_dim) fp32
    runs/qwen_activations/<run_id>/metadata.yaml     # per-row labels
    runs/qwen_activations/<run_id>/manifest.yaml

Usage:

    bash scripts/harvest_qwen_activations.sh

Environment-equivalent flags:

    --trajectories <path>   trajectories.jsonl to replay
    --layer-path <str>      e.g. ``model.layers.14`` (Qwen 2.5 7B has 28)
    --base-model <hf-id>    Qwen/Qwen2.5-7B-Instruct
    --quantize {none,4bit}  bf16 needs ~14GB; 4bit needs ~6GB

Depends on:
    typer, torch, transformers, h5py, pyyaml, peft (for adapter).
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fmllm.connectors.text_annotations import _phase_for  # noqa: E402
from fmllm.orchestrator import DEFAULT_SYSTEM_PROMPT  # noqa: E402
from fmllm.representation.llm_sae import (  # noqa: E402
    ActivationHarvester,
    resolve_layer_module,
)
from fmllm.utils.manifests import write_manifest  # noqa: E402
from fmllm.utils.run_ids import generate_run_id  # noqa: E402


app = typer.Typer(add_completion=False, no_args_is_help=False)


def _latest_trajectories(root: Path, pattern: str) -> Path | None:
    cands = sorted(
        root.glob(f"{pattern}/*/trajectories.jsonl"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    return cands[0] if cands else None


def _load_trajectories(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _ground_truth(h5_path: Path, sid: int) -> dict:
    """Load motif / atom_count / temperature for one specimen."""
    with h5py.File(h5_path, "r") as f:
        motif_names: list[str] = []
        if "motif_names" in f.attrs:
            motif_names = [
                s.decode() if isinstance(s, bytes) else str(s)
                for s in f.attrs["motif_names"]
            ]
        mid = int(np.asarray(f["motif_ids"][sid]))
        n = int(np.asarray(f["atom_counts"][sid]))
        t = float(np.asarray(f["temperatures"][sid]))
        motif = motif_names[mid] if 0 <= mid < len(motif_names) else str(mid)
        return {"motif": motif, "n_atoms": n, "temperature": t,
                "phase": _phase_for(t)}


def _is_correct(claim: dict, gt: dict) -> bool:
    """Compare a parsed final_claim to ground truth.

    Correctness is the conjunction of: motif matches, n_atoms exact,
    |temperature_pred - temperature_gt| <= 0.10. Mirrors the goal-
    accuracy scorer's logic without depending on its full machinery.
    """
    if not claim:
        return False
    motif_ok = str(claim.get("motif", "")).strip().lower() == gt["motif"].lower()
    try:
        n_pred = int(claim.get("n_atoms", -1))
    except (TypeError, ValueError):
        n_pred = -1
    n_ok = n_pred == gt["n_atoms"]
    try:
        t_pred = float(claim.get("temperature", -999.0))
    except (TypeError, ValueError):
        t_pred = -999.0
    t_ok = abs(t_pred - gt["temperature"]) <= 0.10
    return motif_ok and n_ok and t_ok


def _build_chat(query: str, final_claim: dict | None) -> list[dict]:
    """Minimal chat that captures the commit state.

    The user message is the recorded enriched query. The assistant
    message is the final_claim rendered as the JSON object the
    model would have emitted at commit. Tokenizing this produces a
    sequence whose last assistant tokens are exactly the closing
    structure of the commit JSON.
    """
    assistant_text = (
        json.dumps(final_claim, sort_keys=True)
        if final_claim is not None
        else "{}"
    )
    return [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": query},
        {"role": "assistant", "content": assistant_text},
    ]


@app.command()
def main(
    trajectories: Path | None = typer.Option(
        None, "--trajectories",
        help="Path to a trajectories.jsonl. Default: latest under "
             "runs/holdout/full/.",
    ),
    h5_path: Path = typer.Option(
        Path("data/synthetic_lj_v1/specimens.h5"), "--h5-path",
    ),
    base_model: str = typer.Option(
        "Qwen/Qwen2.5-7B-Instruct", "--base-model",
    ),
    adapter_path: Path | None = typer.Option(
        None, "--adapter-path",
        help="Optional LoRA adapter to stack on the base LLM (e.g. the "
             "Phase 11 cot_sft adapter).",
    ),
    layer_path: str = typer.Option(
        "model.layers.14", "--layer-path",
        help="Dotted path to the transformer layer to hook. For "
             "Qwen 2.5 7B (28 layers), 14 is the middle of the "
             "residual stream where Templeton et al. trained Golden "
             "Gate Claude's SAE.",
    ),
    quantize: str = typer.Option(
        "4bit", "--quantize",
        help="'none' (bf16) or '4bit' (BitsAndBytes nf4 + double-quant).",
    ),
    max_tokens: int = typer.Option(
        2048, "--max-tokens",
        help="Truncate chats whose tokenized form exceeds this length.",
    ),
    out: Path = typer.Option(
        Path("runs/qwen_activations"), "--out", "-o",
    ),
    device: str = typer.Option("auto", "--device"),
    log_every: int = typer.Option(20, "--log-every"),
) -> None:
    """Harvest Qwen residual-stream activations from a baseline run."""
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if trajectories is None:
        cand = _latest_trajectories(Path("runs/holdout"), "full")
        if cand is None:
            cand = _latest_trajectories(Path("runs/baselines"), "full")
        if cand is None:
            raise typer.BadParameter(
                "no trajectories.jsonl under runs/holdout/full/ or "
                "runs/baselines/full/. Pass --trajectories explicitly."
            )
        trajectories = cand

    run_id = generate_run_id("qwen-activations")
    out_dir = out / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"==> Run id      : {run_id}")
    typer.echo(f"==> Output      : {out_dir}")
    typer.echo(f"==> Source      : {trajectories}")
    typer.echo(f"==> Base model  : {base_model}")
    typer.echo(f"==> Adapter     : {adapter_path or '(none)'}")
    typer.echo(f"==> Layer hook  : {layer_path}")
    typer.echo(f"==> Quantize    : {quantize}")

    # Load model -----------------------------------------------------------
    quant_kwargs: dict = {}
    if quantize == "4bit":
        from transformers import BitsAndBytesConfig  # noqa: PLC0415

        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantize != "none":
        raise typer.BadParameter(f"unknown quantize={quantize!r}")

    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if quantize == "none" else None,
        device_map=device if device != "cpu" else None,
        **quant_kwargs,
    )
    if adapter_path is not None:
        from peft import PeftModel  # noqa: PLC0415

        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    base_for_hook = model.base_model.model if hasattr(model, "base_model") else model
    layer_module = resolve_layer_module(base_for_hook, layer_path)
    typer.echo(f"==> Hooking     : {type(layer_module).__name__}")

    # Replay trajectories --------------------------------------------------
    trajs = _load_trajectories(trajectories)
    typer.echo(f"==> Trajectories: {len(trajs)}")
    if not trajs:
        raise typer.BadParameter(f"empty trajectory file: {trajectories}")

    activations_rows: list[np.ndarray] = []
    metadata_rows: list[dict] = []
    skipped = 0

    typer.echo("-" * 64)
    for i, traj in enumerate(trajs):
        sid = traj.get("specimen_id")
        query = traj.get("query")
        fc = traj.get("final_claim") or {}
        if sid is None or query is None:
            skipped += 1
            continue

        # Build prompt -------------------------------------------------------
        try:
            chat = _build_chat(query, fc)
            prompt = tok.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=False,
            )
            inputs = tok(
                prompt, return_tensors="pt", truncation=True,
                max_length=max_tokens,
            )
        except Exception as exc:
            typer.echo(f"  skip sid={sid}: tokenization failed ({exc!r})")
            skipped += 1
            continue

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Forward + harvest --------------------------------------------------
        with ActivationHarvester(layer_module) as harv, torch.no_grad():
            _ = model(**inputs)
            buf = harv.pop()                       # (T, hidden_dim) on cpu

        if buf.shape[0] == 0:
            typer.echo(f"  skip sid={sid}: empty activation buffer")
            skipped += 1
            continue

        # Take the LAST token; that's the closing position of the commit JSON.
        last = buf[-1].numpy().astype(np.float32)
        activations_rows.append(last)

        # Per-row metadata ---------------------------------------------------
        try:
            gt = _ground_truth(h5_path, int(sid))
        except Exception as exc:
            gt = {"motif": "?", "n_atoms": -1, "temperature": -1.0, "phase": "?"}
            typer.echo(f"  warn sid={sid}: gt lookup failed ({exc!r})")
        verdict = (traj.get("final_verdict") or {}).get(
            "aggregate_decision", "null",
        )
        metadata_rows.append({
            "row_idx": len(activations_rows) - 1,
            "specimen_id": int(sid),
            "verdict": verdict,
            "is_correct": _is_correct(fc, gt),
            "claim": fc,
            "ground_truth": gt,
            "n_tokens_in_chat": int(buf.shape[0]),
        })

        if (i + 1) % log_every == 0 or (i + 1) == len(trajs):
            typer.echo(
                f"  processed {i + 1}/{len(trajs)} | "
                f"kept={len(activations_rows)} skipped={skipped}"
            )
    typer.echo("-" * 64)

    if not activations_rows:
        raise typer.BadParameter("no activations harvested; aborting.")

    # Persist --------------------------------------------------------------
    arr = np.stack(activations_rows, axis=0)
    np.save(out_dir / "activations.npy", arr)
    typer.echo(f"==> Activations : {out_dir / 'activations.npy'} {arr.shape}")

    with (out_dir / "metadata.yaml").open("w") as f:
        yaml.safe_dump(
            {
                "rows": metadata_rows,
                "hidden_dim": int(arr.shape[1]),
                "n_rows": int(arr.shape[0]),
            },
            f,
            sort_keys=False,
        )
    typer.echo(f"==> Metadata    : {out_dir / 'metadata.yaml'}")

    write_manifest(
        out_dir / "manifest.yaml",
        script="scripts.harvest_qwen_activations",
        inputs={
            "trajectories": str(trajectories),
            "h5_path": str(h5_path),
            "base_model": base_model,
            "adapter_path": str(adapter_path) if adapter_path else None,
            "layer_path": layer_path,
        },
        config={
            "run_id": run_id,
            "quantize": quantize,
            "max_tokens": max_tokens,
            "started_utc": datetime.now(UTC).isoformat(),
        },
        extra={
            "n_rows": int(arr.shape[0]),
            "hidden_dim": int(arr.shape[1]),
            "n_skipped": int(skipped),
        },
    )


if __name__ == "__main__":
    app()
