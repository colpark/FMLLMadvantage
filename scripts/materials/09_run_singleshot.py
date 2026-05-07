"""Stage 9: single-shot LLM inference on materials holdout.

Mirrors ``scripts/run_baseline_cot_sft_sae.py`` from the LJ pipeline
but for the materials port. For each held-out specimen:

  1. Forward CHGNet (live, no embeddings cache dependency).
  2. Run the materials probe bank on the pooled embedding.
  3. Run the SAE encoder, label the top-K active features.
  4. Build the user message via ``fmllm.materials.synthetic_cot._user_message``.
  5. Generate one CoT from the SFT-tuned LoRA + Qwen base.
  6. Parse "Final commit:" JSON; check joint correctness.

Writes a materials-shaped JSONL where each record carries the raw
generation, the parsed claim, the ground truth, the probe outputs,
and a precomputed ``is_correct`` boolean. No LJ ``Trajectory``
schema involved — evaluators consume the JSONL directly.

Output:

    runs/materials/holdout/cot_sft_sae/<run_id>/records.jsonl
    runs/materials/holdout/cot_sft_sae/<run_id>/summary.yaml
    runs/materials/holdout/cot_sft_sae/<run_id>/manifest.yaml

Usage:

    bash scripts/materials/09_run_singleshot.sh

Depends on:
    typer, h5py, numpy, torch, transformers, peft, pyyaml, chgnet,
    pymatgen.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import typer
import yaml


app = typer.Typer(add_completion=False, no_args_is_help=False)


_FINAL_RE = re.compile(r"Final commit:\s*(\{.*?\})", re.DOTALL)


def _now_utc() -> str:
    return datetime.now(UTC).isoformat()


def _generate_run_id(slug: str) -> str:
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


def _parse_final_commit(text: str) -> dict[str, Any] | None:
    """Extract the first balanced JSON object after 'Final commit:'."""
    idx = text.find("Final commit:")
    if idx < 0:
        return None
    sub = text[idx:]
    brace_start = sub.find("{")
    if brace_start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    end = -1
    for i in range(brace_start, len(sub)):
        c = sub[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        m = _FINAL_RE.search(text)
        if m is None:
            return None
        payload = m.group(1)
    else:
        payload = sub[brace_start:end]
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


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


def _top_k_features_for_row(
    z_row: np.ndarray, labels: dict[int, str], top_k: int,
) -> list[tuple[str, float]]:
    nz = np.nonzero(z_row)[0]
    if nz.size == 0:
        return []
    nz_acts = z_row[nz]
    order = np.argsort(nz_acts)[::-1][:top_k]
    return [
        (labels.get(int(nz[i]), f"f{int(nz[i])}"), float(nz_acts[i]))
        for i in order
    ]


@app.command()
def main(
    h5_path: Path = typer.Option(
        Path("data/materials_project_v1/specimens.h5"), "--h5-path",
    ),
    holdout_ids_path: Path = typer.Option(
        Path("data/materials_project_v1/holdout_lock/ids.json"),
        "--holdout-ids-path",
    ),
    adapter_path: Path | None = typer.Option(
        None, "--adapter-path",
        help="LoRA adapter from Stage 8. Default: latest under "
             "checkpoints/materials/cot-sft-sae/<run_id>/adapter/.",
    ),
    probe_bank_dir: Path | None = typer.Option(
        None, "--probe-bank-dir",
        help="Probe bank dir. Default: latest under "
             "checkpoints/materials/probes/.",
    ),
    sae_dir: Path | None = typer.Option(
        None, "--sae-dir",
        help="SAE dir. Default: latest under checkpoints/materials/sae/.",
    ),
    sae_labels_path: Path | None = typer.Option(
        None, "--sae-labels-path",
        help="labels.json. Default: latest under runs/materials/sae_labels/.",
    ),
    chgnet_model_name: str = typer.Option(
        "0.3.0", "--chgnet-model-name",
    ),
    max_atoms: int = typer.Option(80, "--max-atoms"),
    top_k_features: int = typer.Option(
        8, "--top-k-features",
    ),
    base_model: str = typer.Option(
        "Qwen/Qwen2.5-7B-Instruct", "--base-model",
    ),
    out: Path = typer.Option(
        Path("runs/materials/holdout"), "--out", "-o",
    ),
    max_new_tokens: int = typer.Option(768, "--max-new-tokens"),
    quantize: str = typer.Option(
        "4bit", "--quantize",
        help="'none' | '4bit' | '8bit'.",
    ),
    batch_size: int = typer.Option(
        16, "--batch-size",
        help="Number of specimens to process through the LLM in one "
             "batched generate() call. Qwen 7B 4-bit on H100 80GB fits "
             "BS=32 comfortably; 16 is conservative.",
    ),
    device: str = typer.Option("auto", "--device"),
    log_every: int = typer.Option(10, "--log-every"),
) -> None:
    """Single-shot inference with the materials CoT-SFT adapter."""
    from peft import PeftModel  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
    from fmllm.materials.chgnet_wrap import (  # noqa: PLC0415
        CHGNetWrap, structure_from_arrays,
    )
    from fmllm.materials.ground_truth import is_correct, truth_dict  # noqa: PLC0415
    from fmllm.materials.synthetic_cot import _SYSTEM_PROMPT, _user_message  # noqa: PLC0415
    from fmllm.training.probe_bank import ProbeBank  # noqa: PLC0415

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if adapter_path is None:
        run_dir = _latest_dir(Path("checkpoints/materials/cot-sft-sae"))
        if run_dir is None or not (run_dir / "adapter").exists():
            raise typer.BadParameter(
                "no adapter under checkpoints/materials/cot-sft-sae/. Run "
                "scripts/materials/08_train_sft.sh first."
            )
        adapter_path = run_dir / "adapter"

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
            "no labels.json under runs/materials/sae_labels/."
        )

    if not holdout_ids_path.exists():
        raise typer.BadParameter(f"missing {holdout_ids_path}")
    with holdout_ids_path.open("r") as f:
        holdout_ids = [int(s) for s in json.load(f)]

    run_id = _generate_run_id(f"mat-cot-sft-sae-{len(holdout_ids)}-holdout")
    out_dir = out / "cot_sft_sae" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    typer.echo("==> Materials port Stage 9: single-shot inference")
    typer.echo(f"    Run id          : {run_id}")
    typer.echo(f"    Output          : {out_dir}")
    typer.echo(f"    Adapter         : {adapter_path}")
    typer.echo(f"    Probe bank      : {probe_bank_dir}")
    typer.echo(f"    SAE             : {sae_path}")
    typer.echo(f"    SAE labels      : {sae_labels_path}")
    typer.echo(f"    Top-K features  : {top_k_features}")
    typer.echo(f"    Holdout ids     : {len(holdout_ids)}")
    typer.echo("")

    typer.echo("==> Loading CHGNet...")
    wrap = CHGNetWrap.load(device=device, model_name=chgnet_model_name)
    bank = ProbeBank.load(probe_bank_dir, device=device).eval()
    sae, cls_mean, cls_std = _load_sae(sae_path, device=device)
    labels = _load_labels(sae_labels_path)

    typer.echo(f"==> Loading LLM (quantize={quantize})...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Left-pad for batched causal generation: padding goes to the
    # left of each prompt so all rows generate from the same final
    # position regardless of prompt length.
    tokenizer.padding_side = "left"

    load_kwargs: dict = {"device_map": device}
    if quantize == "4bit":
        from transformers import BitsAndBytesConfig  # noqa: PLC0415

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif quantize == "8bit":
        from transformers import BitsAndBytesConfig  # noqa: PLC0415

        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quantize == "none":
        load_kwargs["torch_dtype"] = (
            torch.bfloat16 if device == "cuda" else torch.float32
        )
    else:
        raise typer.BadParameter(
            f"--quantize must be one of none/8bit/4bit, got {quantize!r}"
        )
    base_llm = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
    llm = PeftModel.from_pretrained(base_llm, str(adapter_path), is_trainable=False)
    llm.eval()

    jsonl_path = out_dir / "records.jsonl"
    counters = {
        "total": 0,
        "committed": 0,
        "parse_failure": 0,
        "correct": 0,
        "skipped_chgnet_error": 0,
    }
    started_run = _now_utc()

    typer.echo(
        f"==> Starting generation ({len(holdout_ids)} specimens, "
        f"batch_size={batch_size})"
    )
    with h5py.File(h5_path, "r") as h5, jsonl_path.open("w") as out_f:
        element_names_attr = h5.attrs.get("element_names")
        element_names = (
            [s.decode() if isinstance(s, bytes) else str(s)
             for s in element_names_attr]
            if element_names_attr is not None else []
        )

        # Chunked two-phase processing:
        #   Phase A: per-specimen forward through CHGNet + probes + SAE
        #            (CHGNet is per-cell so we still loop, but it's fast)
        #   Phase B: tokenize all prompts in chunk, batched generate(),
        #            decode each row
        #
        # The LLM generation is the bottleneck (KV-cache + autoregressive
        # decode); batching N=16 prompts shares that cost.
        for chunk_start in range(0, len(holdout_ids), batch_size):
            chunk_ids = holdout_ids[chunk_start : chunk_start + batch_size]

            # Phase A: build per-specimen context for the chunk.
            ctxs: list[dict] = []
            for sid in chunk_ids:
                n_atoms = int(np.asarray(h5["nsites"][sid]))
                if n_atoms > max_atoms or n_atoms < 1:
                    counters["skipped_chgnet_error"] += 1
                    continue
                species_ids = np.asarray(h5["n_atoms_padded"][sid])[:n_atoms]
                positions = np.asarray(h5["positions_padded"][sid])[:n_atoms]
                lattice = np.asarray(h5["lattice"][sid])
                try:
                    structure = structure_from_arrays(
                        species_ids=species_ids,
                        positions=positions,
                        lattice=lattice,
                        element_names=element_names,
                    )
                    _, pooled = wrap.encode(structure)
                except Exception as exc:
                    counters["skipped_chgnet_error"] += 1
                    if counters["skipped_chgnet_error"] <= 5:
                        typer.echo(f"    skip sid={sid}: {exc!r}")
                    continue

                x = pooled.detach().to(device).float().reshape(1, -1)
                with torch.no_grad():
                    x_norm = (x - cls_mean) / cls_std.clamp_min(1.0e-6)
                    z_row = sae.encode(x_norm).detach().cpu().numpy()[0]
                probe_outputs_batch = bank.evaluate(x)
                probe_out = probe_outputs_batch[0]

                sae_feats = _top_k_features_for_row(
                    z_row, labels=labels, top_k=top_k_features,
                )
                user_text = _user_message(probe_out, sae_feats)
                prompt = tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_text},
                    ],
                    tokenize=False, add_generation_prompt=True,
                )
                ctxs.append({
                    "sid": int(sid),
                    "prompt": prompt,
                    "probe_out": probe_out,
                    "sae_feats": sae_feats,
                })

            if not ctxs:
                continue

            # Phase B: tokenize + generate batched, then decode each row.
            prompts = [c["prompt"] for c in ctxs]
            enc = tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=False,
            ).to(device)
            t0 = _now_utc()
            with torch.no_grad():
                out_ids = llm.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            t1 = _now_utc()
            input_len = int(enc["input_ids"].shape[1])
            gen_ids_batch = out_ids[:, input_len:]
            raw_texts = tokenizer.batch_decode(
                gen_ids_batch, skip_special_tokens=True,
            )

            for ctx, raw_text in zip(ctxs, raw_texts, strict=True):
                raw_text = raw_text.strip()
                claim = _parse_final_commit(raw_text)
                truth = truth_dict(h5, ctx["sid"])
                correct = (
                    is_correct(claim or {}, truth)
                    if claim is not None else False
                )

                record = {
                    "specimen_id": ctx["sid"],
                    "started_utc": t0,
                    "finished_utc": t1,
                    "raw_text": raw_text,
                    "claim": claim,
                    "is_correct": bool(correct),
                    "ground_truth": {
                        k: (
                            bool(v) if isinstance(v, np.bool_)
                            else float(v) if isinstance(v, (np.floating, float))
                            else int(v) if isinstance(v, (np.integer, int))
                            else str(v)
                        )
                        for k, v in truth.items()
                    },
                    "probe_outputs": {
                        name: {
                            "prediction": ctx["probe_out"][name].get("prediction"),
                            "confidence": float(
                                ctx["probe_out"][name].get("confidence", 0.0)
                            ),
                        }
                        for name in bank.names()
                    },
                    "sae_features": [
                        [str(lab), float(act)] for lab, act in ctx["sae_feats"]
                    ],
                    "n_sae_features_used": len(ctx["sae_feats"]),
                }
                counters["total"] += 1
                if claim is not None:
                    counters["committed"] += 1
                    if correct:
                        counters["correct"] += 1
                else:
                    counters["parse_failure"] += 1
                out_f.write(json.dumps(record) + "\n")
            out_f.flush()

            del out_ids, gen_ids_batch, enc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            typer.echo(
                f"    {counters['total']:>4}/{len(holdout_ids)} "
                f"chunk_done={len(ctxs)} "
                f"committed={counters['committed']} "
                f"correct={counters['correct']} "
                f"parse_failure={counters['parse_failure']} "
                f"chgnet_skip={counters['skipped_chgnet_error']}"
            )

    typer.echo(f"==> JSONL: {jsonl_path}")
    accuracy = counters["correct"] / max(counters["total"], 1)
    typer.echo(
        f"    accuracy: {counters['correct']}/{counters['total']} "
        f"= {accuracy:.4f}"
    )
    with (out_dir / "summary.yaml").open("w") as f:
        yaml.safe_dump(
            {
                "baseline": "cot_sft_sae",
                "domain": "materials",
                "counters": counters,
                "accuracy": float(accuracy),
                "started_utc": started_run,
                "finished_utc": _now_utc(),
            },
            f,
            sort_keys=False,
        )

    with (out_dir / "manifest.yaml").open("w") as f:
        yaml.safe_dump(
            {
                "run_id": run_id,
                "completed_utc": datetime.now(UTC).isoformat(),
                "h5_path": str(h5_path),
                "holdout_ids_path": str(holdout_ids_path),
                "adapter_path": str(adapter_path),
                "probe_bank_dir": str(probe_bank_dir),
                "sae_dir": str(sae_dir),
                "sae_labels_path": str(sae_labels_path),
                "chgnet_model_name": chgnet_model_name,
                "base_model": base_model,
                "max_atoms": max_atoms,
                "top_k_features": top_k_features,
                "max_new_tokens": max_new_tokens,
                "quantize": quantize,
                "batch_size": batch_size,
                "n_holdout": len(holdout_ids),
                "counters": counters,
                "accuracy": float(accuracy),
            },
            f,
            sort_keys=False,
        )


if __name__ == "__main__":
    app()
