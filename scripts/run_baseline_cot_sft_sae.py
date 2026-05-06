"""CLI: single-shot Phase 16 baseline -- LoRA-tuned Qwen reading probes + SAE features.

Mirrors scripts/run_baseline_cot.py but the user message also
contains the top-K labelled SAE features per specimen, matching the
format of the records that scripts/build_cot_dataset_with_sae.py
produced for the SFT trainer.

No verifier, no OHVD loop. Single forward, single LLM commit per
specimen. Output trajectory is schema-compatible with the held-out
evaluator so it discovers a new column ``cot_sft_sae``.

Usage:

    bash scripts/run_baseline_cot_sft_sae.sh
    SPECIMEN_IDS_FILE=runs/holdout_lock/ids.json \\
        bash scripts/run_baseline_cot_sft_sae.sh

Depends on:
    typer, torch, h5py, transformers, peft, numpy, pyyaml.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import torch
import typer
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fmllm.fms.common import load_checkpoint  # noqa: E402
from fmllm.fms.fm2_rdf.model import build_fm2_model  # noqa: E402
from fmllm.orchestrator.trajectory import (  # noqa: E402
    ActionType,
    LLMAction,
    Step,
    StepType,
    TerminationReason,
    Trajectory,
)
from fmllm.representation.sae import TopKSAE  # noqa: E402
from fmllm.training.probe_bank import ProbeBank  # noqa: E402
from fmllm.training.synthetic_cot import _user_message  # noqa: E402
from fmllm.utils.manifests import write_manifest  # noqa: E402
from fmllm.utils.run_ids import generate_run_id  # noqa: E402
from fmllm.utils.config import load_config  # noqa: E402
from fmllm.verifier.schema import PhysicalStateClaim  # noqa: E402


app = typer.Typer(add_completion=False, no_args_is_help=False)


_FINAL_RE = re.compile(r"Final commit:\s*(\{.*?\})", re.DOTALL)


def _now_utc() -> str:
    return datetime.now(UTC).isoformat()


def _latest_dir(parent: Path) -> Path | None:
    if not parent.exists():
        return None
    cands = sorted(
        parent.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True,
    )
    return next((c for c in cands if c.is_dir()), None)


def _latest_fm2(checkpoint_root: Path, train_split: str) -> Path:
    parent = checkpoint_root / "fm2_rdf" / train_split
    cands = sorted(parent.glob("*"), key=lambda p: p.name, reverse=True)
    cands = [c for c in cands if (c / "model.pt").exists()]
    if not cands:
        raise typer.BadParameter(f"no fm2_rdf checkpoint under {parent}")
    return cands[0]


def _parse_final_commit(text: str) -> PhysicalStateClaim | None:
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
    if not isinstance(data, dict):
        return None
    return PhysicalStateClaim(
        n_atoms=int(data["n_atoms"]) if "n_atoms" in data else None,
        temperature=(
            float(data["temperature"]) if "temperature" in data else None
        ),
        motif=str(data["motif"]) if "motif" in data else None,
    )


def _load_sae(sae_path: Path, device: str) -> tuple[TopKSAE, torch.Tensor, torch.Tensor]:
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
        Path("data/synthetic_lj_v1/specimens.h5"), "--h5-path",
    ),
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    checkpoint_root: Path = typer.Option(
        Path("checkpoints"), "--checkpoint-root",
    ),
    train_split: str = typer.Option("train_50k", "--train-split"),
    adapter_path: Path | None = typer.Option(
        None, "--adapter-path",
        help="LoRA adapter trained on the SAE-augmented CoT dataset. "
             "Default: latest under checkpoints/cot-sft-sae/.",
    ),
    probe_bank_dir: Path | None = typer.Option(
        None, "--probe-bank-dir",
        help="Probe bank directory. Default: latest under checkpoints/probes/.",
    ),
    sae_dir: Path | None = typer.Option(
        None, "--sae-dir",
        help="Trained SAE directory. Default: latest under checkpoints/sae/.",
    ),
    sae_labels_path: Path | None = typer.Option(
        None, "--sae-labels-path",
        help="labels.json for the SAE. Default: latest under runs/sae_labels/.",
    ),
    top_k_features: int = typer.Option(
        8, "--top-k-features",
        help="Top-active SAE features surfaced per specimen.",
    ),
    base_model: str = typer.Option(
        "Qwen/Qwen2.5-7B-Instruct", "--base-model",
    ),
    start: int = typer.Option(0, "--start"),
    count: int = typer.Option(200, "--count"),
    specimen_ids_file: Path | None = typer.Option(
        None, "--specimen-ids-file",
    ),
    out: Path = typer.Option(Path("runs/holdout"), "--out", "-o"),
    max_new_tokens: int = typer.Option(
        768, "--max-new-tokens",
        help="Generation budget per specimen. Phase 16 CoTs include "
             "Step 1 (5 probes) + Step 1b (top-K SAE features, "
             "~80-120 chars each) + Step 2 + Step 3 + Final commit. "
             "Top-K=8 records run ~600-800 tokens. Default 768 covers "
             "the long tail; raise to 1024 if Step 1b uses verbose "
             "SAE labels.",
    ),
    batch_size: int = typer.Option(16, "--batch-size"),
    quantize: str = typer.Option(
        "4bit", "--quantize",
        help="'none' | '4bit' | '8bit'. Default 4bit for memory.",
    ),
    device: str = typer.Option("auto", "--device"),
    log_every: int = typer.Option(10, "--log-every"),
) -> None:
    """Single-shot inference with the SAE-augmented CoT-SFT adapter."""
    from peft import PeftModel  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    cfg = load_config(config)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if adapter_path is None:
        run_dir = _latest_dir(Path("checkpoints/cot-sft-sae"))
        if run_dir is None or not (run_dir / "adapter").exists():
            raise typer.BadParameter(
                "no adapter under checkpoints/cot-sft-sae/. Run "
                "scripts/train_cot_sft_with_sae.sh first."
            )
        adapter_path = run_dir / "adapter"

    if probe_bank_dir is None:
        probe_bank_dir = _latest_dir(Path("checkpoints/probes"))
        if probe_bank_dir is None:
            raise typer.BadParameter("no probe bank under checkpoints/probes/.")

    if sae_dir is None:
        sae_dir = _latest_dir(Path("checkpoints/sae"))
        if sae_dir is None:
            raise typer.BadParameter("no SAE under checkpoints/sae/.")
    sae_path = sae_dir / "sae.pt"
    if not sae_path.exists():
        raise typer.BadParameter(f"no sae.pt under {sae_dir}")

    if sae_labels_path is None:
        cands = sorted(
            Path("runs/sae_labels").glob("*/labels.json"),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        sae_labels_path = cands[0] if cands else None
    if sae_labels_path is None or not sae_labels_path.exists():
        raise typer.BadParameter(
            "no SAE labels found. Run scripts/label_sae_features.sh first."
        )

    if specimen_ids_file is not None:
        with specimen_ids_file.open("r") as f:
            specimen_ids = list(json.load(f))
        run_slug = f"baseline-cot-sft-sae-{len(specimen_ids)}-holdout"
    else:
        specimen_ids = list(range(start, start + count))
        run_slug = f"baseline-cot-sft-sae-{count}"

    # Resume detection
    resume_already_done: set[int] = set()
    resume_dir: Path | None = None
    base_root = out / "cot_sft_sae"
    if base_root.exists():
        for d in sorted(base_root.iterdir(), key=lambda p: p.name, reverse=True):
            jsonl = d / "trajectories.jsonl"
            if jsonl.exists() and jsonl.stat().st_size > 0:
                resume_dir = d
                with jsonl.open("r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        sid_existing = obj.get("specimen_id")
                        if isinstance(sid_existing, int):
                            resume_already_done.add(sid_existing)
                break

    if resume_already_done and resume_dir is not None:
        out_dir = resume_dir
        run_id = resume_dir.name
        run_mode = "resume"
    else:
        run_id = generate_run_id(run_slug)
        out_dir = out / "cot_sft_sae" / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        run_mode = "fresh"

    typer.echo(f"==> Run mode    : {run_mode}")
    typer.echo(f"==> Run id      : {run_id}")
    typer.echo(f"==> Output      : {out_dir}")
    typer.echo(f"==> Adapter     : {adapter_path}")
    typer.echo(f"==> Probe bank  : {probe_bank_dir}")
    typer.echo(f"==> SAE         : {sae_path}")
    typer.echo(f"==> Labels      : {sae_labels_path}")
    typer.echo(f"==> Top-K feats : {top_k_features}")
    typer.echo(f"==> Specimens   : {len(specimen_ids)}")

    fm2_ckpt = _latest_fm2(checkpoint_root, train_split)
    typer.echo(f"==> FM2 ckpt    : {fm2_ckpt}")
    fm2 = build_fm2_model(cfg.fm2).to(device)
    load_checkpoint(fm2_ckpt / "model.pt", model=fm2, map_location=device)
    fm2.eval()
    for p in fm2.parameters():
        p.requires_grad = False

    bank = ProbeBank.load(probe_bank_dir, device=device).eval()
    sae, cls_mean, cls_std = _load_sae(sae_path, device=device)
    labels = _load_labels(sae_labels_path)

    typer.echo(f"==> Loading LLM (quantize={quantize})...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

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

    system_text = (
        "You are a scientific reasoner working with a 2D Lennard-Jones "
        "cluster testbed. You receive probe outputs derived from a "
        "frozen foundation model and must reason explicitly about the "
        "evidence before committing a typed claim about the specimen's "
        "atom count, motif, and temperature."
    )

    jsonl_path = out_dir / "trajectories.jsonl"
    counters = {
        "total": 0,
        "committed": 0,
        "parse_failure": 0,
        "skipped_resume": 0,
    }
    started_run = _now_utc()

    todo = [s for s in specimen_ids if int(s) not in resume_already_done]
    counters["skipped_resume"] = len(specimen_ids) - len(todo)
    typer.echo(f"==> Starting generation ({run_mode}, {len(todo)} to do)")

    write_mode = "a" if run_mode == "resume" else "w"
    with h5py.File(h5_path, "r") as h5, jsonl_path.open(write_mode) as out_f:
        for start_i in range(0, len(todo), batch_size):
            batch_ids = todo[start_i : start_i + batch_size]
            rdfs_np = np.stack(
                [np.asarray(h5["rdfs"][i]) for i in batch_ids], axis=0,
            ).astype(np.float32)
            rdfs = torch.from_numpy(rdfs_np).to(device).float()
            with torch.no_grad():
                hidden = fm2.encode(rdfs)
                cls = hidden[:, 0, :]
                cls_norm = (cls - cls_mean) / cls_std.clamp_min(1.0e-6)
                z = sae.encode(cls_norm).detach().cpu().numpy()
            probe_outputs_batch = bank.evaluate(cls)

            for sid, probe_out, z_row in zip(
                batch_ids, probe_outputs_batch, z, strict=True,
            ):
                sae_feats = _top_k_features_for_row(
                    z_row, labels=labels, top_k=top_k_features,
                )
                user_text = _user_message(probe_out, sae_feats)
                messages = [
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": user_text},
                ]
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                t0 = _now_utc()
                with torch.no_grad():
                    out_ids = llm.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                gen_ids = out_ids[0, inputs["input_ids"].shape[1] :]
                raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                t1 = _now_utc()

                claim = _parse_final_commit(raw_text)
                action = LLMAction(
                    action_type=(
                        ActionType.COMMIT if claim is not None else ActionType.ERROR
                    ),
                    claim=claim,
                    error=None if claim is not None else "could not parse Final commit",
                    raw_text=raw_text,
                )
                step = Step(
                    step_index=0,
                    step_type=(
                        StepType.FINAL if claim is not None else StepType.ERROR
                    ),
                    timestamp_utc=t1,
                    llm_action=action,
                    claim=claim,
                )
                traj = Trajectory(
                    run_id=run_id,
                    query="cot_sft_sae baseline (probes+SAE -> Qwen+adapter -> commit)",
                    specimen_id=int(sid),
                    started_utc=t0,
                    finished_utc=t1,
                    termination=(
                        TerminationReason.COMMITTED
                        if claim is not None
                        else TerminationReason.PARSE_FAILURE
                    ),
                    final_claim=claim,
                    final_verdict=None,
                    steps=[step],
                    metadata={
                        "baseline": "cot_sft_sae",
                        "adapter_path": str(adapter_path),
                        "probe_bank_dir": str(probe_bank_dir),
                        "sae_dir": str(sae_dir),
                        "sae_labels_path": str(sae_labels_path),
                        "top_k_features": top_k_features,
                        "n_sae_features_used": len(sae_feats),
                        "probe_outputs": {
                            name: {
                                "prediction": probe_out[name].get("prediction"),
                                "confidence": float(
                                    probe_out[name].get("confidence", 0.0)
                                ),
                            }
                            for name in bank.names()
                        },
                    },
                )
                counters["total"] += 1
                if claim is not None:
                    counters["committed"] += 1
                else:
                    counters["parse_failure"] += 1
                out_f.write(traj.model_dump_json() + "\n")
                out_f.flush()

                del out_ids, gen_ids, inputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if (
                    counters["total"] == 1
                    or counters["total"] % log_every == 0
                    or counters["total"] == len(todo)
                ):
                    typer.echo(
                        f"    {counters['total']:>4}/{len(todo)} "
                        f"sid={int(sid):<6} "
                        f"committed={counters['committed']} "
                        f"parse_failure={counters['parse_failure']}"
                    )

    typer.echo(f"==> JSONL: {jsonl_path}")
    with (out_dir / "summary.yaml").open("w") as f:
        yaml.safe_dump(
            {
                "baseline": "cot_sft_sae",
                "counters": counters,
                "started_utc": started_run,
                "finished_utc": _now_utc(),
            },
            f,
        )
    write_manifest(
        out_dir / "manifest.yaml",
        script="scripts.run_baseline_cot_sft_sae",
        inputs={
            "h5_path": str(h5_path),
            "adapter_path": str(adapter_path),
            "probe_bank_dir": str(probe_bank_dir),
            "sae_dir": str(sae_dir),
            "sae_labels_path": str(sae_labels_path),
            "fm2_checkpoint": str(fm2_ckpt),
            "base_model": base_model,
            "n_specimens": len(specimen_ids),
            "specimen_ids_file": (
                str(specimen_ids_file) if specimen_ids_file is not None else None
            ),
        },
        config={
            "run_id": run_id,
            "max_new_tokens": max_new_tokens,
            "batch_size": batch_size,
            "quantize": quantize,
            "top_k_features": top_k_features,
        },
        extra={"counters": counters},
    )


if __name__ == "__main__":
    app()
