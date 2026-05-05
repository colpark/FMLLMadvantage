"""CLI: run the Phase 11.B CoT-SFT baseline.

Single-shot inference baseline that bypasses the OHVD loop and the
verifier entirely:

    1. Load FM2 (frozen) + probe bank + Qwen + LoRA adapter from
       Phase 11 Stage 2.
    2. For each specimen ID (range or explicit list):
         - Forward through FM2 to get the CLS embedding.
         - Run every probe in the bank.
         - Build the user message in the same format the SFT trainer
           used.
         - Generate from Qwen+adapter, deterministic decoding.
         - Parse the ``Final commit: {...}`` JSON out of the
           generated text.
         - Write a single-step Trajectory containing the parsed claim
           so the existing held-out evaluator can score it.
    3. Output ``trajectories.jsonl`` + ``summary.yaml`` + ``manifest.yaml``
       under ``runs/holdout/cot_sft/<run_id>/`` (or the configured root).

The output is schema-compatible with the trajectories produced by
the other Phase 8a baselines, so ``scripts/evaluate_baselines.sh``
auto-discovers it.

Usage:
    bash scripts/run_baseline_cot.sh
    SPECIMEN_IDS_FILE=runs/holdout_lock/ids.json \\
        bash scripts/run_baseline_cot.sh

Depends on:
    typer, torch, h5py, transformers, peft (lazy).
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
from fmllm.training.probe_bank import ProbeBank  # noqa: E402
from fmllm.training.synthetic_cot import _user_message  # noqa: E402
from fmllm.utils.config import load_config  # noqa: E402
from fmllm.utils.manifests import write_manifest  # noqa: E402
from fmllm.utils.run_ids import generate_run_id  # noqa: E402
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
    """Extract the first balanced JSON object after ``Final commit:``.

    Falls back to the regex-greedy match when the brace counter
    cannot find a balanced object.
    """
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
        help="Path to the LoRA adapter directory. Default: latest under "
             "checkpoints/cot-sft/.",
    ),
    probe_bank_dir: Path | None = typer.Option(
        None, "--probe-bank-dir",
        help="Path to the probe bank directory. Default: latest under "
             "checkpoints/probes/.",
    ),
    base_model: str = typer.Option(
        "Qwen/Qwen2.5-7B-Instruct", "--base-model",
    ),
    start: int = typer.Option(0, "--start"),
    count: int = typer.Option(200, "--count"),
    specimen_ids_file: Path | None = typer.Option(
        None, "--specimen-ids-file",
        help="JSON list of specimen IDs to run; overrides --start/--count.",
    ),
    out: Path = typer.Option(Path("runs/holdout"), "--out", "-o"),
    max_new_tokens: int = typer.Option(384, "--max-new-tokens"),
    batch_size: int = typer.Option(64, "--batch-size"),
    device: str = typer.Option("auto", "--device"),
    log_every: int = typer.Option(50, "--log-every"),
) -> None:
    """Run the CoT-SFT baseline on a list of specimens."""
    from peft import PeftModel  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    cfg = load_config(config)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if adapter_path is None:
        run_dir = _latest_dir(Path("checkpoints/cot-sft"))
        if run_dir is None or not (run_dir / "adapter").exists():
            raise typer.BadParameter(
                "no adapter under checkpoints/cot-sft/. Run "
                "scripts/train_cot_sft.sh first."
            )
        adapter_path = run_dir / "adapter"
    if probe_bank_dir is None:
        probe_bank_dir = _latest_dir(Path("checkpoints/probes"))
        if probe_bank_dir is None:
            raise typer.BadParameter(
                "no probe bank under checkpoints/probes/."
            )

    if specimen_ids_file is not None:
        with specimen_ids_file.open("r") as f:
            specimen_ids = list(json.load(f))
        if not all(isinstance(x, int) for x in specimen_ids):
            raise typer.BadParameter(
                f"{specimen_ids_file} must be a JSON list of ints"
            )
        run_slug = f"baseline-cot-sft-{len(specimen_ids)}-holdout"
    else:
        specimen_ids = list(range(start, start + count))
        run_slug = f"baseline-cot-sft-{count}"

    run_id = generate_run_id(run_slug)
    out_dir = out / "cot_sft" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"==> Run id      : {run_id}")
    typer.echo(f"==> Output      : {out_dir}")
    typer.echo(f"==> Adapter     : {adapter_path}")
    typer.echo(f"==> Probe bank  : {probe_bank_dir}")
    typer.echo(f"==> Base model  : {base_model}")
    typer.echo(f"==> Specimens   : {len(specimen_ids)}")

    fm2_ckpt = _latest_fm2(checkpoint_root, train_split)
    typer.echo(f"==> FM2 ckpt    : {fm2_ckpt}")
    fm2 = build_fm2_model(cfg.fm2).to(device)
    load_checkpoint(fm2_ckpt / "model.pt", model=fm2, map_location=device)
    fm2.eval()
    for p in fm2.parameters():
        p.requires_grad = False

    bank = ProbeBank.load(probe_bank_dir, device=device).eval()

    typer.echo("==> Loading LLM + adapter...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base_llm = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device,
    )
    llm = PeftModel.from_pretrained(
        base_llm, str(adapter_path), is_trainable=False,
    )
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
    }
    started_run = _now_utc()

    with h5py.File(h5_path, "r") as h5, jsonl_path.open("w") as out_f:
        for start_i in range(0, len(specimen_ids), batch_size):
            batch_ids = specimen_ids[start_i : start_i + batch_size]
            rdfs_np = np.stack(
                [np.asarray(h5["rdfs"][i]) for i in batch_ids], axis=0,
            ).astype(np.float32)
            rdfs = torch.from_numpy(rdfs_np).to(device).float()
            with torch.no_grad():
                hidden = fm2.encode(rdfs)
                cls = hidden[:, 0, :]
            probe_outputs_batch = bank.evaluate(cls)

            for sid, probe_out in zip(batch_ids, probe_outputs_batch, strict=True):
                user_text = _user_message(probe_out)
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
                raw_text = tokenizer.decode(
                    gen_ids, skip_special_tokens=True,
                ).strip()
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
                    query="cot_sft baseline (probes -> Qwen+adapter -> commit)",
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
                        "baseline": "cot_sft",
                        "adapter_path": str(adapter_path),
                        "probe_bank_dir": str(probe_bank_dir),
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

            done = min(start_i + batch_size, len(specimen_ids))
            if done == len(specimen_ids) or (done % log_every == 0):
                typer.echo(
                    f"    {done}/{len(specimen_ids)} specimens "
                    f"(committed={counters['committed']}, "
                    f"parse_failure={counters['parse_failure']})"
                )

    typer.echo(f"==> JSONL: {jsonl_path}")
    with (out_dir / "summary.yaml").open("w") as f:
        yaml.safe_dump(
            {
                "baseline": "cot_sft",
                "counters": counters,
                "started_utc": started_run,
                "finished_utc": _now_utc(),
            },
            f,
        )
    write_manifest(
        out_dir / "manifest.yaml",
        script="scripts.run_baseline_cot",
        inputs={
            "h5_path": str(h5_path),
            "adapter_path": str(adapter_path),
            "probe_bank_dir": str(probe_bank_dir),
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
            "device": device,
        },
        extra={
            "counters": counters,
        },
    )
    typer.echo(json.dumps(counters, indent=2))


if __name__ == "__main__":
    app()
