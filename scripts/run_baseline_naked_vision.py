"""CLI: naked LLM + generic VLM caption baseline.

Fixes the original ``naked`` baseline's "zero-observation floor"
unfairness. The original naked LLM saw only a textual dataset
description and a specimen ID, so its 0.000 goal accuracy was a
pure guess against the marginal. This baseline gives the LLM
something a non-specialist AI system would actually have access to:
the specimen's rasterized image piped through a generic
image-captioning model (BLIP), then the resulting natural-language
caption is injected into the LLM's user prompt.

The point: a generic VLM is the "what would a competent AI system
do without our domain-specific FMs" floor. Whatever caption BLIP
produces ("a black background with white dots", etc.) is what a
non-specialist would have to work with.

The orchestrator LLM (Qwen 2.5 7B Instruct) sees:

    [system] dataset description (LJ, 3 motifs, N in [5,30], etc.)
    [user]   specimen_id + "A generic VLM described the image as:
             '<BLIP caption>'. Commit a JSON answer."

No FM tools, no verifier, no probes, no SAE features. Single LLM
call, parse Final commit, write a one-step Trajectory.

Output goes to ``runs/holdout/naked_vision/<run_id>/`` so
``scripts/evaluate_baselines.sh`` auto-discovers it as a new column.

Usage:

    bash scripts/run_baseline_naked_vision.sh
    SPECIMEN_IDS_FILE=runs/holdout_lock/ids.json \\
        bash scripts/run_baseline_naked_vision.sh

Depends on:
    typer, torch, h5py, transformers (BLIP and Qwen lazy).
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

from fmllm.baselines.naked import NAKED_SYSTEM_PROMPT  # noqa: E402
from fmllm.orchestrator.llm import parse_llm_response  # noqa: E402
from fmllm.orchestrator.trajectory import (  # noqa: E402
    ActionType,
    LLMAction,
    Step,
    StepType,
    TerminationReason,
    Trajectory,
)
from fmllm.utils.manifests import write_manifest  # noqa: E402
from fmllm.utils.run_ids import generate_run_id  # noqa: E402


app = typer.Typer(add_completion=False, no_args_is_help=False)


def _now_utc() -> str:
    return datetime.now(UTC).isoformat()


def _to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """Convert a 2D float intensity image to a 3-channel uint8 RGB array
    suitable for a generic image captioner."""
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(
            f"expected (H, W) intensity image, got shape {arr.shape}"
        )
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo > 1.0e-6:
        arr = (arr - lo) / (hi - lo)
    arr = (arr * 255.0).clip(0.0, 255.0).astype(np.uint8)
    return np.stack([arr, arr, arr], axis=-1)


def _format_user_message(*, specimen_id: int, caption: str) -> str:
    return (
        f"Specimen ID: {specimen_id}.\n\n"
        f"A general-purpose image-captioning model (BLIP) describes "
        f"the specimen's image as: \"{caption}\".\n\n"
        f"Use that caption plus your prior knowledge of the testbed "
        f"distribution to commit one JSON action of the form "
        f"{{\"action\": \"commit\", \"claim\": {{\"n_atoms\": <int>, "
        f"\"motif\": \"<str>\", \"temperature\": <float>}}}}."
    )


@app.command()
def main(
    h5_path: Path = typer.Option(
        Path("data/synthetic_lj_v1/specimens.h5"), "--h5-path",
    ),
    base_model: str = typer.Option(
        "Qwen/Qwen2.5-7B-Instruct", "--base-model",
    ),
    captioner_model: str = typer.Option(
        "Salesforce/blip-image-captioning-base", "--captioner-model",
        help="Generic image captioner. BLIP base is small (~250MB), "
             "fast, and not specialized for our domain -- exactly the "
             "kind of model a non-specialist AI system would have.",
    ),
    start: int = typer.Option(0, "--start"),
    count: int = typer.Option(200, "--count"),
    specimen_ids_file: Path | None = typer.Option(
        None, "--specimen-ids-file",
        help="JSON list of specimen IDs; overrides --start/--count.",
    ),
    out: Path = typer.Option(Path("runs/holdout"), "--out", "-o"),
    max_new_tokens_caption: int = typer.Option(40, "--max-new-tokens-caption"),
    max_new_tokens_commit: int = typer.Option(192, "--max-new-tokens-commit"),
    quantize: str = typer.Option(
        "4bit", "--quantize",
        help="LLM quantization: none / 8bit / 4bit. Same options as "
             "scripts/run_baseline_cot.py.",
    ),
    log_every: int = typer.Option(10, "--log-every"),
    device: str = typer.Option("auto", "--device"),
) -> None:
    """Run the naked-LLM + generic-VLM baseline."""
    from transformers import (  # noqa: PLC0415
        AutoModelForCausalLM,
        AutoTokenizer,
        BlipForConditionalGeneration,
        BlipProcessor,
    )

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if specimen_ids_file is not None:
        with specimen_ids_file.open("r") as f:
            specimen_ids = list(json.load(f))
        if not all(isinstance(x, int) for x in specimen_ids):
            raise typer.BadParameter(
                f"{specimen_ids_file} must be a JSON list of ints"
            )
        run_slug = f"baseline-naked-vision-{len(specimen_ids)}-holdout"
    else:
        specimen_ids = list(range(start, start + count))
        run_slug = f"baseline-naked-vision-{count}"

    # Resume detection.
    base_root = out / "naked_vision"
    resume_already_done: set[int] = set()
    resume_dir: Path | None = None
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
                        sid = obj.get("specimen_id")
                        if isinstance(sid, int):
                            resume_already_done.add(sid)
                break

    if resume_already_done and resume_dir is not None:
        out_dir = resume_dir
        run_id = resume_dir.name
        run_mode = "resume"
        typer.echo(
            f"==> Resuming run {run_id} ({len(resume_already_done)} "
            f"specimens already processed)"
        )
    else:
        run_id = generate_run_id(run_slug)
        out_dir = base_root / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        run_mode = "fresh"

    todo = [s for s in specimen_ids if int(s) not in resume_already_done]

    typer.echo(f"==> Run id    : {run_id}")
    typer.echo(f"==> Output    : {out_dir}")
    typer.echo(f"==> Captioner : {captioner_model}")
    typer.echo(f"==> Base model: {base_model}")
    typer.echo(f"==> Specimens : {len(specimen_ids)} ({len(todo)} to do)")

    # Captioner ------------------------------------------------------------
    typer.echo("==> Loading captioner...")
    cap_processor = BlipProcessor.from_pretrained(captioner_model)
    cap_model = BlipForConditionalGeneration.from_pretrained(
        captioner_model,
        torch_dtype=torch.float32,
    ).to(device)
    cap_model.eval()
    for p in cap_model.parameters():
        p.requires_grad = False

    # LLM ------------------------------------------------------------------
    typer.echo(f"==> Loading LLM (quantize={quantize})...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    llm_kwargs: dict = {"device_map": device}
    if quantize == "4bit":
        from transformers import BitsAndBytesConfig  # noqa: PLC0415

        llm_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif quantize == "8bit":
        from transformers import BitsAndBytesConfig  # noqa: PLC0415

        llm_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quantize == "none":
        llm_kwargs["torch_dtype"] = (
            torch.bfloat16 if device == "cuda" else torch.float32
        )
    else:
        raise typer.BadParameter(
            f"--quantize must be none/8bit/4bit, got {quantize!r}"
        )
    llm = AutoModelForCausalLM.from_pretrained(base_model, **llm_kwargs)
    llm.eval()

    # Caption + commit loop ------------------------------------------------
    jsonl_path = out_dir / "trajectories.jsonl"
    counters = {
        "total": 0,
        "committed": 0,
        "parse_failure": 0,
        "skipped_resume": len(specimen_ids) - len(todo),
    }

    typer.echo("")
    typer.echo(f"==> Starting generation ({run_mode}, {len(todo)} to do)")
    typer.echo("")

    write_mode = "a" if run_mode == "resume" else "w"
    with h5py.File(h5_path, "r") as h5, jsonl_path.open(write_mode) as out_f:
        for i, sid in enumerate(todo):
            t0 = _now_utc()
            image = np.asarray(h5["images"][int(sid)])
            rgb = _to_rgb_uint8(image)

            # Caption the image with BLIP. Greedy decoding for
            # determinism across runs.
            from PIL import Image  # noqa: PLC0415
            pil = Image.fromarray(rgb)
            cap_inputs = cap_processor(images=pil, return_tensors="pt").to(device)
            with torch.no_grad():
                cap_ids = cap_model.generate(
                    **cap_inputs,
                    max_new_tokens=max_new_tokens_caption,
                    do_sample=False,
                )
            caption = cap_processor.decode(
                cap_ids[0], skip_special_tokens=True,
            ).strip()

            # Build the LLM chat with the caption embedded.
            messages = [
                {"role": "system", "content": NAKED_SYSTEM_PROMPT},
                {"role": "user", "content": _format_user_message(
                    specimen_id=int(sid), caption=caption,
                )},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
            with torch.no_grad():
                out_ids = llm.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens_commit,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gen_ids = out_ids[0, inputs["input_ids"].shape[1] :]
            raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            t1 = _now_utc()

            action = parse_llm_response(raw_text)
            claim = (
                action.claim
                if action.action_type is ActionType.COMMIT and action.claim is not None
                else None
            )
            traj = Trajectory(
                run_id=run_id,
                query="naked_vision baseline (image + generic captioner)",
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
                steps=[
                    Step(
                        step_index=0,
                        step_type=(
                            StepType.FINAL if claim is not None else StepType.ERROR
                        ),
                        timestamp_utc=t1,
                        llm_action=action,
                        claim=claim,
                    )
                ],
                metadata={
                    "baseline": "naked_vision",
                    "captioner_model": captioner_model,
                    "vlm_caption": caption,
                },
            )
            counters["total"] += 1
            if claim is not None:
                counters["committed"] += 1
            else:
                counters["parse_failure"] += 1
            out_f.write(traj.model_dump_json() + "\n")
            out_f.flush()

            del out_ids, gen_ids, inputs, cap_ids, cap_inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if (
                counters["total"] == 1
                or counters["total"] % log_every == 0
                or counters["total"] == len(todo)
            ):
                typer.echo(
                    f"    {counters['total']:>4}/{len(todo)} "
                    f"sid={int(sid):<6} caption={caption[:60]!r} "
                    f"committed={counters['committed']} "
                    f"parse_failure={counters['parse_failure']}"
                )

    typer.echo(f"==> JSONL: {jsonl_path}")
    with (out_dir / "summary.yaml").open("w") as f:
        yaml.safe_dump(
            {
                "baseline": "naked_vision",
                "counters": counters,
                "completed_utc": datetime.now(UTC).isoformat(),
            },
            f,
        )
    write_manifest(
        out_dir / "manifest.yaml",
        script="scripts.run_baseline_naked_vision",
        inputs={
            "h5_path": str(h5_path),
            "captioner_model": captioner_model,
            "base_model": base_model,
            "n_specimens": len(specimen_ids),
            "specimen_ids_file": (
                str(specimen_ids_file) if specimen_ids_file is not None else None
            ),
        },
        config={
            "run_id": run_id,
            "max_new_tokens_caption": max_new_tokens_caption,
            "max_new_tokens_commit": max_new_tokens_commit,
            "quantize": quantize,
            "device": device,
        },
        extra={"counters": counters},
    )
    typer.echo(json.dumps(counters, indent=2))


if __name__ == "__main__":
    app()
