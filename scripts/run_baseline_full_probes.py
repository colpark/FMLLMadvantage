"""CLI: Phase 12 -- Pipeline A with probe outputs injected.

Combines the architectural strengths of the verifier-gated OHVD
loop (Pipeline A) and Phase 11's probe-derived typed evidence:

  - Pre-compute probe outputs for each specimen by forwarding
    through the frozen FM2 backbone and running the probe bank.
  - Inject the probe summary into the per-specimen user message
    so the LLM sees them upfront alongside the standard "identify
    the specimen" query.
  - Run the canonical OHVD loop with the multi-source verifier
    active. The LLM still calls FM tools, hypothesizes, gets
    verifier feedback, revises, commits.
  - Optionally stack a LoRA adapter (e.g., the Phase 11 cot_sft
    adapter) on top of the base LLM via --adapter-path.

Output writes to runs/holdout/full_probes/<run_id>/ so
scripts/evaluate_baselines.sh auto-discovers it as a new baseline
column alongside naked / cot_sft / no_verifier / full.

Usage:
    bash scripts/run_baseline_full_probes.sh
    SPECIMEN_IDS_FILE=runs/holdout_lock/ids.json \\
        bash scripts/run_baseline_full_probes.sh

Depends on:
    typer, torch, h5py, transformers, peft (lazy).
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import typer
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fmllm.bridges import load_fm_context  # noqa: E402
from fmllm.data.dataset import LJSpecimenDataset  # noqa: E402
from fmllm.fms.common import load_checkpoint  # noqa: E402
from fmllm.fms.fm2_rdf.model import build_fm2_model  # noqa: E402
from fmllm.orchestrator import (  # noqa: E402
    OHVDLoop,
    TransformersLLM,
    build_runners_from_checkpoints,
)
from fmllm.training.probe_bank import ProbeBank  # noqa: E402
from fmllm.utils.config import load_config  # noqa: E402
from fmllm.utils.logging import configure_logging  # noqa: E402
from fmllm.utils.manifests import write_manifest  # noqa: E402
from fmllm.utils.run_ids import generate_run_id  # noqa: E402
from fmllm.verifier import SourcesConfig, build_default_verifier  # noqa: E402


app = typer.Typer(add_completion=False, no_args_is_help=False)


_BASE_QUERY = (
    "Identify the specimen's atom count, motif, and temperature. "
    "Use FM tools to gather evidence, propose a claim, and commit "
    "when confident."
)


def _latest_dir(parent: Path) -> Path | None:
    if not parent.exists():
        return None
    cands = sorted(
        parent.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True,
    )
    return next((c for c in cands if c.is_dir()), None)


def _latest_completed_fm2(checkpoint_root: Path, train_split: str) -> Path:
    parent = checkpoint_root / "fm2_rdf" / train_split
    cands = sorted(parent.glob("*"), key=lambda p: p.name, reverse=True)
    cands = [c for c in cands if (c / "model.pt").exists()]
    if not cands:
        raise typer.BadParameter(f"no fm2_rdf checkpoint under {parent}")
    return cands[0]


def _format_probes_for_prompt(probe_outputs: dict[str, dict[str, Any]]) -> str:
    """Render the probe summary as a compact JSON-like string the LLM
    sees in the user message."""
    summary = {}
    for name, value in probe_outputs.items():
        prediction = value.get("prediction")
        if isinstance(prediction, float):
            prediction = round(prediction, 3)
        summary[name] = {
            "prediction": prediction,
            "confidence": round(float(value.get("confidence", 0.0)), 3),
        }
    return json.dumps(summary, sort_keys=True)


def _enriched_query(probes_json: str) -> str:
    return (
        f"{_BASE_QUERY}\n\n"
        f"PROBES (derived from a frozen FM2 representation, treat as "
        f"approximate hints, not ground truth): {probes_json}"
    )


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
    literature_db: Path = typer.Option(
        Path("data/literature/clusters.json"), "--literature-db",
    ),
    probe_bank_dir: Path | None = typer.Option(
        None, "--probe-bank-dir",
        help="Probe bank directory. Default: latest under checkpoints/probes/.",
    ),
    base_model: str = typer.Option(
        "Qwen/Qwen2.5-7B-Instruct", "--base-model",
    ),
    adapter_path: Path | None = typer.Option(
        None, "--adapter-path",
        help="Optional LoRA adapter to stack on the base LLM. The "
             "Phase 11 cot_sft adapter is a candidate; format mismatch "
             "may apply since the OHVD loop's prompts differ from the "
             "Phase 11 SFT format.",
    ),
    start: int = typer.Option(0, "--start"),
    count: int = typer.Option(200, "--count"),
    specimen_ids_file: Path | None = typer.Option(
        None, "--specimen-ids-file",
        help="JSON list of specimen IDs; overrides --start/--count.",
    ),
    out: Path = typer.Option(Path("runs/holdout"), "--out", "-o"),
    max_steps: int = typer.Option(16, "--max-steps"),
    ablation: str = typer.Option("V4", "--ablation"),
    llm_temperature: float = typer.Option(0.4, "--llm-temperature"),
    literature_compare_energy: bool = typer.Option(
        False, "--literature-compare-energy/--no-literature-compare-energy",
    ),
    device: str = typer.Option("auto", "--device"),
    log_every: int = typer.Option(10, "--log-every"),
) -> None:
    """Run probe-augmented Pipeline A on a list of specimens."""
    cfg = load_config(config)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if specimen_ids_file is not None:
        with specimen_ids_file.open("r") as f:
            specimen_ids = list(json.load(f))
        if not all(isinstance(x, int) for x in specimen_ids):
            raise typer.BadParameter(
                f"{specimen_ids_file} must be a JSON list of ints"
            )
        run_slug = f"baseline-full-probes-{len(specimen_ids)}-holdout"
    else:
        specimen_ids = list(range(start, start + count))
        run_slug = f"baseline-full-probes-{count}"

    if probe_bank_dir is None:
        probe_bank_dir = _latest_dir(Path("checkpoints/probes"))
        if probe_bank_dir is None:
            raise typer.BadParameter(
                "no probe bank under checkpoints/probes/. Run "
                "scripts/train_probe_bank.sh first."
            )

    # Resume detection: if a previous run-id under runs/holdout/full_probes/
    # has a non-empty trajectories.jsonl, append rather than overwrite
    # and skip already-processed specimens.
    full_probes_root = out / "full_probes"
    resume_already_done: set[int] = set()
    resume_dir: Path | None = None
    if full_probes_root.exists():
        for d in sorted(full_probes_root.iterdir(), key=lambda p: p.name, reverse=True):
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
        out_dir = out / "full_probes" / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        run_mode = "fresh"
    configure_logging(out_dir)

    typer.echo(f"==> Run id      : {run_id}")
    typer.echo(f"==> Output      : {out_dir}")
    typer.echo(f"==> Probe bank  : {probe_bank_dir}")
    typer.echo(f"==> Adapter     : {adapter_path or '(none, base LLM only)'}")
    typer.echo(f"==> Specimens   : {len(specimen_ids)}")

    # Dataset, FMs, runners ------------------------------------------------
    dataset = LJSpecimenDataset(h5_path)
    runners = build_runners_from_checkpoints(
        checkpoint_root=checkpoint_root,
        train_split=train_split,
        dataset=dataset,
        cfg=cfg,
        device=device,
    )

    # FM2 + probe bank for probe extraction --------------------------------
    fm2_ckpt = _latest_completed_fm2(checkpoint_root, train_split)
    fm2_for_probes = build_fm2_model(cfg.fm2).to(device)
    load_checkpoint(fm2_ckpt / "model.pt", model=fm2_for_probes, map_location=device)
    fm2_for_probes.eval()
    for p in fm2_for_probes.parameters():
        p.requires_grad = False

    bank = ProbeBank.load(probe_bank_dir, device=device).eval()
    typer.echo(f"==> Probes      : {bank.names()}")

    # Verifier -------------------------------------------------------------
    verifier = build_default_verifier(
        literature_db_path=literature_db,
        literature_compare_energy=literature_compare_energy,
    )

    # LLM ------------------------------------------------------------------
    typer.echo(f"==> Loading LLM : {base_model}"
               f"{' + adapter' if adapter_path is not None else ''}")
    llm = TransformersLLM(
        model_name=base_model,
        device=device,
        temperature=llm_temperature,
        adapter_path=str(adapter_path) if adapter_path is not None else None,
    )

    # OHVD loop ------------------------------------------------------------
    loop = OHVDLoop(
        llm=llm,
        verifier=verifier,
        runners=runners,
        max_steps=max_steps,
        sources_config=SourcesConfig.for_ablation(ablation),
    )

    # Pre-compute probes for all specimens ---------------------------------
    todo = [s for s in specimen_ids if int(s) not in resume_already_done]
    typer.echo(f"==> Computing probes for {len(todo)} specimens")
    probes_by_sid: dict[int, dict[str, dict[str, Any]]] = {}
    with h5py.File(h5_path, "r") as h5:
        for start_i in range(0, len(todo), 64):
            batch_ids = todo[start_i : start_i + 64]
            rdfs_np = np.stack(
                [np.asarray(h5["rdfs"][i]) for i in batch_ids], axis=0,
            ).astype(np.float32)
            rdfs = torch.from_numpy(rdfs_np).to(device).float()
            with torch.no_grad():
                hidden = fm2_for_probes.encode(rdfs)
                cls = hidden[:, 0, :]
            outs = bank.evaluate(cls)
            for sid, out in zip(batch_ids, outs, strict=True):
                probes_by_sid[int(sid)] = out

    # Run OHVD loop with enriched per-specimen query -----------------------
    jsonl_path = out_dir / "trajectories.jsonl"
    counters: dict[str, int] = {
        "total": 0,
        "committed_pass": 0,
        "committed_caveat": 0,
        "committed_fail": 0,
        "committed_skip": 0,
        "budget_exhausted": 0,
        "parse_failure": 0,
        "llm_error": 0,
        "skipped_resume": len(specimen_ids) - len(todo),
    }
    typer.echo("")
    typer.echo(f"==> Running OHVD ({run_mode})")
    typer.echo("")

    write_mode = "a" if run_mode == "resume" else "w"
    with jsonl_path.open(write_mode) as out_f:
        for i, sid in enumerate(todo):
            probes = probes_by_sid.get(int(sid))
            if probes is None:
                continue
            probes_json = _format_probes_for_prompt(probes)
            query = _enriched_query(probes_json)

            traj = loop.run(query=query, specimen_id=int(sid))
            counters["total"] += 1

            term = traj.termination.value
            if term == "committed":
                if traj.final_verdict is not None:
                    decision = traj.final_verdict.aggregate_decision.value
                    counters[f"committed_{decision}"] = counters.get(
                        f"committed_{decision}", 0,
                    ) + 1
                else:
                    counters["committed_skip"] += 1
            elif term in counters:
                counters[term] += 1

            # Inject probe metadata into the trajectory for traceability.
            if traj.metadata is None:
                traj.metadata = {}
            traj.metadata["baseline"] = "full_probes"
            traj.metadata["probe_bank_dir"] = str(probe_bank_dir)
            traj.metadata["probes_summary"] = {
                name: {
                    "prediction": probes[name].get("prediction"),
                    "confidence": float(probes[name].get("confidence", 0.0)),
                }
                for name in bank.names()
            }

            out_f.write(traj.model_dump_json() + "\n")
            out_f.flush()

            if (i + 1) % log_every == 0 or i == 0 or i == len(todo) - 1:
                typer.echo(
                    f"    {i + 1:>4}/{len(todo)} sid={int(sid):<6} "
                    f"pass={counters['committed_pass']} "
                    f"caveat={counters['committed_caveat']} "
                    f"fail={counters['committed_fail']}"
                )

    typer.echo(f"==> JSONL: {jsonl_path}")
    typer.echo(json.dumps(counters, indent=2))

    with (out_dir / "summary.yaml").open("w") as f:
        yaml.safe_dump(
            {
                "baseline": "full_probes",
                "counters": counters,
                "completed_utc": datetime.now(UTC).isoformat(),
            },
            f,
        )
    write_manifest(
        out_dir / "manifest.yaml",
        script="scripts.run_baseline_full_probes",
        inputs={
            "h5_path": str(h5_path),
            "fm2_checkpoint": str(fm2_ckpt),
            "probe_bank_dir": str(probe_bank_dir),
            "base_model": base_model,
            "adapter_path": str(adapter_path) if adapter_path is not None else None,
            "n_specimens": len(specimen_ids),
        },
        config={
            "run_id": run_id,
            "max_steps": max_steps,
            "ablation": ablation,
            "llm_temperature": llm_temperature,
            "literature_compare_energy": literature_compare_energy,
        },
        extra={"counters": counters},
    )

    dataset.close()


if __name__ == "__main__":
    app()
