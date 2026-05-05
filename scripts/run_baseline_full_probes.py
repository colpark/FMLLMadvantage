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
from fmllm.representation.sae import TopKSAE  # noqa: E402
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


def _enriched_query(probes_json: str, sae_features_str: str = "") -> str:
    parts = [
        _BASE_QUERY,
        "",
        f"PROBES (derived from a frozen FM2 representation, treat as "
        f"approximate hints, not ground truth): {probes_json}",
    ]
    if sae_features_str:
        parts.append("")
        parts.append(
            f"SAE_FEATURES (top-k labelled directions in FM2's "
            f"representation that activated for this specimen): "
            f"{sae_features_str}"
        )
    return "\n".join(parts)


def _load_sae_and_labels(
    sae_dir: Path, labels_path: Path | None, device: str,
) -> tuple[TopKSAE, "np.ndarray", "np.ndarray", dict[int, str]]:
    """Load the trained SAE plus its feature label mapping."""
    payload = torch.load(sae_dir / "sae.pt", map_location=device, weights_only=False)
    sae = TopKSAE(
        in_dim=int(payload["in_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        k=int(payload["k"]),
    ).to(device)
    sae.load_state_dict(payload["state_dict"], strict=True)
    sae.eval()
    cls_mean = np.asarray(payload["cls_mean"], dtype=np.float32)
    cls_std = np.asarray(payload["cls_std"], dtype=np.float32)
    if labels_path is None:
        # Auto-discover the latest under runs/sae_labels/.
        cands = sorted(
            Path("runs/sae_labels").glob("*/labels.json"),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        labels_path = cands[0] if cands else None
    if labels_path is None or not labels_path.exists():
        raise typer.BadParameter(
            "no SAE labels found. Run scripts/label_sae_features.sh first."
        )
    with labels_path.open("r") as f:
        labels_raw = json.load(f)
    labels = {int(k): str(v) for k, v in labels_raw.items()}
    return sae, cls_mean, cls_std, labels


def _format_sae_features(
    *,
    activations: "np.ndarray",       # (hidden_dim,)
    labels: dict[int, str],
    top_k_prompt: int,
    causal_filter: set[int] | None = None,
) -> str:
    """Take the top-k active features for a specimen and render their
    labels + activation values as a compact string.

    When ``causal_filter`` is provided, restricts the candidate
    feature set to those indices first; this is the Phase 14 path
    that surfaces only causally meaningful features to the LLM.
    """
    if activations.size == 0:
        return ""
    nonzero_idx = np.nonzero(activations)[0]
    if causal_filter is not None:
        nonzero_idx = np.array(
            [i for i in nonzero_idx if int(i) in causal_filter], dtype=np.int64,
        )
    if nonzero_idx.size == 0:
        return ""
    nonzero_acts = activations[nonzero_idx]
    order = np.argsort(nonzero_acts)[::-1][:top_k_prompt]
    top_idx = nonzero_idx[order]
    top_act = nonzero_acts[order]
    items = []
    for i, a in zip(top_idx, top_act, strict=True):
        label = labels.get(int(i), f"f{int(i)}")
        items.append(f'"{label}": {float(a):.2f}')
    return "{" + ", ".join(items) + "}"


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
    sae_dir: Path | None = typer.Option(
        None, "--sae-dir",
        help="Trained SAE directory. When set, the runner forwards "
             "each specimen through FM2 + SAE and injects the top-k "
             "labelled features into the user message alongside the "
             "PROBES payload. Output then goes to runs/holdout/full_sae/ "
             "instead of runs/holdout/full_probes/.",
    ),
    sae_labels_path: Path | None = typer.Option(
        None, "--sae-labels-path",
        help="labels.json for the SAE. Default: latest under "
             "runs/sae_labels/.",
    ),
    sae_top_k_prompt: int = typer.Option(
        8, "--sae-top-k-prompt",
        help="How many top-active SAE features to surface per specimen.",
    ),
    causal_filter_path: Path | None = typer.Option(
        None, "--causal-filter-path",
        help="Path to a causal_filter.json from scripts/sae_causal_audit.py. "
             "When set, only feature indices listed under "
             "passing_feature_ids are eligible for the SAE prompt slot. "
             "Output then goes to runs/holdout/full_sae_causal/ instead "
             "of runs/holdout/full_sae/.",
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

    use_sae = sae_dir is not None
    use_causal_filter = use_sae and causal_filter_path is not None
    if use_causal_filter:
        baseline_subdir = "full_sae_causal"
        baseline_label = "full_sae_causal"
    elif use_sae:
        baseline_subdir = "full_sae"
        baseline_label = "full_sae"
    else:
        baseline_subdir = "full_probes"
        baseline_label = "full_probes"

    causal_filter_set: set[int] | None = None
    if use_causal_filter:
        with causal_filter_path.open("r") as f:
            cf = json.load(f)
        causal_filter_set = {int(x) for x in cf.get("passing_feature_ids", [])}
        if not causal_filter_set:
            raise typer.BadParameter(
                f"causal_filter_path={causal_filter_path} has zero "
                f"passing_feature_ids; nothing to surface to the LLM."
            )

    if specimen_ids_file is not None:
        with specimen_ids_file.open("r") as f:
            specimen_ids = list(json.load(f))
        if not all(isinstance(x, int) for x in specimen_ids):
            raise typer.BadParameter(
                f"{specimen_ids_file} must be a JSON list of ints"
            )
        run_slug = f"baseline-{baseline_label}-{len(specimen_ids)}-holdout"
    else:
        specimen_ids = list(range(start, start + count))
        run_slug = f"baseline-{baseline_label}-{count}"

    if probe_bank_dir is None:
        probe_bank_dir = _latest_dir(Path("checkpoints/probes"))
        if probe_bank_dir is None:
            raise typer.BadParameter(
                "no probe bank under checkpoints/probes/. Run "
                "scripts/train_probe_bank.sh first."
            )

    # Resume detection: scan the matching subdir for a non-empty
    # trajectories.jsonl. With SAE on, the subdir is full_sae;
    # otherwise full_probes.
    base_subdir_root = out / baseline_subdir
    resume_already_done: set[int] = set()
    resume_dir: Path | None = None
    if base_subdir_root.exists():
        for d in sorted(base_subdir_root.iterdir(), key=lambda p: p.name, reverse=True):
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
        out_dir = out / baseline_subdir / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        run_mode = "fresh"
    configure_logging(out_dir)

    typer.echo(f"==> Run id      : {run_id}")
    typer.echo(f"==> Output      : {out_dir}")
    typer.echo(f"==> Probe bank  : {probe_bank_dir}")
    typer.echo(f"==> SAE         : {sae_dir or '(none)'}")
    typer.echo(
        f"==> Causal flt  : "
        f"{causal_filter_path if use_causal_filter else '(none)'}"
        + (
            f" ({len(causal_filter_set)} features)"
            if use_causal_filter and causal_filter_set
            else ""
        )
    )
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

    # SAE + labels (optional) ----------------------------------------------
    sae: TopKSAE | None = None
    sae_cls_mean = None
    sae_cls_std = None
    sae_labels: dict[int, str] = {}
    if use_sae:
        sae, sae_cls_mean, sae_cls_std, sae_labels = _load_sae_and_labels(
            sae_dir=sae_dir, labels_path=sae_labels_path, device=device,
        )
        typer.echo(
            f"==> SAE config  : in_dim={sae.in_dim} hidden_dim={sae.hidden_dim} "
            f"k={sae.k} | {len(sae_labels)} labels loaded"
        )

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

    # Pre-compute probes (and SAE features) for all specimens -------------
    todo = [s for s in specimen_ids if int(s) not in resume_already_done]
    typer.echo(f"==> Computing probes/SAE for {len(todo)} specimens")
    probes_by_sid: dict[int, dict[str, dict[str, Any]]] = {}
    sae_features_by_sid: dict[int, np.ndarray] = {}

    sae_cls_mean_t = (
        torch.from_numpy(sae_cls_mean).to(device) if use_sae else None
    )
    sae_cls_std_t = (
        torch.from_numpy(sae_cls_std).to(device) if use_sae else None
    )

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
            if use_sae and sae is not None:
                with torch.no_grad():
                    cls_norm = (cls - sae_cls_mean_t) / sae_cls_std_t
                    z = sae.encode(cls_norm)
                z_np = z.detach().cpu().numpy()
                for sid, row in zip(batch_ids, z_np, strict=True):
                    sae_features_by_sid[int(sid)] = row

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
            sae_features_str = ""
            if use_sae:
                acts = sae_features_by_sid.get(int(sid))
                if acts is not None:
                    sae_features_str = _format_sae_features(
                        activations=acts,
                        labels=sae_labels,
                        top_k_prompt=sae_top_k_prompt,
                        causal_filter=causal_filter_set,
                    )
            query = _enriched_query(probes_json, sae_features_str)

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
            traj.metadata["baseline"] = baseline_label
            traj.metadata["probe_bank_dir"] = str(probe_bank_dir)
            traj.metadata["probes_summary"] = {
                name: {
                    "prediction": probes[name].get("prediction"),
                    "confidence": float(probes[name].get("confidence", 0.0)),
                }
                for name in bank.names()
            }
            if use_sae:
                traj.metadata["sae_dir"] = str(sae_dir)
                traj.metadata["sae_features_in_prompt"] = sae_features_str
                if use_causal_filter:
                    traj.metadata["causal_filter_path"] = str(causal_filter_path)
                    traj.metadata["causal_filter_n_passing"] = (
                        len(causal_filter_set) if causal_filter_set else 0
                    )

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
                "baseline": baseline_label,
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
            "sae_dir": str(sae_dir) if sae_dir is not None else None,
            "sae_labels_path": str(sae_labels_path) if sae_labels_path is not None else None,
            "causal_filter_path": (
                str(causal_filter_path) if causal_filter_path is not None else None
            ),
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
