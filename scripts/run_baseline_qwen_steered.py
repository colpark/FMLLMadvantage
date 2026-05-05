"""CLI: run Pipeline A with Qwen activation steering applied at inference.

Phase 15 Stage D. The full Phase 8a pipeline (FM tools + multi-source
verifier), but every Qwen forward pass during the OHVD loop has an
:class:`ActivationSteerer` hook attached at a specified residual-
stream layer. The hook adds ``coefficient * decoder_column[fid]``
to the residual at the hooked layer, broadcasting across every
generated token.

Output goes to ``runs/holdout/full_steered_<fid>_<coef>/<run_id>/``
so ``scripts/evaluate_baselines.sh`` auto-discovers it as a new
column. The directory name encodes the steering parameters so
multiple experiments coexist:

    full_steered_8421_p200    # feature 8421, coefficient = +2.00
    full_steered_8421_n100    # feature 8421, coefficient = -1.00

The expected use case:

  1. Stage C produced ``steering_candidates.yaml`` with a list of
     "wrong-PASS" features (those that fire on confidently-wrong
     commits).
  2. For one such feature, run this CLI with a *negative*
     coefficient to ablate it during inference.
  3. Compare against ``full`` on the held-out set; a positive
     result is reduced hallucination_rate at unchanged commit_rate.

Usage:

    bash scripts/run_baseline_qwen_steered.sh

Env-equivalent flags:

    --feature-idx <int>         which SAE feature direction to inject
    --coefficient <float>       additive multiplier (negative ablates)
    --sae-dir <path>            checkpoints/qwen_sae/<run_id> (default: latest)
    --layer-path <str>          must match the SAE's training layer

Depends on:
    typer, torch, pyyaml.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import typer
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fmllm.data.dataset import LJSpecimenDataset  # noqa: E402
from fmllm.orchestrator import (  # noqa: E402
    TransformersLLM,
    build_runners_from_checkpoints,
)
from fmllm.representation.steered_llm import SteeredLLMWrapper  # noqa: E402
from fmllm.training import collect_trajectories  # noqa: E402
from fmllm.utils.config import load_config  # noqa: E402
from fmllm.utils.logging import configure_logging  # noqa: E402
from fmllm.utils.manifests import write_manifest  # noqa: E402
from fmllm.utils.run_ids import generate_run_id  # noqa: E402
from fmllm.verifier import SourcesConfig, build_default_verifier  # noqa: E402


app = typer.Typer(add_completion=False, no_args_is_help=False)


def _latest_dir(parent: Path) -> Path | None:
    if not parent.exists():
        return None
    cands = sorted(
        parent.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True,
    )
    return next((c for c in cands if c.is_dir()), None)


def _coef_str(coef: float) -> str:
    """Encode a steering coefficient as a directory-safe string.

    Examples: 2.0 -> 'p200', -1.5 -> 'n150', 0.5 -> 'p050'.
    """
    sign = "p" if coef >= 0 else "n"
    return f"{sign}{int(round(abs(coef) * 100)):03d}"


@app.command()
def main(
    feature_idx: int = typer.Option(
        ..., "--feature-idx",
        help="Index of the SAE feature whose decoder direction is "
             "added to the residual stream during generation.",
    ),
    coefficient: float = typer.Option(
        -1.0, "--coefficient",
        help="Multiplier on the feature direction. Negative values "
             "ablate the feature; positive amplify. Templeton et al. "
             "used 5-10x for 'obsessive' steering; 0.5-3x is more "
             "useful for behaviorally meaningful nudges.",
    ),
    sae_dir: Path | None = typer.Option(
        None, "--sae-dir",
        help="Trained Qwen SAE directory. Default: latest under "
             "checkpoints/qwen_sae/.",
    ),
    layer_path: str = typer.Option(
        "model.layers.14", "--layer-path",
        help="Dotted path to the layer being steered. Must match the "
             "SAE's training layer.",
    ),
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
    base_model: str = typer.Option(
        "Qwen/Qwen2.5-7B-Instruct", "--base-model",
    ),
    adapter_path: Path | None = typer.Option(
        None, "--adapter-path",
    ),
    start: int = typer.Option(0, "--start"),
    count: int = typer.Option(200, "--count"),
    specimen_ids_file: Path | None = typer.Option(
        None, "--specimen-ids-file",
    ),
    out: Path = typer.Option(Path("runs/holdout"), "--out", "-o"),
    max_steps: int = typer.Option(16, "--max-steps"),
    ablation: str = typer.Option("V4", "--ablation"),
    llm_temperature: float = typer.Option(0.4, "--llm-temperature"),
    literature_compare_energy: bool = typer.Option(
        False, "--literature-compare-energy/--no-literature-compare-energy",
    ),
    query: str = typer.Option(
        "Identify the specimen's atom count, motif, and temperature. "
        "Use FM tools to gather evidence, propose a claim, and commit when confident.",
        "--query",
    ),
    device: str = typer.Option("auto", "--device"),
) -> None:
    """Run Pipeline A on held-out specimens with one SAE feature steered."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if sae_dir is None:
        sae_dir = _latest_dir(Path("checkpoints/qwen_sae"))
        if sae_dir is None:
            raise typer.BadParameter(
                "no Qwen SAE under checkpoints/qwen_sae/. Run Stages A/B first."
            )
    if not (sae_dir / "sae.pt").exists():
        raise typer.BadParameter(f"no sae.pt under {sae_dir}")

    # Resolve specimens ----------------------------------------------------
    if specimen_ids_file is not None:
        with specimen_ids_file.open("r") as f:
            specimen_ids = list(json.load(f))
        if not all(isinstance(x, int) for x in specimen_ids):
            raise typer.BadParameter(
                f"{specimen_ids_file} must be a JSON list of ints"
            )
    else:
        specimen_ids = list(range(start, start + count))

    coef_str = _coef_str(coefficient)
    baseline_label = f"full_steered_{feature_idx}_{coef_str}"
    run_slug = (
        f"baseline-{baseline_label}-{len(specimen_ids)}-holdout"
        if specimen_ids_file is not None
        else f"baseline-{baseline_label}-{len(specimen_ids)}"
    )
    run_id = generate_run_id(run_slug)
    out_dir = out / baseline_label / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(out_dir)

    typer.echo(f"==> Run id      : {run_id}")
    typer.echo(f"==> Output      : {out_dir}")
    typer.echo(f"==> SAE dir     : {sae_dir}")
    typer.echo(f"==> Layer       : {layer_path}")
    typer.echo(f"==> Feature idx : {feature_idx}")
    typer.echo(f"==> Coefficient : {coefficient:+.3f}")
    typer.echo(f"==> Specimens   : {len(specimen_ids)}")

    # Build LLM with steering --------------------------------------------
    base_llm = TransformersLLM(
        model_name=base_model,
        device=device,
        temperature=llm_temperature,
        adapter_path=str(adapter_path) if adapter_path is not None else None,
    )
    llm = SteeredLLMWrapper(
        llm=base_llm,
        sae_dir=sae_dir,
        feature_idx=feature_idx,
        coefficient=coefficient,
        layer_path=layer_path,
    )

    # Standard full-pipeline scaffolding ---------------------------------
    cfg = load_config(config)
    dataset = LJSpecimenDataset(h5_path)
    runners = build_runners_from_checkpoints(
        checkpoint_root=checkpoint_root,
        train_split=train_split,
        dataset=dataset,
        cfg=cfg,
        device=device,
    )
    verifier = build_default_verifier(
        literature_db_path=literature_db,
        literature_compare_energy=literature_compare_energy,
    )

    summary = collect_trajectories(
        llm=llm,
        verifier=verifier,
        runners=runners,
        specimen_ids=specimen_ids,
        out_dir=out_dir,
        query=query,
        max_steps=max_steps,
        sources_config=SourcesConfig.for_ablation(ablation),
        filter_passing=False,
        progress_every=10,            # log every 10 specimens so the
                                       # steered run shows progress
                                       # well before specimen 50.
    )

    write_manifest(
        out_dir / "manifest.yaml",
        script="scripts.run_baseline_qwen_steered",
        inputs={
            "h5_path": str(h5_path),
            "checkpoint_root": str(checkpoint_root),
            "train_split": train_split,
            "literature_db": str(literature_db),
            "literature_compare_energy": literature_compare_energy,
            "base_model": base_model,
            "adapter_path": str(adapter_path) if adapter_path is not None else None,
            "sae_dir": str(sae_dir),
            "specimen_ids_file": (
                str(specimen_ids_file) if specimen_ids_file is not None else None
            ),
            "n_specimens": len(specimen_ids),
        },
        config={
            "run_id": run_id,
            "feature_idx": int(feature_idx),
            "coefficient": float(coefficient),
            "layer_path": layer_path,
            "ablation": ablation,
            "max_steps": max_steps,
            "llm_temperature": llm_temperature,
        },
        extra={"counters": summary["counters"]},
    )


if __name__ == "__main__":
    app()
