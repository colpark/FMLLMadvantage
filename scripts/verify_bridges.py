"""CLI: load FMContext + bridges for each trained checkpoint and emit
a synthetic bridged output for inspection.

The script does not run an FM forward pass. It only confirms that the
on-disk artifacts (``metadata.yaml``, ``probe_report.yaml``,
``calibration.json``) compose into a working :class:`FMContext` and
that both bridges produce well-formed outputs against synthetic raw
inputs. Useful as a quick post-training verification of Phase 3
wiring before Phase 4 lands.

Usage:
    uv run python scripts/verify_bridges.py \\
        --checkpoint-root checkpoints \\
        --scale train_50k \\
        --out runs/bridge-verify

For each FM in (fm1_image, fm2_rdf, fm3_traj) the script writes:
    <out>/<fm>/structure.json   # BridgedFMOutput as JSON
    <out>/<fm>/caption.txt      # language-anchored caption
    <out>/<fm>/context.yaml     # snapshot of the FMContext

Plus a top-level summary.yaml with a per-FM pass/fail line.

Depends on:
    torch, typer, pyyaml, loguru.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import typer
import yaml

# Ensure the package is importable when running from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fmllm.bridges import (  # noqa: E402
    load_fm_context,
    make_language_bridge,
    make_structure_bridge,
)
from fmllm.utils.manifests import write_manifest  # noqa: E402
from fmllm.utils.run_ids import generate_run_id  # noqa: E402


FM_TO_DIR = {
    "fm1_image": "fm1_image",
    "fm2_rdf": "fm2_rdf",
    "fm3_traj": "fm3_traj",
}


def _synthetic_raw(fm_name: str) -> dict:
    """A minimal raw FM output the bridges accept."""
    if fm_name == "fm1_image":
        return {
            "count_logits": torch.cat([
                torch.full((30,), -3.0),
                torch.tensor([5.0]),
            ]),
            "positions": torch.tensor([
                [0.5, 0.3],
                [-1.2, 0.7],
                [0.0, -1.5],
            ]),
            "confidence_logits": torch.tensor([3.0, 2.5, 1.5]),
        }
    if fm_name == "fm2_rdf":
        return {"energy": torch.tensor(-1.42)}
    if fm_name == "fm3_traj":
        return {"alpha": torch.tensor(2.0), "beta": torch.tensor(0.55)}
    raise ValueError(f"unknown fm_name {fm_name!r}")


def _latest_checkpoint_dir(checkpoint_root: Path, fm_name: str, scale: str) -> Path | None:
    pattern = checkpoint_root / FM_TO_DIR[fm_name] / scale / "*"
    matches = sorted(checkpoint_root.glob(f"{FM_TO_DIR[fm_name]}/{scale}/*"))
    matches = [m for m in matches if m.is_dir()]
    if not matches:
        return None
    return matches[-1]


def _verify_one(fm_name: str, ckpt_dir: Path, out_dir: Path) -> dict:
    """Run the bridges for one FM checkpoint. Returns a status dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = load_fm_context(fm_name=fm_name, checkpoint_dir=ckpt_dir)
    raw = _synthetic_raw(fm_name)

    struct_bridge = make_structure_bridge(ctx)
    lang_bridge = make_language_bridge(ctx)

    bridged = struct_bridge.emit(raw, input_provenance={"specimen_id": "synthetic"})
    caption = lang_bridge.emit(raw, input_provenance={"specimen_id": "synthetic"})

    (out_dir / "structure.json").write_text(
        json.dumps(json.loads(bridged.model_dump_json()), indent=2, sort_keys=True)
    )
    (out_dir / "caption.txt").write_text(caption + "\n")

    ctx_snapshot = {
        "fm_name": ctx.fm_name,
        "metadata_constraints": [c.name for c in ctx.metadata.physics_constraints],
        "metadata_dependencies": [
            {"target": d.target_variable, "relationship": d.relationship}
            for d in ctx.metadata.dependencies
        ],
        "probe_report": {
            "n_results": len(ctx.probe_report.results),
            "scores": {
                r.constraint_name: float(r.satisfaction_score)
                for r in ctx.probe_report.results
            },
        },
        "calibration_thresholds": (ctx.calibration or {}).get("thresholds", {}),
    }
    with (out_dir / "context.yaml").open("w") as f:
        yaml.safe_dump(ctx_snapshot, f, sort_keys=False)

    return {
        "fm_name": fm_name,
        "checkpoint_dir": str(ckpt_dir),
        "status": "ok",
        "n_constraints": len(bridged.applicable_constraints),
        "n_dependencies": len(bridged.dependencies),
        "uncertainty_present": bridged.prediction.uncertainty is not None,
        "calibration_thresholds_loaded": bool(ctx.calibration),
        "probe_results_loaded": len(ctx.probe_report.results),
    }


app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def main(
    checkpoint_root: Path = typer.Option(
        Path("checkpoints"), "--checkpoint-root",
        help="Root directory containing per-FM checkpoint subtrees.",
    ),
    scale: str = typer.Option(
        "train_50k", "--scale",
        help="Training scale subdirectory under each FM (e.g. train_10k, train_30k, train_50k).",
    ),
    out: Path = typer.Option(
        Path("runs/bridge-verify"), "--out", "-o",
        help="Output directory. The script creates a run-id subdirectory inside.",
    ),
) -> None:
    """Verify bridges load and emit correctly against actual checkpoint artifacts."""
    if not checkpoint_root.exists():
        raise typer.BadParameter(f"checkpoint root not found: {checkpoint_root}")

    run_id = generate_run_id(f"bridge-verify-{scale}")
    out_dir = out / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    typer.echo(f"==> Run id: {run_id}")
    typer.echo(f"==> Output: {out_dir}")

    summaries: list[dict] = []
    all_ok = True
    for fm_name in FM_TO_DIR:
        ckpt = _latest_checkpoint_dir(checkpoint_root, fm_name, scale)
        if ckpt is None:
            summary = {
                "fm_name": fm_name,
                "status": "missing_checkpoint",
                "scale": scale,
            }
            summaries.append(summary)
            all_ok = False
            typer.echo(f"  {fm_name:12s} : MISSING (no run under {checkpoint_root}/{fm_name}/{scale}/)")
            continue
        try:
            summary = _verify_one(fm_name, ckpt, out_dir / fm_name)
            summaries.append(summary)
            typer.echo(
                f"  {fm_name:12s} : OK "
                f"(constraints={summary['n_constraints']}, "
                f"deps={summary['n_dependencies']}, "
                f"uncertainty={summary['uncertainty_present']}, "
                f"calibration={summary['calibration_thresholds_loaded']})"
            )
        except Exception as exc:  # noqa: BLE001
            summaries.append({
                "fm_name": fm_name,
                "status": "error",
                "error": repr(exc),
                "checkpoint_dir": str(ckpt),
            })
            typer.echo(f"  {fm_name:12s} : ERROR {exc!r}")
            all_ok = False

    with (out_dir / "summary.yaml").open("w") as f:
        yaml.safe_dump(
            {"run_id": run_id, "scale": scale, "results": summaries},
            f, sort_keys=False,
        )

    write_manifest(
        out_dir / "manifest.yaml",
        script="scripts.verify_bridges",
        inputs={"checkpoint_root": str(checkpoint_root), "scale": scale},
        config={"run_id": run_id},
        extra={"summaries": summaries, "all_ok": all_ok},
    )
    typer.echo(f"==> Summary: {out_dir}/summary.yaml")
    raise typer.Exit(0 if all_ok else 1)


if __name__ == "__main__":
    app()
