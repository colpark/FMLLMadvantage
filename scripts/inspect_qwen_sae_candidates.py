"""CLI: inspect Stage C steering candidates and diagnose wrong-PASS pattern.

Phase 15 helper. Two outputs:

  1. Pretty-print the top-K candidates from
     ``runs/qwen_sae_labels/<latest>/steering_candidates.yaml`` with
     None-safe formatting (so caveat candidates with no correctness
     lock don't crash the formatter).

  2. Diagnose the actual wrong-PASS distribution in the source
     ``runs/holdout/full/<latest>/trajectories.jsonl``: of all PASS
     commits, group the wrong ones by (motif, phase). The Stage C
     SAE features lock heavily on ``(triangular_disk, solid-like)``,
     and this diagnostic confirms or refutes that pattern in the
     ground-truth data, validating the candidate features before
     Stage D steering.

Usage:

    bash scripts/inspect_qwen_sae_candidates.sh
    # Or: uv run python scripts/inspect_qwen_sae_candidates.py --top-k 8

Depends on:
    typer, h5py, numpy, pyyaml.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import typer
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


app = typer.Typer(add_completion=False, no_args_is_help=False)


def _phase_for(temperature: float) -> str:
    """Same heuristic as fmllm.connectors.text_annotations._phase_for."""
    return "solid-like" if temperature < 0.6 else "liquid-like"


def _fmt_float(x: object) -> str:
    """Format a maybe-None numeric field for table output."""
    if isinstance(x, (int, float)):
        return f"{x:.2f}"
    return "----"


def _latest_with(parent: Path, sub: str) -> Path | None:
    if not parent.exists():
        return None
    cands = sorted(
        parent.glob(f"*/{sub}"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    return cands[0] if cands else None


def _print_candidates(
    candidates_path: Path, top_k: int,
    sections: tuple[str, ...] = ("wrong_pass", "wrong_any", "caveat"),
) -> None:
    typer.echo(f"==> Candidates source: {candidates_path}")
    data = yaml.safe_load(candidates_path.read_text())
    for k in sections:
        items = data.get(k, []) or []
        typer.echo(f"\n--- {k} (n={len(items)}) ---")
        for c in items[:top_k]:
            typer.echo(
                f"  fid={c['feature_idx']:>5}  "
                f"verdict={(c.get('verdict_top') or '--'):>7}"
                f"({_fmt_float(c.get('verdict_purity'))})  "
                f"correct={str(c.get('correct_top')):>5}"
                f"({_fmt_float(c.get('correct_purity'))})  "
                f"n_top={c.get('n_top_activators', 0):>3}"
            )
            typer.echo(f"          {c.get('label', '?')}")


def _diagnose_wrong_pass(
    trajectories_path: Path, h5_path: Path,
) -> None:
    """Group wrong-PASS commits by (motif, phase) and print a table."""
    typer.echo(f"\n==> 'full' source     : {trajectories_path}")
    typer.echo(f"==> Specimens HDF5    : {h5_path}")

    wrong_by: collections.Counter = collections.Counter()
    right_by: collections.Counter = collections.Counter()
    total_pass = 0
    wrong_pass = 0

    with h5py.File(h5_path, "r") as h5:
        motif_names: list[str] = []
        if "motif_names" in h5.attrs:
            motif_names = [
                s.decode() if isinstance(s, bytes) else str(s)
                for s in h5.attrs["motif_names"]
            ]
        with trajectories_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    t = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sid = t.get("specimen_id")
                if not isinstance(sid, int):
                    continue
                v = (t.get("final_verdict") or {}).get("aggregate_decision")
                if v != "pass":
                    continue
                total_pass += 1
                fc = t.get("final_claim") or {}
                mid = int(np.asarray(h5["motif_ids"][sid]))
                n_gt = int(np.asarray(h5["atom_counts"][sid]))
                t_gt = float(np.asarray(h5["temperatures"][sid]))
                motif_gt = (
                    motif_names[mid] if 0 <= mid < len(motif_names) else str(mid)
                )
                phase_gt = _phase_for(t_gt)
                try:
                    motif_ok = (
                        str(fc.get("motif", "")).lower() == motif_gt.lower()
                    )
                    n_ok = int(fc.get("n_atoms", -1)) == n_gt
                    t_ok = abs(
                        float(fc.get("temperature", -999.0)) - t_gt
                    ) <= 0.10
                except (TypeError, ValueError):
                    motif_ok = n_ok = t_ok = False
                key = (motif_gt, phase_gt)
                if motif_ok and n_ok and t_ok:
                    right_by[key] += 1
                else:
                    wrong_pass += 1
                    wrong_by[key] += 1

    typer.echo(
        f"\n--- 'full' PASS commits ({total_pass} total, "
        f"{wrong_pass} wrong) by (motif, phase) ---"
    )
    typer.echo(f"  {'group':<32} {'wrong':>5}  {'right':>5}  {'frac_wrong':>10}")
    keys = sorted(set(list(wrong_by) + list(right_by)))
    for k in keys:
        w = wrong_by.get(k, 0)
        r = right_by.get(k, 0)
        frac = (w / (w + r)) if (w + r) else 0.0
        label = f"{k[0]}, {k[1]}"
        typer.echo(f"  {label:<32} {w:>5}  {r:>5}  {frac:>10.2f}")


@app.command()
def main(
    candidates: Path | None = typer.Option(
        None, "--candidates",
        help="Path to a steering_candidates.yaml. Default: latest "
             "under runs/qwen_sae_labels/.",
    ),
    trajectories: Path | None = typer.Option(
        None, "--trajectories",
        help="Path to the 'full' baseline trajectories.jsonl. "
             "Default: latest under runs/holdout/full/.",
    ),
    h5_path: Path = typer.Option(
        Path("data/synthetic_lj_v1/specimens.h5"), "--h5-path",
    ),
    top_k: int = typer.Option(
        8, "--top-k",
        help="How many candidates to show per section.",
    ),
    diagnose: bool = typer.Option(
        True, "--diagnose/--no-diagnose",
        help="If true, also tabulate wrong-PASS distribution by "
             "(motif, phase) from the source 'full' trajectories.",
    ),
) -> None:
    """Pretty-print steering candidates and (optionally) diagnose wrong-PASS."""
    if candidates is None:
        candidates = _latest_with(
            Path("runs/qwen_sae_labels"), "steering_candidates.yaml",
        )
    if candidates is None or not candidates.exists():
        raise typer.BadParameter(
            "no steering_candidates.yaml found under runs/qwen_sae_labels/. "
            "Run scripts/label_qwen_sae_features.sh first."
        )
    _print_candidates(candidates, top_k=top_k)

    if not diagnose:
        return

    if trajectories is None:
        trajectories = _latest_with(
            Path("runs/holdout/full"), "trajectories.jsonl",
        )
    if trajectories is None or not trajectories.exists():
        typer.echo(
            "\n(skipping wrong-PASS diagnostic: no trajectories.jsonl under "
            "runs/holdout/full/)"
        )
        return
    if not h5_path.exists():
        typer.echo(
            f"\n(skipping wrong-PASS diagnostic: H5 file not found at {h5_path})"
        )
        return

    _diagnose_wrong_pass(
        trajectories_path=trajectories, h5_path=h5_path,
    )


if __name__ == "__main__":
    app()
