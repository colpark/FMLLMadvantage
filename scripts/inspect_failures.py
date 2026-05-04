"""CLI: summarize FAIL trajectories from a baseline run.

Loads a trajectories.jsonl, picks every trajectory whose final
verdict aggregates to FAIL (or whose termination is parse_failure /
llm_error / budget_exhausted), and prints:

* Counts by termination type and aggregate verdict.
* Most-common failing source (rule_library, cross_fm, ...) with
  counts.
* A handful of example specimens with the failing source's message,
  the LLM's last claim, and (if available) ground truth.

Usage:

    uv run python scripts/inspect_failures.py \\
        --trajectories runs/baselines/full/<run-id>/trajectories.jsonl \\
        --h5-path data/synthetic_lj_v1/specimens.h5 \\
        --max-examples 10

Depends on:
    typer, pyyaml, h5py.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fmllm.evaluation.utils import load_ground_truth  # noqa: E402
from fmllm.orchestrator import Trajectory  # noqa: E402


app = typer.Typer(add_completion=False, no_args_is_help=True)


def _load_trajectories(path: Path) -> list[Trajectory]:
    out: list[Trajectory] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(Trajectory.model_validate_json(line))
    return out


def _is_failure(t: Trajectory) -> bool:
    if t.termination.value != "committed":
        return True
    if t.final_verdict is None:
        return True
    return t.final_verdict.aggregate_decision.value == "fail"


@app.command()
def main(
    trajectories: Path = typer.Option(..., "--trajectories", "-t"),
    h5_path: Path = typer.Option(
        Path("data/synthetic_lj_v1/specimens.h5"), "--h5-path",
    ),
    max_examples: int = typer.Option(10, "--max-examples", "-n"),
) -> None:
    """Summarize failure modes in a baseline trajectories.jsonl."""
    if not trajectories.exists():
        typer.echo(f"ERROR: not found: {trajectories}")
        raise typer.Exit(code=1)

    trajs = _load_trajectories(trajectories)
    typer.echo(f"==> Loaded {len(trajs)} trajectories from {trajectories}")

    failing = [t for t in trajs if _is_failure(t)]
    typer.echo(f"==> Failing: {len(failing)} ({100*len(failing)/len(trajs):.1f}%)")
    if not failing:
        typer.echo("No failures to inspect.")
        return

    # Truth lookup (only for specimens we have).
    seen_ids = sorted({
        int(t.specimen_id) for t in failing if t.specimen_id is not None
    })
    truth: dict[int, dict[str, Any]] = {}
    if seen_ids and h5_path.exists():
        truth = load_ground_truth(h5_path, specimen_ids=seen_ids)

    # Aggregate counts.
    by_termination: Counter[str] = Counter()
    by_aggregate: Counter[str] = Counter()
    failing_sources: Counter[str] = Counter()
    by_motif: Counter[str] = Counter()
    by_n_bucket: Counter[str] = Counter()
    failing_constraints: Counter[str] = Counter()

    for t in failing:
        by_termination[t.termination.value] += 1
        if t.final_verdict is not None:
            by_aggregate[t.final_verdict.aggregate_decision.value] += 1
            for sv in t.final_verdict.source_verdicts:
                if sv.decision.value in ("fail", "caveat"):
                    failing_sources[f"{sv.source_name}:{sv.decision.value}"] += 1
                if sv.source_name == "rule_library" and sv.decision.value == "fail":
                    for c in (sv.evidence or {}).get("checks", []) or []:
                        if not c.get("passed", True):
                            cname = c.get("constraint_name", "?")
                            fname = c.get("fm_name", "?")
                            failing_constraints[f"{cname}@{fname}"] += 1
        else:
            by_aggregate["no_verdict"] += 1

        if t.specimen_id is not None and t.specimen_id in truth:
            gt = truth[t.specimen_id]
            by_motif[gt["motif"]] += 1
            n = gt["n"]
            bucket = f"{(n // 5) * 5}-{(n // 5) * 5 + 4}"
            by_n_bucket[bucket] += 1

    typer.echo("")
    typer.echo("Failure breakdown")
    typer.echo("-" * 56)
    typer.echo(f"By termination       : {dict(by_termination)}")
    typer.echo(f"By aggregate verdict : {dict(by_aggregate)}")
    typer.echo(f"By ground-truth motif: {dict(by_motif)}")
    typer.echo(f"By N bucket          : {dict(by_n_bucket)}")
    typer.echo("")
    typer.echo("Top flagging sources (FAIL or CAVEAT)")
    typer.echo("-" * 56)
    for src, n in failing_sources.most_common(10):
        typer.echo(f"  {src:<30} {n:>5}")

    if failing_constraints:
        typer.echo("")
        typer.echo("Failing rule_library constraints")
        typer.echo("-" * 56)
        for c, n in failing_constraints.most_common(10):
            typer.echo(f"  {c:<40} {n:>5}")

    typer.echo("")
    typer.echo(f"First {min(max_examples, len(failing))} failure examples")
    typer.echo("-" * 80)
    for t in failing[:max_examples]:
        gt = truth.get(int(t.specimen_id), {}) if t.specimen_id is not None else {}
        gt_str = (
            f"truth: N={gt['n']} T={gt['t']:.2f} motif={gt['motif']}"
            if gt else "truth: (not loaded)"
        )
        claim = t.final_claim
        claim_str = (
            f"claim: N={claim.n_atoms} T={claim.temperature} motif={claim.motif}"
            if claim else "claim: (none)"
        )
        agg = (
            t.final_verdict.aggregate_decision.value
            if t.final_verdict is not None else "no_verdict"
        )
        flagged = (
            ", ".join(
                f"{sv.source_name}:{sv.decision.value}"
                for sv in (t.final_verdict.source_verdicts if t.final_verdict else [])
                if sv.decision.value in ("fail", "caveat")
            )
            or "-"
        )
        first_msg = ""
        first_constraint = ""
        if t.final_verdict is not None:
            # Prefer FAIL over CAVEAT for the headline message, and for
            # rule_library specifically drill into the individual check
            # that triggered the fail (the top-level message just says
            # "one or more hard constraints failed").
            for sv in t.final_verdict.source_verdicts:
                if sv.decision.value != "fail":
                    continue
                first_msg = sv.message
                if sv.source_name == "rule_library":
                    checks = (sv.evidence or {}).get("checks") or []
                    failing = [
                        c for c in checks
                        if not c.get("passed", True)
                    ]
                    if failing:
                        names = ", ".join(
                            f"{c.get('constraint_name', '?')}:{c.get('fm_name', '?')}"
                            for c in failing
                        )
                        first_constraint = names
                        first_msg = (
                            f"{first_msg} | "
                            f"{failing[0].get('constraint_name', '?')}: "
                            f"{failing[0].get('message', '')}"
                        )
                break
            if not first_msg:
                for sv in t.final_verdict.source_verdicts:
                    if sv.decision.value == "caveat":
                        first_msg = sv.message
                        break

        typer.echo(
            f"specimen={t.specimen_id:<5}  termination={t.termination.value:<18}  agg={agg}"
        )
        typer.echo(f"  {gt_str}")
        typer.echo(f"  {claim_str}")
        typer.echo(f"  flagged: {flagged}")
        if first_constraint:
            typer.echo(f"  failing checks: {first_constraint}")
        if first_msg:
            typer.echo(f"  msg: {first_msg[:240]}")
        typer.echo("")


if __name__ == "__main__":
    app()
