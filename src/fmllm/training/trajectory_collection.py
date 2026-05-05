"""Collect Pipeline A trajectories at scale and serialize them as JSONL.

This module wraps :class:`fmllm.orchestrator.OHVDLoop` in a small
loop over a list of specimen IDs and writes one JSON line per
trajectory. Phase 6's training stage reads the JSONL and converts
it into datasets for SFT, DPO, and GRPO.

The collection step is the heavy lifter at runtime: each specimen
runs a full loop that calls every selected FM, evaluates verifier
sources, and stores the result. The user runs this on the remote.

Produces:
    A JSONL file under ``runs/trajectories/<run_id>/`` with one
    :class:`Trajectory` per line. A summary YAML lists pass/fail
    counts and the run config.

Depends on:
    fmllm.orchestrator, fmllm.verifier, fmllm.bridges.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from fmllm.orchestrator import (
    BaseLLM,
    FMRunnerFn,
    OHVDLoop,
    Trajectory,
)
from fmllm.utils.manifests import write_manifest
from fmllm.utils.run_ids import generate_run_id
from fmllm.verifier import (
    SourceDecision,
    SourcesConfig,
    Verifier,
)


def collect_trajectories(
    *,
    llm: BaseLLM,
    verifier: Verifier,
    runners: dict[str, FMRunnerFn],
    specimen_ids: Iterable[int],
    out_dir: Path | str,
    query: str = (
        "Identify the specimen's atom count, motif, and temperature. "
        "Use FM tools to gather evidence, propose a claim, and commit when confident."
    ),
    max_steps: int = 16,
    sources_config: SourcesConfig | None = None,
    filter_passing: bool = False,
    progress_every: int = 50,
    resume: bool = False,
) -> dict[str, Any]:
    """Run the OHVD loop on each specimen and write the trajectories to JSONL.

    Args:
        llm: An instantiated chat LLM (real or mock).
        verifier: The Phase 4 verifier, already configured with the
            literature DB plus optional cross-FM tolerance matrix.
        runners: ``{"fm1": ..., "fm2": ..., "fm3": ...}`` runner
            callables.
        specimen_ids: Iterable of specimen IDs to run.
        out_dir: Directory the script writes ``trajectories.jsonl``,
            ``summary.yaml``, and ``manifest.yaml`` into.
        query: The user message the loop puts in front of the LLM.
        max_steps: Per-specimen step budget for the loop.
        sources_config: Optional :class:`SourcesConfig` override.
        filter_passing: When True, only verifier-PASS trajectories
            land in the JSONL. The summary still records the totals.
        progress_every: Log a progress line after every N specimens.
        resume: When True and ``out_dir / 'trajectories.jsonl'``
            already exists, scan it for completed specimen_ids,
            skip those, and append the new trajectories. When False
            (default), the file is overwritten -- this matches the
            original behavior.

    Returns:
        A summary dict with ``run_id``, ``out_dir``, counts by
        termination type, and the JSONL path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = generate_run_id("traj-collect")
    jsonl_path = out_dir / "trajectories.jsonl"

    counters = {
        "total": 0,
        "committed_pass": 0,
        "committed_caveat": 0,
        "committed_fail": 0,
        "committed_skip": 0,
        "budget_exhausted": 0,
        "parse_failure": 0,
        "llm_error": 0,
        "filtered_kept": 0,
    }

    loop = OHVDLoop(
        llm=llm,
        verifier=verifier,
        runners=runners,
        max_steps=max_steps,
        sources_config=sources_config,
    )

    # Resume support: scan the existing trajectories.jsonl, drop the
    # specimen_ids we already have, and reopen in append mode.
    done_sids: set[int] = set()
    if resume and jsonl_path.exists() and jsonl_path.stat().st_size > 0:
        import json as _json
        with jsonl_path.open("r") as _existing:
            for line in _existing:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = _json.loads(line)
                except _json.JSONDecodeError:
                    continue
                sid_existing = obj.get("specimen_id")
                if isinstance(sid_existing, int):
                    done_sids.add(sid_existing)
        logger.info(
            "Resume mode: {} specimens already in trajectories.jsonl; skipping.",
            len(done_sids),
        )

    logger.info(
        "Trajectory collection: out={}, max_steps={}, filter_passing={}, resume={}",
        out_dir, max_steps, filter_passing, resume,
    )

    write_mode = "a" if (resume and done_sids) else "w"
    with jsonl_path.open(write_mode) as f:
        for sid in specimen_ids:
            sid_int = int(sid)
            if sid_int in done_sids:
                continue
            traj = loop.run(query=query, specimen_id=sid_int)
            counters["total"] += 1

            term = traj.termination.value
            if term == "committed":
                if traj.final_verdict is not None:
                    decision = traj.final_verdict.aggregate_decision
                    if decision is SourceDecision.PASS:
                        counters["committed_pass"] += 1
                    elif decision is SourceDecision.CAVEAT:
                        counters["committed_caveat"] += 1
                    elif decision is SourceDecision.FAIL:
                        counters["committed_fail"] += 1
                    else:
                        counters["committed_skip"] += 1
            elif term == "budget_exhausted":
                counters["budget_exhausted"] += 1
            elif term == "parse_failure":
                counters["parse_failure"] += 1
            elif term == "llm_error":
                counters["llm_error"] += 1

            keep = (
                (not filter_passing)
                or (
                    traj.final_verdict is not None
                    and traj.final_verdict.aggregate_decision is SourceDecision.PASS
                )
            )
            if keep:
                f.write(traj.model_dump_json() + "\n")
                counters["filtered_kept"] += 1

            if counters["total"] % progress_every == 0:
                logger.info(
                    "Collected {} specimens (pass={}, caveat={}, fail={}, kept={})",
                    counters["total"],
                    counters["committed_pass"],
                    counters["committed_caveat"],
                    counters["committed_fail"],
                    counters["filtered_kept"],
                )

    summary = {
        "run_id": run_id,
        "out_dir": str(out_dir),
        "jsonl_path": str(jsonl_path),
        "filter_passing": bool(filter_passing),
        "counters": counters,
    }
    summary_path = out_dir / "summary.yaml"
    with summary_path.open("w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    write_manifest(
        out_dir / "manifest.yaml",
        script="fmllm.training.trajectory_collection",
        inputs={"max_steps": max_steps, "filter_passing": filter_passing},
        config={"run_id": run_id, "query": query},
        extra=counters,
    )
    logger.info("Saved {} trajectories to {}", counters["filtered_kept"], jsonl_path)
    return summary


def load_trajectories_jsonl(path: Path | str) -> list[Trajectory]:
    """Load a JSONL trajectory file into memory."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"trajectory JSONL not found: {path}")
    out: list[Trajectory] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(Trajectory.model_validate_json(line))
    return out


def write_trajectories_jsonl(
    trajectories: Iterable[Trajectory], path: Path | str,
) -> Path:
    """Write trajectories to JSONL (one trajectory per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for t in trajectories:
            f.write(t.model_dump_json() + "\n")
    return path


__all__ = [
    "collect_trajectories",
    "load_trajectories_jsonl",
    "write_trajectories_jsonl",
]
