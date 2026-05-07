"""Categorize parse failures in a Stage 9 records.jsonl.

Auto-discovers the latest Stage 9 inference run under
``runs/materials/holdout/cot_sft_sae/`` (override with --records).
For every record where ``claim is None`` (parse failure), classify
the cause and report aggregate counts plus a few representative
last-200-char samples per category.

Categories:

    A. NO_FINAL_COMMIT_TAG
       The string "Final commit:" never appears in raw_text. The
       LLM didn't reach the commit step -- usually because
       generation hit the max_new_tokens cap before the CoT
       finished, or the LoRA degraded the format.

    B. NO_OPEN_BRACE_AFTER_TAG
       "Final commit:" present but no '{' afterwards. Rare; the
       LLM forgot to start the JSON object.

    C. UNCLOSED_JSON
       '{' present but never balanced. Almost always truncation:
       max_new_tokens cap hit mid-JSON.

    D. INVALID_JSON_SYNTAX
       Balanced braces but ``json.loads`` rejects the payload.
       Causes: Python-style True/False, trailing commas, unquoted
       keys, embedded comments. Indicates LoRA didn't fully
       internalize JSON formatting rules.

    E. NOT_A_DICT
       JSON parses but yields a list/string/etc. instead of an
       object. Very rare.

A high A+C count points to truncation; a high D count points to
formatting drift in the SFT.

Output:

    runs/materials/diagnostics/<run_id>/parse_failures.yaml

Usage:

    bash scripts/materials/inspect_parse_failures.sh
    bash scripts/materials/inspect_parse_failures.sh --records <jsonl>

Depends on:
    typer, pyyaml.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer
import yaml


app = typer.Typer(add_completion=False, no_args_is_help=False)


_FINAL_TAG = "Final commit:"
_FINAL_RE_FALLBACK = re.compile(r"Final commit:\s*(\{.*?\})", re.DOTALL)


def _latest_records(parent: Path) -> Path | None:
    if not parent.exists():
        return None
    cands = sorted(
        parent.glob("*/records.jsonl"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    return cands[0] if cands else None


def _classify(raw_text: str) -> tuple[str, dict[str, Any]]:
    """Return (category, details) for a parse-failed record's raw_text."""
    if _FINAL_TAG not in raw_text:
        return "NO_FINAL_COMMIT_TAG", {
            "tail": raw_text[-200:] if len(raw_text) >= 200 else raw_text,
            "raw_length_chars": len(raw_text),
        }
    sub = raw_text[raw_text.find(_FINAL_TAG):]
    brace_start = sub.find("{")
    if brace_start < 0:
        return "NO_OPEN_BRACE_AFTER_TAG", {"sub_tail": sub[-200:]}

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
        return "UNCLOSED_JSON", {
            "sub_length": len(sub),
            "max_depth_reached": depth,
            "tail": sub[-200:],
        }

    payload = sub[brace_start:end]
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as e:
        # Try the regex fallback the parser uses.
        m = _FINAL_RE_FALLBACK.search(raw_text)
        if m is not None:
            try:
                data = json.loads(m.group(1))
            except json.JSONDecodeError:
                return "INVALID_JSON_SYNTAX", {
                    "payload": payload[:300],
                    "json_error": str(e),
                }
        else:
            return "INVALID_JSON_SYNTAX", {
                "payload": payload[:300],
                "json_error": str(e),
            }
    if not isinstance(data, dict):
        return "NOT_A_DICT", {"parsed_type": type(data).__name__}

    # The official parser would have succeeded -- flag this row as
    # "PARSER_DISAGREES" so we know to re-check the parser logic.
    return "PARSER_DISAGREES", {"payload": payload[:300]}


@app.command()
def main(
    records: Path | None = typer.Option(
        None, "--records",
        help="records.jsonl to inspect. Default: latest under "
             "runs/materials/holdout/cot_sft_sae/.",
    ),
    n_examples: int = typer.Option(
        3, "--n-examples",
        help="How many representative examples to show per category.",
    ),
    out: Path = typer.Option(
        Path("runs/materials/diagnostics"), "--out", "-o",
    ),
) -> None:
    """Classify why Stage 9 parse-failure records failed."""
    if records is None:
        records = _latest_records(Path("runs/materials/holdout/cot_sft_sae"))
    if records is None or not records.exists():
        raise typer.BadParameter(
            "no records.jsonl found. Run scripts/materials/09_run_singleshot.sh "
            "first or pass --records explicitly."
        )

    run_id = f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-parse-failures"
    out_dir = out / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    typer.echo("==> Parse-failure analysis")
    typer.echo(f"    records : {records}")
    typer.echo("")

    n_total = 0
    n_committed = 0
    n_failed = 0
    cat_counts: Counter = Counter()
    examples: dict[str, list[dict]] = {}
    raw_lengths_failed: list[int] = []
    raw_lengths_ok: list[int] = []

    with records.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_total += 1
            raw_text = rec.get("raw_text") or ""
            if rec.get("claim") is not None:
                n_committed += 1
                raw_lengths_ok.append(len(raw_text))
                continue
            n_failed += 1
            raw_lengths_failed.append(len(raw_text))
            category, details = _classify(raw_text)
            cat_counts[category] += 1
            if len(examples.get(category, [])) < n_examples:
                examples.setdefault(category, []).append({
                    "specimen_id": rec.get("specimen_id"),
                    "details": details,
                })

    if n_total == 0:
        typer.echo("ERROR: no records found.", err=True)
        raise typer.Exit(1)

    typer.echo(f"==> {n_total} records, {n_committed} parsed, "
               f"{n_failed} failed ({100.0 * n_failed / n_total:.1f}%)")
    typer.echo("")
    typer.echo("Category counts:")
    for cat, n in cat_counts.most_common():
        typer.echo(f"  {cat:<25s} {n:>4d} ({100.0 * n / max(n_failed, 1):.1f}%)")
    typer.echo("")

    if raw_lengths_failed:
        avg_failed = sum(raw_lengths_failed) / len(raw_lengths_failed)
        max_failed = max(raw_lengths_failed)
        typer.echo("Raw-text length (chars):")
        if raw_lengths_ok:
            avg_ok = sum(raw_lengths_ok) / len(raw_lengths_ok)
            max_ok = max(raw_lengths_ok)
            typer.echo(f"  parsed   avg={avg_ok:>8.0f}  max={max_ok:>6d}")
        typer.echo(f"  failed   avg={avg_failed:>8.0f}  max={max_failed:>6d}")
        typer.echo("")

    if examples:
        typer.echo("Examples per category:")
        for cat, exs in examples.items():
            typer.echo(f"  [{cat}]")
            for ex in exs:
                sid = ex["specimen_id"]
                d = ex["details"]
                tail_key = next(
                    (k for k in ("tail", "sub_tail", "payload") if k in d),
                    None,
                )
                tail = d.get(tail_key, "") if tail_key else ""
                tail_repr = (
                    repr(tail)[:300]
                    if isinstance(tail, str) else str(tail)[:300]
                )
                typer.echo(f"    sid={sid}: {tail_repr}")
                if "json_error" in d:
                    typer.echo(f"      json_error: {d['json_error']}")
            typer.echo("")

    # Cause-based recommendation
    typer.echo("DIAGNOSIS:")
    truncation_count = (
        cat_counts["NO_FINAL_COMMIT_TAG"] + cat_counts["UNCLOSED_JSON"]
    )
    json_drift_count = cat_counts["INVALID_JSON_SYNTAX"]
    if n_failed == 0:
        typer.echo("  No parse failures. Nothing to do.")
    elif truncation_count >= 0.7 * n_failed:
        typer.echo(
            f"  Dominant cause: TRUNCATION ({truncation_count}/{n_failed} = "
            f"{100.0 * truncation_count / n_failed:.0f}%). The CoT didn't "
            f"finish before max_new_tokens ran out. Fix:"
        )
        typer.echo(
            "    MAX_NEW_TOKENS=1024 bash scripts/materials/09_run_singleshot.sh"
        )
    elif json_drift_count >= 0.5 * n_failed:
        typer.echo(
            f"  Dominant cause: JSON SYNTAX DRIFT ({json_drift_count}/{n_failed}). "
            f"The LoRA didn't internalize strict JSON formatting. Fix: "
            f"either add a permissive json5 parser, or audit a few "
            f"INVALID_JSON_SYNTAX examples and repair the most common "
            f"format errors via post-processing."
        )
    else:
        typer.echo(
            f"  Mixed causes (truncation={truncation_count}, "
            f"json_drift={json_drift_count}). Address truncation first "
            f"by bumping MAX_NEW_TOKENS; remaining drift after that is "
            f"the real format-quality signal."
        )

    summary = {
        "records": str(records),
        "examined_utc": datetime.now(UTC).isoformat(),
        "n_total": n_total,
        "n_committed": n_committed,
        "n_failed": n_failed,
        "parse_failure_rate": float(n_failed / max(n_total, 1)),
        "category_counts": dict(cat_counts),
        "raw_length_avg_failed": (
            float(sum(raw_lengths_failed) / len(raw_lengths_failed))
            if raw_lengths_failed else None
        ),
        "raw_length_avg_parsed": (
            float(sum(raw_lengths_ok) / len(raw_lengths_ok))
            if raw_lengths_ok else None
        ),
        "examples": examples,
    }
    summary_path = out_dir / "parse_failures.yaml"
    with summary_path.open("w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    typer.echo(f"==> Report: {summary_path}")


if __name__ == "__main__":
    app()
