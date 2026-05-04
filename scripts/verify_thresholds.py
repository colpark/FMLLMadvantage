"""CLI: verify locked thresholds match the test modules' current defaults.

Reads ``configs/evaluation_thresholds_locked.yaml``, imports each
test module, inspects ``measure(...)``'s signature, and asserts every
locked threshold matches the default argument value. Exits non-zero
on drift so a held-out evaluation refuses to run when a threshold
has silently changed.

Usage:
    uv run python scripts/verify_thresholds.py
    uv run python scripts/verify_thresholds.py --lock-file configs/evaluation_thresholds_locked.yaml

Depends on:
    typer, pyyaml.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import typer
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fmllm.evaluation import (  # noqa: E402
    accuracy,
    calibrated_uncertainty,
    federated_factorability,
    goal_competence,
    prediction_compression,
    prediction_distinction,
    step_recoverability,
    trajectory_compression,
    trajectory_distinction,
)


app = typer.Typer(add_completion=False, no_args_is_help=False)


_MODULE_BY_NAME = {
    "trajectory_compression": trajectory_compression,
    "trajectory_distinction": trajectory_distinction,
    "step_recoverability": step_recoverability,
    "prediction_compression": prediction_compression,
    "prediction_distinction": prediction_distinction,
    "goal_competence": goal_competence,
    "federated_factorability": federated_factorability,
    "calibrated_uncertainty": calibrated_uncertainty,
    "goal_accuracy": accuracy,
}


def _signature_defaults(module) -> dict[str, object]:
    sig = inspect.signature(module.measure)
    out: dict[str, object] = {}
    for name, param in sig.parameters.items():
        if param.default is inspect.Parameter.empty:
            continue
        out[name] = param.default
    return out


@app.command()
def main(
    lock_file: Path = typer.Option(
        Path("configs/evaluation_thresholds_locked.yaml"), "--lock-file",
    ),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    """Verify every locked threshold matches the current code default."""
    if not lock_file.exists():
        typer.echo(f"ERROR: lock file not found: {lock_file}")
        raise typer.Exit(code=1)

    with lock_file.open("r") as f:
        lock = yaml.safe_load(f)

    if not quiet:
        typer.echo(
            f"==> Verifying thresholds against {lock_file} "
            f"(locked at {lock.get('locked_at_commit', '?')[:12]})"
        )

    drift: list[str] = []
    expected = lock.get("thresholds") or {}
    for test_name, expected_kwargs in expected.items():
        module = _MODULE_BY_NAME.get(test_name)
        if module is None:
            drift.append(f"{test_name}: no matching module in fmllm.evaluation")
            continue
        actual = _signature_defaults(module)
        for kwarg, expected_value in (expected_kwargs or {}).items():
            actual_value = actual.get(kwarg, "<missing>")
            if actual_value != expected_value:
                drift.append(
                    f"{test_name}.{kwarg}: locked={expected_value!r} "
                    f"vs code={actual_value!r}"
                )
            elif not quiet:
                typer.echo(f"  OK  {test_name}.{kwarg} = {expected_value}")

    if drift:
        typer.echo("")
        typer.echo("THRESHOLD DRIFT DETECTED:")
        for d in drift:
            typer.echo(f"  - {d}")
        typer.echo("")
        typer.echo(
            "Either revert the code defaults to match the lock, "
            "or generate a new lock with a new version number."
        )
        raise typer.Exit(code=1)

    if not quiet:
        typer.echo("==> All thresholds match the lock.")


if __name__ == "__main__":
    app()
