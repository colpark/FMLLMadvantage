"""The eight world-model evaluation tests.

Phase 7 implements the trajectory-level and prediction-level tests,
plus the federated-factorability and calibrated-uncertainty checks.

Each test module exposes a ``measure(...)`` function returning a
:class:`fmllm.evaluation.schema.TestResult`. The CLI runner under
``scripts/run_evaluation.py`` collects every test into an
:class:`EvaluationReport`.

Layers:

* **trajectory** (Layer 1): trajectory_compression, trajectory_distinction,
  step_recoverability.
* **prediction** (Layer 2): prediction_compression, prediction_distinction,
  goal_competence.
* **cross_layer**: federated_factorability, calibrated_uncertainty.
"""

from fmllm.evaluation import (
    calibrated_uncertainty,
    federated_factorability,
    goal_competence,
    prediction_compression,
    prediction_distinction,
    step_recoverability,
    trajectory_compression,
    trajectory_distinction,
)
from fmllm.evaluation.schema import (
    EvaluationReport,
    TestResult,
    make_skipped,
    threshold_check,
)

__all__ = [
    "EvaluationReport",
    "TestResult",
    "calibrated_uncertainty",
    "federated_factorability",
    "goal_competence",
    "make_skipped",
    "prediction_compression",
    "prediction_distinction",
    "step_recoverability",
    "threshold_check",
    "trajectory_compression",
    "trajectory_distinction",
]
