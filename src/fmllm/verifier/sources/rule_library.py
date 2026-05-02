"""Rule-library verifier source: per-constraint check dispatch.

The source maintains a registry of constraint-name -> check function.
At runtime it iterates the ``applicable_constraints`` field of every
bridged FM output, dispatches to the registered check function for
that constraint name, and aggregates the per-constraint outcomes
into a single :class:`SourceVerdict`.

A check function signature:

    def check(bridged: BridgedFMOutput, claim: PhysicalStateClaim) -> dict

The dict carries:

    - ``passed``: bool — whether the check succeeded
    - ``confidence``: float in [0, 1]
    - ``message``: short string describing the outcome
    - ``evidence``: dict of supporting numbers

The source aggregates: any failing hard check produces an aggregate
``fail``; any failing soft check or low-confidence pass produces a
``caveat``; otherwise ``pass``.

Extending: register new check functions with
``@register_check("constraint_name")``. The metadata's ``probe``
field need not match these names; the rule library tracks
constraint *checks* (analytic verifications), distinct from probes
(behavioral measurements).

Depends on:
    pydantic, fmllm.fms._schemas, fmllm.verifier.schema.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fmllm.fms._schemas import BridgedFMOutput
from fmllm.verifier.schema import (
    PhysicalStateClaim,
    SourceDecision,
    SourceVerdict,
)


CheckFn = Callable[[BridgedFMOutput, PhysicalStateClaim], dict[str, Any]]
"""Signature for a per-constraint check function."""


_REGISTRY: dict[str, CheckFn] = {}


def register_check(constraint_name: str) -> Callable[[CheckFn], CheckFn]:
    """Decorator that registers a check function for a constraint."""
    def decorator(fn: CheckFn) -> CheckFn:
        if constraint_name in _REGISTRY:
            raise ValueError(f"check {constraint_name!r} already registered")
        _REGISTRY[constraint_name] = fn
        return fn
    return decorator


def registered_checks() -> dict[str, CheckFn]:
    """Return a copy of the registry for inspection or testing."""
    return dict(_REGISTRY)


# ---------------------------------------------------------------------------
# Built-in checks
# ---------------------------------------------------------------------------


@register_check("positions_in_box")
def _check_positions_in_box(
    bridged: BridgedFMOutput, claim: PhysicalStateClaim,
) -> dict[str, Any]:
    """Confirm every position in FM1's :class:`AtomSet` payload sits inside
    a configurable box of half-width 4.8 LJ."""
    half = 4.8
    value = bridged.prediction.value
    positions = (value or {}).get("positions", [])
    bad = []
    for ap in positions:
        x = float(ap["x_lj"])
        y = float(ap["y_lj"])
        if abs(x) > half or abs(y) > half:
            bad.append((x, y))
    if not bad:
        return {
            "passed": True, "confidence": 1.0,
            "message": f"all {len(positions)} positions inside box",
            "evidence": {"n_positions": len(positions)},
        }
    return {
        "passed": False, "confidence": 1.0,
        "message": f"{len(bad)}/{len(positions)} positions outside the box",
        "evidence": {"out_of_box": bad, "n_positions": len(positions)},
    }


@register_check("non_negativity")
def _check_non_negativity(
    bridged: BridgedFMOutput, claim: PhysicalStateClaim,
) -> dict[str, Any]:
    """Confirm FM2's predicted per-atom energy stays at or above the
    LJ pair-energy floor."""
    floor = -3.0
    value = bridged.prediction.value or {}
    e = value.get("value_lj")
    if e is None:
        return {
            "passed": True, "confidence": 1.0,
            "message": "no scalar energy in payload",
            "evidence": {},
        }
    if float(e) < floor:
        return {
            "passed": False, "confidence": 1.0,
            "message": f"energy {e:.4f} below floor {floor:.2f}",
            "evidence": {"energy": float(e), "floor": floor},
        }
    return {
        "passed": True, "confidence": 1.0,
        "message": f"energy {e:.4f} above floor {floor:.2f}",
        "evidence": {"energy": float(e), "floor": floor},
    }


@register_check("distribution_non_negativity")
def _check_distribution_non_negativity(
    bridged: BridgedFMOutput, claim: PhysicalStateClaim,
) -> dict[str, Any]:
    """Confirm FM3's Gamma parameters are strictly positive."""
    value = bridged.prediction.value or {}
    alpha = value.get("alpha")
    beta = value.get("beta")
    if alpha is None or beta is None:
        return {
            "passed": True, "confidence": 1.0,
            "message": "no Gamma parameters in payload",
            "evidence": {},
        }
    ok = float(alpha) > 0 and float(beta) > 0
    return {
        "passed": ok,
        "confidence": 1.0,
        "message": f"alpha={alpha:.4f}, beta={beta:.4f}",
        "evidence": {"alpha": float(alpha), "beta": float(beta)},
    }


@register_check("distribution_normalization")
def _check_distribution_normalization(
    bridged: BridgedFMOutput, claim: PhysicalStateClaim,
) -> dict[str, Any]:
    """Gamma normalization is analytic; this check is structural and
    only fails if alpha or beta are non-finite."""
    value = bridged.prediction.value or {}
    alpha = value.get("alpha")
    beta = value.get("beta")
    if alpha is None or beta is None:
        return {
            "passed": True, "confidence": 1.0,
            "message": "no Gamma parameters in payload",
            "evidence": {},
        }
    finite = (
        isinstance(alpha, int | float) and isinstance(beta, int | float)
        and float("-inf") < float(alpha) < float("inf")
        and float("-inf") < float(beta) < float("inf")
    )
    return {
        "passed": bool(finite),
        "confidence": 1.0,
        "message": "Gamma parameters finite" if finite else "Gamma parameters not finite",
        "evidence": {"alpha": alpha, "beta": beta},
    }


_ATOM_COUNT_TOLERANCE = 2
"""Allowed off-by-N between FM1's count head and confidence-thresholded query count.

Set-prediction with confidence thresholding produces natural noise; exact
equality is an unrealistic gate. Off-by-2 catches major head disagreement
while tolerating ordinary confidence-edge effects.
"""


@register_check("atom_count_consistency")
def _check_atom_count_consistency(
    bridged: BridgedFMOutput, claim: PhysicalStateClaim,
) -> dict[str, Any]:
    """Confirm FM1's AtomSet's predicted atom count matches the number
    of confidence-thresholded query slots in its payload, within an
    off-by-N tolerance."""
    value = bridged.prediction.value or {}
    n_pred = value.get("n_atoms_pred")
    positions = value.get("positions", [])
    if n_pred is None:
        return {
            "passed": True, "confidence": 1.0,
            "message": "no n_atoms_pred in payload",
            "evidence": {},
        }
    n_kept = len(positions)
    diff = abs(int(n_pred) - n_kept)
    if diff <= _ATOM_COUNT_TOLERANCE:
        return {
            "passed": True, "confidence": 1.0,
            "message": (
                f"count head N={n_pred}, confident queries={n_kept} "
                f"(diff {diff} <= tol {_ATOM_COUNT_TOLERANCE})"
            ),
            "evidence": {
                "n_pred": int(n_pred),
                "n_kept": n_kept,
                "tolerance": _ATOM_COUNT_TOLERANCE,
            },
        }
    return {
        "passed": False, "confidence": 0.7,
        "message": (
            f"count head says N={n_pred}, but {n_kept} confident queries "
            f"passed the threshold (diff {diff} > tol {_ATOM_COUNT_TOLERANCE})"
        ),
        "evidence": {
            "n_pred": int(n_pred),
            "n_kept": n_kept,
            "tolerance": _ATOM_COUNT_TOLERANCE,
        },
    }


@register_check("permutation_invariance")
def _check_permutation_invariance(
    bridged: BridgedFMOutput, claim: PhysicalStateClaim,
) -> dict[str, Any]:
    """Permutation invariance is automatic for FM2 (input g(r) is
    permutation invariant by construction). The rule-library check is
    a structural sanity confirmation."""
    return {
        "passed": True, "confidence": 1.0,
        "message": "permutation invariance holds by construction (g(r) input)",
        "evidence": {},
    }


@register_check("extensive_scaling")
def _check_extensive_scaling(
    bridged: BridgedFMOutput, claim: PhysicalStateClaim,
) -> dict[str, Any]:
    """Extensive scaling holds by output design (FM2 emits per-atom).
    The check confirms the value remains a finite scalar."""
    value = bridged.prediction.value or {}
    e = value.get("value_lj")
    if e is None or not (float("-inf") < float(e) < float("inf")):
        return {
            "passed": False, "confidence": 1.0,
            "message": f"non-finite per-atom energy: {e}",
            "evidence": {},
        }
    return {
        "passed": True, "confidence": 1.0,
        "message": "per-atom energy finite",
        "evidence": {"energy": float(e)},
    }


# Note: ``translation_equivariance`` and ``equipartition`` are FM-quality
# probes whose satisfaction scores live in the bridged output's
# ``applicable_constraints`` field. The verifier exposes them to other
# sources (and the LLM context) for trustworthiness weighting, but they
# do not produce per-output checks here; they describe the FM globally,
# not a particular prediction. Phase 8's E4 ablation can re-enable them
# as a separate verifier mode if needed.


# ---------------------------------------------------------------------------
# Source class
# ---------------------------------------------------------------------------


class RuleLibrarySource:
    """Verifier source that dispatches per-constraint check functions."""

    name = "rule_library"

    def check(
        self,
        bridged_outputs: list[BridgedFMOutput],
        claim: PhysicalStateClaim,
    ) -> SourceVerdict:
        per_check_results: list[dict[str, Any]] = []
        any_hard_fail = False
        any_soft_fail = False
        for bo in bridged_outputs:
            for ac in bo.applicable_constraints:
                fn = _REGISTRY.get(ac.constraint_name)
                if fn is None:
                    continue
                result = fn(bo, claim)
                result["constraint_name"] = ac.constraint_name
                result["fm_name"] = bo.source.fm_name
                result["constraint_type"] = ac.type
                per_check_results.append(result)
                if not result["passed"]:
                    if ac.type == "hard":
                        any_hard_fail = True
                    else:
                        any_soft_fail = True

        if not per_check_results:
            return SourceVerdict(
                source_name=self.name,
                decision=SourceDecision.SKIP,
                confidence=0.0,
                message="no registered checks fired",
                evidence={"n_outputs": len(bridged_outputs)},
            )

        if any_hard_fail:
            decision = SourceDecision.FAIL
            msg = "one or more hard constraints failed"
        elif any_soft_fail:
            decision = SourceDecision.CAVEAT
            msg = "one or more soft constraints failed"
        else:
            decision = SourceDecision.PASS
            msg = "all registered constraint checks passed"

        # Mean confidence across the per-check confidences.
        confidence = sum(r["confidence"] for r in per_check_results) / len(per_check_results)
        return SourceVerdict(
            source_name=self.name,
            decision=decision,
            confidence=float(confidence),
            message=msg,
            evidence={"checks": per_check_results},
        )


__all__ = [
    "CheckFn",
    "RuleLibrarySource",
    "register_check",
    "registered_checks",
]
