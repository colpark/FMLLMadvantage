"""Cross-FM verifier source: pairwise consistency on shared variables.

The source reads the ``dependencies`` block of every bridged FM output,
groups by ``target_variable``, and compares the values different FMs
derived for the same shared variable. When a calibrated tolerance
matrix from :mod:`fmllm.fms._calibration` is available, the source
uses its threshold; otherwise it applies a hard-coded fallback.

For our year-1 testbed the practical shared variables are:

    - ``atom_count``: derived by FM1 from its count head, also possibly
      claimed by the LLM. The source compares FM1's value against the
      claim and against any other FM's derived value.
    - ``temperature``: derived by FM3 as ``alpha * beta``. The source
      compares against the claim's temperature.

Sources that find no shared-variable pair return ``SKIP``. Sources
that find disagreement above the tolerance return ``CAVEAT``;
disagreement above twice the tolerance returns ``FAIL``.

Depends on:
    fmllm.fms._calibration, fmllm.verifier.schema.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from fmllm.fms._calibration import CrossFMToleranceMatrix
from fmllm.fms._schemas import BridgedFMOutput
from fmllm.verifier.schema import (
    PhysicalStateClaim,
    SourceDecision,
    SourceVerdict,
)


# Fallback tolerances applied when the calibrated matrix lacks a
# pair, keyed by the shared variable name. Units match the variable.
_DEFAULT_TOLERANCE: dict[str, float] = {
    "atom_count": 1.0,        # off by one is a caveat
    "temperature": 0.20,       # 0.20 LJ tolerance on T
    "per_atom_potential_energy": 0.10,
}


class CrossFMSource:
    """Pairwise agreement source over shared causal variables."""

    name = "cross_fm"

    def __init__(
        self,
        tolerance_matrix: CrossFMToleranceMatrix | None = None,
        *,
        fail_factor: float = 2.0,
    ) -> None:
        self.tolerance_matrix = tolerance_matrix
        self.fail_factor = fail_factor

    # ----- helpers ---------------------------------------------------------

    def _calibrated_tolerance(
        self, variable: str, fm_a: str, fm_b: str, alpha: float = 0.10,
    ) -> float | None:
        if self.tolerance_matrix is None:
            return None
        wanted = (sorted([fm_a, fm_b]))
        for p in self.tolerance_matrix.pairwise:
            if p.variable != variable:
                continue
            if sorted([p.fm_a, p.fm_b]) == wanted:
                return p.thresholds.get(f"alpha_{alpha:.4f}")
        return None

    def _collect_per_variable(
        self,
        bridged_outputs: list[BridgedFMOutput],
        claim: PhysicalStateClaim,
    ) -> dict[str, dict[str, float]]:
        """Return ``{variable: {source_name: value}}`` over real numbers."""
        per_var: dict[str, dict[str, float]] = defaultdict(dict)
        for bo in bridged_outputs:
            fm = bo.source.fm_name
            for d in bo.dependencies:
                if d.derived_value is None:
                    continue
                try:
                    per_var[d.target_variable][fm] = float(d.derived_value)
                except (TypeError, ValueError):
                    continue
        # The claim contributes too.
        if claim.n_atoms is not None:
            per_var["atom_count"]["claim"] = float(claim.n_atoms)
        if claim.temperature is not None:
            per_var["temperature"]["claim"] = float(claim.temperature)
        if claim.per_atom_potential_energy is not None:
            per_var["per_atom_potential_energy"]["claim"] = float(
                claim.per_atom_potential_energy,
            )
        return per_var

    # ----- main check ------------------------------------------------------

    def check(
        self,
        bridged_outputs: list[BridgedFMOutput],
        claim: PhysicalStateClaim,
    ) -> SourceVerdict:
        per_var = self._collect_per_variable(bridged_outputs, claim)

        comparisons: list[dict[str, Any]] = []
        any_caveat = False
        any_fail = False
        for variable, sources in per_var.items():
            names = sorted(sources.keys())
            if len(names) < 2:
                continue
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a, b = names[i], names[j]
                    diff = abs(sources[a] - sources[b])
                    tol_calibrated = self._calibrated_tolerance(variable, a, b)
                    tol = (
                        tol_calibrated
                        if tol_calibrated is not None
                        else _DEFAULT_TOLERANCE.get(variable, 1.0)
                    )
                    if diff <= tol:
                        outcome = "agree"
                    elif diff <= self.fail_factor * tol:
                        outcome = "caveat"
                        any_caveat = True
                    else:
                        outcome = "fail"
                        any_fail = True
                    comparisons.append({
                        "variable": variable,
                        "fm_a": a, "fm_b": b,
                        "value_a": sources[a], "value_b": sources[b],
                        "abs_diff": diff,
                        "tolerance": tol,
                        "tolerance_source": "calibrated" if tol_calibrated is not None else "default",
                        "outcome": outcome,
                    })

        if not comparisons:
            return SourceVerdict(
                source_name=self.name,
                decision=SourceDecision.SKIP,
                confidence=0.0,
                message="no shared variables across FMs and claim",
                evidence={"n_outputs": len(bridged_outputs)},
            )

        if any_fail:
            decision = SourceDecision.FAIL
            msg = "at least one cross-FM pair exceeds the fail tolerance"
        elif any_caveat:
            decision = SourceDecision.CAVEAT
            msg = "at least one cross-FM pair sits above the agreement tolerance"
        else:
            decision = SourceDecision.PASS
            msg = "all cross-FM pairs within tolerance"

        # Confidence: fraction of comparisons that passed.
        n_pass = sum(1 for c in comparisons if c["outcome"] == "agree")
        confidence = n_pass / max(1, len(comparisons))
        return SourceVerdict(
            source_name=self.name,
            decision=decision,
            confidence=float(confidence),
            message=msg,
            evidence={"comparisons": comparisons},
        )


__all__ = ["CrossFMSource"]
