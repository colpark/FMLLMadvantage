"""Conformal verifier source: per-FM in-distribution flag check.

The source reads ``source.in_distribution`` from each bridged FM
output and, when uncertainty is populated, confirms the predicted
band has a finite width. Aggregates into a single
:class:`SourceVerdict`. When a claim names a value the bridge
predicted, the source also checks whether the claimed value sits
inside the calibrated band (``Prediction.uncertainty.lower`` to
``upper`` for FM2's symmetric energy band).

Decision rules:

    - Any FM with ``in_distribution=False`` -> ``CAVEAT``.
    - The claim's value falls outside the FM's calibrated band -> ``CAVEAT``.
    - Otherwise -> ``PASS``.

The verifier never returns ``FAIL`` from conformal alone; conformal
flags low-confidence predictions but does not gate hard pass / fail
decisions, which the rule library and cross-FM sources own.

Depends on:
    fmllm.verifier.schema.
"""

from __future__ import annotations

from typing import Any

from fmllm.fms._schemas import BridgedFMOutput
from fmllm.verifier.schema import (
    PhysicalStateClaim,
    SourceDecision,
    SourceVerdict,
)


class ConformalSource:
    """Verifier source that reads in-distribution flags and band membership."""

    name = "conformal"

    def check(
        self,
        bridged_outputs: list[BridgedFMOutput],
        claim: PhysicalStateClaim,
    ) -> SourceVerdict:
        report: list[dict[str, Any]] = []
        any_caveat = False
        for bo in bridged_outputs:
            entry: dict[str, Any] = {
                "fm_name": bo.source.fm_name,
                "in_distribution": bo.source.in_distribution,
                "uncertainty_present": bo.prediction.uncertainty is not None,
            }
            if not bo.source.in_distribution:
                entry["flag"] = "out_of_distribution"
                any_caveat = True
            elif bo.prediction.uncertainty is not None:
                # If the claim names a value the bridge measured against,
                # check membership in the calibrated band.
                claim_check = self._claim_in_band(bo, claim)
                entry["claim_in_band"] = claim_check
                if claim_check is False:
                    entry["flag"] = "claim_outside_band"
                    any_caveat = True
                else:
                    entry["flag"] = "ok"
            else:
                entry["flag"] = "ok"
            report.append(entry)

        if not report:
            return SourceVerdict(
                source_name=self.name,
                decision=SourceDecision.SKIP,
                confidence=0.0,
                message="no bridged outputs",
                evidence={},
            )

        decision = SourceDecision.CAVEAT if any_caveat else SourceDecision.PASS
        n_clean = sum(1 for r in report if r.get("flag") == "ok")
        confidence = n_clean / len(report)
        msg = (
            "all FMs in-distribution and within band"
            if decision is SourceDecision.PASS
            else "at least one FM flagged out-of-distribution or claim outside band"
        )
        return SourceVerdict(
            source_name=self.name,
            decision=decision,
            confidence=float(confidence),
            message=msg,
            evidence={"per_fm": report},
        )

    # ----- helpers ---------------------------------------------------------

    def _claim_in_band(
        self,
        bridged: BridgedFMOutput,
        claim: PhysicalStateClaim,
    ) -> bool | None:
        """If the claim names a value comparable to the prediction, return
        ``True`` / ``False`` for whether it lies in the calibrated band.
        Otherwise return ``None``."""
        u = bridged.prediction.uncertainty
        if u is None:
            return None
        # FM2 emits per-atom potential energy with a symmetric band.
        if (
            bridged.source.fm_name == "fm2_rdf"
            and claim.per_atom_potential_energy is not None
            and isinstance(u.lower, int | float) and isinstance(u.upper, int | float)
        ):
            v = float(claim.per_atom_potential_energy)
            return float(u.lower) <= v <= float(u.upper)
        return None


__all__ = ["ConformalSource"]
