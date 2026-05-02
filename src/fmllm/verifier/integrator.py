"""Multi-source verifier integrator.

The integrator holds the five sources (rule_library, literature,
cross_fm, simulator, conformal), runs each enabled source against a
list of bridged FM outputs plus the LLM's claim, and aggregates the
per-source verdicts into a single :class:`VerifierVerdict`.

Aggregate decision rule:

    - any ``FAIL`` from any source -> ``FAIL``;
    - else any ``CAVEAT`` -> ``CAVEAT``;
    - else if at least one source ``PASS`` -> ``PASS``;
    - else ``SKIP``.

The aggregate also produces a structured :class:`Hint` listing the
sources that flagged issues plus suggested revisions tied to the
flagged sources.

E4 ablation: the ``verify`` method accepts a ``sources_config`` that
overrides the integrator's default. Disabled sources contribute a
``SKIP`` verdict so the trace stays consistent shape-wise across
ablation conditions.

Depends on:
    fmllm.verifier.{schema, sources}, fmllm.fms._calibration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fmllm.fms._calibration import (
    CrossFMToleranceMatrix,
    load_tolerance_matrix,
)
from fmllm.fms._schemas import BridgedFMOutput
from fmllm.fms._schemas.probe_schema import now_utc_iso
from fmllm.verifier.schema import (
    Hint,
    PhysicalStateClaim,
    SourceDecision,
    SourceVerdict,
    SourcesConfig,
    VerifierVerdict,
)
from fmllm.verifier.sources import (
    ConformalSource,
    CrossFMSource,
    LiteratureSource,
    RuleLibrarySource,
    SimulatorSource,
)


_SOURCE_ORDER = ("rule_library", "literature", "cross_fm", "simulator", "conformal")


_REVISION_SUGGESTIONS = {
    "rule_library": "Re-check the FM outputs against declared physics constraints; the LLM may need to retract or refine the typed claim.",
    "literature": "The FM-derived energy or temperature disagrees with a curated reference cluster; consider whether the structural motif assumption is correct.",
    "cross_fm": "Two FMs disagree on a shared causal variable; consult the FM with the strongest probe satisfaction or call additional FMs.",
    "simulator": "A short MD rollout from the claim's state did not reproduce the FM3-derived temperature; revisit the temperature estimate.",
    "conformal": "At least one FM flagged itself out-of-distribution or the claim sits outside the calibrated band; reduce confidence in that FM's contribution.",
}


class Verifier:
    """Multi-source verifier with runtime ablation.

    Args:
        default_config: Default :class:`SourcesConfig`. The
            :meth:`verify` method accepts a per-call override.
        cross_fm_tolerance: Optional calibrated tolerance matrix for
            cross-FM agreement. When ``None``, the cross-FM source
            falls back to its hard-coded defaults.
        literature_db_path: Optional path to ``clusters.json``. When
            ``None`` or missing, the literature source skips.
        simulator_kwargs: Optional kwargs forwarded to
            :class:`SimulatorSource`.
    """

    def __init__(
        self,
        *,
        default_config: SourcesConfig | None = None,
        cross_fm_tolerance: CrossFMToleranceMatrix | None = None,
        literature_db_path: Path | str | None = None,
        simulator_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.default_config = default_config or SourcesConfig()
        self._sources = {
            "rule_library": RuleLibrarySource(),
            "literature": (
                LiteratureSource(literature_db_path)
                if literature_db_path is not None
                else None
            ),
            "cross_fm": CrossFMSource(tolerance_matrix=cross_fm_tolerance),
            "simulator": SimulatorSource(**(simulator_kwargs or {})),
            "conformal": ConformalSource(),
        }

    def available_sources(self) -> list[str]:
        return [name for name, src in self._sources.items() if src is not None]

    def verify(
        self,
        bridged_outputs: list[BridgedFMOutput],
        claim: PhysicalStateClaim,
        *,
        sources_config: SourcesConfig | None = None,
    ) -> VerifierVerdict:
        cfg = sources_config or self.default_config
        per_source: list[SourceVerdict] = []
        for name in _SOURCE_ORDER:
            if not getattr(cfg, name):
                per_source.append(
                    SourceVerdict(
                        source_name=name,
                        decision=SourceDecision.SKIP,
                        confidence=0.0,
                        message="source disabled in sources_config",
                        evidence={},
                    )
                )
                continue
            source = self._sources.get(name)
            if source is None:
                per_source.append(
                    SourceVerdict(
                        source_name=name,
                        decision=SourceDecision.SKIP,
                        confidence=0.0,
                        message="source not available (missing dependency)",
                        evidence={},
                    )
                )
                continue
            verdict = source.check(bridged_outputs, claim)
            per_source.append(verdict)

        aggregate = self._aggregate(per_source)
        hint = self._build_hint(per_source, aggregate)
        return VerifierVerdict(
            aggregate_decision=aggregate,
            source_verdicts=per_source,
            hint=hint,
            timestamp=now_utc_iso(),
            sources_config=cfg,
        )

    # ----- helpers ---------------------------------------------------------

    def _aggregate(self, verdicts: list[SourceVerdict]) -> SourceDecision:
        active = [v for v in verdicts if v.decision is not SourceDecision.SKIP]
        if not active:
            return SourceDecision.SKIP
        if any(v.decision is SourceDecision.FAIL for v in active):
            return SourceDecision.FAIL
        if any(v.decision is SourceDecision.CAVEAT for v in active):
            return SourceDecision.CAVEAT
        return SourceDecision.PASS

    def _build_hint(
        self,
        verdicts: list[SourceVerdict],
        aggregate: SourceDecision,
    ) -> Hint:
        if aggregate is SourceDecision.PASS:
            return Hint(direction="all active sources agree; commit when ready")
        flagged = [
            v.source_name for v in verdicts
            if v.decision in (SourceDecision.FAIL, SourceDecision.CAVEAT)
        ]
        suggestions = [
            _REVISION_SUGGESTIONS.get(s, "Revisit the FM outputs and the claim.")
            for s in flagged
        ]
        if aggregate is SourceDecision.FAIL:
            direction = "at least one source signaled a hard failure; revise the claim"
        elif aggregate is SourceDecision.CAVEAT:
            direction = "at least one source flagged a soft inconsistency; refine or annotate"
        else:
            direction = "no active sources passed; cannot commit"
        return Hint(
            flagged_sources=flagged,
            suggested_revisions=suggestions,
            direction=direction,
        )


def build_default_verifier(
    *,
    sources_config: SourcesConfig | None = None,
    tolerance_matrix_path: Path | str | None = None,
    literature_db_path: Path | str | None = None,
    simulator_kwargs: dict[str, Any] | None = None,
) -> Verifier:
    """Convenience constructor that loads the cross-FM tolerance matrix
    from disk if a path is given."""
    tolerance_matrix: CrossFMToleranceMatrix | None = None
    if tolerance_matrix_path is not None:
        path = Path(tolerance_matrix_path)
        if path.exists():
            tolerance_matrix = load_tolerance_matrix(path)
    return Verifier(
        default_config=sources_config,
        cross_fm_tolerance=tolerance_matrix,
        literature_db_path=literature_db_path,
        simulator_kwargs=simulator_kwargs,
    )


__all__ = ["Verifier", "build_default_verifier"]
