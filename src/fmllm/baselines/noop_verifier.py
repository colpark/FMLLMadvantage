"""No-op verifier for the B2 baseline.

The OHVD loop calls ``self.verifier.verify(bridged_outputs, claim,
sources_config=...)`` and gets a :class:`VerifierVerdict`. The
:class:`NoOpVerifier` exposes the same method signature but always
returns an aggregate ``PASS`` with one stub source verdict. The
LLM's hypothesize / commit turns therefore never see a CAVEAT or
FAIL signal, which is the whole point of B2: isolate whether the
multi-source verifier loop teaches the LLM anything.

Depends on:
    fmllm.verifier.schema, fmllm.fms._schemas.bridge_schema.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fmllm.fms._schemas import BridgedFMOutput
from fmllm.verifier.schema import (
    Hint,
    PhysicalStateClaim,
    SourceDecision,
    SourceVerdict,
    SourcesConfig,
    VerifierVerdict,
)


class NoOpVerifier:
    """Drop-in :class:`fmllm.verifier.Verifier` replacement.

    Always returns aggregate ``PASS`` with one stub ``noop`` source
    verdict. The ``verify(...)`` signature matches
    :meth:`fmllm.verifier.Verifier.verify` exactly so the OHVD loop
    can swap one for the other.
    """

    def __init__(self) -> None:
        self.default_config = SourcesConfig(
            rule_library=False,
            literature=False,
            cross_fm=False,
            simulator=False,
            conformal=False,
        )

    def available_sources(self) -> list[str]:
        return ["noop"]

    def verify(
        self,
        bridged_outputs: list[BridgedFMOutput],
        claim: PhysicalStateClaim,
        *,
        sources_config: SourcesConfig | None = None,
    ) -> VerifierVerdict:
        del bridged_outputs, claim, sources_config  # unused
        return VerifierVerdict(
            aggregate_decision=SourceDecision.PASS,
            source_verdicts=[
                SourceVerdict(
                    source_name="noop",
                    decision=SourceDecision.PASS,
                    confidence=1.0,
                    message="no-op verifier; baseline B2 has no verification",
                    evidence={},
                )
            ],
            hint=Hint(direction="no verifier signal in this baseline"),
            timestamp=datetime.now(UTC).isoformat(),
            sources_config=self.default_config,
        )


__all__ = ["NoOpVerifier"]
