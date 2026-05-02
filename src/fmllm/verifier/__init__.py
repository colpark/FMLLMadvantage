"""Multi-source verifier with rule, literature, cross-FM, simulator, and
conformal sources.

The verifier reads :class:`BridgedFMOutput` objects produced by the
bridges in :mod:`fmllm.bridges`, evaluates the LLM's
:class:`PhysicalStateClaim`, and emits a structured
:class:`VerifierVerdict` with a per-source breakdown plus a
revision :class:`Hint`.

The integrator accepts a runtime :class:`SourcesConfig` that disables
specific sources at call time, supporting the E4 ablation experiment.
"""

from fmllm.verifier.integrator import Verifier, build_default_verifier
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

__all__ = [
    "ConformalSource",
    "CrossFMSource",
    "Hint",
    "LiteratureSource",
    "PhysicalStateClaim",
    "RuleLibrarySource",
    "SimulatorSource",
    "SourceDecision",
    "SourceVerdict",
    "SourcesConfig",
    "Verifier",
    "VerifierVerdict",
    "build_default_verifier",
]
