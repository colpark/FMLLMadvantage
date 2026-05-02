"""The five verifier sources.

Each source exposes a ``check(bridged_outputs, claim) -> SourceVerdict``
method and consumes a list of :class:`BridgedFMOutput` objects plus the
LLM's :class:`PhysicalStateClaim`.
"""

from fmllm.verifier.sources.conformal import ConformalSource
from fmllm.verifier.sources.cross_fm import CrossFMSource
from fmllm.verifier.sources.literature import LiteratureSource
from fmllm.verifier.sources.rule_library import RuleLibrarySource
from fmllm.verifier.sources.simulator import SimulatorSource

__all__ = [
    "ConformalSource",
    "CrossFMSource",
    "LiteratureSource",
    "RuleLibrarySource",
    "SimulatorSource",
]
