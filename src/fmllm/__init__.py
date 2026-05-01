"""FMLLMadvantage package root.

This package implements the compositional world-model pipeline described
in the project README. Submodules group code by component: data
generation, foundation models, bridges, verifier, orchestrator, training,
evaluation, physics, and shared utilities.

The public API stays minimal at the top level. Importers reach into the
relevant submodule directly.
"""

__version__ = "0.1.0"
