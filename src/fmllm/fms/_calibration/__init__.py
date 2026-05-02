"""Cross-FM calibration utilities.

Each FM trains its conformal calibrator independently against its own
ground truth. After all three FMs at a given training scale finish,
``cross_fm_tolerance.py`` measures pairwise empirical agreement on
shared causal variables and writes a tolerance matrix the verifier's
cross-FM source reads.
"""

from fmllm.fms._calibration.cross_fm_tolerance import (
    CrossFMToleranceMatrix,
    compute_cross_fm_tolerances,
    load_tolerance_matrix,
    save_tolerance_matrix,
)

__all__ = [
    "CrossFMToleranceMatrix",
    "compute_cross_fm_tolerances",
    "load_tolerance_matrix",
    "save_tolerance_matrix",
]
