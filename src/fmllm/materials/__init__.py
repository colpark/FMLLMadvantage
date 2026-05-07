"""Materials Project port of the Phase 16 CoT + SAE + SFT recipe.

Mirrors the structure of the LJ pipeline (`fmllm.fms.fm2_rdf`,
`fmllm.training.synthetic_cot`, `fmllm.data.dataset`) but for
inorganic crystalline materials sourced from Materials Project.

Submodules:
    dataset            HDF5 reader for Materials Project specimens
    chgnet_wrap        wraps CHGNet pretrained model, exposes pooled embeddings
    ground_truth       extracts (formation_energy, e_above_hull,
                       is_stable, band_gap_class, space_group) per material
    synthetic_cot      materials-specific CoT generator with Step 1 / Step 1b
                       structure (probes + SAE features + final commit)
"""

from fmllm.materials.synthetic_cot import (
    GroundTruthMaterials,
    MaterialsCoT,
    build_sft_record,
    generate_cot,
)

__all__ = [
    "GroundTruthMaterials",
    "MaterialsCoT",
    "build_sft_record",
    "generate_cot",
]
