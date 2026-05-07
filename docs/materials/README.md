# Materials Project port — CoT + SAE + SFT pipeline

This folder is the entry point for the inorganic crystalline
materials port of the Phase 16 positive recipe (probes + SAE
features → synthetic CoT → LoRA SFT → single-shot inference).

The recipe transfers from our Lennard-Jones testbed without
architectural changes:

```
LJ port (Phase 11/13/16)            Materials port
-------------------------            ----------------
FM2 (RDF transformer)        ->     CHGNet (crystal graph net)
probe targets                ->     formation_energy, e_above_hull,
                                    band_gap_class, space_group, magmom
SAE on FM2 CLS               ->     SAE on CHGNet pooled embedding
synthetic CoT (Step 1/1b...) ->     same shape, materials labels
LoRA SFT (Qwen 2.5 7B)       ->     same trainer
single-shot inference        ->     same runner shape
```

The LLM remains the same (Qwen 2.5 7B Instruct). The verifier is
*out of scope for the initial CoT+SAE+SFT goal* per the user's
direction; we'll port it later.

## Read in this order

| Doc | Purpose |
|---|---|
| `data_pipeline.md` | Concrete download + HDF5 build instructions, including Materials Project API key setup |
| `storage_budget.md` | Disk and GPU memory budget table for every artefact in the pipeline |
| `pipeline_stages.md` | The 9-stage end-to-end recipe from raw download to held-out inference |

## End-to-end recipe (cheat sheet)

```
# Stage 0: prerequisite -- API key + uv extras
export MP_API_KEY=<your-key-from-materialsproject.org>
uv sync --extra dev --extra materials

# Stage 1: download structures + properties from Materials Project
bash scripts/materials/00_download_mp.sh

# Stage 2: pack into HDF5 (positions, cell, species, properties)
bash scripts/materials/01_build_mp_h5.sh

# Stage 3: lock the held-out 200-specimen split
bash scripts/materials/02_lock_holdout.sh

# Stage 4: forward CHGNet over the dataset, cache pooled embeddings
bash scripts/materials/03_encode.sh

# Stage 5: train probe bank (5 probes on the cached embeddings)
bash scripts/materials/04_train_probes.sh

# Stage 6: train Top-K SAE on the cached embeddings
bash scripts/materials/05_train_sae.sh

# Stage 7: label SAE features by attribute correlation
bash scripts/materials/06_label_sae.sh

# Stage 8: build synthetic CoT records (probes + SAE features + ground truth)
bash scripts/materials/07_build_cot.sh

# Stage 9: SFT a LoRA adapter on the records (multi-GPU DDP)
NUM_GPUS=4 PER_DEVICE_BS=2 GRAD_ACCUM=4 MAX_SEQ=1536 bash scripts/materials/08_train_sft.sh

# Stage 10: single-shot inference on the held-out range, score
SPECIMEN_IDS_FILE=data/materials_project_v1/holdout_lock/ids.json bash scripts/materials/09_run_singleshot.sh
```

## Project structure

```
src/fmllm/materials/
  __init__.py
  dataset.py            # MaterialsSpecimenDataset (HDF5 reader)
  chgnet_wrap.py        # load CHGNet pretrained, expose .encode()
  ground_truth.py       # extract (e_form, e_above_hull, band_gap, space_group, magmom)
  synthetic_cot.py      # materials-specific CoT generator

scripts/materials/
  00_download_mp.{py,sh}
  01_build_mp_h5.{py,sh}
  02_lock_holdout.{py,sh}
  03_encode.{py,sh}                   # to be ported -- mirrors LJ encode steps
  04_train_probes.{py,sh}             # to be ported -- mirrors train_probe_bank
  05_train_sae.{py,sh}                # to be ported -- mirrors train_sae
  06_label_sae.{py,sh}                # to be ported -- mirrors label_sae_features
  07_build_cot.{py,sh}                # to be ported -- mirrors build_cot_dataset_with_sae
  08_train_sft.sh                     # thin wrapper over existing train_cot_sft.py
  09_run_singleshot.{py,sh}           # to be ported -- mirrors run_baseline_cot_sft_sae

data/materials_project_v1/
  raw/                     # raw API responses (compressed JSON)
  specimens.h5             # HDF5 -- positions, cell, species, properties
  splits.yaml              # train / held-out manifest
  holdout_lock/ids.json    # the 200 held-out specimen IDs

checkpoints/materials/
  chgnet/                  # downloaded CHGNet checkpoint (or pretrained reference)
  probes/                  # trained probe bank
  sae/                     # trained SAE
  cot-sft-sae-mat/         # trained LoRA adapter

runs/materials/
  cot_datasets_sae/        # synthetic CoT records
  sae_labels/              # SAE feature labels
  holdout/cot_sft_sae/     # single-shot inference outputs
  comparisons/             # side-by-side outputs
```

## Status of stages at handover

| Stage | File | Status |
|---|---|---|
| 1 download | `scripts/materials/00_download_mp.{py,sh}` | **Implemented** in this commit |
| 2 build h5 | `scripts/materials/01_build_mp_h5.{py,sh}` | **Implemented** |
| 3 lock holdout | `scripts/materials/02_lock_holdout.{py,sh}` | **Implemented** |
| 4 encode | `scripts/materials/03_encode.{py,sh}` | **To be ported** -- minor wrapper around CHGNet wrap class |
| 5 probes | `scripts/materials/04_train_probes.{py,sh}` | **To be ported** -- changes vs LJ are the probe targets |
| 6 SAE | `scripts/materials/05_train_sae.{py,sh}` | **To be ported** -- reuses `TopKSAE` class directly |
| 7 SAE labels | `scripts/materials/06_label_sae.{py,sh}` | **To be ported** -- swap LJ attribute set for materials attribute set |
| 8 CoT records | `scripts/materials/07_build_cot.{py,sh}` | **To be ported** -- uses materials `synthetic_cot` (already implemented) |
| 9 SFT | `scripts/materials/08_train_sft.sh` | Direct wrapper over existing `train_cot_sft.py` (general) |
| 10 Inference | `scripts/materials/09_run_singleshot.{py,sh}` | **To be ported** -- mirrors `run_baseline_cot_sft_sae.py` |
| Module: `src/fmllm/materials/dataset.py` | | **Implemented** |
| Module: `src/fmllm/materials/chgnet_wrap.py` | | **Implemented** |
| Module: `src/fmllm/materials/synthetic_cot.py` | | **Implemented** |
| Module: `src/fmllm/materials/ground_truth.py` | | **Implemented** |
| Tests | `tests/test_materials_synthetic_cot.py` | **Implemented** (CPU) |

The "to be ported" scripts are mechanically obvious copies of the
LJ scripts; each is ~150-300 lines that change three things (FM
loader, probe targets / attribute extractor, output paths). I've
left them as intentional follow-ups for one of two reasons:
they need at least a smoke run on the actual data before the
implementation can be locked, *and* the fastest dev loop is to
get the data through the pipeline first, then iterate on the
analysis scripts.

## Why the "verifier later" decision

Per user direction the initial goal is CoT + SAE + SFT. The
LJ Phase 16 result showed the no-verifier single-shot CoT-SFT
recipe reached 0.650 on a 0.695 ceiling -- 94% of the
verifier-using ceiling without any inference-time orchestration.
On materials we expect a similar pattern: the verifier is an
additional +5-15 point lift, but the bulk of the value is in
the SFT'd LLM reading rich evidence. Building the pipeline in
this order (CoT+SAE+SFT first, verifier second) lets us
validate the recipe transfer before committing to the
materials verifier infrastructure (PyMatGen rules, MP API
queries, DFT pseudo-oracle).

The Phase 16 negative on `cot_sft_sae + verifier` (where the
SFT'd over-confidence broke the verifier's calibrated
abstention) is a separate caveat for the materials project
once we add the verifier. We'll need verifier-aware SFT (Stage
3-4 of Phase 11) to make them genuinely additive.

## Open scope notes

- **Cross-FM verifier source** in materials means CHGNet vs
  MACE-MP-0 vs M3GNet on energy / forces. The cross-FM check
  is unusually strong here because all three are universal
  potentials trained on overlapping datasets; their
  disagreements are concentrated on out-of-distribution
  structures, which is exactly where the verifier should
  raise CAVEAT.
- **Larger Phase-15-style SAE on Qwen** is more interesting
  here than on LJ because materials inputs (composition, space
  group, formula) are richer than LJ tuples; the LLM
  representation may have meatier interpretable features.
- **Continuous + discrete output composite**: the suggested
  goal is `(formation_energy, e_above_hull, is_stable,
  band_gap_class, space_group_top1)`. Continuous regression
  on two scalars + binary classification + 3-way discrete +
  230-way discrete. Much richer label space than LJ's
  `(motif, n_atoms, T)`, which addresses the "discretized
  output saturates" caveat from the LJ project.
