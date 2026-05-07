# Materials port — Storage and compute budget

The full pipeline from cold start (no Materials Project data on
disk) to a trained `cot_sft_sae` adapter on materials and a 200-
specimen held-out evaluation needs the following.

## Per-stage disk budget

Estimates assume ~50,000 stable-or-near-stable structures from
Materials Project (filter: `e_above_hull < 0.5 eV/atom`). Smaller
samples scale roughly linearly.

| Stage | Artefact | Size (est.) | Path |
|---|---|---|---|
| 1 | Raw API responses (compressed JSON) | **2-4 GB** | `data/materials_project_v1/raw/` |
| 2 | HDF5 packed dataset | **0.8-1.5 GB** | `data/materials_project_v1/specimens.h5` |
| 3 | Held-out IDs | <10 KB | `data/materials_project_v1/holdout_lock/ids.json` |
| 3 | Splits manifest | <100 KB | `data/materials_project_v1/splits.yaml` |
| 4 | CHGNet pretrained checkpoint | **~5 MB** | `checkpoints/materials/chgnet/` |
| 4 | Cached pooled embeddings (50K × 64-128 dim, fp32) | **15-30 MB** | `runs/materials/embeddings/<run>/embeddings.npy` |
| 5 | Probe bank checkpoint (5 small MLPs) | **~5 MB** | `checkpoints/materials/probes/<run>/` |
| 6 | SAE checkpoint (in_dim 128, hidden 1024, k 32) | **~10 MB** | `checkpoints/materials/sae/<run>/sae.pt` |
| 6 | SAE training history | <1 MB | `checkpoints/materials/sae/<run>/training.yaml` |
| 7 | SAE feature labels (1024 entries) | **~500 KB** | `runs/materials/sae_labels/<run>/labels.json` |
| 7 | Detailed feature records | **~5 MB** | `runs/materials/sae_labels/<run>/details.yaml` |
| 8 | Synthetic CoT records (50K JSONL records) | **150-300 MB** | `runs/materials/cot_datasets_sae/<run>/records.jsonl` |
| 9 | LoRA adapter (r=16, α=32) on Qwen 2.5 7B | **~50 MB** | `checkpoints/materials/cot-sft-sae-mat/<run>/adapter/` |
| 9 | Trainer state + checkpoints (epoch saves) | **150-400 MB** | `checkpoints/materials/cot-sft-sae-mat/<run>/trainer/` (deleteable post-training) |
| 10 | Held-out trajectories.jsonl (200 commits) | **~5 MB** | `runs/materials/holdout/cot_sft_sae/<run>/trajectories.jsonl` |
| 10 | Comparison reports | <1 MB | `runs/materials/comparisons/<run>/` |

**Working-set total (post-cleanup of trainer intermediates): ~1.5-2 GB**
**Peak total (with raw cache + trainer state pre-cleanup): ~3-5 GB**

## Required tools and their sizes

These are needed in the environment but not part of the project's
on-disk artefacts. They sit under `~/.cache/...` or the python
venv:

| Tool | Approx size | Where it lands |
|---|---|---|
| Qwen 2.5 7B Instruct (HF download) | ~14 GB (bf16) or ~5 GB (4-bit) | `~/.cache/huggingface/hub/` |
| CHGNet pretrained | ~5 MB | downloaded by `chgnet.model.CHGNet.load()` |
| MACE-MP-0 medium (optional, for cross-FM verifier later) | ~30 MB | downloaded on first use |
| pymatgen + dependencies | ~200 MB | venv |
| chgnet + ase + others | ~100 MB | venv |
| mp-api | ~20 MB | venv |

## GPU memory per stage

Single H100 (80 GB) is more than enough for every stage. Multi-GPU
DDP (4× H100) only helps Stage 9.

| Stage | Component | Peak VRAM |
|---|---|---|
| 4 (encode) | CHGNet forward (batch 64 structures) | ~2 GB |
| 5 (probes) | Tiny MLPs on cached embeddings (CPU-fine) | ~500 MB |
| 6 (SAE) | TopKSAE on cached embeddings (CPU-fine) | ~500 MB |
| 7 (label) | CHGNet forward + SAE encode + correlations | ~2 GB |
| 8 (build CoT) | CHGNet + probes + SAE forward, no LLM | ~2-3 GB |
| 9 (SFT) | Qwen 2.5 7B bf16 + LoRA + grad ckpt + DDP | ~30-40 GB / GPU |
| 10 (inference) | Qwen + LoRA + CHGNet forward | ~14-16 GB |

## Compute time on a single H100 (rough estimates)

For a 50K-specimen pipeline:

| Stage | Wall clock |
|---|---|
| 1 download (rate-limited by MP API) | **30 min - 2 hours** |
| 2 build h5 | 5-15 min |
| 3 lock holdout | <1 min |
| 4 encode 50K via CHGNet | **30-60 min** |
| 5 train probes (5 probes × 50K × 30 epochs) | 5-10 min |
| 6 train SAE (1024 features × 50K × 30 epochs) | 5-15 min |
| 7 label SAE features | 5-10 min |
| 8 build synthetic CoT records | 15-30 min |
| 9 SFT LoRA on 50K records (4-GPU DDP) | **45-75 min** |
| 10 inference on 200 held-out (bf16 Qwen) | 10-15 min |

**End-to-end from cold start: ~3-5 hours of wall clock.**

If you need to shave time:

- **Smaller sample**: 10K specimens cuts everything roughly 5× while still giving a meaningful SFT signal. Use `N_SPECIMENS=10000 bash scripts/materials/01_build_mp_h5.sh`.
- **Skip stage 4 caching**: if RAM holds 50K embeddings (it does), we can compute them on the fly in stages 5-7. The cached version is for amortization across stages and quick re-runs.

## Materials Project API rate limits and key

The download stage requires:

1. **Free Materials Project account**: <https://materialsproject.org>
2. **API key**: <https://next-gen.materialsproject.org/api> (settings → API key)
3. **Set the env var** before stage 1: `export MP_API_KEY=<your-key>`

Rate limits as of handover (subject to change):

- ~600 requests / minute on the legacy API
- ~3000 requests / minute with `mp-api` library (newer)
- One query can return up to 10000 documents per page

The script auto-paginates and handles 429s with exponential
backoff. Expect ~30-60 minutes for the 50K-specimen download
on the newer API. If it's slower than that, your account may
be on the legacy quota.

## Checkpoints we need

For the materials port, before running stage 4, you need:

1. **CHGNet pretrained weights** -- automatically downloaded by
   `chgnet.model.CHGNet.load()` on first call. ~5 MB.
2. **Materials Project structures + properties** -- downloaded
   by stage 1.

That's it. We do not pretrain a CHGNet from scratch; we use the
public pretrained checkpoint as our "FM2 equivalent."

## Cleanup commands

After the pipeline has run end-to-end and you have the trained
adapter + trajectories, you can free disk by deleting:

- `checkpoints/materials/cot-sft-sae-mat/*/trainer/` -- HF
  Trainer's intermediate checkpoints, ~200-400 MB.
- `data/materials_project_v1/raw/` -- raw API JSON cache, ~2-4 GB.
  Once the HDF5 is built, the raw cache is only useful if you want
  to add new fields without re-downloading.

These two together typically free ~3-4 GB.
