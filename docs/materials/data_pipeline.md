# Materials port — Data pipeline

End-to-end procedure to get the Materials Project data on disk
and into the HDF5 format the rest of the pipeline consumes.

## Prerequisites

```
# 1. Get a free Materials Project account
#    https://materialsproject.org

# 2. Get the API key
#    https://next-gen.materialsproject.org/api  (settings -> API key)

# 3. Export it (single line)
export MP_API_KEY=<your-key>

# 4. Sync the materials extras
uv sync --extra dev --extra materials
```

The `materials` extra brings in `mp-api` (REST client),
`pymatgen` (structure handling + rule library), and `chgnet`
(the FM-equivalent for materials).

## Stage 1 — Download structures + properties

```
bash scripts/materials/00_download_mp.sh
```

**What it does:** queries the Materials Project new API for all
materials matching the filter (default: `e_above_hull < 0.5
eV/atom`), pulls structures + properties, caches each as a
gzipped JSON in `data/materials_project_v1/raw/`.

**Configurable env vars** (single line each):

| Var | Default | Purpose |
|---|---|---|
| `N_MAX` | 50000 | Cap on number of materials to fetch |
| `E_ABOVE_HULL_MAX` | 0.5 | Stability filter (eV/atom) |
| `RAW_DIR` | `data/materials_project_v1/raw` | Output directory |
| `BATCH` | 500 | Materials per API page |
| `RESUME` | 1 | Skip already-downloaded materials |

**Storage**: 2-4 GB of gzipped JSON depending on N_MAX.

**Time**: 30 min - 2 hours (rate-limited by Materials Project
API and your network). The script supports interruption +
resume so you can stop and restart freely.

The download script extracts the following fields per material
(this is the canonical schema we'll use throughout):

```yaml
material_id: str         # MP id, e.g. "mp-149"
formula_pretty: str      # e.g. "Si"
elements: [str]          # e.g. ["Si"]
nsites: int              # number of atoms in the unit cell
volume: float            # cell volume in Å^3
density: float           # g/cm^3
formation_energy_per_atom: float    # eV/atom (DFT)
energy_above_hull: float            # eV/atom (DFT, stability)
band_gap: float                     # eV
is_metal: bool
is_magnetic: bool
total_magnetization: float | null   # μB (None for non-magnetic)
symmetry:
  crystal_system: str               # one of {triclinic, monoclinic, ...}
  space_group_symbol: str
  space_group_number: int           # 1..230
structure:                          # pymatgen Structure dict
  lattice:
    matrix: 3x3 list                # cell vectors in Å
  sites:
    - species: [{element: "Si", occu: 1.0}]
      xyz: [x, y, z]                # Cartesian Å
```

## Stage 2 — Pack into HDF5

```
bash scripts/materials/01_build_mp_h5.sh
```

**What it does:** consumes the raw JSON cache and writes a
single HDF5 file `data/materials_project_v1/specimens.h5` with
fixed-shape arrays (padded for variable atom counts).

**HDF5 layout** (mirrors our LJ HDF5 shape):

```
specimens.h5
├── attrs:
│     n_specimens: int
│     element_names: [str]              # 100-element vocabulary
│     space_group_names: [str]
│     created_utc: str
├── material_id: (N,) string
├── formula_pretty: (N,) string
├── nsites: (N,) int32
├── volume: (N,) float32
├── density: (N,) float32
├── formation_energy_per_atom: (N,) float32
├── energy_above_hull: (N,) float32
├── band_gap: (N,) float32
├── is_metal: (N,) bool
├── total_magnetization: (N,) float32     # NaN for non-magnetic
├── space_group_number: (N,) int32
├── crystal_system_id: (N,) int32         # 0..6
├── n_atoms_padded: (N, MAX_ATOMS) int8   # element index, -1 = pad
├── positions_padded: (N, MAX_ATOMS, 3) float32
├── lattice: (N, 3, 3) float32
└── padding_mask: (N, MAX_ATOMS) bool
```

`MAX_ATOMS` defaults to 80 (covers 99% of MP unit cells with
`e_above_hull < 0.5 eV/atom`). Larger cells are dropped during
the build with a logged count.

**Configurable env vars**:

| Var | Default | Purpose |
|---|---|---|
| `RAW_DIR` | `data/materials_project_v1/raw` | Source |
| `H5_PATH` | `data/materials_project_v1/specimens.h5` | Output |
| `MAX_ATOMS` | 80 | Atom-count truncation |
| `MIN_ATOMS` | 1 | Drop below |

**Time**: 5-15 minutes for 50K materials.

## Stage 3 — Lock the held-out split

```
bash scripts/materials/02_lock_holdout.sh
```

**What it does:** Picks 200 materials for held-out evaluation
with stratified sampling so each major crystal system and
band-gap class is represented. Writes:

- `data/materials_project_v1/splits.yaml` -- train / holdout assignments
- `data/materials_project_v1/holdout_lock/ids.json` -- the 200 ids

The held-out 200 are excluded from probe training, SAE training,
and CoT-SFT training. Mirrors the LJ project's locked
`runs/holdout_lock/ids.json` pattern.

**Configurable env vars**:

| Var | Default | Purpose |
|---|---|---|
| `N_HOLDOUT` | 200 | Held-out size |
| `STRATIFY_BY` | `crystal_system,is_metal` | Comma-separated stratification keys |
| `SEED` | 0 | Reproducibility |

The stratification ensures the held-out 200 has a
representative mix of {triclinic, monoclinic, ..., cubic} ×
{metal, non-metal}. Without this, ~85% of the held-out would be
the dominant cubic non-metal bucket and the eval would be
narrow.

## Sanity checks after stages 1-3

```
ls -lh data/materials_project_v1/
```

Expected output (single line each):

- `raw/` directory with several thousand `.json.gz` files
- `specimens.h5` -- 0.8-1.5 GB
- `splits.yaml` -- a few KB
- `holdout_lock/ids.json` -- 200-element JSON list

```
uv run python -c "import h5py; f=h5py.File('data/materials_project_v1/specimens.h5','r'); print('n=',f.attrs['n_specimens']); print('keys=',list(f.keys())); print('e_form range=',f['formation_energy_per_atom'][:].min(),f['formation_energy_per_atom'][:].max()); print('e_above_hull range=',f['energy_above_hull'][:].min(),f['energy_above_hull'][:].max())"
```

You should see something like:

```
n= 50000
keys= ['band_gap', 'crystal_system_id', 'density', ...]
e_form range= -5.2 0.5
e_above_hull range= 0.0 0.499
```

## What the rest of the pipeline does with this data

Stages 4-10 run unchanged for materials vs LJ. The only
difference is the FM (CHGNet vs FM2-RDF) and the per-specimen
ground truth (formation_energy/e_above_hull/etc vs motif/n/T).

The synthetic CoT generator at `src/fmllm/materials/synthetic_cot.py`
emits chains of the form:

```
Step 1 — Read the probes:
  - formation-energy probe : −2.85 eV/atom (confidence 0.91)
  - e-above-hull probe     : 0.018 eV/atom (confidence 0.85)
  - band-gap probe         : 1.23 eV (confidence 0.88)
  - magnetization probe    : 0.0 μB    (confidence 0.93)
  - space-group probe      : Fm-3m (confidence 0.79)

Step 1b — Read the SAE-derived features:
  - f127: crystal=cubic + e_form-low(r=−0.62) + non-metal (activation 3.04)
  - f342: e_above_hull-low (purity 0.92, n=50) (activation 2.81)
  - f78: heavy-element + magnetic (activation 1.12)
  ...

Step 2 — Cross-check stability:
  e_above_hull ≈ 0.018 eV/atom is below the standard 0.025 eV/atom
  stability cutoff. Formation energy −2.85 eV/atom is consistent
  with the FCC structure family.

Step 3 — Resolution:
  All probes agree this is a stable cubic non-metal with a
  ~1.2 eV gap. The space-group probe disambiguates Fm-3m vs Fd-3m;
  Fm-3m has higher confidence.

Final commit: {"formation_energy": -2.85, "e_above_hull": 0.018,
"is_stable": true, "band_gap_class": "narrow", "space_group": 225}
```

Same structure as LJ's CoTs, materials labels.

## Troubleshooting

**Stage 1 errors with `MP_API_KEY` not set**: export it before
running. The script will not write any data without a valid key.

**Stage 1 stuck at low rate**: legacy quota. Sign up for a new
account; the new API has 5x higher limits.

**Stage 1 hangs on a single material**: kill and rerun. Resume
will skip everything that's already cached. Check `RAW_DIR` line
count: `ls data/materials_project_v1/raw/ | wc -l`.

**Stage 2 OOM**: drop `BATCH_BUILD` to 1000 (default 5000) and
the script will stream-build the HDF5 instead of loading all
records into memory.

**Stage 3 stratification leaves gaps**: lower `STRATIFY_BY` to
just `crystal_system` if some buckets have <10 specimens after
filtering.
