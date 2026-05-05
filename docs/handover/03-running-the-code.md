# 03 — Running the Code

## Operating discipline (read first)

The user runs **all** tests, training, and baselines on a remote
4xH100 host. The local Mac copy is for editing and committing
only.

> Do not run `pytest`, `uv sync`, `uv run`, `bash scripts/...`,
> or `uv run python scripts/...` locally. These have caused NFS /
> venv issues in past sessions and produce results the user does
> not need.

The locally-saved memory note that records this rule:
`~/.claude/projects/-Users-davidpark-Documents-Claude-FMLLM/memory/feedback_no_local_testing.md`.

Workflow:

1. Edit code locally.
2. Syntax-check via `python -c "import ast; ast.parse(open('path').read())"`.
3. Commit and push.
4. User runs `git pull && bash scripts/<thing>.sh` on remote.
5. Read the user's pasted output, iterate.

## Remote environment

- Host pattern: `<container-id>:~/workspace/Temporary/dpark1/scratch/FMLLM/FMLLMadvantage`
- Filesystem: NFS-backed scratch (this is the source of several
  pain points; see "Infrastructure issues" below).
- Container is shared and **periodically recycles**, killing
  long-running Python processes.
- Python: 3.11
- Package manager: `uv` 0.10.x
- GPU: 4x H100 (use `CUDA_VISIBLE_DEVICES=0` to pin to one
  unless explicitly parallelizing).

## First-time setup on a fresh container

```bash
cd ~/workspace/Temporary/dpark1/scratch/FMLLM/FMLLMadvantage

# Drop conda if it's the active env
conda deactivate 2>/dev/null

# Make sure uv is on PATH
export PATH="$HOME/.local/bin:$PATH"

# NFS hates hardlinks; force copy mode permanently
echo 'export UV_LINK_MODE=copy' >> ~/.bashrc
export UV_LINK_MODE=copy

# Move venv off NFS to local disk (much more stable)
echo 'export UV_PROJECT_ENVIRONMENT="$HOME/.cache/fmllm-venv"' >> ~/.bashrc
export UV_PROJECT_ENVIRONMENT="$HOME/.cache/fmllm-venv"

# Sync once
uv sync --extra dev
```

Verify with `uv run python --version` (should be 3.11).

## Common workflows

### A. Run the held-out evaluator only

If trajectory files already exist on disk:

```bash
BASELINES_ROOT=runs/holdout bash scripts/evaluate_baselines.sh
```

Auto-discovers any `runs/holdout/*/<run_id>/trajectories.jsonl`
and produces a side-by-side. ~5-10 min.

### B. Run a Phase 8a baseline from scratch

```bash
SPECIMEN_IDS_FILE=runs/holdout_lock/ids.json \
    bash scripts/run_baseline.sh full
```

Or `naked` / `no_verifier`. Output to `runs/holdout/full/<run_id>/`.
~30-50 min for 200 specimens with Qwen 2.5 7B 4-bit.

### C. Run the Phase 13 SAE-injection baseline

```bash
# Stages 0-1-2 in one shot (resume-aware on stages 0 and 1)
SPECIMEN_IDS_FILE=runs/holdout_lock/ids.json \
    bash scripts/run_phase13_full_sae.sh

# Or directly via the simpler wrapper if the above fails
bash scripts/run_full_sae_direct.sh
```

### D. Run Phase 14 causal audit + filtered baseline

```bash
bash scripts/sae_causal_audit.sh                       # Stage A, ~10 min

# Then run the LLM-side baseline with the filter:
SAE_DIR=$(ls -td checkpoints/sae/*/ | head -1) \
SAE_LABELS_PATH=$(ls -td runs/sae_labels/*/labels.json | head -1) \
CAUSAL_FILTER_PATH=$(ls -td runs/sae_causal/*/causal_filter.json | head -1) \
SPECIMEN_IDS_FILE=runs/holdout_lock/ids.json \
    bash scripts/run_baseline_full_probes.sh
```

### E. Phase 15 (full pipeline)

```bash
# Stage A: harvest Qwen activations (~5-10 min for 200 specimens)
bash scripts/harvest_qwen_activations.sh

# Stage B: train Top-K SAE on harvested activations (~3-15 min)
bash scripts/train_qwen_sae.sh

# Stage C: label features (CPU, ~30 sec)
bash scripts/label_qwen_sae_features.sh

# Inspect candidates and diagnose ground-truth wrong-PASS pattern
bash scripts/inspect_qwen_sae_candidates.sh

# Stage D: steered Pipeline A baseline (~30-50 min, resume-aware)
FEATURE_IDX=<picked-from-step-above> COEFFICIENT=-2.0 \
    SPECIMEN_IDS_FILE=runs/holdout_lock/ids.json \
    bash scripts/run_baseline_qwen_steered.sh
```

### F. Re-evaluate everything

```bash
BASELINES_ROOT=runs/holdout bash scripts/evaluate_baselines.sh
```

## Long-running jobs: nohup + resume

The container can recycle mid-run. To survive:

```bash
nohup bash -c '
SPECIMEN_IDS_FILE=runs/holdout_lock/ids.json \
    bash scripts/run_baseline_qwen_steered.sh
' > /tmp/job.log 2>&1 &

# Monitor
tail -f /tmp/job.log
```

If the container does recycle:

```bash
cd ~/workspace/Temporary/dpark1/scratch/FMLLM/FMLLMadvantage
git pull   # in case fixes were pushed
# Re-run the same command — resume picks up the partial work
SPECIMEN_IDS_FILE=runs/holdout_lock/ids.json \
    bash scripts/run_baseline_qwen_steered.sh
```

The Phase 15 steered runner (`run_baseline_qwen_steered.py`) and
the Phase 13 probe-augmented runner
(`run_baseline_full_probes.py`) both have resume logic. The
canonical `run_baseline.py` does NOT yet have resume; if a `full`
run is killed mid-stream the partial JSONL is wiped on rerun.
**Beware.**

## Infrastructure issues encountered (and fixes)

### NFS hardlink failures during `uv sync`

Symptom:

```
warning: Failed to hardlink files; falling back to full copy.
error: Failed to install: h5py-...whl
  Caused by: No such file or directory at .../site-packages/h5py/.tmp...
```

Cause: `uv` defaults to hardlinking, which fails on NFS.

Fix: `export UV_LINK_MODE=copy` (added to `.bashrc` in step A).
Move the venv off NFS to `$HOME/.cache/...` for further isolation.

### Container recycles killing long jobs

Symptom: GPU usage drops to 0%, prompt prefix changes (e.g.
`(base)` -> `(py39)`), container ID in the host name changes.

Cause: shared host periodically refreshes containers.

Fix: resume support in long-running runners + `nohup` for
detaching from terminal lifecycle. Specifically the steered
runner skips already-done specimen IDs in the existing
trajectories.jsonl and appends the new ones.

### Concurrent `uv` / `python` processes corrupting venv

Symptom: random h5py / transformers import errors after running
multiple commands in parallel.

Cause: two terminals running `uv run python ...` simultaneously
fight over the same venv; one's installation can wipe the other's
references.

Fix: one venv per parallel job:

```bash
UV_PROJECT_ENVIRONMENT="$HOME/.cache/fmllm-venv-A" \
    bash scripts/job_a.sh   # in terminal A

UV_PROJECT_ENVIRONMENT="$HOME/.cache/fmllm-venv-B" \
    bash scripts/job_b.sh   # in terminal B
```

### Loguru / typer line truncation in pasted output

Sometimes the user pastes terminal output that wraps oddly. Long
labels in the side-by-side table truncate (e.g.
`full_steered_6844_n200` -> `full_steered_`). Don't infer
truncated state; ask for a re-paste or `cat` the comparison.yaml
directly.

### 4-bit model + activation steering dtype concerns

The `ActivationSteerer` resolves the model's compute dtype via
`for p in model.parameters(): if p.dtype in (fp16, bf16, fp32): ...`.
On 4-bit Qwen this picks bf16 (the LayerNorms / embeddings). The
hook adds bf16 deltas to a bf16 residual. Tested working in
Phase 15 Stage D.

If you ever see `RuntimeError: expected ... to be ...` from a
hook, that's the dtype path; trace through `_maybe_load_steering`
in `fmllm.representation.steered_llm`.

### Resume + duplication trap

If a runner has resume but writes mode='a' without checking
done-IDs, you can end up with duplicate trajectories. The Phase
13 `full_probes` run hit this — JSONL ended up with 400 lines
for 200 specimens.

Sanity check after any resumed run:

```bash
LATEST=$(ls -td runs/holdout/<baseline>/*/ | head -1)
wc -l "${LATEST}/trajectories.jsonl"   # should match specimen count
```

## Useful one-liners

```bash
# What baselines are on disk?
ls runs/holdout/

# Newest run of a given baseline
ls -td runs/holdout/full/*/ | head -1

# Read first trajectory of a run
LATEST=$(ls -td runs/holdout/full/*/ | head -1)
head -1 "${LATEST}/trajectories.jsonl" | uv run python -m json.tool | head -50

# Verdict mix in a run
LATEST=$(ls -td runs/holdout/full/*/ | head -1)
uv run python -c "
import json, collections
ctr = collections.Counter()
for line in open('${LATEST}/trajectories.jsonl'):
    t = json.loads(line)
    v = (t.get('final_verdict') or {}).get('aggregate_decision', 'null')
    ctr[v] += 1
print(dict(ctr))
"

# Disk usage by run-dir (when cleaning up)
du -sh runs/holdout/*/ | sort -h
```

## Cost rough estimates

(Wall-clock on one H100 with 4-bit quantization)

- Pipeline A on 200 specimens: ~30-50 min
- FM2 SAE training (`train_sae.sh`, 16K hidden, 30 epochs on
  20K rows): ~5-15 min
- Qwen activation harvesting (200 trajectories): ~5-10 min
- Qwen SAE training (16384 hidden, 30 epochs on 200 rows): ~30 sec
  (small data); 5-15 min on larger sets
- Qwen SAE labelling (CPU): ~30 sec
- Phase 14 causal audit: ~5-10 min on 2000 specimens

## When something doesn't run

Trace the chain backwards:

1. The runner script's banner — does it print sane paths and
   parameters? If `SAE dir: (none)` in a Phase 13 run, your env
   var was forgotten.
2. The first ~30 lines of the python CLI's stdout — does it find
   its inputs? Most "missing X" errors print before model
   loading.
3. The `run.log` inside the run directory — `tail -50` for
   tracebacks.
4. The user's GPU / process state — `nvidia-smi` and `ps -ef |
   grep python`.

If a fix needs new code, push it; the user pulls and reruns. The
cycle is fast as long as the user's iteration loop is not
blocked.
