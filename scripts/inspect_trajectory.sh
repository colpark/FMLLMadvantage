#!/usr/bin/env bash
#
# inspect_trajectory.sh
#
# Pretty-print a trajectory.json from a run directory. Shows the
# step sequence, final claim, final verdict, and a per-step summary.
#
# Usage:
#   bash scripts/inspect_trajectory.sh                              # latest run
#   bash scripts/inspect_trajectory.sh runs/<run_id>                # specific run dir
#   bash scripts/inspect_trajectory.sh runs/<run_id>/trajectory.json  # exact file

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [ "$#" -eq 0 ]; then
    LATEST_DIR=$(ls -td runs/*pipeline-a* 2>/dev/null | head -1 || true)
    if [ -z "${LATEST_DIR}" ]; then
        echo "No trajectory directory found under runs/."
        echo "Run scripts/run_pipeline_smoke.sh or scripts/run_pipeline_real.sh first."
        exit 1
    fi
    TRAJ_PATH="${LATEST_DIR}/trajectory.json"
elif [ -d "$1" ]; then
    TRAJ_PATH="$1/trajectory.json"
else
    TRAJ_PATH="$1"
fi

if [ ! -f "${TRAJ_PATH}" ]; then
    echo "Trajectory file not found: ${TRAJ_PATH}"
    exit 1
fi

echo "==> Inspecting ${TRAJ_PATH}"
echo

uv run python - "${TRAJ_PATH}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path) as f:
    t = json.load(f)

def short(x, w=80):
    s = json.dumps(x, sort_keys=True) if not isinstance(x, str) else x
    return s if len(s) <= w else s[: w - 3] + "..."

print(f"Run ID         : {t.get('run_id')}")
print(f"Query          : {short(t.get('query'))}")
print(f"Specimen ID    : {t.get('specimen_id')}")
print(f"Termination    : {t.get('termination')}")
print(f"Steps          : {len(t.get('steps', []))}")
print()

print("Step sequence:")
for s in t.get("steps", []):
    idx = s.get("step_index")
    typ = s.get("step_type")
    note = s.get("notes") or ""
    detail = ""
    if typ == "observation":
        bo = s.get("bridged_output") or {}
        fm = (bo.get("source") or {}).get("fm_name", "?")
        quantity = (bo.get("prediction") or {}).get("quantity", "?")
        detail = f"  fm={fm}  quantity={quantity}"
    elif typ in ("hypothesis", "final"):
        c = s.get("claim") or {}
        detail = "  claim=" + short({k: v for k, v in c.items() if v is not None}, 60)
    elif typ == "verifier_verdict":
        vd = s.get("verdict") or {}
        decision = vd.get("aggregate_decision")
        detail = f"  aggregate={decision}"
        flagged = (vd.get("hint") or {}).get("flagged_sources") or []
        if flagged:
            detail += f"  flagged={flagged}"
    elif typ == "error":
        detail = f"  note={short(note, 60)}"
    print(f"  {idx:>3}  {typ:<18}{detail}")

if t.get("final_claim") is not None:
    print()
    print("Final claim:")
    for k, v in t["final_claim"].items():
        if v is not None:
            print(f"  {k}: {v}")

if t.get("final_verdict") is not None:
    fv = t["final_verdict"]
    print()
    print(f"Final verdict   : {fv.get('aggregate_decision')}")
    for sv in fv.get("source_verdicts", []):
        msg = short(sv.get("message", ""), 60)
        print(f"  {sv.get('source_name'):<14} {sv.get('decision'):<8} conf={sv.get('confidence'):.2f}  {msg}")
    hint = fv.get("hint") or {}
    if hint.get("flagged_sources"):
        print(f"Hint            : {hint.get('direction')}")
        for s, r in zip(hint.get("flagged_sources", []), hint.get("suggested_revisions", [])):
            print(f"  {s}: {short(r, 80)}")
PY
