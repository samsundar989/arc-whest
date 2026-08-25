#!/usr/bin/env bash
# P2-P0 gate clause G4 -- provoke RLIMIT_AS on Linux, where the limit exists.
# Run from WSL. Writes its raw transcript and machine-readable evidence beside it.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="$HOME/.venv-whest-p2"
PY="$VENV/bin/python"
WHEST="$VENV/bin/whest"

export HF_HOME=/mnt/c/Users/samsu/.cache/huggingface     # reuse the Windows cache
export PYTHONIOENCODING=utf-8
export ARC_WHEST_MEM_SENTINEL=/tmp/arc_whest_mem_sentinel
export ARC_WHEST_MEM_REPORT="$HERE/rlimit_as_worker_report.json"
export ARC_WHEST_MEM_REQUEST=$((9 * 1024 * 1024 * 1024))   # 9 GiB, never touched
export ARC_WHEST_MEM_CONTROL=$((256 * 1024 * 1024))        # control, must succeed
rm -f "$ARC_WHEST_MEM_SENTINEL" "$ARC_WHEST_MEM_REPORT"

echo "=============================================================="
echo "P2-P0 G4 -- RLIMIT_AS provocation, Ubuntu WSL, official stack"
echo "=============================================================="
echo "--- environment ---"
grep -E '^(NAME|VERSION)=' /etc/os-release
uname -srm
"$PY" - <<'PY'
import importlib.metadata as md, sys, resource
print("python    ", sys.version.split()[0])
for p in ("whestbench", "flopscope", "numpy"):
    print(f"{p:<10}", md.version(p))
print("RLIMIT_AS available:", hasattr(resource, "RLIMIT_AS"))
PY
echo
echo "--- configuration ---"
echo "  memory_limit_mb (ContestSpec default) : 8192   = 8 GiB"
echo "  provocation request                   : ${ARC_WHEST_MEM_REQUEST} bytes = 9 GiB (NEVER touched)"
echo "  control request                       : ${ARC_WHEST_MEM_CONTROL} bytes = 256 MiB (must succeed)"
echo "  runner                                : subprocess (the only one that sets RLIMIT_AS)"
echo
echo "--- run ---"
"$WHEST" run --estimator "$HERE/sacrificial_memory.py" \
  --dataset hf://aicrowd/arc-whestbench-public-2026@v2-phase2 \
  --split mini --runner subprocess --n-mlps 3 --json > "$HERE/memory_gate_report.json" 2>"$HERE/memory_gate_stderr.txt"
echo "whest exit: $?"
echo
echo "--- what the worker saw (written BEFORE either mmap) ---"
cat "$ARC_WHEST_MEM_REPORT" 2>/dev/null || echo "  (no worker report written)"
echo
echo "--- per-MLP outcome ---"
"$PY" - "$HERE/memory_gate_report.json" <<'PY'
import json, sys
raw = open(sys.argv[1], encoding="utf-8").read()
i = raw.find("{")
d = json.loads(raw[i:])
if not d.get("results"):
    print("  RUN REFUSED:", json.dumps(d)[:1500]); sys.exit(0)
cfg, r = d["run_config"], d["results"]
print(f"  whestbench {d['whestbench_version']}   shape {cfg['width']}x{cfg['depth']}")
print(f"  n_failed_mlps = {r['n_failed_mlps']}   failure_breakdown = {r.get('failure_breakdown')}")
for m in r["per_mlp"]:
    flags = [k for k in ("budget_exhausted","time_exhausted",
                         "residual_wall_time_exhausted","combined_budget_exhausted")
             if m[k]]
    tb = (m["traceback"] or "").strip().splitlines()
    print(f"    MLP {m['mlp_index']}  mse={m['final_layer_mse']:.6e}  "
          f"flops={m['flops_used']:,}  flags={flags or 'none'}  "
          f"error={tb[-1][:90] if tb else 'none'}")
PY
