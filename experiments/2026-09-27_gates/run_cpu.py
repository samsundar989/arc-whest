"""CPU worker-q job: official numpy-path identity parity (1 DEV net, parity read only) and the
real-meter cost of both identity gate files at 1024 x 16 (FIT net 0; FLOPs are net-independent).

    .venv/Scripts/python.exe experiments/2026-09-27_gates/run_cpu.py
"""
import json
import os
import subprocess
import sys

R = "C:/Users/samsu/Documents/arc-whest"
D = R + "/experiments/2026-09-27_gates"
BASE = R + "/estimator_k3_v68_wino.py"
ARMS = {a: f"{D}/estimator_k3_v70g_{a}.py" for a in ("birth", "trans")}
out = {"parity_numpy": {}, "cost": {}}
env = dict(os.environ, WHEST_BACKEND="numpy", PYTHONIOENCODING="utf-8")

for arm, f in ({} if "--cost-only" in sys.argv else ARMS).items():
    p = subprocess.run([sys.executable, "tools/_parity.py", BASE, f, "1"], cwd=R, env=env,
                       capture_output=True, text=True)
    out["parity_numpy"][arm] = {"rc": p.returncode,
                                "lines": [l for l in p.stdout.splitlines() if "[parity]" in l
                                          or "PARITY" in l]}
    print(arm, out["parity_numpy"][arm], flush=True)

for tag, f in (("v68_wino", BASE), *ARMS.items()):
    p = subprocess.run([sys.executable, "tools/_cost_worker.py", f, "fit", "0"], cwd=R, env=env,
                       capture_output=True, text=True)
    line = [l for l in p.stdout.splitlines() if l.startswith("{")]
    out["cost"][tag] = json.loads(line[-1]) if line else {"rc": p.returncode,
                                                          "stderr": p.stderr[-2000:]}
    print(tag, json.dumps(out["cost"][tag]), flush=True)

b = out["cost"].get("v68_wino", {})
for arm in ARMS:
    c = out["cost"].get(arm, {})
    if "flops" in c and "flops" in b:
        c["dflops"] = c["flops"] - b["flops"]
        c["dmult"] = c["multiplier"] - b["multiplier"]
        c["dops"] = c["ops"] - b["ops"]
with open(D + "/check_cpu.json", "w") as fh:
    json.dump(out, fh, indent=1)
print("wrote", D + "/check_cpu.json")
