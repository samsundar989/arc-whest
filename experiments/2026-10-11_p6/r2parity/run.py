"""P6.0d: twin parity file <-> arm on DEV-100 for v82c_r2 (arm = R2's v82b arm with R2's params_best)."""
import json
import subprocess
import sys
from pathlib import Path

R = Path("C:/Users/samsu/Documents/arc-whest")
OUT = R / "experiments/2026-10-11_p6/r2parity"
K = R / "experiments/2026-10-06_mask/ktab/aggr_q50.json"
RHO = R / "experiments/2026-10-10_p5/d2/rho.json"
P = R / "experiments/2026-10-10_p5/r2/v82b/params_best.npz"
tw = [sys.executable, "-u", str(R / "tools/twin/twin.py"), "eval", "--panel", "dev"]
subprocess.run(tw + ["--raw", "--cand", "estimator_k3_v82c_r2.py", "--json", str(OUT / "dev_raw_v82c_r2.json")],
               check=True, cwd=R)
subprocess.run(tw + ["--cand", "experiments/2026-09-27_gates/estimator_k3_v62_gb.py", "--arm", "mask+dress+bank",
                     "--bank", "star,starall", "--mask", f"R=2.0,Hd=2.5,K={K},D2=2.5,rho={RHO},J=2",
                     "--dress", "all", "--params", str(P), "--json", str(OUT / "dev_arm_r2.json")],
               check=True, cwd=R)
a = json.loads((OUT / "dev_raw_v82c_r2.json").read_text())
b = json.loads((OUT / "dev_arm_r2.json").read_text())
rel = abs(a["raw_mse"] / b["raw_mse"] - 1)
pn = max(abs(x / y - 1) for x, y in zip(a["per_net"], b["per_net"]))
print(f"file {a['raw_mse']:.6e} arm {b['raw_mse']:.6e} parity {rel:.2e} max per-net {pn:.2e}",
      "PASS" if rel <= 1e-3 else "FAIL")
