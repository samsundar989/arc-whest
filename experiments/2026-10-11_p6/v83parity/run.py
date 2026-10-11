"""v83 (fold): twin parity file <-> R3 fold arm on DEV-100."""
import json
import subprocess
import sys
from pathlib import Path

R = Path("C:/Users/samsu/Documents/arc-whest")
OUT = R / "experiments/2026-10-11_p6/v83parity"
K = R / "experiments/2026-10-06_mask/ktab/aggr_q50.json"
RHO = R / "experiments/2026-10-10_p5/d2/rho.json"
P = R / "experiments/2026-10-11_p6/r3/fold/params_best.npz"
tw = [sys.executable, "-u", str(R / "tools/twin/twin.py"), "eval", "--panel", "dev"]
subprocess.run(tw + ["--raw", "--cand", "estimator_k3_v83_fold.py", "--json", str(OUT / "dev_raw.json")], check=True, cwd=R)
subprocess.run(tw + ["--cand", "experiments/2026-09-27_gates/estimator_k3_v62_gb.py", "--arm", "mask+dress+bank+fold",
                     "--fold", "tau=2,mode=rec", "--bank", "star,starall",
                     "--mask", f"R=2.0,Hd=2.5,D2=2.5,rho={RHO},J=2,K={K}", "--dress", "all", "--params", str(P),
                     "--json", str(OUT / "dev_arm.json")], check=True, cwd=R)
a = json.loads((OUT / "dev_raw.json").read_text())
b = json.loads((OUT / "dev_arm.json").read_text())
rel = abs(a["raw_mse"] / b["raw_mse"] - 1)
print(f"file {a['raw_mse']:.6e} arm {b['raw_mse']:.6e} parity {rel:.2e}", "PASS" if rel <= 1e-3 else "FAIL")
