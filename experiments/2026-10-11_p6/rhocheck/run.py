"""P6.0b post-merge check: rerun the P5.2 FIT-100 --mask-stats control chain and compute_rho; must
reproduce experiments/2026-10-10_p5/d2/rho.json (else the merge lost the e_all/e_kept hook)."""
import json
import subprocess
import sys
from pathlib import Path

R = Path("C:/Users/samsu/Documents/arc-whest")
OUT = R / "experiments/2026-10-11_p6/rhocheck"
K = R / "experiments/2026-10-06_mask/ktab/aggr_q50.json"
P = R / "experiments/2026-10-08_p3/refit_starall2/params_best.npz"
J = OUT / "rho_stats_fit100.json"
subprocess.run([sys.executable, "-u", str(R / "tools/twin/twin.py"), "eval",
                "--cand", "experiments/2026-09-27_gates/estimator_k3_v62_gb.py",
                "--arm", "mask+dress+bank", "--bank", "star,starall",
                "--mask", f"R=2.0,Hd=2.5,K={K},Dz=2.5", "--mask-stats",
                "--dress", "all", "--params", str(P), "--panel", "train", "--n", "100",
                "--json", str(J)], check=True, cwd=R)
subprocess.run([sys.executable, str(R / "experiments/2026-10-10_p5/d2/compute_rho.py"), str(J)], check=True)
new = json.loads((OUT / "rho.json").read_text())["rho"]
old = json.loads((R / "experiments/2026-10-10_p5/d2/rho.json").read_text())["rho"]
d = max(abs(a - b) for a, b in zip(new, old))
print(f"max |rho_new - rho_old| = {d:.3e}", "PASS" if d < 1e-6 and max(new) > 1.0 else "FAIL")
