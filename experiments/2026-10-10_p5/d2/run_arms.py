"""P5.2 step 1 + P5.1 item 6: twin arms, VAL-100 unrefit, on v81d's twin arm.
Run under workerq with the GPU venv: python run_arms.py <arm,arm,...>"""
import subprocess
import sys
from pathlib import Path

R = Path("C:/Users/samsu/Documents/arc-whest")
OUT = R / "experiments/2026-10-10_p5/d2"
K = R / "experiments/2026-10-06_mask/ktab/aggr_q50.json"
P = R / "experiments/2026-10-08_p3/refit_starall2/params_best.npz"
RHO = OUT / "rho.json"
BASE = f"R=2.0,Hd=2.5,K={K}"
ARMS = {
    "ctl": f"{BASE},Dz=2.5",
    "d2": f"{BASE},D2=2.5",
    "d2rho": f"{BASE},D2=2.5,rho={RHO}",
    "d2rholeg": f"{BASE},D2=2.5,rho={RHO},Dleg=1",
    "d2c": f"{BASE},D2c=2.5",
    "dzleg": f"{BASE},Dz=2.5,Dleg=1",
    "cov3": f"{BASE},Dz=2.5,C=3.0",
    "cov4": f"{BASE},Dz=2.5,C=4.0",
}
py = sys.executable
for name in sys.argv[1].split(","):
    cmd = [py, "-u", str(R / "tools/twin/twin.py"), "eval",
           "--cand", "experiments/2026-09-27_gates/estimator_k3_v62_gb.py",
           "--arm", "mask+dress+bank", "--bank", "star,starall", "--mask", ARMS[name],
           "--dress", "all", "--params", str(P), "--panel", "hold100",
           "--json", str(OUT / f"val_{name}.json")]
    print(name, flush=True)
    subprocess.run(cmd, check=True)
