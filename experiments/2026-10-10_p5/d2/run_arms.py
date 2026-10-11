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
    "j2": f"{BASE},Dz=2.5,J=2",
    "v82b": f"{BASE},D2=2.5,rho={RHO},J=2",
}
py = sys.executable
PANEL = sys.argv[2] if len(sys.argv) > 2 else "hold100"
for name in sys.argv[1].split(","):
    if name.startswith("raw:"):          # raw:<estimator file> -- the file itself in the twin
        f = name[4:]
        subprocess.run([py, "-u", str(R / "tools/twin/twin.py"), "eval", "--raw", "--cand", f,
                        "--panel", PANEL, "--json", str(OUT / f"{PANEL}_raw_{Path(f).stem}.json")],
                       check=True)
        continue
    cmd = [py, "-u", str(R / "tools/twin/twin.py"), "eval",
           "--cand", "experiments/2026-09-27_gates/estimator_k3_v62_gb.py",
           "--arm", "mask+dress+bank", "--bank", "star,starall", "--mask", ARMS[name],
           "--dress", "all", "--params", str(P), "--panel", PANEL,
           "--json", str(OUT / (f"val_{name}.json" if PANEL == "hold100" else f"{PANEL}_{name}.json"))]
    print(name, flush=True)
    subprocess.run(cmd, check=True)
