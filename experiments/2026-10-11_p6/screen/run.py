"""P6.1 step 2: unrefit VAL-100 screen. `raw` = the price.py variant files (R2 constants) in the twin;
`arm` = the same arms through the twin mask with R2's params (arm-vs-file parity before refitting).
    python run.py raw|arm [name,...]"""
import subprocess
import sys
from pathlib import Path

R = Path("C:/Users/samsu/Documents/arc-whest")
OUT = R / "experiments/2026-10-11_p6/screen"
K = R / "experiments/2026-10-06_mask/ktab/aggr_q50.json"
RHO = R / "experiments/2026-10-10_p5/d2/rho.json"
P = R / "experiments/2026-10-10_p5/r2/v82b/params_best.npz"
M = f"R=2.0,Hd=2.5,D2=2.5,rho={RHO},J=2"
FOLD = ["--arm", "mask+dress+bank+fold", "--fold", "tau=2,mode=rec"]
NOF = ["--arm", "mask+dress+bank"]
ARMS = {
    "base": (f"{M},K={K}", NOF),
    "fold": (f"{M},K={K}", FOLD),
    "fold+Habs": (f"{M},K={OUT / 'K_habs25.json'},Hrank=abs,Hfull=1", FOLD),
    "fold+Hasym2030": (f"{M},K={OUT / 'K_asym2030.json'},Hrank=asym:2.0:3.0,Hfull=1", FOLD),
    "fold+Hasym2025": (f"{M},K={OUT / 'K_asym2025.json'},Hrank=asym:2.0:2.5,Hfull=1", FOLD),
    "fold+Habs20": (f"{M},K={OUT / 'K_habs20.json'},Hrank=abs,Hfull=1", FOLD),
}
mode = sys.argv[1]
names = sys.argv[2].split(",") if len(sys.argv) > 2 else list(ARMS)
tw = [sys.executable, "-u", str(R / "tools/twin/twin.py"), "eval", "--panel", "hold100"]
for nm in names:
    tag = nm.replace("+", "_")
    print(mode, nm, flush=True)
    if mode == "raw":
        cmd = tw + ["--raw", "--cand", str(OUT / f"variants/est_{tag}.py"), "--json", str(OUT / f"val_raw_{tag}.json")]
    else:
        mask, extra = ARMS[nm]
        cmd = tw + ["--cand", "experiments/2026-09-27_gates/estimator_k3_v62_gb.py", "--bank", "star,starall",
                    "--mask", mask, "--dress", "all", "--params", str(P), *extra,
                    "--json", str(OUT / f"val_arm_{tag}.json")]
    subprocess.run(cmd, check=True, cwd=R)
