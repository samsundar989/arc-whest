"""Lane M6.2 refits and twin parity. The recipe is P6's R4 (fold arm, R2 mask, S2tsf), but
1 epoch from R3 fold's params_best, run locally against a local same-recipe control.

    .venv-gpu/Scripts/python.exe experiments/2026-10-11_m6/refit/batch.py parity
    .venv-gpu/Scripts/python.exe experiments/2026-10-11_m6/refit/batch.py refit <arm>   (ctl|J3|age6|jsvd)

Arms: J3 = mask J=3; age6 / jsvd = the twin candidate with price.py's age(6) / join_svd patch
(cand_*.py here, built from estimator_k3_v62_gb.py). Every eval writes <tag>.json and is skipped
when that file already exists.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAIN = "C:/Users/samsu/Documents/arc-whest"
K = f"{MAIN}/experiments/2026-10-06_mask/ktab/aggr_q50.json"
RHO = f"{MAIN}/experiments/2026-10-10_p5/d2/rho.json"
P_R2 = f"{MAIN}/experiments/2026-10-10_p5/r2/v82b/params_best.npz"
P_R3 = f"{MAIN}/experiments/2026-10-11_p6/r3/fold/params_best.npz"
BASE_CAND = "experiments/2026-09-27_gates/estimator_k3_v62_gb.py"
COMMON = ["--dress", "all", "--arm", "mask+dress+bank+fold", "--fold", "tau=2,mode=rec"]


def bank(name):
    """M6.3 `acov` = the alpha-dependent covariance and variance gains (zero-init, stage S2tsfa)."""
    return ["--bank", {"acov": "star,starall,acov,avar",
                       "cfeat": "star,starall,cfeat"}.get(name, "star,starall")]
TW = [sys.executable, "-u", str(HERE / "tw.py")]


def arm(name):
    """(cand, mask) for an arm."""
    j = 3 if name == "J3" else 2
    cand = str(HERE / f"cand_{name}.py") if name in ("age6", "jsvd") else BASE_CAND
    return cand, f"R=2.0,Hd=2.5,D2=2.5,rho={RHO},J={j},K={K}"


def ev(tag, name, params, panel="hold100", n=None):
    out = HERE / "val" / f"{tag}.json"
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    cand, mask = arm(name)
    print(">>", tag, flush=True)
    r = subprocess.run(TW + ["eval", "--cand", cand, "--mask", mask, "--params", params,
                             "--panel", panel, *(["--n", str(n)] if n else []), *COMMON,
                             *bank(name), "--json", str(out)])
    if r.returncode:
        print(f"!! {tag} failed ({r.returncode})", flush=True)


def parity():
    # (1) arm parity: R2 params, to set beside the R2 price.py files
    #     (../screen/val_raw_fold_*.json, and P6's val_raw_fold.json for ctl)
    for nm in ("ctl", "J3", "age6", "jsvd"):
        ev(f"r2_{nm}", nm, P_R2)
    # (2) the unrefit starting point of every refit: R3 fold params
    for nm in ("ctl", "J3", "age6", "jsvd"):
        ev(f"r3_{nm}", nm, P_R3)


def refit(name):
    cand, mask = arm(name)
    cmd = TW + ["train", "--cand", cand, "--mask", mask, *COMMON, *bank(name), "--init", P_R3,
                "--epochs", "1",
                "--accum", "8", "--lr", "1e-4", "--eval-every", "28", "--patience", "99",
                "--hold-panel", "hold100", "--stage", {"acov": "S2tsfa", "cfeat": "S2tsfc"}.get(name, "S2tsf"),
                # zero-init gains step at lr x zero_scale (1e-6) unless multiplied (M5 froze so)
                *({"acov": ["--lr-mult", "gain/ab=1000,gain/va=1000"],
                   "cfeat": ["--lr-mult", "gain/cf=1"]}.get(name, [])), "--out", str(HERE / f"refit_{name}")]
    sys.exit(subprocess.run(cmd).returncode)


def inert():
    """acov,avar with zero gains must reproduce r3_ctl (first 10 VAL nets) exactly."""
    ev("r3_acov_inert10", "acov", P_R3, n=10)
    ev("r3_ctl10", "ctl", P_R3, n=10)
    ev("r3_cfeat_inert10", "cfeat", P_R3, n=10)


def fin(*names):
    """Per-net VAL-100 of each finished refit's params_best (paired comparison vs ctl)."""
    for nm in names:
        ev(f"fin_{nm}", nm, str(HERE / f"refit_{nm}" / "params_best.npz"))


if __name__ == "__main__":
    {"parity": parity, "inert": inert,
     "ev": lambda: ev(sys.argv[2], sys.argv[3], {"r2": P_R2, "r3": P_R3}[sys.argv[4]]), "fin": lambda: fin(*sys.argv[2].split(",")), "refit": lambda: refit(sys.argv[2])}[sys.argv[1]]()
