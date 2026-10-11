"""Lane M5 batches (replaces the .sh scripts: worker-q's `bash` cannot see Windows paths).

    .venv-gpu/Scripts/python.exe experiments/2026-10-10_m5/batch.py <batch> [args]

batches: smoke | val | refit <tag> <stage> [twin args...]
Each eval writes <out>/<tag>.json and is skipped if that file exists.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAIN = "C:/Users/samsu/Documents/arc-whest"
OUT = "C:/Users/samsu/Documents/arc-whest-lane-m/experiments/2026-10-10_m5"
KT = f"{MAIN}/experiments/2026-10-06_mask/ktab/aggr_q50.json"
KRD = f"{OUT}/ktab_RD.json"
P = f"{MAIN}/experiments/2026-10-08_p3/refit_starall2/params_best.npz"
C = ["--cand", "experiments/2026-09-27_gates/estimator_k3_v62_gb.py", "--bank", "star,starall",
     "--dress", "all"]
MASK = ["--mask", f"R=2.0,Hd=2.5,Dz=2.5,K={KT}"]
TW = [sys.executable, "-u", str(HERE / "tw.py")]
ARM = "mask+dress+bank"


def ev(d, tag, *args, params=P, panel=("--panel", "hold100")):
    out = Path(d) / f"{tag}.json"
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = TW + ["eval"] + C + ["--params", params, *panel] + list(args) + ["--json", str(out)]
    print(">>", tag, " ".join(args), flush=True)
    r = subprocess.run(cmd)
    if r.returncode:
        print(f"!! {tag} failed ({r.returncode})", flush=True)


def smoke():
    d = f"{OUT}/smoke"
    n = ("--panel", "hold100", "--n", "3")
    ev(d, "base", *MASK, "--arm", ARM, panel=n)
    ev(d, "inert", *MASK, "--arm", ARM + "+fold+rad", panel=n)
    ev(d, "fold0", *MASK, "--arm", ARM + "+fold", "--fold", "scale=0", panel=n)
    ev(d, "rad0", *MASK, "--arm", ARM + "+rad", "--rad", "all,scale=0", panel=n)
    ev(d, "fold_ns", *MASK, "--arm", ARM + "+fold", "--fold", "tau=2,mode=ns,stats=1", panel=n)
    ev(d, "fold_st", *MASK, "--arm", ARM + "+fold", "--fold", "tau=2,mode=st", panel=n)
    ev(d, "fold_rec", *MASK, "--arm", ARM + "+fold", "--fold", "tau=2,mode=rec", panel=n)
    ev(d, "rad", *MASK, "--arm", ARM + "+rad", "--rad", "all", panel=n)


def val():
    d = f"{OUT}/val"
    ev(d, "base", *MASK, "--arm", ARM)
    ev(d, "fold_ns", *MASK, "--arm", ARM + "+fold", "--fold", "tau=2,mode=ns,stats=1")
    ev(d, "fold_st", *MASK, "--arm", ARM + "+fold", "--fold", "tau=2,mode=st")
    ev(d, "fold_rec", *MASK, "--arm", ARM + "+fold", "--fold", "tau=2,mode=rec")
    ev(d, "rad_all", *MASK, "--arm", ARM + "+rad", "--rad", "all")
    ev(d, "rad_q", *MASK, "--arm", ARM + "+rad", "--rad", "q=1,d22=0,d4=0")
    ev(d, "rad_d22", *MASK, "--arm", ARM + "+rad", "--rad", "q=0,d22=1,d4=0")
    ev(d, "rad_d4", *MASK, "--arm", ARM + "+rad", "--rad", "q=0,d22=0,d4=1")
    # M5.4 step 0: H by threshold (the K table keeps R and D only)
    ev(d, "h_d25", "--mask", f"R=2.0,Hd=2.5,Dz=2.5,K={KRD}", "--arm", ARM)
    ev(d, "h_kh25", "--mask", f"R=2.0,Hkh=2.5,Dz=2.5,K={KRD}", "--arm", ARM)
    ev(d, "h_lit25", "--mask", f"R=2.0,H=2.5,Dz=2.5,K={KRD}", "--arm", ARM)
    # hubs dropped on always-on columns (A_L kept): what the always-on route-(i)+(ii) hubs are worth
    ev(d, "h_k25", "--mask", f"R=2.0,Hk=2.5,Dz=2.5,K={KRD}", "--arm", ARM)


def radsweep():
    d = f"{OUT}/val"
    for sc in ("0.1", "0.3", "0.5"):
        ev(d, f"rad_q_s{sc}", *MASK, "--arm", ARM + "+rad", "--rad", f"q=1,d22=0,d4=0,scale={sc}")
        ev(d, f"rad_d4_s{sc}", *MASK, "--arm", ARM + "+rad", "--rad", f"q=0,d22=0,d4=1,scale={sc}")
        ev(d, f"rad_d22_s{sc}", *MASK, "--arm", ARM + "+rad", "--rad", f"q=0,d22=1,d4=0,scale={sc}")


def parity(f, tag, *armargs):
    """DEV-100: the built file (--raw) vs the twin arm (same g*), and the twin base."""
    d = f"{OUT}/parity"
    dev = ("--panel", "dev")
    out = Path(d) / f"{tag}_file.json"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(TW + ["eval", "--raw", "--cand", f, "--panel", "dev", "--json", str(out)])
    ev(d, f"{tag}_arm", *MASK, *armargs, panel=dev)
    ev(d, "base_arm", *MASK, "--arm", ARM, panel=dev)


def refit(tag, stage, *extra):
    """1 epoch TRAIN-900, lr 1e-4, cosine, from refit_starall2 (v81d's params)."""
    if "--arm" not in extra:
        extra = ("--arm", ARM) + extra
    cmd = TW + ["train"] + C + MASK + ["--init", P, "--epochs", "1", "--accum", "8", "--lr", "1e-4",
                                     "--eval-every", "28", "--patience", "3", "--hold-panel",
                                     "hold100", "--stage", stage, "--out", f"{OUT}/refit_{tag}",
                                     *extra]
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    b = sys.argv[1]
    {"smoke": smoke, "val": val, "radsweep": radsweep,
     "parity": lambda: parity(*sys.argv[2:])}.get(b, lambda: refit(*sys.argv[2:]))()
