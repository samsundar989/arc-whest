"""P2.1b unrefit screen: v79_dress (raw twin of the shipped file) + the spurious-term correction.

    .venv-gpu/Scripts/python.exe experiments/2026-10-07_p2/corr_screen.py --fracs 0,0.0625,0.15,0.4,1.0

`_v79_corr.py` is v79 built with `tools/build_dress.py --corr`; `_DRESS_CORR` is switched per
config (0 = v79 exactly). VAL-100 (full 900-999), per-net final MSE, paired vs f = 0.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAIN = Path(r"C:/Users/samsu/Documents/arc-whest")
sys.path.insert(0, str(ROOT / "tools" / "twin"))
import numpy as np  # noqa: E402
import twin  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--fracs", default="0,0.15")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--out", default=str(MAIN / "experiments" / "2026-10-07_p2" / "corr_unrefit.json"))
a = ap.parse_args()
tw = twin.Twin(ROOT / "experiments" / "2026-10-07_p2" / "_v79_corr.py", raw=True)
res = {}
prog = os.environ.get("WORKERQ_PROGRESS")
fr = [float(x) for x in a.fracs.split(",")]
for k, f in enumerate(fr):
    tw.mod._DRESS_CORR = f
    ns = argparse.Namespace(panel="hold100", n=a.n, cand="", params=None, subst="")
    r = twin.cmd_eval(ns, tw=tw, quiet=True)
    res[str(f)] = {"raw_mse": r["raw_mse"], "per_net": r["per_net"]}
    line = {"f": f, "raw_mse": r["raw_mse"]}
    if "0.0" in res and f != 0.0:
        b, x = np.array(res["0.0"]["per_net"]), np.array(r["per_net"])
        d = x / b - 1
        line.update(ratio=float(x.mean() / b.mean()), wins=int((x < b).sum()),
                    paired_mean=float(d.mean()), paired_se=float(d.std(ddof=1) / len(d) ** .5))
        res[str(f)].update({kk: line[kk] for kk in ("ratio", "wins", "paired_mean", "paired_se")})
    print(json.dumps(line), flush=True)
    Path(a.out).write_text(json.dumps(res))
    if prog:
        Path(prog).write_text(str((k + 1) / len(fr)))
