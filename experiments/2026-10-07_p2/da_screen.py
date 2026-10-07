"""P2.4.1 screen: always-on D21 rows dropped (mask key Da=tau) on the v79-equivalent twin.

v62_gb + mask (v77's R2.0/Hd2.5/Dz2.5, q50 K table) + dress all + the P2.1 refit params.
VAL-100; per-net final MSE paired vs the base mask; per-layer alpha fractions logged (stats)
so the export's fixed per-layer counts can be read off.
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

CAND = ROOT / "experiments" / "2026-09-27_gates" / "estimator_k3_v62_gb.py"
PARAMS = MAIN / "experiments" / "2026-10-07_p2" / "refit_dress_all" / "params_best.npz"
BASE = "R=2.0,Hd=2.5,Dz=2.5,K=" + str(MAIN / "experiments" / "2026-10-06_mask" / "ktab" / "aggr_q50.json")

ap = argparse.ArgumentParser()
ap.add_argument("--configs", default=";Da=3.0;Da=2.5;Da=2.0", help="mask suffixes, ;-separated (empty = base)")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--out", default=str(MAIN / "experiments" / "2026-10-07_p2" / "da_screen.json"))
a = ap.parse_args()
tw = twin.Twin(CAND, arm="mask+dress")
tw.install(twin.load_params(PARAMS))
res = {}
prog = os.environ.get("WORKERQ_PROGRESS")
taus = a.configs.split(";")
for k, t in enumerate(taus):
    mask = BASE + (f",{t}" if t else "")
    ns = argparse.Namespace(panel="hold100", n=a.n, cand=str(CAND), params=None, subst="",
                            mask=mask, arm="mask+dress", dress="all", gm="",
                            mask_stats=(k == 0))
    r = twin.cmd_eval(ns, tw=tw, quiet=True)
    res[t or "base"] = {"raw_mse": r["raw_mse"], "per_net": r["per_net"]}
    if "stats" in r:
        # mean over nets of the per-layer always-on fractions alpha > t (dec - dead)
        L = {}
        for rows in r["stats"].values():
            for st in rows:
                for tt in (2.0, 2.5, 3.0):
                    L.setdefault((st["l"], tt), []).append(st["dec%.1f" % tt] - st["dead%.1f" % tt])
        res["on_frac"] = {f"{l}:{tt}": float(np.mean(v)) for (l, tt), v in sorted(L.items())}
    line = {"mask": t or "base", "raw_mse": r["raw_mse"]}
    if "base" in res and t:
        b, x = np.array(res["base"]["per_net"]), np.array(r["per_net"])
        d = x / b - 1
        line.update(ratio=float(x.mean() / b.mean()), wins=int((x < b).sum()),
                    paired_mean=float(d.mean()), paired_se=float(d.std(ddof=1) / len(d) ** .5))
    print(json.dumps(line), flush=True)
    Path(a.out).write_text(json.dumps(res))
    if prog:
        Path(prog).write_text(str((k + 1) / len(taus)))
