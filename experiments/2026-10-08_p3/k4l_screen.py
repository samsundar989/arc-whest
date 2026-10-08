"""P3.2 unrefit screen: the v79/v80-equivalent twin (v62_gb + mask + dress, refit_dress_all params)
with the kappa4-core loop arms, VAL-100 (hold100) free-running, paired vs `off`.

    .venv-gpu/Scripts/python.exe experiments/2026-10-08_p3/k4l_screen.py --n 100 \
        --configs "off;core=1,k211=exact,rad=exact;core=1,k211=exact,rad=exact|sr=128" --out X.json

A config is a --k4l spec, optionally `|sr=<rank>` (old-tier rank). `off` = no k4l (the baseline).
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "twin"))
import numpy as np  # noqa: E402
import twin  # noqa: E402
import k4l_arm  # noqa: E402

MAIN = Path(r"C:/Users/samsu/Documents/arc-whest")
CAND = ROOT / "experiments" / "2026-09-27_gates" / "estimator_k3_v62_gb.py"
INIT = MAIN / "experiments" / "2026-10-07_p2" / "refit_dress_all" / "params_best.npz"
MASK = ("R=2.0,Hd=2.5,Dz=2.5,K=" + str(MAIN / "experiments" / "2026-10-06_mask" / "ktab"
                                       / "aggr_q50.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--panel", default="hold100")
    ap.add_argument("--configs", default="off")
    ap.add_argument("--params", default=str(INIT))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tw = twin.Twin(CAND, arm="mask+dress+k4l")
    tw.install(twin.load_params(a.params))
    base_rank = tw.mod._SHARED_RANK
    res = {"cand": str(CAND), "params": a.params, "mask": MASK, "panel": a.panel, "configs": {}}
    prog = os.environ.get("WORKERQ_PROGRESS")
    cfgs = a.configs.split(";")
    for ci, c in enumerate(cfgs):
        spec, sr = c, 0
        if "|" in c:
            spec, extra = c.split("|")
            sr = int(extra.split("=")[1])
        tw.mod._K4L = None
        tw.mod._SHARED_RANK = sr or base_rank
        ns = argparse.Namespace(panel=a.panel, n=a.n, cand=str(CAND), params=None, subst="",
                                mask=MASK, arm="mask+dress+k4l", dress="all", gm="",
                                k4l="" if spec == "off" else spec, shared_rank=0)
        r = twin.cmd_eval(ns, tw=tw, quiet=True)
        res["configs"][c] = {"raw_mse": r["raw_mse"], "per_net": r["per_net"], "k4l": r["k4l"],
                             "shared_rank": tw.mod._SHARED_RANK}
        line = {"config": c, "raw_mse": r["raw_mse"]}
        if "off" in res["configs"] and c != "off":
            b = np.array(res["configs"]["off"]["per_net"])
            x = np.array(r["per_net"])
            d = x / b - 1
            line.update(ratio=float(x.mean() / b.mean()), wins=int((x < b).sum()),
                        paired_mean=float(d.mean()), paired_se=float(d.std(ddof=1) / len(d) ** .5))
            res["configs"][c].update({k: line[k] for k in ("ratio", "wins", "paired_mean",
                                                            "paired_se")})
        print(json.dumps(line), flush=True)
        Path(a.out).write_text(json.dumps(res))
        if prog:
            Path(prog).write_text(str((ci + 1) / len(cfgs)))


if __name__ == "__main__":
    main()
