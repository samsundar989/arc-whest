"""P2.1 unrefit screen: v77-equivalent twin (v62_gb + mask R2.0/Hd2.5/Dz2.5 q50) vs dressed legs.

    .venv-gpu/Scripts/python.exe experiments/2026-10-07_p2/dress_screen.py --n 100 \
        --configs off,all,c2=1,d21=1,d22=1 --out experiments/2026-10-07_p2/dress_unrefit.json

One twin process, one module (arm mask+dress); `_DRESS` is switched per config, so every
config sees the same nets, params and mask. Writes per-net final MSEs and paired ratios vs
`off` (which must equal the plain mask arm: the dress arm is an exact identity at None).
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

CAND = ROOT / "experiments" / "2026-09-27_gates" / "estimator_k3_v62_gb.py"
INIT = ROOT / "experiments" / "2026-09-27_gates" / "G1bg" / "params_best.npz"
MASK = ("R=2.0,Hd=2.5,Dz=2.5,K=" + str(ROOT / "experiments" / "2026-10-06_mask" / "ktab"
                                       / "aggr_q50.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--panel", default="hold100")
    ap.add_argument("--configs", default="off,all")
    ap.add_argument("--params", default=str(INIT))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tw = twin.Twin(CAND, arm="mask+dress")
    params = twin.load_params(a.params)
    tw.install(params)
    res = {"cand": str(CAND), "params": a.params, "mask": MASK, "panel": a.panel, "configs": {}}
    cfgs = a.configs.split(";") if ";" in a.configs else a.configs.split(",")
    # "c2=1" style single-key configs come through the comma split as their own items
    prog = os.environ.get("WORKERQ_PROGRESS")
    for ci, c in enumerate(cfgs):
        ns = argparse.Namespace(panel=a.panel, n=a.n, cand=str(CAND), params=None, subst="",
                                mask=MASK, arm="mask+dress", dress="" if c == "off" else c)
        tw.mod._DRESS = None
        r = twin.cmd_eval(ns, tw=tw, quiet=True)
        res["configs"][c] = {"raw_mse": r["raw_mse"], "per_net": r["per_net"],
                             "dress": r["dress"]}
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
