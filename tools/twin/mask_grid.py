"""P1 mask grid (HANDOFF 2026-10-06 Lane P): free-running raw MSE per decided-neuron mask.

    .venv-gpu/Scripts/python.exe -u tools/twin/mask_grid.py --out <ABS dir> [--panel hold100]

One Twin load (v73 with the `mask` arm, gains baked in -> --raw), one JSON per config
in --out, skipped when it already exists, so a preempted job resumes where it stopped.
Every config also logs the per-layer alpha kept fractions (`--mask-stats`). The
control is the SAME code with no mask; `--fit-stats` adds the unmasked FIT-100 run
the FLOP model reads its kept fractions from.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

if "--cpu" in sys.argv:          # read by twin at import: a CPU-only job beside a GPU one
    os.environ["TWIN_DEVICE"] = "cpu"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import twin  # noqa: E402  (puts the torch shim on the path)

KT = twin.ROOT / "experiments" / "2026-10-06_mask" / "ktab"

GRID = [
    ("ctl", ""),
    ("R2.0", "R=2.0"), ("R2.5", "R=2.5"),
    ("H2.5", "H=2.5"), ("H3.0", "H=3.0"), ("H3.5", "H=3.5"),
    ("Hk2.5", "Hk=2.5"), ("Hk3.0", "Hk=3.0"), ("Hk3.5", "Hk=3.5"),
    ("D2.5", "D=2.5"), ("D3.0", "D=3.0"),
    ("Dc2.5", "Dc=2.5"), ("Dc3.0", "Dc=3.0"),
    ("safe", "R=2.0,H=3.0,D=3.0"), ("safeK", "R=2.0,Hk=3.0,D=3.0"),
    ("safeKc", "R=2.0,Hk=3.0,Dc=3.0"),
    ("aggr", "R=2.0,H=2.5,D=2.5"), ("aggrK", "R=2.0,Hk=2.5,D=2.5"),
    # H decomposition (after the first grid: H ~ Hk, so e_b's carrier is not the cost)
    ("Hd2.5", "Hd=2.5"), ("Hd3.0", "Hd=3.0"), ("Hkh3.0", "Hkh=3.0"),
    ("Hw3.0", "Hw=3.0"), ("Hs3.0", "Hs=3.0"),
    # R decomposition: inherited legs (post-Wick, incl. old-tier Q) vs newborn rows
    ("Ri2.0", "Ri=2.0"), ("Rn2.0", "Rn=2.0"),
    # D on dead ROWS only (the corrector's d21e stays exact for live rows)
    ("Dr2.5", "Dr=2.5"), ("Dr3.0", "Dr=3.0"), ("Dr2.0", "Dr=2.0"),
    ("safe-r", "R=2.0,Hd=3.0,Dr=3.0"), ("aggr-r", "R=2.0,Hd=2.5,Dr=2.5"),
    ("lean-r", "R=2.5,Hd=2.5,Dr=2.5"),
    # P2 variant A: fixed per-layer top-K (median kept count on FIT-100, nearest 32)
    ("aggr-rK", "R=2.0,Hd=2.5,Dr=2.5,K=" + str(KT / "aggr-r_m32.json")),
    ("safe-rK", "R=2.0,Hd=3.0,Dr=3.0,K=" + str(KT / "safe-r_m32.json")),
    ("lean-rK", "R=2.5,Hd=2.5,Dr=2.5,K=" + str(KT / "lean-r_m32.json")),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", default=str(twin.ROOT / "estimator_k3_v73_gb.py"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--panel", default="hold100")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--only", default="", help="comma list of config names")
    ap.add_argument("--fit-stats", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="TWIN_DEVICE=cpu (set before import)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tw = twin.Twin(Path(a.cand), arm="mask", raw=True)
    jobs = [(name, spec, a.panel) for name, spec in GRID
            if not a.only or name in a.only.split(",")]
    if a.fit_stats:
        jobs.append(("fit_ctl", "", "fit100"))
    prog = os.environ.get("WORKERQ_PROGRESS")
    for j, (name, spec, pnl) in enumerate(jobs):
        dst = out / f"{name}.json"
        if dst.exists():
            continue
        t0 = time.time()
        ns = argparse.Namespace(panel=pnl, n=a.n, params="", subst="", mask=spec,
                                mask_stats=True, json=str(dst) + ".tmp")
        r = twin.cmd_eval(ns, tw=tw, quiet=True)
        Path(str(dst) + ".tmp").replace(dst)
        print(json.dumps({"config": name, "mask": spec, "panel": pnl, "n": r["n"],
                          "raw_mse": r["raw_mse"], "seconds": round(time.time() - t0, 1)}),
              flush=True)
        if prog:
            Path(prog).write_text(str((j + 1) / len(jobs)))


if __name__ == "__main__":
    main()
