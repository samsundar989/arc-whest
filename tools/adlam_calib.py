"""Calibrate REF_l for the adaptive lambda (HANDOFF 09-23 Step 4a).

    python tools/adlam_calib.py estimator_k3_v57_adlam.py --panel fit --n 100 --out <json>

Runs the candidate at `_ADAPT_BETA = 0` (so the chain is exactly the base) with
the `_ADAPT_LOG` hook on, and records `r_l = mean(dG_l) / mean(var_l)` for every
non-final layer of every net. REF_l is the panel mean. cupy, accuracy-side only.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "tools"))
import probe  # noqa: E402

import numpy as np  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("--panel", default="fit")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import cupy as cp
    import panels
    import real_data as rd
    cp.get_default_memory_pool().set_limit(size=20 * 2 ** 30)
    mod = probe.load_module(ROOT / a.candidate)
    mod._ADAPT_BETA = 0.0
    est = mod.Estimator()
    rows = panels.nets(a.panel)[:a.n]
    per = []
    for i, (split, idx, seed, _name) in enumerate(rows):
        w, _tr, _ = rd.get(idx, split, dtype=np.float32)
        depth, width = w.shape[0], w.shape[1]
        if i == 0:
            est.setup(probe._Ctx(width, depth, probe.BUDGET, 0))
        wf = [cp.asarray(np.asarray(x, dtype=np.float64)) for x in w]
        mod._ADAPT_LOG = []
        est.predict(probe._MLP(wf, width, depth, int(seed)), probe.BUDGET)
        per.append([float(v) for v in mod._ADAPT_LOG])
        mod._ADAPT_LOG = None
        cp.get_default_memory_pool().free_all_blocks()
        if i % 10 == 0:
            print(f"[calib] {i + 1}/{len(rows)} r = {np.round(per[-1], 4).tolist()}", flush=True)
    R = np.array(per)
    ref = R.mean(axis=0)
    cv = R.std(axis=0) / np.abs(ref)
    res = {"candidate": a.candidate, "panel": a.panel, "n": len(per),
           "ref": ref.tolist(), "cv": cv.tolist(), "per_net": per}
    Path(ROOT / a.out).write_text(json.dumps(res) + "\n", encoding="utf-8")
    print("REF =", [round(float(x), 6) for x in ref])
    print("CV  =", [round(float(x), 4) for x in cv])


if __name__ == "__main__":
    main()
