"""What tau grid does the adaptive-age sweep need? Measure the energy ratios.

    python tools/adage_calib.py [--n 3]

`estimator_k3_v60_adage.py` folds the oldest exact lineage when its carried Gram
energy drops below `tau` x the youngest's. Choosing a tau grid blind is how a
sweep comes back with every arm identical: if every ratio is 1e-4 and the grid is
[0.5, 1, 2], every arm folds at the floor and the sweep measures nothing.

So this records the ratio `e_old / e_new` at every layer where the gate is live,
with the tolerance OFF (so the chain is exactly v55 and the ratios are the ones
v55 itself would have seen), and prints the quantiles the grid should straddle.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "local_sim"))
BACKEND = os.environ.get("WHEST_BACKEND", "cupy").lower()
if BACKEND == "cupy":
    sys.path.insert(0, str(ROOT / "tools" / "gpu_shim"))
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "tools"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import numpy as np  # noqa: E402

BUDGET = 2 ** 41


class _MLP:
    __slots__ = ("weights", "width", "depth", "seed")

    def __init__(self, weights, width, depth, seed):
        self.weights, self.width, self.depth, self.seed = weights, width, depth, seed


class _Ctx:
    def __init__(self, width, depth, budget, seed):
        self.width, self.depth = width, depth
        self.flop_budget, self.seed = budget, seed
        self.api_version, self.scratch_dir, self.submission_dir = "1.0", None, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--candidate", default="estimator_k3_v60_adage.py")
    a = ap.parse_args()

    xp = None
    if BACKEND == "cupy":
        import cupy as xp
        xp.get_default_memory_pool().set_limit(size=12 * 2 ** 30)

    import panels
    import real_data as rd
    import importlib.util
    spec = importlib.util.spec_from_file_location("_adc", ROOT / a.candidate)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_adc"] = mod
    spec.loader.exec_module(mod)

    rows = []
    orig = mod._adapt_gate

    def spy(tw, LA, LP, n, c):
        ex = tw.k - tw.ko
        # Record only in the band the tolerance would actually decide in.
        if int(mod._ADAPT_FLOOR) <= ex < int(mod._ADAPT_CEIL):
            eo = float(((LA[:, c:c + n] * LA[:, c:c + n]).sum()
                        + (LP[:, c:c + n] * LP[:, c:c + n]).sum()))
            en = float(((LA[:, -n:] * LA[:, -n:]).sum()
                        + (LP[:, -n:] * LP[:, -n:]).sum()))
            rows.append((ex, eo, en, eo / en if en else np.nan))
        return orig(tw, LA, LP, n, c)

    mod._adapt_gate = spy
    est = mod.Estimator()
    for k, (split, idx, seed, _nm) in enumerate(panels.nets("dev")[: a.n]):
        w, _tr, _ = rd.get(idx, split, dtype=np.float32)
        depth, width = w.shape[0], w.shape[1]
        wf = []
        for x in w:
            arr = np.asarray(x, dtype=np.float64)
            arr.setflags(write=False)
            wf.append(xp.asarray(arr) if xp is not None else arr)
        est.setup(_Ctx(width, depth, BUDGET, 0))
        est.predict(_MLP(wf, width, depth, int(seed)), BUDGET)
        print(f"[calib] net {k} done, {len(rows)} live-band observations so far",
              flush=True)
        if xp is not None:
            xp.get_default_memory_pool().free_all_blocks()

    if not rows:
        print("NO observations in the [floor, ceil) band -- with _ADAPT_TAU=None "
              "the chain folds at _SHARED_AGE and the band is never reached with "
              "a decision pending. Widen _ADAPT_FLOOR/_ADAPT_CEIL or read the "
              "ratios at the fixed age instead.")
        return 1
    r = np.array([x[3] for x in rows], float)
    ex = np.array([x[0] for x in rows], int)
    print(f"\n{len(r)} observations, exact-lineage counts seen: "
          f"{sorted(set(ex.tolist()))}")
    print("e_old / e_new quantiles:")
    for q in (0, 5, 25, 50, 75, 95, 100):
        print(f"  p{q:<3d} {np.percentile(r, q):.6e}")
    print("\nby exact-lineage count:")
    for e in sorted(set(ex.tolist())):
        m = ex == e
        print(f"  ex={e}  n={m.sum():4d}  median {np.median(r[m]):.4e}  "
              f"p5 {np.percentile(r[m],5):.4e}  p95 {np.percentile(r[m],95):.4e}")
    print("\nA tau at the pXX of this distribution folds ~XX% of the time, so a "
          "grid straddling p25..p95 is what moves the mean age.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
