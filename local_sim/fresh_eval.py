"""
Run the SHIPPED estimator file over fresh minted MLPs (CPU, through the shims).

`robustness.py --dump` writes fresh MLPs with high-sample MC truth; this runs the
actual submission file over them, so the fresh-re-run readiness claim rests on the
file the grader executes rather than on the GPU mirror of it.

Run: uv run python local_sim/fresh_eval.py --data <fresh.npz> [--estimator estimator.py]
"""

import argparse
import importlib.util
import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_HERE, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

BUDGET = 272_000_000_000


class MLP:
    def __init__(self, w, seed):
        self.weights = [np.asarray(x, dtype=np.float32) for x in w]
        self.width = self.weights[0].shape[1]
        self.seed = int(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--estimator", default="estimator.py")
    ap.add_argument("--mlps", type=int, default=0)
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("shipped", os.path.join(_ROOT, args.estimator))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["shipped"] = mod
    spec.loader.exec_module(mod)
    est = mod.Estimator()

    d = np.load(args.data)
    W, T, S = d["weights"], d["truth"], d["seeds"]
    n = len(W) if args.mlps == 0 else min(args.mlps, len(W))

    class Ctx:
        width, depth, flop_budget, seed = int(W.shape[2]), int(W.shape[1]), BUDGET, 0

    est.setup(Ctx())

    mses, t0, bad = [], time.perf_counter(), 0
    for i in range(n):
        m = MLP(W[i], S[i])
        pred = np.asarray(est.predict(m, BUDGET), dtype=np.float64)
        if not np.isfinite(pred).all():
            bad += 1
            continue
        mses.append(float(((pred[-1] - T[i]) ** 2).mean()))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n}  {time.perf_counter() - t0:.0f}s", flush=True)

    a = np.sort(np.array(mses))
    print(f"\nSHIPPED FILE ({args.estimator}) on {len(a)} fresh MLPs "
          f"({bad} non-finite)")
    print(f"  final-layer MSE   mean {a.mean():.4e}   median {np.median(a):.4e}")
    for q in (0.9, 0.99, 1.0):
        print(f"  {'p' + str(int(q * 100)):>16} {a[min(len(a) - 1, int(math.ceil(q * len(a)) - 1))]:.4e}")
    print(f"  max/median        {a[-1] / np.median(a):.2f}x")


if __name__ == "__main__":
    main()
