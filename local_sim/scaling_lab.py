"""
Error-vs-samples slope for each sampling scheme — the datum that picks our
operating point on the FLOP budget.

Score = MSE * max(0.1, C/B) and C is linear in N, so with MSE ~ N^-alpha:
  * alpha = 1 (plain MC): score is FLAT for every N above the 10% floor —
    spend exactly to the floor, no more.
  * alpha > 1 (working QMC): score ~ N^(1-alpha) above the floor — spend
    everything.
Measuring alpha therefore decides how many samples the submission should take.

Run: uv run python local_sim/scaling_lab.py [--reps 8] [--mlps 3]
"""

import argparse
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])

import real_data as rd
from variance_lab import METHODS, final_mean

COST_PER_SAMPLE = 32 * 2 * 256 * 256 + 32 * 256
BUDGET = 2.72e11


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--mlps", type=int, default=3)
    ap.add_argument("--methods", type=str, default="mc,sobol,radial+anti")
    ap.add_argument("--ns", type=str, default="2048,8192,32768")
    args = ap.parse_args()

    ns = [int(v) for v in args.ns.split(",")]
    names = [m for m in args.methods.split(",") if m in METHODS]
    d = 256

    print(f"{'N':>7} {'C/B':>7} {'mult':>6} " + " ".join(f"{m:>22}" for m in names))
    print("-" * (23 + 23 * len(names)))
    table = {m: [] for m in names}
    for n in ns:
        cb = n * COST_PER_SAMPLE / BUDGET
        mult = max(0.1, cb)
        cells = []
        for m in names:
            fn = METHODS[m]
            mses = []
            t0 = time.perf_counter()
            for mi in range(args.mlps):
                w, truth, _ = rd.get(mi)
                t = truth[-1]
                for r in range(args.reps):
                    rng = np.random.default_rng(50_000 + 1000 * mi + r)
                    mses.append(float(((final_mean(fn(n, d, rng), w) - t) ** 2).mean()))
            mse = float(np.mean(mses))
            table[m].append((n, mse, mse * mult))
            cells.append(f"{mse:.3e}/{mse * mult:.2e}")
            del t0
        print(f"{n:>7} {cb:>7.3f} {mult:>6.3f} " + " ".join(f"{c:>22}" for c in cells), flush=True)

    print("\nfitted slope  MSE ~ N^-alpha   (alpha>1 ⇒ spend the whole budget)")
    print("-" * 66)
    for m in names:
        n_ = np.log(np.array([r[0] for r in table[m]]))
        y_ = np.log(np.array([r[1] for r in table[m]]))
        alpha = -np.polyfit(n_, y_, 1)[0]
        # extrapolate the adjusted score to the full-budget operating point
        n_full = int(BUDGET / COST_PER_SAMPLE)
        c0 = table[m][-1][1] * table[m][-1][0] ** alpha
        proj_full = c0 * n_full**-alpha * 1.0
        proj_floor = c0 * (0.1 * BUDGET / COST_PER_SAMPLE) ** -alpha * 0.1
        print(
            f"{m:<14} alpha={alpha:5.2f}   projected adjusted: "
            f"floor(N={int(0.1 * BUDGET / COST_PER_SAMPLE)})={proj_floor:.3e}   "
            f"full(N={n_full})={proj_full:.3e}"
        )


if __name__ == "__main__":
    main()
