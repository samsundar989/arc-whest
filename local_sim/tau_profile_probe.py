"""
Sprint 2 item 3: does the pruning threshold want to vary with depth?

A single global tau = -2.5 was a sharp optimum (1.47x; -3.0 gives 1.38x, -2.0
gives 1.23x). Sharp global optima often hide mild structure -- early layers carry
the signal everything downstream depends on, so they may deserve a more
conservative threshold than late ones, or vice versa.

We test tau_k = a + b * k/(depth-1) over a small grid, plus a two-block variant,
scoring on V*c/B over the FULL 100-MLP mini as usual.

Run: uv run python local_sim/tau_profile_probe.py [--mlps 100]
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_HERE, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import real_data as rd  # noqa: E402
from prune_probe import covprop_all, matmul_cost  # noqa: E402
from rotation_probe import sobol_normals  # noqa: E402

F32 = np.float32
BUDGET = 2.72e11
FIXED = 3.27e9 + 1.04e9 + 3.0e8  # closure + Jacobian product + eigh
PPF_SOBOL = 256 * 168.0 + 30 * 3 * 256


def forward(x, w, mus, taus):
    n = w[0].shape[1]
    depth = len(w)
    N = len(x)
    cost = 0
    h = x
    prev = np.ones(n, dtype=bool)
    for k in range(depth):
        alive = (taus[k] is None) and np.ones(n, dtype=bool) or None
        if taus[k] is None:
            alive = np.ones(n, dtype=bool)
        else:
            alive = taus[k]
        wk = np.asarray(w[k], dtype=F32)
        z = h @ wk[np.ix_(prev, alive)]
        cost += matmul_cost(N, int(prev.sum()), int(alive.sum()))
        dead = ~prev
        if dead.any():
            z = z + (mus[k - 1][dead] @ wk[np.ix_(dead, alive)].astype(np.float64)).astype(F32)
        h = np.maximum(z, F32(0))
        prev = alive
    out = mus[-1].copy()
    out[prev] = h.mean(axis=0, dtype=np.float64)
    return out, cost


def build_masks(alphas, depth, n, a, b):
    """tau_k = a + b * k/(depth-1); last layer never pruned."""
    masks = []
    for k in range(depth):
        if k == depth - 1:
            masks.append(np.ones(n, dtype=bool))
        else:
            tau = a + b * (k / max(depth - 2, 1))
            masks.append(alphas[k] > tau)
    return masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlps", type=int, default=100)
    ap.add_argument("--split", default="mini")
    ap.add_argument("--n", type=int, default=16384)
    args = ap.parse_args()

    # (a, b): a = tau at layer 1, b = drift across depth
    grid = [
        (-2.5, 0.0),   # the incumbent global threshold
        (-2.5, +0.5), (-2.5, -0.5),
        (-2.75, +0.5), (-2.25, -0.5),
        (-3.0, +1.0), (-2.0, -1.0),
        (-2.75, 0.0), (-2.25, 0.0),
    ]
    res = {g: {"mse": [], "cost": []} for g in grid}

    n_mlps = min(args.mlps, rd.n_mlps(args.split))
    for i in range(n_mlps):
        w, truth, _ = rd.get(i, split=args.split, dtype=np.float64)
        mlp = rd.as_mlp(i, split=args.split)
        alphas, mus = covprop_all(w)
        x = sobol_normals(args.n, 256, int(mlp.seed) & 0x7FFFFFFF)
        t = truth[-1]
        depth, n = len(w), w[0].shape[1]
        for g in grid:
            masks = build_masks(alphas, depth, n, *g)
            est, c = forward(x, w, mus, masks)
            res[g]["mse"].append(float(((est - t) ** 2).mean()))
            res[g]["cost"].append(c)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{n_mlps}", flush=True)

    print(f"\n{n_mlps} MLPs, N={args.n}   tau_k = a + b*k/(depth-2)\n")
    print(f"{'a':>7} {'b':>6} {'tau@1':>7} {'tau@31':>7} {'MSE':>12} {'cost/sample':>13} {'V*c/B':>11} {'gain':>7}")
    print("-" * 80)
    base = None
    rows = []
    for g in grid:
        m = float(np.mean(res[g]["mse"]))
        tot = float(np.mean(res[g]["cost"])) + args.n * PPF_SOBOL + FIXED
        adj = m * tot / BUDGET
        if base is None:
            base = adj
        rows.append((g, m, tot / args.n, adj, base / adj))
    for (a, b), m, cps, adj, gain in rows:
        print(f"{a:>7.2f} {b:>6.2f} {a:>7.2f} {a + b:>7.2f} {m:>12.4e} {cps:>13,.0f} {adj:>11.4e} {gain:>6.3f}x")
    best = max(rows, key=lambda r: r[4])
    print(f"\nbest: a={best[0][0]} b={best[0][1]}  ->  {best[3]:.4e}  ({best[4]:.3f}x vs global -2.5)")


if __name__ == "__main__":
    main()
