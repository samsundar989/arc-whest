"""
Sprint 2 item 2: can a rank-r factor model replace the full covariance closure
as the pruning scaffold?

The closure exists in v2 only to produce alpha = mu_pre/sigma (which neurons are
dead) and mu (the analytic patch for the ones we drop). It costs 4.3e9 FLOPs and,
more importantly, ~832 of the estimator's 1794 ops per MLP, and op count is what
drives residual wall time.

A factor model tracks  Cov ~ F F^T + diag(d)  with F of shape (n, r):
    mu_pre  = W^T mu
    var_pre = rowsum((W^T F)^2) + (W*W)^T d        <- diagonal only, which is all
                                                      alpha needs
    F_new   = Phi * (W^T F)
    d_new   = var - rowsum(F_new^2)
costing ~n^2*r per layer instead of 2*n^3, i.e. 8x less at r=64, with roughly a
third the ops.

HARD GATE (from the plan): byte-identical prune sets AND <=0.5% MSE delta versus
full float64 covariance propagation, on the full 100-MLP mini. Anything less and
we keep the full closure -- a scaffold that silently reshuffles which neurons get
dropped is not a cost optimisation, it is a different estimator.

Run: uv run python local_sim/rank_scaffold_probe.py [--ranks 32,64,128]
"""

import argparse
import os
import sys

import numpy as np
from scipy.special import ndtr

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_HERE, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import real_data as rd  # noqa: E402
from prune_probe import covprop_all, matmul_cost  # noqa: E402
from rotation_probe import sobol_normals  # noqa: E402

F32 = np.float32
TAU = -2.5
BUDGET = 2.72e11


def _npdf(x):
    return (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x * x)


def rank_scaffold(w, r):
    """alpha and mu per layer from a rank-r + diagonal factor model."""
    n = w[0].shape[1]
    mu = np.zeros(n)
    F = np.zeros((n, r))
    d = np.ones(n)  # input covariance = I  ->  F = 0, d = 1
    A, M = [], []
    cost = 0
    for wl in w:
        wl = np.asarray(wl, dtype=np.float64)
        mu_pre = wl.T @ mu
        WF = wl.T @ F
        cost += matmul_cost(n, n, r) + matmul_cost(1, n, n)
        var_pre = np.maximum((WF**2).sum(axis=1) + (wl * wl).T @ d, 1e-12)
        cost += 2 * n * r + matmul_cost(1, n, n) + n * n
        sig = np.sqrt(var_pre)
        a = mu_pre / sig
        Phi, phi = ndtr(a), _npdf(a)
        mu = mu_pre * Phi + sig * phi
        ez2 = (mu_pre**2 + var_pre) * Phi + mu_pre * sig * phi
        var = np.maximum(ez2 - mu * mu, 0.0)
        F = Phi[:, None] * WF
        d = np.maximum(var - (F**2).sum(axis=1), 0.0)
        cost += 3 * n * r
        A.append(a.copy())
        M.append(mu.copy())
    return np.stack(A), np.stack(M), cost


def forward_pruned(x, w, mus, alphas, tau):
    n = w[0].shape[1]
    depth = len(w)
    N = len(x)
    cost = 0
    h = x
    prev = np.ones(n, dtype=bool)
    for k in range(depth):
        alive = (alphas[k] > tau) if k < depth - 1 else np.ones(n, dtype=bool)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlps", type=int, default=100)
    ap.add_argument("--split", default="mini")
    ap.add_argument("--n", type=int, default=16384)
    ap.add_argument("--ranks", default="32,64,128")
    args = ap.parse_args()
    ranks = [int(v) for v in args.ranks.split(",")]

    n_mlps = min(args.mlps, rd.n_mlps(args.split))
    base_mse, base_cost = [], []
    res = {r: {"mse": [], "cost": [], "ident": [], "flip": []} for r in ranks}

    for i in range(n_mlps):
        w, truth, _ = rd.get(i, split=args.split, dtype=np.float64)
        mlp = rd.as_mlp(i, split=args.split)
        x = sobol_normals(args.n, 256, int(mlp.seed) & 0x7FFFFFFF)
        t = truth[-1]

        A_full, M_full = covprop_all(w)
        est, c = forward_pruned(x, w, M_full, A_full, TAU)
        base_mse.append(float(((est - t) ** 2).mean()))
        base_cost.append(c + 3.27e9)
        keep_full = np.stack([A_full[k] > TAU for k in range(31)])

        for r in ranks:
            A_r, M_r, sc = rank_scaffold(w, r)
            est_r, c_r = forward_pruned(x, w, M_r, A_r, TAU)
            res[r]["mse"].append(float(((est_r - t) ** 2).mean()))
            res[r]["cost"].append(c_r + sc)
            keep_r = np.stack([A_r[k] > TAU for k in range(31)])
            res[r]["ident"].append(bool(np.array_equal(keep_full, keep_r)))
            res[r]["flip"].append(int((keep_full != keep_r).sum()))
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{n_mlps}", flush=True)

    bm, bc = float(np.mean(base_mse)), float(np.mean(base_cost))
    print(f"\n{n_mlps} MLPs, tau={TAU}, N={args.n}")
    print(f"full f64 closure : MSE {bm:.4e}  cost/MLP {bc:,.0f}\n")
    print(f"{'rank':>6} {'MSE':>12} {'dMSE':>8} {'cost/MLP':>14} {'saved':>8} {'identical':>10} {'flips/MLP':>10}")
    print("-" * 76)
    for r in ranks:
        m = float(np.mean(res[r]["mse"]))
        c = float(np.mean(res[r]["cost"]))
        ident = sum(res[r]["ident"])
        print(
            f"{r:>6} {m:>12.4e} {100 * (m / bm - 1):>+7.2f}% {c:>14,.0f} "
            f"{100 * (1 - c / bc):>7.1f}% {ident:>4}/{n_mlps:<5} {float(np.mean(res[r]['flip'])):>10.1f}"
        )
    print("\nHARD GATE: byte-identical prune sets on ALL MLPs and |dMSE| <= 0.5%")
    for r in ranks:
        ok = sum(res[r]["ident"]) == n_mlps and abs(float(np.mean(res[r]["mse"])) / bm - 1) <= 0.005
        print(f"  rank {r:>4}: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
