"""
Is alpha the right thing to prune on?

alpha = mu_pre/sigma says how OFTEN a neuron fires, not how much the final layer
CARES. A rarely-firing neuron feeding a strongly-amplified path may matter more
than a frequently-firing one whose signal dies downstream. The natural criterion
is the neuron's contribution to the output's variance:

    c_{k,i} = Var(h_{k,i}) * || B_k[i, :] ||^2 ,
    B_k = W_{k+1} diag(Phi_{k+1}) ... W_L      (sensitivity of z_L to h_k)

B_k comes from a backward product costing 31 matmuls (~1.0e9 FLOPs in f32, 0.4%
of budget) — the mirror of the forward product we already run for the rotation.

We compare, at matched COST (same fraction of neurons kept), the MSE from:
  alpha    threshold on the firing index               (what v2.0 ships)
  contrib  keep the top-f neurons by c_{k,i}
  var      keep the top-f by Var(h) alone (ablation: is the sensitivity needed?)

Run: uv run python local_sim/prune_criterion_probe.py [--mlps 100]
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
from prune_probe import matmul_cost  # noqa: E402
from rotation_probe import sobol_normals  # noqa: E402

F32 = np.float32
BUDGET = 2.72e11
COVPROP_COST = 3.27e9
PPF_COST = 168.0


def closure_full(w):
    """Per-layer alpha, post-ReLU mean, post-ReLU variance, and Phi."""
    n = w[0].shape[1]
    mu = np.zeros(n)
    cov = np.eye(n)
    off = np.ones((n, n)) - np.eye(n)
    A, M, V, P = [], [], [], []
    for wl in w:
        wl = np.asarray(wl, dtype=np.float64)
        mu_pre = wl.T @ mu
        cov_pre = wl.T @ cov @ wl
        var_pre = np.maximum(np.diag(cov_pre), 1e-12)
        sig = np.sqrt(var_pre)
        a = mu_pre / sig
        Phi = ndtr(a)
        phi = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * a * a)
        mu = mu_pre * Phi + sig * phi
        ez2 = (mu_pre**2 + var_pre) * Phi + mu_pre * sig * phi
        var = np.maximum(ez2 - mu * mu, 0.0)
        cov = np.outer(Phi, Phi) * cov_pre * off + np.diag(var)
        cov = 0.5 * (cov + cov.T)
        A.append(a.copy()); M.append(mu.copy()); V.append(var.copy()); P.append(Phi.copy())
    return np.stack(A), np.stack(M), np.stack(V), np.stack(P)


def sensitivity(w, P):
    """B_k[i,:] = sensitivity of the final pre-activation to h_{k,i}."""
    depth = len(w)
    B = [None] * depth
    B[depth - 1] = np.eye(w[0].shape[1])
    cur = np.asarray(w[depth - 1], dtype=np.float64)
    B[depth - 2] = cur
    for k in range(depth - 3, -1, -1):
        cur = np.asarray(w[k + 1], dtype=np.float64) * P[k + 1][None, :] @ cur
        B[k] = cur
    return B


def forward_masked(x, w, mus, keep, depth):
    n = w[0].shape[1]
    N = len(x)
    cost = 0
    h = x
    prev = np.ones(n, dtype=bool)
    for k in range(depth):
        alive = keep[k]
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
    ap.add_argument("--fracs", default="0.70,0.75,0.80,0.85")
    args = ap.parse_args()
    fracs = [float(v) for v in args.fracs.split(",")]

    crits = ("alpha", "contrib", "var")
    res = {(c, f): {"mse": [], "cost": []} for c in crits for f in fracs}
    n_mlps = min(args.mlps, rd.n_mlps(args.split))

    for i in range(n_mlps):
        w, truth, _ = rd.get(i, split=args.split, dtype=np.float64)
        mlp = rd.as_mlp(i, split=args.split)
        A, M, V, P = closure_full(w)
        B = sensitivity(w, P)
        depth = len(w)
        n = w[0].shape[1]
        x = sobol_normals(args.n, n, int(mlp.seed) & 0x7FFFFFFF)
        t = truth[-1]

        score = {
            "alpha": A,
            "var": V,
            "contrib": np.stack([V[k] * (B[k] ** 2).sum(axis=1) for k in range(depth)]),
        }
        for c in crits:
            for f in fracs:
                # GLOBAL threshold across all prunable layers, matched to the
                # target alive fraction. A per-layer fixed fraction is wrong:
                # layer 1 has mu_pre = 0 for every neuron (the input mean is
                # zero), so alpha = 0 everywhere there and nothing is dead --
                # forcing a quota there prunes fully live neurons.
                pool = score[c][: depth - 1].ravel()
                thr = np.quantile(pool, 1.0 - f)
                keep = [score[c][k] > thr for k in range(depth - 1)]
                keep.append(np.ones(n, dtype=bool))
                est, cost = forward_masked(x, w, M, keep, depth)
                res[(c, f)]["mse"].append(float(((est - t) ** 2).mean()))
                res[(c, f)]["cost"].append(cost)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{n_mlps}", flush=True)

    print(f"\n{n_mlps} MLPs, N={args.n}   (cost includes closure + ppf + sensitivity where used)\n")
    print(f"{'criterion':>10} {'keep':>6} {'MSE':>12} {'cost/sample':>13} {'V*c/B':>11} {'vs alpha':>9}")
    print("-" * 68)
    base = {}
    for f in fracs:
        for c in crits:
            m = float(np.mean(res[(c, f)]["mse"]))
            extra = COVPROP_COST + (1.04e9 if c == "contrib" else 0.0)
            tot = float(np.mean(res[(c, f)]["cost"])) + args.n * (256 * PPF_COST + 23040) + extra
            adj = m * tot / BUDGET
            if c == "alpha":
                base[f] = adj
            print(
                f"{c:>10} {f:>6.2f} {m:>12.4e} {tot / args.n:>13,.0f} {adj:>11.4e} "
                f"{base[f] / adj:>8.2f}x"
            )
        print()


if __name__ == "__main__":
    main()
