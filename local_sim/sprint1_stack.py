"""
Sprint 1: measure every component solo and stacked, on the FULL 100-MLP mini
split, with HONEST end-to-end cost accounting.

The score above the multiplier floor is  V*c/B  (per-sample variance x per-sample
cost), so every component is judged on that single number, never on MSE alone --
a component that cuts MSE 5% while adding 10% cost is a loss.

Cost model uses flopscope's real pricing:
  matmul (M,K)@(K,N) = (2K-1)*M*N        (shapes only -- pruning must materialise
                                          smaller dense matrices to pay)
  stats.norm.ppf     = 168 / element
  elementwise        = 1 / element (f32 rate 1.0; f16/int8 are ALSO 1.0, so there
                                    is no dtype trick to be had)
  eigh               = 9 n^3,  thin SVD = 26 n^3   (per-MLP one-offs, amortised)

Components:
  prune  drop neurons with covprop alpha <= tau on both sides of each matmul,
         compensating with the exact constant offset; dead outputs reported with
         their closed-form rectified-Gaussian mean
  rot    fold an alignment basis Q into W1 (zero marginal per-sample cost):
         'w1'  = eigenvectors of W1 W1^T          (eigh, 1.5e8)
         'jac' = left singular vectors of the covprop-linearised end-to-end map
                 W1 D1 W2 D2 ... (31 matmuls + thin SVD, ~1.5e9)
  radius put an exactly stratified chi_256 radius on one extra Sobol coordinate
         and rescale each point to it

Run: uv run python local_sim/sprint1_stack.py [--mlps 100] [--n 16384]
"""

import argparse
import itertools
import os
import sys

import numpy as np
from scipy.special import ndtri
from scipy.stats import chi

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_HERE, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import real_data as rd  # noqa: E402
from prune_probe import covprop_all, matmul_cost  # noqa: E402

F32 = np.float32
BUDGET = 2.72e11
PPF_COST = 168.0
COVPROP_COST = 3.27e9  # measured f64 Gaussian-closure cost per MLP
SOBOL_GEN = 30 * 3 * 256


_VT_CACHE = {}


def _dirnums(d, bits=30):
    """Joe-Kuo direction numbers for d dimensions. The shipped estimator embeds
    256; the radius variant needs 257, so build the table here (dev-only) and
    regenerate the embedded blob at the wider size only if radius wins."""
    if d not in _VT_CACHE:
        from scipy.stats import _sobol

        v = np.zeros((d, bits), dtype=np.uint32)
        _sobol._initialize_v(v, d, bits)
        _VT_CACHE[d] = np.ascontiguousarray(v.T)
    return _VT_CACHE[d]


def sobol_uniforms(n, d, seed):
    bits = 30
    vt = _dirnums(d, bits)
    idx = np.arange(n, dtype=np.uint32)
    idx = (idx ^ (idx >> np.uint32(1)))[:, None]
    pts = np.zeros((n, d), dtype=np.uint32)
    for b in range(bits):
        pts = np.bitwise_xor(pts, ((idx >> np.uint32(b)) & np.uint32(1)) * vt[b])
    shift = np.random.default_rng(seed).integers(0, 1 << bits, size=d, dtype=np.uint32)
    u = (np.bitwise_xor(pts, shift).astype(np.float64) + 0.5) / float(1 << bits)
    return np.clip(u, 1e-12, 1 - 1e-12)


def make_points(n, seed, radius: bool):
    d = 257 if radius else 256
    u = sobol_uniforms(n, d, seed)
    z = ndtri(u[:, :256]).astype(F32)
    cost = 256 * PPF_COST + SOBOL_GEN
    if radius:
        r = chi.ppf(u[:, 256], df=256).astype(F32)
        nrm = np.sqrt((z.astype(np.float64) ** 2).sum(axis=1)).astype(F32)
        z = z * (r / nrm)[:, None]
        cost += PPF_COST + 3 * 256  # one extra quantile + norm/rescale
    return z, cost


def basis_w1(w):
    a = np.asarray(w[0], dtype=np.float64)
    _, Q = np.linalg.eigh(a @ a.T)
    return Q[:, ::-1].astype(F32), 9 * 256**3


def basis_jac(w, alphas):
    from scipy.special import ndtr

    J = np.asarray(w[0], dtype=np.float64)
    cost = 0
    for k in range(1, len(w)):
        J = (J * ndtr(alphas[k - 1])[None, :]) @ np.asarray(w[k], dtype=np.float64)
        cost += matmul_cost(256, 256, 256)
    U, _, _ = np.linalg.svd(J, full_matrices=False)
    cost += 26 * 256**3
    return U.astype(F32), cost


def run(x, w, alphas, mus, tau, Q):
    """Forward pass with optional pruning and an alignment basis folded into W1."""
    n = w[0].shape[1]
    depth = len(w)
    N = len(x)
    cost = 0
    h = x
    alive_prev = np.ones(n, dtype=bool)
    for k in range(depth):
        alive = (alphas[k] > tau) if (k < depth - 1 and tau is not None) else np.ones(n, dtype=bool)
        wk = np.asarray(w[k], dtype=F32)
        if k == 0 and Q is not None:
            wk = (Q.T @ wk.astype(np.float64)).astype(F32)  # fold Q: x=Qu  =>  u @ (Q^T W1)
        wk_s = wk[np.ix_(alive_prev, alive)]
        z = h @ wk_s
        cost += matmul_cost(N, int(alive_prev.sum()), int(alive.sum()))
        dead_prev = ~alive_prev
        if dead_prev.any():
            z = z + (mus[k - 1][dead_prev] @ wk[np.ix_(dead_prev, alive)].astype(np.float64)).astype(F32)
        h = np.maximum(z, F32(0))
        alive_prev = alive
    out = mus[-1].copy()
    out[alive_prev] = h.mean(axis=0, dtype=np.float64)
    return out, cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlps", type=int, default=100)
    ap.add_argument("--split", default="mini")
    ap.add_argument("--n", type=int, default=16384)
    ap.add_argument("--tau", type=float, default=-2.5)
    args = ap.parse_args()

    combos = list(itertools.product([None, args.tau], ["none", "w1", "jac"], [False, True]))
    res = {c: {"mse": [], "cost": []} for c in combos}

    n_mlps = min(args.mlps, rd.n_mlps(args.split))
    for i in range(n_mlps):
        w, truth, _ = rd.get(i, split=args.split, dtype=np.float64)
        mlp = rd.as_mlp(i, split=args.split)
        alphas, mus = covprop_all(w)
        t = truth[-1]
        seed = int(mlp.seed) & 0x7FFFFFFF
        bases = {"none": (None, 0)}
        bases["w1"] = basis_w1(w)
        bases["jac"] = basis_jac(w, alphas)
        pts = {False: make_points(args.n, seed, False), True: make_points(args.n, seed, True)}

        for tau, rotname, rad in combos:
            x, pcost = pts[rad]
            Q, qcost = bases[rotname]
            est, mcost = run(x, w, alphas, mus, tau, Q)
            res[(tau, rotname, rad)]["mse"].append(float(((est - t) ** 2).mean()))
            res[(tau, rotname, rad)]["cost"].append(mcost + args.n * pcost + qcost + COVPROP_COST)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{n_mlps}", flush=True)

    print(f"\n{n_mlps} MLPs, N={args.n}, tau={args.tau}\n")
    print(f"{'prune':>6} {'rot':>5} {'radius':>7} {'MSE':>12} {'cost/sample':>13} {'C/B':>7} {'V*c/B':>11} {'gain':>7}")
    print("-" * 78)
    base = None
    rows = []
    for c in combos:
        m = float(np.mean(res[c]["mse"]))
        tot = float(np.mean(res[c]["cost"]))
        cps = tot / args.n
        adj = m * tot / BUDGET
        if base is None:
            base = adj
        rows.append((c, m, cps, tot / BUDGET, adj, base / adj))
    for (tau, rotname, rad), m, cps, cb, adj, gain in rows:
        print(
            f"{'yes' if tau else 'no':>6} {rotname:>5} {'yes' if rad else 'no':>7} "
            f"{m:>12.4e} {cps:>13,.0f} {cb:>7.3f} {adj:>11.4e} {gain:>6.2f}x"
        )
    best = max(rows, key=lambda r: r[5])
    print(f"\nbest: prune={best[0][0]} rot={best[0][1]} radius={best[0][2]}  ->  {best[4]:.4e}  ({best[5]:.2f}x)")


if __name__ == "__main__":
    main()
