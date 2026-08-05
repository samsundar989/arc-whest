"""
Sprint 1 item 1: dead-neuron pruning.

Why this is the strongest lever available. Above the 0.1 multiplier floor the
adjusted score is

    MSE * (C/B) = (V/N) * (N*c/B) = V*c/B

i.e. per-sample VARIANCE times per-sample COST, with N cancelling entirely.
(Sanity: V = 1.4893e-6*16384 = 2.44e-2, c = 4.273e6, B = 2.72e11 -> 3.83e-7,
matching v1.1's graded 3.893e-7.) So halving the cost of a sample halves the
score just as surely as halving the variance.

flopscope bills matmul by SHAPES ONLY -- (2K-1)*M*N for (M,K)@(K,N) -- so the
only way pruning pays is to materialise smaller DENSE matrices. We therefore
predict each layer's dead neurons analytically (from covariance propagation's
alpha = mu_pre/sigma), drop those columns of W_k AND the matching rows of
W_{k+1}, and compensate exactly for what was dropped by adding the constant
offset  c_{k+1} = sum_{i dead} W_{k+1}[i,:] * mu_i  to the next layer's
pre-activations. Dead neurons are reported with their closed-form rectified
Gaussian mean, never with 0.

Pruning both sides makes the per-layer cost scale with f^2 (f = alive fraction),
not f.

Run: uv run python local_sim/prune_probe.py [--mlps 100] [--taus -3,-3.5,-4,-5]
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
from covprop_features import covprop  # noqa: E402
from rotation_probe import sobol_normals  # noqa: E402

F32 = np.float32
BUDGET = 2.72e11


def covprop_all(w):
    """Per-layer alpha and post-ReLU mean, from the same closure v0 uses."""
    from scipy.special import ndtr

    n = w[0].shape[1]
    mu = np.zeros(n)
    cov = np.eye(n)
    off = np.ones((n, n)) - np.eye(n)
    alphas, mus = [], []
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
        alphas.append(a.copy())
        mus.append(mu.copy())
    return np.stack(alphas), np.stack(mus)


def matmul_cost(M, K, N):
    return (2 * K - 1) * M * N


def forward_pruned(x, w, alphas, mus, tau):
    """Forward pass keeping only alive neurons; returns final-layer estimate for
    ALL 256 outputs (dead ones filled with their analytic mean) and the FLOP cost."""
    n = w[0].shape[1]
    depth = len(w)
    N = len(x)
    cost = 0
    h = x  # (N, n) -- input layer is never pruned
    alive_prev = np.ones(n, dtype=bool)
    for k in range(depth):
        alive = alphas[k] > tau if k < depth - 1 else np.ones(n, dtype=bool)
        wk = np.asarray(w[k], dtype=F32)
        wk_s = wk[np.ix_(alive_prev, alive)]
        z = h @ wk_s
        cost += matmul_cost(N, int(alive_prev.sum()), int(alive.sum()))
        dead_prev = ~alive_prev
        if dead_prev.any():
            offs = (mus[k - 1][dead_prev].astype(np.float64) @ wk[np.ix_(dead_prev, alive)].astype(np.float64))
            z = z + offs.astype(F32)
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
    ap.add_argument("--taus", default="-3,-3.5,-4,-4.5,-5")
    args = ap.parse_args()
    taus = [float(v) for v in args.taus.split(",")]

    n_mlps = min(args.mlps, rd.n_mlps(args.split))
    hist = np.zeros((32, 0)).tolist()
    alive_frac = {t: [] for t in taus}
    mses = {t: [] for t in taus}
    costs = {t: [] for t in taus}
    dead_bias = {t: [] for t in taus}
    base_mse, base_cost = [], []

    for i in range(n_mlps):
        w, truth, _ = rd.get(i, split=args.split, dtype=np.float64)
        mlp = rd.as_mlp(i, split=args.split)
        alphas, mus = covprop_all(w)
        if i == 0:
            all_alphas = [alphas]
        elif i < 20:
            all_alphas.append(alphas)
        x = sobol_normals(args.n, 256, int(mlp.seed) & 0x7FFFFFFF)
        t = truth[-1]

        est, c = forward_pruned(x, w, alphas, mus, -np.inf)
        base_mse.append(float(((est - t) ** 2).mean()))
        base_cost.append(c)

        for tau in taus:
            est, c = forward_pruned(x, w, alphas, mus, tau)
            mses[tau].append(float(((est - t) ** 2).mean()))
            costs[tau].append(c)
            alive_frac[tau].append(float(np.mean([(alphas[k] > tau).mean() for k in range(31)])))
            dead_last = alphas[-1] <= tau
            if dead_last.any():
                dead_bias[tau].append(float(((mus[-1][dead_last] - t[dead_last]) ** 2).mean()))
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{n_mlps}", flush=True)

    A = np.concatenate([a.ravel() for a in all_alphas])
    print(f"\nalpha histogram over layers 1-32 ({len(all_alphas)} MLPs):")
    for lo, hi in ((-99, -5), (-5, -4), (-4, -3), (-3, -2), (-2, 0), (0, 2), (2, 4), (4, 99)):
        f = float(((A >= lo) & (A < hi)).mean())
        print(f"  alpha in [{lo:>4},{hi:>4}) : {100 * f:5.1f}%")

    b_mse, b_cost = float(np.mean(base_mse)), float(np.mean(base_cost))
    per_sample_base = b_cost / args.n
    print(f"\nbaseline (no pruning): MSE {b_mse:.4e}  cost/sample {per_sample_base:,.0f}")
    print(f"  V*c/B = {b_mse * args.n * per_sample_base / BUDGET:.4e}  (predicted adjusted score)\n")

    print(f"{'tau':>6} {'alive':>7} {'cost/sample':>13} {'cost cut':>9} {'MSE':>12} {'MSE chg':>9} {'V*c/B':>11} {'gain':>7}")
    print("-" * 84)
    base_adj = b_mse * args.n * per_sample_base / BUDGET
    for tau in taus:
        c = float(np.mean(costs[tau])) / args.n
        m = float(np.mean(mses[tau]))
        adj = m * args.n * c / BUDGET
        print(
            f"{tau:>6.1f} {100 * float(np.mean(alive_frac[tau])):>6.1f}% {c:>13,.0f} "
            f"{100 * (1 - c / per_sample_base):>8.1f}% {m:>12.4e} {100 * (m / b_mse - 1):>+8.1f}% "
            f"{adj:>11.4e} {base_adj / adj:>6.2f}x"
        )
        if dead_bias[tau]:
            print(f"       pruned-neuron MSE vs their true means: {float(np.mean(dead_bias[tau])):.3e}")


if __name__ == "__main__":
    main()
