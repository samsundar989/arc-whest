"""
B2, measured properly: replace the LAST k layers of sampling with an analytic
Gaussian step taken from the SAMPLED state at layer L-k.

This is not the same as covariance propagation from the input (which the layer
crossover shows is 18x worse than sampling by layer 32). Here the analytic step
starts from an empirically estimated mean+covariance that is already accurate, so
its only errors are (a) the Gaussian-closure bias over k layers and (b) the noise
in estimating those moments from N samples.

The trade: it removes k layers' worth of ReLU sampling noise, but pays for
estimating a 256x256 covariance from N=16384 draws.

Run: uv run python local_sim/b2_tail_probe.py [--mlps 40] [--ks 1,2,3]
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
from scale_source_probe import sobol_points_for  # noqa: E402

F32 = np.float32
INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def npdf(x):
    return INV_SQRT_2PI * np.exp(-0.5 * x * x)


def analytic_steps(mu, cov, weights):
    """Gaussian-closure propagation through `weights`, starting from (mu, cov)."""
    n = len(mu)
    off = np.ones((n, n)) - np.eye(n)
    for w in weights:
        mu_pre = w.T @ mu
        cov_pre = w.T @ cov @ w
        var_pre = np.maximum(np.diag(cov_pre), 1e-12)
        sig = np.sqrt(var_pre)
        a = mu_pre / sig
        Phi, phi = ndtr(a), npdf(a)
        mu = mu_pre * Phi + sig * phi
        ez2 = (mu_pre * mu_pre + var_pre) * Phi + mu_pre * sig * phi
        var = np.maximum(ez2 - mu * mu, 0.0)
        cov = np.outer(Phi, Phi) * cov_pre * off + np.diag(var)
        cov = 0.5 * (cov + cov.T)
    return mu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlps", type=int, default=40)
    ap.add_argument("--offset", type=int, default=800)
    ap.add_argument("--split", default="full")
    ap.add_argument("--ks", default="1,2,3")
    args = ap.parse_args()
    ks = [int(v) for v in args.ks.split(",")]

    res = {k: [] for k in ks}
    base = []
    for j in range(args.mlps):
        i = args.offset + j
        w_np, truth, _ = rd.get(i, split=args.split, dtype=np.float64)
        mlp = rd.as_mlp(i, split=args.split)
        x, N = sobol_points_for(mlp)
        wl = [np.asarray(ww, dtype=F32) for ww in mlp.weights]
        depth = len(wl)

        acts = []
        h = x
        for ww in wl:
            h = np.maximum(h @ ww, F32(0))
            acts.append(h)
        base.append(float(((acts[-1].mean(axis=0, dtype=np.float64) - truth[-1]) ** 2).mean()))

        for k in ks:
            src = acts[depth - 1 - k].astype(np.float64)  # sampled state at L-k
            mu = src.mean(axis=0)
            cov = np.cov(src, rowvar=False, bias=True)
            tail = [np.asarray(ww, dtype=np.float64) for ww in mlp.weights[depth - k :]]
            est = analytic_steps(mu, cov, tail)
            res[k].append(float(((est - truth[-1]) ** 2).mean()))
        if (j + 1) % 10 == 0:
            print(f"  {j + 1}/{args.mlps}", flush=True)

    b = float(np.mean(base))
    print(f"\n{args.mlps} MLPs, N={N} samples\n")
    print(f"{'method':<44} {'final-layer MSE':>17} {'vs sampling':>12}")
    print("-" * 76)
    print(f"{'pure sampling (v1.1 tail)':<44} {b:>17.4e} {'--':>12}")
    for k in ks:
        v = float(np.mean(res[k]))
        print(f"{'analytic last ' + str(k) + ' layer(s) from sampled moments':<44} {v:>17.4e} {100 * (1 - v / b):>+11.1f}%")


if __name__ == "__main__":
    main()
