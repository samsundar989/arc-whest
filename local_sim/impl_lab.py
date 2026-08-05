"""
Implementability experiments: which variance-reduction scheme can we actually
ship inside `estimator.py` under the grader's constraints?

Scrambled Sobol wins offline, but scipy is not importable in the sandbox, so the
point set must either be (a) shipped as data and re-randomized cheaply in-estimator,
or (b) generated from a handful of numbers with pure fnp arithmetic. This script
compares:

  sobol            scipy scrambled Sobol           (offline reference / upper bound)
  sobol_shift      FIXED Sobol points + Cranley-Patterson random shift mod 1
                   -> shippable as an .npz, randomization is 2 flops/element
  korobov_a        rank-1 Korobov lattice + random shift
                   -> shippable as 256 integers, generated with pure arithmetic
  *_cv1            the above, plus a layer-1 regression control variate whose
                   mean is known in closed form (so it stays unbiased)

Run: uv run python local_sim/impl_lab.py [--n 16384] [--reps 8] [--mlps 3]
"""

import argparse
import sys
import time

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])

import real_data as rd

F32 = np.float32
CHUNK = 16384
COST_PER_SAMPLE = 32 * 2 * 256 * 256 + 32 * 256
BUDGET = 2.72e11

_SOBOL_CACHE = {}


def fixed_sobol(n, d):
    """One deterministic (unscrambled) Sobol point set — the thing we would ship."""
    key = (n, d)
    if key not in _SOBOL_CACHE:
        eng = qmc.Sobol(d=d, scramble=False, seed=0)
        u = eng.random(n)
        u[0] = 0.5 / n  # avoid the exact-zero first point (ndtri -> -inf)
        _SOBOL_CACHE[key] = u
    return _SOBOL_CACHE[key]


def to_normal(u):
    np.clip(u, 1e-9, 1 - 1e-9, out=u)
    return ndtri(u).astype(F32)


def pts_sobol_scrambled(n, d, rng):
    eng = qmc.Sobol(d=d, scramble=True, seed=int(rng.integers(2**31)))
    return to_normal(eng.random(n))


def pts_sobol_shift(n, d, rng):
    u = fixed_sobol(n, d) + rng.random(d)[None, :]
    return to_normal(np.mod(u, 1.0))


def korobov_vector(n, d, a):
    g = np.ones(d, dtype=np.int64)
    for j in range(1, d):
        g[j] = (g[j - 1] * a) % n
    return g


def make_korobov(a):
    def pts(n, d, rng):
        g = korobov_vector(n, d, a)
        i = np.arange(n, dtype=np.float64)[:, None]
        u = np.mod(i * (g[None, :] / n) + rng.random(d)[None, :], 1.0)
        return to_normal(u)

    return pts


# --------------------------------------------------------------------- run ---
def run_batch(x, w, want_h1=False):
    """Forward pass; return (final activations mean, optionally per-sample h1/hL)."""
    n = len(x)
    d = w[0].shape[1]
    fin = np.empty((n, d), dtype=F32) if want_h1 else None
    h1s = np.empty((n, d), dtype=F32) if want_h1 else None
    acc = np.zeros(d, dtype=np.float64)
    for s in range(0, n, CHUNK):
        h = x[s : s + CHUNK]
        for li, wi in enumerate(w):
            h = np.maximum(h @ wi, F32(0))
            if want_h1 and li == 0:
                h1s[s : s + len(h)] = h
        if want_h1:
            fin[s : s + len(h)] = h
        acc += h.sum(axis=0, dtype=np.float64)
    return acc / n, fin, h1s


def exact_h1_mean(w0):
    s = np.sqrt(np.maximum(np.diag(w0.astype(np.float64).T @ w0.astype(np.float64)), 0.0))
    return s / np.sqrt(2 * np.pi)


def estimate(x, w, cv: bool):
    """Final-layer mean estimate, optionally with the layer-1 control variate."""
    if not cv:
        return run_batch(x, w)[0]
    _, fin, h1 = run_batch(x, w, want_h1=True)
    mu1 = exact_h1_mean(w[0])
    a = h1.astype(np.float64)
    y = fin.astype(np.float64)
    ac = a - a.mean(axis=0)
    yc = y - y.mean(axis=0)
    Saa = ac.T @ ac / len(a)
    Say = ac.T @ yc / len(a)
    ridge = 1e-8 * np.trace(Saa) / Saa.shape[0]
    B = np.linalg.solve(Saa + ridge * np.eye(Saa.shape[0]), Say)
    return y.mean(axis=0) - (a.mean(axis=0) - mu1) @ B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16384)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--mlps", type=int, default=3)
    args = ap.parse_args()

    d = 256
    n = args.n
    cb = n * COST_PER_SAMPLE / BUDGET
    methods = {
        "mc": (lambda n, d, rng: rng.standard_normal((n, d), dtype=F32), False),
        "sobol (scipy scrambled)": (pts_sobol_scrambled, False),
        "sobol_shift (shippable)": (pts_sobol_shift, False),
        "korobov a=17797": (make_korobov(17797), False),
        "korobov a=76129": (make_korobov(76129), False),
        "sobol_shift + cv1": (pts_sobol_shift, True),
        "mc + cv1": (lambda n, d, rng: rng.standard_normal((n, d), dtype=F32), True),
    }

    print(f"N={n}  C/B={cb:.3f}  multiplier={max(0.1, cb):.3f}  reps={args.reps} mlps={args.mlps}\n")
    print(f"{'method':<26} {'MSE':>11} {'vs mc':>7} {'adjusted':>11} {'sec':>6}")
    print("-" * 66)
    base = None
    for name, (fn, cv) in methods.items():
        t0 = time.perf_counter()
        mses = []
        for mi in range(args.mlps):
            w, truth, _ = rd.get(mi)
            t = truth[-1]
            for r in range(args.reps):
                rng = np.random.default_rng(90_000 + 1000 * mi + r)
                mses.append(float(((estimate(fn(n, d, rng), w, cv) - t) ** 2).mean()))
        mse = float(np.mean(mses))
        base = base or mse
        print(
            f"{name:<26} {mse:>11.4e} {base / mse:>6.2f}x {mse * max(0.1, cb):>11.3e} "
            f"{time.perf_counter() - t0:>6.1f}"
        )


if __name__ == "__main__":
    main()
