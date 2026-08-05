"""
Phase A measurement rig: where does sampling error in the final layer come from,
and which variance-reduction trick actually removes it?

Runs against the REAL dataset (1e9-sample ground truth), so measurements down to
~1e-9 MSE are meaningful. Every method returns an unbiased-or-nearly-unbiased
(256,) estimate of the final-layer means; we report E[MSE] over independent
replications, which is what the leaderboard scores.

Run:  uv run python local_sim/variance_lab.py [--n 8192] [--reps 24] [--mlps 4]
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
CHUNK = 32768


# ----------------------------------------------------------------- forward ---
def final_mean(x, w):
    """Mean of the final-layer post-ReLU activations over the rows of x."""
    acc = np.zeros(w[0].shape[1], dtype=np.float64)
    for s in range(0, len(x), CHUNK):
        h = x[s : s + CHUNK]
        for wi in w:
            h = np.maximum(h @ wi, F32(0))
        acc += h.sum(axis=0, dtype=np.float64)
    return acc / len(x)


# ----------------------------------------------------- input point sets ------
def pts_mc(n, d, rng):
    return rng.standard_normal((n, d), dtype=F32)


def pts_antithetic(n, d, rng):
    half = rng.standard_normal((n // 2, d), dtype=F32)
    return np.concatenate([half, -half], axis=0)


def pts_sobol(n, d, rng):
    eng = qmc.Sobol(d=d, scramble=True, seed=int(rng.integers(2**31)))
    u = eng.random(n)
    np.clip(u, 1e-12, 1 - 1e-12, out=u)
    return ndtri(u).astype(F32)


def _sphere(n, d, rng):
    g = rng.standard_normal((n, d), dtype=F32)
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    return g


def _chi_ppf(p, k):
    """Quantile of chi_k via Wilson-Hilferty on chi^2_k (exact enough for k=256,
    and expressible with only norm.ppf — which is what the real estimator has)."""
    z = ndtri(p)
    a = 2.0 / (9.0 * k)
    return np.sqrt(k * (1.0 - a + z * np.sqrt(a)) ** 3)


def pts_radial(n, d, rng):
    """Stratify the input radius ||x|| into n equal-probability strata (one draw
    each), with an independent uniform direction. Removes all radial variance."""
    u = (np.arange(n) + rng.random(n)) / n
    r = _chi_ppf(u, d).astype(F32)
    return _sphere(n, d, rng) * r[:, None]


def pts_radial_anti(n, d, rng):
    half = n // 2
    u = (np.arange(half) + rng.random(half)) / half
    r = _chi_ppf(u, d).astype(F32)
    dirs = _sphere(half, d, rng)
    x = dirs * r[:, None]
    return np.concatenate([x, -x], axis=0)


def pts_sobol_radial(n, d, rng):
    """Sobol directions + stratified radius: dimension 0 of the Sobol point
    drives the radius, the remaining d dims give the direction."""
    eng = qmc.Sobol(d=d + 1, scramble=True, seed=int(rng.integers(2**31)))
    u = eng.random(n)
    np.clip(u, 1e-12, 1 - 1e-12, out=u)
    r = _chi_ppf(u[:, 0], d).astype(F32)
    g = ndtri(u[:, 1:]).astype(F32)
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    return g * r[:, None]


METHODS = {
    "mc": pts_mc,
    "antithetic": pts_antithetic,
    "sobol": pts_sobol,
    "radial": pts_radial,
    "radial+anti": pts_radial_anti,
    "sobol+radial": pts_sobol_radial,
}


# ------------------------------------------------------------ diagnostics ----
def error_structure(errs, truth_final):
    """Decompose replication errors (reps, width) into common-mode vs residual.

    The 'scale mode' is the component along the truth vector: an error that
    multiplies the whole layer by (1+eps). If most of the variance lives there,
    a single scalar control variate would remove most of the sampling error.
    """
    t = truth_final
    live = t > 0
    tn = t[live] / np.linalg.norm(t[live])
    e = errs[:, live]
    along = e @ tn  # (reps,)
    resid = e - np.outer(along, tn)
    tot = float((e**2).sum(axis=1).mean())
    scale_share = float((along**2).mean()) / tot if tot else 0.0
    # uniform (constant-offset) mode
    un = np.ones(live.sum()) / np.sqrt(live.sum())
    flat = e @ un
    flat_share = float((flat**2).mean()) / tot if tot else 0.0
    return {
        "scale_mode_share": scale_share,
        "flat_mode_share": flat_share,
        "residual_mse": float((resid**2).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8192, help="samples per estimate")
    ap.add_argument("--reps", type=int, default=24)
    ap.add_argument("--mlps", type=int, default=4)
    ap.add_argument("--methods", type=str, default=",".join(METHODS))
    args = ap.parse_args()

    names = [m for m in args.methods.split(",") if m in METHODS]
    d = 256
    cost_per_sample = 32 * 2 * d * d + 32 * d  # f32 flopscope price
    cb = args.n * cost_per_sample / 2.72e11
    print(f"N={args.n} samples/estimate  ->  C/B ≈ {cb:.3f}  (multiplier {max(0.1, cb):.3f})")
    print(f"reps={args.reps}  mlps={args.mlps}\n")

    results = {m: [] for m in names}
    struct = {m: [] for m in names}

    for mi in range(args.mlps):
        w, truth, meta = rd.get(mi)
        t = truth[-1]
        line = f"MLP {mi} ({meta['mlp_name'][:18]:<18} var={meta['avg_variance']:.4f})"
        for m in names:
            fn = METHODS[m]
            t0 = time.perf_counter()
            errs = np.empty((args.reps, d))
            for r in range(args.reps):
                rng = np.random.default_rng(1000 * mi + r)
                errs[r] = final_mean(fn(args.n, d, rng), w) - t
            mse = float((errs**2).mean())
            results[m].append(mse)
            struct[m].append(error_structure(errs, t))
            line += f"\n    {m:<14} MSE={mse:.4e}  adj={mse * max(0.1, cb):.3e}  ({time.perf_counter() - t0:5.1f}s)"
        print(line, flush=True)

    print("\n" + "=" * 78)
    print(f"{'method':<14} {'mean MSE':>12} {'vs mc':>7} {'adjusted':>11} {'scale%':>8} {'flat%':>7}")
    print("-" * 78)
    base = float(np.mean(results[names[0]])) if names else 1.0
    for m in names:
        mse = float(np.mean(results[m]))
        sc = float(np.mean([s["scale_mode_share"] for s in struct[m]]))
        fl = float(np.mean([s["flat_mode_share"] for s in struct[m]]))
        print(
            f"{m:<14} {mse:>12.4e} {base / mse:>6.2f}x {mse * max(0.1, cb):>11.3e} "
            f"{100 * sc:>7.1f}% {100 * fl:>6.1f}%"
        )
    print(
        "\nscale% = share of error variance along the truth direction (a pure "
        "rescaling of the layer)\nflat%  = share along the constant vector"
    )


if __name__ == "__main__":
    main()
