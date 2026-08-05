"""
Error anatomy of the two estimators — the evidence the B1 fusion is built on.

Decomposes each estimator's final-layer residual into
  * a per-MLP COMMON SCALE mode (the whole layer multiplied by 1+g), and
  * the per-neuron remainder,
and checks how the remainder's size varies with covprop's own in-flight
diagnostics (sigma, alpha). Also reports the cross-correlation between the two
estimators' errors, which is what decides whether fusing them can help at all.

Run: uv run python local_sim/fusion_anatomy.py [--split mini]
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DATA = os.path.join(_ROOT, "experiments", "fusion_data")


def load(split):
    return dict(np.load(os.path.join(DATA, f"{split}_fusion_data.npz"), allow_pickle=True))


def scale_split(resid, t, weights=None):
    """Split residual r (m, n) into the component along t and the remainder.

    Returns (g, r_perp) where g is the per-MLP scale coefficient minimising
    ||r - g*t||^2 (optionally weighted), i.e. r ≈ g*t + r_perp.
    """
    if weights is None:
        weights = np.ones_like(t)
    num = (weights * resid * t).sum(axis=1)
    den = (weights * t * t).sum(axis=1)
    g = num / np.maximum(den, 1e-300)
    return g, resid - g[:, None] * t


def anatomy(name, resid, t):
    g, rp = scale_split(resid, t)
    tot = float((resid**2).mean())
    perp = float((rp**2).mean())
    print(f"\n{name}")
    print(f"  MSE                        {tot:.4e}")
    print(f"  common-scale share         {100 * (1 - perp / tot):5.1f}%")
    print(f"  per-neuron remainder MSE   {perp:.4e}")
    print(f"  g: mean {g.mean():+.5f}  sd {g.std():.5f}  (a pure rescale of the layer)")
    return g, rp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="mini")
    args = ap.parse_args()
    d = load(args.split)
    t, p, s = d["t"], d["p"], d["s"]
    sig, alpha, var = d["sig"], d["alpha"], d["var"]
    m, n = t.shape
    print(f"split={args.split}  {m} MLPs x {n} neurons")

    rp_cov = p - t
    rs = s - t
    g_cov, perp_cov = anatomy("COVARIANCE PROPAGATION (v0)", rp_cov, t)
    g_smp, perp_smp = anatomy("SOBOL SAMPLING (v1.1)", rs, t)

    # --- are the two errors independent? that is what makes fusion pay -------
    num = float((rp_cov * rs).mean())
    den = float(np.sqrt((rp_cov**2).mean() * (rs**2).mean()))
    print(f"\ncorr(covprop error, sampling error) = {num / den:+.4f}   (near 0 => fusion pays)")

    # --- sampling noise vs covprop's predicted variance ----------------------
    # theory: Var(eps_i) = Var(h_i) / (N * kappa); covprop gives var ~ Var(h_i)
    N = 16384
    emp = perp_smp**2  # per-neuron squared remainder, (m, n)
    ratio = emp.mean(axis=0) / np.maximum(var.mean(axis=0), 1e-300) * N
    print("\nsampling noise model  Var(eps_i) = var_covprop_i / (N*kappa)")
    print(f"  implied 1/kappa (median over neurons): {np.median(ratio):.4f}")
    print(f"  => kappa ~ {1 / max(np.median(ratio), 1e-12):.2f}  (QMC gain over plain MC, expect ~2)")
    # covprop's variance is itself biased low; measure that separately
    emp_var_true = None
    print(f"  covprop mean predicted Var(h): {var.mean():.5f}")

    # --- covprop residual size vs its own diagnostics ------------------------
    print("\ncovprop |residual| by alpha bin (alpha = mu_pre/sigma, firing index)")
    a = alpha.ravel()
    r = np.abs(perp_cov).ravel()
    sg = sig.ravel()
    edges = np.array([-np.inf, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, np.inf])
    print(f"  {'alpha bin':>14} {'n':>8} {'rms resid':>11} {'mean sigma':>11} {'rms/sigma':>10}")
    for k in range(len(edges) - 1):
        msk = (a >= edges[k]) & (a < edges[k + 1])
        if msk.sum() < 50:
            continue
        rms = float(np.sqrt((r[msk] ** 2).mean()))
        ms = float(sg[msk].mean())
        print(f"  [{edges[k]:>5.1f},{edges[k + 1]:>5.1f}) {msk.sum():>8} {rms:>11.4e} {ms:>11.4f} {rms / ms:>10.4f}")

    # is residual/sigma roughly constant? that would make the model 1-parameter
    rel = (np.abs(perp_cov) / np.maximum(sig, 1e-12)).ravel()
    print(f"\n  |resid|/sigma : mean {rel.mean():.4e}  median {np.median(rel):.4e}  sd {rel.std():.4e}")

    # --- what an ORACLE fusion would achieve (upper bound) ------------------
    # per-neuron optimal convex blend using the realised errors
    ec, es = perp_cov, perp_smp
    wopt = (es**2) / np.maximum(es**2 + ec**2, 1e-300)
    oracle = wopt * p + (1 - wopt) * s
    print(f"\noracle per-neuron blend MSE   {np.mean((oracle - t) ** 2):.4e}  (unreachable upper bound)")
    vc, vs = float((ec**2).mean()), float((es**2).mean())
    w_static = vs / (vs + vc)
    static = w_static * p + (1 - w_static) * s
    print(f"single global weight w={w_static:.4f} -> MSE {np.mean((static - t) ** 2):.4e}")
    print(f"sampling alone                 {np.mean(rs**2):.4e}")


if __name__ == "__main__":
    main()
