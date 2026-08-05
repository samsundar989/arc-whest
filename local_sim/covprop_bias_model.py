"""
Is covariance propagation's residual a predictable BIAS, or just noise?

The per-layer calibration already removes a single scalar per layer. What is
left, we now measure against 1e9-sample truth on up to 1000 real MLPs. The
question that decides how much B1 fusion can win:

    covprop residual  c_i = p_i - t_i
    Does E[c_i | features] differ meaningfully from 0?

If the conditional mean is a large fraction of the conditional rms, then a
low-dimensional correction curve (shipped as literals) removes it, which both
improves covprop directly AND makes it a better fusion partner. If the residual
is conditionally centred, only its VARIANCE is modellable and fusion is limited
to precision weighting.

The natural conditioning variable is alpha = mu_pre/sigma, covprop's own firing
index: it is scale-free, available online, and the residual clearly varies with it.

Run: uv run python local_sim/covprop_bias_model.py [--split mini] [--fit-frac 0.8]
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DATA = os.path.join(_ROOT, "experiments", "fusion_data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="mini")
    ap.add_argument("--fit-frac", type=float, default=0.8)
    args = ap.parse_args()

    d = dict(np.load(os.path.join(DATA, f"{args.split}_fusion_data.npz"), allow_pickle=True))
    t, p, s, sig, alpha, var = d["t"], d["p"], d["s"], d["sig"], d["alpha"], d["var"]
    m = len(t)
    nfit = int(args.fit_frac * m)
    fit = slice(0, nfit)
    hold = slice(nfit, m)
    print(f"split={args.split}  {m} MLPs   fit on {nfit}, held out {m - nfit}\n")

    c = p - t  # covprop residual
    e = s - t  # sampling residual

    # ---- residual normalised by sigma: is it a function of alpha alone? -----
    a_f, cn_f = alpha[fit].ravel(), (c[fit] / np.maximum(sig[fit], 1e-12)).ravel()
    a_h, cn_h = alpha[hold].ravel(), (c[hold] / np.maximum(sig[hold], 1e-12)).ravel()

    edges = np.concatenate([[-np.inf], np.arange(-4.0, 4.01, 0.5), [np.inf]])
    print(f"{'alpha bin':>15} {'n':>9} {'mean c/sig':>12} {'rms c/sig':>11} {'|mean|/rms':>11}")
    print("-" * 62)
    centres, means = [], []
    for k in range(len(edges) - 1):
        msk = (a_f >= edges[k]) & (a_f < edges[k + 1])
        if msk.sum() < 100:
            continue
        mu = float(cn_f[msk].mean())
        rms = float(np.sqrt((cn_f[msk] ** 2).mean()))
        lo = edges[k] if np.isfinite(edges[k]) else edges[k + 1] - 0.5
        hi = edges[k + 1] if np.isfinite(edges[k + 1]) else edges[k] + 0.5
        centres.append(0.5 * (lo + hi))
        means.append(mu)
        print(f"  [{lo:>5.1f},{hi:>5.1f}) {msk.sum():>9} {mu:>12.5f} {rms:>11.5f} {abs(mu) / max(rms, 1e-12):>11.3f}")

    centres = np.array(centres)
    means = np.array(means)

    # ---- does subtracting the fitted curve help on HELD-OUT MLPs? ----------
    corr_h = np.interp(a_h, centres, means)
    mse_before = float((cn_h**2).mean())
    mse_after = float(((cn_h - corr_h) ** 2).mean())
    print(f"\nheld-out, in units of (c/sigma)^2:")
    print(f"  before bias correction   {mse_before:.5e}")
    print(f"  after  bias correction   {mse_after:.5e}   ({100 * (1 - mse_after / mse_before):.1f}% removed)")

    # in absolute MSE terms on the final layer
    c_h = c[hold]
    corr_abs = np.interp(alpha[hold], centres, means) * sig[hold]
    print(f"\nheld-out covprop final-layer MSE:")
    print(f"  raw calibrated covprop   {float((c_h**2).mean()):.4e}")
    print(f"  + alpha bias curve       {float(((c_h - corr_abs) ** 2).mean()):.4e}")

    # ---- where does the SAMPLING error live, by alpha? ---------------------
    print(f"\n{'alpha bin':>15} {'n':>9} {'covprop MSE':>13} {'sampling MSE':>14} {'covprop wins?':>14}")
    print("-" * 70)
    ah, ch, eh = alpha[hold].ravel(), c[hold].ravel(), e[hold].ravel()
    tot_c = tot_e = tot_best = 0.0
    for k in range(len(edges) - 1):
        msk = (ah >= edges[k]) & (ah < edges[k + 1])
        if msk.sum() < 100:
            continue
        mc = float((ch[msk] ** 2).mean())
        me = float((eh[msk] ** 2).mean())
        lo = edges[k] if np.isfinite(edges[k]) else edges[k + 1] - 0.5
        hi = edges[k + 1] if np.isfinite(edges[k + 1]) else edges[k] + 0.5
        frac = msk.sum() / len(ah)
        tot_c += frac * mc
        tot_e += frac * me
        tot_best += frac * min(mc, me)
        print(f"  [{lo:>5.1f},{hi:>5.1f}) {msk.sum():>9} {mc:>13.4e} {me:>14.4e} {'YES' if mc < me else '':>14}")
    print(f"\n  totals: covprop {tot_c:.4e}   sampling {tot_e:.4e}   per-bin best {tot_best:.4e}")
    print(f"  => an alpha-gated hard switch would give {tot_best:.4e} ({100 * (1 - tot_best / tot_e):.1f}% better than sampling)")


if __name__ == "__main__":
    main()
