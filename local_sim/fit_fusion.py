"""
Fit the B1 fusion constants, and honestly report what fusion is worth.

Two mechanisms, fitted on the FULL split (1e9-sample truth = no label noise) and
reported on a held-out block plus the disjoint mini split:

  (i)  scale-mode kill.  The sampling run's common scale fluctuation g_L is
       estimated online by  ghat = <s_k - p_k, p_k>/<p_k,p_k>  at a mid-depth
       layer k (covprop is trustworthy there, and g has largely accumulated).
       We fit  g_L ~ a*ghat + b  and shrink by the fit's own reliability.

  (ii) precision blend.  t_hat = w*p + (1-w)*s' with w from the two error
       variances. Because covprop's residual variance dwarfs sampling's, w is
       small and this is worth only a few percent — measured, not assumed.

Run: uv run python local_sim/fit_fusion.py [--split full] [--holdout 200]
"""

import argparse
import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DATA = os.path.join(_ROOT, "experiments", "fusion_data")


def load(split):
    return dict(np.load(os.path.join(DATA, f"{split}_fusion_data.npz"), allow_pickle=True))


def scale_of(a, ref):
    return (a * ref).sum(axis=1) / np.maximum((ref * ref).sum(axis=1), 1e-300)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="full")
    ap.add_argument("--holdout", type=int, default=200)
    ap.add_argument("--probe-layer", type=int, default=12, help="1-based layer for ghat")
    args = ap.parse_args()

    d = load(args.split)
    t, p, s, sig, alpha, var = d["t"], d["p"], d["s"], d["sig"], d["alpha"], d["var"]
    m = len(t)
    nfit = m - args.holdout
    F, H = slice(0, nfit), slice(nfit, m)
    print(f"split={args.split}: {m} MLPs, fit {nfit}, holdout {args.holdout}\n")

    ec, es = p - t, s - t
    vc = float((ec[F] ** 2).mean())
    vs = float((es[F] ** 2).mean())
    print(f"covprop  MSE (fit)  {vc:.4e}")
    print(f"sampling MSE (fit)  {vs:.4e}")

    # ---------- (ii) precision blend -------------------------------------
    # global weight
    w_glob = vs / (vs + vc)
    # per-neuron weight from modelled variances:
    #   sampling: Var(eps_i) = var_i / (N*kappa)   (kappa = QMC gain, fitted)
    #   covprop : Var(c_i)   = (rho(alpha_i) * sig_i)^2, rho a binned curve
    N = 16384
    kappa = float(np.median(var[F].mean(axis=0) / np.maximum((es[F] ** 2).mean(axis=0), 1e-300) / N))
    edges = np.concatenate([[-np.inf], np.arange(-4.0, 4.01, 0.5), [np.inf]])
    a_f = alpha[F].ravel()
    rel_f = (np.abs(ec[F]) / np.maximum(sig[F], 1e-12)).ravel()
    centres, rhos = [], []
    for k in range(len(edges) - 1):
        msk = (a_f >= edges[k]) & (a_f < edges[k + 1])
        if msk.sum() < 50:
            continue
        lo = edges[k] if np.isfinite(edges[k]) else edges[k + 1] - 0.5
        hi = edges[k + 1] if np.isfinite(edges[k + 1]) else edges[k] + 0.5
        centres.append(0.5 * (lo + hi))
        rhos.append(float(np.sqrt((rel_f[msk] ** 2).mean())))
    centres, rhos = np.array(centres), np.array(rhos)
    print(f"\nkappa (QMC gain over plain MC) = {1 / kappa:.2f}" if kappa else "")

    def per_neuron_w(alpha_, sig_, var_):
        vs_i = var_ / (N * (1.0 / kappa))
        vc_i = (np.interp(alpha_, centres, rhos) * sig_) ** 2
        return vs_i / np.maximum(vs_i + vc_i, 1e-300)

    # ---------- (i) scale-mode kill ---------------------------------------
    # ghat needs a mid-layer probe; the cached data only has the final layer, so
    # here we use the FINAL-layer observable and let the fit tell us its worth.
    ghat_L = scale_of(s - p, p)
    gL = scale_of(s - t, t)
    A = np.stack([ghat_L[F], np.ones(nfit)], axis=1)
    beta = np.linalg.lstsq(A, gL[F], rcond=None)[0]
    pred_fit = A @ beta
    r2_fit = 1 - ((gL[F] - pred_fit) ** 2).mean() / gL[F].var()
    Ah = np.stack([ghat_L[H], np.ones(m - nfit)], axis=1)
    pred_h = Ah @ beta
    r2_h = 1 - ((gL[H] - pred_h) ** 2).mean() / gL[H].var()
    print(f"\nscale-mode regression  g_L ~ {beta[0]:+.4f}*ghat {beta[1]:+.6f}")
    print(f"  R^2 fit {r2_fit:.3f}   R^2 holdout {r2_h:.3f}")

    # ---------- evaluate the combinations on the holdout ------------------
    def report(label, est, sl):
        mse = float(((est - t[sl]) ** 2).mean())
        print(f"  {label:<44} {mse:.4e}   ({100 * (1 - mse / float(((s[sl] - t[sl]) ** 2).mean())):+.1f}% vs sampling)")
        return mse

    for name, sl in (("HOLDOUT", H),):
        print(f"\n{name}:")
        report("sampling alone (v1.1)", s[sl], sl)
        gh = np.stack([ghat_L[sl], np.ones(sl.stop - sl.start)], axis=1) @ beta
        s_corr = s[sl] / (1.0 + gh[:, None])
        report("+ scale-mode kill", s_corr, sl)
        report("+ global precision blend", w_glob * p[sl] + (1 - w_glob) * s[sl], sl)
        wpn = per_neuron_w(alpha[sl], sig[sl], var[sl])
        report("+ per-neuron GLS blend", wpn * p[sl] + (1 - wpn) * s[sl], sl)
        both = wpn * p[sl] + (1 - wpn) * s_corr
        report("+ scale kill AND per-neuron blend", both, sl)
        print(f"  realized w: mean {wpn.mean():.4f}  median {np.median(wpn):.4f}  max {wpn.max():.4f}")

    out = {
        "N": N,
        "kappa": 1.0 / kappa,
        "w_global": w_glob,
        "scale_beta": beta.tolist(),
        "scale_r2_holdout": float(r2_h),
        "rho_centres": centres.tolist(),
        "rho_values": rhos.tolist(),
        "vc_fit": vc,
        "vs_fit": vs,
    }
    path = os.path.join(DATA, "fusion_constants.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
