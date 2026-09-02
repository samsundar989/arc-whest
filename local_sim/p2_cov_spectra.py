"""Is the PRE-activation covariance rank-collapsed, or only the POST-ReLU one?

WP13 measured r90(Sigma) falling 237 -> 26 over 16 layers. That was taken inside
the NONLINEAR hook, i.e. on the POST-ReLU covariance. WP14's range finder is
called at the LINEAR hook, i.e. on cov_pre = W^T Sigma_post W. If W^T (.) W
re-spreads the spectrum, the two are entirely different objects and the range
finder has been fed the wrong one.

Pure K2, numpy, CPU. No seeds drawn: the seeds are the eight P01 networks WP13
already claimed and this reads no label and decides no gate.
"""
import os, sys
sys.path.append(os.path.join(r"c:\Users\samsu\Documents\arc-whest", "local_sim"))
import numpy as np
from mint_provenance import canonical_mint

def eff_rank(sv, frac):
    e = sv.astype(np.float64) ** 2
    c = np.cumsum(e)
    return int(np.searchsorted(c / c[-1], frac) + 1)

W, D = 1024, 16
seeds = [40_821_001_168, 40_821_001_169]
LAYERS = (1, 4, 7, 10, 13, 14)
print(f"{'layer':>6}{'r90(cov_post)':>15}{'r99(cov_post)':>15}"
      f"{'r90(cov_pre)':>14}{'r99(cov_pre)':>14}")
agg = {}
for seed in seeds:
    w, _e, _s = canonical_mint(W, D, seed)
    mu = np.zeros(W); cov = np.eye(W)
    off = np.ones((W, W)) - np.eye(W)
    for k in range(D):
        wl = w[k]
        mu_pre = wl.T @ mu
        cov_pre = wl.T @ cov @ wl
        var_pre = np.maximum(np.diagonal(cov_pre), 1e-12)
        sig = np.sqrt(var_pre); al = mu_pre / sig
        from scipy.stats import norm
        Phi = norm.cdf(al); phi = norm.pdf(al)
        mu = sig * (al * Phi + phi)
        ez2 = (var_pre + mu_pre * mu_pre) * Phi + mu_pre * sig * phi
        var = np.maximum(ez2 - mu * mu, 0.0)
        cov_post = np.outer(Phi, Phi) * cov_pre * off + np.diag(var)
        if k in LAYERS:
            svp = np.linalg.svd(cov_post, compute_uv=False)
            svr = np.linalg.svd(cov_pre, compute_uv=False)
            agg.setdefault(k, []).append(
                (eff_rank(svp, .90), eff_rank(svp, .99),
                 eff_rank(svr, .90), eff_rank(svr, .99)))
        cov = cov_post
for k in LAYERS:
    v = np.array(agg[k]).mean(axis=0)
    print(f"{k:>6}{v[0]:>15.0f}{v[1]:>15.0f}{v[2]:>14.0f}{v[3]:>14.0f}")
