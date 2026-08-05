"""
NumPy reference for the v0 covariance-propagation estimator, instrumented to
also return the per-neuron quantities the B1 fusion needs.

The math here is a line-for-line mirror of `experiments/estimator_v0_archive.py`
(Gaussian closure with the gain off-diagonal + exact rectified-Gaussian diagonal,
then the offline per-layer calibration). `verify_matches_v0()` asserts the two
agree, so anything fitted here is fitted to what the shipped estimator computes.

Per-MLP outputs at the FINAL layer (the scored one):
    p     calibrated covprop prediction of E[ReLU]        <- the analytic estimate
    mu_pre, sig  pre-activation mean / std
    alpha = mu_pre / sig                                  <- firing-probability index
    Phi   = P(neuron fires)
    var   post-ReLU variance  Var(h_i)                    <- drives sampling noise
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from scipy.special import ndtr as _ndtr  # noqa: E402

_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def _npdf(x):
    return _INV_SQRT_2PI * np.exp(-0.5 * x * x)


# v0's baked-in per-layer calibration (fitted at width 256).
_CAL_WIDTH = 256
_CAL_LAYER_SCALE = [
    1.0, 0.998414, 0.997486, 0.996835, 0.996253, 0.995831, 0.995492,
    0.994881, 0.994833, 0.994597, 0.994315, 0.994229, 0.993970, 0.993850,
    0.993429, 0.993266, 0.993268, 0.993313, 0.992924, 0.992802, 0.992655,
    0.992586, 0.992734, 0.992925, 0.992730, 0.992784, 0.992899, 0.993107,
    0.993026, 0.993091, 0.993310, 0.993142,
]


def calibration_column(width, depth):
    if width != _CAL_WIDTH:
        return np.ones(depth)
    table = list(_CAL_LAYER_SCALE)
    if depth <= len(table):
        return np.array(table[:depth])
    tail = table[-6:]
    step = (np.log(tail[-1]) - np.log(tail[0])) / (len(tail) - 1)
    out = list(table)
    last = np.log(table[-1])
    for _ in range(len(table), depth):
        last += step
        out.append(float(np.exp(last)))
    return np.array(out)


def covprop(weights, calibrate=True):
    """Full Gaussian-closure propagation. weights: (depth, n, n) float64.

    Returns dict with `rows` (depth, n) predictions plus final-layer diagnostics.
    """
    w0 = np.asarray(weights[0], dtype=np.float64)
    n = w0.shape[1]
    depth = len(weights)

    mu = np.zeros(n)
    cov = np.eye(n)
    off_mask = np.ones((n, n)) - np.eye(n)
    rows = []
    diag = {}
    for li in range(depth):
        w = np.asarray(weights[li], dtype=np.float64)
        mu_pre = w.T @ mu
        cov_pre = w.T @ cov @ w
        var_pre = np.maximum(np.diag(cov_pre), 1e-12)
        sig = np.sqrt(var_pre)
        alpha = mu_pre / sig

        Phi = _ndtr(alpha)
        phi = _npdf(alpha)

        mu = mu_pre * Phi + sig * phi
        ez2 = (mu_pre * mu_pre + var_pre) * Phi + mu_pre * sig * phi
        var = np.maximum(ez2 - mu * mu, 0.0)

        cov = np.outer(Phi, Phi) * cov_pre * off_mask + np.diag(var)
        cov = 0.5 * (cov + cov.T)
        rows.append(mu.copy())
        if li == depth - 1:
            diag = dict(mu_pre=mu_pre, sig=sig, alpha=alpha, Phi=Phi, var=var)

    pred = np.stack(rows, axis=0)
    if calibrate:
        pred = pred * calibration_column(n, depth)[:, None]
    diag["rows"] = pred
    diag["p"] = pred[-1]
    return diag


def verify_matches_v0(weights, atol=1e-10):
    """Assert this reference reproduces the archived v0 estimator exactly."""
    import importlib.util

    root = os.path.dirname(_HERE)
    for p in (_HERE, root):
        if p not in sys.path:
            sys.path.insert(0, p)
    path = os.path.join(root, "experiments", "estimator_v0_archive.py")
    spec = importlib.util.spec_from_file_location("v0_archive", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v0_archive"] = mod
    spec.loader.exec_module(mod)

    class _MLP:
        def __init__(self, w):
            self.width = w[0].shape[1]
            self.weights = list(w)
            self.seed = 0

    ref = np.asarray(mod.Estimator().predict(_MLP(weights), 272_000_000_000), dtype=np.float64)
    mine = covprop(weights)["rows"]
    err = float(np.abs(ref - mine).max())
    assert err < atol, f"covprop reference diverges from v0 by {err:.3e}"
    return err


if __name__ == "__main__":
    import real_data as rd

    w, truth, meta = rd.get(0, dtype=np.float64)
    err = verify_matches_v0(w)
    print(f"reference matches estimator_v0_archive.py to {err:.3e} (max abs)")

    d = covprop(w)
    t = truth[-1]
    c = d["p"] - t
    print(f"\nMLP 0 ({meta['mlp_name']}) final layer:")
    print(f"  covprop MSE          {np.mean(c**2):.4e}")
    print(f"  mean |residual|      {np.abs(c).mean():.4e}")
    print(f"  corr(residual, t)    {np.corrcoef(c, t)[0, 1]:+.3f}")
    print(f"  common-mode share    {(c @ (t / np.linalg.norm(t))) ** 2 / (c @ c):.3f}")
    print(f"  alpha range          [{d['alpha'].min():.2f}, {d['alpha'].max():.2f}]")
    print(f"  post-ReLU var mean   {d['var'].mean():.4f}  (shipped avg_variance {meta['avg_variance']:.4f})")
