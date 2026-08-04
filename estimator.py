"""
ARC White-Box Estimation Challenge 2026 — submission estimator (v0).

Method: covariance propagation ("Gaussian closure", K=2) with an offline-fitted
per-layer scale calibration.

Idea in one paragraph
---------------------
The input x ~ N(0, I) is Gaussian, so the first layer's pre-activations
z = W^T x are *exactly* Gaussian with mean 0 and covariance W^T W. For a
Gaussian z_i ~ N(mu, s^2) the post-ReLU moments are closed-form:

    E[ReLU(z)]   = mu * Phi(mu/s) + s * phi(mu/s)
    E[ReLU(z)^2] = (mu^2 + s^2) * Phi(mu/s) + mu * s * phi(mu/s)

and for a jointly-Gaussian pair the post-ReLU covariance is, to first order in
the correlation, Cov_post[i,j] ~= Phi(a_i) * Phi(a_j) * Cov_pre[i,j]  (the
"gain" method: Phi(a) = E[ReLU'(z)] is each neuron's pass-through probability).
We *pretend* every layer's pre-activations are Gaussian (they aren't, quite —
that is the approximation), propagate (mean, covariance) layer by layer, and
read off the predicted per-neuron means. Cost ~ O(depth * width^3) FLOPs:
~3e9 for 256x32 in float64, i.e. ~1% of the phase-1 budget of 2.72e11 —
comfortably below the 10% floor of the score multiplier.

The Gaussian-closure bias is real but *systematic*: at width 256 it shows up
as a stable per-layer multiplicative drift of the predicted means (documented
publicly on the forum, thread 18063). We fitted per-layer correction factors
offline on self-generated He-init MLPs against high-sample Monte Carlo (see
local_sim/fit_calibration.py); they are baked in as _CAL_LAYER_SCALE below and
applied row-wise. Layer statistics do not depend on later layers, so factors
fitted at depth 32 apply to any depth <= 32 (prefix property); beyond 32 we
extrapolate the drift log-linearly.

Grader compliance
-----------------
* Only `flopscope`, `whestbench`, and the Python stdlib are imported.
* Deterministic: no randomness anywhere, so the mlp.seed contract is
  trivially satisfied.
* No in-place array mutation (grader arrays are immutable).
* Budget failsafe: for exotic shapes where full covariance propagation would
  not fit the budget, falls back to diagonal (mean) propagation, then zeros.
"""

import os

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP, BaseEstimator

# --------------------------------------------------------------------------
# Offline-fitted per-layer calibration (width 256, He init N(0, 2/n)).
# _CAL_LAYER_SCALE[k] multiplies the predicted means of layer k+1 (1-based
# layer k+1 = row k). Fitted by local_sim/fit_calibration.py; see
# experiments/calibration.json for provenance. None -> disabled (identity).
# --------------------------------------------------------------------------
_CAL_WIDTH = 256
# Fitted 2026-08-04 on 5 self-generated He-init 256x32 MLPs vs 200k-sample MC
# (local_sim/fit_calibration.py; provenance in experiments/calibration.json).
# Layer 1 pinned to exactly 1.0 (closed-form exact). Depth-8/16 fits agree with
# this table layer-by-layer within MC noise (prefix property verified).
_CAL_LAYER_SCALE = [
    1.0,      0.998414, 0.997486, 0.996835, 0.996253, 0.995831, 0.995492,
    0.994881, 0.994833, 0.994597, 0.994315, 0.994229, 0.993970, 0.993850,
    0.993429, 0.993266, 0.993268, 0.993313, 0.992924, 0.992802, 0.992655,
    0.992586, 0.992734, 0.992925, 0.992730, 0.992784, 0.992899, 0.993107,
    0.993026, 0.993091, 0.993310, 0.993142,
]


def _calibration_column(width, depth):
    """Per-row multiplicative factors, shape (depth,) as a python list."""
    if (
        _CAL_LAYER_SCALE is None
        or width != _CAL_WIDTH
        or os.environ.get("ARC_WHEST_DISABLE_CAL") == "1"
    ):
        return [1.0] * depth
    table = list(_CAL_LAYER_SCALE)
    if depth <= len(table):
        return table[:depth]
    # extrapolate: average log-drift of the last few fitted layers
    import math

    tail = table[-6:]
    if len(tail) >= 2 and all(t > 0 for t in tail):
        step = (math.log(tail[-1]) - math.log(tail[0])) / (len(tail) - 1)
    else:
        step = 0.0
    out = table[:]
    last = math.log(table[-1]) if table[-1] > 0 else 0.0
    for k in range(len(table), depth):
        last += step
        out.append(math.exp(last))
    return out


class Estimator(BaseEstimator):
    """Covariance propagation + offline per-layer calibration."""

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        n = mlp.width
        depth = len(mlp.weights)

        # ---- crude analytic cost estimates (float64 rate = 2x) ------------
        cov_cost = depth * (2.2 * n**3 + 40.0 * n * n) * 2.0
        mean_cost = depth * (30.0 * n * n) * 2.0

        if budget is not None and cov_cost > 0.5 * budget:
            if mean_cost > 0.5 * budget:
                return fnp.zeros((depth, n))
            rows = self._mean_propagation(mlp)
        else:
            rows = self._covariance_propagation(mlp)

        pred = fnp.stack(rows, axis=0)

        cal = _calibration_column(n, depth)
        if any(abs(c - 1.0) > 1e-12 for c in cal):
            factors = fnp.array(cal).reshape((depth, 1))
            pred = pred * factors
        return pred

    # ------------------------------------------------------------------ #
    def _covariance_propagation(self, mlp):
        """Track (mean vector, full covariance) through every layer."""
        n = mlp.width
        mu = fnp.zeros(n)
        cov = fnp.eye(n)  # input covariance: x ~ N(0, I)
        off_mask = fnp.ones((n, n)) - fnp.eye(n)  # zeros on the diagonal
        rows = []
        for w in mlp.weights:
            # exact linear-layer update: z = W^T h
            mu_pre = w.T @ mu
            cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)  # W^T Cov W
            var_pre = fnp.maximum(fnp.diagonal(cov_pre), 1e-12)
            sig = fnp.sqrt(var_pre)
            alpha = mu_pre / sig

            Phi = flops.stats.norm.cdf(alpha)  # pass-through probability
            phi = flops.stats.norm.pdf(alpha)

            # exact rectified-Gaussian marginal moments
            mu = mu_pre * Phi + sig * phi
            ez2 = (mu_pre * mu_pre + var_pre) * Phi + mu_pre * sig * phi
            var = fnp.maximum(ez2 - mu * mu, 0.0)

            # gain approximation off-diagonal, exact variance on the diagonal
            cov = fnp.outer(Phi, Phi) * cov_pre * off_mask + fnp.diag(var)
            if hasattr(flops, "as_symmetric"):
                try:
                    cov = flops.as_symmetric(cov, symmetry=(0, 1))
                except TypeError:
                    pass
            rows.append(mu)
        return rows

    # ------------------------------------------------------------------ #
    def _mean_propagation(self, mlp):
        """Cheap fallback: diagonal (independent-neuron) propagation."""
        n = mlp.width
        mu = fnp.zeros(n)
        var = fnp.ones(n)
        rows = []
        for w in mlp.weights:
            mu_pre = w.T @ mu
            var_pre = fnp.maximum((w * w).T @ var, 1e-12)
            sig = fnp.sqrt(var_pre)
            alpha = mu_pre / sig
            Phi = flops.stats.norm.cdf(alpha)
            phi = flops.stats.norm.pdf(alpha)
            mu = mu_pre * Phi + sig * phi
            ez2 = (mu_pre * mu_pre + var_pre) * Phi + mu_pre * sig * phi
            var = fnp.maximum(ez2 - mu * mu, 0.0)
            rows.append(mu)
        return rows
