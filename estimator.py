"""
ARC White-Box Estimation Challenge 2026 — submission estimator (v1.0).

Method: antithetic Monte Carlo in float32, with the first layer replaced by its
exact closed form.

Idea in one paragraph
---------------------
The quantity we are asked for, E[ReLU-activations] under x ~ N(0, I), is just an
expectation, so we can estimate it by sampling: draw N inputs, push them through
the network, average each layer's post-ReLU activations. That is unbiased by
construction, and its error falls as 1/N. Sampling in *float32* rather than
float64 halves the flopscope bill (measured: matmul is 2 FLOPs/MAC in f32 vs 4 in
f64) and costs nothing in accuracy, because the shipped weights are float32
values and the challenge's own ground truth was computed in float32. We draw
N/2 points and use each one twice, as x and -x ("antithetic pairs"), which cancels
the odd part of the integrand for free.

Layer 1 needs no sampling at all: z = W1^T x is *exactly* Gaussian with mean 0 and
variance ||W1[:, i]||^2, and E[ReLU(z)] for zero-mean Gaussian z is sigma/sqrt(2*pi).
We use the closed form for row 0 and sampling for the rest.

Choosing N (the operating point)
--------------------------------
Score = final_layer_MSE * max(0.1, C/B), and C is linear in N while MSE ~ 1/N.
Below C/B = 0.1 the multiplier is pinned at its floor, so error falls for free;
above it the two effects cancel and the score is flat. The optimum is therefore
the knee at C/B ~ 0.1, and we sit just past it (~0.15) for margin against
cost-model surprises. N is computed analytically from the budget — we never call
flopscope's introspection API inside predict(), because budget_summary_dict() is
O(op-history) and has been measured to inflate residual wall time ~157x
(forum thread 18092), which at lambda = 1e11 FLOP/s would cost far more than the
whole budget.

Grader compliance
-----------------
* Only `flopscope`, `flopscope.numpy`, `whestbench` and the Python stdlib.
* All randomness comes from `fnp.random.default_rng(mlp.seed)`, so repeated runs
  on the same MLP are bit-identical.
* No in-place mutation of grader arrays; no flopscope introspection in predict.
* Work is done in a few large chunks rather than many small ops: releasing big
  temporaries is billed as residual wall time (forum 18101).
"""

import flopscope.numpy as fnp
from whestbench import MLP, BaseEstimator

# --- operating point ------------------------------------------------------
# Fraction of the FLOP budget to spend. The score multiplier floor is 0.1, so
# anything below that is free; we sit just above the knee for safety margin.
_TARGET_CB = 0.15
_MAX_CB = 0.20

# Analytic float32 cost of one sample, in flopscope FLOPs (measured 2026-08-05):
#   matmul  2 FLOPs/MAC  -> depth * 2 * width^2
#   ReLU    0.5 FLOP/elt -> depth * 0.5 * width
#   RNG    16 FLOPs/elt  -> width  (standard_normal, float32)
#   plus per-layer accumulation of the running column sums (~1 FLOP/elt)
_RNG_COST_PER_ELT = 16.0

# Never take fewer than this many samples (a tiny run would be worse than useless)
# nor more than this (guards absurd budgets / memory).
_MIN_SAMPLES = 512
_MAX_SAMPLES = 1 << 22

# Rows per forward chunk. Large enough that per-op residual overhead is
# amortised, small enough that a chunk's activations stay well under the 4 GiB
# single-array cap (8192 x 256 float32 = 8 MiB).
_CHUNK = 8192


def _cost_per_sample(width: int, depth: int) -> float:
    matmul = depth * 2.0 * width * width
    relu = depth * 0.5 * width
    accum = depth * 1.0 * width
    rng = _RNG_COST_PER_ELT * width
    return matmul + relu + accum + rng


class Estimator(BaseEstimator):
    """Antithetic Monte Carlo (float32) with an exact first layer."""

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        width = mlp.width
        depth = len(mlp.weights)

        n = self._n_samples(width, depth, budget)
        if n < _MIN_SAMPLES:
            # Budget too small to sample meaningfully — fall back to the exact
            # first layer and zeros elsewhere rather than returning noise.
            rows = [self._exact_first_layer(mlp)]
            rows.extend(fnp.zeros(width) for _ in range(depth - 1))
            return fnp.stack(rows, axis=0)

        # float32 weights: identical values (the shipped weights are float32
        # upcast to float64), half the flopscope price.
        weights = [w.astype(fnp.float32) for w in mlp.weights]

        rng = fnp.random.default_rng(mlp.seed)
        totals = [fnp.zeros(width, dtype=fnp.float64) for _ in range(depth)]

        pairs = n // 2  # antithetic: each draw is used as +x and -x
        drawn = 0
        while drawn < pairs:
            b = _CHUNK // 2 if (pairs - drawn) > _CHUNK // 2 else (pairs - drawn)
            half = rng.standard_normal((b, width), dtype=fnp.float32)
            x = fnp.concatenate([half, -half], axis=0)
            for li, w in enumerate(weights):
                x = fnp.maximum(x @ w, fnp.float32(0.0))
                totals[li] = totals[li] + x.sum(axis=0, dtype=fnp.float64)
            drawn += b

        scale = 1.0 / float(2 * pairs)
        rows = [t * scale for t in totals]
        rows[0] = self._exact_first_layer(mlp)
        return fnp.stack(rows, axis=0)

    # ------------------------------------------------------------------ #
    def _n_samples(self, width: int, depth: int, budget) -> int:
        """Largest even sample count whose analytic cost fits the target share."""
        if budget is None or budget <= 0:
            return _MIN_SAMPLES
        per = _cost_per_sample(width, depth)
        if per <= 0:
            return _MIN_SAMPLES
        n = int(_TARGET_CB * float(budget) / per)
        n_cap = int(_MAX_CB * float(budget) / per)
        n = min(n, n_cap, _MAX_SAMPLES)
        return (n // 2) * 2

    # ------------------------------------------------------------------ #
    def _exact_first_layer(self, mlp: MLP):
        """E[ReLU(z_i)] with z = W1^T x exactly N(0, ||W1[:, i]||^2).

        For zero-mean Gaussian z with std s, E[ReLU(z)] = s / sqrt(2*pi).
        """
        w0 = mlp.weights[0].astype(fnp.float32)
        var = (w0 * w0).sum(axis=0, dtype=fnp.float64)
        return fnp.sqrt(var) * 0.3989422804014327  # 1/sqrt(2*pi)
