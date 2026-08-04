"""
Local simulation harness: generate challenge-distribution MLPs, compute Monte
Carlo ground truth, and run the actual submission estimator against it.

The challenge's data distribution is fully known (He-init N(0, 2/n) float32
weights, square width x depth ReLU MLPs, N(0, I) inputs), so we can produce
unlimited private training/validation data — this is explicitly allowed and
uncounted by the FLOP budget (offline work).
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_HERE, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

from whestbench import MLP  # noqa: E402  (shim)


def gen_mlp(width: int, depth: int, seed: int) -> MLP:
    """He-Gaussian init N(0, 2/n), float32 — matches the starter kit's
    local_engine.build_mlp / whestbench.sample_mlp distribution."""
    rng = np.random.default_rng(seed)
    scale = np.float32(np.sqrt(2.0 / width))
    weights = [
        rng.standard_normal((width, width), dtype=np.float32) * scale
        for _ in range(depth)
    ]
    return MLP(width=width, depth=depth, weights=weights, seed=seed)


def mc_layer_means(mlp: MLP, n_samples: int, seed: int, batch: int = 50_000):
    """Monte Carlo per-neuron post-ReLU means for every layer.

    Forward pass: x <- ReLU(x @ W) (weights stored (input, output)).
    float32 math, float64 accumulation.
    Returns (depth, width) float64.
    """
    acc = np.zeros((mlp.depth, mlp.width), dtype=np.float64)
    rng = np.random.default_rng(seed)
    done = 0
    while done < n_samples:
        b = min(batch, n_samples - done)
        x = rng.standard_normal((b, mlp.width), dtype=np.float32)
        for li, w in enumerate(mlp.weights):
            x = np.maximum(x @ w, np.float32(0.0))
            acc[li] += x.sum(axis=0, dtype=np.float64)
        done += b
    return acc / float(n_samples)


def run_estimator(mlp: MLP, budget: int = 272_000_000_000):
    """Run the REAL submission file (estimator.py) through the shims."""
    import estimator as est_mod

    est = est_mod.Estimator()
    est.setup_called = False
    pred = est.predict(mlp, budget)
    return np.asarray(pred, dtype=np.float64)


def final_layer_mse(pred, truth):
    return float(np.mean((pred[-1] - truth[-1]) ** 2))


def all_layers_mse(pred, truth):
    return float(np.mean((pred - truth) ** 2))
