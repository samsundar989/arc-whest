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
    """A grader-identical MLP for input seed `seed`.

    Delegates to `mint_provenance.canonical_mint`, which is proven bit-identical
    to `whestbench.sample_mlp` driven by `whestbench.seeds.derive_seed_streams`
    (100/100 across 5 shapes x 20 seeds — see `mint_provenance.verify()` and
    docs/VALIDATION.md). `mlp.seed` therefore carries the evaluator's INDEPENDENT
    estimator substream, not the weight seed.
    """
    from mint_provenance import canonical_mint

    weights, estimator_seed, _sample_ss = canonical_mint(width, depth, seed)
    return MLP(width=width, depth=depth, weights=weights, seed=estimator_seed)


def gen_mlp_legacy(width: int, depth: int, seed: int) -> MLP:
    """The pre-2026-08-06 generator. Same distribution, different realisation,
    and it aliased `mlp.seed` to the weight seed.

    Kept only to reproduce historical numbers; never use it for new decisions.
    """
    rng = np.random.default_rng(seed)
    scale = np.float32(np.sqrt(2.0 / width))
    weights = [
        rng.standard_normal((width, width), dtype=np.float32) * scale
        for _ in range(depth)
    ]
    return MLP(width=width, depth=depth, weights=weights, seed=seed)


def gt_rng(seed: int):
    """The evaluator's ground-truth sampling substream for input seed `seed`.

    Using it means our MC labels are drawn from the same stream the grader would
    have used, so a vault MLP's truth is reproducible from its seed alone.
    """
    from mint_provenance import seed_streams

    return np.random.default_rng(seed_streams(seed)[1])


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
