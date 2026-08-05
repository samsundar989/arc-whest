"""
Offline access to the REAL competition dataset (weights + 1e9-sample ground
truth), for experiments that need a ground-truth floor far below our target.

Why this exists
---------------
`harness.py` self-generates MLPs and computes Monte Carlo truth at 200k samples,
whose own measurement noise is ~5e-7 final-layer MSE. Phase A targets ~1e-6 MSE,
so that harness literally cannot see the signal. The public phase-1 dataset ships
ground truth computed at N = 1e9 samples per MLP (noise ~2e-10) — three orders of
magnitude below anything we are trying to measure.

Use this for every Phase-A/B measurement. Keep `harness.py` for quick sanity work
and for anything that needs MLPs the grader has never seen.
"""

import functools
import os
import sys

import numpy as np

DATASET = "aicrowd/arc-whestbench-public-2026"
REVISION = "v1-phase1"

_HERE = os.path.dirname(os.path.abspath(__file__))


def _real_whestbench():
    """Import the REAL whestbench, not the `local_sim/whestbench` shim.

    The shim exists so `estimator.py` can run offline; it has no `.dataset`
    submodule. If it (or this directory) is on the path first, drop it for the
    duration of the import and evict any shim already in sys.modules.
    """
    saved_path = list(sys.path)
    saved_mods = {k: v for k, v in sys.modules.items() if k == "whestbench" or k.startswith("whestbench.")}
    sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
    for k in saved_mods:
        del sys.modules[k]
    try:
        import whestbench.dataset as wd

        return wd
    finally:
        sys.path = saved_path
        # leave the real package imported; restoring the shim here would break
        # any dataset object we just handed out.


@functools.lru_cache(maxsize=4)
def _load(split: str = "mini"):
    return _real_whestbench().load_dataset(DATASET, revision=REVISION, split=split)


def n_mlps(split: str = "mini") -> int:
    return len(_load(split))


def get(index: int, split: str = "mini", dtype=np.float32):
    """Return (weights, truth_all_layers, meta) for one dataset MLP.

    weights : (depth, width, width) `dtype`, stored (input, output) — forward is
              h <- ReLU(h @ W), matching the challenge's `y = ReLU(W.T @ x)`.
    truth   : (depth, width) float64, ground-truth post-ReLU means (N = 1e9).
    meta    : dict with mlp_id, mlp_name, mlp_seed, avg_variance.
    """
    row = _load(split)[index]
    w = np.asarray(row["weights"], dtype=dtype)
    truth = np.asarray(row["all_layer_means"], dtype=np.float64)
    meta = {
        "mlp_id": row["mlp_id"],
        "mlp_name": row["mlp_name"],
        "mlp_seed": row["mlp_seed"],
        "avg_variance": row["avg_variance"],
    }
    return w, truth, meta


def forward(x, weights, collect=False):
    """Batched forward pass. x: (n, width). Returns final activations, or a
    (depth, width) array of per-layer column means when collect=True."""
    if collect:
        out = np.empty((len(weights), weights[0].shape[1]), dtype=np.float64)
    for li, w in enumerate(weights):
        x = np.maximum(x @ w, x.dtype.type(0.0))
        if collect:
            out[li] = x.mean(axis=0, dtype=np.float64)
    return out if collect else x


def as_mlp(index: int, split: str = "mini"):
    """The dataset row as a whestbench MLP object (for running estimator.py)."""
    return _real_whestbench().mlp_at(_load(split), index)


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    print(f"{n_mlps()} MLPs in the mini split")
    avgs = []
    for i in range(8):
        w, truth, meta = get(i)
        avgs.append(meta["avg_variance"])
        print(
            f"  [{i}] {meta['mlp_name']:<22} avg_variance={meta['avg_variance']:.5f}  "
            f"final mean={truth[-1].mean():.4f}  max={truth[-1].max():.3f}  "
            f"dead={(truth[-1] == 0).sum()}/{truth.shape[1]}"
        )
    print(f"mean avg_variance over 8: {np.mean(avgs):.5f}")
