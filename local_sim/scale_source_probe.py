"""
Where does the sampling run's common scale fluctuation COME FROM, and what is
the best online predictor of it?

Candidates tested, all computable inside predict():
  r2dev    realized radial imbalance of the drawn point set,
           mean(||x||^2)/width - 1   -- if this drives g, the fix is a free rescale
  g1       sampled layer-1 scale deviation against the EXACT closed-form mean
           (zero contamination: we know layer 1 analytically)
  ghat_k   <s_k - p_k, p_k>/<p_k,p_k> at layer k against covprop

Reports leave-one-out cross-validated R^2 for predicting the final-layer scale
g_L, so nothing here is in-sample optimism, and the MSE each predictor would
actually deliver.

Run: uv run python local_sim/scale_source_probe.py [--split mini] [--mlps 100]
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_HERE, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import real_data as rd  # noqa: E402
from covprop_features import covprop  # noqa: E402

BUDGET = 272_000_000_000
F32 = np.float32


def sobol_points_for(mlp, n_expect=16384):
    """Regenerate exactly the point set estimator_v1_1 would draw for this MLP."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("v11_src", os.path.join(_ROOT, "estimator_v1_1.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v11_src"] = mod
    spec.loader.exec_module(mod)

    width = mlp.width
    depth = len(mlp.weights)
    k = mod.Estimator()._log2_samples(width, depth, BUDGET)
    n = 1 << k
    rng = np.random.default_rng(mlp.seed)
    shift = rng.integers(0, 1 << mod._SOBOL_BITS, size=width, dtype=np.uint32)
    dirnum = mod._direction_numbers()
    vt = np.array([dirnum[b][:width] for b in range(mod._SOBOL_BITS)], dtype=np.uint32)

    idx = np.arange(n, dtype=np.uint32)
    idx = idx ^ (idx >> np.uint32(1))
    idx = idx[:, None]
    pts = np.zeros((n, width), dtype=np.uint32)
    for b in range(mod._SOBOL_BITS):
        bit = (idx >> np.uint32(b)) & np.uint32(1)
        pts = np.bitwise_xor(pts, bit * vt[b])
    pts = np.bitwise_xor(pts, shift)
    u = (pts.astype(np.float64) + 0.5) / float(1 << mod._SOBOL_BITS)
    from scipy.special import ndtri

    return ndtri(u).astype(F32), n


def loo_r2(X, y):
    """Leave-one-out cross-validated R^2 for OLS (hat-matrix shortcut)."""
    X = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    loo = resid / np.maximum(1 - h, 1e-12)
    return 1 - (loo**2).mean() / y.var(), beta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="mini")
    ap.add_argument("--mlps", type=int, default=100)
    args = ap.parse_args()

    n_mlps = min(args.mlps, rd.n_mlps(args.split))
    rows = []
    T_last, S_last = [], []
    for i in range(n_mlps):
        w, truth, _ = rd.get(i, split=args.split, dtype=np.float64)
        mlp = rd.as_mlp(i, split=args.split)
        x, n = sobol_points_for(mlp)

        # radial imbalance of the realised point set
        r2dev = float((x.astype(np.float64) ** 2).sum(axis=1).mean() / mlp.width - 1.0)

        # forward pass, keeping every layer's sampled mean
        wl = [np.asarray(ww, dtype=F32) for ww in mlp.weights]
        h = x
        s_layers = []
        for ww in wl:
            h = np.maximum(h @ ww, F32(0))
            s_layers.append(h.mean(axis=0, dtype=np.float64))
        s_layers = np.stack(s_layers)

        P = covprop(w)["rows"]
        # exact layer-1 mean: sigma_i/sqrt(2pi)
        w0 = np.asarray(mlp.weights[0], dtype=np.float64)
        mu1_exact = np.sqrt((w0 * w0).sum(axis=0)) / np.sqrt(2 * np.pi)
        g1 = float(((s_layers[0] - mu1_exact) * mu1_exact).sum() / (mu1_exact * mu1_exact).sum())

        ghat = np.array(
            [
                float(((s_layers[k] - P[k]) * P[k]).sum() / max((P[k] * P[k]).sum(), 1e-300))
                for k in range(len(wl))
            ]
        )
        gL = float(((s_layers[-1] - truth[-1]) * truth[-1]).sum() / (truth[-1] ** 2).sum())

        rows.append((r2dev, g1, ghat, gL))
        T_last.append(truth[-1])
        S_last.append(s_layers[-1])
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{n_mlps}", flush=True)

    r2dev = np.array([r[0] for r in rows])
    g1 = np.array([r[1] for r in rows])
    ghat = np.stack([r[2] for r in rows])
    gL = np.array([r[3] for r in rows])
    T_last = np.stack(T_last)
    S_last = np.stack(S_last)
    depth = ghat.shape[1]

    print(f"\n{n_mlps} MLPs; sd(g_L) = {gL.std():.5f}\n")
    print(f"{'predictor':<44} {'corr':>8} {'LOO R^2':>9}")
    print("-" * 64)

    def report(label, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, None]
        r2, _ = loo_r2(X, gL)
        c = np.corrcoef(X[:, 0], gL)[0, 1] if X.shape[1] == 1 else np.nan
        print(f"{label:<44} {c:>8.3f} {r2:>9.3f}")
        return r2

    report("radial imbalance mean||x||^2/n - 1", r2dev)
    report("layer-1 exact-reference scale g1", g1)
    report("ghat at layer 12", ghat[:, 11])
    report("ghat layers 1-8", ghat[:, :8])
    report("ghat layers 1-16", ghat[:, :16])
    report("ghat layers 1-24", ghat[:, :24])
    report("ghat all 32 layers", ghat)
    report("mean of ghat over layers 8-20", ghat[:, 7:20].mean(axis=1))
    report("mean ghat 8-20 + g1", np.stack([ghat[:, 7:20].mean(axis=1), g1], axis=1))
    report("mean ghat 4-12 + mean ghat 12-24", np.stack([ghat[:, 3:12].mean(axis=1), ghat[:, 11:24].mean(axis=1)], axis=1))

    # --- MSE delivered by the best low-dimensional predictors ---------------
    resid = S_last - T_last
    mse0 = float((resid**2).mean())
    print(f"\nfinal-layer sampling MSE                     {mse0:.4e}")
    print(f"  oracle (g_L known)                         {float(((resid - gL[:, None] * T_last) ** 2).mean()):.4e}")
    for label, X in (
        ("mean ghat 8-20", ghat[:, 7:20].mean(axis=1)[:, None]),
        ("ghat layers 1-16", ghat[:, :16]),
        ("mean ghat 8-20 + g1", np.stack([ghat[:, 7:20].mean(axis=1), g1], axis=1)),
    ):
        Xa = np.concatenate([X, np.ones((len(X), 1))], axis=1)
        beta = np.linalg.pinv(Xa.T @ Xa) @ (Xa.T @ gL)
        pred = Xa @ beta
        # use the estimator's OWN prediction p as the direction (truth unavailable online)
        corrected = resid - pred[:, None] * S_last
        print(f"  corrected via {label:<28} {float((corrected**2).mean()):.4e}")


if __name__ == "__main__":
    main()
