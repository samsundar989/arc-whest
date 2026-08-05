"""
Can we OBSERVE the sampling run's common scale fluctuation at predict time?

47.9% of v1.1's final-layer error is one scalar: the whole layer scaled by 1+g.
Killing it needs an estimate of g built only from things predict() can see.

Reading g off the FINAL layer against covprop does not work — covprop's own
final-layer scale error has sd 0.0028 versus the sampling scale's 0.0009, so the
regression measures covprop's error, not ours.

But covprop's error COMPOUNDS WITH DEPTH: at early layers it is tiny, while the
sampling run's scale fluctuation is a property of the drawn point set and should
be visible from layer 1 onward. So the observable

    g_hat_k = <s_k - p_k, p_k> / <p_k, p_k>        (layer k, both computable online)

may estimate the final layer's g_L using an EARLY k, where p_k is trustworthy.
Layer 1 is the extreme case: there we know the answer in closed form exactly, so
g_hat_1 is contaminated by nothing at all.

This script measures, per layer k:
  * corr(g_k, g_L)      — does the scale fluctuation persist across depth?
  * corr(g_hat_k, g_L)  — is the ONLINE observable predictive?
  * the achievable MSE reduction from regressing g_L on g_hat_k

Run: uv run python local_sim/scale_mode_probe.py [--mlps 100] [--split mini]
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


def load_sampler(path="estimator_v1_1.py"):
    import importlib.util

    spec = importlib.util.spec_from_file_location("v11_probe", os.path.join(_ROOT, path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v11_probe"] = mod
    spec.loader.exec_module(mod)
    return mod.Estimator()


def scale_of(a, ref):
    """<a, ref> / <ref, ref> per row."""
    return (a * ref).sum(axis=1) / np.maximum((ref * ref).sum(axis=1), 1e-300)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="mini")
    ap.add_argument("--mlps", type=int, default=100)
    args = ap.parse_args()

    est = load_sampler()
    n = min(args.mlps, rd.n_mlps(args.split))
    S, P, T = [], [], []
    for i in range(n):
        w, truth, _ = rd.get(i, split=args.split, dtype=np.float64)
        mlp = rd.as_mlp(i, split=args.split)
        S.append(np.asarray(est.predict(mlp, BUDGET), dtype=np.float64))
        P.append(covprop(w)["rows"])
        T.append(truth)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{n}", flush=True)
    S, P, T = np.stack(S), np.stack(P), np.stack(T)  # (m, depth, n)
    m, depth, width = T.shape
    print(f"\n{m} MLPs, depth {depth}, width {width}\n")

    # true per-layer sampling scale mode, and the online observable
    g_true = np.stack([scale_of(S[:, k] - T[:, k], T[:, k]) for k in range(depth)], axis=1)
    g_hat = np.stack([scale_of(S[:, k] - P[:, k], P[:, k]) for k in range(depth)], axis=1)
    gL = g_true[:, -1]

    print(f"{'layer':>6} {'sd(g_k)':>10} {'corr(g_k,g_L)':>14} {'corr(ghat_k,g_L)':>17} {'MSE red. if used':>18}")
    print("-" * 70)
    varL = gL.var()
    for k in list(range(0, 8)) + [11, 15, 19, 23, 27, 30, 31]:
        c_true = np.corrcoef(g_true[:, k], gL)[0, 1]
        c_hat = np.corrcoef(g_hat[:, k], gL)[0, 1]
        # regressing gL on g_hat_k removes a fraction corr^2 of var(gL)
        red = c_hat**2
        print(f"{k + 1:>6} {g_true[:, k].std():>10.5f} {c_true:>14.3f} {c_hat:>17.3f} {100 * red:>17.1f}%")

    # --- best single predictor, and a multi-layer combination ---------------
    best_k = int(np.argmax([abs(np.corrcoef(g_hat[:, k], gL)[0, 1]) for k in range(depth)]))
    print(f"\nbest single layer for the observable: layer {best_k + 1}")

    # multi-layer least squares (how much better than the best single layer?)
    for cols in ([0], [best_k], [0, 1, 2, 3], list(range(0, 8)), list(range(0, 16))):
        X = np.concatenate([g_hat[:, cols], np.ones((m, 1))], axis=1)
        beta, *_ = np.linalg.lstsq(X, gL, rcond=None)
        pred = X @ beta
        r2 = 1 - ((gL - pred) ** 2).mean() / varL
        print(f"  layers {str([c + 1 for c in cols])[:34]:<34} R^2(gL) = {r2:.3f}")

    # --- what the scale kill would be worth on the final layer --------------
    tL = T[:, -1]
    resid = S[:, -1] - tL
    mse0 = float((resid**2).mean())
    print(f"\nfinal-layer sampling MSE                {mse0:.4e}")
    print(f"  if g_L known exactly (oracle)         {float(((resid - gL[:, None] * tL) ** 2).mean()):.4e}")
    for cols in ([0], [0, 1, 2, 3], list(range(0, 8))):
        X = np.concatenate([g_hat[:, cols], np.ones((m, 1))], axis=1)
        beta, *_ = np.linalg.lstsq(X, gL, rcond=None)
        ghat = X @ beta
        corrected = resid - ghat[:, None] * tL
        print(f"  using layers {str([c + 1 for c in cols])[:22]:<22} {float((corrected**2).mean()):.4e}")


if __name__ == "__main__":
    main()
