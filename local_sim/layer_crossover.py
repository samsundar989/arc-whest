"""
B2's key input: at which DEPTH does covariance propagation stop beating sampling?

B2 proposes pushing the last k layers analytically. That only pays where the
analytic error is smaller than the sampling noise it replaces, so the honest way
to pick k is to measure both estimators' error at every layer against the
shipped 1e9-sample truth (which exists for all 32 layers, not just the scored one).

Also produces the "fusion-hurt map": per layer and per alpha band, whether the
precision blend helped or hurt relative to sampling alone.

Run: uv run python local_sim/layer_crossover.py [--split full] [--mlps 200]
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


def load_sampler():
    import importlib.util

    spec = importlib.util.spec_from_file_location("v11_x", os.path.join(_ROOT, "estimator_v1_1.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v11_x"] = mod
    spec.loader.exec_module(mod)
    return mod.Estimator()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="full")
    ap.add_argument("--mlps", type=int, default=200)
    ap.add_argument("--offset", type=int, default=800, help="start index (use the fit holdout)")
    args = ap.parse_args()

    est = load_sampler()
    n = min(args.mlps, rd.n_mlps(args.split) - args.offset)
    P, S, T = [], [], []
    for j in range(n):
        i = args.offset + j
        w, truth, _ = rd.get(i, split=args.split, dtype=np.float64)
        mlp = rd.as_mlp(i, split=args.split)
        P.append(covprop(w)["rows"])
        S.append(np.asarray(est.predict(mlp, BUDGET), dtype=np.float64))
        T.append(truth)
        if (j + 1) % 50 == 0:
            print(f"  {j + 1}/{n}", flush=True)
    P, S, T = np.stack(P), np.stack(S), np.stack(T)
    depth = T.shape[1]
    print(f"\n{n} MLPs (held out from the fusion fit), depth {depth}\n")

    print(f"{'layer':>6} {'covprop MSE':>13} {'sampling MSE':>14} {'ratio cov/smp':>14} {'winner':>9}")
    print("-" * 62)
    cross = None
    for k in range(depth):
        mc = float(((P[:, k] - T[:, k]) ** 2).mean())
        ms = float(((S[:, k] - T[:, k]) ** 2).mean())
        win = "covprop" if mc < ms else "sampling"
        if cross is None and mc > ms:
            cross = k
        if k < 12 or k >= depth - 8 or k % 4 == 0:
            print(f"{k + 1:>6} {mc:>13.4e} {ms:>14.4e} {mc / ms:>14.3g} {win:>9}")
    print(f"\ncovprop stops winning at layer {cross + 1 if cross is not None else 'never'} (1-based)")

    # ---- how far back would we have to start to make the last k analytic? ---
    print("\nIf the last k layers were replaced by EXACT analytic propagation from")
    print("a perfectly known layer (L-k), the error at L would be at least the")
    print("closure error accumulated over those k layers. Lower bound per k:")
    print(f"  {'k':>3} {'covprop error from scratch':>28} {'sampling error at L':>21}")
    ms_L = float(((S[:, -1] - T[:, -1]) ** 2).mean())
    for k in (1, 2, 3, 4, 6, 8):
        mc_k = float(((P[:, depth - 1] - T[:, depth - 1]) ** 2).mean())
        print(f"  {k:>3} {mc_k:>28.4e} {ms_L:>21.4e}")
        break  # single line: full-depth covprop is the only measurement we have

    # ---- fusion-hurt map on the final layer --------------------------------
    d = dict(np.load(os.path.join(_ROOT, "experiments", "fusion_data", "full_fusion_data.npz"), allow_pickle=True))
    sl = slice(args.offset, args.offset + n)
    alpha, sig, var = d["alpha"][sl], d["sig"][sl], d["var"][sl]
    p, s, t = d["p"][sl], d["s"][sl], d["t"][sl]
    vc_g = float(((p - t) ** 2).mean())
    vs_g = float(((s - t) ** 2).mean())
    w = vs_g / (vs_g + vc_g)
    blend = w * p + (1 - w) * s
    err_b = (blend - t) ** 2
    err_s = (s - t) ** 2

    print(f"\nfusion-hurt map (global blend w={w:.4f}), final layer, by alpha band")
    print(f"  {'alpha band':>15} {'n':>8} {'sampling MSE':>14} {'blended MSE':>13} {'change':>9}")
    edges = np.array([-np.inf, -3, -2, -1, 0, 1, 2, 3, np.inf])
    a = alpha.ravel()
    for k in range(len(edges) - 1):
        m = (a >= edges[k]) & (a < edges[k + 1])
        if m.sum() < 100:
            continue
        es_, eb_ = float(err_s.ravel()[m].mean()), float(err_b.ravel()[m].mean())
        print(
            f"  [{edges[k]:>5.1f},{edges[k + 1]:>5.1f}) {m.sum():>8} {es_:>14.4e} {eb_:>13.4e} "
            f"{100 * (1 - eb_ / es_):>+8.1f}%"
        )


if __name__ == "__main__":
    main()
