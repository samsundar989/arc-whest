"""
Collect everything the B1 fusion needs, per MLP, on a whole dataset split.

For each MLP we store the FINAL-layer vectors (256 each):
    t     ground truth (1e9-sample, so effectively noiseless labels)
    p     calibrated covariance-propagation prediction
    s     the v1.1 Sobol-RQMC sampling estimate (exact same points the shipped
          estimator would draw: same N, same digital shift from mlp.seed)
    sig, alpha, Phi, var   covprop's in-flight diagnostics at the final layer

Everything downstream (error models, fusion weights, diagnostics) is fitted from
this cache, so the fit sees exactly what `predict()` will see at grading time.

Run:
    uv run python local_sim/collect_fusion_data.py --split full   (1000 MLPs)
    uv run python local_sim/collect_fusion_data.py --split mini   (100 MLPs)
"""

import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_HERE, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import real_data as rd  # noqa: E402
from covprop_features import covprop  # noqa: E402

BUDGET = 272_000_000_000
OUT_DIR = os.path.join(_ROOT, "experiments", "fusion_data")


def load_sampler(path="estimator_v1_1.py"):
    import importlib.util

    spec = importlib.util.spec_from_file_location("v1_1_for_fit", os.path.join(_ROOT, path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v1_1_for_fit"] = mod
    spec.loader.exec_module(mod)
    return mod.Estimator()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="full")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--sampler", default="estimator_v1_1.py")
    args = ap.parse_args()

    est = load_sampler(args.sampler)
    n_total = rd.n_mlps(args.split)
    n = args.limit if args.limit else n_total
    n = min(n, n_total)

    keys = ("t", "p", "s", "sig", "alpha", "Phi", "var")
    acc = {k: [] for k in keys}
    seeds, names = [], []

    t0 = time.perf_counter()
    for i in range(n):
        w, truth, meta = rd.get(i, split=args.split, dtype=np.float64)
        mlp = rd.as_mlp(i, split=args.split)

        d = covprop(w)
        s = np.asarray(est.predict(mlp, BUDGET), dtype=np.float64)[-1]

        acc["t"].append(truth[-1])
        acc["p"].append(d["p"])
        acc["s"].append(s)
        acc["sig"].append(d["sig"])
        acc["alpha"].append(d["alpha"])
        acc["Phi"].append(d["Phi"])
        acc["var"].append(d["var"])
        seeds.append(meta["mlp_seed"])
        names.append(meta["mlp_name"])

        if (i + 1) % 25 == 0 or i + 1 == n:
            el = time.perf_counter() - t0
            print(f"  {i + 1}/{n}   {el:6.1f}s   eta {el / (i + 1) * (n - i - 1):6.1f}s", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{args.split}_fusion_data.npz")
    np.savez_compressed(
        out,
        **{k: np.stack(acc[k], axis=0) for k in keys},
        seed=np.array(seeds, dtype=np.int64),
        name=np.array(names),
    )
    print(f"\nsaved {out}  ({os.path.getsize(out) / 1e6:.1f} MB, {n} MLPs)")

    T = np.stack(acc["t"])
    P = np.stack(acc["p"])
    S = np.stack(acc["s"])
    print(f"  covprop  MSE {np.mean((P - T) ** 2):.4e}")
    print(f"  sampling MSE {np.mean((S - T) ** 2):.4e}")


if __name__ == "__main__":
    main()
