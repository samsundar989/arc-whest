"""
Evaluate any submission estimator against the REAL dataset's 1e9-sample ground
truth, through the local shims (fast: no FLOP counting).

This replaces the old self-generated-MLP + 200k-sample-MC route for anything at
v1 accuracy or better: that harness carried ~5e-7 of its own measurement noise,
which is the same size as the numbers we now care about. The shipped ground truth
has ~2e-10 noise instead.

Usage:
    uv run python local_sim/eval_estimator.py --estimator estimator_v1.py
    uv run python local_sim/eval_estimator.py --estimator estimator_v1.py --scale 0.25,0.5,1.0
        (--scale multiplies the budget, giving the error-vs-N curve)
"""

import argparse
import importlib.util
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# local_sim first so `flopscope` resolves to the shim (plain numpy, fast)
for p in (_HERE, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

import real_data as rd  # noqa: E402  (imports the REAL whestbench internally)

BUDGET = 272_000_000_000


def load_estimator(path):
    path = path if os.path.isabs(path) else os.path.join(_ROOT, path)
    name = "loaded_" + os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod.Estimator()


def _analytic_cb(est, budget, width=256, depth=32):
    """The estimator's own analytic C/B share, whichever sizing hook it exposes.

    This is the FLOP part only — the grader additionally bills residual wall
    time at 1e11 FLOP/s, which added ~17% on top for v0. `whest run` is the
    authoritative figure; this is for steering offline.
    """
    mod = sys.modules.get(est.__class__.__module__)
    cps = getattr(mod, "_cost_per_sample", None)
    if cps is None:
        return None
    if hasattr(est, "_n_samples"):
        n = est._n_samples(width, depth, budget)
    elif hasattr(est, "_log2_samples"):
        k = est._log2_samples(width, depth, budget)
        n = (1 << k) if k >= 0 else 0
    else:
        return None
    return n * cps(width, depth) / BUDGET


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimator", default="estimator_v1.py")
    ap.add_argument("--mlps", type=int, default=12)
    ap.add_argument("--scale", default="1.0", help="comma-separated budget multipliers")
    ap.add_argument("--repeat-seed-check", action="store_true", help="verify bit-identical reruns")
    args = ap.parse_args()

    est = load_estimator(args.estimator)
    scales = [float(s) for s in args.scale.split(",")]

    print(f"estimator: {args.estimator}   mlps: {args.mlps}   ground truth: 1e9-sample (real dataset)\n")
    curve = []
    for sc in scales:
        budget = int(BUDGET * sc)
        finals, alls, t0 = [], [], time.perf_counter()
        for i in range(args.mlps):
            w, truth, meta = rd.get(i, dtype=np.float64)
            mlp = rd.as_mlp(i)
            pred = np.asarray(est.predict(mlp, budget), dtype=np.float64)
            finals.append(float(((pred[-1] - truth[-1]) ** 2).mean()))
            alls.append(float(((pred - truth) ** 2).mean()))
            if args.repeat_seed_check and i == 0:
                again = np.asarray(est.predict(mlp, budget), dtype=np.float64)
                ident = bool(np.array_equal(pred, again))
                print(f"  determinism (bit-identical rerun on MLP 0): {ident}")
                if not ident:
                    raise SystemExit("NON-DETERMINISTIC — violates the seed contract")
        mse = float(np.mean(finals))
        cb = _analytic_cb(est, budget)
        curve.append((sc, budget, mse, cb))
        tag = f"C/B={cb:.4f} mult={max(0.1, cb):.4f}" if cb is not None else ""
        adj = mse * (max(0.1, cb) if cb is not None else 0.1)
        print(
            f"budget x{sc:<5} final-layer MSE = {mse:.4e}   all-layers = {float(np.mean(alls)):.4e}   "
            f"{tag}   adjusted = {adj:.3e}   ({time.perf_counter() - t0:.1f}s)"
        )

    if len(curve) > 1:
        print("\nerror-vs-N scaling")
        print(f"  {'budget x':>9} {'MSE':>12} {'ratio vs prev':>14}  (1/N scaling => ratio ~ the budget ratio)")
        for k, (sc, _, mse, _) in enumerate(curve):
            r = f"{curve[k - 1][2] / mse:.2f}x" if k else "-"
            print(f"  {sc:>9} {mse:>12.4e} {r:>14}")
        xs = np.log(np.array([c[0] for c in curve]))
        ys = np.log(np.array([c[2] for c in curve]))
        alpha = -np.polyfit(xs, ys, 1)[0]
        print(f"  fitted MSE ~ N^-alpha  =>  alpha = {alpha:.2f}   (1.0 = textbook Monte Carlo)")

    v0 = 2.67e-5
    best = curve[-1][2]
    print(f"\nvs v0 local held-out ({v0:.2e}): {v0 / best:.2f}x better")


if __name__ == "__main__":
    main()
