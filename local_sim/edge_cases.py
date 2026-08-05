"""
Contract and numerical-edge-case suite for the SHIPPED estimator file.

`robustness.py` measures the score distribution using a GPU mirror of the
pipeline. This runs the actual `estimator.py` through the flopscope/whestbench
shims, because the things that turn into a zero-scored MLP on the grader —
an exception, a NaN, a shape violation, a non-deterministic rerun — live in the
real file, not in the mirror.

Cases, each chosen to hit a specific line rather than to be generically nasty:

  degenerate pruning   every neuron in a layer below tau  -> the argmax fallback
  tiny sigma           near-zero pre-activation variance  -> alpha = mu/sigma
  extreme alpha        saturated layers                   -> norm.cdf/pdf tails
  wide dynamic range   one layer scaled 100x / 0.01x      -> f32 overflow paths
  seed extremes        seed 0, 1, 2**63-1                 -> rng contract
  shape generality     width 64/128, depth 2/8            -> off-calibration path
  tiny budget          budget forcing log2n below the floor -> analytic fallback

Run: uv run python local_sim/edge_cases.py
"""

import importlib.util
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_HERE, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

BUDGET = 272_000_000_000


def load(path="estimator.py"):
    full = os.path.join(_ROOT, path)
    spec = importlib.util.spec_from_file_location("shipped_est", full)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["shipped_est"] = mod
    spec.loader.exec_module(mod)
    return mod


class MLP:
    def __init__(self, weights, seed=0):
        self.weights = [np.asarray(w, dtype=np.float32) for w in weights]
        self.width = self.weights[0].shape[1]
        self.seed = seed


def he(depth, width, seed, scale=None):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((depth, width, width)).astype(np.float32) * np.float32(np.sqrt(2.0 / width))
    if scale is not None:
        w = w.copy()
        for li, s in scale.items():
            w[li] = w[li] * np.float32(s)
    return w


def check(name, est, mlp, budget=BUDGET, expect_depth=None, rerun=True):
    try:
        out = est.predict(mlp, budget)
    except Exception as e:  # a raise on the grader is a zero-scored MLP
        print(f"  {name:<34} RAISED {type(e).__name__}: {e}")
        return False
    a = np.asarray(out, dtype=np.float64)
    depth = expect_depth if expect_depth is not None else len(mlp.weights)
    ok = True
    if a.shape != (depth, mlp.width):
        print(f"  {name:<34} BAD SHAPE {a.shape}, expected {(depth, mlp.width)}")
        ok = False
    if not np.isfinite(a).all():
        print(f"  {name:<34} NON-FINITE ({np.sum(~np.isfinite(a))} entries)")
        ok = False
    if (a < 0).any():
        print(f"  {name:<34} NEGATIVE MEANS (min {a.min():.3e}) — post-ReLU means cannot be < 0")
        ok = False
    ident = True
    if rerun and ok:
        b = np.asarray(est.predict(mlp, budget), dtype=np.float64)
        ident = bool(np.array_equal(a, b))
        if not ident:
            print(f"  {name:<34} NON-DETERMINISTIC (max diff {np.abs(a - b).max():.3e})")
            ok = False
    if ok:
        print(f"  {name:<34} ok   shape {a.shape}  range [{a.min():.3e}, {a.max():.3e}]  "
              f"final mean {a[-1].mean():.4f}  deterministic={ident}")
    return ok


def main():
    est_mod = load()
    est = est_mod.Estimator()

    class Ctx:
        width, depth, flop_budget, seed = 256, 32, BUDGET, 0

    est.setup(Ctx())
    print(f"shipped estimator: tau={est_mod._PRUNE_TAU}  target C/B={est_mod._TARGET_CB}  "
          f"chunk={est_mod._CHUNK}\n")

    results = []

    print("1. NORMAL CASE")
    results.append(check("he 256x32", est, MLP(he(32, 256, 1), seed=7)))

    print("\n2. DEGENERATE PRUNING (whole layers pushed below tau)")
    # A strongly negative-biased layer drives the next layer's alpha very low.
    for s, label in ((0.02, "layer 5 scaled 0.02"), (0.001, "layer 5 scaled 0.001")):
        results.append(check(label, est, MLP(he(32, 256, 2, {5: s}), seed=11)))
    w = he(32, 256, 3)
    w[7] = (w[7] * np.float32(0.0)).copy()  # an exactly zero layer: sigma -> 0
    results.append(check("layer 7 identically zero", est, MLP(w, seed=13)))

    print("\n3. EXTREME DYNAMIC RANGE")
    for s, label in ((100.0, "layer 12 scaled 100x"), (1e-3, "layer 12 scaled 1e-3"),
                     (1e4, "layer 12 scaled 1e4")):
        results.append(check(label, est, MLP(he(32, 256, 4, {12: s}), seed=17)))
    w = he(32, 256, 5)
    w = w.copy()
    w[3] = w[3] + np.float32(2.0) * np.float32(np.sqrt(2.0 / 256))  # strong positive bias
    results.append(check("layer 3 mean-shifted (all on)", est, MLP(w, seed=19)))

    print("\n4. SEED EXTREMES (the rng contract)")
    base = he(32, 256, 6)
    for sd in (0, 1, 2**31 - 1, 2**63 - 1, 12345678901234567890 % (2**64)):
        results.append(check(f"seed {sd}", est, MLP(base, seed=sd)))

    print("\n5. SHAPE GENERALITY (calibration must disable itself off-width)")
    for d, wd in ((2, 256), (8, 128), (32, 64), (4, 256), (64, 256)):
        results.append(check(f"depth {d} x width {wd}", est, MLP(he(d, wd, 8), seed=23)))

    print("\n6. BUDGET EDGES (the analytic fallback path)")
    m = MLP(he(32, 256, 9), seed=29)
    for b, label in ((0, "budget 0"), (1, "budget 1"), (10**7, "budget 1e7 (below min N)"),
                     (10**9, "budget 1e9"), (10 * BUDGET, "budget 10x")):
        results.append(check(label, est, m, budget=b))

    print("\n7. GRADER-ARRAY IMMUTABILITY (weights must be untouched)")
    m = MLP(he(32, 256, 10), seed=31)
    before = [w.copy() for w in m.weights]
    est.predict(m, BUDGET)
    same = all(np.array_equal(a, b) for a, b in zip(before, m.weights))
    print(f"  {'input weights unmodified':<34} {'ok' if same else 'MUTATED INPUT'}")
    results.append(same)

    n_ok = sum(1 for r in results if r)
    print(f"\n{n_ok}/{len(results)} checks passed")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
