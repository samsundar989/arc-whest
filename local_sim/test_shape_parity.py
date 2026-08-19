"""
SHAPE PARITY — the fallback path must be v2.4.1, exactly.

Stage 0 / P0.3 of `docs/FRONTIER_ARCHITECTURE_PROPOSAL.md`.

v2.6 adds one Strassen level, which requires every matmul dimension to be even.
It buys that by trimming the weakest survivor out of any odd alive set, moving it
into `dead` where its closed-form rectified-Gaussian mean is carried instead of a
sampled value. That trade is deliberate and measured: 0.51% MSE, LOG 2026-08-19.

The defect this file exists to catch is paying that cost and getting NOTHING for
it. If the chain turns out to be Strassen-ineligible — an odd global width, a
single-survivor layer, a degenerate layer — the estimator falls back to the dense
chain, which is v2.4.1's algorithm exactly. On that path v2.6 must produce
v2.4.1's numbers **bit for bit**. Anything else is a pure accuracy loss.

THE STANDARD, and why it is equality and not a tolerance:
  * `use_str` False  -> v2.6 and v2.4.1 execute the same dense chain on the same
                        float32 inputs in the same order. Equality is achievable,
                        so equality is demanded.
  * `use_str` True   -> the trim is intentional. Parity is NOT required; the
                        check is that the result is finite, correctly shaped, and
                        close enough to v2.4.1 to be the same estimator.

Run: uv run python local_sim/test_shape_parity.py
"""

import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]

import flopscope as flops  # noqa: E402  isort:skip
import flopscope.numpy as fnp  # noqa: E402  isort:skip

import numpy as np  # noqa: E402

for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

flops.configure(symmetry_warnings=False)

import harness  # noqa: E402
from mint_provenance import canonical_mint  # noqa: E402

BUDGET = 272_000_000_000
RESULTS = []

# Widths chosen to hit every branch: odd below the Sobol dim, even non-power-of-two,
# the shipped width, and one above the Sobol dim where the analytic path takes over.
# Depth 1 is included because `depth < 2` is its own early return.
SHAPES = [
    (63, 4), (65, 4), (96, 4), (127, 4), (192, 4), (255, 4), (256, 4), (257, 4),
    (65, 2), (65, 32), (256, 32), (256, 2), (64, 1), (65, 1), (256, 1),
    (2, 4), (3, 4),
]

# Legacy-namespace seeds: this file mints throwaway networks for shape checks and
# decides nothing, so it must not draw on any registered decision block.
# seed-ledger: excludes V
# (the trim scan also mints from 4_300_000+, still the legacy namespace)
SEEDS = [4_242_001, 4_242_002, 4_242_003]


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"  {name:<54} {'ok ' if ok else 'FAILED'}  {detail}")
    return ok


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _predict(mod, w_np, width, depth, seed):
    mlp = harness.MLP(weights=[np.asarray(x) for x in w_np],
                      width=width, depth=depth, seed=seed)
    est = mod.Estimator()
    return np.asarray(est.predict(mlp, BUDGET), dtype=np.float64)


def _uses_strassen(shipped, w_np, width, depth, seed):
    """Did the shipped file actually take the Strassen path for this shape?

    Read out of the estimator's own decision rather than guessed, by rerunning
    its pre-sampling logic. Only the dimensions matter, so this is cheap.
    """
    if not getattr(shipped, "_STRASSEN", False):
        return False
    mlp = harness.MLP(weights=[np.asarray(x) for x in w_np],
                      width=width, depth=depth, seed=seed)
    est = shipped.Estimator()
    log2n = est._log2_samples(width, depth, BUDGET)
    if log2n < shipped._MIN_LOG2_N or width > shipped._SOBOL_DIM or depth < 2:
        return False
    w64 = [w.astype(fnp.float64) for w in mlp.weights]
    alphas, _mus, _phis = est._closure(w64, width, depth)
    if hasattr(est, "_alive_sets"):
        trim = (width & 1) == 0
        alive, _dead = est._alive_sets(alphas, width, depth, trim)
        return trim and all((int(x.shape[0]) & 1) == 0 for x in alive)
    # Pre-fix layout: reproduce the old inline decision.
    counts = []
    for li in range(depth):
        if li == depth - 1:
            counts.append(width)
            continue
        a = alphas[li]
        idx = fnp.flatnonzero(a > shipped._PRUNE_TAU)
        k = int(idx.shape[0])
        if k == 0:
            counts.append(int(fnp.flatnonzero(a >= a.max()).shape[0]))
            continue
        if (k & 1) and k > 1:
            k = int(fnp.flatnonzero(a > fnp.take(a, idx).min()).shape[0])
        counts.append(k)
    return (width & 1) == 0 and all((c & 1) == 0 for c in counts)


def t_trim_cannot_annihilate(shipped):
    """The trim must shed at most ONE neuron, and never the whole layer.

    The defect this catches is not hypothetical — it is what v2.6 did. Raising the
    threshold past the weakest survivor (`a > min(alpha[alive])`) sheds exactly one
    neuron only when that survivor is unique. At layer 0 it never is: x ~ N(0, I)
    makes every pre-activation mean zero, so every alpha is EXACTLY 0.0 and
    `a > 0.0` keeps nothing. On any odd width v2.6 therefore deleted the entire
    first layer's sampled path and replaced it with the closure's means.
    """
    print("\nTRIM SAFETY — the layer-0 annihilation")
    print("-" * 40)
    if not hasattr(shipped.Estimator, "_alive_sets"):
        check("shipped file exposes _alive_sets", False, "pre-fix layout")
        return
    est = shipped.Estimator()
    for width in (65, 127, 255, 63, 3, 256):
        w_np, _s, _ = canonical_mint(width, 4, SEEDS[0])
        w64 = [fnp.asarray(w).astype(fnp.float64) for w in w_np]
        alphas, _mus, _phis = est._closure(w64, width, 4)
        a0 = np.asarray(alphas[0])
        untrimmed, _ = est._alive_sets(alphas, width, 4, False)
        trimmed, _ = est._alive_sets(alphas, width, 4, True)
        u = [int(x.shape[0]) for x in untrimmed]
        t = [int(x.shape[0]) for x in trimmed]
        shed = [ui - ti for ui, ti in zip(u, t)]
        ok = min(t) > 0 and max(shed) <= 1 and min(shed) >= 0
        check(
            f"width {width}: trim sheds <=1 per layer and never empties one",
            ok,
            f"layer-0 alphas all equal ({bool((a0 == a0[0]).all())})  shed {shed}",
        )


def t_trim_scan(shipped, n_nets=200, width=256, depth=32):
    """The Skeptic's follow-up, promoted into the suite.

    D4 — "v2.6.1 is bit-identical to v2.6 wherever Strassen runs" — is NOT an
    identity. The guard refuses a trim that would shed more than one neuron, and a
    survivor minimum with an exact tie of odd multiplicity >= 3 is a case where
    v2.6 sheds 3 (odd -> even, Strassen runs) and v2.6.1 refuses (stays odd ->
    dense). The independent Skeptic constructed exactly that at width 8. This
    bounds how often it happens at the competition shape.

    Reads alphas only: no truth, no sampling, no gate. Legacy-namespace seeds.
    """
    print(f"\nTRIM SCAN — {n_nets} fresh {width}x{depth} mints, alphas only")
    print("-" * 52)
    if not hasattr(shipped.Estimator, "_alive_sets"):
        check("shipped file exposes _alive_sets", False, "pre-fix layout")
        return
    est = shipped.Estimator()
    ties, odd_after, worst_shed, degenerate = 0, 0, 0, 0
    for k in range(n_nets):
        w_np, _s, _ = canonical_mint(width, depth, 4_300_000 + k)
        w64 = [fnp.asarray(w).astype(fnp.float64) for w in w_np]
        alphas, _mus, _phis = est._closure(w64, width, depth)
        untrimmed, _ = est._alive_sets(alphas, width, depth, False)
        trimmed, _ = est._alive_sets(alphas, width, depth, True)
        for li in range(depth):
            u, t = int(untrimmed[li].shape[0]), int(trimmed[li].shape[0])
            worst_shed = max(worst_shed, u - t)
            if u == 0:
                degenerate += 1
            # Layers 1..depth-2 only. Layer 0's alphas are ALL exactly 0.0 by
            # construction, so it ties by identity — counting it here would
            # report 200/200 forever and drown the case this scan exists to
            # find: an exact tie among layer 1+ survivors, where the trim would
            # shed more than one neuron.
            if 0 < li < depth - 1 and u > 1:
                a = alphas[li]
                m = fnp.take(a, untrimmed[li]).min()
                mult = int(fnp.flatnonzero(a == m).shape[0])
                if mult > 1:
                    ties += 1
            if t & 1:
                odd_after += 1
    check("no layer sheds more than one neuron", worst_shed <= 1, f"max shed {worst_shed}")
    check("no exact tie at a survivor minimum, layers 1..30", ties == 0,
          f"{ties} tie(s) in {n_nets * (depth - 2):,} layer-draws")
    check("no layer left odd after trimming (so Strassen always runs)",
          odd_after == 0, f"{odd_after} odd")
    check("no degenerate all-pruned layer", degenerate == 0, f"{degenerate}")


def main():
    print("=" * 104)
    print("SHAPE PARITY — v2.6 fallback vs v2.4.1")
    print("=" * 104)
    shipped = _load("estimator.py", "_ship")
    ref = _load("estimator_v2_4_1.py", "_ref241")
    print(f"  shipped _STRASSEN = {getattr(shipped, '_STRASSEN', None)}")
    t_trim_cannot_annihilate(shipped)
    t_trim_scan(shipped, n_nets=int(os.environ.get("SHAPE_SCAN_NETS", "200")))
    print()

    print(f"  {'shape':>10} {'path':>9} {'max rel diff':>14} {'pred MSE':>12}  verdict")
    print("  " + "-" * 96)
    for width, depth in SHAPES:
        worst_rel, worst_mse, ok_all = 0.0, 0.0, True
        used_flags = []
        for s in SEEDS:
            w_np, est_seed, _ss = canonical_mint(width, depth, s)
            p_new = _predict(shipped, w_np, width, depth, est_seed)
            p_ref = _predict(ref, w_np, width, depth, est_seed)
            if p_new.shape != (depth, width):
                ok_all = False
                break
            if not np.all(np.isfinite(p_new)):
                ok_all = False
                break
            used_flags.append(_uses_strassen(shipped, w_np, width, depth, est_seed))
            den = np.maximum(np.abs(p_ref), 1e-30)
            worst_rel = max(worst_rel, float(np.abs(p_new - p_ref).max() / den.max()))
            worst_mse = max(worst_mse, float(np.mean((p_new[-1] - p_ref[-1]) ** 2)))
        # Grade under the STRICT bound unless EVERY seed took the Strassen path.
        # `any(...)` would let a shape that is dense on two seeds and Strassen on
        # one be judged entirely by the loose 5% bound. (Skeptic, 2026-08-19.)
        all_str = bool(used_flags) and all(used_flags)
        path = "strassen" if all_str else ("MIXED" if any(used_flags) else "DENSE")
        if all_str:
            verdict = "trim is intentional here"
            ok = ok_all and worst_rel < 0.05
        else:
            verdict = "must be BIT-IDENTICAL to v2.4.1"
            ok = ok_all and worst_rel == 0.0
        print(f"  {f'{width}x{depth}':>10} {path:>9} {worst_rel:>14.4e} {worst_mse:>12.4e}"
              f"  {'ok ' if ok else 'FAILED'}  {verdict}")
        RESULTS.append(bool(ok))

    n_ok = sum(1 for r in RESULTS if r)
    print(f"\n{n_ok}/{len(RESULTS)} shapes passed")
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
