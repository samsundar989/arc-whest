"""WP13 / E0-COST: what does a REGENERATING K3 flow cost, in flopscope 0.12?

Contract: `experiments/LOG.md`, 2026-09-02, "CONTRACT (Rule 1): WP13", HEAD
`8b46574`. Gate **G-COST**: the minimal flopscope-priced total is `<= 0.20 B`;
`> 0.35 B` closes the lane on arithmetic alone.

WHY THIS FILE EXISTS
--------------------
`experiments/2026-08-28_pb_oracle/drift_oracle.json` prices the `dsproj` cell --
drop `kappa_3`'s all-distinct block, keep both read-slices exact, rank `n` -- at
`0.2977 x budget`. That figure is **upstream's own torch-side runtime counter**,
which `docs/GOTCHAS.md` refuses to equate with flopscope in either direction, and
it is the cost of a GENERIC `O(n^2 r)` spelling that is never told about the
structure it is manipulating.

The structure, read out of `factor_k3.py:252-274` on 2026-09-02:

    FactoredTensor.from_dstensor(ds) -> factors ( diag(S3) + 3*S21^T ,  I ,  I )

**Two of the three legs are exactly the identity.** `contract_W` on an identity
leg is `W` itself and needs no matmul; a diagonal leg inside a diagonal-slice
einsum is a row scaling, not an `(n x R)(R x n)` product. A regenerating flow
therefore pays for far fewer `n^3` operations than the generic spelling bills,
and G-COST is the measurement of how many.

WHAT IS MEASURED, AND WHAT IS MODELLED
--------------------------------------
**Measured.** Every `n^3`-class operation in the per-layer op sequence is
EXECUTED, at the live shape `1024 x 16`, in `flopscope.numpy`, inside a real
`flopscope.BudgetContext`, and its counted FLOPs are read off the meter. Nothing
in the third-order increment is a hand-derived `2 n^3`.

**Modelled.** The K2 backbone is taken as the shipped file's official one-row
subprocess anchor `137,600,326,656` rather than re-executed, and the total is the
SUM of that anchor and the measured increment. That sum is a **projection**: a
real candidate may share intermediates with the backbone (cheaper) or need extra
casts and temporaries (dearer), and it must still clear `whest validate`, the
`4 GiB` single-array rule and the `0.4 s` residual cliff, none of which a FLOP
count can speak to.

**Values are arbitrary.** Counted FLOPs on this dense path are value-independent
-- the same assumption `local_sim/p2_flop_anchor.py` already rests on, and
`--value-invariance` re-measures it here rather than inheriting it.
`# seed-ledger: no block named` -- no seed block is drawn and nothing is selected.

Run:
    uv run python local_sim/p2_slice_cost.py
    uv run python local_sim/p2_slice_cost.py --width 1024 --depth 16 --out <path>
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# APPEND, never insert: `local_sim/` holds numpy shims named `flopscope` and
# `whestbench`, and prepending this directory turns every counted FLOP below into
# fiction. `_require_real_flopscope` asserts the real package before any number.
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np  # noqa: E402

WIDTH, DEPTH = 1024, 16
BUDGET = 2 ** 41

#: The shipped file's official Phase-2 one-row subprocess anchor (v2.6.1,
#: sha256 00240eb1...). docs/CURRENT_STATE.md. This is the K2 backbone the
#: third-order increment is added to.
K2_ANCHOR_FLOPS = 137_600_326_656

#: Upstream's torch-side counter for the generic `dsproj` spelling, for the
#: P2 ratio only. NOT flopscope, and no gate is scored on it.
UPSTREAM_DSPROJ_X_BUDGET = 0.2977

#: The grading server bills 137,617,103,872 where the local official-package
#: runner bills 137,600,326,656 -- an unexplained +2**24 (+0.0122%). The contract
#: requires it added to any operating point within 0.5% of the 0.1 floor.
SERVER_OFFSET_FLOPS = 2 ** 24


def _require_real_flopscope(flops) -> None:
    path = os.path.abspath(getattr(flops, "__file__", "") or "")
    if path.startswith(os.path.abspath(_HERE) + os.sep):
        raise RuntimeError(
            f"REFUSED: `flopscope` resolved to the local shim at {path}. Every "
            "counted FLOP below would be fiction."
        )
    if not hasattr(flops, "BudgetContext"):
        raise RuntimeError(f"flopscope at {path} has no BudgetContext.")


# --------------------------------------------------------------------------
# the metered op sequence
# --------------------------------------------------------------------------

def _third_order_layer(fnp, W, M, w1, Sig_off, *, last_layer: bool):
    """ONE layer of the regenerating third-order step, in `fnp`. Metered.

    State entering: `kappa_3` of the previous post-ReLU activations, carried in
    the regenerated form -- factors `(M, I, I)` with `M = diag(S3) + 3 S21^T`,
    which is EXACTLY what `FactoredTensor.from_dstensor` builds and therefore
    exactly what a rank-`n` regenerating flow holds.

    The three `n^3`-class operations, and why there are only three:

    1. `P = W.T @ M` -- `contract_W` on leg 1. Legs 2 and 3 are the identity, so
       `contract_W` returns `W.T` for both and costs nothing.
    2/3. the `(2, 1)` slice of the wick-scaled contracted tensor. Its three block
       permutations are `(A1.A2) A3^T`, `(A1.A3) A2^T`, `(A2.A3) A1^T`; with
       `A2 == A3 == W.T` two of them coincide, leaving TWO distinct matmuls.

    The `(3,)` slice, the wick row scalings, the diagonal zeroing and the whole
    newly-generated two-leg block are `O(n^2)`: the block's middle leg is `3 I`,
    so every one of ITS block permutations degenerates to a row scaling. They are
    executed here anyway, at full width, so the meter -- not this docstring --
    decides whether "negligible" is true.
    """
    # 1. contract_W on the only non-identity leg.
    P = W.T @ M
    Q = W.T

    # the (3,) slice of the contracted tensor: read by the degree-1 readout.
    S3_z = (P * Q * Q).sum(axis=1)

    if last_layer:
        # nothing downstream reads the (2, 1) slice at the final layer, so the
        # two matmuls below are not paid. Charging them would overstate the
        # method by 1/16 of its dominant term.
        return S3_z, None

    # wick row scalings -- O(n^2), the ReLU's Hermite coefficients.
    Pw = P * w1[:, None]
    Qw = Q * w1[:, None]

    # 2/3. the (2, 1) slice. Two distinct matmuls because A2 == A3.
    PQ = Pw * Qw
    QQ = Qw * Qw
    S21 = (2.0 * (PQ @ Qw.T) + (QQ @ Pw.T)) / 3.0

    # the newly generated two-leg block, whose middle leg is 3*I: every block
    # permutation is a row scaling. Executed at full width so the meter judges it.
    fac1 = Sig_off * w1[:, None]
    fac3 = Sig_off * w1[None, :]
    S21 = S21 + (fac1 * np.float64(3.0)) + (fac1 * fac3) * np.float64(3.0) + fac3 * np.float64(3.0)
    S3_z = S3_z + (fac1 * fac3).sum(axis=1)

    # zero the diagonal: the (2, 1) slice is defined with repeated indices removed.
    S21 = S21 - fnp.diag(fnp.diag(S21))

    return S3_z, S21


def _rebuild_M(fnp, S3, S21):
    """`from_dstensor`'s own reconstruction: `M = diag(S3) + 3 * S21^T`. O(n^2)."""
    return fnp.diag(S3) + np.float64(3.0) * S21.T


def _generic_third_order_layer(fnp, W, A1, A2, A3, w1):
    """The GENERIC spelling: three real legs, no identity, rank `r`. For P2 only.

    `contract_W` on all three legs (3 matmuls) plus a `(2, 1)` slice with three
    distinct block permutations (3 matmuls) = six `(n x n)(n x r)` products, twice
    the three the structured spelling needs.
    """
    B1, B2, B3 = W.T @ A1, W.T @ A2, W.T @ A3
    B1, B2, B3 = B1 * w1[:, None], B2 * w1[:, None], B3 * w1[:, None]
    S21 = ((B1 * B2) @ B3.T + (B1 * B3) @ B2.T + (B2 * B3) @ B1.T) / 3.0
    S3 = (B1 * B2 * B3).sum(axis=1)
    return S3, S21


# --------------------------------------------------------------------------
# drivers
# --------------------------------------------------------------------------

def measure(width: int, depth: int, *, spelling: str, seed: int = 1_000_003) -> dict:
    """Counted flopscope FLOPs for `depth` layers of the third-order increment."""
    import flopscope as flops
    import flopscope.numpy as fnp

    _require_real_flopscope(flops)

    rng = np.random.default_rng(seed)
    n = width

    # Built OUTSIDE the BudgetContext, exactly as the grader builds weights
    # (`MLP.from_row` runs during dataset load, not during `predict`).
    Ws = [fnp.array(rng.standard_normal((n, n)) * (2.0 / n) ** 0.5) for _ in range(depth)]
    M0 = fnp.array(rng.standard_normal((n, n)) * 1e-3)
    w1 = fnp.array(np.abs(rng.standard_normal(n)))
    Sig = fnp.array(rng.standard_normal((n, n)) * 1e-3)
    A0 = [fnp.array(rng.standard_normal((n, n)) * 1e-3) for _ in range(3)]

    with flops.BudgetContext(flop_budget=10 * BUDGET, quiet=True) as ctx:
        if spelling == "regen":
            M = M0
            for l in range(depth):
                S3, S21 = _third_order_layer(
                    fnp, Ws[l], M, w1, Sig, last_layer=(l == depth - 1))
                if S21 is not None:
                    M = _rebuild_M(fnp, S3, S21)
        elif spelling == "generic":
            A1, A2, A3 = A0
            for l in range(depth):
                S3, S21 = _generic_third_order_layer(fnp, Ws[l], A1, A2, A3, w1)
                A1 = _rebuild_M(fnp, S3, S21)
                A2, A3 = Ws[l].T, Ws[l].T
        else:
            raise ValueError(spelling)
        used = float(ctx.flops_used) if hasattr(ctx, "flops_used") else None

    if used is None:
        used = float(getattr(ctx, "used", getattr(ctx, "total", float("nan"))))
    return {"spelling": spelling, "width": n, "depth": depth, "flops": used,
            "x_budget": used / BUDGET, "per_layer": used / depth,
            "per_layer_over_n3": used / depth / n ** 3}


def primitive_prices(width: int) -> dict:
    """flopscope 0.12's price for each primitive, so the total is ATTRIBUTED.

    The finding this exists to record: flopscope bills an `(n, n) @ (n, n)`
    matmul at **`4 n^3`**, not the textbook `2 n^3`. Every hand cost model in
    this repository that assumed `2 n^3` was a factor of two optimistic, and the
    whole third-order increment below is exactly three such matmuls per layer.
    """
    import flopscope as flops
    import flopscope.numpy as fnp

    _require_real_flopscope(flops)
    n = width
    rng = np.random.default_rng(5)
    A = fnp.array(rng.standard_normal((n, n)))
    B = fnp.array(rng.standard_normal((n, n)))
    v = fnp.array(rng.standard_normal(n))

    def price(fn):
        with flops.BudgetContext(flop_budget=10 ** 14, quiet=True) as c:
            fn()
        return float(c.flops_used)

    ops = {
        "matmul_nn": price(lambda: A @ B),
        "matmul_T_nn": price(lambda: A.T @ B),
        "hadamard_nn": price(lambda: A * B),
        "rowsum_nn": price(lambda: (A * B).sum(axis=1)),
        "transpose_nn": price(lambda: A.T),
        "diag_from_vec": price(lambda: fnp.diag(v)),
        "diag_of_mat": price(lambda: fnp.diag(A)),
        "add_nn": price(lambda: A + B),
        "rowscale_nn": price(lambda: A * v[:, None]),
    }
    return {"width": n, "flops": ops,
            "over_n3": {k: v_ / n ** 3 for k, v_ in ops.items()},
            "matmul_price_n3": ops["matmul_nn"] / n ** 3}


def value_invariance(width: int, depth: int) -> dict:
    """The counted total does not depend on the VALUES. Measured, not assumed."""
    a = measure(width, depth, spelling="regen", seed=11)
    b = measure(width, depth, spelling="regen", seed=97)
    return {"seed_11": a["flops"], "seed_97": b["flops"],
            "identical": a["flops"] == b["flops"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--depth", type=int, default=DEPTH)
    ap.add_argument("--out", default="experiments/2026-09-02_wp13_slice/e0_cost.json")
    ap.add_argument("--skip-generic", action="store_true")
    a = ap.parse_args()

    print("=" * 78)
    print("WP13 / E0-COST  --  regenerating K3 flow, priced in flopscope 0.12")
    print("=" * 78)
    print(f"  shape {a.width} x {a.depth}    budget 2**41 = {BUDGET:,}")
    print()

    regen = measure(a.width, a.depth, spelling="regen")
    print(f"  third-order increment, STRUCTURED spelling")
    print(f"    counted FLOPs      {regen['flops']:>20,.0f}")
    print(f"    x budget           {regen['x_budget']:>20.6f}")
    print(f"    per layer / n^3    {regen['per_layer_over_n3']:>20.3f}")
    print()

    rows = {"regen": regen}
    if not a.skip_generic:
        gen = measure(a.width, a.depth, spelling="generic")
        rows["generic"] = gen
        print(f"  third-order increment, GENERIC spelling (three real legs)")
        print(f"    counted FLOPs      {gen['flops']:>20,.0f}")
        print(f"    x budget           {gen['x_budget']:>20.6f}")
        print(f"    per layer / n^3    {gen['per_layer_over_n3']:>20.3f}")
        print(f"    structured saving  {gen['flops'] / regen['flops']:>20.3f}x")
        print()

    prices = primitive_prices(a.width)
    print("  primitive prices, flopscope 0.12, measured at this width")
    for k, v_ in sorted(prices["over_n3"].items(), key=lambda kv: -kv[1]):
        print(f"    {k:<18} {prices['flops'][k]:>16,.0f}   = {v_:8.4f} n^3")
    print(f"    -> a dense (n,n)@(n,n) matmul costs {prices['matmul_price_n3']:.2f} n^3, "
          "NOT the textbook 2 n^3")
    print(f"    -> the structured spelling is {regen['per_layer_over_n3']:.2f} n^3/layer "
          f"= {regen['per_layer_over_n3'] / prices['matmul_price_n3']:.2f} matmuls/layer")
    print()

    vi = value_invariance(a.width, min(a.depth, 4))
    print(f"  value invariance   two different random draws, same op sequence")
    print(f"    identical counts   {vi['identical']}")
    print()

    total = K2_ANCHOR_FLOPS + regen["flops"]
    x = total / BUDGET
    near_floor = abs(x - 0.1) / 0.1 <= 0.005
    x_srv = (total + SERVER_OFFSET_FLOPS) / BUDGET if near_floor else x
    mult = max(0.1, x_srv)

    print("  PROJECTED TOTAL  (K2 backbone anchor + measured third-order increment)")
    print(f"    K2 anchor (shipped v2.6.1, official subprocess)  {K2_ANCHOR_FLOPS:>18,}")
    print(f"    third-order increment (measured here)            {regen['flops']:>18,.0f}")
    print(f"    total                                            {total:>18,.0f}")
    print(f"    x budget                                         {x:>18.6f}")
    if near_floor:
        print(f"    within 0.5% of the floor: +2**24 server offset applied -> {x_srv:.6f}")
    print(f"    multiplier = max(0.1, F/2**41)                   {mult:>18.6f}")
    print()

    print("  GATE  G-COST  (contract WP13, threshold fixed before this ran)")
    print(f"    <= 0.20 B  -> PASS, the lane continues")
    print(f"    >  0.35 B  -> the lane closes on arithmetic, no seeds spent")
    verdict = "PASS" if x <= 0.20 else ("CLOSE-LANE" if x > 0.35 else "MARGINAL")
    print(f"    measured {x:.6f} B   ->   **{verdict}**")
    print()
    print("  Score implied at this multiplier, for three raw accuracies:")
    for name, raw in (("full K3 (measured, 24 DESIGN mints)", 4.0047e-08),
                      ("dsproj bracket midpoint", 6.9e-07),
                      ("shipped K2p1", 4.050566e-06)):
        print(f"    {name:<38} raw {raw:.4e}  ->  adjusted {raw * mult:.4e}")
    print()
    print("  CAVEAT: this is a PROJECTION. The increment is measured; the sum with")
    print("  the K2 anchor is not an end-to-end run of any candidate, and no")
    print("  residual, memory or contract check is implied by it.")
    print("=" * 78)

    doc = {
        "check": "WP13 E0-COST: flopscope-priced regenerating K3 third-order increment",
        "contract": "experiments/LOG.md 2026-09-02 CONTRACT WP13, HEAD 8b46574",
        "width": a.width, "depth": a.depth, "budget": BUDGET,
        "flopscope_version": _fs_version(),
        "spellings": rows,
        "value_invariance": vi,
        "primitive_prices": prices,
        "k2_anchor_flops": K2_ANCHOR_FLOPS,
        "projected_total_flops": total,
        "projected_x_budget": x,
        "server_offset_applied": near_floor,
        "projected_x_budget_with_server_offset": x_srv,
        "multiplier": mult,
        "upstream_dsproj_x_budget": UPSTREAM_DSPROJ_X_BUDGET,
        "upstream_caveat": "upstream torch-side counter, NOT flopscope; ratios within itself only",
        "gate_g_cost_pass_at": 0.20,
        "gate_g_cost_close_at": 0.35,
        "verdict": verdict,
        "caveat": "the increment is MEASURED; the total is a PROJECTION onto the shipped K2 anchor",
    }
    out = os.path.abspath(os.path.join(_ROOT, a.out))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
    print(f"  wrote {os.path.relpath(out, _ROOT)}")
    return 0


def _fs_version() -> str:
    import flopscope
    return str(getattr(flopscope, "__version__", "unknown"))


if __name__ == "__main__":
    raise SystemExit(main())
