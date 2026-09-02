"""WP13 / E0-COST addendum: the two levers on the multiplier, priced.

Contract: `experiments/LOG.md`, 2026-09-02, "CONTRACT (Rule 1): WP13", HEAD
`8b46574`. **This moves no gate.** G-COST was scored by `p2_slice_cost.py` at
`0.1527 B` and stands. This file answers the follow-on question that number
raises and that the contract explicitly recorded as *"noticed while reading the
meter, not measured"*:

    0.1527 B is a multiplier of 0.153, not 0.100. Fifty-three percent of the
    cost buys nothing at all. Can either untried lever reach the floor?

Two levers were named in the E0-COST entry. Both are measured here, at the live
width, in `flopscope 0.12` inside a real `BudgetContext`, and **one of them is
dead**:

- **symmetry tagging.** `Sigma` is symmetric and `W^T Sigma W` is two of the five
  matmuls a regenerating K3 flow spends. `flops.as_symmetric(S, symmetry=(0, 1))`
  validates and tags. It buys **nothing**: the tagged and untagged congruences
  bill the identical `7.996 n^3`, and the tagging itself costs `0.0137 n^3`. A
  symmetric operand is not a cheaper operand.
- **Strassen.** It pays, and by more than the shipped file is currently taking.

`# seed-ledger: no block named` -- no network is minted, no truth is read, and
counted FLOPs on this dense path are value-independent.

WHAT THIS FILE DOES NOT ESTABLISH, AND IT IS THE HALF THAT DECIDES
------------------------------------------------------------------
**Counted FLOPs are not the only budget.** `docs/GOTCHAS.md` records that residual
wall time tracks the NUMBER of operations and the size of the temporaries they
free, not FLOPs -- and that is why `estimator.py` v2.4 was re-engineered for
operation count in the first place. Strassen at `L` levels replaces one matmul
with `7^L` leaf matmuls and `~18 * (7^L - 1) / 6` full-size additions. At `L = 4`
that is 2,401 leaves and about 7,200 additions **per matmul**, against a `0.4 s`
per-MLP cliff that annihilates a submission rather than penalising it.

So the table below is a counted-FLOP ceiling on what the lever could be worth. It
is not a licence to use `L = 4`, or any `L`. The operation counts are reported
beside the FLOPs precisely so the two budgets are never conflated, and any
operating point must be soaked through `gpuq` before it means anything.

Run:
    uv run python local_sim/p2_cost_levers.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# APPEND, never insert -- `local_sim/` shadows `flopscope` with a numpy shim.
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np  # noqa: E402

WIDTH = 1024
BUDGET = 2 ** 41

#: E0-COST's measured projection for a regenerating K3 flow at 1024 x 16:
#: K2 backbone 137,600,326,656 + third-order increment 198,233,321,472.
REGEN_TOTAL_FLOPS = 137_600_326_656 + 198_233_321_472

#: The grading server's unexplained per-MLP offset. The contract requires it
#: added to any operating point within 0.5% of the 0.1 floor -- and every row
#: below that reaches the floor is inside that band by construction.
SERVER_OFFSET_FLOPS = 2 ** 24


def _require_real_flopscope(flops) -> None:
    path = os.path.abspath(getattr(flops, "__file__", "") or "")
    if path.startswith(os.path.abspath(_HERE) + os.sep):
        raise RuntimeError(f"REFUSED: flopscope resolved to the shim at {path}.")


def _strassen(fnp, X, Y, levels, counter):
    """Textbook Strassen. `counter` tallies OPERATIONS, not FLOPs -- the two
    budgets are different and this file refuses to report only one."""
    if levels == 0:
        counter["matmul"] += 1
        return X @ Y
    m = X.shape[0] // 2
    a11, a12, a21, a22 = X[:m, :m], X[:m, m:], X[m:, :m], X[m:, m:]
    b11, b12, b21, b22 = Y[:m, :m], Y[:m, m:], Y[m:, :m], Y[m:, m:]
    counter["slice"] += 8
    s = lambda U, V: _strassen(fnp, U, V, levels - 1, counter)
    counter["add"] += 10
    m1 = s(a11 + a22, b11 + b22)
    m2 = s(a21 + a22, b11)
    m3 = s(a11, b12 - b22)
    m4 = s(a22, b21 - b11)
    m5 = s(a11 + a12, b22)
    m6 = s(a21 - a11, b11 + b12)
    m7 = s(a12 - a22, b21 + b22)
    counter["add"] += 8
    counter["concat"] += 3
    top = fnp.concatenate([m1 + m4 - m5 + m7, m3 + m5], axis=1)
    bot = fnp.concatenate([m2 + m4, m1 - m2 + m3 + m6], axis=1)
    return fnp.concatenate([top, bot], axis=0)


def measure(width: int, max_levels: int = 4) -> dict:
    import flopscope as flops
    import flopscope.numpy as fnp

    _require_real_flopscope(flops)
    n = width
    rng = np.random.default_rng(0)
    A = fnp.array(rng.standard_normal((n, n)))
    B = fnp.array(rng.standard_normal((n, n)))
    Sn = rng.standard_normal((n, n))
    Sn = (Sn + Sn.T) / 2.0
    S = fnp.array(Sn)
    W = fnp.array(rng.standard_normal((n, n)))

    def price(fn):
        with flops.BudgetContext(flop_budget=10 ** 15, quiet=True) as c:
            r = fn()
        return float(c.flops_used), r

    plain, _ = price(lambda: A @ B)

    # ---- lever 1: symmetry tagging ----
    tag_cost, Ssym = price(lambda: flops.as_symmetric(S, symmetry=(0, 1)))
    cong_plain, _ = price(lambda: (W.T @ S) @ W)
    cong_sym, _ = price(lambda: (W.T @ Ssym) @ W)
    sym = {"tag_cost": tag_cost, "tag_cost_over_n3": tag_cost / n ** 3,
           "congruence_plain": cong_plain, "congruence_tagged": cong_sym,
           "discount": cong_sym / cong_plain,
           "verdict": "NO DISCOUNT" if cong_sym >= cong_plain else "discount"}

    # ---- lever 2: Strassen ----
    strassen_rows = []
    for L in range(1, max_levels + 1):
        ctr = {"matmul": 0, "add": 0, "slice": 0, "concat": 0}
        f, _ = price(lambda L=L, ctr=ctr: _strassen(fnp, A, B, L, ctr))
        ops = sum(ctr.values())
        strassen_rows.append({
            "levels": L, "flops": f, "over_n3": f / n ** 3,
            "ratio_to_plain": f / plain,
            "leaf_matmuls": ctr["matmul"], "array_ops": ops,
            "ops_vs_plain": ops,
        })

    return {"width": n, "plain_matmul_flops": plain,
            "plain_matmul_over_n3": plain / n ** 3,
            "symmetry": sym, "strassen": strassen_rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--max-levels", type=int, default=4)
    ap.add_argument("--out",
                    default="experiments/2026-09-02_wp13_slice/e0_cost_levers.json")
    a = ap.parse_args()

    m = measure(a.width, a.max_levels)
    n = a.width

    print("=" * 78)
    print("WP13 / E0-COST addendum  --  the two levers on the multiplier")
    print("=" * 78)
    print(f"  shape {n} x {n}   plain matmul {m['plain_matmul_flops']:,.0f}"
          f"  = {m['plain_matmul_over_n3']:.4f} n^3")
    print()
    s = m["symmetry"]
    print("  LEVER 1  symmetry tagging on W^T Sigma W")
    print(f"    as_symmetric(S, symmetry=(0,1)) costs {s['tag_cost']:,.0f}"
          f"  = {s['tag_cost_over_n3']:.4f} n^3")
    print(f"    congruence, plain    {s['congruence_plain']:>16,.0f}")
    print(f"    congruence, tagged   {s['congruence_tagged']:>16,.0f}")
    print(f"    -> {s['verdict']}  (ratio {s['discount']:.4f}).  "
          "A symmetric operand is not a cheaper operand.")
    print()
    print("  LEVER 2  Strassen")
    print(f"    {'L':>3} {'counted FLOPs':>18} {'/n^3':>9} {'vs plain':>9} "
          f"{'leaf matmuls':>13} {'array ops':>10}")
    for r in m["strassen"]:
        print(f"    {r['levels']:>3} {r['flops']:>18,.0f} {r['over_n3']:>9.4f} "
              f"{r['ratio_to_plain']:>9.4f} {r['leaf_matmuls']:>13,} "
              f"{r['array_ops']:>10,}")
    print()
    print("  What that does to the regenerating K3 flow's multiplier")
    print(f"    {'L':>3} {'projected FLOPs':>18} {'x budget':>10} "
          f"{'multiplier':>11}   score at raw 4.0047e-08")
    base = REGEN_TOTAL_FLOPS
    rows = []
    for r in [{"levels": 0, "ratio_to_plain": 1.0}] + m["strassen"]:
        f = base * r["ratio_to_plain"]
        x = f / BUDGET
        near = abs(x - 0.1) / 0.1 <= 0.005
        x_eff = (f + SERVER_OFFSET_FLOPS) / BUDGET if near else x
        mult = max(0.1, x_eff)
        rows.append({"levels": r["levels"], "flops": f, "x_budget": x,
                     "server_offset_applied": near, "multiplier": mult,
                     "score_at_k3_raw": 4.0047e-08 * mult})
        flag = "  <- floor pinned" if mult <= 0.1 + 1e-12 else ""
        print(f"    {r['levels']:>3} {f:>18,.0f} {x:>10.5f} {mult:>11.5f}"
              f"   {4.0047e-08 * mult:.4e}{flag}")
    print()
    print("  READ THIS BEFORE USING ANY ROW ABOVE")
    print("    Residual wall time tracks OPERATION COUNT, not FLOPs (GOTCHAS;")
    print("    forum 18101), and the cliff at 0.4 s/MLP ANNIHILATES a submission")
    print("    rather than penalising it -- one overrun row is scored ~0.92")
    print("    instead of ~1e-7. The 'array ops' column is the warning: L = 4")
    print("    is 2,401 leaf matmuls per matmul. No level is authorised by a")
    print("    FLOP count; each must be soaked through gpuq at a declared")
    print("    footprint, and Strassen's own numerical error is unmeasured here.")
    print("=" * 78)

    doc = {"check": "WP13 E0-COST addendum: symmetry and Strassen levers",
           "contract": "LOG 2026-09-02 CONTRACT WP13, HEAD 8b46574",
           "moves_no_gate": True,
           "width": n, "budget": BUDGET,
           "regen_total_flops": REGEN_TOTAL_FLOPS,
           "measurements": m, "projection": rows,
           "caveat": ("counted-FLOP ceiling only; residual wall time tracks "
                      "operation count and is unmeasured for every Strassen "
                      "level here, as is Strassen's numerical error")}
    out = os.path.abspath(os.path.join(_ROOT, a.out))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
    print(f"  wrote {os.path.relpath(out, _ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
