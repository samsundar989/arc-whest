"""Lever 2a, the feasible frontier: how deep can batched Strassen actually go here?

This supersedes the shape-class roll-up in `p2_strassen_price.py`, which was
optimistic in a way worth stating plainly. That roll-up grouped matmuls by SHAPE
and priced one Strassen invocation per shape class. Shape is not the unit that
matters: **simultaneity** is. Two matmuls can share one batched Strassen only if
neither's operands depend on the other's result, and the chain is a sequential
16-layer recursion, so calls in the same shape class that happen at different
layers cannot be batched together at all.

WHAT THE OP LOG ACTUALLY SHOWS
------------------------------
148 matmul calls, and their positions in the operation log fall into tight
clusters separated by long gaps. Per layer there are three simultaneous groups:

    _advance      3 calls, adjacent      (n,n) @ (n, kn) x2  and  (n,n) @ (n,(k-1)n)
    the birth     2 calls, adjacent      (n,n) @ (n,n)
    _dslice21     4 calls, gaps 3-4      (n, kn) @ (kn, n)

so there are roughly `3 x 16 = 48` Strassen invocation sites, not 28. That matters
because operations are charged PER INVOCATION: at L=4 with `split_top=1` a single
invocation costs 511 operations, and `48 x 511 = 24,528` is already over the
24,000 gate on its own.

THE TWO CONSTRAINTS, AND WHY THE HANDOFF NAMED ONLY ONE
-------------------------------------------------------
    operations   <= 24,000 per MLP     (0.290 s at the graded 12.1 us/op)
    memory       <= 8,192 MB           (this candidate already peaks at 2,827.5 MB)

`split_top` trades one against the other and there is no single setting that wins:

    worst invocation, (1024,14336)@(14336,1024) x4, MEASURED peak over operands
        L      split_top=0        split_top=1
        1          +896 MB            +145 MB
        2        +2,688 MB            +475 MB
        3     MemoryError*            +922 MB
        4     MemoryError*          +1,706 MB
    * at 5,555 MB peak on a host with room to spare -- the fully batched stack is
      `(7/4)^L` times the operands and simply does not fit.

    operations per invocation
        L      split_top=0        split_top=1
        1               24                 28
        2               47                189
        3               70                350
        4               93                511

So `split_top=0` is operation-cheap and dies on memory; `split_top=1` is
memory-cheap and dies on operations. **The answer is to choose `split_top` PER
INVOCATION**: the early layers carry a small `k` and fit fully batched, while only
the widest late-layer invocations need the sequential split. This module reads the
real invocation groups out of the operation log and reports the frontier.

ROUTINE: a lab script. Spends no seed block (`# seed-ledger: no block named`),
moves no gate, changes no shipped file. Nothing here is a promotion -- it prices a
candidate that has not been built.

Run:  uv run python local_sim/p2_strassen_plan.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]

B_FLOPS = float(2 ** 41)
GRADED_RAW = 3.7793e-08          # graded final-layer MSE of submission 330212
MEM_LIMIT_MB = 8192.0
BASELINE_PEAK_MB = 2827.5        # linux_memory_gate on sha 220d2314

#: MEASURED extra peak, in MB, over the operands, for the WIDEST invocation
#: ((1024,14336)@(14336,1024) x4) at each (L, split_top). One process per point,
#: `K32GetProcessMemoryInfo` peak working set. None = MemoryError.
MEASURED_MEM = {
    (0, 0): 0.0, (1, 0): 896.3, (2, 0): 2688.3, (3, 0): None, (4, 0): None,
    (0, 1): 0.0, (1, 1): 145.4, (2, 1): 475.1, (3, 1): 922.2, (4, 1): 1706.0,
}
#: MEASURED operations for one invocation at each (L, split_top). Shape-free:
#: a batched level costs a flat 23 operations however wide the batch is.
OPS_PER_INVOCATION = {
    (0, 0): 1, (1, 0): 24, (2, 0): 47, (3, 0): 70, (4, 0): 93,
    (0, 1): 1, (1, 1): 28, (2, 1): 189, (3, 1): 350, (4, 1): 511,
}
#: MEASURED FLOP ratio vs L=0 for the widest invocation. `split_top` barely
#: moves it (105.38 vs 105.59 GFLOP at L=1 -- the split spelling is marginally
#: CHEAPER, having fewer concatenates), so one ratio per L is used for both.
FLOP_RATIO = {0: 1.0, 1: 0.87633, 2: 0.77228, 3: 0.68589, 4: 0.61846}


def invocation_groups(records, max_gap: int = 6):
    """Cluster matmul calls into SIMULTANEOUS groups by op-log adjacency.

    Two matmuls whose op-log positions are within `max_gap` of each other are
    doing the same step of the algorithm on independent operands; a long gap is a
    layer boundary, and a layer boundary is a data dependency. This is a
    heuristic read of a real execution trace, and it is deliberately reported
    alongside the raw gaps so a reader can check it rather than trust it.
    """
    mm = [(i, r) for i, r in enumerate(records) if r.op_name == "matmul"]
    groups, cur, prev = [], [], None
    for idx, rec in mm:
        sh = rec.shapes
        if len(sh) != 2 or len(sh[0]) != 2 or len(sh[1]) != 2:
            continue                                  # a matvec: not a Strassen site
        if prev is not None and idx - prev > max_gap:
            groups.append(cur)
            cur = []
        cur.append((idx, tuple(sh[0]), tuple(sh[1]), int(rec.flop_cost)))
        prev = idx
    if cur:
        groups.append(cur)
    return groups


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--candidate", default="estimator_k3_full.py")
    ap.add_argument("--max-gap", type=int, default=6)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    import official_config as oc
    from p2_strassen_price import meter

    gate, per_op = oc.PHASE2_OP_COUNT_GATE, oc.GRADED_RESIDUAL_S_PER_OP
    records, used = meter(a.candidate, width=1024, depth=16, seed=1_000_003)
    base_ops = len(records)
    groups = invocation_groups(records, a.max_gap)
    mm_flops = sum(f for g in groups for (_i, _sa, _sb, f) in g)

    print("=" * 94)
    print(f"STRASSEN FEASIBLE FRONTIER  --  {a.candidate}  1024 x 16")
    print("=" * 94)
    print(f"  base chain      {used:,} FLOPs   {base_ops:,} operations"
          f"   {used / B_FLOPS:.5f} B")
    print(f"  matmul          {mm_flops:,} FLOPs in {sum(len(g) for g in groups)}"
          f" calls across {len(groups)} SIMULTANEOUS invocation groups")
    print(f"  op gate         {gate:,}    memory {MEM_LIMIT_MB:,.0f} MB"
          f"   (baseline peak {BASELINE_PEAK_MB:,.1f} MB)")

    # Each group's operand footprint, used to decide whether it can afford
    # split_top=0. Scaled off the widest group, which is what MEASURED_MEM is.
    sizes = []
    for g in groups:
        elems = max(sa[0] * sa[1] + sb[0] * sb[1] for (_i, sa, sb, _f) in g)
        elems *= len(g)
        sizes.append(elems)
    widest = max(sizes)

    print(f"\n  {'group':>6}{'calls':>6}{'shape (m,k,n)':>22}{'GFLOP':>10}"
          f"{'operand MB':>12}{'vs widest':>11}")
    for gi, g in enumerate(groups):
        (_i, sa, sb, _f) = g[0]
        gf = sum(f for (_a, _b, _c, f) in g) / 1e9
        mb = sizes[gi] * 4 / 2 ** 20
        print(f"  {gi:>6}{len(g):>6}"
              f"{f'({sa[0]},{sa[1]},{sb[1]})':>22}{gf:>10.2f}"
              f"{mb:>12,.0f}{sizes[gi] / widest:>11.3f}")

    # ---- the frontier ------------------------------------------------------
    print("\n  FRONTIER: greedily give each group the CHEAPEST split_top that fits")
    print("  memory, then check the operation budget.")
    print(f"\n  {'L':>3}{'s=0 groups':>12}{'s=1 groups':>12}{'ops':>9}{'gate':>7}"
          f"{'peak MB':>10}{'mem':>6}{'chain B':>10}{'adj score':>12}")
    rows = []
    for L in (0, 1, 2, 3, 4):
        n0 = n1 = 0
        ops = base_ops
        worst_extra = 0.0
        infeasible = None
        for gi, g in enumerate(groups):
            frac = sizes[gi] / widest
            m0 = MEASURED_MEM[(L, 0)]
            m1 = MEASURED_MEM[(L, 1)]
            # A group smaller than the widest scales its extra proportionally.
            e0 = None if m0 is None else m0 * frac
            e1 = None if m1 is None else m1 * frac
            room = MEM_LIMIT_MB - BASELINE_PEAK_MB
            if e0 is not None and e0 <= room:
                n0 += 1
                ops += OPS_PER_INVOCATION[(L, 0)] - 1
                worst_extra = max(worst_extra, e0)
            elif e1 is not None and e1 <= room:
                n1 += 1
                ops += OPS_PER_INVOCATION[(L, 1)] - 1
                worst_extra = max(worst_extra, e1)
            else:
                infeasible = gi
                break
        if infeasible is not None:
            print(f"  {L:>3}   group {infeasible} fits at no split_top: INFEASIBLE")
            continue
        flops = used - mm_flops + mm_flops * FLOP_RATIO[L]
        mult = max(0.1, flops / B_FLOPS)
        peak = BASELINE_PEAK_MB + worst_extra
        print(f"  {L:>3}{n0:>12}{n1:>12}{ops:>9,}"
              f"{'ok' if ops <= gate else 'OVER':>7}{peak:>10,.0f}"
              f"{'ok' if peak <= MEM_LIMIT_MB else 'OVER':>6}"
              f"{flops / B_FLOPS:>10.5f}{mult * GRADED_RAW:>12.4e}")
        rows.append({"L": L, "groups_split0": n0, "groups_split1": n1,
                     "ops": ops, "ops_ok": ops <= gate,
                     "peak_mb": peak, "mem_ok": peak <= MEM_LIMIT_MB,
                     "chain_flops": flops, "chain_B": flops / B_FLOPS,
                     "adj_score_at_graded_raw": mult * GRADED_RAW})

    print("\n  READ THIS BEFORE QUOTING THE TABLE.")
    print("  * 'adj score' holds the raw MSE at the GRADED 3.7793e-08 of 330212")
    print("    and moves ONLY the multiplier. Strassen is exact in real")
    print("    arithmetic; float32 parity is a SEPARATE measurement and the probe")
    print("    already reads 1.3e-06 relative at L=3, against a 1e-06 bar.")
    print("  * the memory column is ACCOUNTING. The 8,192 MB limit is")
    print("    unenforceable on this host (no RLIMIT_AS on Windows) --")
    print("    `local_sim/linux_memory_gate.py` is the check. docs/GOTCHAS.md.")
    print("  * per-group memory is SCALED from one measured invocation, not")
    print("    measured per group. Confirm the chosen L before building on it.")

    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump({"candidate": a.candidate, "base_flops": used,
                       "base_ops": base_ops, "matmul_flops": mm_flops,
                       "n_groups": len(groups), "op_gate": gate,
                       "graded_s_per_op": per_op, "graded_raw": GRADED_RAW,
                       "measured_mem_mb": {f"L{k[0]}s{k[1]}": v
                                           for k, v in MEASURED_MEM.items()},
                       "ops_per_invocation": {f"L{k[0]}s{k[1]}": v
                                              for k, v in OPS_PER_INVOCATION.items()},
                       "frontier": rows}, fh, indent=2)
        print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
