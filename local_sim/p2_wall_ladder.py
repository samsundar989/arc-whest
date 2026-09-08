"""How much GRADED wall does each Strassen depth cost? The instrument the cost model lacked.

WHY THIS EXISTS
---------------
Submission `330290` (L=4) cleared the `120 s` per-MLP wall cap by `5.3 seconds` --
`1.046x`, or `1.23` standard deviations on its own 100 rows. The frozen cost model
priced FLOPs, operations, memory and residual and had **no wall term at all**,
because until Strassen every candidate here ran two orders of magnitude inside the
cap. Strassen inverts that: it trades few enormous matmuls for thousands of small
batched ones, so arithmetic intensity collapses and wall rises even as FLOPs fall.

    submission     FLOPs      kernel wall   GFLOP/s   worst predict wall
    330212  L=0  1,714.6 G       7.97 s      215.1         9.54 s
    330290  L=4  1,068.2 G      98.96 s       10.8       114.68 s

**The risk this measures is not hypothetical.** v2.7's IDENTICAL bytes were graded
twice, `328664` and `329889`. The median wall moved `-0.6%`; the MAXIMUM moved
`1.0527 -> 1.5979 s`, `+51.8%`, with 16 of 100 rows above `1.15x` the median in a
contiguous index window -- a server-contention episode, on the same 0.16.1 stack
`330290` ran on. Applied to a 105.7 s row that is `~169 s`, and those rows FAIL.
A failed MLP is zeroed at multiplier 1.0, which at our operating point is a
`~3e5` score penalty; L=4 buys only `1.106x` over L=3.

WHAT IT MEASURES, AND WHY IT NEEDS NO SEED BLOCK
------------------------------------------------
Local predict wall for `L in {0, 3, 4}` in ONE process, ONE host, ONE run, at the
live shape under `f64_readonly` weights -- the same instrument certify's
`metered_runs` used. We hold GRADED anchors at L=0 (`9.219 s` median) and L=4
(`105.716 s` median), so the two ends pin the local->graded map and L=3's graded
wall follows by interpolation instead of by a per-level guess.

**It estimates no accuracy quantity**, only wall, so it spends no seed block
(`# seed-ledger: no block named`) and moves no gate. It is the cheapest thing that
discriminates on which depth should be nominee A.

A LOCAL WALL IS NEVER THE AUTHORITY (`AGENTS.md`). It is used here only as a
RATIO between arms measured in the same process, which is the transferable form.

Run:  uv run python local_sim/p2_wall_ladder.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]

import numpy as np  # noqa: E402

WIDTH, DEPTH, BUDGET = 1024, 16, 2 ** 41

#: GRADED medians we already hold, seconds of `participant_predict_wall_s`.
GRADED_ANCHORS = {0: 9.219, 4: 105.716}
#: GRADED worst rows, seconds.
GRADED_WORST = {0: 9.540, 4: 114.680}
#: The contention factor observed on IDENTICAL bytes across two graded runs of
#: v2.7 (328664 -> 329889): the maximum row moved +51.8% while the median moved
#: -0.6%. This is what a wall margin has to survive.
CONTENTION_FACTOR = 1.60
WALL_CAP = 120.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--candidate", default="estimator_k3_strassen.py")
    ap.add_argument("--levels", default="0,3,4")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    levels = [int(x) for x in a.levels.split(",")]

    import flopscope
    import flopscope.numpy as fnp
    import whestbench

    from lifecycle_eval import build_weights
    from mint_provenance import canonical_mint
    from p2_wp21_score import _Ctx, _load

    mod = _load(a.candidate)
    est = mod.Estimator()
    est.setup(_Ctx(1_000_003))

    # Mint ONCE, outside the timing loop, so the weights are identical across
    # arms and the mint's own cost never enters a wall reading.
    nets = []
    for i in range(a.reps):
        w_np, _e, _s = canonical_mint(WIDTH, DEPTH, 1_000_003 + i)
        nets.append((1_000_003 + i, build_weights(w_np, "f64_readonly", fnp)))

    print("=" * 92)
    print("STRASSEN DEPTH vs WALL -- one process, one host, one run")
    print("=" * 92)
    print(f"  candidate  {a.candidate}   {WIDTH} x {DEPTH}   f64_readonly")
    print(f"  cap {WALL_CAP:.0f} s/MLP   graded anchors: "
          + ", ".join(f"L={k} {v:.2f} s" for k, v in sorted(GRADED_ANCHORS.items())))

    rows = []
    print(f"\n  {'L':>3}{'local wall s (min/med/max)':>34}{'FLOPs':>18}"
          f"{'B':>9}{'ops':>8}{'GFLOP/s':>10}")
    for L in levels:
        mod._STRASSEN_DEPTH = L
        walls, flops, ops = [], None, None
        for (seed, w) in nets:
            mlp = whestbench.MLP(weights=w, width=WIDTH, depth=DEPTH, seed=seed)
            t0 = time.perf_counter()
            with flopscope.BudgetContext(BUDGET) as ctx:
                est.predict(mlp, BUDGET)
                flops, ops = int(ctx.flops_used), len(ctx.op_log)
            walls.append(time.perf_counter() - t0)
        walls.sort()
        med = walls[len(walls) // 2]
        rows.append({"L": L, "local_min": walls[0], "local_med": med,
                     "local_max": walls[-1], "flops": flops + 32768,
                     "ops": ops + 1})
        print(f"  {L:>3}{walls[0]:>11.2f}{med:>11.2f}{walls[-1]:>12.2f}"
              f"{flops + 32768:>18,}{(flops + 32768) / 2 ** 41:>9.5f}"
              f"{ops + 1:>8,}{(flops + 32768) / 1e9 / med:>10.1f}")

    # ---- local -> graded, calibrated on the two anchors we actually hold ----
    have = [r for r in rows if r["L"] in GRADED_ANCHORS]
    print("\n  LOCAL -> GRADED MAP, from the anchors (never a bare local second)")
    if len(have) < 2:
        print("    need L=0 and L=4 in --levels to calibrate; skipping")
        return 0
    lo, hi = sorted(have, key=lambda r: r["L"])[0], sorted(have, key=lambda r: r["L"])[-1]
    for r in (lo, hi):
        print(f"    L={r['L']}  local {r['local_med']:7.2f} s ->"
              f" graded {GRADED_ANCHORS[r['L']]:7.2f} s"
              f"   ratio {GRADED_ANCHORS[r['L']] / r['local_med']:.3f}x")
    # log-linear in local wall: graded = exp(a + b*ln(local))
    import math
    b = ((math.log(GRADED_ANCHORS[hi["L"]]) - math.log(GRADED_ANCHORS[lo["L"]]))
         / (math.log(hi["local_med"]) - math.log(lo["local_med"])))
    aa = math.log(GRADED_ANCHORS[lo["L"]]) - b * math.log(lo["local_med"])
    print(f"    fit: graded = exp({aa:+.4f}) * local^{b:.4f}")

    print(f"\n  {'L':>3}{'projected graded median':>26}{'projected worst':>18}"
          f"{'margin':>9}{'margin x1.60 contention':>26}{'verdict':>10}")
    for r in rows:
        g = math.exp(aa + b * math.log(r["local_med"]))
        # worst/median from the graded runs we hold: 1.035 (L=0), 1.085 (L=4)
        wm = 1.085
        worst = g * wm
        m = WALL_CAP / worst
        mc = WALL_CAP / (worst * CONTENTION_FACTOR)
        v = "SAFE" if mc >= 1.0 else ("THIN" if m >= 1.15 else "AT RISK")
        print(f"  {r['L']:>3}{g:>26.1f}{worst:>18.1f}{m:>9.2f}x{mc:>25.2f}x{v:>10}")
        r.update({"projected_graded_med": g, "projected_graded_worst": worst,
                  "margin": m, "margin_under_contention": mc, "verdict": v})

    print(f"\n  'margin x1.60 contention' applies the factor measured on IDENTICAL")
    print(f"  bytes graded twice (328664 -> 329889: worst row +51.8%). A depth that")
    print(f"  is under 1.0x there can FAIL rows on a busy grading server, and a")
    print(f"  failed MLP is zeroed at multiplier 1.0 -- a ~3e5 score penalty against")
    print(f"  a 1.106x gain from L=4 over L=3.")

    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump({"candidate": a.candidate, "reps": a.reps,
                       "graded_anchors": GRADED_ANCHORS,
                       "contention_factor": CONTENTION_FACTOR,
                       "wall_cap_s": WALL_CAP, "arms": rows}, fh, indent=2)
        print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
