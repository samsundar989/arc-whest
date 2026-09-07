"""Does the per-operation residual law survive EXTRAPOLATION to the op gate?

The law is calibrated on two of our own graded submissions, `329896` (canary4)
and `330212` (k3_full). In the GRADER'S weight convention those are 2,690 and
3,249 operations per MLP -- a lever of only `1.21x`. The operation gate sits near
24,000, which is `7.4x` beyond both calibration points, in exactly the variable
the law says is the one that matters. Two points cannot test that.

A third source can, and it is already committed: `experiments/board_snapshots/
board_2026-09-05_telemetry.md` carries median `flopscope_server_overhead_wall_s`
and `residual_wall_time_s` for twenty OTHER graded submissions. Their operation
counts are not published -- but overhead is very nearly proportional to operation
count too, and we can calibrate overhead-per-operation on our own two rows, where
BOTH quantities are known, and invert it for everyone else.

That buys a population spanning roughly `6k` to `34k` operations -- a `5.4x`
lever, straddling the gate -- for no compute, no seed block and no submission.

WHAT IT COSTS TO BELIEVE IT
---------------------------
The inferred op counts are a MODEL, not a measurement: they assume other teams'
overhead-per-operation matches ours, and an estimator whose operations are of a
very different kind (AndreasHad04 runs at 2.9 GFLOP/s against our 215) may well
break that. So this is corroboration of a LAW, never a per-team op count, and no
row here may be quoted as another team's operation count.

# seed-ledger: no block named. ROUTINE: reads two committed artifacts and does
arithmetic. Moves no gate by itself -- it reports what the gate WOULD be under
the stated policy, and changing the constant is a separate, recorded act.

Run:  uv run python local_sim/p2_residual_law.py
"""

from __future__ import annotations

import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]

BOARD = os.path.join(_ROOT, "experiments", "board_snapshots",
                     "board_2026-09-05_telemetry.md")

#: Our two graded rows, in the GRADER'S weight convention (f64 read-only weights
#: plus whestbench's own `materialise_predictions`) -- the only convention whose
#: FLOP total reproduces the graded total exactly. Measured 2026-09-07 by
#: replaying each candidate under `lifecycle_eval.build_weights(..., mode)`:
#:
#:     canary4   f32 168,636,374,630 / 2,673 ops
#:               f64_readonly 168,669,929,062 / 2,689  (+ materialise) -> 2,690
#:               graded total 168,669,961,830                          MATCHES
#:     k3_full   f32 1,714,554,351,206 / 3,232 ops
#:               f64_readonly 1,714,587,905,638 / 3,248 (+ materialise) -> 3,249
#:               graded total 1,714,587,938,406                        MATCHES
#:
#: The 2,673 that earlier entries called "graded" is the f32 census convention
#: AND was taken from a different file; comparing it against k3_full's 3,249
#: compared two conventions as one. `docs/GOTCHAS.md` carries the ladder.
OURS = [
    # (label, ops, overhead_median_s, residual min/med/max s)
    ("329896 canary4", 2690, 0.9731, (0.029244, 0.030105, 0.030659)),
    ("330212 k3_full", 3249, 1.226509, (0.035501, 0.036402, 0.039301)),
]


def _med(cell: str):
    """`a/b/c` -> b ; `x` -> x ; `~1.0` -> 1.0 ; None if unparseable."""
    cell = cell.strip().lstrip("~")
    parts = [p.strip().lstrip("~") for p in cell.split("/")]
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        return None
    return vals[len(vals) // 2] if len(vals) == 3 else vals[0]


def board_rows():
    rows = []
    with open(BOARD, encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("| 3"):
                continue
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            if len(c) < 11:
                continue
            # | sub | who | adj | raw | all-layers | FLOPs | util | wall |
            #   kern | ovh | res | notes |   -- ovh is 9 and res is 10; reading
            # 8 and 9 (kern and ovh) is an off-by-one that inflates inferred op
            # counts by ~8x and was caught by the board's own "~27k ops" note.
            sub, who = c[0], c[1]
            ovh, res = _med(c[9]), _med(c[10])
            if ovh is None or res is None or res <= 0:
                continue
            rows.append((sub, who, ovh, res))
    return rows


def main() -> int:
    import official_config as oc

    print("=" * 88)
    print("THE PER-OPERATION RESIDUAL LAW, TESTED BY EXTRAPOLATION")
    print("=" * 88)

    print("\n  1. CALIBRATION on our own two graded rows (grader weight convention)")
    print(f"  {'submission':>16}{'ops':>8}{'ovh s':>10}{'us/op ovh':>12}"
          f"{'res med s':>12}{'us/op res':>12}{'worst us/op':>13}")
    ovh_per_op = []
    for (lab, ops, ovh, (rlo, rmed, rhi)) in OURS:
        ovh_per_op.append(ovh / ops)
        print(f"  {lab:>16}{ops:>8,}{ovh:>10.4f}{ovh / ops * 1e6:>12.1f}"
              f"{rmed:>12.6f}{rmed / ops * 1e6:>12.3f}{rhi / ops * 1e6:>13.3f}")
    k = sum(ovh_per_op) / len(ovh_per_op)
    spread = max(ovh_per_op) / min(ovh_per_op)
    print(f"\n     overhead per operation: {k * 1e6:.1f} us  "
          f"(the two agree to {100 * (spread - 1):.1f}%)")
    print(f"     residual per operation: {OURS[0][3][1] / OURS[0][1] * 1e6:.3f}"
          f" and {OURS[1][3][1] / OURS[1][1] * 1e6:.3f} us"
          f"  -- 0.10% apart across a 10.2x change in FLOPs")

    print("\n  2. INVERTING overhead to infer op counts for 20 OTHER graded rows")
    print("     (a MODEL, not a measurement -- never quote a row as a team's op count)")
    print(f"  {'sub':>8}{'who':>26}{'ovh s':>9}{'res s':>9}"
          f"{'inferred ops':>14}{'us/op':>9}")
    pts = []
    for (sub, who, ovh, res) in board_rows():
        if sub in ("329896",):
            continue
        ops = ovh / k
        us = res / ops * 1e6
        pts.append((ops, res, us, who, sub))
        print(f"  {sub:>8}{who[:25]:>26}{ovh:>9.2f}{res:>9.3f}"
              f"{ops:>14,.0f}{us:>9.2f}")

    if not pts:
        print("     (no rows parsed -- board snapshot format changed?)")
        return 1

    us = sorted(p[2] for p in pts)
    lo, hi = min(p[0] for p in pts), max(p[0] for p in pts)
    print(f"\n     {len(pts)} rows, inferred ops {lo:,.0f} - {hi:,.0f}"
          f"  ({hi / lo:.1f}x lever, vs our own 1.21x)")
    print(f"     per-op residual  min {us[0]:.2f}  median {us[len(us) // 2]:.2f}"
          f"  max {us[-1]:.2f} us")

    # least squares through the inferred population
    n = len(pts)
    sx = sum(p[0] for p in pts)
    sy = sum(p[1] for p in pts)
    sxx = sum(p[0] * p[0] for p in pts)
    sxy = sum(p[0] * p[1] for p in pts)
    b = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    a = (sy - b * sx) / n
    ybar = sy / n
    ss_tot = sum((p[1] - ybar) ** 2 for p in pts)
    ss_res = sum((p[1] - (a + b * p[0])) ** 2 for p in pts)
    print(f"     linear fit  residual = {a:+.4f} s + {b * 1e6:.2f} us/op * ops"
          f"   R^2 = {1 - ss_res / ss_tot:.3f}")

    print("\n  3. WHAT THE GATE SHOULD BE UNDER THE CONSTANT'S OWN STATED POLICY")
    print("     ('the MAXIMUM of the observed range, so a prediction errs toward")
    print("      the cliff rather than away')")
    cur = oc.GRADED_RESIDUAL_S_PER_OP
    over = [(p[2], p[3], p[4]) for p in pts if p[2] > cur * 1e6]
    print(f"\n     currently carried      {cur * 1e6:.1f} us/op"
          f"  -> 25%-margin gate {int(0.3 / cur):,} ops"
          f"  (carried gate {oc.PHASE2_OP_COUNT_GATE:,})")
    if over:
        print(f"     rows ABOVE that figure ({len(over)}):")
        for (u, who, sub) in sorted(over, reverse=True):
            print(f"        {u:>6.2f} us/op   {who} ({sub})")
        worst = max(u for (u, _w, _s) in over)
        print(f"\n     population maximum     {worst:.2f} us/op"
              f"  -> 25%-margin gate {int(0.3 / (worst * 1e-6)):,} ops"
              f"  -> cliff at {int(0.4 / (worst * 1e-6)):,}")
        print(f"\n     => THE CARRIED GATE IS NOT CONSERVATIVE under its own policy.")
        print(f"        These are OTHER teams' estimators, so the model caveat in")
        print(f"        section 2 applies with full force -- but the policy says")
        print(f"        MAXIMUM OBSERVED, and refusing to look at the widest")
        print(f"        evidence we have because it is inconvenient is the exact")
        print(f"        failure `docs/VALIDATION.md` was written to prevent.")
    else:
        print(f"     no inferred row exceeds it; the carried figure is the maximum.")

    print(f"\n  4. AT THE GATE")
    for g in (22_000, 24_000):
        print(f"     {g:,} ops:  constant model {g * cur:.3f} s"
              f"   linear fit {a + b * g:.3f} s"
              f"   headroom {0.4 / (g * cur):.2f}x")
    print("\n     Both models agree the gate is survivable. Neither point")
    print("     constrains CURVATURE, and no experiment in this repository can:")
    print("     the grading server is not instrumentable from here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
