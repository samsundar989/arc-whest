"""Mutation battery for invariant 9, the permutation-equivariance item.

WHY THIS EXISTS
---------------
On 2026-09-03 invariant 9's absolute clause `d_k / scale < 1e-6` was replaced by
"every row's drift, in MSE units, <= 1% of the candidate's own raw MSE"
(strategist Ruling 1; `experiments/LOG.md`, *IMPORTANT (instrument)*). The old
clause was an f64-K2-era float32-precision bar, and the live design runs the
covariance backbone in f32 on purpose -- curing it was measured at exactly 2.0x
billing, which is 1.53x worse score. So the clause did not ask a candidate to be
correct, it asked it to be expensive.

**An amendment that makes a failing gate pass has to prove it still bites, or it
is not an amendment, it is a hole.** This file is that proof, and it is a gate:
it exits non-zero if any deliberately broken prediction is ACCEPTED by
`edge_cases.perm_gate`, which is the same function invariant 9 itself calls, so
the two cannot drift apart.

It also measures and prints the thing the amendment genuinely costs. **The first
version of this file promised that and did not deliver it** -- it printed the new
floor and never the old one, and every mutation in the battery sits far above
the band between them, so the battery was silent exactly where the change did
something. An independent Skeptic found that on 2026-09-03 and measured the gap
at **794x-1,585x**. `old_bar_floor()` now computes both floors on the same ramp
and prints the ratio, so the cost is a number in the output rather than a
sentence in a docstring.

What the amendment buys back is smaller than the first version of this file
claimed. The `other rows` and `final row` clauses are new, but they leave the
canary `1e5x` of slack and v2.6.1 `1e9x`: they discriminate nothing, and calling
them an offset to the row-`k` widening does not survive arithmetic. The honest
statement is that the item is now MEETABLE by a candidate that is correct rather
than only by one that is expensive, at the price of a 3-orders-of-magnitude
coarser floor -- and that `docs/VALIDATION.md` section 3.2 had to be rewritten,
because it assigned invariant 9 a second job that lived entirely inside the
band that was given away.

The battery is deliberately run on SMALL networks (128x8, 64x4). Detection here
is a property of the gate arithmetic, not of the shape, and the invariant itself
already runs the live shapes.

Run:  uv run python local_sim/test_edge_cases_inv9.py [candidate.py]
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import edge_cases as ec  # noqa: E402

#: Small, and both shapes are off the 1024x16 calibration path on purpose.
CASES = ((8, 128, 30_806_006_013, 3), (4, 64, 30_806_006_015, 1))


def _truth_pair(est, depth, width, seed, k, rng):
    """The honest (base, perm, P) triple for one case -- no mutation yet."""
    m = ec.canon(depth, width, seed)
    base = np.asarray(est.predict(m, ec.BUDGET), dtype=np.float64)
    P = rng.permutation(width)
    w = [np.array(x) for x in m.weights]
    w[k] = w[k][:, P]
    w[k + 1] = w[k + 1][P, :]
    perm = np.asarray(est.predict(ec.MLP(w, seed=m.seed), ec.BUDGET),
                      dtype=np.float64)
    return base, perm, P


# ---------------------------------------------------------------------------
# The mutations. Each is a defect applied to the estimator's OUTPUT in a fixed,
# label-blind way -- which is exactly what a transposed index or a swapped
# reduction axis is: an operation that does not commute with relabelling the
# neurons. Applying it to both `base` and `perm` is the correct model, because a
# real defect is in the code and fires on every network it sees.
# ---------------------------------------------------------------------------

def mut_row_reversed(out, k):
    """A transposed index: row k comes out in reversed neuron order."""
    o = out.copy()
    o[k] = o[k][::-1]
    return o


def mut_positional_ramp(out, k):
    """An axis swapped in a reduction: row k scaled by a POSITIONAL ramp."""
    o = out.copy()
    n = o.shape[1]
    o[k] = o[k] * (1.0 + 0.10 * np.arange(n) / n)
    return o


def mut_depth_roll(out, k):
    """An off-by-one on the DEPTH axis: every row shifted down by one."""
    return np.roll(out, 1, axis=0)


def mut_fixed_shuffle(out, k):
    """A basis applied on the wrong side: row k hit by a FIXED permutation."""
    o = out.copy()
    n = o.shape[1]
    q = np.random.default_rng(20_260_903).permutation(n)
    o[k] = o[k][q]
    return o


def mut_swap_adjacent(out, k):
    """The minimal index bug: positions 0 and 1 of row k exchanged.

    An off-by-one WITHIN a row, and the smallest structural defect there is --
    two entries, adjacent, fixed positions. It is positional, so it does not
    commute with relabelling and invariant 9 must see it.
    """
    o = out.copy()
    o[k, [0, 1]] = o[k, [1, 0]]
    return o


#: Positions chosen once, from the first prediction the battery sees, and then
#: applied POSITIONALLY -- see `_choose_closest_positions`.
_CLOSEST = {}


def _choose_closest_positions(base, k):
    """The two POSITIONS in row k whose base values are closest."""
    row = base[k]
    order = np.argsort(row)
    i = int(np.argmin(np.diff(row[order])))
    return int(order[i]), int(order[i + 1])


def mut_swap_closest_positional(out, k):
    """Swap the two POSITIONS whose base values are closest.

    Not the true worst case, and the printed margin says so: swapping positions
    `a,b` moves FOUR neurons once the relabelling is applied (a and b in the
    unpermuted row, and whatever P sends there), so the deviation is set by the
    row's spread, not by the chosen gap. The authoritative floor is the probe
    below, which isolates a genuine two-entry swap.

    The positions are chosen once from the unmutated prediction and then applied
    positionally to both predictions, which is what a defect in the code does.
    Choosing them by value and re-choosing per prediction -- which the first
    draft of this file did -- builds a mutation that COMMUTES with relabelling
    and is therefore invisible by construction. That version is kept below as
    `ctl_value_selected_swap`, because a battery has to contain at least one
    thing invariant 9 is right not to catch.
    """
    a, b = _CLOSEST[k]
    o = out.copy()
    o[k, [a, b]] = o[k, [b, a]]
    return o


def ctl_value_selected_swap(out, k):
    """CONTROL, and it must be ACCEPTED.

    Swap the two closest-VALUED neurons, re-selected per prediction. Selection
    by value is equivariant, so the whole defect commutes with relabelling and
    invariant 9 cannot see it -- correctly. This is the boundary of what the
    item claims: it detects defects that are POSITIONAL, not defects that are
    merely wrong.
    """
    o = out.copy()
    row = o[k]
    order = np.argsort(row)
    i = int(np.argmin(np.diff(row[order])))
    a, b = int(order[i]), int(order[i + 1])
    o[k, [a, b]] = o[k, [b, a]]
    return o


def ctl_global_scale(out, k):
    """CONTROL, and it must be ACCEPTED.

    A 0.1% multiplicative error on the whole output is badly WRONG, and
    invariant 9 must not notice, because it commutes with relabelling.
    Invariant 9 tests equivariance; accuracy is invariants 8 and 13's job. A
    battery in which every mutation fails proves nothing about specificity.
    """
    return out * 1.001


def mut_nan_final(out, k):
    """A NaN on the SCORED final row. Must be REJECTED.

    `SKEPTIC` (2026-09-03) found that `perm_gate` accepted exactly this: `max()`
    orders with `>`, `nan > x` is False, so a NaN anywhere but the first position
    was dropped before the finiteness check saw it, and the gate returned
    `worst = 0.0, ok = True`. The pre-amendment code rejected it. The candidate
    family under test is the one that returns NaN on degenerate inputs, and a
    NaN produced only on the PERMUTED network is seen by no other invariant --
    so this case is not hypothetical, it is the one that mattered.
    """
    o = out.copy()
    o[-1, 0] = np.nan
    return o


def mut_nan_rowk(out, k):
    """A NaN on the permuted row. Must be REJECTED."""
    o = out.copy()
    o[k, 0] = np.nan
    return o


def mut_inf_other(out, k):
    """An infinity on a row that is neither k nor final. Must be REJECTED."""
    o = out.copy()
    j = 0 if k != 0 else 1
    if j == out.shape[0] - 1:
        j = max(0, out.shape[0] - 2)
    o[j, 0] = np.inf
    return o


MUTATIONS = (
    ("transposed index (row reversed)", mut_row_reversed, True),
    ("reduction axis swapped (positional ramp)", mut_positional_ramp, True),
    ("off-by-one on the depth axis", mut_depth_roll, True),
    ("basis on the wrong side (fixed shuffle)", mut_fixed_shuffle, True),
    ("minimal index bug (adjacent positions 0<->1)", mut_swap_adjacent, True),
    ("closest-valued positions swapped",
     mut_swap_closest_positional, True),
    ("NaN on the SCORED final row", mut_nan_final, True),
    ("NaN on the permuted row k", mut_nan_rowk, True),
    ("inf on a non-k, non-final row", mut_inf_other, True),
    ("CONTROL: equivariant 0.1% scale error", ctl_global_scale, False),
    ("CONTROL: value-selected swap (commutes)", ctl_value_selected_swap, False),
)


def positional_ramp(out, k, amp):
    """`mut_positional_ramp` with the amplitude exposed, for the floor sweep."""
    o = out.copy()
    n = o.shape[1]
    o[k] = o[k] * (1.0 + amp * np.arange(n) / n)
    return o


def detection_floor(base, perm, P, k, depth, denom):
    """The smallest POSITIONAL distortion this gate still rejects.

    The first draft asked a different and unrealisable question -- "the smallest
    two-entry swap" -- and its probe kept rejecting at half the claimed floor.
    The reason is worth keeping: a positional swap applied to BOTH predictions
    moves FOUR neurons once the relabelling is applied (positions 0 and 1, and
    whichever neurons P sends there), so its deviation is set by the row's
    spread and never by the chosen gap. There is no realisable defect whose
    deviation is a free small parameter of that shape.

    A positional RAMP is the realisable one. Scaling `row * (1 + a*t)` with
    `t = arange(n)/n` gives a deviation `v[P] * a * (t - t[P])`, so `mse_k` is
    exactly quadratic in `a` and the floor has a closed form:
    `a* = a_ref / sqrt(mse_ref / bar)`. That is the number this returns, and it
    is then CHECKED at 0.5x and 2.0x rather than asserted.
    """
    bar = ec.DRIFT_SHARE * denom
    a_ref = 0.10
    g = ec.perm_gate(positional_ramp(base, k, a_ref),
                     positional_ramp(perm, k, a_ref), P, k, depth, denom=denom)
    a_star = a_ref / float(np.sqrt(g["mse_k"] / bar))
    return a_star, g["mse_k"]


def old_bar_floor(base, perm, P, k, depth):
    """The amplitude at which the RETIRED clause `d_k/scale < 1e-6` rejected.

    `SKEPTIC` (2026-09-03): the battery's own docstring promised to report what
    the amendment surrendered and did not, and every one of its mutations sits
    far ABOVE the surrendered band, so the battery was silent exactly where the
    change did something. This closes that gap by measuring both floors on the
    same ramp and printing the ratio.

    The old clause was `max|perm[k] - base[k][P]| / mean|base[-1]| < 1e-6`. Under
    `row * (1 + a*t)` the deviation is `v[P] * a * (t - t[P])`, linear in `a`, so
    `a*_old = a_ref * 1e-6 * scale / d_k(a_ref)`.
    """
    a_ref = 0.10
    b2 = positional_ramp(base, k, a_ref)
    p2 = positional_ramp(perm, k, a_ref)
    d_k = float(np.abs(p2[k] - b2[k][P]).max())
    scale = float(np.abs(base[-1]).mean())
    return a_ref * 1e-6 * scale / max(d_k, 1e-300)


def main():
    cand = sys.argv[1] if len(sys.argv) > 1 else "estimator.py"
    est = ec.load(cand).Estimator()

    class Ctx:
        width, depth, flop_budget, seed = 256, 32, ec.BUDGET, 0

    est.setup(Ctx())

    denom, denom_src = ec.raw_mse_denominator()
    bar = ec.DRIFT_SHARE * denom
    print("MUTATION BATTERY for invariant 9 -- candidate: %s" % cand)
    print("gate: edge_cases.perm_gate, bar = %.0f%% of raw MSE = %.3e  [%s]\n"
          % (ec.DRIFT_SHARE * 100, bar, denom_src))

    rng = np.random.default_rng(30_806_006_100)
    failures, surrendered = [], []
    for depth, width, seed, k in CASES:
        base, perm, P = _truth_pair(est, depth, width, seed, k, rng)

        clean = ec.perm_gate(base, perm, P, k, depth, denom=denom)
        tag = "%dx%d layer %d" % (width, depth, k)
        print("%s   UNMUTATED   worst %.3e = %.4f%% of bar   %s"
              % (tag, clean["worst"], clean["worst"] / bar * 100,
                 "accepted" if clean["ok"] else "REJECTED"))
        if not clean["ok"]:
            failures.append("%s: the UNMUTATED prediction was rejected" % tag)

        a_star, mse_ref = detection_floor(base, perm, P, k, depth, denom)
        a_old = old_bar_floor(base, perm, P, k, depth)
        print("%s   detection floor: a POSITIONAL distortion is rejected once "
              "it exceeds %.3e = %.4f%% of the row" % (tag, a_star, a_star * 100))
        print("%s   THE SENSITIVITY SURRENDERED: the retired clause rejected at "
              "%.3e (%.6f%%); the amendment gives away a factor of %.0fx"
              % (tag, a_old, a_old * 100, a_star / max(a_old, 1e-300)))
        surrendered.append((tag, a_old, a_star, a_star / max(a_old, 1e-300)))

        for mult, want_reject in ((0.5, False), (2.0, True)):
            gg = ec.perm_gate(positional_ramp(base, k, a_star * mult),
                              positional_ramp(perm, k, a_star * mult),
                              P, k, depth, denom=denom)
            got_reject = not gg["ok"]
            okf = (got_reject == want_reject)
            print("%s   floor probe amp = %.2fx floor -> mse_k %.3e, %s   %s"
                  % (tag, mult, gg["mse_k"],
                     "REJECTED" if got_reject else "accepted",
                     "ok" if okf else "*** FLOOR MISSTATED ***"))
            if not okf:
                failures.append("%s: floor probe at %.2fx behaved wrongly"
                                % (tag, mult))

        _CLOSEST[k] = _choose_closest_positions(base, k)
        print()

        for name, fn, must_reject in MUTATIONS:
            if name.startswith(("NaN", "inf")):
                # A non-finite value must be caught wherever it appears, so it
                # is injected into the PERMUTED prediction only -- injecting it
                # into both would leave the DIFFERENCE finite on some rows and
                # test the wrong thing.
                b2, p2 = base, fn(perm, k)
            else:
                b2, p2 = fn(base, k), fn(perm, k)
            g = ec.perm_gate(b2, p2, P, k, depth, denom=denom)
            margin = g["worst"] / bar if np.isfinite(g["worst"]) else float("inf")
            verdict = "REJECTED" if not g["ok"] else "accepted"
            want = "REJECTED" if must_reject else "accepted"
            good = (verdict == want)
            print("  %-42s worst %.3e = %8.3gx bar   %-8s  %s"
                  % (name, g["worst"], margin, verdict,
                     "ok" if good else "*** WRONG, wanted %s ***" % want))
            if not good:
                failures.append("%s / %s: got %s, wanted %s"
                                % (tag, name, verdict, want))
        print()

    if failures:
        print("BATTERY FAILED -- %d problem(s):" % len(failures))
        for f in failures:
            print("  - %s" % f)
        print("\nInvariant 9's amendment is NOT legitimate as it stands. Revert "
              "it (LOG, IMPORTANT (instrument), 'what would make me revert').")
        return 1

    print("SURRENDERED SENSITIVITY, and it is the honest cost of the amendment:")
    for tag, a_old, a_new, ratio in surrendered:
        print("  %-18s old %.3e -> new %.3e   = %.0fx given away"
              % (tag, a_old, a_new, ratio))
    print("  Every mutation above sits FAR ABOVE this band, so the battery says "
          "nothing about\n  what lives inside it. docs/VALIDATION.md section 3.2 "
          "names one thing that does\n  (rotation stability at 4.4e-08) and it "
          "has been rewritten to record that this\n  instrument no longer "
          "measures it.\n")
    print("BATTERY PASSED -- every deliberate equivariance defect is still "
          "rejected, the equivariant control is still accepted, and the "
          "unmutated prediction passes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
