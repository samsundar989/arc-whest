"""Did the weight-representation repair MOVE cost from residual into counted?

P2-P0 review, second Skeptic pass. `experiments/LOG.md`, 2026-08-24.

THE QUESTION
------------
P2-P0 changed how `lifecycle_eval` builds local MLP weights:

    weights=[np.asarray(x) for x in w_np]      ->   weights=[fnp.array(x) for x in w_np]

so that weight-METHOD operations are metered the way the grader meters them. The
shipped Phase-2 path opens with `[w.astype(fnp.float64) for w in mlp.weights]`, and
after the change that costs a measured `33,554,432` counted FLOPs where before it
cost zero.

Zero *counted* FLOPs. Not zero seconds. flopscope computes

    residual = wall - backend - overhead

so an expensive operation the meter cannot see does not vanish -- **it lands in
residual**. Sixteen 1024x1024 float32->float64 conversions per `predict` is real
work either way.

That predicts something specific and checkable: the repair should have moved a
large, fixed slab of time OUT of residual, at the same moment it moved
`37,779,456` FLOPs INTO counted. And the frozen P2-CP1 ladder, run on
2026-08-21 before the repair, does record much larger residuals than today:

    arm        2026-08-21 (pre-repair)      2026-08-24 (post-repair)
    k2p1       140.9 ms median              63.3 / 67.5 ms
    k2p2       260.7 ms median              102.4 / 93.7 ms

P2-P0's review entry attributed that gap to **machine contention**. That was
wrong -- the older numbers are a committed artifact from a different day, not a
contended re-run -- and this file exists to find out what the gap actually is.

WHY IT MATTERS BEYOND THE BOOKKEEPING
-------------------------------------
If the mechanism is the weight representation, then the repaired evaluator did not
merely start counting something: it moved a real cost from the half of `C` that
Phase 2 stopped charging (residual, `lambda = 0`) into the half it does charge
(counted FLOPs). The number to design the carrier against is then today's ~95 ms,
and the older figures are simply the wrong instrument.

If instead today's residual is low for some other reason -- the flopscope version,
or plain session drift -- then a **between-session term** is real and large
(2.2-2.7x), and any headroom claim that omits it is unsupported.

The two answers imply opposite protocols, which is what makes this worth two
minutes.

WHAT IT DOES
------------
Runs the SAME candidate, on the SAME host, in the SAME session, through
`lifecycle_eval`, twice -- once with each weight representation. Everything else is
held fixed, so the difference is the representation and nothing else. This is the
comparison the 2026-08-21-versus-today one could not be, because that one crossed
a stack, a representation, an instrument and three days at once.

Run on a QUIET machine:
    uv run python local_sim/p2_weight_repr_residual.py [--n-mlps 20]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np  # noqa: E402

OUT = os.path.join(_ROOT, "experiments", "2026-08-24_p2_p0",
                   "p2_weight_repr_residual.json")

#: Legacy-namespace throwaways. This file measures COST and decides nothing, so it
#: draws no registered block. seed-ledger: no block named
SEEDS_BASE = 5_151_000


def run(candidate: str, n_mlps: int, repr_mode: str) -> dict:
    """One pass of `lifecycle_eval.evaluate` with weights built one way or the other.

    The patch is applied to the module's own `harness.MLP` construction by
    swapping the array constructor `evaluate` uses, so the two arms differ in
    EXACTLY one expression and share every other line of the evaluator.
    """
    import flopscope.numpy as fnp

    import lifecycle_eval as le

    spec = le.LifecycleSpec.for_round("v2-phase2")
    seeds = [SEEDS_BASE + i for i in range(n_mlps)]
    truth = np.zeros((n_mlps, spec.width), dtype=np.float32)

    import harness

    real_mlp = harness.MLP
    ctor = np.asarray if repr_mode == "np" else fnp.array

    class _PatchedMLP(real_mlp):  # type: ignore[misc,valid-type]
        def __init__(self, *, weights, width, depth, seed):
            super().__init__(weights=[ctor(w) for w in weights],
                             width=width, depth=depth, seed=seed)

    harness.MLP = _PatchedMLP
    try:
        rows, _prov = le.evaluate(candidate, seeds, truth, spec=spec)
    finally:
        harness.MLP = real_mlp

    resid = np.asarray([r.residual_wall_time_s for r in rows], float)
    wall = np.asarray([r.wall_time_s for r in rows], float)
    flops = np.asarray([r.flops_used for r in rows], float)
    return {
        "repr": repr_mode,
        "candidate": candidate,
        "residual_s": resid.tolist(),
        "wall_s": wall.tolist(),
        "flops": flops.tolist(),
        "residual_median": float(np.median(resid)),
        "residual_max": float(resid.max()),
        "wall_median": float(np.median(wall)),
        "flops_median": float(np.median(flops)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-mlps", type=int, default=20)
    ap.add_argument("--candidates",
                    default="local_sim/p2_arms/arm_k2p2.py,local_sim/p2_arms/arm_k2p1.py")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    cands = [c.strip() for c in args.candidates.split(",") if c.strip()]
    print("=" * 96)
    print("WEIGHT REPRESENTATION vs RESIDUAL -- same host, same session, one line apart")
    print("=" * 96)
    print(f"  n = {args.n_mlps} per cell.  RUN THIS ON A QUIET MACHINE.")
    print()

    results = []
    for cand in cands:
        for mode in ("np", "fnp"):
            print(f"  {os.path.basename(cand):<16} weights={mode:<4} ...", flush=True)
            results.append(run(cand, args.n_mlps, mode))

    print()
    print(f"  {'candidate':<18}{'weights':>9}{'resid med':>12}{'resid max':>12}"
          f"{'wall med':>11}{'counted FLOPs':>18}")
    for r in results:
        print(f"  {os.path.basename(r['candidate']):<18}{r['repr']:>9}"
              f"{r['residual_median'] * 1e3:>10.1f}ms{r['residual_max'] * 1e3:>10.1f}ms"
              f"{r['wall_median'] * 1e3:>9.1f}ms{r['flops_median']:>18,.0f}")

    print()
    print("  THE COMPARISON")
    verdict = {}
    for cand in cands:
        a = next(r for r in results if r["candidate"] == cand and r["repr"] == "np")
        b = next(r for r in results if r["candidate"] == cand and r["repr"] == "fnp")
        d_res = a["residual_median"] - b["residual_median"]
        d_flops = b["flops_median"] - a["flops_median"]
        verdict[cand] = {
            "residual_np_s": a["residual_median"],
            "residual_fnp_s": b["residual_median"],
            "residual_drop_s": d_res,
            "residual_ratio_np_over_fnp": (a["residual_median"] / b["residual_median"]
                                           if b["residual_median"] else None),
            "counted_flops_added": d_flops,
        }
        print(f"    {os.path.basename(cand)}")
        print(f"      residual  np {a['residual_median'] * 1e3:.1f} ms  ->  "
              f"fnp {b['residual_median'] * 1e3:.1f} ms"
              f"   ({a['residual_median'] / b['residual_median']:.2f}x, "
              f"{d_res * 1e3:+.1f} ms)")
        print(f"      counted   +{d_flops:,.0f} FLOPs moved INTO the meter")

    print()
    print("  HOW TO READ IT")
    print("  - a LARGE residual drop with the +37,779,456 counted rise means the")
    print("    repair MOVED real cost from residual into counted. The 2026-08-21")
    print("    ladder's larger residuals are then the pre-repair instrument, not")
    print("    contention and not session drift.")
    print("  - a SMALL drop means the representation is not the mechanism, and the")
    print("    2.2-2.7x gap against 2026-08-21 is a genuine between-session term")
    print("    that any headroom claim must carry.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({
            "artifact": "p2_weight_repr_residual",
            "artifact_version": 1,
            "n_mlps": args.n_mlps,
            "runs": results,
            "verdict": verdict,
            "note": (
                "Same host, same session, same stack; the two arms differ in one "
                "expression. Draws no seed block and selects nothing. Wall-time "
                "absolutes are host-specific; the RATIO between the two arms is "
                "the quantity of interest and it is measured within one session "
                "by construction."
            ),
        }, fh, indent=2)
    print(f"\n  wrote {os.path.relpath(args.out, _ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
