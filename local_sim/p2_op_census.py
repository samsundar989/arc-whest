"""Where do a candidate's OPERATIONS go, and what would a 45% cut have to remove?

Opened 2026-09-04 to SIZE the op work before the canary's graded telemetry lands,
so the strategist's `< 2.5x` branch is actionable the day it fires rather than
four days later. **It moves no gate, spends no seed block, and changes no shipped
file.** `# seed-ledger: no block named` -- `canonical_mint` is called on a fixed
seed outside every registered block, purely to give `predict` something to run on.

WHY OPERATIONS AND NOT FLOPS
----------------------------
Phase 2 sets `lambda = 0`, so counted FLOPs below the `0.1` multiplier floor are
free and the only cost that can still ruin a submission is the **`0.4 s` per-MLP
residual cliff, which FAILS the MLP**. Write-up 6.8 established what that cliff is
actually paid in, on the one clean coefficient this project owns: the evaluator's
mean-residual ratio between two of our files was `5.82x` where their operation
ratio was `5.98x` -- agreement to `3%`, where every attempt to establish the same
thing from local wall-clock had been contaminated by host load.

So: **count the operations.** This module reads `BudgetContext.op_log` directly,
which carries per-operation `flopscope_overhead_duration_s` -- the meter's own
attribution of the time that is NOT backend arithmetic, i.e. the residual itself.
That makes the attribution measured rather than modelled, on this host.

**The wall-clock columns are NOT a residual prediction and no decision may be made
on them.** `docs/GOTCHAS.md`: the deployment authority is `whest run --runner
subprocess` on the organisers' data, two runs of the same evaluator on files
differing by `2.3%` in op count reported mean residuals differing by `31%`, and
this host reads high besides. **The COUNT column is the decision column**; the
time columns are here to say which operations are unusually expensive per call,
which a count alone cannot.

THE ARITHMETIC THIS IS PREPARING
--------------------------------
`experiments/2026-09-03_wp17_official/official_residual.json` -- ONE official
100-row run, snapshot `22db10a`, `2,646` operations per MLP:

    worst-row residual   0.1797 s   ->   headroom  2.226x   against the 0.4 s cliff
    median                0.1537 s
    mean headroom                          2.595x
    failures                               0 / 100

The guarded canary bills `2,674` operations (`+1.06%`), so its expected worst-row
headroom is about `2.20x`, and the nomination bar is `>= 4x`. Reaching it needs
the worst-row residual multiplied by `2.20 / 4.0 = 0.55`.

Fitting the two graded points we have -- `(442 ops, 0.0264 s)` for the shipped
second-order file and `(2646 ops, 0.1537 s)` for the canary, both medians, both
from the evaluator -- gives `residual ~ 0.00087 + 5.776e-5 * ops` seconds, i.e.
**essentially proportional with a sub-millisecond fixed cost**. With the observed
worst/median ratio of `1.169`, `>= 4x` headroom needs

    1.169 * (0.00087 + 5.776e-5 * N)  <=  0.100      ->      N  <=  ~1,465 ops

**a `45%` cut from `2,674`**, which is above the `> 40%` the ruling names as the
abandon threshold. **That is the number this census exists to make targetable**,
and it is a projection from two graded points, not a measurement of a file that
exists. It is stated so it can be falsified by the canary's own telemetry.

Run:
    uv run python local_sim/p2_op_census.py --candidate estimator_wp18_canary.py
    uv run python local_sim/p2_op_census.py --candidate estimator.py --top 25
    uv run python local_sim/p2_op_census.py --candidate estimator_wp18_canary.py \
        --out experiments/<slice>/op_census.json
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# APPEND, never insert: `local_sim/` holds numpy shims named `flopscope` and
# `whestbench`, and prepending this directory would meter the shim.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]

import numpy as np  # noqa: E402

WIDTH, DEPTH = 1024, 16
BUDGET = 2 ** 41

#: `experiments/2026-09-03_wp17_official/official_residual.json`, gpuq 182.
GRADED = {"snapshot": "22db10a", "ops": 2646, "n_mlps": 100,
          "worst_residual_s": 0.1797, "median_residual_s": 0.1537,
          "worst_headroom_x": 2.2262120890959607,
          "mean_headroom_x": 2.5951266374968225}
#: The guarded canary's own metered op count, `certify_guarded.json`.
GUARDED_OPS = 2674
#: The nomination bar the strategist's decision table is written on.
NOMINATION_HEADROOM_X = 4.0


def _require_real_flopscope(flops) -> None:
    path = os.path.abspath(getattr(flops, "__file__", "") or "")
    if path.startswith(os.path.abspath(_HERE) + os.sep):
        raise RuntimeError(
            f"REFUSED: `flopscope` resolved to the local shim at {path}.")


def _load(path):
    name = "cand_" + os.path.basename(path).replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class _Ctx:
    """The minimum `SetupContext` the candidates read."""

    def __init__(self, seed, tmp):
        self.seed = seed
        self.scratch_dir = tmp
        self.submission_dir = _ROOT


def _mint(seed, width, depth):
    import flopscope.numpy as fnp

    from mint_provenance import canonical_mint

    w, _e, _s = canonical_mint(width, depth, seed)
    return [fnp.array(np.ascontiguousarray(np.asarray(x, dtype=np.float32)))
            for x in w]


def census(path: str, *, width: int, depth: int, seed: int, mlp_index: int) -> dict:
    """Meter ONE `predict` and return its per-operation-group ledger.

    `mlp_index` exists because the grader constructs ONE `Estimator` and calls
    `predict` on every MLP of the split, so anything cached on the instance is
    paid by the FIRST MLP and by none of the others. `--mlp-index 1` reports the
    STEADY-STATE cost, which is what `100` graded rows are nearly all made of and
    therefore what the residual bar is really written on. `--mlp-index 0` reports
    the first row, which is the only one that pays for the cache.
    """
    import flopscope
    import whestbench

    _require_real_flopscope(flopscope)
    mod = _load(path)
    est = mod.Estimator()
    est.setup(_Ctx(seed, _ROOT))          # OUTSIDE the meter: its own 5 s window

    records = None
    for i in range(mlp_index + 1):
        weights = _mint(seed + i, width, depth)
        mlp = whestbench.MLP(weights=weights, width=width, depth=depth,
                             seed=seed + i)
        with flopscope.BudgetContext(BUDGET) as ctx:
            est.predict(mlp, BUDGET)
            used = int(ctx.flops_used)
            records = list(ctx.op_log)

    groups = collections.defaultdict(
        lambda: {"calls": 0, "flops": 0, "backend_s": 0.0, "overhead_s": 0.0})
    for rec in records:
        key = (rec.op_name, str(rec.resolved_dtype),
               "x".join("(" + ",".join(str(d) for d in sh) + ")"
                        for sh in rec.shapes))
        g = groups[key]
        g["calls"] += 1
        g["flops"] += int(rec.flop_cost)
        g["backend_s"] += float(rec.flopscope_backend_duration_s or 0.0)
        g["overhead_s"] += float(rec.flopscope_overhead_duration_s or 0.0)

    rows = [{"op": k[0], "dtype": k[1], "shapes": k[2], **v}
            for k, v in groups.items()]
    rows.sort(key=lambda r: (-r["calls"], -r["flops"]))
    return {"candidate": path, "width": width, "depth": depth, "seed": seed,
            "mlp_index": mlp_index, "flops": used,
            "n_ops": len(records), "groups": rows,
            "total_backend_s": sum(r["backend_s"] for r in rows),
            "total_overhead_s": sum(r["overhead_s"] for r in rows)}


def _target(ops_now: int) -> dict:
    """How many operations `>= 4x` headroom allows, from the two graded points.

    A PROJECTION, not a measurement. Two points, both medians, both from the
    evaluator: the shipped second-order file and the canary. Write-up 6.8 records
    the ratio pair that gives the shipped file's operating point
    (`5.82x` residual against `5.98x` operations).
    """
    ops_lo = GRADED["ops"] / 5.98
    res_lo = GRADED["median_residual_s"] / 5.82
    b = (GRADED["median_residual_s"] - res_lo) / (GRADED["ops"] - ops_lo)
    a = res_lo - b * ops_lo
    worst_over_median = GRADED["worst_residual_s"] / GRADED["median_residual_s"]
    allowed_worst = 0.4 / NOMINATION_HEADROOM_X
    n_max = ((allowed_worst / worst_over_median) - a) / b
    return {"fit_intercept_s": a, "fit_slope_s_per_op": b,
            "worst_over_median": worst_over_median,
            "allowed_worst_residual_s": allowed_worst,
            "max_ops_for_4x": n_max,
            "ops_now": ops_now,
            "required_cut_frac": 1.0 - n_max / float(ops_now),
            "abandon_threshold_frac": 0.40}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--candidate", default="estimator_wp18_canary.py")
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--depth", type=int, default=DEPTH)
    ap.add_argument("--seed", type=int, default=1_000_003)
    ap.add_argument("--mlp-index", type=int, default=1,
                    help="1 = steady state (what 100 graded rows are made of); "
                         "0 = the first MLP, the only one that pays for setup's "
                         "cache")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    c = census(a.candidate, width=a.width, depth=a.depth, seed=a.seed,
               mlp_index=a.mlp_index)
    tgt = _target(c["n_ops"])

    print("=" * 78)
    print(f"OPERATION CENSUS  --  {a.candidate}")
    print("=" * 78)
    print(f"  shape {a.width} x {a.depth}   MLP index {a.mlp_index} "
          f"({'steady state' if a.mlp_index else 'FIRST -- pays for the cache'})")
    print(f"  metered           {c['flops']:,} FLOPs   "
          f"{c['n_ops']:,} operations")
    print(f"  guarded canary    {GUARDED_OPS:,} operations "
          f"(certify_guarded.json)")
    print()
    print("  THE BAR, projected from the two graded points -- NOT a measurement")
    print(f"    graded worst-row headroom, {GRADED['n_mlps']} rows, snapshot "
          f"{GRADED['snapshot']}       {GRADED['worst_headroom_x']:.3f}x")
    print(f"    nomination bar                                        "
          f"{NOMINATION_HEADROOM_X:.1f}x")
    print(f"    fit: residual ~ {tgt['fit_intercept_s']*1e3:.2f} ms + "
          f"{tgt['fit_slope_s_per_op']*1e6:.2f} us/op ; worst/median "
          f"{tgt['worst_over_median']:.3f}")
    print(f"    operations that fit under {NOMINATION_HEADROOM_X:.1f}x           "
          f"           {tgt['max_ops_for_4x']:>8,.0f}")
    print(f"    required cut from {c['n_ops']:,}                              "
          f"     {tgt['required_cut_frac']*100:>7.1f}%")
    print(f"    the ruling abandons the flow above                     "
          f"{tgt['abandon_threshold_frac']*100:>7.1f}%")
    print(f"    -> {'REACHABLE' if tgt['required_cut_frac'] <= 0.40 else 'ABOVE THE ABANDON THRESHOLD'}"
          f" on this projection")
    print()

    rows = c["groups"]
    print(f"  {'ops':>6}{'%':>7}{'cum%':>7}  {'operation':<14}{'dtype':<9}"
          f"{'overhead ms':>13}{'shapes'}")
    print("  " + "-" * 92)
    cum = 0
    for r in rows[:a.top]:
        cum += r["calls"]
        print(f"  {r['calls']:>6}{100*r['calls']/c['n_ops']:>7.2f}"
              f"{100*cum/c['n_ops']:>7.2f}  {r['op']:<14}{r['dtype']:<9}"
              f"{r['overhead_s']*1e3:>13.2f}  {r['shapes'][:44]}")
    if len(rows) > a.top:
        rest = sum(r["calls"] for r in rows[a.top:])
        print(f"  {rest:>6}{100*rest/c['n_ops']:>7.2f}{100.0:>7.2f}  "
              f"{'(' + str(len(rows) - a.top) + ' further groups)':<23}")
    print()
    print(f"  {len(rows)} distinct (operation, dtype, shape) groups.")
    print(f"  metered backend {c['total_backend_s']*1e3:.1f} ms, "
          f"meter overhead {c['total_overhead_s']*1e3:.1f} ms "
          f"-- THIS HOST ONLY, not a residual prediction.")

    out = {"check": "operation census -- where a candidate's operations go, and "
                    "what a 4x-headroom cut would have to remove",
           "authority_note": "COUNT is the decision column. Wall-clock columns "
                             "are this host's and are NOT a residual prediction; "
                             "the authority is `whest run --runner subprocess`.",
           "graded_reference": GRADED, "guarded_canary_ops": GUARDED_OPS,
           "target": tgt, **c}
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
