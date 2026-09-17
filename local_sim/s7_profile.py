"""Where do the counted FLOPs go? Label-free per-op profile of candidates at 1024x16.

Same canonical network as `s2_cost_meter.py` (seed 90_000_016_001, outside every
block), MLP index 1 (steady state), f64 read-only weights. Aggregates `op_log` by
operation name and SYMBOLIC shape (n = width, r = the arm's shared rank, k*n widths
written as `Kn`), so the step-3 cost can be attributed to transports, formings,
contractions, joins and rotations. Diagnostic only.

    uv run python local_sim/s7_profile.py --candidate estimator_k3_v36_v32_a4r384.py --out <json>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]

SEED = 90_000_016_001      # seed-ledger: no block named (as s2_cost_meter)
WIDTH, DEPTH = 1024, 16


def sym(d, n, r):
    if d == n:
        return "n"
    if r and d == r:
        return "r"
    if d % n == 0:
        return "%dn" % (d // n)
    if r and d % r == 0:
        return "%dr" % (d // r)
    return str(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import numpy as np
    import flopscope as flops
    import k4feed_oracle as K


    from mint_provenance import canonical_mint
    flops.configure(symmetry_warnings=False)
    result = {}
    for cand in a.candidate:
        mod = K._load(os.path.join(_ROOT, cand))
        r = int(getattr(mod, "_SHARED_RANK", 0)) if getattr(mod, "_SHARED_ON", False) else 0
        import whestbench

        class _Ctx:
            seed = SEED
            scratch_dir = submission_dir = _ROOT
            width, depth, flop_budget, n_mlps = WIDTH, DEPTH, 2 ** 41, 2
        est = mod.Estimator()
        est.setup(_Ctx())
        w, es, _ = canonical_mint(WIDTH, DEPTH, SEED + 1)   # MLP index 1, as the meter
        weights = []
        for x in w:
            arr = np.array(x, dtype=np.float64, copy=True)
            arr.setflags(write=False)
            weights.append(flops.numpy.asarray(arr))
        mlp = whestbench.MLP(weights=weights, width=WIDTH, depth=DEPTH, seed=es)
        with flops.BudgetContext(2 ** 41, quiet=True) as ctx:
            est.predict(mlp, 2 ** 41)
            log = list(ctx.op_log)
        agg = defaultdict(lambda: [0, 0])
        for rec in log:
            shp = "x".join("(" + ",".join(sym(int(d), WIDTH, r) for d in s) + ")"
                           for s in (rec.shapes or ()) if isinstance(s, tuple))
            key = "%s %s" % (rec.op_name, shp)
            agg[key][0] += int(rec.flop_cost)
            agg[key][1] += 1
        tot = sum(v[0] for v in agg.values())
        rows = sorted(({"op": k, "flops": v[0], "calls": v[1], "share": v[0] / tot} for k, v in agg.items()),
                      key=lambda x: -x["flops"])
        result[cand] = {"total_flops": tot, "c_over_b": tot / 2 ** 41, "ops": len(log), "top": rows[:40]}
        print("\n%s  C/B %.5f  ops %d" % (cand, tot / 2 ** 41, len(log)))
        for row in rows[:25]:
            print("  %6.2f%%  %8.5f B  %5d  %s" % (100 * row["share"], row["flops"] / 2 ** 41, row["calls"], row["op"]))
        with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(result, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
