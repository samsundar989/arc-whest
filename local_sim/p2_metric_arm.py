"""Does `use_avg_metric=False` buy accuracy AT THE LIVE SHAPE, on the kind that ships?

Contract: `experiments/LOG.md`, 2026-09-10, *"CONTRACT (Rule 1): the exact-metric
arm at 1024x16"*, committed before this file was run.

WHERE THIS CAME FROM
---------------------
The Slice-4 ladder (job `474`) found that replacing upstream's averaged metric
`E[W W^T] = 2I` with the exact `W W^T` is worth a lot, and worth MORE on
`kind=SIMPLE` -- the only kind with a budget -- than on AUGMENT:

    exact-metric / avg      n=128 L=2   n=192 L=3   n=256 L=4
    AUGMENT                    0.778       0.764       0.766
    SIMPLE                     0.595       0.686       0.763

**But the SIMPLE row is TRENDING and the AUGMENT row is not.** As a gain that is
`1.68x`, `1.46x`, `1.31x` -- shrinking monotonically with width, and the live
shape is four times wider again. The ladder has already been shown unable to
extrapolate (its own proxy check fails by ~40% at the one calibrated point, and
it contains a method ratio that provably moves with `n` at fixed `L/n`). So the
ladder cannot say whether anything survives at `1024 x 16`.

**This module asks the live shape directly.** It is the cheapest possible form of
the question: two arms, same 24 P01 networks, same labels, one flag apart.

WHY IT IS WORTH ASKING BEFORE ANY COST WORK
---------------------------------------------
`use_avg_metric=False` is NOT free and its cost is awkward in our chain:
the metric becomes a real `(n, n)` object needing `W metric W^T` each layer
(~`0.031 B` over 16 layers, cheap), but `metric.ndim == 2` ALSO switches on the
`211` case-2 block in `nonlin_kprop` (`factor_k3.py:439-446`), and in the shipped
chain it breaks `_proj_coef_d4_r2`, which currently reduces the entire degree-4
object to one scalar under an isotropic-metric assumption.

**So the order is: measure the accuracy at the live shape first. If the gain has
decayed to nothing by `n = 1024`, no cost model is needed and the lane closes for
free.** Pricing first would be pricing a lever that may not exist.

TIER
----
**DESIGN.** Re-reads the P01 asset (`experiments/2026-08-26_p2_gt`), as D1, A3,
A5, D2 and the capped run did. `# seed-ledger: no block named`. It orients
priority and **promotes nothing**; that asset's own manifest says in writing that
nothing measured on it may close a lane, and this run does not.

Run:
    .venv-gpu\\Scripts\\python.exe local_sim/p2_metric_arm.py
    .venv-gpu\\Scripts\\python.exe local_sim/p2_metric_arm.py --report
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np  # noqa: E402

OUT_DIR = os.path.join(_ROOT, "experiments", "2026-09-09_slice4")
OUT = os.path.join(OUT_DIR, "metric_arm.json")
GT_DIR = "experiments/2026-08-26_p2_gt"

#: Pre-registered. The control must reproduce this or nothing else is read.
SIMPLE_BASELINE_XMSE = 4.004685e-08
#: The ladder's SIMPLE readings, for the trend this run is testing.
LADDER_SIMPLE_RATIOS = {"128x2": 0.595, "192x3": 0.686, "256x4": 0.763}
#: Pre-registered gate: the exact metric is worth pursuing only if it clears
#: 1.15x at the LIVE shape. Below that its cost (a ninth born block, and the
#: loss of the degree-4 scalar reduction) cannot plausibly be repaid.
GATE_RATIO = 1.15


def xmse(pred, A, B):
    """Unbiased two-half cross estimator. E[(p-A)(p-B)] = MSE(p) exactly."""
    return float(np.mean((pred - A) * (pred - B)))


def paired(a, b):
    from competition_score import _t975
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    m = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    h = _t975(n - 1) * se if n > 1 else float("nan")
    return {"paired_diff": m, "se": se, "ci95": [m - h, m + h], "n": n,
            "arithmetic_ratio": float(np.mean(a) / np.mean(b)),
            # NAME IT FOR WHAT IT COUNTS. `b` is the avg-metric arm and `a` the
            # exact one, so this is the number of nets on which the EXACT
            # metric WINS. The first spelling called it `n_exact_worse`
            # and printed it as "nets worse" -- the same sign confusion
            # that already misread `unfactored/factored` once this slice.
            "n_exact_better": int((np.asarray(b, float) > np.asarray(a, float)).sum())}


def _progress(frac):
    p = os.environ.get("WORKERQ_PROGRESS")
    if not p:
        return
    try:
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(str(min(max(frac, 0.0), 1.0)))
    except OSError:
        pass


def run(a) -> int:
    sys.path.insert(0, a.upstream)
    import torch
    from mlp_kprop.kprop_harmonic import SIMPLE
    import p2_kprop_bridge as BRIDGE
    from mint_provenance import canonical_mint

    v = BRIDGE.verify_upstream(a.upstream)
    if not v["pass"]:
        raise SystemExit("REFUSED: upstream is not the pinned object: %r" % v)

    gt = os.path.join(_ROOT, GT_DIR)
    man = json.load(open(os.path.join(gt, "gt_manifest.json"), encoding="utf-8"))
    if not man.get("all_pass"):
        raise SystemExit("REFUSED: the P2-GT asset did not pass its own fitness "
                         "checks. An arm measured against a void label is not a "
                         "measurement.")
    zA = np.load(os.path.join(gt, "gt_halfA.npz"))
    zB = np.load(os.path.join(gt, "gt_halfB.npz"))
    seeds = [int(s) for s in zA["seeds"]]
    assert seeds == [int(s) for s in zB["seeds"]], "halves disagree about seeds"
    A_all, B_all = zA["means"], zB["means"]
    if a.limit:
        seeds, A_all, B_all = seeds[:a.limit], A_all[:a.limit], B_all[:a.limit]
    width, depth = man["width"], man["depth"]
    dtype = getattr(torch, a.dtype)

    print("=" * 78)
    print("  THE EXACT-METRIC ARM AT THE LIVE SHAPE  --  %d x %d, %d nets"
          % (width, depth, len(seeds)))
    print("=" * 78)
    print("  asset            %s   label noise/half %.4e"
          % (GT_DIR, man["gt_noise_per_half_mean"]))
    print("  upstream         %s + recorded patch, VERIFIED" % v["rev"][:12])
    print("  dtype            %s   (D1 ran float32; the control is an f32 number)"
          % a.dtype)
    print("  ladder said      SIMPLE exact/avg %s -- TRENDING toward 1"
          % LADDER_SIMPLE_RATIOS)
    print("  PRE-REGISTERED   gate: exact metric must clear %.2fx at 1024x16\n"
          % GATE_RATIO)

    rows, t0 = [], time.perf_counter()
    for i, s in enumerate(seeds):
        w, _e, _ss = canonical_mint(width, depth, s)
        Af, Bf = A_all[i, -1, :], B_all[i, -1, :]
        row = {"seed": s}
        for tag, avg in (("avg_metric", True), ("exact_metric", False)):
            try:
                with torch.no_grad():
                    m = BRIDGE.kprop_means(w, k_max=3, factor=True, kind=SIMPLE,
                                           device=a.device, dtype=dtype,
                                           use_avg_metric=avg)
                row[tag] = xmse(m[-1], Af, Bf)
                row[tag + "_all_layers"] = xmse(m, A_all[i], B_all[i])
            except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
                row[tag] = None
                row[tag + "_error"] = "%s: %s" % (type(exc).__name__, exc)
                torch.cuda.empty_cache()
        rows.append(row)
        _progress((i + 1) / len(seeds))
        drift = (row["avg_metric"] - SIMPLE_BASELINE_XMSE
                 if row.get("avg_metric") is not None else float("nan"))
        print("    %3d/%d seed %d  (%.0fs)  avg %.4e  exact %.4e   "
              "[control drift %+.2e]"
              % (i + 1, len(seeds), s, time.perf_counter() - t0,
                 row.get("avg_metric", float("nan")),
                 row.get("exact_metric", float("nan")), drift))

    doc = {
        "slice": "Slice 4 -- the exact-metric arm at the live shape",
        "contract": ("experiments/LOG.md 2026-09-10 CONTRACT (Rule 1): the "
                     "exact-metric arm at 1024x16"),
        "tier": ("DESIGN. Re-reads the P01 asset. Orients priority; promotes "
                 "NOTHING. That asset's manifest forbids closing a lane on it."),
        "asset": GT_DIR, "width": width, "depth": depth,
        "dtype": a.dtype, "device": a.device,
        "gt_noise_per_half": man["gt_noise_per_half_mean"],
        "upstream": v,
        "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "workerq_job": os.environ.get("WORKERQ_JOB_ID"),
        "gate_ratio": GATE_RATIO,
        "control_baseline": SIMPLE_BASELINE_XMSE,
        "ladder_simple_ratios": LADDER_SIMPLE_RATIOS,
        "cost_note": ("NOT free: metric.ndim == 2 switches on the 211 case-2 "
                      "block (factor_k3.py:439-446), taking births 8n -> 9n, and "
                      "in the shipped fnp chain it breaks _proj_coef_d4_r2's "
                      "reduction of the whole degree-4 object to one scalar. No "
                      "cost model is offered here -- accuracy first, on purpose."),
        "rows": rows,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    print("\n  wrote %s" % os.path.relpath(a.out, _ROOT))
    _progress(1.0)
    return report(a)


def report(a) -> int:
    with open(a.out, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    rows = doc["rows"]
    ok = [(r["exact_metric"], r["avg_metric"]) for r in rows
          if r.get("exact_metric") is not None and r.get("avg_metric") is not None]
    print("\n" + "=" * 78)
    print("  RESULT")
    print("=" * 78)
    avg = np.array([y for _, y in ok])
    ex = np.array([x for x, _ in ok])
    print("  control (avg metric)  %.6e   vs pre-registered %.6e   drift %+.3e"
          % (avg.mean(), doc["control_baseline"], avg.mean() - doc["control_baseline"]))
    print("  exact metric          %.6e" % ex.mean())
    st = paired(ex, avg)
    gain = avg.mean() / ex.mean()
    print("\n  gain at 1024x16       %.4fx   (gate %.2fx -> %s)"
          % (gain, doc["gate_ratio"],
             "PASS" if gain >= doc["gate_ratio"] else "FAIL"))
    print("  paired (exact - avg)  %+.4e   95%% [%+.4e, %+.4e]   exact wins on %d/%d"
          % (st["paired_diff"], st["ci95"][0], st["ci95"][1],
             st["n_exact_better"], st["n"]))
    lr = doc["ladder_simple_ratios"]
    print("\n  the trend this run tested:")
    for k in ("128x2", "192x3", "256x4"):
        print("      ladder %-7s  gain %.3fx" % (k, 1.0 / lr[k]))
    print("      LIVE   1024x16  gain %.3fx" % gain)
    print("\n  %s" % doc["tier"])
    print("  %s" % doc["cost_note"])
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--upstream",
                    default=r"C:\Users\samsu\Documents\mlp_cumulant_propagation\src")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    return report(a) if a.report else run(a)


if __name__ == "__main__":
    raise SystemExit(main())
