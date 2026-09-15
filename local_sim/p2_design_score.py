"""Score one candidate against one control on DESIGN truth. The cadence tool.

Slice 6 contract §0.4: *"every candidate gets a 24-net fresh-truth score the day
it compiles."* This is that. It is deliberately small and general — two file
paths, one truth file, a paired arithmetic score difference — because the thing
that has been missing since September 7 is not an instrument, it is a habit.

WHY NOT `p2_lab.py`
--------------------
`p2_lab.py` was locked on 2026-09-14 to a specific four-arm preregistered
mechanism experiment: it demands a freeze receipt naming exactly
`("exact", "coherent", "consumers-only", "births-only")` and a `--control` of
`"exact"`. Those locks are correct for that experiment and this is a different
one, so this module exists beside it rather than loosening it. **A lock you
route around is not a lock.**

WHAT IT IS AND IS NOT
----------------------
DESIGN tier. It reads a reusable design truth file through the read register,
so the look is counted. It produces the paired arithmetic score difference with
a studentised-t interval — the binding statistic — and, when the truth file
carries `gt_noise`, the noise-corrected true-function difference and the
official-precision forecast beside it, because label noise does **not** cancel
from an adjusted difference when the two arms have different multipliers.

It **decides nothing**. `--gate` is accepted only so a preregistered contract
can have its verdict printed; with no gate it prints the statistics and stops.
A design-tier pass licenses a DECIDE-block contract, never a promotion.

Run:
    uv run python local_sim/p2_design_score.py \\
        --control estimator_k3_strassen_L4_f21.py \\
        --candidate estimator_k3_v30.py \\
        --truth experiments/2026-08-21_p2_cp1/truth_P01.npz \\
        --n 24 --ts <utc> --out experiments/<slice>/score.json
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--control", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--ts", required=True,
                    help="UTC timestamp for the read register. REQUIRED: this "
                         "module never reads the clock, so a replay carries "
                         "the historical date.")
    ap.add_argument("--block", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--gate", type=float, default=None,
                    help="a PRE-REGISTERED gate. Omit unless a contract set "
                         "one; this tool does not invent thresholds.")
    ap.add_argument("--residual-cap", type=float, default=0.0,
                    help="<=0 disables the local residual cliff. Default: "
                         "disabled, because section 0.1 removed local seconds "
                         "from every decision path and an armed cliff here "
                         "replaces an accuracy comparison with a comparison of "
                         "zero-prediction MSEs.")
    ap.add_argument("--official-samples", type=float, default=1e9)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    import flopscope as flops
    flops.configure(symmetry_warnings=False)
    import competition_score as cs
    import lifecycle_eval as le
    import truth_reads as tr

    block = a.block or tr.resolve_block(a.truth)
    with tr.reserve(block, a.truth, consumer="local_sim/p2_design_score.py",
                    purpose="design-tier paired score: %s vs %s"
                            % (os.path.basename(a.candidate),
                               os.path.basename(a.control)),
                    ts=a.ts, instrument=os.path.abspath(__file__),
                    result_artifact=a.out, note=a.note) as truth_path:
        z = dict(np.load(truth_path))

    seeds = [int(s) for s in z["seeds"]][:a.n]
    n = len(seeds)
    width = int(z["width"])
    depth = int(z["depth"])
    n_samples = float(z["n_samples"])
    y = ((z["means_r1"][:n, -1, :].astype(np.float64)
          + z["means_r2"][:n, -1, :].astype(np.float64)) / 2.0)
    gt = z["gt_noise"][:n].astype(np.float64) if "gt_noise" in z else None
    # The AVERAGED truth has half the per-replica noise.
    gt_avg = None if gt is None else gt / 2.0

    res = {} if a.residual_cap > 0 else {"residual_wall_time_limit_s": None}
    if a.residual_cap > 0:
        res = {"residual_wall_time_limit_s": a.residual_cap}
    spec = le.LifecycleSpec(width=width, depth=depth,
                            flop_budget=2 ** 41, lam=0.0, **res)

    print("=" * 96)
    print("DESIGN-TIER PAIRED SCORE -- licenses a contract, never a promotion")
    print("=" * 96)
    print(f"  control    {a.control}")
    print(f"  candidate  {a.candidate}")
    print(f"  truth      {os.path.relpath(a.truth, _ROOT)}   block {block}   "
          f"{n} nets   {width}x{depth}   2 replicas x {n_samples:,.0f}")
    if a.residual_cap <= 0:
        print("  *** RESIDUAL CAP DISABLED -- ACCURACY-ONLY. This run's failure")
        print("  *** accounting cannot see a 0.4 s violation and it is NOT")
        print("  *** residual evidence for anything (contract section 0.1).")
    print()

    arms = {}
    for label, path in (("control", a.control), ("candidate", a.candidate)):
        t0 = time.perf_counter()
        rows, prov = le.evaluate(path, seeds, y, spec=spec,
                                 gt_noise=None if gt_avg is None
                                 else list(gt_avg), progress_every=0)
        suite = le.score(rows, spec=spec)
        arms[label] = {"rows": rows, "suite": suite, "prov": prov,
                       "secs": time.perf_counter() - t0}
        f = float(np.mean([r.flops_used for r in rows]))
        print(f"  {label:<10} score {suite.objective:.6e}   "
              f"raw MSE {suite.mean_mse:.6e}   C/B {f / 2 ** 41:.6f}   "
              f"mult {suite.mean_multiplier:.6f}   failed {suite.n_failed}/{n}"
              f"   {arms[label]['secs']:.0f}s")

    c_rows = [r.to_row() for r in arms["control"]["rows"]]
    k_rows = [r.to_row() for r in arms["candidate"]["rows"]]

    print()
    if gt_avg is not None:
        nac = cs.noise_corrected_compare(
            c_rows, k_rows, local_samples=n_samples * 2.0,
            official_samples=a.official_samples, require_ids=True)
        print(nac)
        paired = nac.observed
    else:
        paired = cs.paired_compare_rows(c_rows, k_rows, require_ids=True)
        print("  no gt_noise in the truth file: observed difference only")

    print()
    print(f"  PAIRED arithmetic score difference (the binding statistic)")
    print(f"    mean difference   {paired.mean_difference:+.6e}")
    print(f"    studentised-t 95% [{paired.t_ci_difference[0]:+.4e}, "
          f"{paired.t_ci_difference[1]:+.4e}]")
    print(f"    arithmetic gain   {paired.arithmetic_gain:.4f}x   "
          f"candidate better on {paired.n_candidate_better}/{paired.n}   "
          f"sign p {paired.sign_test_p:.4f}")
    print(f"    geometric gain    {paired.geometric_gain:.4f}x   "
          "DIAGNOSTIC ONLY, no authority")

    verdict = None
    if a.gate is not None:
        v = cs.decide(arms["control"]["suite"].scores,
                      arms["candidate"]["suite"].scores, gate=a.gate)
        print()
        print(v)
        verdict = {"passed": bool(v.passed), "gain": float(v.gain),
                   "gate": float(v.gate),
                   "ci_excludes_zero": bool(v.ci_excludes_zero),
                   "reasons": list(v.reasons)}
    else:
        print()
        print("  no --gate given, so no verdict. This tool does not invent "
              "thresholds.")

    if a.out:
        d = os.path.dirname(os.path.abspath(a.out))
        if d:
            os.makedirs(d, exist_ok=True)
        payload = {
            "control": a.control, "candidate": a.candidate,
            "truth": os.path.relpath(a.truth, _ROOT), "block": block,
            "n": n, "width": width, "depth": depth,
            "n_samples_per_replica": n_samples,
            "accuracy_only": a.residual_cap <= 0,
            "tier": "DESIGN", "promotes_nothing": True,
            "arms": {k: {"objective": float(v["suite"].objective),
                         "raw_mse": float(v["suite"].mean_mse),
                         "multiplier": float(v["suite"].mean_multiplier),
                         "n_failed": int(v["suite"].n_failed),
                         "flops_mean": float(np.mean(
                             [r.flops_used for r in v["rows"]])),
                         "rows_hash": le.rows_hash(v["rows"]),
                         "candidate_sha256": v["prov"].get("candidate_sha256",
                                                           ""),
                         "seconds": v["secs"]}
                     for k, v in arms.items()},
            "paired": {"mean_difference": float(paired.mean_difference),
                       "t_ci": [float(x) for x in paired.t_ci_difference],
                       "arithmetic_gain": float(paired.arithmetic_gain),
                       "geometric_gain_DIAGNOSTIC": float(paired.geometric_gain),
                       "n_candidate_better": int(paired.n_candidate_better),
                       "sign_test_p": float(paired.sign_test_p),
                       "notes": list(paired.notes)},
            "verdict": verdict,
        }
        if gt_avg is not None:
            payload["noise_aware"] = {
                "true_function_difference": float(
                    nac.true_function.mean_difference),
                "forecast_difference": float(nac.forecast.mean_difference),
                "noise_bias": float(nac.noise_bias),
                "noise_bias_share": float(nac.noise_bias_share),
                "n_negative_corrected": int(nac.n_negative_corrected),
            }
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, indent=2, default=str) + "\n")
        print(f"\n  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
