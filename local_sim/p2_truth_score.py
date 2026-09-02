"""WP14 / C0-i: what the regenerating K3 flow actually SCORES against minted truth.

Contract: `experiments/LOG.md`, 2026-09-02, "CONTRACT (Rule 1): WP14", HEAD
`d915f83`, cell `C0-i`. Prediction **P2**: raw final-layer MSE in
`[7.5e-07, 1.0e-06]`, point `8.9e-07` -- the *orthogonal-error* point
`MSE_full + drift = 4.0e-08 + 8.48e-07`, not the bracket's midpoint.

Research only. Never imported by `estimator.py`.

WHY THIS CELL EXISTS
--------------------
WP13 closed with a fallback it explicitly refused to oversell: the regenerating
flow needs no correction at all to be a candidate, and at the multiplier floor its
raw MSE brackets `[5.20e-07, 1.26e-06]` -- adjusted `[5.2e-08, 1.26e-07]` against
graded v2.7's `1.374e-07`, so somewhere between `1.1x` and `2.6x` better.

**That is a triangle-inequality bracket, not a measurement.** It comes from a
drift against the uncapped flow plus an assumed `MSE_full`, and nothing narrower
is available without a real label:

    | sqrt(MSE_full) - sqrt(drift) |  <=  sqrt(MSE_cap)  <=  sqrt(MSE_full) + sqrt(drift)

`2.4x` of uncertainty is too wide to decide anything on, and the bracket's low end
is barely an improvement over the file we already shipped. WP13 named the fix and
declined to run it: one truth mint on the eight networks already claimed. That
mint is `experiments/2026-09-02_wp14_range/truth_P01_wp13nets.npz` (gpuq job 121,
sha256 `a3c27a7d...`), and this module spends it.

THE NOISE FLOOR IS QUOTED WITH EVERY NUMBER, AND HERE IT MATTERS
-----------------------------------------------------------------
`p2_truth.py` draws TWO independent replicas, so an ordinary MSE against either is
biased UPWARD by exactly the label's own Monte-Carlo variance. The cross-product

    mean_j (p_j - y1_j) (p_j - y2_j)

has expectation `MSE(p)` exactly, because `E[e1 e2] = 0`. Both are reported.

At `2^22` samples the measured truth noise is **`1.72e-08`** per replica. Against
a regenerating flow near `9e-07` that is `2%` and the cross-product removes even
that. Against **full K3 near `4.0e-08` it is `43%`**, so the `uncapped` arm's
absolute MSE here is a diagnostic and NOT a re-measurement of P-A's `4.0047e-08`,
which was taken on a different and better-resolved mint. Said here rather than
discovered later.

Run (GPU venv; heavy -- submit through gpuq, never launch):
    .venv-gpu\\Scripts\\python.exe local_sim/p2_truth_score.py \\
        --truth <abs path to truth npz> --out <abs path to json>
"""

from __future__ import annotations

import argparse
import json
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

import p2_kprop_bridge as bridge  # noqa: E402
import p2_range_lab as rlab  # noqa: E402

BUDGET = 2 ** 41

#: C0-iii's composed f32 projection. The multiplier every adjusted score below is
#: quoted at, and it is a PROJECTION -- see `p2_k3_fnp.py`.
F32_TOTAL_FLOPS = 68_802_510_336 + 99_116_660_736
F32_MULTIPLIER = max(0.1, F32_TOTAL_FLOPS / BUDGET)

#: The number the fallback has to beat, from the graded board, not from here.
V27_GRADED_ADJUSTED = 1.3738e-07
V27_GRADED_RAW = 1.3738e-06

#: P-A's full-K3 raw on 24 DESIGN mints, for orientation only.
MSE_FULL_ANCHOR = 4.0047e-08


def _mse_pair(p, y1, y2):
    """Ordinary MSE against each replica, and the UNBIASED cross-product."""
    p = np.asarray(p, dtype=np.float64)
    y1 = np.asarray(y1, dtype=np.float64)
    y2 = np.asarray(y2, dtype=np.float64)
    return {
        "mse_r1": float(((p - y1) ** 2).mean()),
        "mse_r2": float(((p - y2) ** 2).mean()),
        "mse_cross": float(((p - y1) * (p - y2)).mean()),
        "mse_avg_replica": float(((p - 0.5 * (y1 + y2)) ** 2).mean()),
    }


def main() -> int:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["f32", "f64"], default="f32")
    ap.add_argument("--arms", nargs="*",
                    default=["regen", "uncapped", "k2p1"])
    a = ap.parse_args()

    src = bridge._src()
    v = bridge.verify_upstream(src)
    if not v["pass"]:
        raise SystemExit("REFUSED: upstream is not rev 93d091a4 plus the patch.")

    d = np.load(a.truth)
    seeds = [int(x) for x in d["seeds"]]
    m1, m2 = d["means_r1"], d["means_r2"]
    gt_noise = d["gt_noise"]
    n_samples = int(d["n_samples"])
    if "probe" in d.files and int(d["probe"]) == 1:
        raise SystemExit("REFUSED: this is a PROBE truth file, not evidence.")

    dtype = torch.float64 if a.dtype == "f64" else torch.float32
    print("=" * 78)
    print("WP14 / C0-i  --  the regenerating flow scored against MINTED TRUTH")
    print("=" * 78)
    print(f"  truth      {os.path.relpath(a.truth, _ROOT)}")
    print(f"  networks   {len(seeds)}   seeds {seeds[0]:,} .. {seeds[-1]:,}")
    print(f"  samples    2^{int(round(np.log2(n_samples)))} per replica, 2 replicas")
    print(f"  GT noise   {float(gt_noise.mean()):.4e} per replica  "
          f"(the cross-product estimate removes its bias exactly)")
    print(f"  dtype      {a.dtype}")
    print()

    from mint_provenance import canonical_mint
    import p2_cal_lab as cal

    rows = []
    for i, seed in enumerate(seeds):
        w, _e, _s = canonical_mint(1024, 16, seed)
        t0 = time.perf_counter()
        preds = {}
        if "uncapped" in a.arms:
            preds["uncapped"] = rlab.means_of(w, dtype=dtype, device=a.device)
        if "regen" in a.arms:
            preds["regen"] = rlab.means_of(w, hook_nl=rlab._regen_hook_nl(),
                                           dtype=dtype, device=a.device)
        if "k2p1" in a.arms:
            preds["k2p1"] = cal.closure(w)
        for arm, p in preds.items():
            r = _mse_pair(p[-1], m1[i, -1], m2[i, -1])
            r.update({"seed": seed, "arm": arm,
                      "gt_noise": float(gt_noise[i])})
            rows.append(r)
            print(f"    seed {seed}  {arm:<9} cross {r['mse_cross']:.6e}   "
                  f"r1 {r['mse_r1']:.6e}")
        if "regen" in preds and "uncapped" in preds:
            drift = float(((preds["regen"][-1] - preds["uncapped"][-1]) ** 2).mean())
            rows.append({"seed": seed, "arm": "drift_regen_vs_uncapped",
                         "mse_cross": drift, "mse_r1": drift, "mse_r2": drift,
                         "mse_avg_replica": drift, "gt_noise": float(gt_noise[i])})
            print(f"    seed {seed}  {'drift':<9} {drift:.6e}  "
                  f"(regen vs uncapped, the WP13 quantity)")
        print(f"    seed {seed}  [{time.perf_counter() - t0:.1f}s]")

    print()
    print("  SUMMARY -- mean over networks, cross-product (unbiased) estimator")
    print(f"  {'arm':<26}{'raw MSE':>14}{'adjusted @ 0.1':>17}{'vs v2.7':>10}")
    summary = {}
    for arm in sorted({r["arm"] for r in rows}):
        vals = [r["mse_cross"] for r in rows if r["arm"] == arm]
        mean = float(np.mean(vals))
        adj = mean * F32_MULTIPLIER
        summary[arm] = {"n": len(vals), "mean_cross": mean,
                        "adjusted_at_f32_multiplier": adj,
                        "x_better_than_v27_graded": V27_GRADED_ADJUSTED / adj,
                        "per_net": vals}
        print(f"  {arm:<26}{mean:>14.4e}{adj:>17.4e}"
              f"{V27_GRADED_ADJUSTED / adj:>9.2f}x")
    print()
    print(f"  multiplier used  {F32_MULTIPLIER:.5f}  "
          f"(C0-iii's composed f32 PROJECTION, {F32_TOTAL_FLOPS:,} FLOPs)")
    print(f"  graded v2.7      raw {V27_GRADED_RAW:.4e}   adjusted "
          f"{V27_GRADED_ADJUSTED:.4e}   -- the number to beat, from the board")
    print()
    print("  P2 (pre-registered): raw in [7.5e-07, 1.0e-06], point 8.9e-07")
    if "regen" in summary:
        got = summary["regen"]["mean_cross"]
        held = 7.5e-07 <= got <= 1.0e-06
        print(f"  measured {got:.4e}  ->  P2 {'HELD' if held else 'REFUTED'}")
    print("=" * 78)

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump({
                "check": "WP14 C0-i: regenerating flow vs minted truth",
                "contract": "LOG 2026-09-02 CONTRACT WP14, HEAD d915f83",
                "truth": os.path.relpath(a.truth, _ROOT).replace("\\", "/"),
                "seed_block": "P01", "seeds": seeds, "dtype": a.dtype,
                "n_samples_per_replica": n_samples,
                "gt_noise_mean": float(gt_noise.mean()),
                "multiplier": F32_MULTIPLIER,
                "multiplier_source": "C0-iii composed f32 projection -- NOT an "
                                     "end-to-end measurement of any candidate",
                "v27_graded_adjusted": V27_GRADED_ADJUSTED,
                "mse_full_anchor_for_orientation": MSE_FULL_ANCHOR,
                "caveat_uncapped": "at 2^22 samples the GT noise is ~43% of full "
                                   "K3's own MSE, so the 'uncapped' row is a "
                                   "diagnostic and NOT a re-measurement of P-A",
                "rows": rows, "summary": summary,
            }, fh, indent=1)
        print(f"  wrote {os.path.relpath(a.out, _ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
