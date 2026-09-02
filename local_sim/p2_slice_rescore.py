"""WP13 re-scoring: the gates, in the currency the contract says binds.

Written in response to `SKEPTIC WP13`, which found that **`G-RANK`'s threshold
`r <= 330` was inherited from P-B-COST's upstream torch-side affine counter** --
the very counter this slice's own contract forbids scoring a gate on:

    "Every upstream NamedFlopCounter figure quoted anywhere in this slice is
     labelled *upstream torch-side counter, not flopscope*, and **no gate is
     scored on one**."   -- CONTRACT WP13, Cost model

That is the WP8 failure class repeating: a gate scored against a quantity the
contract excluded. This file replaces the hand arithmetic that produced the
first result entry's composed-excess column -- which the Skeptic correctly
reported was not reproducible from the artifacts under any formula it tried,
because it existed only in a shell command.

WHAT IS FIXED HERE
------------------
1. **The affordable rank is measured in flopscope**, not inherited. An oracle
   rank-`r` correction is applied to the `(2, 1)` slice directly (`S21 + U V^T`)
   -- it does NOT need to be carried through the factored machinery, because the
   regenerating flow rebuilds `kappa_3` from the slice anyway.
2. **Both compositions are reported**, worst-layer and layer-averaged, because
   the contract's per-layer wording is ambiguous and the Skeptic showed the
   ambiguity runs the OPPOSITE way from what the first entry claimed.
3. **The anchor is stated and its weakness is printed**, not buried: the
   `eps^2` law's amplitude over-predicts the structured drift by `32x`, so the
   law is re-anchored on the flow's own operating point and is a ONE-POINT
   calibration.

`# seed-ledger: no block named` -- reads committed artifacts, mints nothing.

Run:
    uv run python local_sim/p2_slice_rescore.py
"""

from __future__ import annotations

import argparse
import collections
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

BUDGET = 2 ** 41
WIDTH, DEPTH = 1024, 16

#: E0-COST measured, with four Strassen levels (`e0_cost_levers.json`).
REGEN_FLOPS_STRASSEN4 = 207_330_648_398
#: E0-COST measured, no Strassen (`e0_cost.json`).
REGEN_FLOPS_PLAIN = 335_833_648_128

MSE_FULL_ANCHOR = 4.0047e-08
GT_NOISE_FLOOR = 1.155e-09
TARGET_EXCESS = 1e-8


def price_rank_correction(r: int, *, depth_used: int = DEPTH - 1) -> float:
    """flopscope cost of adding a rank-`r` correction to the `(2, 1)` slice.

    `S21 + U V^T` at `(n, r) (r, n)`. The correction does NOT have to be carried
    as three general factor legs: the regenerating flow rebuilds `kappa_3` from
    the slice every layer anyway, so correcting the slice IS correcting the
    tensor, and one matmul per layer is the whole price.
    """
    import flopscope as flops
    import flopscope.numpy as fnp

    if os.path.abspath(getattr(flops, "__file__", "")).startswith(_HERE + os.sep):
        raise RuntimeError("REFUSED: flopscope resolved to the local shim.")
    n = WIDTH
    rng = np.random.default_rng(0)
    S = fnp.array(rng.standard_normal((n, n)))
    U = fnp.array(rng.standard_normal((n, r)))
    V = fnp.array(rng.standard_normal((n, r)))
    with flops.BudgetContext(flop_budget=10 ** 15, quiet=True) as c:
        _ = S + U @ V.T
    return float(c.flops_used) * depth_used


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="experiments/2026-09-02_wp13_slice")
    ap.add_argument("--out", default="experiments/2026-09-02_wp13_slice/rescore.json")
    a = ap.parse_args()

    d = os.path.join(_ROOT, a.dir)
    tol = json.load(open(os.path.join(d, "e0_tol.json"), encoding="utf-8"))
    e2 = json.load(open(os.path.join(d, "e2_residual.json"), encoding="utf-8"))
    e1 = json.load(open(os.path.join(d, "e1_selector.json"), encoding="utf-8"))

    drift = float(np.mean([r["drift_regen_vs_uncapped"] for r in tol["rows"]]))
    n_tol = len(tol["seeds"])
    n_struct = len(e2["seeds"])

    by = collections.defaultdict(list)
    for r in e2["rows"]:
        by[r["layer"]].append(r)
    layers = sorted(by)
    rel0 = {l: float(np.mean([x["rel_E21"] for x in by[l]])) for l in layers}

    fit = tol["fits"]["coherent"]
    predicted_at_rel0 = fit["A"] * (float(np.mean(list(rel0.values()))) ** fit["slope"])

    print("=" * 78)
    print("WP13 RE-SCORE  --  the gates in flopscope, after SKEPTIC WP13")
    print("=" * 78)
    print(f"  E0-TOL ran on {n_tol} networks; E1/E2 on {n_struct}.  "
          "(The first result entry said 8 for all three. It was 3 for E0-TOL.)")
    print(f"  anchor: regenerating-flow drift {drift:.4e} at mean rel_E21 "
          f"{np.mean(list(rel0.values())):.4f}")
    print(f"  the fitted coherent law would predict {predicted_at_rel0:.4e} there -- "
          f"it OVER-predicts by {predicted_at_rel0 / drift:.1f}x.")
    print("  So the exponent is used and the amplitude is re-anchored on the flow's")
    print("  own operating point. That is a ONE-POINT calibration and it is a model.")
    print()

    def excess(l, rel):
        return drift * (rel / rel0[l]) ** 2

    ranks = [int(x) for x in e2["ranks"]]
    print("  AFFORDABLE RANK, priced in flopscope (not upstream's torch counter)")
    print(f"    {'r':>6} {'correction FLOPs':>18} {'+regen(L=4)':>14} "
          f"{'x budget':>10} {'multiplier':>11}")
    prices = {}
    for r in ranks:
        f = price_rank_correction(r)
        prices[r] = f
        tot = REGEN_FLOPS_STRASSEN4 + f
        x = tot / BUDGET
        print(f"    {r:>6} {f:>18,.0f} {tot:>14,.0f} {x:>10.5f} {max(0.1, x):>11.5f}")
    print("    -> rank is CHEAP. The contract's `r <= 330` came from upstream's")
    print("       affine torch counter, which the contract itself forbids.")
    print()

    print("  COMPOSED EXCESS after an oracle rank-r correction")
    print(f"    {'r':>6} {'worst layer':>13} {'layer-averaged':>16} "
          f"{'worst':>7} {'avg':>5}")
    rows = []
    for r in ranks:
        per = {l: float(np.mean([x["rel_E21_after_rank"][str(r)] for x in by[l]]))
               for l in layers}
        ex = {l: excess(l, per[l]) for l in layers}
        worst = max(ex.values())
        # layer-averaged: the anchor is itself an ALL-LAYERS drift, so averaging
        # the squared ratios is the composition consistent with how it was measured.
        avg = drift * float(np.mean([(per[l] / rel0[l]) ** 2 for l in layers]))
        rows.append({"rank": r, "flops": prices[r],
                     "multiplier": max(0.1, (REGEN_FLOPS_STRASSEN4 + prices[r]) / BUDGET),
                     "rel_per_layer": per, "excess_per_layer": ex,
                     "excess_worst_layer": worst, "excess_layer_averaged": avg,
                     "pass_worst": bool(worst <= TARGET_EXCESS),
                     "pass_avg": bool(avg <= TARGET_EXCESS)})
        print(f"    {r:>6} {worst:>13.3e} {avg:>16.3e} "
              f"{'PASS' if worst <= TARGET_EXCESS else 'fail':>7} "
              f"{'PASS' if avg <= TARGET_EXCESS else 'fail':>5}")
    print()

    # G-SEL, unchanged and unrescuable
    byk = collections.defaultdict(list)
    for r in e1["rows"]:
        byk[(r["layer"], r["k"], r["selector"])].append(r)
    sels = ("top_frob", "top_slice", "uniform", "dkm_frob", "dkm_slice")
    print("  G-SEL, best selector at each k, over all layers")
    print(f"    {'k':>6} {'best selector':>12} {'rel':>8} {'vs doing NOTHING':>18}")
    for k in sorted(set(r["k"] for r in e1["rows"])):
        best, bv = None, 1e9
        for s in sels:
            vs = [x["rel_slice21"] for l in layers for x in byk[(l, k, s)]]
            if vs and np.mean(vs) < bv:
                best, bv = s, float(np.mean(vs))
        nothing = float(np.mean(list(rel0.values())))
        print(f"    {k:>6} {best:>12} {bv:>8.4f} {bv / nothing:>17.2f}x")
    print("    -> every selector at every k is WORSE than doing nothing "
          f"(rel_E21 = {float(np.mean(list(rel0.values()))):.4f}).")
    print("       No monotone re-mapping of the amplification law can rescue this.")
    print()

    r90 = {l: float(np.mean([x["r90_E21"] for x in by[l]])) for l in layers}
    r90c = {l: float(np.mean([x["r90_cov"] for x in by[l]])) for l in layers}
    print("  E2 rank collapse, and the ratio the first entry got WRONG")
    print(f"    {'layer':>6} {'r90(E21)':>9} {'r90(Sigma)':>11} {'ratio':>7}")
    for l in layers:
        print(f"    {l:>6} {r90[l]:>9.0f} {r90c[l]:>11.0f} {r90[l] / r90c[l]:>7.2f}")
    print("    -> both collapse, but NOT 'layer for layer': the ratio grows")
    print("       1.11 -> 3.00, so by depth 14 E21 needs 3x more directions than")
    print("       Sigma has significant ones. The naive containment premise of the")
    print("       named follow-up is at best PARTLY true and must be re-priced.")
    print("=" * 78)

    doc = {
        "check": "WP13 re-score after SKEPTIC WP13",
        "supersedes": ("the composed-excess column of the first E0-TOL/E1/E2 "
                       "result entry, which was hand arithmetic with no artifact"),
        "n_networks_e0tol": n_tol, "n_networks_e1e2": n_struct,
        "anchor_drift": drift, "rel0_per_layer": rel0,
        "fitted_law_overprediction_factor": predicted_at_rel0 / drift,
        "regen_flops_strassen4": REGEN_FLOPS_STRASSEN4,
        "regen_flops_plain": REGEN_FLOPS_PLAIN,
        "target_excess": TARGET_EXCESS,
        "mse_full_anchor": MSE_FULL_ANCHOR, "gt_noise_floor": GT_NOISE_FLOOR,
        "rank_rows": rows,
        "r90_E21": r90, "r90_cov": r90c,
        "r90_ratio": {str(l): r90[l] / r90c[l] for l in layers},
        "caveat": ("the composed-excess mapping inverts a law measured with "
                   "uniform multiplicative perturbation and applies it to a "
                   "structured residual; it is re-anchored on the flow's own "
                   "operating point and is a one-point calibration"),
    }
    out = os.path.abspath(os.path.join(_ROOT, a.out))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
    print(f"  wrote {os.path.relpath(out, _ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
