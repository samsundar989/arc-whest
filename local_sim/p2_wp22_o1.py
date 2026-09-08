"""WP22 / O1: is the K3 chain's residual error a SCALE error the readout can fix?

Handoff section 3a asks for "fitted renormalisation of the K3 chain: per-layer
scalars on the mean, and separately on the (3,) and (2,1) kappa_3 slices;
sequential closed-form least squares", predicting `1.2-1.5x`.

**This module answers the cheap half of that question first, and it answers a
question that decides whether the expensive half is worth building.** Scalars
applied to the RETURNED array are a readout fix: they cost one multiply per layer
and cannot propagate, because nothing downstream of the readout exists. Scalars
applied INSIDE the chain propagate, and are what the `1.2-1.5x` prediction is
really about -- but they need the chain's internal state exposed, which is a real
piece of work.

So: if a fitted scale on the readout removes a material fraction of the error, the
chain has a systematic multiplicative bias, and the internal version is very
likely worth building. If it removes nearly nothing, the error is NOT a scale
error, the internal version inherits that, and section 3a should be re-scoped
toward the shape-correcting arms (3b Edgeworth-4, 3d SSC) instead.

Only the FINAL layer is scored (`score = final_layer_MSE * multiplier`), so the
final row is the decision row; the other rows are reported because a chain whose
intermediate rows are biased and whose last row is not would be a surprise worth
knowing about.

FOUR NESTED ARMS, each strictly richer than the last, all closed-form:

    A0  none                            the control
    A1  one global scalar               1 parameter
    A2  a scalar per layer              16
    A3  a scalar and an offset per layer 32

FIT and VALIDATION are disjoint by construction:

    FIT         P02, 96 nets, `truth_P02.npz` -- the DESIGN block allocated for
                exactly this ("the offline learned cumulant-feature corrector"),
                where tuning is PERMITTED
    VALIDATE    the 24 high-resolution P01 nets of `gt_halfA/B.npz`, 2**26 samples
                per half, which no coefficient here is fitted on

A scalar fitted on 96 nets x 1024 neurons is over-determined by ~10^5 to 1, so
P02's coarser `2**22` truth is not a limitation for FITTING; the validation set is
the high-resolution one because that is where the number is quoted.

# seed-ledger: P02 is claimed by `truth_P02.npz`; this module draws no new seeds.
Research only. Never imported by `estimator.py`.

Run:  uv run python local_sim/p2_wp22_o1.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]

import numpy as np  # noqa: E402

FIT_TRUTH = os.path.join(_ROOT, "experiments", "2026-09-01_wp9_corrector",
                         "truth_P02.npz")
VAL_A = os.path.join(_ROOT, "experiments", "2026-08-26_p2_gt", "gt_halfA.npz")
VAL_B = os.path.join(_ROOT, "experiments", "2026-08-26_p2_gt", "gt_halfB.npz")


def _truth_means(path, n):
    d = np.load(path)
    for key in ("means", "means_r1"):
        if key in d.files:
            m = d[key][:n].astype(np.float64)
            if "means_r2" in d.files:
                m = 0.5 * (m + d["means_r2"][:n].astype(np.float64))
            return d["seeds"][:n], m
    raise KeyError(f"no means in {path}")


def fit_scales(pred, truth, per_layer, with_offset):
    """Closed-form least squares. `pred`/`truth` are `(nets, depth, width)`.

    Returns `(a, b)` broadcastable over `(1, depth, 1)`; `b` is zero when no
    offset is fitted. Least squares on `a*pred + b ~ truth`, pooled over nets and
    neurons, per layer or globally.
    """
    depth = pred.shape[1]
    a = np.ones((1, depth, 1))
    b = np.zeros((1, depth, 1))
    groups = range(depth) if per_layer else [None]
    for g in groups:
        sl = slice(None) if g is None else slice(g, g + 1)
        x = pred[:, sl, :].ravel()
        y = truth[:, sl, :].ravel()
        if with_offset:
            n = x.size
            sx, sy = x.sum(), y.sum()
            sxx, sxy = (x * x).sum(), (x * y).sum()
            den = n * sxx - sx * sx
            aa = (n * sxy - sx * sy) / den if den else 1.0
            bb = (sy - aa * sx) / n
        else:
            sxx = (x * x).sum()
            aa = (x * y).sum() / sxx if sxx else 1.0
            bb = 0.0
        if g is None:
            a[:] = aa
            b[:] = bb
        else:
            a[0, g, 0] = aa
            b[0, g, 0] = bb
    return a, b


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--candidate", default="estimator_k3_full.py")
    ap.add_argument("--n-fit", type=int, default=96)
    ap.add_argument("--n-val", type=int, default=24)
    ap.add_argument("--preds-cache", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    from p2_wp21_score import run_arm

    fit_seeds, fit_truth = _truth_means(FIT_TRUTH, a.n_fit)
    va, vb = np.load(VAL_A), np.load(VAL_B)
    val_seeds = va["seeds"][:a.n_val]
    val_truth = 0.5 * (va["means"][:a.n_val].astype(np.float64)
                       + vb["means"][:a.n_val].astype(np.float64))
    d = (va["means"][:a.n_val].astype(np.float64)
         - vb["means"][:a.n_val].astype(np.float64))
    gt_noise = float((d[:, -1, :] ** 2).mean() / 4.0)

    print("=" * 92)
    print("WP22 / O1 -- is the chain's error a SCALE error a readout can fix?")
    print("=" * 92)
    print(f"  candidate   {a.candidate}")
    print(f"  FIT         P02, {len(fit_seeds)} nets  (DESIGN block; tuning permitted)")
    print(f"  VALIDATE    {len(val_seeds)} nets, gt_halfA/B, 2**26 per half"
          f"   -- DISJOINT, nothing here is fitted on them")
    print(f"  GT noise floor on the validation final layer   {gt_noise:.4e}")

    cache = a.preds_cache or os.path.join(
        _ROOT, "experiments", "2026-09-05_slice1", "wp22_preds.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        pf, pv = z["fit"], z["val"]
        print(f"\n  predictions loaded from {os.path.relpath(cache, _ROOT)}")
    else:
        print(f"\n  running the chain on {len(fit_seeds)} + {len(val_seeds)} nets...")
        pf, _f, _o = run_arm(a.candidate, 0, fit_seeds)
        pv, _f, _o = run_arm(a.candidate, 0, val_seeds)
        np.savez_compressed(cache, fit=pf, val=pv,
                            fit_seeds=fit_seeds, val_seeds=val_seeds)
        print(f"  wrote {os.path.relpath(cache, _ROOT)}")

    def mse_final(p, t):
        return float(((p[:, -1, :] - t[:, -1, :]) ** 2).mean())

    def mse_all(p, t):
        return float(((p - t) ** 2).mean())

    base_f = mse_final(pv, val_truth)
    base_a = mse_all(pv, val_truth)

    arms = [("A0  none", False, False, False),
            ("A1  one global scalar", True, False, False),
            ("A2  scalar per layer", True, True, False),
            ("A3  scalar + offset per layer", True, True, True)]

    print(f"\n  {'arm':>32}{'params':>8}{'final-layer MSE':>18}{'x base':>9}"
          f"{'all-layers MSE':>17}{'x base':>9}")
    rows = []
    for (name, active, per_layer, offset) in arms:
        if not active:
            fm, am, npar = base_f, base_a, 0
        else:
            A, B = fit_scales(pf, fit_truth, per_layer, offset)
            q = pv * A + B
            fm, am = mse_final(q, val_truth), mse_all(q, val_truth)
            npar = (16 if per_layer else 1) * (2 if offset else 1)
        print(f"  {name:>32}{npar:>8}{fm:>18.4e}{fm / base_f:>9.4f}"
              f"{am:>17.4e}{am / base_a:>9.4f}")
        rows.append({"arm": name.strip(), "params": npar,
                     "final_mse": fm, "final_ratio": fm / base_f,
                     "all_mse": am, "all_ratio": am / base_a})

    best = min(rows[1:], key=lambda r: r["final_ratio"])
    gain = 1.0 / best["final_ratio"]
    print(f"\n  Best readout arm: {best['arm']}  ->  {gain:.4f}x on final-layer MSE")
    print(f"  Handoff section 3a predicts 1.2-1.5x for the INTERNAL version.")
    print("\n  WHAT THIS DOES AND DOES NOT SETTLE")
    print("  * These scalars are on the RETURNED array. They cost ~1 multiply per")
    print("    layer and CANNOT propagate -- nothing runs after the readout.")
    print("  * A large gain here means the chain carries a systematic")
    print("    MULTIPLICATIVE bias, and the internal per-layer version (which")
    print("    does propagate, and which section 3a actually asks for) should be")
    print("    strictly better. A gain near 1.00x means the error is NOT a scale")
    print("    error, the internal version inherits that, and section 3a should be")
    print("    re-scoped toward the shape-correcting arms (3b, 3d).")
    print("  * Fitted on P02, quoted on 24 nets neither arm was fitted on, against")
    print(f"    a GT noise floor of {gt_noise:.4e}.")

    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump({"candidate": a.candidate,
                       "n_fit": int(len(fit_seeds)), "n_val": int(len(val_seeds)),
                       "gt_noise_final": gt_noise,
                       "base_final_mse": base_f, "base_all_mse": base_a,
                       "arms": rows}, fh, indent=2)
        print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
