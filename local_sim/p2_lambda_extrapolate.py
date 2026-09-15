"""Does our own λ transfer from a width that fits to the width that scores?

Slice 6a step 1, second half. `local_sim/p2_lambda_fit.py` measured λ at the
widths a 32 GiB card can hold; the live shape is 1024 and the dense AUGMENT
state is O(n³), so 1024 is 6,369 GiB away. This module asks whether the
measured widths determine λ(1024), and answers with a model and a residual
rather than a hope.

THE MODEL, AND WHY THIS ONE
----------------------------
Per layer, fit `log λ_l = log a_l − p_l · log n` by least squares over the
measured widths. A power law is the right family to try first because the chain
is self-similar in width: every layer is the same operation at a different `n`,
and He initialisation keeps the pre-activation variance scale fixed as `n`
grows, so a dimensionless coefficient that moves with `n` should move as a
power of it. The fit's own R² is reported per layer, so the family is judged
rather than assumed.

WIDTH 48 IS EXCLUDED, AND THE REASON IS MEASURED
-------------------------------------------------
The n=48 chain goes numerically degenerate past layer 11 — λ reads `1.02e+01`
at l=12, `3.11e+04` at l=13 and `nan` at l=14, against a per-layer scale of
order `0.5`, with the upstream chain printing "snapping negative variance to
zero" throughout. Including it drops the median power-law R² from 0.972 to
0.988-with-two-garbage-rows and puts an exponent of 10.85 on layer 13. It is
excluded as a degenerate arm, not as an inconvenient one, and `--widths` makes
that choice visible and reversible.

WHAT AN 8x EXTRAPOLATION IS WORTH
----------------------------------
The largest width measured is 128 and the target is 1024. That is three
doublings beyond the data. A power law with R² 0.97 over 64-128 is evidence
that λ moves smoothly in width; it is **not** evidence that the same exponent
holds to 1024, and no amount of R² inside the measured range can be. This
module prints the extrapolation and that sentence together, because the number
without the sentence is the kind of thing that ends up in a write-up.

Run:
    uv run python local_sim/p2_lambda_extrapolate.py \\
        --fit experiments/2026-09-14_slice6a_lambda/lambda_fit.npz \\
        --target 1024 --out experiments/2026-09-14_slice6a_lambda/lam_own.npz
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

ALDO_CSV = os.path.join(_ROOT, "experiments", "2026-09-14_exp_lambda",
                        "lam_compare_summary.csv")


def aldo_table(n_layers):
    if not os.path.exists(ALDO_CSV):
        return None
    rows = np.genfromtxt(ALDO_CSV, delimiter=",", names=True, comments="#")
    return np.asarray(rows[rows.dtype.names[1]], dtype=np.float64)[:n_layers]


def powerfit(widths, lam):
    """Per-layer `(a, p, r2)` for `lam ~ a * n**(-p)`, or NaN where unusable."""
    lw = np.log(np.asarray(widths, dtype=np.float64))
    A = np.vstack([lw, np.ones_like(lw)]).T
    L = lam.shape[1]
    a = np.full(L, np.nan)
    p = np.full(L, np.nan)
    r2 = np.full(L, np.nan)
    for l in range(L):
        y = lam[:, l]
        if not np.all(np.isfinite(y)) or np.any(y <= 0):
            continue
        ly = np.log(y)
        coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
        pred = A @ coef
        den = float(((ly - ly.mean()) ** 2).sum())
        r2[l] = 1.0 - float(((ly - pred) ** 2).sum()) / den if den > 0 else np.nan
        p[l] = -coef[0]
        a[l] = float(np.exp(coef[1]))
    return a, p, r2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fit", required=True)
    ap.add_argument("--widths", default="",
                    help="comma-separated subset. Default: every width in the "
                         "file EXCEPT any that is non-positive or non-finite at "
                         "some layer.")
    ap.add_argument("--target", type=int, default=1024)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    z = np.load(a.fit)
    all_w = [int(w) for w in z["widths"]]
    L = min(len(z["n%d_lam_mean" % w]) for w in all_w)
    if a.widths:
        ws = [int(x) for x in a.widths.split(",") if x]
    else:
        ws = []
        for w in all_w:
            y = z["n%d_lam_mean" % w][:L]
            if np.all(np.isfinite(y)) and np.all(y > 0):
                ws.append(w)
    dropped = [w for w in all_w if w not in ws]

    lam = np.stack([z["n%d_lam_mean" % w][:L] for w in ws], axis=0)
    r2fit = np.stack([z["n%d_r2_mean" % w][:L] for w in ws], axis=0)
    coef_a, p, r2 = powerfit(ws, lam)
    ext = coef_a * float(a.target) ** (-p)
    al = aldo_table(L)

    print("=" * 96)
    print(f"DOES OUR OWN LAMBDA REACH n = {a.target}?")
    print("=" * 96)
    print(f"  fitted on widths {ws}" + (f"   DROPPED {dropped}" if dropped else ""))
    if dropped:
        print("  A dropped width was non-positive or non-finite at some layer:")
        print("  a degenerate arm, excluded as such and named here.")
    print()
    print("  l     exponent p   power R2    lambda(%4d)      504aldo      ratio"
          % a.target)
    for l in range(L):
        if not np.isfinite(p[l]):
            print(f"  {l:<4d}  (unusable)")
            continue
        line = (f"  {l:<4d}  {p[l]:>10.4f}  {r2[l]:>9.5f}  {ext[l]:>14.4e}")
        if al is not None and l < len(al):
            line += f"  {al[l]:>11.4e}  {ext[l] / al[l]:>9.3f}"
        print(line)

    ok = np.isfinite(p)
    print()
    print(f"  power-law R2   min {np.nanmin(r2[ok]):.4f}   "
          f"median {np.nanmedian(r2[ok]):.4f}")
    print(f"  exponent p     {np.nanmin(p[ok]):.3f} .. {np.nanmax(p[ok]):.3f}"
          f"   median {np.nanmedian(p[ok]):.3f}")
    print(f"  the fit's OWN R2 (kappa-4 core on C_off), at n={ws[-1]}: "
          f"{r2fit[-1].min():.3f} .. {r2fit[-1].max():.3f}")
    if al is not None:
        rat = ext[ok] / al[:L][ok]
        within = int(np.sum((rat >= 0.85) & (rat <= 1.15)))
        print()
        print(f"  ours / 504aldo at n={a.target}: median {np.median(rat):.3f}, "
              f"range [{np.min(rat):.3f}, {np.max(rat):.3f}]")
        print(f"  layers within 15% of their table: {within} of {int(ok.sum())}")
        print()
        print("  PRE-REGISTERED PREDICTION (strategist, before this ran):")
        print("    'within 15% per layer at l=1-14'  -> "
              + ("CONFIRMED" if within >= int(ok.sum()) - 1 else "FALSIFIED"))
        print("    'R2 >= 0.6 by layer 4'            -> "
              + ("CONFIRMED" if float(r2fit[-1][4]) >= 0.6 else "FALSIFIED")
              + f"  (R2 = {float(r2fit[-1][4]):.3f} at n={ws[-1]})")
    print()
    print(f"  THE EXTRAPOLATION SPANS {a.target / ws[-1]:.0f}x IN WIDTH beyond the")
    print("  largest measured point. A power law with this R2 shows lambda moves")
    print("  smoothly in width over 64-128. It does NOT show the same exponent")
    print("  holds to 1024, and no R2 inside the measured range can show that.")
    print("  Treat the table as a STARTING POINT for P1, to be checked end to")
    print("  end against score, not as a fitted constant.")

    if a.out:
        d = os.path.dirname(os.path.abspath(a.out))
        if d:
            os.makedirs(d, exist_ok=True)
        np.savez_compressed(
            a.out, lam_target=ext, exponent=p, coef_a=coef_a,
            power_r2=r2, widths=np.asarray(ws), target=np.int64(a.target),
            lam_measured=lam, fit_r2_measured=r2fit,
            aldo=al if al is not None else np.zeros(0))
        meta = {"source_fit": os.path.relpath(a.fit, _ROOT),
                "widths_used": ws, "widths_dropped": dropped,
                "target_width": a.target,
                "model": "lambda_l(n) = a_l * n**(-p_l), per-layer LS in log-log",
                "extrapolation_factor": a.target / ws[-1],
                "caveat": ("the extrapolation spans %.0fx beyond the largest "
                           "measured width; the power law is evidence of smooth "
                           "width dependence over the measured range and not "
                           "evidence that the exponent holds to the target"
                           % (a.target / ws[-1])),
                "label_free": True, "public_labels_used": False,
                "aldo_table_used_as": "comparison target only, never as a "
                                      "fitted constant"}
        with open(os.path.splitext(a.out)[0] + "_meta.json", "w",
                  encoding="utf-8") as fh:
            fh.write(json.dumps(meta, indent=2) + "\n")
        print(f"\n  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
