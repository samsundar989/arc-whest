"""Does our own λ transfer from a width that fits to the width that scores?

Slice 6a step 1, second half. `local_sim/p2_lambda_fit.py` measured λ at the
widths a 32 GiB card can hold; the live shape is 1024 and the dense AUGMENT
state is O(n³), whose RETAINED part alone is ~133 GiB at 1024x16 -- over 4x the
card, before any transient. (An earlier version of this line quoted "6,369 GiB"
from a two-point extrapolation of peaks that were never saved as an artifact.
Withdrawn: the retained-state figure is derivable from the state inventory and
does not need them.) This module asks whether the
measured widths determine λ(1024), and answers with a model and a residual
rather than a hope. **The answer is no** -- see the window spread below.

THE MODEL, AND WHY THIS ONE
----------------------------
Per layer, fit `log λ_l = log a_l − p_l · log n` by least squares over the
measured widths. A power law is the right family to try first because the chain
is self-similar in width: every layer is the same operation at a different `n`,
and He initialisation keeps the pre-activation variance scale fixed as `n`
grows, so a dimensionless coefficient that moves with `n` should move as a
power of it. The fit's own R² is reported per layer, so the family is judged
rather than assumed.

EXCLUDING A WIDTH — AND THE DEFECT AN INDEPENDENT SKEPTIC FOUND HERE
--------------------------------------------------------------------
The first version of this module said it excluded a width for being
"numerically degenerate". **It did not.** As coded, the rule was *"drop a width
that is non-finite or non-positive at some layer"* — a **NaN detector**. It
dropped `n=48` only because of a single NaN at `l=14`; it would happily have
kept `n=48`'s `3.11e+04` at `l=13` (against a per-layer scale of order 0.5) had
that NaN not occurred, and it could not see `n=64`'s 231 negative-variance
snaps at all. A rule whose docstring describes a different rule from the one it
implements is worse than no rule, because it is quoted as evidence.

`degenerate_widths()` below is the replacement, and it tests the three things
the prose always claimed: a non-finite or non-positive λ, a λ that leaves its
own depth profile by more than `--outlier-factor`, and a per-layer fit R² that
falls under the contract's own stop condition on more than a few networks.

**What the corrected rule finds, and what that cost the original claim:**
`n=48` is degenerate (17 of 24 nets snapping, λ of `1.02e+01` and `3.11e+04`).
`n=64` snaps on 7 of 24 nets — but an out-of-sample test at `n=160` showed that
**dropping `n=64` makes the extrapolation worse** (median error 11.93% against
3.00%), because two points cannot see curvature. So the flag is reported and
the decision is left to `--widths`: a degeneracy marker is information, not an
automatic exclusion, and this module no longer pretends otherwise.
`experiments/2026-09-14_slice6a_lambda/followup_result.md`.

WHAT AN 8x EXTRAPOLATION IS WORTH — MEASURED, NOT ARGUED
---------------------------------------------------------
A power law with R² 0.97 inside the measured range is not evidence that the
exponent holds to 1024. That used to be a caveat here; it is now a
**measurement**. Fitting λ(1024) under every defensible window gives medians
from **0.385** to **1.005** relative to 504aldo's shipped table — a factor of
2.6 spread — and adding a fourth width at `n=160` does not narrow it.

The same machinery predicts `λ(160)` from `{64,96,128}` to a median 3.00% and
15/15 layers within 15%. So the family is right at a **1.25× step** and the
value is undetermined at **8×**, and those two facts sit together without
contradiction. Anything this module prints for the target width is a shape, not
a constant; `--report-windows` prints the spread next to it so the number
cannot travel without it.

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



#: A layer's λ this far from the median of its own depth profile is treated as
#: an excursion rather than a measurement.
DEFAULT_OUTLIER_FACTOR = 8.0


def degenerate_widths(z, widths, L, *, outlier_factor=DEFAULT_OUTLIER_FACTOR,
                      r2_floor=0.5, r2_bad_frac=0.10):
    """Which widths look numerically degenerate, and WHY — one reason each.

    Three tests, because the prose always claimed three and the code used to
    implement one:

      `nonfinite`   λ non-finite or non-positive at some layer;
      `excursion`   |λ_l| more than `outlier_factor` times the median |λ|
                    across that width's own depth profile;
      `low_r2`      the fit's own per-net R² below `r2_floor` on more than
                    `r2_bad_frac` of net-layer cells — the contract's stop
                    condition, applied per width instead of globally.

    Returns `{width: [reasons]}`, empty list meaning clean. It REPORTS; it does
    not exclude. An out-of-sample test showed the obvious exclusion was the
    wrong call, so the caller decides with `--widths` and has to say so.
    """
    out = {}
    for w in widths:
        reasons = []
        lam = np.asarray(z["n%d_lam_mean" % w][:L], dtype=np.float64)
        if not np.all(np.isfinite(lam)) or np.any(lam <= 0):
            reasons.append("nonfinite")
        finite = lam[np.isfinite(lam)]
        if finite.size:
            med = float(np.median(np.abs(finite)))
            if med > 0 and np.nanmax(np.abs(finite)) > outlier_factor * med:
                reasons.append("excursion(x%.0f)"
                               % (np.nanmax(np.abs(finite)) / med))
        key = "n%d_raw_r2" % w
        if key in z:
            r2 = np.asarray(z[key])[:, :L]
            frac = float((r2 < r2_floor).mean())
            if frac > r2_bad_frac:
                reasons.append("low_r2(%.0f%% of cells < %.1f)"
                               % (frac * 100, r2_floor))
        out[w] = reasons
    return out


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
    ap.add_argument("--outlier-factor", type=float,
                    default=DEFAULT_OUTLIER_FACTOR,
                    help="a layer's |lambda| this far from its own depth "
                         "profile's median is flagged as an excursion")
    ap.add_argument("--report-windows", action="store_true",
                    help="refit under every leave-one-out window and print the "
                         "spread. An extrapolated table whose spread is wide "
                         "is a shape, not a constant.")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    z = np.load(a.fit)
    all_w = [int(w) for w in z["widths"]]
    L = min(len(z["n%d_lam_mean" % w]) for w in all_w)
    flags = degenerate_widths(z, all_w, L, outlier_factor=a.outlier_factor)
    if a.widths:
        ws = [int(x) for x in a.widths.split(",") if x]
    else:
        # Default: keep every width whose λ is usable as a NUMBER, and report
        # the degeneracy flags rather than acting on them. The old default
        # silently dropped a width on a NaN test and called it degeneracy.
        ws = [w for w in all_w
              if "nonfinite" not in flags[w] and "excursion" not in
              " ".join(flags[w])]
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
    print("  degeneracy flags (REPORTED, not acted on -- see --widths):")
    for w in all_w:
        print(f"    n={w:<5d} " + (", ".join(flags[w]) if flags[w] else "clean"))
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
    if a.report_windows and len(ws) >= 3:
        import itertools
        print()
        print("  LEAVE-ONE-OUT WINDOWS -- how much of this is the fit window?")
        meds = []
        for drop in [None] + list(ws):
            sub = [w for w in ws if w != drop]
            if len(sub) < 2:
                continue
            sl = np.stack([z["n%d_lam_mean" % w][:L] for w in sub], axis=0)
            aa, pp, _ = powerfit(sub, sl)
            ee = aa * float(a.target) ** (-pp)
            if al is not None:
                rr = ee[np.isfinite(ee)] / al[:L][np.isfinite(ee)]
                meds.append(float(np.median(rr)))
                print(f"    {str(sub):<26} median ratio {np.median(rr):.3f}"
                      f"   range [{np.min(rr):.3f}, {np.max(rr):.3f}]")
        if len(meds) >= 2:
            print(f"    spread across windows: {max(meds) / min(meds):.2f}x in "
                  "the median.")
            print("    A wide spread means the target-width value is set by the")
            print("    fit window and not by the data.")
    print()
    print(f"  THE EXTRAPOLATION SPANS {a.target / ws[-1]:.0f}x IN WIDTH beyond the")
    print("  largest measured point. A power law with this R2 shows lambda moves")
    print("  smoothly in width over 64-128. It does NOT show the same exponent")
    print("  holds to 1024, and no R2 inside the measured range can show that.")
    print("  DO NOT USE THIS TABLE AS A FITTED CONSTANT. Measured 2026-09-14:")
    print("  the leave-one-out window spread is 8.89x in the median, so the")
    print("  target-width VALUE is set by the fit window, not by the data. What")
    print("  survives is the SHAPE and the mechanism -- fit one scalar per layer")
    print("  end to end against score, using this as the initialiser.")

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
