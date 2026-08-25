"""
THE GRADER'S RESIDUAL, calibrated on FIVE graded submissions instead of one.

Contract and predictions R1-R5: `experiments/LOG.md`, 2026-08-19 (night).

WHY
---
`docs/CURRENT_STATE.md` classifies the cost half of the evaluator **E1 —
directional only**, and says improving it outranks any further score
optimisation. The reason is that `C = counted_FLOPs + 1e11 * residual_seconds`,
counted FLOPs are exact and reproducible, and residual wall time on the grader's
CPU cannot be measured on this machine — it bills the same file ~4.3x.

But residual is a SMALL part of `C` at our operating point: 9.845 ms x 1e11 =
9.85e8 against 5.29e10 counted, i.e. 1.8%. That is why the shipped file's local
score is only ~8% off the graded one. It is v2.6 that breaks this, by trading
counted FLOPs for operations — 1.7x the ops makes residual ~21% of `C` locally.

WHAT IS NEW HERE
----------------
The repository's standing number is "7.0 us/op", derived from ONE submission
(327743: 1,416 ops in 9.845 ms) as a ratio through the origin. There are **five**
graded submissions with residual telemetry and all five estimator files are
archived. Five points at different operation counts fit a LINE:

    residual_graded  ~  a + b * ops

which separates fixed per-MLP overhead from marginal per-operation cost. A ratio
through the origin cannot do that, and it over-charges every candidate whose op
count differs from 327743's.

THE VALIDATION THAT MAKES IT TRUSTWORTHY
-----------------------------------------
Counted FLOPs are deterministic. If running each archived file locally reproduces
its GRADED counted FLOPs, we are provably running the same programs the grader
ran — and the operation counts are trustworthy by the same argument. That check
comes first and the fit is not reported if it fails.

Run: uv run python local_sim/residual_model.py
"""

import importlib.util
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

import flopscope as flops  # noqa: E402  isort:skip
import flopscope.numpy as fnp  # noqa: E402  isort:skip

import numpy as np  # noqa: E402

flops.configure(symmetry_warnings=False)

import harness  # noqa: E402
from mint_provenance import canonical_mint  # noqa: E402

WIDTH, DEPTH = 256, 32
BUDGET = 272_000_000_000
OUT = os.path.join(_ROOT, "experiments", "2026-08-19_carriers", "residual_model.json")
RESULTS = []

# Legacy-namespace throwaway seeds: this file measures COST, decides nothing, and
# must not draw on a registered block.
# seed-ledger: excludes V
# 24, not 6. Counted FLOPs are deterministic GIVEN a network, but v2.4+ prunes,
# and the alive fraction varies by network: measured per-network sd 3.22% for
# v2.4.1 and exactly 0.00% for v1.1, which does not prune. Six seeds gives a
# standard error of 1.3% and a 2.5% gap then looks like a failure when it is
# noise. This is the same mis-specification that bit predictions B4 and the
# rotation check earlier tonight: a tolerance that ignores the estimator's own
# sampling error.
SEEDS = [5_150_000 + i for i in range(24)]

REPORTS = os.path.join(_ROOT, "experiments", "run_reports")
VERSIONS = [
    ("v0",     "experiments/estimator_v0_archive.py", "2026-08-04_v0_submission_323872_graded_report.json"),
    ("v1.0",   "estimator_v1.py",                     "2026-08-05_v1.0_324075_graded_report.json"),
    ("v1.1",   "estimator_v1_1.py",                   "2026-08-05_v1.1_324094_graded_report.json"),
    ("v2.4",   "estimator_v2_4.py",                   "2026-08-05_v2.4_324420_graded_report.json"),
    ("v2.4.1", "estimator_v2_4_1.py",                 "2026-08-10_v2.4.1_phase1_327743_graded_report.json"),
]
#: Measured locally, never graded. The whole point of the fit is to predict these.
CANDIDATES = [("v2.6", "estimator_v2_6.py"), ("v2.6.1", "estimator_v2_6_1.py")]


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"  {name:<56} {'ok ' if ok else 'FAILED'}  {detail}")
    return ok


def head(t):
    print(f"\n{t}\n{'-' * len(t)}")


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def graded(report):
    with open(os.path.join(REPORTS, report), encoding="utf-8") as fh:
        d = json.load(fh)
    pm = d["evaluation"]["results"]["per_mlp"]
    tel = [e.get("telemetry") or {} for e in pm]
    fl = np.array([float(t.get("flops_used", 0.0)) for t in tel])
    rs = np.array([float(t.get("residual_wall_time_s", 0.0)) for t in tel])
    return {"counted": float(fl.mean()), "residual": float(rs.mean()),
            "residual_median": float(np.median(rs)), "n": len(pm),
            "score": float(d["evaluation"]["results"]["aggregates"]["public"]["score"])}


def measure_local(rel, label):
    """Counted FLOPs and OPERATION COUNT per MLP, with their standard errors."""
    mod = _load(rel, f"_ver_{label.replace('.', '_')}")
    fl, ops = [], []
    for s in SEEDS:
        w_np, est_seed, _ = canonical_mint(WIDTH, DEPTH, s)
        mlp = harness.MLP(weights=[np.asarray(x) for x in w_np],
                          width=WIDTH, depth=DEPTH, seed=est_seed)
        est = mod.Estimator()
        if hasattr(est, "setup"):
            try:
                est.setup(type("Ctx", (), {"seed": 0, "width": WIDTH, "depth": DEPTH})())
            except Exception:
                pass
        ctx = flops.BudgetContext(flop_budget=BUDGET, quiet=True)
        with ctx:
            est.predict(mlp, BUDGET)
        summary = ctx.summary_dict(by_namespace=True)
        ns = summary.get("by_namespace") or {}
        calls = sum(float(v.get("calls", 0.0)) for v in ns.values())
        fl.append(float(ctx.flops_used))
        ops.append(calls)
    fl = np.asarray(fl)
    return (float(fl.mean()), float(np.mean(ops)),
            float(fl.std(ddof=1) / np.sqrt(len(fl))) if len(fl) > 1 else 0.0)


def fit_line(x, y):
    """Least squares y = a + b x, with R^2 and leave-one-out prediction error."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    pred = a + b * x
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    loo = []
    for i in range(len(x)):
        m = np.ones(len(x), dtype=bool)
        m[i] = False
        A2 = np.vstack([np.ones(m.sum()), x[m]]).T
        c2, *_ = np.linalg.lstsq(A2, y[m], rcond=None)
        loo.append(abs((c2[0] + c2[1] * x[i]) - y[i]))
    return a, b, r2, float(np.mean(loo))


def main():
    print("=" * 100)
    print("THE GRADER'S RESIDUAL — calibrated on five graded submissions")
    print("=" * 100)

    head("1. do we reproduce the graded programs? (prediction R1)")
    rows = []
    print(f"  {'ver':<8}{'local counted':>16}{'graded counted':>16}{'gap':>9}"
          f"{'local ops':>11}{'graded resid':>14}{'n sigma':>10}")
    print("  " + "-" * 86)
    for label, rel, rep in VERSIONS:
        g = graded(rep)
        lf, lo, se = measure_local(rel, label)
        gap = (lf - g["counted"]) / g["counted"]
        nsig = abs(lf - g["counted"]) / se if se > 0 else 0.0
        rows.append({"ver": label, "local_counted": lf, "graded_counted": g["counted"],
                     "gap": gap, "se": se, "n_sigma": nsig, "ops": lo,
                     "residual": g["residual"],
                     "residual_median": g["residual_median"], "score": g["score"]})
        print(f"  {label:<8}{lf:>16.4e}{g['counted']:>16.4e}{gap * 100:>8.2f}%"
              f"{lo:>11.0f}{g['residual'] * 1e3:>13.3f}ms{nsig:>10.1f}")
    worst = max(r["n_sigma"] for r in rows)
    check("R1: local counted reproduces graded counted within 3 standard errors",
          worst < 3.0, f"worst {worst:.1f} sigma "
          f"(the estimator's own per-network spread, not a program mismatch)")

    head("2. what explains the grader's residual? (prediction R2)")
    y = np.array([r["residual"] for r in rows])
    ops = np.array([r["ops"] for r in rows])
    cnt = np.array([r["graded_counted"] for r in rows])
    a_o, b_o, r2_o, loo_o = fit_line(ops, y)
    a_c, b_c, r2_c, loo_c = fit_line(cnt, y)
    print(f"  residual ~ a + b * OPS      a = {a_o * 1e3:+.3f} ms   "
          f"b = {b_o * 1e6:.3f} us/op    R^2 = {r2_o:.4f}   LOO {loo_o * 1e3:.3f} ms")
    print(f"  residual ~ a + b * COUNTED  a = {a_c * 1e3:+.3f} ms   "
          f"b = {b_c * 1e12:.3f} ps/FLOP  R^2 = {r2_c:.4f}   LOO {loo_c * 1e3:.3f} ms")
    check("R2: operation count explains residual better than counted FLOPs",
          r2_o > r2_c, f"R^2 {r2_o:.4f} vs {r2_c:.4f}")
    v0, v10 = rows[0], rows[1]
    check("R2: v1.0 is the discriminating case — 12x the FLOPs, LESS residual",
          v10["graded_counted"] / v0["graded_counted"] > 10
          and v10["residual"] < v0["residual"],
          f"{v10['graded_counted'] / v0['graded_counted']:.1f}x FLOPs, "
          f"{v10['residual'] * 1e3:.2f} ms vs {v0['residual'] * 1e3:.2f} ms")

    head("3. the fitted model (predictions R3, R4)")
    check("R3: positive fixed overhead, 1-5 ms", 1e-3 <= a_o <= 5e-3,
          f"intercept {a_o * 1e3:+.3f} ms")
    check("R4: marginal cost 1-5 us/op, BELOW the 7.0 us/op ratio-through-origin",
          1e-6 <= b_o <= 5e-6,
          f"{b_o * 1e6:.3f} us/op vs the standing 7.0")
    print(f"\n  {'ver':<8}{'ops':>9}{'observed':>12}{'fitted':>10}{'resid':>10}"
          f"{'ratio-thru-origin':>20}")
    print("  " + "-" * 70)
    for r in rows:
        f = a_o + b_o * r["ops"]
        thru = 6.953e-6 * r["ops"]
        print(f"  {r['ver']:<8}{r['ops']:>9.0f}{r['residual'] * 1e3:>11.3f}ms"
              f"{f * 1e3:>9.3f}ms{(f - r['residual']) * 1e3:>+9.3f}ms"
              f"{thru * 1e3:>18.3f}ms")

    head("4. what it says about the un-graded candidates (prediction R5)")
    cand = []
    for label, rel in CANDIDATES:
        if not os.path.exists(os.path.join(_ROOT, rel)):
            continue
        lf, lo, _se = measure_local(rel, label)
        pred = a_o + b_o * lo
        thru = 6.953e-6 * lo
        cand.append({"ver": label, "counted": lf, "ops": lo,
                     "residual_fit": pred, "residual_thru": thru})
        print(f"  {label:<8} counted {lf:.4e}   ops {lo:>6.0f}   "
              f"projected graded residual: fitted {pred * 1e3:.3f} ms   "
              f"ratio-through-origin {thru * 1e3:.3f} ms")

    base = [r for r in rows if r["ver"] == "v2.4.1"][0]
    print(f"\n  {'candidate':<10}{'model':<26}{'C/B ctrl':>10}{'C/B cand':>10}"
          f"{'cost ratio':>12}{'net V.c (MSE 0.9942x)':>24}")
    print("  " + "-" * 92)
    out_pairs = {}
    for c in cand:
        for mname, rc, ra in (
            ("fitted line (5 points)", a_o + b_o * base["ops"], c["residual_fit"]),
            ("ratio through origin", 6.953e-6 * base["ops"], c["residual_thru"]),
        ):
            cb_c = (base["graded_counted"] + 1e11 * rc) / BUDGET
            cb_a = (c["counted"] + 1e11 * ra) / BUDGET
            cost_ratio = cb_c / cb_a
            net = cost_ratio * 0.9942
            out_pairs[f"{c['ver']} | {mname}"] = {
                "cb_control": cb_c, "cb_candidate": cb_a,
                "cost_ratio": cost_ratio, "net": net}
            print(f"  {c['ver']:<10}{mname:<26}{cb_c:>10.4f}{cb_a:>10.4f}"
                  f"{cost_ratio:>11.4f}x{net:>23.4f}x")

    lo_ops, hi_ops = float(ops.min()), float(ops.max())
    for c in cand:
        if lo_ops <= c["ops"] <= hi_ops:
            note = "interpolation"
        else:
            note = "EXTRAPOLATION by %.2fx beyond the largest calibration point" % (
                c["ops"] / hi_ops)
        print("\n  !! %s sits at %.0f ops; the fit is calibrated on %.0f-%.0f."
              % (c["ver"], c["ops"], lo_ops, hi_ops))
        print("     %s. This is the new load-bearing assumption -- much weaker than"
              % note)
        print("     the old one, but still an assumption.")
    # This was `check("extrapolation is reported, not hidden", True, ...)`: a
    # literal condition, so it printed green whether or not anything had been
    # reported. There is no property here to test -- the block above prints the
    # extrapolation unconditionally -- so this is a REPORT and is now spelled as
    # one. Found 2026-08-25 by the third independent P1-PRE review, using that
    # slice's own literal-condition scanner against the wider tree.
    print("  extrapolation is reported, not hidden: calibrated %.0f-%.0f ops; "
          "candidates at %s"
          % (lo_ops, hi_ops, [int(c["ops"]) for c in cand]))

    v261 = out_pairs.get("v2.6.1 | fitted line (5 points)")
    if v261:
        check("R5: the calibrated net V.c lands inside the 1.068-1.096x bracket",
              1.068 <= v261["net"] <= 1.096, f"{v261['net']:.4f}x")

    res = {"versions": rows, "candidates": cand,
           "fit_ops": {"intercept_s": a_o, "slope_s_per_op": b_o, "r2": r2_o,
                       "loo_mae_s": loo_o},
           "fit_counted": {"intercept_s": a_c, "slope_s_per_flop": b_c, "r2": r2_c,
                           "loo_mae_s": loo_c},
           "projections": out_pairs}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(f"\nsaved {OUT}")

    n_ok = sum(1 for r in RESULTS if r)
    print(f"\n{n_ok}/{len(RESULTS)} checks passed")
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
