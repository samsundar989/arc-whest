"""
The proof that `local_sim/competition_score.py` IS the grader's arithmetic.

Stage 0 / P0.1. Contract and written predictions P1-P7: `experiments/LOG.md`,
2026-08-19 (night).

`competition_score.py` is a reimplementation, not a wrapper, so that it runs in
the torch-only GPU venv. A reimplementation is worth nothing without this file.
Three standards of evidence, in increasing order of how much they can catch:

  1. SCALAR PARITY  — bitwise `==` against `whestbench.budget` on randomised rows.
  2. EVALUATOR PARITY — drive the REAL `whestbench.scoring.evaluate_estimator`
     over synthetic contests (clean, raising, and budget-exhausting estimators)
     and demand our aggregate reproduce its aggregate exactly. This is the only
     check that exercises the failure semantics on the real code path.
  3. GRADED REPLAY — replay the 50 published per-MLP rows of submissions 324420
     (v2.4) and 327743 (v2.4.1) through our scorer. Read-only; no network.

Plus the regression fixture the whole slice exists for: a case where arithmetic
aggregation and the geometric mean of per-MLP ratios select OPPOSITE winners.

Run: uv run python local_sim/test_competition_score.py
"""

import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]

import flopscope as flops  # noqa: E402  isort:skip
import flopscope.numpy as fnp  # noqa: E402  isort:skip

import numpy as np  # noqa: E402

for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

flops.configure(symmetry_warnings=False)

import whestbench  # noqa: E402
from whestbench import budget as wb_budget  # noqa: E402
from whestbench.scoring import (ContestSpec, evaluate_estimator,  # noqa: E402
                                make_contest)
from whestbench.sdk import BaseEstimator  # noqa: E402

from competition_score import (LAMBDA_FLOPS_PER_SECOND,  # noqa: E402
                               DEFAULT_LAMBDA_FLOPS_PER_SECOND,
                               PHASE1_LAMBDA_FLOPS_PER_SECOND,
                               DEFAULT_FLOP_BUDGET,
                               MULTIPLIER_FLOOR, PHASE1_FLOP_BUDGET, MlpRow,
                               effective_compute, format_paired,
                               is_combined_budget_exhausted, mlp_score,
                               naive_factorised_objective, over_budget_rows,
                               paired_compare, paired_compare_rows,
                               reported_scores_by_index,
                               rows_from_graded_report, score_multiplier,
                               score_suite, zero_prediction_mse, decide, Verdict,
                               align_rows_by_identity, PairingError, _t_ppf, _t975,
                               paired_diff_ci,
                               noise_corrected_compare,
                               NoiseAwareCompare)

RESULTS = []
REPORTS = os.path.join(_ROOT, "experiments", "run_reports")
V24 = os.path.join(REPORTS, "2026-08-05_v2.4_324420_graded_report.json")
V241 = os.path.join(REPORTS, "2026-08-10_v2.4.1_phase1_327743_graded_report.json")

#: EVERY published graded report this repository holds is from PHASE 1
#: (`v1-phase1`: lambda 1e11, B = 2.72e11). Since P2-P0 the module default is the
#: LIVE round, where lambda is **0**, so a Phase-1 replay must name its own rate.
#: This is the only sanctioned way to do it, and it exists so the naming is
#: visible at every call site rather than inherited.
#:
#: Replaying a Phase-1 report at lambda 0 does not error. It silently drops
#: `lambda * R` from every row -- 10-13% of C on this hardware -- and the
#: reproduction checks below (P4/P5/P7) are what caught it when the default moved.
P1 = dict(lam=PHASE1_LAMBDA_FLOPS_PER_SECOND)


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"  {name:<58} {'ok ' if ok else 'FAILED'}  {detail}")
    return ok


def head(title):
    print(f"\n{title}\n{'-' * len(title)}")


# --------------------------------------------------------------------------- #
# 1. Scalar parity against the installed whestbench — prediction P1
# --------------------------------------------------------------------------- #


def t_scalar_parity():
    head("1. scalar parity vs whestbench.budget (bitwise ==)")
    rng = np.random.default_rng(20260819)
    n = 12_000

    # Deliberately adversarial coverage: budgets that are zero and negative, C/B
    # straddling the floor, residual terms from 0 to seconds, and huge/small MSE.
    budgets = np.concatenate([
        rng.uniform(1e9, 1e12, n - 400),
        np.zeros(200),
        -rng.uniform(1, 1e9, 100),
        np.full(100, PHASE1_FLOP_BUDGET),
    ])
    rng.shuffle(budgets)
    fl = rng.uniform(0.0, 3e11, n)
    rs = np.concatenate([np.zeros(300), rng.uniform(0.0, 1.0, n - 300)])
    rng.shuffle(rs)
    failed = rng.random(n) < 0.15

    bad_c = bad_m = bad_x = 0
    for i in range(n):
        c_ours = effective_compute(fl[i], rs[i])
        c_theirs = wb_budget.effective_compute(fl[i], rs[i])
        if c_ours != c_theirs:
            bad_c += 1
        m_ours = score_multiplier(c_ours, budgets[i], failed=bool(failed[i]))
        m_theirs = wb_budget.score_multiplier(c_theirs, budgets[i], failed=bool(failed[i]))
        if m_ours != m_theirs:
            bad_m += 1
        x_ours = is_combined_budget_exhausted(fl[i], rs[i], budgets[i])
        x_theirs = wb_budget.is_combined_budget_exhausted(fl[i], rs[i], budgets[i])
        if x_ours != x_theirs:
            bad_x += 1

    check("effective_compute exact on 12,000 rows", bad_c == 0, f"{bad_c} mismatches")
    check("score_multiplier exact on 12,000 rows", bad_m == 0, f"{bad_m} mismatches")
    check("is_combined_budget_exhausted exact", bad_x == 0, f"{bad_x} mismatches")
    check(
        "the DEPRECATED lambda alias matches the package's deprecated alias",
        LAMBDA_FLOPS_PER_SECOND == wb_budget.LAMBDA_FLOPS_PER_SECOND,
        f"{LAMBDA_FLOPS_PER_SECOND:.3e}",
    )
    # The alias above is 1e11 in both, and it is NOT the live rate. That
    # coincidence of spelling is exactly what let a Phase-1 default survive into
    # Phase 2 unnoticed, so the LIVE rate gets its own check.
    check(
        "the LIVE default lambda matches the package's live default",
        DEFAULT_LAMBDA_FLOPS_PER_SECOND == wb_budget.DEFAULT_LAMBDA_FLOPS_PER_SECOND,
        f"{DEFAULT_LAMBDA_FLOPS_PER_SECOND:.3e}",
    )
    check(
        "...and it is ZERO, so C == F and residual is gated, not priced",
        DEFAULT_LAMBDA_FLOPS_PER_SECOND == 0.0
        and effective_compute(1.0e9, 1.0e6) == 1.0e9,
        f"C({1.0e9:.0e} FLOPs, 1e6 s) = {effective_compute(1.0e9, 1.0e6):.6e}",
    )
    check(
        "the Phase-1 rate is still available, explicitly, for replay",
        PHASE1_LAMBDA_FLOPS_PER_SECOND == wb_budget.PHASE1_LAMBDA_FLOPS_PER_SECOND
        == 1e11,
        f"{PHASE1_LAMBDA_FLOPS_PER_SECOND:.3e}",
    )
    check(
        "the LIVE budget matches the package's live default",
        DEFAULT_FLOP_BUDGET == float(wb_budget.DEFAULT_FLOP_BUDGET) == float(2 ** 41),
        f"{DEFAULT_FLOP_BUDGET:,.0f}",
    )
    check(
        "the Phase-1 budget is still available, explicitly, for replay",
        PHASE1_FLOP_BUDGET == float(wb_budget.PHASE1_FLOP_BUDGET) == 272e9,
        f"{PHASE1_FLOP_BUDGET:,.0f}",
    )

    # A non-0.1 floor cannot be checked against the package (0.14.0 hardcodes
    # 0.1), so check the two properties that must hold for ANY future floor.
    floor = 0.5
    ok = True
    for i in range(2000):
        c = effective_compute(fl[i], rs[i])
        m = score_multiplier(c, budgets[i], failed=False, floor=floor)
        if budgets[i] <= 0:
            ok &= m == 1.0
        else:
            ok &= m == max(floor, c / budgets[i]) and m >= floor
    check("configurable floor: max(floor, C/B), B<=0 -> 1.0", ok, "floor=0.5")

    # The floor boundary itself, exactly.
    b = 1e11
    check(
        "floor boundary C/B = 0.1 exactly",
        score_multiplier(0.1 * b, b, failed=False) == 0.1
        and wb_budget.score_multiplier(0.1 * b, b, failed=False) == 0.1,
    )
    check(
        "below floor is clamped, above floor is not capped",
        score_multiplier(0.01 * b, b, failed=False) == 0.1
        and score_multiplier(50.0 * b, b, failed=False) == 50.0,
    )
    check(
        "failure multiplier is 1.0, NOT the 0.1 floor",
        score_multiplier(0.0, b, failed=True) == 1.0
        and wb_budget.score_multiplier(0.0, b, failed=True) == 1.0,
    )


def t_aggregation_parity():
    head("2. aggregation parity — arithmetic mean, same accumulation as the grader")
    rng = np.random.default_rng(7)
    for trial in range(200):
        vals = [float(v) for v in rng.lognormal(-15.0, 1.5, rng.integers(2, 400))]
        theirs = float(fnp.mean(fnp.asarray(vals)))
        ours = float(np.mean(np.asarray(vals, dtype=np.float64)))
        if ours != theirs:
            check("np.mean == fnp.mean on score lists", False, f"trial {trial}")
            return
    check("np.mean == fnp.mean on score lists (200 trials)", True, "bitwise")


# --------------------------------------------------------------------------- #
# 2. Evaluator parity — drive the real scoring loop, including failures
# --------------------------------------------------------------------------- #


class _CleanEstimator(BaseEstimator):
    """Cheap, valid, and deliberately heteroscedastic so MSE and cost correlate.

    `work` sets how many counted matmuls to burn. Raising it makes the per-MLP
    effective compute dominated by *counted* FLOPs rather than by this machine's
    noisy residual wall time, which is the only way to place multipliers reliably
    above the floor in a test.
    """

    def __init__(self, scale=1.0, work=0):
        self.scale = scale
        self.work = work
        self._i = 0

    def predict(self, mlp, budget):
        self._i += 1
        # Vary work per MLP so effective_compute varies across the suite.
        reps = self.work + 1 + (self._i % 4)
        a = mlp.weights[0]
        acc = fnp.zeros_like(a)
        for _ in range(reps):
            acc = acc + a @ a
        col = fnp.linalg.norm(mlp.weights[0], axis=0) / math.sqrt(2.0 * math.pi)
        rows = [col * (self.scale + 0.01 * k) for k in range(mlp.depth)]
        return fnp.stack(rows, axis=0) + 0.0 * acc[0]


class _RaisingEstimator(_CleanEstimator):
    """Raises on selected MLPs — the grader's exception path (C recorded as 0)."""

    def __init__(self, fail_on, **kw):
        super().__init__(**kw)
        self.fail_on = set(fail_on)
        self._n = -1

    def predict(self, mlp, budget):
        self._n += 1
        if self._n in self.fail_on:
            raise RuntimeError(f"synthetic failure on mlp {self._n}")
        return super().predict(mlp, budget)


class _GreedyEstimator(_CleanEstimator):
    """Burns FLOPs until the budget is exhausted — the exhaustion path."""

    def __init__(self, **kw):
        super().__init__(work=200, **kw)


def _rows_from_inprocess(result, spec):
    """Rebuild MlpRows from an in-process `evaluate_estimator` result."""
    out = []
    for e in result["per_mlp"]:
        out.append(
            MlpRow(
                mse=float(e["final_layer_mse"]),
                flops_used=float(e.get("flops_used", 0.0) or 0.0),
                residual_wall_time_s=float(e.get("residual_wall_time_s", 0.0) or 0.0),
                budget=float(spec.flop_budget),
                failed=bool(e.get("error_code"))
                or bool(e.get("budget_exhausted"))
                or bool(e.get("time_exhausted"))
                or bool(e.get("residual_wall_time_exhausted"))
                or bool(e.get("combined_budget_exhausted")),
                name=str(e.get("mlp_name", "")),
            )
        )
    return out


def _run_case(label, estimator, spec, data, expect_failures=None):
    res = evaluate_estimator(estimator, data)
    rows = _rows_from_inprocess(res, spec)
    ours = score_suite(rows)
    theirs = float(res["adjusted_final_layer_score"])

    per_ok = all(
        ours.scores[i] == float(res["per_mlp"][i]["adjusted_final_layer_score"])
        for i in range(len(rows))
    )
    check(f"{label}: per-MLP score bitwise == grader", per_ok)
    check(
        f"{label}: aggregate objective == grader",
        ours.objective == theirs,
        f"{ours.objective:.12e} vs {theirs:.12e}",
    )
    check(
        f"{label}: n_failed == grader",
        ours.n_failed == int(res["n_failed_mlps"]),
        f"{ours.n_failed} vs {res['n_failed_mlps']}",
    )
    check(
        f"{label}: mean multiplier == grader",
        abs(ours.mean_multiplier - float(res["mean_score_multiplier"])) < 1e-12,
        f"{ours.mean_multiplier:.6f}",
    )
    if expect_failures is not None:
        check(
            f"{label}: expected {expect_failures} failure(s) actually occurred",
            ours.n_failed == expect_failures,
            f"got {ours.n_failed}",
        )
    return res, ours


def t_evaluator_parity():
    head("3. evaluator parity — the REAL whestbench scoring loop")
    # The budget must be large enough that lambda * residual does not by itself
    # exhaust it: at lambda = 1e11 a 12 ms residual is 1.2e9 FLOP-equivalents, so
    # a "small" budget of 4e7 makes EVERY MLP a combined-budget failure. That is
    # the grader's real arithmetic, not a quirk — it is why the first draft of
    # this test scored zero clean rows. Use the Phase-1 budget for the clean arm.
    spec = ContestSpec(
        width=16,
        depth=4,
        n_mlps=8,
        flop_budget=int(PHASE1_FLOP_BUDGET),
        ground_truth_samples=4096,
        seed=20260819,
        wall_time_limit_s=120.0,
    )
    data = make_contest(spec)

    res_c, ours_c = _run_case("clean (at floor)", _CleanEstimator(), spec, data, expect_failures=0)

    # A second clean arm whose budget is derived from the FIRST arm's observed
    # compute, so multipliers sit strictly above the floor and vary per MLP.
    # This is where an MSE/cost correlation bug would show up under the real loop.
    #
    # The budget has to be re-derived rather than guessed: at width 16 the counted
    # FLOPs are trivial, so C is dominated by lambda * residual, and this machine's
    # residual is noisy enough that a single-shot budget lands either below the
    # floor or over the combined cap. Widen until every row is strictly inside
    # (floor*B, B]. Parity is asserted on every attempt regardless.
    WORK = 1400  # ~1.5e10 counted FLOPs/MLP at width 128, vs ~1e9 from lambda*R
    probe = ContestSpec(
        width=128, depth=4, n_mlps=6, flop_budget=int(1e13),
        ground_truth_samples=4096, seed=515, wall_time_limit_s=180.0,
    )
    probe_data = make_contest(probe)
    _, probe_score = _run_case("clean (budget probe)", _CleanEstimator(work=WORK), probe, probe_data)
    b_above = float(probe_score.effective_computes.max()) * 1.3
    ours_a = None
    for attempt in range(3):
        spec_a = ContestSpec(
            width=128, depth=4, n_mlps=6, flop_budget=int(b_above),
            ground_truth_samples=4096, seed=515, wall_time_limit_s=180.0,
        )
        data_a = make_contest(spec_a)
        _, ours_a = _run_case(
            f"clean (above floor, try {attempt + 1})", _CleanEstimator(work=WORK), spec_a, data_a
        )
        if ours_a.n_failed == 0 and float(ours_a.multipliers.min()) > MULTIPLIER_FLOOR:
            break
        b_above = float(ours_a.effective_computes.max()) * 1.3
    check(
        "above-floor arm: no failures, every multiplier strictly above the floor",
        ours_a.n_failed == 0
        and float(ours_a.multipliers.min()) > MULTIPLIER_FLOOR
        and float(ours_a.multipliers.std()) > 0.0,
        f"mult in [{ours_a.multipliers.min():.3f}, {ours_a.multipliers.max():.3f}]",
    )

    res_f, ours_f = _run_case(
        "raising (2 of 8)", _RaisingEstimator(fail_on=(2, 5)), spec, data, expect_failures=2
    )

    # The failure path must use the ZERO-prediction MSE and multiplier 1.0.
    e = res_f["per_mlp"][2]
    zmse = zero_prediction_mse(data.final_targets[2])
    check(
        "failure MSE == zero_prediction_mse(target)",
        abs(zmse - float(e["final_layer_mse"])) <= 1e-12 * max(1.0, abs(zmse)),
        f"{zmse:.6e} vs {float(e['final_layer_mse']):.6e}",
    )
    check(
        "failure row: multiplier forced to 1.0 and C recorded as 0",
        float(e["effective_compute"]) == 0.0
        and abs(float(e["adjusted_final_layer_score"]) - float(e["final_layer_mse"])) < 1e-30,
    )

    # Exhaustion path: real flops recorded, multiplier still 1.0.
    tight = ContestSpec(
        width=16,
        depth=4,
        n_mlps=4,
        flop_budget=200_000,
        ground_truth_samples=2048,
        seed=99,
        wall_time_limit_s=120.0,
    )
    tdata = make_contest(tight)
    res_x, ours_x = _run_case("budget-exhausting", _GreedyEstimator(), tight, tdata)
    ex = [e for e in res_x["per_mlp"] if e.get("budget_exhausted")]
    check(
        "exhaustion path exercised (flops recorded, mult still 1.0)",
        len(ex) > 0
        and all(
            float(e["flops_used"]) > 0
            and abs(float(e["adjusted_final_layer_score"]) - float(e["final_layer_mse"])) < 1e-30
            for e in ex
        ),
        f"{len(ex)}/{len(res_x['per_mlp'])} exhausted",
    )


def t_heterogeneous_budgets():
    head("4. heterogeneous per-MLP budgets (whestbench 0.14.0 has one B; we must not assume it)")
    rng = np.random.default_rng(4242)
    rows, expect = [], []
    for _ in range(500):
        b = float(rng.uniform(1e9, 1e12))
        f = float(rng.uniform(0, 3e11))
        r = float(rng.uniform(0, 0.5))
        m = float(rng.lognormal(-14, 1.0))
        rows.append(MlpRow(mse=m, flops_used=f, residual_wall_time_s=r, budget=b))
        expect.append(
            m * wb_budget.score_multiplier(wb_budget.effective_compute(f, r), b, failed=False)
        )
    # "ignore": this test probes the RAW identity against whestbench's scalar
    # primitives, which do not themselves apply the combined-budget check.
    s = score_suite(rows, on_over_budget="ignore")
    check(
        "per-row scores exact against whestbench with per-row B",
        all(s.scores[i] == expect[i] for i in range(len(rows))),
    )
    check(
        "objective == arithmetic mean of those rows",
        s.objective == float(np.mean(np.asarray(expect, dtype=np.float64))),
    )


def t_over_budget_guard():
    head("4b. combined-budget guard — C > B with failed=False must not be scored silently")
    b = 1e11

    # --- PHASE 1: residual is PRICED, so a slow row can go over budget on time
    # alone. This is the regime the guard was written for, and it must keep
    # working for replay, so the Phase-1 rate is passed EXPLICITLY. It is no
    # longer the default and must never be reached by inheritance again.
    lam1 = PHASE1_LAMBDA_FLOPS_PER_SECOND
    ok_row = MlpRow(mse=1e-6, flops_used=5e10, residual_wall_time_s=0.1, budget=b)
    bad_row = MlpRow(mse=1e-6, flops_used=5e10, residual_wall_time_s=1.0, budget=b)
    check(
        "P1: the grader itself would fail that row",
        is_combined_budget_exhausted(5e10, 1.0, b, lam1)
        and not is_combined_budget_exhausted(5e10, 0.1, b, lam1),
        f"C = {bad_row.effective_compute(lam1):.3e} vs B = {b:.3e}",
    )
    check("P1: default on_over_budget='raise' refuses it",
          _raises(lambda: score_suite([ok_row, bad_row], lam=lam1)))
    check("P1: over_budget_rows names the offender",
          over_budget_rows([ok_row, bad_row], lam=lam1) == [1])
    failed = score_suite([ok_row, bad_row], lam=lam1, on_over_budget="fail")
    check(
        "P1: on_over_budget='fail' forces multiplier 1.0 on that row only",
        failed.multipliers[1] == 1.0 and abs(failed.multipliers[0] - 0.6) < 1e-12,
        f"mults {failed.multipliers.tolist()}",
    )
    ignored = score_suite([ok_row, bad_row], lam=lam1, on_over_budget="ignore")
    check(
        "P1: on_over_budget='ignore' keeps the raw uncapped identity",
        abs(ignored.multipliers[1] - 1.5) < 1e-12,
        f"mult {ignored.multipliers[1]:.4f}",
    )

    # --- PHASE 2: lambda is 0, so C == F. Residual CANNOT put a row over budget
    # by any amount — it is gated at 0.4 s instead, which is a failure path the
    # lifecycle owns, not an arithmetic one this module can see. The only way over
    # budget now is FLOPs. Both halves are asserted, because "the guard still
    # fires" and "it no longer fires on residual" are different claims.
    slow = MlpRow(mse=1e-6, flops_used=5e10, residual_wall_time_s=1e6, budget=b)
    check(
        "P2: an absurdly slow row is NOT over budget — lambda is 0, so C == F",
        not is_combined_budget_exhausted(5e10, 1e6, b),
        f"C = {slow.effective_compute():.3e} vs B = {b:.3e} at 1e6 s of residual",
    )
    check("P2: ...and the guard therefore does not fire on it",
          over_budget_rows([slow]) == [])
    fat = MlpRow(mse=1e-6, flops_used=1.5e11, residual_wall_time_s=0.0, budget=b)
    check(
        "P2: a row over budget on FLOPs alone still trips the guard",
        is_combined_budget_exhausted(1.5e11, 0.0, b)
        and over_budget_rows([ok_row, fat]) == [1],
        f"C = {fat.effective_compute():.3e} vs B = {b:.3e}",
    )
    check("P2: default on_over_budget='raise' refuses it",
          _raises(lambda: score_suite([ok_row, fat])))
    p2_failed = score_suite([ok_row, fat], on_over_budget="fail")
    check(
        "P2: 'fail' forces multiplier 1.0 on that row only",
        p2_failed.multipliers[1] == 1.0 and abs(p2_failed.multipliers[0] - 0.5) < 1e-12,
        f"mults {p2_failed.multipliers.tolist()}",
    )

    check("an unknown mode is rejected", _raises(lambda: score_suite([ok_row], on_over_budget="x")))
    check("clean suites are unaffected by the guard", score_suite([ok_row]).n == 1)


# --------------------------------------------------------------------------- #
# 3. The failure penalty — prediction P2
# --------------------------------------------------------------------------- #


def t_failure_cost():
    head("5. what one failure costs — prediction P2, FALSIFIED AS STATED; the exact law")
    with open(V241, encoding="utf-8") as fh:
        rep = json.load(fh)
    rows = rows_from_graded_report(rep)
    base = score_suite(rows, **P1)

    # P2 predicted "> 5x". That was wrong, and the reason is worth keeping: the
    # multiplier-only failure penalty is EXACTLY 1 / mult_i, and mult_i is a
    # per-MLP quantity. P2 was computed off the SUITE MEAN multiplier (0.19896 ->
    # 5.03x); row 0's own multiplier is 0.2137 -> 4.68x. Any row with C/B > 0.2
    # falls under 5x. The test now asserts the exact law instead of the wrong
    # threshold, and reports the suite-wide range.
    i = 0
    r = rows[i]
    normal = base.scores[i]
    mult_only = MlpRow(
        mse=r.mse,
        flops_used=r.flops_used,
        residual_wall_time_s=r.residual_wall_time_s,
        budget=r.budget,
        failed=True,
    ).score(**P1)
    # `base` above was scored at the Phase-1 rate, and these row methods take the
    # rate too. Mixing a P1 suite with a lambda-0 row method compares two
    # different identities and the ratio stops being 1/mult_i at all.
    ratio = mult_only / normal
    check(
        "failure penalty is EXACTLY 1/mult_i (multiplier half only)",
        abs(ratio - 1.0 / r.multiplier(**P1)) < 1e-12,
        f"{ratio:.3f}x   (mult {r.multiplier(**P1):.4f} -> 1.0)",
    )
    pen = 1.0 / base.multipliers
    check(
        "P2 as written (> 5x) is FALSE for part of the real suite",
        float(pen.min()) < 5.0,
        f"penalty range [{pen.min():.2f}x, {pen.max():.2f}x], "
        f"at suite-mean mult {base.mean_multiplier:.5f} -> {1 / base.mean_multiplier:.2f}x",
    )

    flipped = list(rows)
    flipped[i] = MlpRow(
        mse=r.mse,
        flops_used=r.flops_used,
        residual_wall_time_s=r.residual_wall_time_s,
        budget=r.budget,
        failed=True,
    )
    inflated = score_suite(flipped)
    check(
        "one failed MLP in 50 moves the objective measurably",
        inflated.objective > base.objective,
        f"{base.objective:.6e} -> {inflated.objective:.6e} "
        f"({(inflated.objective / base.objective - 1) * 100:+.2f}%, MSE held fixed)",
    )
    print(
        f"    note: a real failure ALSO replaces mse with mean(target^2); this "
        f"check isolates the multiplier half only, which is the falsifiable one."
    )


# --------------------------------------------------------------------------- #
# 4. MSE/cost correlation — prediction P6
# --------------------------------------------------------------------------- #


def t_correlation_preserved():
    head("6. MSE/cost correlation — prediction P6 (the factorised form is biased)")
    with open(V241, encoding="utf-8") as fh:
        rep = json.load(fh)
    rows = rows_from_graded_report(rep)
    s = score_suite(rows, **P1)
    naive = naive_factorised_objective(rows, **P1)
    rel = abs(naive - s.objective) / s.objective
    rho = float(np.corrcoef(s.mses, s.multipliers)[0, 1])
    check(
        "P6: mean(mse)*mean(mult) != mean(mse*mult) on real graded rows",
        rel > 1e-12,
        f"rel bias {rel * 100:+.4f}%   corr(mse, mult) = {rho:+.4f}",
    )

    # A synthetic case with strong positive correlation, where the bias is large
    # and its SIGN is predictable: positive correlation => factorised UNDERstates.
    rng = np.random.default_rng(11)
    z = rng.normal(size=400)
    rows2 = [
        MlpRow(
            mse=float(np.exp(-14 + 1.2 * zi)),
            flops_used=float(5e10 * np.exp(0.25 * zi)),
            residual_wall_time_s=0.01,
            budget=PHASE1_FLOP_BUDGET,
        )
        for zi in z
    ]
    s2 = score_suite(rows2)
    n2 = naive_factorised_objective(rows2)
    check(
        "positive corr => factorised UNDERstates the objective",
        n2 < s2.objective,
        f"{n2:.6e} < {s2.objective:.6e}  ({(n2 / s2.objective - 1) * 100:+.2f}%)",
    )


# --------------------------------------------------------------------------- #
# 5. The regression fixture: arithmetic and geometric pick opposite winners — P3
# --------------------------------------------------------------------------- #


def t_sign_flip_fixture():
    head("7. REGRESSION FIXTURE — arithmetic vs geometric select opposite winners (P3)")
    with open(V241, encoding="utf-8") as fh:
        rep = json.load(fh)
    rows = rows_from_graded_report(rep)
    ctrl = score_suite(rows, **P1).scores.copy()
    n = ctrl.size
    worst = int(np.argmax(ctrl))
    g = 0.03

    # The pre-registered threshold was  X* = g * n * mean / worst  = 41.80%.
    # It is WRONG: it drops a -g term. Solving mean(ctrl) - mean(cand) = 0 for the
    # fixture cand = ctrl*(1-g) with cand[worst] = ctrl[worst]*(1+x) gives
    #     g*mean - (worst/n)*(x + g) = 0   =>   X_flip = g * (n*mean/worst - 1)
    # which is 38.80%. The independent Skeptic caught this on 2026-08-19; the
    # contract's +-15% tolerance was wide enough to pass a wrong formula, and the
    # [25%, 60%] band was wide enough that it tested nothing (it is equivalent to
    # predicting worst/mean in [2.5, 6.0], and 3.59 was already in the JSON).
    # P3 is therefore recorded as an ARITHMETIC IDENTITY, not a confirmed
    # prediction. What survives is the MECHANISM and the fixture.
    x_star_contract = g * n * float(ctrl.mean()) / float(ctrl[worst])
    x_flip_exact = g * (n * float(ctrl.mean()) / float(ctrl[worst]) - 1.0)
    check(
        "P3 as pre-registered was ALGEBRAICALLY WRONG (dropped the -g term)",
        abs(x_star_contract - x_flip_exact - g) < 1e-12,
        f"contract {x_star_contract * 100:.2f}%  exact {x_flip_exact * 100:.2f}%  "
        f"error {(x_star_contract - x_flip_exact) * 100:.2f}pp = g",
    )

    # Scan X and find the actual flip.
    flip_at = None
    for x in np.arange(0.01, 2.0, 0.005):
        cand = ctrl * (1.0 - g)
        cand[worst] = ctrl[worst] * (1.0 + x)
        r = paired_compare(ctrl, cand, boot_resamples=2000)
        if r.estimands_disagree:
            flip_at = float(x)
            break
    check("a sign flip exists", flip_at is not None,
          f"first grid point at X = {flip_at:.2%}" if flip_at is not None else "none found")
    if flip_at is not None:
        check(
            "the flip lands on the first grid point above the exact law (0.5% grid)",
            0.0 <= flip_at - x_flip_exact < 0.005,
            f"observed {flip_at:.2%}  exact {x_flip_exact:.2%}  "
            f"gap {(flip_at - x_flip_exact) * 100:.3f}pp",
        )

    # The canonical fixture, frozen at a value comfortably inside the flip region.
    x = (flip_at if flip_at is not None else x_flip_exact) + 0.10
    cand = ctrl * (1.0 - g)
    cand[worst] = ctrl[worst] * (1.0 + x)
    res = paired_compare(ctrl, cand)
    check(
        "FIXTURE: geometric says candidate WINS, arithmetic says it LOSES",
        res.geometric_says_candidate_better and not res.arithmetic_says_candidate_better,
        f"geo {res.geometric_gain:.4f}x  vs  arith {res.arithmetic_gain:.4f}x",
    )
    print("\n" + format_paired(res, control="v2.4.1 graded", candidate="tail-regressing candidate"))
    print(
        f"\n    mechanism: the candidate improves {n - 1}/{n} MLPs by {g:.0%} and "
        f"regresses one by {x:.0%}.\n"
        f"    The geometric mean weights every MLP's LOG equally, so a single tail "
        f"blow-up is\n    invisible to it. The competition sums scores, so that one "
        f"MLP is {ctrl[worst] / ctrl.sum() * 100:.1f}% of the objective."
    )


def t_paired_determinism_and_sanity():
    head("8. paired statistics — determinism and analytic sanity")
    a = np.array([1.0, 2.0, 4.0, 8.0])
    b = a / 2.0
    r1 = paired_compare(a, b, boot_resamples=3000)
    r2 = paired_compare(a, b, boot_resamples=3000)
    check(
        "paired_compare is deterministic",
        (r1.boot_ci_difference == r2.boot_ci_difference)
        and (r1.mean_difference == r2.mean_difference),
    )
    check(
        "uniform 2x improvement: arithmetic gain == geometric gain == 2",
        abs(r1.arithmetic_gain - 2.0) < 1e-12 and abs(r1.geometric_gain - 2.0) < 1e-12,
        f"arith {r1.arithmetic_gain:.6f}  geo {r1.geometric_gain:.6f}",
    )
    check("uniform improvement: estimands agree", not r1.estimands_disagree)
    check(
        "identical arms: zero difference, gain 1.0, p = 1",
        paired_compare(a, a.copy(), boot_resamples=500).mean_difference == 0.0
        and abs(paired_compare(a, a.copy(), boot_resamples=500).geometric_gain - 1.0) < 1e-15,
    )
    check(
        "mismatched lengths are refused",
        _raises(lambda: paired_compare(a, b[:2])),
    )


def _raises(fn):
    try:
        fn()
        return False
    except Exception:
        return True


# --------------------------------------------------------------------------- #
# 6. Graded replay — predictions P4, P5, P7. Read-only, no network.
# --------------------------------------------------------------------------- #


def t_graded_replay():
    head("9. GRADED REPLAY — real published rows, read-only (P4, P5, P7)")
    for label, path in (("v2.4  / 324420", V24), ("v2.4.1 / 327743", V241)):
        with open(path, encoding="utf-8") as fh:
            rep = json.load(fh)
        rows = rows_from_graded_report(rep)
        reported = reported_scores_by_index(rep)
        s = score_suite(rows, **P1)

        # P4 — reproduce the grader's own per-MLP adjusted score.
        idx = sorted(reported)
        worst_rel = 0.0
        for j, i in enumerate(idx):
            ours, theirs = s.scores[j], reported[i]
            worst_rel = max(worst_rel, abs(ours - theirs) / abs(theirs))
        check(
            f"P4 {label}: 50 rows reproduce to 1e-12",
            worst_rel < 1e-12,
            f"worst rel {worst_rel:.3e}",
        )

        # P5 — confirm lambda = 1e11 from the grader's OWN telemetry.
        ev = rep["evaluation"]
        wr = 0.0
        for e in ev["results"]["per_mlp"]:
            t = e["telemetry"]
            c = effective_compute(t["flops_used"], t["residual_wall_time_s"],
                                  **P1)
            wr = max(wr, abs(c - float(t["effective_compute"])) / abs(float(t["effective_compute"])))
        check(
            f"P5 {label}: C == F + 1e11*R on all 100 rows to 1e-9",
            wr < 1e-9,
            f"worst rel {wr:.3e}",
        )

        # P7 — is aggregates.public over 50 rows or 100? PREDICTED 100; it is 50.
        agg = ev["results"]["aggregates"]["public"]
        tel = [e["telemetry"] for e in ev["results"]["per_mlp"]]
        c_all = float(np.mean([float(t["effective_compute"]) for t in tel]))
        c_50 = float(np.mean([float(t["effective_compute"]) for t in tel[:50]]))
        rep_c = float(agg["mean_effective_compute"])
        check(
            f"P7 {label}: aggregates.public is over the 50 PUBLIC rows, not 100",
            abs(c_50 - rep_c) < abs(c_all - rep_c),
            f"reported {rep_c:.6e} | 50-row {c_50:.6e} | 100-row {c_all:.6e}",
        )
        # ...and therefore the whole leaderboard number is reproducible offline.
        reported_obj = float(agg["adjusted_final_layer_score"])
        check(
            f"P7 {label}: our 50-row objective IS the published score, bitwise",
            s.objective == reported_obj,
            f"{s.objective:.12e} vs {reported_obj:.12e}",
        )
        print(
            f"      worst MLP {s.worst:.4e}  best {s.best:.4e}  "
            f"worst/mean {s.worst / s.objective:.2f}x  "
            f"top-1 tail share {s.tail_share(1) * 100:.1f}%  "
            f"top-5 {s.tail_share(5) * 100:.1f}%"
        )


def t_real_pair_v24_vs_v241():
    head("10. the real v2.4 -> v2.4.1 pair, scored by the canonical scorer")
    with open(V24, encoding="utf-8") as fh:
        r24 = json.load(fh)
    with open(V241, encoding="utf-8") as fh:
        r241 = json.load(fh)
    a = rows_from_graded_report(r24)
    b = rows_from_graded_report(r241)
    na = [x.name for x in a]
    nb = [x.name for x in b]
    check(
        "the two submissions were scored on the SAME 50 MLPs in the same order",
        na == nb and all(na),
        f"{len(na)} names",
    )
    sa, sb = score_suite(a), score_suite(b)
    res = paired_compare(sa.scores, sb.scores)
    print("\n" + format_paired(res, control="v2.4 (324420)", candidate="v2.4.1 (327743)"))
    check(
        "the real graded pair reproduces the documented estimand disagreement",
        res.estimands_disagree,
        f"arith {res.arithmetic_gain:.4f}x  geo {res.geometric_gain:.4f}x",
    )
    check(
        "arithmetic improvement is NOT resolvable (bootstrap CI spans 0)",
        res.boot_ci_difference[0] < 0.0 < res.boot_ci_difference[1],
        f"[{res.boot_ci_difference[0]:.3e}, {res.boot_ci_difference[1]:.3e}]",
    )


def t_environment():
    head("0. environment")
    from importlib.metadata import version as _ver

    print(f"  whestbench {_ver('whestbench')}   flopscope {_ver('flopscope')}   "
          f"numpy {np.__version__}")
    check("floor constant matches the installed evaluator", MULTIPLIER_FLOOR == 0.1)
    check("graded reports present", os.path.exists(V24) and os.path.exists(V241))


def t_the_binding_gate():
    """CP0.3 — `decide()` IS the gate, and it cannot be talked out of a verdict."""
    head("13. the binding gate — arithmetic primary, t/basic interval only")
    rng = np.random.default_rng(20260820)

    # An arm that is genuinely better by ~11%, with the heavy right tail the real
    # metric has, so the two estimands are both well defined.
    ctrl = rng.lognormal(mean=-15.0, sigma=0.7, size=50)
    cand = ctrl * rng.uniform(0.86, 0.92, size=50)

    v = decide(ctrl, cand, gate=1.05)
    check("a real 11% improvement PASSES a 1.05x gate", v.passed,
          f"gain {v.gain:.4f}x, t CI [{v.ci[0]:+.3e}, {v.ci[1]:+.3e}]")
    check("...and FAILS a 1.25x gate, on the same data",
          not decide(ctrl, cand, gate=1.25).passed)
    check("the gate has NO default — it must be pre-registered",
          _raises(lambda: decide(ctrl, cand)), "TypeError without gate=")

    check("the percentile bootstrap is REFUSED, not deprecated",
          _raises(lambda: decide(ctrl, cand, gate=1.05, interval="percentile"),
                  ValueError))
    check("BCa is REFUSED (measured 9.1-11.5% false promotion at n~50)",
          _raises(lambda: decide(ctrl, cand, gate=1.05, interval="bca"),
                  ValueError))
    check("the basic bootstrap IS permitted",
          decide(ctrl, cand, gate=1.05, interval="basic").interval == "basic")

    # A true null: identical arms plus symmetric noise must not promote.
    null_c = rng.lognormal(mean=-15.0, sigma=0.7, size=50)
    null_k = null_c * rng.normal(1.0, 0.02, size=50)
    vn = decide(null_c, null_k, gate=1.05)
    check("a true null does not clear a 1.05x gate", not vn.passed,
          f"gain {vn.gain:.4f}x; reasons: {len(vn.reasons)}")

    # The verdict is arithmetic even when geometric disagrees.
    a, b = _sign_flip_arms()
    vd = decide(a, b, gate=1.0)
    check("when the estimands DISAGREE the verdict follows ARITHMETIC",
          vd.gain > 1.0 and vd.geometric_gain < 1.0 and vd.estimands_disagree,
          f"arith {vd.gain:.4f}x, geo {vd.geometric_gain:.4f}x")
    check("...and the disagreement is surfaced in the reasons, not hidden",
          any("estimands disagree" in r for r in vd.reasons))
    # NOT `vd.gain >= 1.0 and vd.ci_excludes_zero` -- that recomputes the
    # implementation's own expression from the Verdict's own fields and cannot
    # fail for ANY implementation. The Skeptic called it a tautology and was
    # right. Assert against a hand-computed expectation instead.
    # The GAIN follows arithmetic; the VERDICT still needs the interval. On these
    # arms the whole gain sits on one network out of 50, so `decide()` refuses
    # despite a 1.47x arithmetic gain -- which is correct, and is the behaviour
    # that makes adopting a more outlier-sensitive estimand safe. Asserting
    # `passed is True` here was the builder's error, not the gate's.
    vd10 = decide(a, b, gate=1.0)
    check("the GAIN follows arithmetic even when geometric disagrees",
          vd10.gain > 1.4 and vd10.geometric_gain < 1.0,
          f"arith {vd10.gain:.4f}x vs geo {vd10.geometric_gain:.4f}x")
    check("but a gain resting on ONE network is still refused by the interval",
          not vd10.passed and not vd10.ci_excludes_zero
          and abs(vd10.biggest_mover_share) >= 1.0,
          f"biggest mover carries {vd10.biggest_mover_share * 100:.0f}% of the delta")

    # --- a strictly WORSE candidate must never pass, at ANY legal gate ------
    # This leaked: `excl` was two-sided, so a SIGNIFICANT REGRESSION counted as
    # evidence FOR promotion, and with gate < 1.0 nothing else caught it.
    worse = ctrl / 0.77                       # 30% worse on every single network
    check("a 30%-worse candidate FAILS at every legal gate",
          not any(decide(ctrl, worse, gate=gt).passed
                  for gt in (1.0, 1.0000001, 1.05, 2.0)),
          "one-sided interval + the gate floor, two independent locks")
    check("...and the Verdict SAYS it is worse, rather than staying silent",
          any("WORSE" in r for r in decide(ctrl, worse, gate=1.0).reasons))
    check("a gate below 1.0 is REFUSED (it would ask to promote a regression)",
          all(_raises(lambda g=g: decide(ctrl, worse, gate=g), ValueError)
              for g in (0.0, 0.5, -1.0)))
    check("a non-finite gate is REFUSED",
          all(_raises(lambda g=g: decide(ctrl, cand, gate=g), ValueError)
              for g in (float("nan"), float("inf"), float("-inf"))))
    check("the interval condition is ONE-SIDED (strictly above zero)",
          not decide(ctrl, worse, gate=1.0).ci_excludes_zero,
          "a regression's interval excludes zero two-sidedly; that is not a pass")

    check("Verdict is frozen — a verdict you can edit is not a verdict",
          _raises(lambda: setattr(vd, "passed", True)))


def _raises(fn, exc=Exception) -> bool:
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


def _sign_flip_arms():
    """Arms on which arithmetic and geometric pick opposite winners.

    The real graded pair does this (1.0334x vs 0.9666x); this is a synthetic
    version so the gate is tested even if the reports move.
    """
    ctrl = np.array([1.0] * 49 + [100.0])
    cand = np.array([1.05] * 49 + [50.0])   # worse on 49, far better on the heavy one
    return ctrl, cand



# --------------------------------------------------------------------------- #
# 14. Identity-aligned pairing — the 2026-09-10 Stage 0 repair
#
# These tests FAIL against the code as it stood at commit 0fe164c. That is the
# point of them: a regression test that passes on the defect tests nothing.
# --------------------------------------------------------------------------- #


def _named_arms(n=24, seed=11):
    rs = np.random.default_rng(seed)
    B = 2.0 ** 41
    mse_c = rs.uniform(3e-8, 6e-8, n)
    mse_k = mse_c * rs.uniform(0.85, 1.05, n)
    fl = rs.uniform(0.2, 0.6, n) * B
    ctrl = [MlpRow(mse=float(mse_c[i]), flops_used=float(fl[i]), budget=B,
                   name="mlp%02d" % i) for i in range(n)]
    cand = [MlpRow(mse=float(mse_k[i]), flops_used=float(fl[i]), budget=B,
                   name="mlp%02d" % i) for i in range(n)]
    return ctrl, cand, rs


def t_pairing_identity():
    head("14. pairing is on MLP IDENTITY, not list order")
    ctrl, cand, rs = _named_arms()
    ref = paired_compare_rows(ctrl, cand)

    # (a) A permuted candidate arm is REALIGNED, not silently mispaired.
    #     Before the repair the SE inflated by a median 8.55x while
    #     mean_difference stayed bit-identical, so nothing printed revealed it.
    ses, excl = [], 0
    for _ in range(40):
        perm = rs.permutation(len(cand))
        r = paired_compare_rows(ctrl, [cand[i] for i in perm])
        ses.append(r.se_difference)
        lo, hi = r.t_ci_difference
        excl += int(lo > 0 or hi < 0)
    check("a permuted candidate arm recovers the SAME standard error",
          max(abs(x - ref.se_difference) for x in ses) < 1e-24,
          f"se {ref.se_difference:.4e}, 40 permutations, spread "
          f"{max(ses) - min(ses):.2e}")
    check("...and the t interval still excludes zero on every permutation",
          excl == 40, f"{excl}/40 (before the repair: 0/200)")
    check("...and the result SAYS it reordered, rather than staying silent",
          any("REORDERED" in n for n in
              paired_compare_rows(ctrl, list(reversed(cand))).notes),
          "notes carry the reorder")

    # (b) duplicates move the POINT ESTIMATE, so they are refused outright
    dup = [cand[0]] * len(cand)
    try:
        paired_compare_rows(ctrl, dup)
        ok = False
    except PairingError as e:
        ok = "repeats MLP identities" in str(e)
    check("a candidate arm of duplicated identities is REFUSED", ok,
          "a duplicate is a false-promotion route, not a paired design")

    # (c) different networks are refused
    other = [MlpRow(mse=r.mse, flops_used=r.flops_used, budget=r.budget,
                    name="other%02d" % i) for i, r in enumerate(cand)]
    try:
        paired_compare_rows(ctrl, other)
        ok = False
    except PairingError as e:
        ok = "not the same MLPs" in str(e)
    check("arms naming DIFFERENT MLPs are refused", ok, "no common random numbers")

    # (d) half-named is ambiguous
    half = [MlpRow(mse=r.mse, flops_used=r.flops_used, budget=r.budget,
                   name=(r.name if i % 2 else "")) for i, r in enumerate(cand)]
    try:
        paired_compare_rows(ctrl, half)
        ok = False
    except PairingError:
        ok = True
    check("a PARTLY named arm is refused rather than guessed at", ok, "")

    # (e) the historical unnamed path survives, but says so
    anon_c = [MlpRow(mse=r.mse, flops_used=r.flops_used, budget=r.budget) for r in ctrl]
    anon_k = [MlpRow(mse=r.mse, flops_used=r.flops_used, budget=r.budget) for r in cand]
    r = paired_compare_rows(anon_c, anon_k)
    check("two fully UNNAMED arms still pair positionally", r.n == len(ctrl),
          "existing callers keep working")
    check("...and the result records that it paired on LIST ORDER",
          any("LIST ORDER" in n for n in r.notes), "the allowance is visible")
    try:
        paired_compare_rows(anon_c, anon_k, require_ids=True)
        ok = False
    except PairingError:
        ok = True
    check("...but require_ids=True refuses them", ok,
          "a DECIDE-tier contract may not pair on list order")

    # (f) alignment is a pure permutation: same rows, same scores
    a, b, _ = align_rows_by_identity(ctrl, list(reversed(cand)))
    check("alignment returns the SAME rows, only reordered",
          [x.name for x in a] == [x.name for x in b]
          and sorted(x.mse for x in b) == sorted(x.mse for x in cand),
          "no row invented, none dropped")


def t_t_quantile_alpha():
    head("15. the t interval honours the alpha it was asked for")
    d = np.random.default_rng(7).normal(1e-8, 3e-8, size=40)
    c95 = paired_diff_ci(d, method="t", alpha=0.05)
    c99 = paired_diff_ci(d, method="t", alpha=0.01)
    c80 = paired_diff_ci(d, method="t", alpha=0.20)
    check("a 99% t interval is WIDER than the 95%", (c99[1] - c99[0]) > (c95[1] - c95[0]) * 1.2,
          f"{(c99[1] - c99[0]) / (c95[1] - c95[0]):.3f}x   (before the repair: 1.000x)")
    check("an 80% t interval is NARROWER than the 95%", (c80[1] - c80[0]) < (c95[1] - c95[0]) * 0.85,
          f"{(c80[1] - c80[0]) / (c95[1] - c95[0]):.3f}x   (before the repair: 1.000x)")
    check("the three intervals are no longer bit-identical",
          len({c95, c99, c80}) == 3, "alpha 0.05 / 0.01 / 0.20")
    # the exact quantile must agree with the historical table to the table's
    # own precision, or the 0.05 seam is drifting
    worst = max(abs(_t_ppf(0.975, df) - _t975(df)) for df in list(range(1, 31)))
    check("the exact quantile matches the historical _T975 table to 1e-3",
          worst < 1.1e-3, f"worst |exact - table| = {worst:.2e} over df 1..30")
    check("the DEFAULT 0.05 path still reads the table, bit for bit",
          paired_diff_ci(d, method="t")[0] == c95[0],
          "no published 95% interval moves")
    try:
        import scipy.stats as _st
        err = max(abs(_t_ppf(p, df) - float(_st.t.ppf(p, df)))
                  for df in (1, 5, 17, 48, 199) for p in (0.9, 0.975, 0.995))
        check("...and it matches scipy.stats.t.ppf", err < 1e-9, f"max err {err:.2e}")
    except ImportError:
        check("scipy not importable; table comparison stands alone", True, "skipped")



# --------------------------------------------------------------------------- #
# 16. Label noise does not cancel from an ADJUSTED difference
#
# Validation audit finding 6. These fail against the code at 0fe164c because
# the function did not exist and MlpRow had nowhere to put a noise floor.
# --------------------------------------------------------------------------- #


def _noise_arms(n=60, seed=5, q_ctrl=0.60, q_cand=0.25,
                true_c=4.0e-8, true_k=4.0e-8, v=1.1552e-9, meas_sd=0.0):
    """Two arms on the same networks with KNOWN true errors and a known floor."""
    B = 2.0 ** 41
    rs = np.random.default_rng(seed)
    rows_c, rows_k = [], []
    for i in range(n):
        vi = v * float(rs.uniform(0.9, 1.1))
        name = "seed:%d" % (900000 + i)
        ec = float(rs.normal(0.0, meas_sd)) if meas_sd else 0.0
        ek = float(rs.normal(0.0, meas_sd)) if meas_sd else 0.0
        rows_c.append(MlpRow(mse=true_c + vi + ec, flops_used=q_ctrl * B,
                             budget=B, name=name, gt_noise=vi))
        rows_k.append(MlpRow(mse=true_k + vi + ek, flops_used=q_cand * B,
                             budget=B, name=name, gt_noise=vi))
    return rows_c, rows_k


def t_noise_does_not_cancel():
    head("16. label noise, and the multiplier that stops it cancelling")

    # Two arms with IDENTICAL true error and DIFFERENT cost. On the true
    # functions the arms differ only through cost; the observed difference
    # additionally carries (q_c - q_k) * v.
    rows_c, rows_k = _noise_arms()
    r = noise_corrected_compare(rows_c, rows_k, local_samples=2 ** 24)
    q_c, q_k = 0.60, 0.25
    v_mean = float(np.mean([x.gt_noise for x in rows_c]))
    expected_bias = (q_c - q_k) * v_mean
    check("the noise bias matches the closed form (q_c - q_k) * v",
          abs(r.noise_bias - expected_bias) / expected_bias < 1e-9,
          f"{r.noise_bias:.6e} vs {expected_bias:.6e}")
    check("...and it is exactly the gap between observed and true-function",
          abs((r.observed.mean_difference - r.true_function.mean_difference)
              - r.noise_bias) < 1e-18,
          f"{r.observed.mean_difference - r.true_function.mean_difference:.6e}")
    check("the observed difference OVERSTATES the true-function difference",
          r.observed.mean_difference > r.true_function.mean_difference,
          f"{r.observed.mean_difference:.4e} vs "
          f"{r.true_function.mean_difference:.4e}")
    check("...and the report says what share of it is noise",
          0.02 < r.noise_bias_share < 0.04,
          f"{r.noise_bias_share * 100:.2f}% of the observed difference")

    # Same cost on both arms: the term is exactly zero, and the correction is a
    # no-op. This is the case where the old "it cancels" wording was right, and
    # saying so is the difference between a scope correction and a scare.
    same_c, same_k = _noise_arms(q_ctrl=0.5, q_cand=0.5)
    rs2 = noise_corrected_compare(same_c, same_k, local_samples=2 ** 24)
    check("at EQUAL multipliers the noise bias is exactly zero",
          rs2.noise_bias == 0.0,
          "the old cancellation claim is right here, and only here")
    check("...and observed equals true-function to the last bit",
          rs2.observed.mean_difference == rs2.true_function.mean_difference, "")


def t_noise_forecast_and_refusals():
    head("17. the official-precision forecast, and what it refuses")
    rows_c, rows_k = _noise_arms(true_k=3.0e-8)
    r = noise_corrected_compare(rows_c, rows_k, local_samples=2 ** 24,
                                official_samples=1e9)
    # 2**24 / 1e9 is about 1/60, so the official floor is ~60x smaller and the
    # forecast sits between the corrected and the observed figure.
    lo = min(r.true_function.mean_difference, r.observed.mean_difference)
    hi = max(r.true_function.mean_difference, r.observed.mean_difference)
    check("the official-precision forecast lies between corrected and observed",
          lo <= r.forecast.mean_difference <= hi,
          f"{r.forecast.mean_difference:.6e} in [{lo:.4e}, {hi:.4e}]")
    check("...and much nearer the corrected figure, since 2**24 << 1e9",
          abs(r.forecast.mean_difference - r.true_function.mean_difference)
          < 0.05 * abs(r.observed.mean_difference - r.true_function.mean_difference),
          "the official labels are ~60x more precise than ours")

    # A missing floor is not zero.
    bare = [MlpRow(mse=x.mse, flops_used=x.flops_used, budget=x.budget,
                   name=x.name) for x in rows_c]
    try:
        noise_corrected_compare(bare, rows_k, local_samples=2 ** 24)
        ok = False
    except ValueError as e:
        ok = "gt_noise on EVERY row" in str(e)
    check("a row with no gt_noise is REFUSED, not treated as zero", ok,
          "an unknown floor is not a floor of zero")

    # The same networks have ONE floor.
    bad = [MlpRow(mse=x.mse, flops_used=x.flops_used, budget=x.budget,
                  name=x.name, gt_noise=x.gt_noise * 2) for x in rows_k]
    try:
        noise_corrected_compare(rows_c, bad, local_samples=2 ** 24)
        ok = False
    except PairingError as e:
        ok = "disagree about the LABEL NOISE" in str(e)
    check("two arms disagreeing about one network's floor are REFUSED", ok, "")

    try:
        noise_corrected_compare(rows_c, rows_k, local_samples=0)
        ok = False
    except ValueError:
        ok = True
    check("a non-positive sample count is refused rather than guessed at", ok, "")


def t_negative_corrections_are_kept():
    head("18. a negative corrected MSE is retained, never clipped")
    # An estimator sitting essentially ON the noise floor: half the corrected
    # values must come out below zero by construction.
    rows_c, rows_k = _noise_arms(n=80, true_c=0.0, true_k=0.0, v=2.0e-9,
                                 meas_sd=4.0e-10, seed=17)
    r = noise_corrected_compare(rows_c, rows_k, local_samples=2 ** 24)
    check("negative corrected MSEs occur and are COUNTED",
          r.n_negative_corrected > 0, f"{r.n_negative_corrected} of 160 rows")
    check("...and the result says so in its notes",
          any("NEGATIVE" in n for n in r.notes), "")
    check("...and no network was dropped", r.n == 80, f"n = {r.n}")
    # The mean of the corrected values must stay near zero. Clipping would push
    # it up; this asserts it did not.
    tf = r.true_function
    check("the corrected true-function means stay near zero",
          abs(tf.mean_control) < 1e-10 and abs(tf.mean_candidate) < 1e-10,
          f"ctrl {tf.mean_control:+.3e}, cand {tf.mean_candidate:+.3e}")


def main():
    print("=" * 100)
    print("CANONICAL COMPETITION SCORER — parity with installed whestbench, and the estimand fixture")
    print("=" * 100)
    for fn in (
        t_environment,
        t_scalar_parity,
        t_aggregation_parity,
        t_evaluator_parity,
        t_heterogeneous_budgets,
        t_over_budget_guard,
        t_failure_cost,
        t_correlation_preserved,
        t_sign_flip_fixture,
        t_paired_determinism_and_sanity,
        t_graded_replay,
        t_real_pair_v24_vs_v241,
        t_the_binding_gate,
        t_pairing_identity,
        t_t_quantile_alpha,
        t_noise_does_not_cancel,
        t_noise_forecast_and_refusals,
        t_negative_corrections_are_kept,
    ):
        fn()
    n_ok = sum(1 for r in RESULTS if r)
    print(f"\n{n_ok}/{len(RESULTS)} checks passed")
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
