"""
The canonical competition scorer — one implementation of the grader's arithmetic.

Stage 0 / P0.1 of `docs/FRONTIER_ARCHITECTURE_PROPOSAL.md`. Contract and written
predictions: `experiments/LOG.md`, 2026-08-19 (night).

WHY THIS FILE EXISTS
--------------------
Twelve probes in `local_sim/` carry an inline `mse * max(0.1, C/B)`. Every one of
them is a *partial* copy of the grader's identity: none reproduces the failure
multiplier, none takes a per-MLP budget, several multiply a *mean* MSE by a *mean*
multiplier (which is biased whenever MSE and cost are correlated, and they are),
and the promotion statistic layered on top is the geometric mean of per-MLP ratios
while the competition is decided by an *arithmetic* mean of per-MLP scores. Those
two estimands can and do select opposite winners on a heavy-tailed suite.

This module is the single place that arithmetic lives. It is a reimplementation,
not a wrapper: it imports nothing from `whestbench`, so it runs in the torch-only
GPU venv, exactly as `mint_provenance.py` reimplements the mint protocol. Its
claim to authority is `local_sim/test_competition_score.py`, which checks it
against the installed `whestbench` for *bitwise* equality on randomised rows and
replays the real graded reports through it.

THE IDENTITY (whestbench 0.14.0, `budget.py` + `scoring.py`)
------------------------------------------------------------
    C_i       = F_i + lambda * R_i
    mult_i    = 1.0                    if failed_i
              = 1.0                    if B_i <= 0
              = max(floor, C_i / B_i)  otherwise      # floored, UNCAPPED above
    s_i       = mse_i * mult_i
    objective = arithmetic_mean(s_i)                   # lower is better

    failed_i  = error_code or budget_exhausted or time_exhausted
                or residual_wall_time_exhausted or combined_budget_exhausted

Note the shape of the failure penalty: a failed MLP is scored at multiplier 1.0,
not at the 0.1 floor, *and* against zeroed predictions. At our operating point
(mult ~ 0.199) that is roughly a 5x multiplier penalty on top of a much larger MSE.

PHASE-2 SAFETY
--------------
As of 2026-08-19 round 1429 is UPCOMING and its configuration is unpublished. No
Phase-1 constant is baked in here: `budget`, `lam` and `floor` are per-call
parameters, `budget` may vary per MLP (whestbench 0.14.0 and 0.15.0 both use one
`spec.flop_budget` for every MLP, but nothing in this module assumes it), and
`LAMBDA_FLOPS_PER_SECOND` / `MULTIPLIER_FLOOR` below are labelled as the *installed
0.14.0* values rather than as competition constants.

Run the self-check:  uv run python local_sim/test_competition_score.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Constants — the values the INSTALLED whestbench uses, not competition axioms.
# Verified byte-identical in whestbench 0.14.0 (installed) and 0.15.0 (the
# newest release, 2026-08-17, inspected read-only without installing it).
# --------------------------------------------------------------------------- #

LAMBDA_FLOPS_PER_SECOND: float = 1e11
MULTIPLIER_FLOOR: float = 0.1
FAILURE_MULTIPLIER: float = 1.0

#: Phase-1 budget. Present for replaying Phase-1 artifacts ONLY. Never a default.
PHASE1_FLOP_BUDGET: float = 272_000_000_000.0

_FAILURE_FLAGS = (
    "budget_exhausted",
    "time_exhausted",
    "residual_wall_time_exhausted",
    "combined_budget_exhausted",
)


# --------------------------------------------------------------------------- #
# Scalar primitives — deliberately line-for-line with whestbench/budget.py so a
# reader can diff them, and so float operation order is identical.
# --------------------------------------------------------------------------- #


def effective_compute(
    flops_used: float,
    residual_wall_time_s: float,
    lam: float = LAMBDA_FLOPS_PER_SECOND,
) -> float:
    """C_i = F_i + lambda * R_i."""
    return float(flops_used) + float(lam) * float(residual_wall_time_s)


def is_combined_budget_exhausted(
    flops_used: float,
    residual_wall_time_s: float,
    flop_budget: float,
    lam: float = LAMBDA_FLOPS_PER_SECOND,
) -> bool:
    """True when C_i strictly exceeds B_i. Strict `>`: C_i == B_i is in budget."""
    if flop_budget <= 0:
        return False
    return effective_compute(flops_used, residual_wall_time_s, lam) > float(flop_budget)


def score_multiplier(
    effective_compute_value: float,
    flop_budget: float,
    *,
    failed: bool,
    floor: float = MULTIPLIER_FLOOR,
) -> float:
    """Per-MLP multiplier: 1.0 on failure or no budget, else max(floor, C/B).

    Uncapped above. `floor` is a parameter because the Phase-2 floor is not yet
    published; the installed evaluator hardcodes 0.1.
    """
    if failed or flop_budget <= 0:
        return FAILURE_MULTIPLIER
    return max(float(floor), float(effective_compute_value) / float(flop_budget))


def mlp_score(
    mse: float,
    flops_used: float,
    residual_wall_time_s: float,
    flop_budget: float,
    *,
    failed: bool = False,
    lam: float = LAMBDA_FLOPS_PER_SECOND,
    floor: float = MULTIPLIER_FLOOR,
) -> float:
    """s_i = mse_i * mult_i — the per-MLP competition score. Lower is better."""
    c = effective_compute(flops_used, residual_wall_time_s, lam)
    return float(mse) * score_multiplier(c, flop_budget, failed=failed, floor=floor)


def zero_prediction_mse(final_layer_target: Sequence[float] | np.ndarray) -> float:
    """MSE the grader assigns a failed MLP: mean(target**2), predictions zeroed.

    Matches `scoring.evaluate_estimator`'s failure path, which computes the MSE of
    a zeros array against the same target rather than skipping the MLP.
    """
    t = np.asarray(final_layer_target, dtype=np.float32)
    return float(np.mean(t.astype(np.float32) ** 2))


# --------------------------------------------------------------------------- #
# Per-MLP row and suite scoring
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MlpRow:
    """One MLP's contribution to the objective.

    `budget` is per-row on purpose: the installed evaluator uses a single
    `spec.flop_budget`, but Phase 2 is unpublished and a heterogeneous budget must
    not require a rewrite. `failed` is the grader's *disjunction* — see
    `failed_from_report_entry`.
    """

    mse: float
    flops_used: float = 0.0
    residual_wall_time_s: float = 0.0
    budget: float = 0.0
    failed: bool = False
    name: str = ""

    def effective_compute(self, lam: float = LAMBDA_FLOPS_PER_SECOND) -> float:
        return effective_compute(self.flops_used, self.residual_wall_time_s, lam)

    def multiplier(
        self, lam: float = LAMBDA_FLOPS_PER_SECOND, floor: float = MULTIPLIER_FLOOR
    ) -> float:
        return score_multiplier(
            self.effective_compute(lam), self.budget, failed=self.failed, floor=floor
        )

    def score(
        self, lam: float = LAMBDA_FLOPS_PER_SECOND, floor: float = MULTIPLIER_FLOOR
    ) -> float:
        return float(self.mse) * self.multiplier(lam, floor)


@dataclass(frozen=True)
class SuiteScore:
    """The competition objective for one suite, plus everything needed to audit it."""

    objective: float
    scores: np.ndarray
    multipliers: np.ndarray
    effective_computes: np.ndarray
    mses: np.ndarray
    n_failed: int
    n: int

    # Diagnostics that the aggregate alone hides.
    @property
    def worst(self) -> float:
        return float(self.scores.max()) if self.n else float("inf")

    @property
    def best(self) -> float:
        return float(self.scores.min()) if self.n else float("inf")

    @property
    def mean_multiplier(self) -> float:
        return float(self.multipliers.mean()) if self.n else 1.0

    @property
    def mean_mse(self) -> float:
        return float(self.mses.mean()) if self.n else float("inf")

    def tail_share(self, k: int = 1) -> float:
        """Fraction of the objective carried by the k worst MLPs."""
        if not self.n:
            return float("nan")
        s = np.sort(self.scores)[::-1]
        return float(s[:k].sum() / s.sum())


def over_budget_rows(
    rows: Sequence[MlpRow], lam: float = LAMBDA_FLOPS_PER_SECOND
) -> List[int]:
    """Indices of rows the grader would have FAILED on the combined-budget check.

    `C_i > B_i` triggers `CombinedBudgetExhaustionWarning`, zeroes the predictions
    and forces the multiplier to 1.0. A hand-built row that trips this and still
    carries `failed=False` is scored ~5x too cheaply AND with an MSE the grader
    would have thrown away, so it must never be aggregated silently.
    """
    return [
        i
        for i, r in enumerate(rows)
        if not r.failed
        and is_combined_budget_exhausted(r.flops_used, r.residual_wall_time_s, r.budget, lam)
    ]


def score_suite(
    rows: Iterable[MlpRow],
    *,
    lam: float = LAMBDA_FLOPS_PER_SECOND,
    floor: float = MULTIPLIER_FLOOR,
    on_over_budget: str = "raise",
) -> SuiteScore:
    """Arithmetic mean of per-MLP scores — the competition objective.

    Correlation-preserving by construction: each row's MSE meets its OWN cost
    before anything is averaged. `naive_factorised_objective` below is the biased
    form this replaces.

    `on_over_budget` governs rows with `C_i > B_i` and `failed=False`, which the
    grader would have failed:
      "raise"  (default) refuse to score them;
      "fail"   score them as failures — still optimistic, because this module
               cannot zero a prediction it never saw, so the MSE stays the
               candidate's own;
      "ignore" score them as-is. Only for deliberately exploring the identity.
    """
    rows = list(rows)
    n = len(rows)
    if on_over_budget not in ("raise", "fail", "ignore"):
        raise ValueError(f"on_over_budget must be raise/fail/ignore, got {on_over_budget!r}")
    bad = over_budget_rows(rows, lam) if n else []
    if bad:
        if on_over_budget == "raise":
            raise ValueError(
                f"{len(bad)} row(s) exceed the combined budget C > B but are marked "
                f"failed=False (first at index {bad[0]}: C="
                f"{rows[bad[0]].effective_compute(lam):.6e} > B={rows[bad[0]].budget:.6e}). "
                "The grader would have zeroed these predictions and forced the "
                "multiplier to 1.0. Pass on_over_budget='fail' to score them as "
                "failures (still optimistic: the MSE is not zeroed here), or "
                "'ignore' to keep the raw identity."
            )
        if on_over_budget == "fail":
            bad_set = set(bad)
            rows = [
                MlpRow(r.mse, r.flops_used, r.residual_wall_time_s, r.budget, True, r.name)
                if i in bad_set
                else r
                for i, r in enumerate(rows)
            ]
    if n == 0:
        return SuiteScore(
            objective=float("inf"),
            scores=np.zeros(0),
            multipliers=np.zeros(0),
            effective_computes=np.zeros(0),
            mses=np.zeros(0),
            n_failed=0,
            n=0,
        )
    cs = np.array([r.effective_compute(lam) for r in rows], dtype=np.float64)
    ms = np.array(
        [score_multiplier(c, r.budget, failed=r.failed, floor=floor) for c, r in zip(cs, rows)],
        dtype=np.float64,
    )
    mses = np.array([float(r.mse) for r in rows], dtype=np.float64)
    scores = np.array([float(r.mse) * float(m) for r, m in zip(rows, ms)], dtype=np.float64)
    return SuiteScore(
        objective=float(np.mean(scores)),
        scores=scores,
        multipliers=ms,
        effective_computes=cs,
        mses=mses,
        n_failed=sum(1 for r in rows if r.failed),
        n=n,
    )


def naive_factorised_objective(
    rows: Iterable[MlpRow],
    *,
    lam: float = LAMBDA_FLOPS_PER_SECOND,
    floor: float = MULTIPLIER_FLOOR,
    on_over_budget: str = "raise",
) -> float:
    """mean(mse) * mean(mult) — WRONG, and kept only so the bias is measurable.

    Equals the true objective only when per-MLP MSE and per-MLP multiplier are
    uncorrelated. Several probes in this directory use this shape.
    """
    s = score_suite(rows, lam=lam, floor=floor, on_over_budget=on_over_budget)
    if s.n == 0:
        return float("inf")
    return float(s.mses.mean() * s.multipliers.mean())


# --------------------------------------------------------------------------- #
# Paired inference — the primary decision statistic, and the secondary diagnostic
# --------------------------------------------------------------------------- #


def _binom_two_sided_p(k: int, n: int) -> float:
    """Exact two-sided sign test at p=0.5. stdlib only, so the GPU venv can run it."""
    if n == 0:
        return float("nan")
    k = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0**n)
    return float(min(1.0, 2.0 * tail))


# Two-sided 97.5% Student-t quantiles for df = 1..30; a Cornish-Fisher expansion
# beyond. Avoids a scipy dependency so this module runs unchanged in the torch-only
# GPU venv. MEASURED accuracy of the df > 30 branch against scipy: 3.02e-3 at
# df=31, 1.80e-3 at df=40, 1.20e-3 at df=49 — and it is uniformly BELOW the true
# quantile, so the t interval it produces is systematically a little too narrow.
# Do not use this branch for a coverage claim; use the bootstrap.
_T975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
    27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def _t975(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df in _T975:
        return _T975[df]
    # Cornish-Fisher one-term expansion; approaches 1.959964 from below.
    z = 1.959964
    return z + (z**3 + z) / (4.0 * df)


@dataclass(frozen=True)
class PairedResult:
    """Candidate vs control on the SAME MLPs. Positive improvement = candidate better."""

    n: int
    mean_control: float
    mean_candidate: float

    # PRIMARY — the estimand the competition actually uses.
    mean_difference: float          # mean(s_ctrl - s_cand); > 0 => candidate better
    relative_improvement: float     # mean_difference / mean_control
    arithmetic_gain: float          # mean_ctrl / mean_cand; > 1 => candidate better
    se_difference: float
    t_ci_difference: Tuple[float, float]
    boot_ci_difference: Tuple[float, float]
    bca_ci_difference: Tuple[float, float]
    boot_ci_relative: Tuple[float, float]

    # SECONDARY — stability / mechanism diagnostic only.
    geometric_gain: float           # exp(mean(log(s_ctrl / s_cand))); repo convention
    geometric_ci: Tuple[float, float]

    n_candidate_better: int
    sign_test_p: float
    boot_resamples: int
    boot_seed: int
    notes: List[str] = field(default_factory=list)

    @property
    def arithmetic_says_candidate_better(self) -> bool:
        return self.mean_difference > 0.0

    @property
    def geometric_says_candidate_better(self) -> bool:
        return self.geometric_gain > 1.0

    @property
    def estimands_disagree(self) -> bool:
        """True when the two aggregations pick opposite winners."""
        return self.arithmetic_says_candidate_better != self.geometric_says_candidate_better


def paired_diff_ci(
    d: np.ndarray,
    *,
    method: str = "bca",
    resamples: int = 20_000,
    seed: int = 12345,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """95% CI for mean(d) by one of four methods. `d` is the paired difference.

    Split out of `paired_compare` so a coverage study can drive each method
    identically. Methods, weakest to strongest under skew:

      "t"          Student-t on mean(d). Assumes symmetry; fastest.
      "percentile" bootstrap percentiles of the resampled mean. Biased under skew
                   in the direction OPPOSITE the skew.
      "basic"      the reverse-percentile ("basic") interval, 2*mean - percentiles.
      "bca"        bias-corrected and accelerated. Corrects both median bias and
                   skew via a jackknife acceleration; the standard remedy here.

    Which one this project should USE is an empirical question, not a stylistic
    one — see `local_sim/coverage_lab.py`.
    """
    d = np.asarray(d, dtype=np.float64)
    n = int(d.size)
    theta = float(d.mean())
    if n < 2:
        return (float("nan"), float("nan"))
    if method == "t":
        se = float(d.std(ddof=1) / math.sqrt(n))
        q = _t975(n - 1) if abs(alpha - 0.05) < 1e-12 else _t975(n - 1)
        return (theta - q * se, theta + q * se)

    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, n, size=(resamples, n))].mean(axis=1)
    lo_p, hi_p = 100 * alpha / 2, 100 * (1 - alpha / 2)
    if method == "percentile":
        return (float(np.percentile(boot, lo_p)), float(np.percentile(boot, hi_p)))
    if method == "basic":
        return (
            float(2 * theta - np.percentile(boot, hi_p)),
            float(2 * theta - np.percentile(boot, lo_p)),
        )
    if method != "bca":
        raise ValueError(f"unknown CI method {method!r}")

    # BCa. z0 corrects median bias; the jackknife acceleration corrects skew.
    prop = float(np.mean(boot < theta))
    prop = min(max(prop, 1.0 / (2 * resamples)), 1.0 - 1.0 / (2 * resamples))
    z0 = _norm_ppf(prop)
    total = d.sum()
    jack = (total - d) / (n - 1)          # leave-one-out means
    jbar = jack.mean()
    num = float(((jbar - jack) ** 3).sum())
    den = float(6.0 * (((jbar - jack) ** 2).sum() ** 1.5))
    a_hat = num / den if den > 0 else 0.0
    out = []
    for p in (alpha / 2, 1 - alpha / 2):
        z = _norm_ppf(p)
        denom = 1.0 - a_hat * (z0 + z)
        adj = z0 + (z0 + z) / denom if denom != 0 else z0 + z
        out.append(float(np.percentile(boot, 100.0 * _norm_cdf(adj))))
    lo, hi = out
    return (lo, hi) if lo <= hi else (hi, lo)


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF — Acklam's rational approximation, |err| < 1.2e-9."""
    if p <= 0.0:
        return -np.inf
    if p >= 1.0:
        return np.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    dd = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
          3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((dd[0]*q+dd[1])*q+dd[2])*q+dd[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((dd[0]*q+dd[1])*q+dd[2])*q+dd[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def paired_compare(
    control: Sequence[float] | np.ndarray,
    candidate: Sequence[float] | np.ndarray,
    *,
    boot_resamples: int = 20_000,
    boot_seed: int = 12345,
) -> PairedResult:
    """Paired comparison of two per-MLP score vectors on the same MLPs.

    Primary statistic: the paired ARITHMETIC score difference with a paired
    bootstrap over MLPs. The geometric gain is returned too, because
    `docs/VALIDATION.md` 2b still makes it the binding gate and because a
    disagreement between the two is itself information.

    WHICH INTERVAL TO READ — measured, not assumed. `local_sim/coverage_lab.py`
    ran 192 fresh 256x32 networks (block D18, GT 2^22), four real estimator arms
    under common random numbers, and 4,000 subsamples of n = 50. False-promotion
    rate against a population recentred to a TRUE mean of exactly zero, nominal
    5%:

        t            4.1 - 5.4%     <- at nominal
        basic        4.0 - 5.2%     <- at nominal
        percentile   6.8 - 8.2%     <- inflated ~1.5x
        BCa          9.1 - 11.5%    <- inflated ~2x

    **Read `t_ci_difference` or a basic bootstrap. Do NOT read the percentile
    interval, and do NOT read BCa, at n ~ 50 on this metric.** That is the
    opposite of the textbook ordering, and the reason is measurable: common
    random numbers make the PAIRED DIFFERENCE nearly symmetric (measured skew
    +0.21, -0.55, -0.46) even though each arm's score distribution is strongly
    right-skewed. BCa's jackknife acceleration then spends variance correcting a
    skew that is not there. In the parametric sweep BCa does become the best
    method, but only past skew ~6, which no real arm here reaches.

    `boot_ci_difference` (percentile) is kept because it is what earlier entries
    in `experiments/LOG.md` quote; `bca_ci_difference` is kept because it is the
    right choice at high skew. Neither is the default answer. Full study:
    `experiments/LOG.md` 2026-08-19 (night), and `experiments/2026-08-19_coverage/`.
    """
    a = np.asarray(control, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError(f"control/candidate must be 1-D and equal length; got {a.shape} {b.shape}")
    n = int(a.size)
    if n == 0:
        raise ValueError("empty comparison")

    notes: List[str] = []
    d = a - b
    mean_c, mean_k = float(a.mean()), float(b.mean())
    md = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    tq = _t975(n - 1)
    t_ci = (md - tq * se, md + tq * se) if n > 1 else (float("nan"), float("nan"))

    rng = np.random.default_rng(boot_seed)
    idx = rng.integers(0, n, size=(boot_resamples, n))
    ba, bb = a[idx], b[idx]
    bd = ba.mean(axis=1) - bb.mean(axis=1)
    boot_ci_d = (float(np.quantile(bd, 0.025)), float(np.quantile(bd, 0.975)))
    bca_ci_d = paired_diff_ci(d, method="bca", resamples=boot_resamples, seed=boot_seed)
    with np.errstate(divide="ignore", invalid="ignore"):
        br = bd / ba.mean(axis=1)
    br = br[np.isfinite(br)]
    boot_ci_r = (
        (float(np.quantile(br, 0.025)), float(np.quantile(br, 0.975)))
        if br.size
        else (float("nan"), float("nan"))
    )

    pos = (a > 0) & (b > 0)
    if not np.all(pos):
        notes.append(
            f"{int((~pos).sum())}/{n} rows have a non-positive score; the geometric "
            "statistic is undefined there and was computed on the positive subset only"
        )
    if pos.sum() >= 1:
        lr = np.log(a[pos]) - np.log(b[pos])
        g = float(np.exp(lr.mean()))
        gse = float(lr.std(ddof=1) / math.sqrt(lr.size)) if lr.size > 1 else float("nan")
        g_ci = (float(np.exp(lr.mean() - 1.96 * gse)), float(np.exp(lr.mean() + 1.96 * gse)))
    else:
        g, g_ci = float("nan"), (float("nan"), float("nan"))

    wins = int((b < a).sum())
    return PairedResult(
        n=n,
        mean_control=mean_c,
        mean_candidate=mean_k,
        mean_difference=md,
        relative_improvement=(md / mean_c) if mean_c else float("nan"),
        arithmetic_gain=(mean_c / mean_k) if mean_k else float("inf"),
        se_difference=se,
        t_ci_difference=t_ci,
        boot_ci_difference=boot_ci_d,
        bca_ci_difference=bca_ci_d,
        boot_ci_relative=boot_ci_r,
        geometric_gain=g,
        geometric_ci=g_ci,
        n_candidate_better=wins,
        sign_test_p=_binom_two_sided_p(wins, n),
        boot_resamples=boot_resamples,
        boot_seed=boot_seed,
        notes=notes,
    )


def paired_compare_rows(
    control_rows: Sequence[MlpRow],
    candidate_rows: Sequence[MlpRow],
    *,
    lam: float = LAMBDA_FLOPS_PER_SECOND,
    floor: float = MULTIPLIER_FLOOR,
    on_over_budget: str = "raise",
    **kw: Any,
) -> PairedResult:
    """`paired_compare` over rows, scoring each arm with the canonical identity."""
    cs = score_suite(control_rows, lam=lam, floor=floor, on_over_budget=on_over_budget)
    ks = score_suite(candidate_rows, lam=lam, floor=floor, on_over_budget=on_over_budget)
    if cs.n != ks.n:
        raise ValueError(f"arms have different MLP counts: {cs.n} vs {ks.n}")
    return paired_compare(cs.scores, ks.scores, **kw)


@dataclass(frozen=True)
class Verdict:
    """The binding promotion decision. THE gate, executable.

    Adopted 2026-08-20 (CP0.3, Samuel's decision): the primary statistic is the
    paired ARITHMETIC score difference with the STUDENTISED-t interval, because
    that is the competition's own identity -- `mean_i (mse_i * max(0.1, C_i/B))`
    over the public MLPs. The geometric mean of per-MLP ratios is retained as a
    STABILITY DIAGNOSTIC and has no authority over the outcome.

    Before this, `docs/VALIDATION.md` 2b bound the geometric mean, and on our own
    graded data the two select opposite winners (1.0334x vs 0.9666x, same 50 rows).

    Exists so a caller cannot pick a favourable interval after seeing the answer:
    `passed` is computed from the pre-registered `gate`, and nothing else in this
    module returns a boolean.
    """

    passed: bool
    gain: float                      # arithmetic, control / candidate
    gate: float
    ci: Tuple[float, float]          # studentised-t, on the paired DIFFERENCE
    ci_excludes_zero: bool
    interval: str
    #: Diagnostics. None of these can change `passed`.
    geometric_gain: float
    estimands_disagree: bool
    n: int
    n_candidate_better: int
    sign_test_p: float
    biggest_mover_share: float
    reasons: Tuple[str, ...]

    def __str__(self) -> str:
        lines = [
            f"VERDICT  {'PASS' if self.passed else 'FAIL'}   "
            f"arithmetic gain {self.gain:.4f}x   gate {self.gate:.4f}x",
            f"  {self.interval} 95% CI on the paired difference "
            f"[{self.ci[0]:+.4e}, {self.ci[1]:+.4e}]   "
            f"excludes zero: {self.ci_excludes_zero}",
            f"  n = {self.n}, candidate better on {self.n_candidate_better}, "
            f"sign-test p = {self.sign_test_p:.4f}",
            f"  DIAGNOSTIC geometric {self.geometric_gain:.4f}x"
            + ("   *** ESTIMANDS DISAGREE — the competition is arithmetic ***"
               if self.estimands_disagree else ""),
            f"  DIAGNOSTIC biggest single mover carries "
            f"{self.biggest_mover_share * 100:.1f}% of the delta"
            + ("   *** >=100%: treat as a draw until proven otherwise ***"
               if abs(self.biggest_mover_share) >= 1.0 else ""),
        ]
        for r in self.reasons:
            lines.append(f"  - {r}")
        return chr(10).join(lines)


def decide(
    control: Sequence[float] | np.ndarray,
    candidate: Sequence[float] | np.ndarray,
    *,
    gate: float,
    interval: str = "t",
    **kw: Any,
) -> Verdict:
    """Apply THE gate. Arithmetic primary, studentised-t or basic bootstrap.

    `gate` is the pre-registered threshold and is REQUIRED -- there is no default,
    so a gate cannot be supplied after the data is visible without that being
    obvious in the diff.

    `interval` is "t" (default) or "basic". Measured false-promotion rates against
    a true zero at n ~ 50, nominal 5% (`local_sim/coverage_lab.py`, 192 fresh
    networks, block D18): t **4.1-5.4%**, basic **4.0-5.2%**, percentile 6.8-8.2%,
    BCa **9.1-11.5%**. Percentile and BCa are refused here rather than deprecated,
    because a deprecated option is one somebody still passes.

    A pass requires BOTH the gain to clear the gate AND the interval to exclude
    zero. Both were required under the old geometric gate too; only the estimand
    and the interval have changed.
    """
    if interval not in ("t", "basic"):
        raise ValueError(
            f"interval {interval!r} is not permitted. Use 't' or 'basic'. "
            "The percentile bootstrap and BCa have measured false-promotion rates "
            "of 6.8-8.2% and 9.1-11.5% against a nominal 5% at n ~ 50 on this "
            "metric (docs/GOTCHAS.md); they are not gate-eligible."
        )
    res = paired_compare(control, candidate, **kw)
    a = np.asarray(control, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    d = a - b
    ci = (res.t_ci_difference if interval == "t"
          else paired_diff_ci(d, method="basic",
                              resamples=res.boot_resamples, seed=res.boot_seed))
    excl = ci[0] > 0.0 or ci[1] < 0.0
    tot = float(d.sum())
    mover = float(d[np.argmax(np.abs(d))] / tot) if tot != 0 else float("nan")

    reasons: List[str] = []
    if res.arithmetic_gain < gate:
        reasons.append(f"arithmetic gain {res.arithmetic_gain:.4f}x is below the "
                       f"pre-registered gate {gate:.4f}x")
    if not excl:
        reasons.append(f"the {interval} 95% interval on the paired difference "
                       f"spans zero")
    if res.estimands_disagree:
        reasons.append("the estimands disagree; the improvement is narrow rather "
                       "than broad — report it, it does not change the verdict")
    if abs(mover) >= 1.0:
        reasons.append(f"the biggest single mover carries {mover * 100:.0f}% of the "
                       f"delta; treat as a draw until proven otherwise")
    for n in res.notes:
        reasons.append(n)

    return Verdict(
        passed=bool(res.arithmetic_gain >= gate and excl),
        gain=res.arithmetic_gain, gate=float(gate), ci=tuple(ci),
        ci_excludes_zero=bool(excl), interval=interval,
        geometric_gain=res.geometric_gain,
        estimands_disagree=res.estimands_disagree,
        n=res.n, n_candidate_better=res.n_candidate_better,
        sign_test_p=res.sign_test_p, biggest_mover_share=mover,
        reasons=tuple(reasons),
    )


def format_paired(res: PairedResult, *, control: str = "control", candidate: str = "candidate") -> str:
    """One block of text an evidence packet can paste verbatim."""
    lines = [
        f"paired, n = {res.n} MLPs   ({control} vs {candidate})",
        f"  mean score          {control}: {res.mean_control:.6e}   "
        f"{candidate}: {res.mean_candidate:.6e}",
        "  PRIMARY  arithmetic mean difference (ctrl - cand, >0 = candidate better)",
        f"      delta            {res.mean_difference:+.6e}",
        f"      relative         {res.relative_improvement * 100:+.3f}%   "
        f"(gain {res.arithmetic_gain:.4f}x)",
        f"      bootstrap 95% CI [{res.boot_ci_difference[0]:+.4e}, "
        f"{res.boot_ci_difference[1]:+.4e}]  "
        f"({res.boot_resamples} resamples, seed {res.boot_seed})",
        f"      BCa       95% CI [{res.bca_ci_difference[0]:+.4e}, "
        f"{res.bca_ci_difference[1]:+.4e}]   (2x the false-promotion rate at n~50 "
        f"— do not gate on it)",
        f"      relative  95% CI [{res.boot_ci_relative[0] * 100:+.3f}%, "
        f"{res.boot_ci_relative[1] * 100:+.3f}%]",
        f"      studentised   CI [{res.t_ci_difference[0]:+.4e}, "
        f"{res.t_ci_difference[1]:+.4e}]   <- READ THIS ONE (measured at nominal "
        f"coverage; see coverage_lab)",
        "  SECONDARY  geometric mean of per-MLP ratios (>1 = candidate better)",
        f"      gain             {res.geometric_gain:.4f}x   "
        f"95% CI [{res.geometric_ci[0]:.4f}, {res.geometric_ci[1]:.4f}]",
        f"  candidate better on {res.n_candidate_better}/{res.n}   "
        f"sign-test p = {res.sign_test_p:.4f}",
    ]
    if res.estimands_disagree:
        lines.append(
            "  *** ESTIMANDS DISAGREE: arithmetic says "
            f"{'candidate' if res.arithmetic_says_candidate_better else 'control'}, "
            f"geometric says "
            f"{'candidate' if res.geometric_says_candidate_better else 'control'}. "
            "The competition is arithmetic. ***"
        )
    for note in res.notes:
        lines.append(f"  note: {note}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Graded-report adapter — read-only, dict in / rows out, no I/O and no network
# --------------------------------------------------------------------------- #


def failed_from_report_entry(entry: Dict[str, Any]) -> bool:
    """The grader's own failure disjunction, from a `results.per_mlp` entry.

    Mirrors `scoring._is_failed_entry`. Grader reports nest the exhaustion flags
    under `telemetry`; the in-process evaluator returns them flat. Both are read.
    """
    tel = entry.get("telemetry") or {}
    if entry.get("error_code"):
        return True
    return any(bool(entry.get(k)) or bool(tel.get(k)) for k in _FAILURE_FLAGS)


def rows_from_graded_report(
    report: Dict[str, Any],
    *,
    budget: Optional[float] = None,
    require_mse: bool = True,
) -> List[MlpRow]:
    """Rebuild scorable rows from a durable graded report.

    Per-MLP MSE is only published for the PUBLIC half (`results.public_scores`,
    indices 0-49 of 100). Cost telemetry is published for all rows. With
    `require_mse=True` this returns only the rows that carry both, so the caller
    can never silently score a half-populated suite.

    `budget` defaults to the report's own `config_snapshot.flop_budget`; it is
    never defaulted to a Phase-1 constant.
    """
    ev = report.get("evaluation", report)
    cfg = ev.get("config_snapshot") or {}
    b = float(budget) if budget is not None else float(cfg.get("flop_budget", 0.0) or 0.0)
    if b <= 0:
        raise ValueError(
            "no positive flop_budget in the report's config_snapshot and none passed; "
            "refusing to guess one"
        )
    res = ev.get("results") or {}
    per_mlp = res.get("per_mlp") or []
    mse_by_index: Dict[int, float] = {}
    for row in res.get("public_scores") or []:
        sc = row.get("scores") or {}
        if "final_layer_mse" in sc:
            mse_by_index[int(row["mlp_index"])] = float(sc["final_layer_mse"])

    out: List[MlpRow] = []
    for entry in per_mlp:
        i = int(entry.get("mlp_index", -1))
        tel = entry.get("telemetry") or {}
        if i not in mse_by_index:
            if require_mse:
                continue
            mse = float("nan")
        else:
            mse = mse_by_index[i]
        out.append(
            MlpRow(
                mse=mse,
                flops_used=float(tel.get("flops_used", 0.0) or 0.0),
                residual_wall_time_s=float(tel.get("residual_wall_time_s", 0.0) or 0.0),
                budget=b,
                failed=failed_from_report_entry(entry),
                name=str(entry.get("mlp_name", "")),
            )
        )
    return out


def reported_scores_by_index(report: Dict[str, Any]) -> Dict[int, float]:
    """The grader's OWN per-MLP adjusted score, for replay against ours."""
    ev = report.get("evaluation", report)
    res = ev.get("results") or {}
    out: Dict[int, float] = {}
    for row in res.get("public_scores") or []:
        sc = row.get("scores") or {}
        if "adjusted_final_layer_score" in sc:
            out[int(row["mlp_index"])] = float(sc["adjusted_final_layer_score"])
    return out
