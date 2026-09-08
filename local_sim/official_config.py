"""THE round-aware official configuration. One authority, asserted not assumed.

P2-P0 (Phase 0 of `docs/GAME_PLAN.md`). Contract and gate G2: `experiments/LOG.md`,
2026-08-24.

WHY THIS FILE EXISTS
--------------------
Before today the competition's own constants lived in at least three places and all
three encoded Phase 1:

    local_sim/competition_score.py:66   LAMBDA_FLOPS_PER_SECOND = 1e11
    local_sim/lifecycle_eval.py:127-131 wall 60 s, residual None, memory 65,536 MB,
                                        budget 2.72e11
    a dozen probes                      an inline `mse * max(0.1, C/B)`

Phase 2 changed four of those at once -- `lambda` 1e11 -> **0**, budget 2.72e11 ->
**2**41**, wall 60 -> **120 s**, memory 65,536 -> **8,192 MB** -- and added a
constraint that did not exist at all: **residual wall time above 0.4 s fails the
MLP**. A repository with three copies of a constant does not update a constant; it
updates two of the copies.

THE TRAP THIS FILE IS BUILT AROUND
----------------------------------
whestbench 0.16 keeps a name called `LAMBDA_FLOPS_PER_SECOND`, and it is still
`1e11`. It is a **deprecated alias for the Phase-1 rate**, explicitly not the
current default (`whestbench/budget.py`: "It has always meant the Phase 1 rate and
still does; it is NOT the current default"). The live default is
`DEFAULT_LAMBDA_FLOPS_PER_SECOND = 0.0`.

So the stale local constant and the live package agree on the *spelling* and differ
on the *meaning*. A version bump alone would not have caught it, and neither would
a test that imports the name. That is why `verify()` below compares against
`budget.CURRENT_ROUND` and `scoring.ContestSpec` field by field, and why every
value here is named for the round it belongs to rather than called "the default".

WHY THE VALUES ARE STATED HERE AND ALSO CHECKED
-----------------------------------------------
`competition_score.py` imports nothing from whestbench on purpose, so it runs in the
torch-only GPU venv exactly as `mint_provenance.py` does. If this module *imported*
the package it would destroy that property for every consumer. So the values are
stated -- which makes the module importable anywhere -- and `verify()` asserts them
against the installed package, which makes the statement checkable. A transcription
nobody checks is a rumour; a transcription checked by the gate is a contract.

`scripts/preflight.ps1` runs `verify()`. A stale stack, a re-pinned package, or a
round advance fails the gate rather than passing quietly.

WHAT IS DERIVED FROM THE PACKAGE AND WHAT IS NOT
-------------------------------------------------
* `flop_budget`, `lambda_flops_per_second`, `residual_wall_time_limit_s`,
  `wall_time_limit_s`, `width`, `depth`, `n_samples` -- carried per round by
  `whestbench.budget.RoundConfig`, so all three rounds are checkable.
* `memory_limit_mb`, `setup_timeout_s`, `predict_timeout_s` -- live only on
  `scoring.ContestSpec` **as defaults**, which means the installed package can only
  confirm them for the round it currently grades. For older rounds they are a
  RECORDED observation with the package version they were read from, and they are
  labelled as such. No graded report we hold publishes any of them.
* `multiplier_floor` and `failure_multiplier` -- asserted behaviourally, by calling
  `budget.score_multiplier`, not by reading a constant. A floor that is read is a
  claim; a floor that is exercised is a measurement.
* `aicrowd_round_id` -- observed from the live challenge page, not from any package.

Run the self-check:  uv run python local_sim/official_config.py
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# APPEND, never insert -- `local_sim/` holds numpy shims named `flopscope` and
# `whestbench`, and `verify()` must reach the REAL packages.
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)


@dataclass(frozen=True)
class OfficialRound:
    """Everything the evaluator needs to know about one graded round.

    Frozen, because a configuration that can be mutated after a measurement is not
    a configuration, it is a variable.
    """

    #: Dataset revision tag. The join key between a round and the data baked for it.
    tag: str
    #: Human name, for reports.
    name: str
    #: AIcrowd round id, observed from the live challenge page. Not in any package.
    aicrowd_round_id: Optional[int]

    # --- shape --------------------------------------------------------------
    width: int
    depth: int
    #: Ground-truth Monte-Carlo samples per MLP in the published splits.
    n_samples: int

    # --- the score identity -------------------------------------------------
    flop_budget: int
    lambda_flops_per_second: float
    multiplier_floor: float
    failure_multiplier: float

    # --- the operational envelope -------------------------------------------
    #: None means residual wall time is PRICED (via lambda) and not gated.
    residual_wall_time_limit_s: Optional[float]
    #: "priced" (lambda > 0, no cap) or "gated" (lambda == 0, hard cap).
    residual_mode: str
    wall_time_limit_s: Optional[float]
    memory_limit_mb: int
    setup_timeout_s: float
    predict_timeout_s: float

    # --- provenance ----------------------------------------------------------
    #: The whestbench version these ContestSpec-only limits were read from.
    limits_read_from: str
    #: True only for the round the installed package currently grades. For any
    #: other round `memory_limit_mb` / `setup_timeout_s` / `predict_timeout_s` are
    #: a historical record and the installed package cannot confirm them.
    limits_verifiable: bool
    note: str

    # --- the grading server's own accounting --------------------------------
    #: FLOPs the GRADING SERVER bills per MLP on top of what the local
    #: official-package subprocess runner counts.
    #:
    #: THIS IS A CONSERVATIVE UPPER BOUND, NOT A CONSTANT, and the distinction
    #: is load-bearing. It was preregistered as a constant on 2026-09-01 and
    #: the preregistered check FALSIFIED that: the two graded submissions show
    #: offsets of 2**24 and 2**24 - 32, so the offset is weakly
    #: estimator-dependent. The MAXIMUM observed is carried here deliberately --
    #: over-estimating what the server bills is the safe direction for a
    #: floor-margin check.
    #:
    #: It is measured, never derived: no graded operation ledger exists, so the
    #: offset is attributed to no operation. Read `server_flop_offset_provenance`
    #: before quoting it, and never claim server op-by-op parity from it.
    server_flop_offset_per_mlp: int = 0
    #: (min, max) offsets actually observed, in FLOPs. `None` where unmeasured.
    #: A range whose endpoints differ is proof the offset is not a constant.
    server_flop_offset_observed_range: Optional[Tuple[int, int]] = None
    #: Where the offset came from, and how much evidence stands behind it.
    #: Empty string means it has never been measured for this round -- which is
    #: NOT the same as measured-and-zero, and the two must not be conflated.
    server_flop_offset_provenance: str = ""

    # ---------------------------------------------------------------------- #

    @property
    def residual_is_priced(self) -> bool:
        return self.lambda_flops_per_second > 0.0

    @property
    def residual_is_gated(self) -> bool:
        return self.residual_wall_time_limit_s is not None

    def effective_compute(self, flops_used: float, residual_wall_time_s: float) -> float:
        """C_m = F_m + lambda * R_m, under THIS round's lambda."""
        return float(flops_used) + self.lambda_flops_per_second * float(residual_wall_time_s)

    @property
    def server_offset_is_measured(self) -> bool:
        """True only where the offset was actually observed on graded reports."""
        return bool(self.server_flop_offset_provenance)

    def graded_flops(self, flops_used: float) -> float:
        """What the GRADING SERVER will bill, given what the LOCAL meter counted.

        Adds this round's measured per-MLP offset. Where no offset has been
        measured this returns `flops_used` unchanged -- which is an admission of
        ignorance, not a claim that the server agrees with the local meter.
        """
        return float(flops_used) + float(self.server_flop_offset_per_mlp)

    def multiplier(self, flops_used: float, residual_wall_time_s: float, *,
                   failed: bool, meter: str = "local") -> float:
        """max(floor, C/B), or the failure multiplier. Uncapped above.

        `meter="local"` prices what the local official-package runner counted.
        `meter="server"` adds `server_flop_offset_per_mlp` first, which is the
        number that decides the outcome. They differ only near a boundary; see
        `floor_margin`.
        """
        if meter not in ("local", "server"):
            raise ValueError("meter must be 'local' or 'server', not %r" % (meter,))
        if failed or self.flop_budget <= 0:
            return self.failure_multiplier
        f = self.graded_flops(flops_used) if meter == "server" else float(flops_used)
        c = self.effective_compute(f, residual_wall_time_s)
        return max(self.multiplier_floor, c / float(self.flop_budget))

    def floor_flops(self, *, meter: str = "local") -> float:
        """The largest per-MLP FLOP count that still prices at the floor.

        At `meter="server"` this is the budget's floor share MINUS the offset:
        the local meter must come in that much under, because the server will
        add the offset back before dividing. This is the only place the offset
        can change an outcome, and it is a hard edge, not a gradient.
        """
        share = self.multiplier_floor * float(self.flop_budget)
        if meter == "server":
            return share - float(self.server_flop_offset_per_mlp)
        if meter != "local":
            raise ValueError("meter must be 'local' or 'server', not %r" % (meter,))
        return share

    def floor_margin(self, flops_used: float, *, meter: str = "server") -> float:
        """How many FLOPs of headroom remain before the multiplier leaves the floor.

        Negative means this candidate is ALREADY off the floor at that meter.
        Default is the server meter deliberately: the local one flatters.
        """
        return self.floor_flops(meter=meter) - float(flops_used)

    def residual_headroom(self, residual_wall_time_s: float) -> Optional[float]:
        """How many times the observed residual fits under the cap. None if uncapped.

        Under Phase 2 this is the ONLY way residual enters the outcome: lambda is
        zero, so residual does not price into C at all -- it is a cliff, and what
        matters is the margin to it, not a cost.
        """
        if self.residual_wall_time_limit_s is None:
            return None
        r = float(residual_wall_time_s)
        if r <= 0.0:
            return float("inf")
        return self.residual_wall_time_limit_s / r


# --------------------------------------------------------------------------- #
# The rounds. Values stated so this module imports anywhere; `verify()` asserts
# every one of them against the installed package.
# --------------------------------------------------------------------------- #

WARMUP = OfficialRound(
    tag="v1-warmup",
    name="Warm-up",
    aicrowd_round_id=None,
    width=256, depth=8, n_samples=1_000_000_000,
    flop_budget=68_000_000_000,
    lambda_flops_per_second=1e11,
    multiplier_floor=0.1,
    failure_multiplier=1.0,
    residual_wall_time_limit_s=None,
    residual_mode="priced",
    wall_time_limit_s=60.0,
    memory_limit_mb=65_536,
    setup_timeout_s=5.0,
    predict_timeout_s=30.0,
    limits_read_from="0.14.0",
    limits_verifiable=False,
    note="First public round. Residual priced at 1e11; nothing gated.",
)

PHASE1 = OfficialRound(
    tag="v1-phase1",
    name="Phase 1",
    aicrowd_round_id=None,
    width=256, depth=32, n_samples=1_000_000_000,
    flop_budget=272_000_000_000,
    lambda_flops_per_second=1e11,
    multiplier_floor=0.1,
    failure_multiplier=1.0,
    residual_wall_time_limit_s=None,
    residual_mode="priced",
    wall_time_limit_s=60.0,
    memory_limit_mb=65_536,
    setup_timeout_s=5.0,
    predict_timeout_s=30.0,
    limits_read_from="0.14.0",
    limits_verifiable=False,
    note=("The graded round for estimator v2.4.1. Retained for REPLAY of Phase-1 "
          "artifacts only -- never a default. Its 65,536 MB memory limit already "
          "killed v2.5 (docs/GOTCHAS.md)."),
)

PHASE2 = OfficialRound(
    tag="v2-phase2",
    name="Phase 2",
    aicrowd_round_id=1429,
    width=1024, depth=16, n_samples=1_000_000_000,
    flop_budget=2 ** 41,                      # 2,199,023,255,552
    lambda_flops_per_second=0.0,              # NOT 1e11. See the module docstring.
    multiplier_floor=0.1,
    failure_multiplier=1.0,
    residual_wall_time_limit_s=0.4,           # a CLIFF, not a price
    residual_mode="gated",
    wall_time_limit_s=120.0,
    memory_limit_mb=8_192,                    # eight times tighter than Phase 1
    setup_timeout_s=5.0,
    predict_timeout_s=30.0,
    limits_read_from="0.16.0",
    limits_verifiable=True,
    note=("Wider and shallower. lambda is 0 and residual is capped at 0.4 s "
          "instead, so C == F and the FLOP budget means what it says."),
    server_flop_offset_per_mlp=2 ** 24,       # 16,777,216 -- the MAXIMUM observed
    server_flop_offset_observed_range=(0, 2 ** 24),
    server_flop_offset_provenance=(
        "MEASURED, NOT DERIVED, AND NOT A CONSTANT. FOUR graded submissions, "
        "100 MLPs each, every row within a submission identical. The LOCAL "
        "figure is taken under the GRADER-TWIN ARM "
        "(`lifecycle_eval.build_weights(..., 'f64_readonly')`) with whestbench's "
        "own in-window `materialise_predictions` included -- i.e. what the local "
        "meter counts once it is finally handed the dtype the grader hands:\n"
        "  328662 (v2.6.1, fs 0.12.0): local 137,600,326,656 -> graded "
        "137,617,103,872   offset 16,777,216 = 2**24 exactly\n"
        "  328664 (v2.7,   fs 0.12.0): local 137,600,359,488 -> graded "
        "137,617,136,672   offset 16,777,184 = 2**24 - 32\n"
        "  329896 (canary4, fs 0.12.1): local 168,669,961,830 -> graded "
        "168,669,961,830   offset 0 -- EXACT\n"
        "  330212 (k3_full, fs 0.12.1): local 1,714,587,938,406 -> graded "
        "1,714,587,938,406   offset 0 -- EXACT, and PREDICTED BY THIS MODEL "
        "BEFORE SUBMISSION at 10.2x canary4's FLOPs. The cast reading below is "
        "therefore confirmed out of sample, not merely fitted.\n"
        "The 0.16.0->0.16.1 / 0.12.0->0.12.1 stack move is NOT the explanation: "
        "329889 re-graded v2.7's UNCHANGED bytes on 0.16.1/0.12.1 and billed "
        "137,617,136,672, bit-identical to 328664 on the old stack. What "
        "separates the rows is how the candidate touches the grader's float64 "
        "weights. v2.6.1 and v2.7 open with `w.astype(fnp.float64)`, which on "
        "the grader is a SAME-DTYPE f64 cast; canary4 opens with a real "
        "f64->f32 cast. The local meter bills both at 2 FLOP/element (measured "
        "directly 2026-09-05: f64->f32, f32->f64 and f64->f64 all cost 2/elt "
        "with copy=True; f32->f32 costs 1; `copy=False` on a same-dtype cast "
        "costs 0 -- the keyword that cost five slots). The residual +2**24 on "
        "the two same-dtype rows is exactly ONE extra FLOP per weight element "
        "(16 * 1024**2); the candidate that does a genuine cast shows none. "
        "That is a MODEL, not a mechanism: no graded operation ledger exists, "
        "so nothing here is attributed to an operation and nothing here claims "
        "server op-by-op parity.\n"
        "`server_flop_offset_per_mlp` carries the MAXIMUM observed, so "
        "`graded_flops()` over-estimates rather than under-estimates -- the safe "
        "direction for a floor-margin check and the wrong direction for claiming "
        "parity. Reports: experiments/run_reports/"
        "submission_3286{62,64}_*, canary4_329896_*, s329889_329889_*."),
)

# --------------------------------------------------------------------------
# THE GRADED RESIDUAL LAW
# --------------------------------------------------------------------------
# Phase 2 does not PRICE residual wall time; it caps it at 0.4 s and FAILS the
# MLP above that. So residual is a cliff to clear, and clearing it needs a
# predictor of what the GRADING SERVER will spend -- not what this host spends.
#
# This host is 8x pessimistic and unstable run to run. Measured 2026-09-05 on
# `estimator_wp18_canary4.py` (submission 329896, 100/100 rows, graded):
#
#     local  (certify section 5, 4 seeds): 2,690 ops, 234-244 ms residual
#                                          -> 87-91 us/op
#     graded (329896 telemetry, 100 rows): 2,673 ops,  29.7-31.0 ms residual
#                                          -> 11.1-11.6 us/op
#
# CONFIRMED OUT OF SAMPLE 2026-09-07 by submission 330212 (estimator_k3_full,
# 3,249 ops, 1,714.6 GFLOP -- 10.2x canary4): graded residual 35.5-39.3 ms,
# i.e. 10.93-12.10 us/op, median 11.205 against canary4's 11.263. The median
# moved 0.5%; the tail widened. Gate on the tail. VALIDATION.md 2b-iii.
#
# The per-op figure is what transfers. Two independent corroborations:
#   * the whole Phase-2 top-8 sits at 5-13 s of `flopscope_server_overhead_wall_s`
#     and 0.29-0.39 s of residual, i.e. 25-35k ops at ~11 us each;
#   * marius_binner's Phase-1 write-up self-measures 11.4 us per flopscope call.
#
# So a candidate is gated on ITS OP COUNT, which the local meter reports exactly
# and which does not depend on the host at all. The local residual SECONDS stay
# in the reports as a ratio-only instrument (worst/median, run to run) and must
# never be quoted as a headroom against the 0.4 s cliff -- doing exactly that on
# 2026-09-05 sized a whole work stream against a number that was 5.9x wrong, and
# the erratum that withdrew it is in `experiments/LOG.md`.

#: Seconds of graded residual per flopscope operation. The MAXIMUM of the
#: observed range, so a prediction errs toward the cliff rather than away.
#: Re-fit 2026-09-07 on a second graded point (330212), which widened the tail
#: 11.47 -> 12.10 us while moving the median by 0.5%.
GRADED_RESIDUAL_S_PER_OP: float = 13.4e-6

#: The observed range, kept so nobody can quote the point estimate as exact.
#: The floor is the visible population's minimum (504aldo, 8.99 us/op) and the
#: ceiling is its maximum (neilarmstrong4, 13.31 us/op), rounded out.
GRADED_RESIDUAL_S_PER_OP_RANGE: Tuple[float, float] = (9.0e-6, 13.4e-6)

GRADED_RESIDUAL_PROVENANCE = (
    "MEASURED on TWO graded submissions of ours, CORROBORATED on twenty more, "
    "and quoted throughout in the GRADER'S WEIGHT CONVENTION -- f64 read-only "
    "weights plus whestbench's own `materialise_predictions`, which is the only "
    "convention whose FLOP total reproduces the graded total exactly. Getting "
    "that wrong is the trap this string exists to close: an earlier version of "
    "this constant cited canary4 at '2,673 ops', which is the f32 CENSUS "
    "convention AND was read off a different file (estimator_wp18_canary.py, "
    "submission 329887, which FAILED). Comparing it against k3_full's 3,249 "
    "compared two conventions as one.\n"
    "  329896 (canary4):  2,690 ops/MLP, 168.7 GFLOP, residual 0.0292-0.0307 s "
    "over 100 rows -> 10.87-11.40 us/op, median 11.192.\n"
    "  330212 (k3_full):  3,249 ops/MLP, 1,714.6 GFLOP, residual 0.0355-0.0393 s "
    "over 100 rows -> 10.93-12.10 us/op, median 11.204.\n"
    "THE MEDIANS ARE 0.10% APART ACROSS A 10.2x CHANGE IN FLOPs and a 3.2x "
    "change in arithmetic intensity (66.4 -> 215.1 GFLOP/s): the graded residual "
    "is a per-OPERATION constant and array traffic does not enter it -- that is "
    "billed to flopscope_server_kernel_wall_s, which moved only 3.1x on that "
    "10.2x FLOP change and therefore does NOT track FLOPs either.\n"
    "EXTRAPOLATION, which two points 1.21x apart cannot test on their own: "
    "`local_sim/p2_residual_law.py` calibrates overhead-per-operation on our own "
    "two rows (361.7 and 377.5 us/op, agreeing to 4.3%) and inverts it across the "
    "twenty other graded submissions in experiments/board_snapshots/"
    "board_2026-09-05_telemetry.md. That spans 6,223-33,818 inferred operations, "
    "a 5.4x lever STRADDLING the gate, and fits residual = -0.023 s + 11.66 us/op "
    "with R^2 = 0.939 -- a near-zero intercept, i.e. proportional to operations. "
    "Per-op residual across that population runs 8.99-13.31 us. Those op counts "
    "are INFERRED and are a model, never another team's measured op count.\n"
    "The carried figure is the MAXIMUM of everything visible (13.31 us/op, "
    "neilarmstrong4 329639, rounded to 13.4), because the policy on this constant "
    "is to err toward the cliff. Two rows exceed our own worst row: "
    "neilarmstrong4 13.31 and Camaro 12.17.\n"
    "STILL A PLANNING LAW, not a certified constant. Our two direct points are "
    "matmul-dominated files by the same author; the other twenty are inferred "
    "through an overhead model that an unusual operation mix could break "
    "(AndreasHad04 runs at 2.9 GFLOP/s against our 215). Nothing here constrains "
    "CURVATURE, and no experiment in this repository can: the grading server is "
    "not instrumentable from here. See VALIDATION.md 2b-iii.\n"
    "The LOCAL host measures 87-91 us/op for canary4 and over-reads "
    "estimator_k3_full by 12.3x (0.4486 s local median vs 0.036402 s graded, "
    "like-for-like on the SUBMITTED bytes -- experiments/2026-09-05_slice1/"
    "mini_k3_full_nocap.json); local residual seconds are a ratio-only "
    "instrument here."
)

#: SECONDS OF GRADED PREDICT WALL PER COUNTED GFLOP, by arithmetic intensity.
#:
#: THE COST MODEL DID NOT HAVE THIS UNTIL 2026-09-08 AND IT NEARLY COST A
#: SUBMISSION. Phase 2 caps `participant_predict_wall_s` at 120 s per MLP, and
#: until Strassen every candidate here ran so far inside it that nobody modelled
#: it. Strassen trades few large matmuls for many small ones, which is
#: FLOP-cheap and WALL-EXPENSIVE:
#:
#:     submission        FLOPs      kernel wall   GFLOP/s   predict wall (worst)
#:     330212 k3_full    1,714.6 G     7.97 s      215.1        9.54 s
#:     330290 L=4        1,068.2 G    98.96 s       10.8      114.68 s   <- 1.046x
#:                                                                   (0.65x under contention)
#:
#: A 0.62x cut in FLOPs bought a 12.4x rise in kernel wall, because arithmetic
#: intensity fell 20x. The worst graded row cleared the cap by 5.3 SECONDS.
#:
#: TWO RULES FOLLOW, and both are the kind that only a graded run can establish:
#:   1. **The grader is SLOWER than this host, not faster.** k3_full ran 3.87 s
#:      locally under `--runner subprocess` and 9.22 s graded -- 2.4x. Never
#:      assume local wall has margin the grader will widen.
#:   2. **Quote GFLOP/s next to counted FLOPs for any candidate that changes the
#:      SHAPE of its arithmetic.** A FLOP reduction that lowers intensity can
#:      cost wall faster than it saves budget.
GRADED_WALL_S_PER_MLP_OBSERVED: Dict[str, float] = {
    "329896": 3.86, "330212": 9.54, "330290": 114.68,
}
#: Worst graded predict wall seen, in seconds, against the 120 s cap. A candidate
#: whose projected wall exceeds this needs its own graded measurement first.
GRADED_WALL_S_WORST_OBSERVED: float = 114.68

#: How much the grading server's wall can move BETWEEN RUNS of identical bytes.
#: MEASURED, not assumed: v2.7 was graded twice, `328664` (0.16.0/0.12.0) and
#: `329889` (0.16.1/0.12.1), with `flops_used = 137,617,136,672` on both.
#:
#:     run      wall median   wall max    max/med   rows > 1.15x med
#:     328664     0.9209 s    1.0527 s     1.143          0 / 100
#:     329889     0.9151 s    1.5979 s     1.746         16 / 100
#:
#: The median moved `-0.6%` and the MAXIMUM moved `+51.8%`, in a contiguous index
#: window -- a server-contention episode, on the stack that grades us now. Any
#: wall margin has to survive this, because a failed MLP is zeroed at multiplier
#: 1.0 and costs ~3e5 at our operating point.
#:
#: UNRESOLVED, and it is why this is a planning factor and not a law: nobody here
#: can tell whether a 1.6x episode on a 0.9 s/MLP file scales multiplicatively to
#: a 105 s row or is a fixed per-row overhead invisible at that scale. The
#: grading server is not instrumentable from here.
GRADED_WALL_CONTENTION_FACTOR: float = 1.60

#: The op-count gate for a Phase-2 candidate. 30,000 -> 22,000 on 2026-09-07.
#: 22,000 ops x 13.4 us = 0.295 s against the 0.4 s cliff, a 1.36x margin,
#: replacing 30,000 x 11.6 us = 0.348 s and its 1.15x -- stricter in operations
#: and safer in seconds at once.
#:
#: An intermediate 24,000 was carried for part of 2026-09-07 and was NOT
#: conservative under this constant's own stated policy: it was sized on OUR
#: worst row (12.096 us/op, a single MLP), while two rows of the visible
#: population sit above it (neilarmstrong4 13.31, Camaro 12.17). 24,000 was also
#: not derived from the data -- it was specified in the Slice 2 handoff BEFORE
#: 330212 was graded, and the LOG entry that "derived" it was a retro-derivation.
#: An independent Skeptic caught both on 2026-09-07; see experiments/LOG.md.
#:
#: Leaders sit at 0.10-0.36 s of graded residual (NOT the "0.29-0.39 s" an
#: earlier version of this comment claimed -- that was false for suliman_tadros
#: 0.124, Puffi 0.193 and hyojun_kwon 0.237). Several do operate near the cliff;
#: we deliberately do not. Anything under this gate is safe on the evidence we
#: have; anything over it needs its own graded measurement before it is submitted.
PHASE2_OP_COUNT_GATE: int = 22_000


def predicted_graded_residual_s(n_ops: float,
                                per_op_s: float = GRADED_RESIDUAL_S_PER_OP) -> float:
    """What the GRADING SERVER is predicted to spend in residual, from op count.

    This is the number to gate on. `n_ops` comes from the local flopscope
    ledger (`BudgetContext.summary_dict(by_namespace=True)`, summing `calls`),
    is exact, and does not depend on this host's speed.
    """
    return float(n_ops) * float(per_op_s)


def predicted_graded_residual_headroom(n_ops: float, round_: "OfficialRound" = None
                                       ) -> Optional[float]:
    """`cap / predicted`, or None where the round has no residual cap."""
    r = round_ if round_ is not None else LIVE
    cap = r.residual_wall_time_limit_s
    if not cap:
        return None
    pred = predicted_graded_residual_s(n_ops)
    return float("inf") if pred <= 0 else cap / pred


def op_count_at_cap(round_: "OfficialRound" = None) -> Optional[float]:
    """The op count whose PREDICTED graded residual exactly hits the cliff."""
    r = round_ if round_ is not None else LIVE
    cap = r.residual_wall_time_limit_s
    return None if not cap else cap / GRADED_RESIDUAL_S_PER_OP


ROUNDS: Dict[str, OfficialRound] = {r.tag: r for r in (WARMUP, PHASE1, PHASE2)}

#: The round currently being graded. Everything that does not name a round gets
#: this one, and `verify()` asserts it still matches `budget.CURRENT_ROUND`.
LIVE: OfficialRound = PHASE2


def live() -> OfficialRound:
    return LIVE


def by_tag(tag: str) -> OfficialRound:
    try:
        return ROUNDS[tag]
    except KeyError:
        raise KeyError(
            f"unknown round tag {tag!r}; known: {sorted(ROUNDS)}"
        ) from None


def for_shape(width: int, depth: int) -> OfficialRound:
    """The round whose published shape is (width, depth).

    Refuses on an unknown shape rather than guessing -- a wrong budget silently
    rescales every score in a run.
    """
    hits = [r for r in ROUNDS.values() if r.width == width and r.depth == depth]
    if len(hits) == 1:
        return hits[0]
    raise KeyError(
        f"no unique official round for shape {width}x{depth}; "
        f"candidates {[r.tag for r in hits]}. Name the round explicitly."
    )


# --------------------------------------------------------------------------- #
# verification against the installed package
# --------------------------------------------------------------------------- #

def _real_whestbench():
    """Import whestbench, refusing the local_sim numpy shim."""
    import whestbench

    path = os.path.abspath(getattr(whestbench, "__file__", "") or "")
    if path.startswith(os.path.abspath(_HERE) + os.sep):
        raise RuntimeError(
            f"whestbench resolved to the local_sim SHIM at {path}. verify() would "
            "be checking the constants against themselves."
        )
    return whestbench


def verify() -> Tuple[List[Tuple[str, bool, str]], bool]:
    """Assert every stated value against the installed whestbench.

    Returns `(checks, ok)` where each check is `(name, passed, detail)`.

    Two things are deliberately NOT asserted:
      * the ContestSpec-only limits of a round the installed package does not
        currently grade -- the package has one set of defaults, not three, so
        claiming otherwise would be a check that cannot fail;
      * `aicrowd_round_id`, which no package carries.
    """
    checks: List[Tuple[str, bool, str]] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        checks.append((name, bool(cond), detail))

    wb = _real_whestbench()
    from whestbench import budget as wbudget
    from whestbench.scoring import ContestSpec

    import importlib.metadata as md
    wb_ver = md.version("whestbench")
    fs_ver = md.version("flopscope")
    # EQUALITY against evidence, not a range against belief. Until 2026-09-05
    # these two lines read `startswith("0.16.")` / `startswith("0.12.")`, and on
    # that day the grading server moved to 0.16.1 / 0.12.1 while this host sat on
    # 0.16.0 / 0.12.0 with every gate green. The range accepted both, so the
    # difference between the measurement host and the grading host was invisible.
    # `local_sim/grader_stack.py` derives the pin from the newest
    # `evaluation.runtime_environment` in `experiments/run_reports/`; a missing or
    # partial frozen file is RED here, never green.
    if _HERE not in sys.path:
        sys.path.append(_HERE)
    import grader_stack as _gs
    try:
        _stack = _gs.load()
        _pin = _stack["pinned"]
        _src = _stack.get("derived_from", {})
        _pin_ok = True
    except Exception as _exc:
        _pin, _src, _pin_ok = {}, {}, False
        check("grader_stack.json exists and is complete (the pin is EVIDENCE, "
              "not a default)", False, f"{type(_exc).__name__}: {_exc}")
    if _pin_ok:
        check("grader_stack.json exists and is complete (the pin is EVIDENCE, "
              "not a default)", True,
              f"from {_src.get('source_report')} "
              f"(submission {_src.get('submission_id')}, {_src.get('observed_at')})")
        check(f"whestbench == the GRADED stack ({_pin['whestbench_version']}), "
              f"not merely 0.16.x",
              wb_ver == _pin["whestbench_version"],
              f"installed {wb_ver}, graded {_pin['whestbench_version']}")
        check(f"flopscope == the GRADED stack ({_pin['flopscope_version']}), "
              f"not merely 0.12.x",
              fs_ver == _pin["flopscope_version"],
              f"installed {fs_ver}, graded {_pin['flopscope_version']}")
        # The derived pin must still MATCH the reports on disk: a frozen file
        # that has fallen behind a newer fetched report is itself drift.
        try:
            _derived = _gs.derive()["pinned"]
        except Exception as _exc:
            _derived = None
            check("the frozen pin is re-derivable from the stored reports",
                  False, f"{type(_exc).__name__}: {_exc}")
        if _derived is not None:
            check("the frozen pin still equals what the NEWEST stored graded "
                  "report says (a stale freeze is drift too)",
                  _derived == dict(_pin),
                  f"frozen {dict(_pin)} vs newest report {_derived}")
    # The declared dependency ranges must not be able to admit a stack the
    # grader is not on. A pin asserted at runtime but not written into
    # pyproject.toml is one `uv sync` away from being undone.
    _pyproject = os.path.join(os.path.dirname(_HERE), "pyproject.toml")
    try:
        with open(_pyproject, "r", encoding="utf-8") as _fh:
            _pp = _fh.read()
    except Exception:
        _pp = ""
    if _pin_ok:
        for _dist, _key in (("whestbench", "whestbench_version"),
                            ("flopscope", "flopscope_version")):
            _want = f'"{_dist}=={_pin[_key]}"'
            check(f"pyproject.toml pins {_dist} to =={_pin[_key]} (a runtime pin "
                  f"a `uv sync` can undo is not a pin)",
                  _want in _pp, _want)
    # Not `True`: a check whose condition is a literal cannot fail, and this repo
    # has lost several checks to exactly that. `_real_whestbench()` above raises on
    # the shim, so the property IS enforced -- but a passing line that tests
    # nothing still inflates the denominator, so it tests the property itself.
    check("whestbench is the real package, not the local_sim shim",
          not os.path.abspath(wb.__file__).startswith(os.path.abspath(_HERE) + os.sep),
          os.path.dirname(wb.__file__))

    # --- the live round is the one the package grades -----------------------
    pkg_current = wbudget.CURRENT_ROUND
    check("LIVE round tag == whestbench budget.CURRENT_ROUND.tag",
          LIVE.tag == pkg_current.tag, f"{LIVE.tag} == {pkg_current.tag}")

    # --- every round, field by field, against budget.RoundConfig ------------
    pkg_rounds = {r.tag: r for r in wbudget.ROUNDS.values()}
    check("every round we know is a round the package knows",
          set(ROUNDS) <= set(pkg_rounds),
          f"ours {sorted(ROUNDS)} <= package {sorted(pkg_rounds)}")
    check("and the package knows no round we have not recorded",
          set(pkg_rounds) <= set(ROUNDS),
          f"package {sorted(pkg_rounds)} <= ours {sorted(ROUNDS)}")

    for tag, ours in sorted(ROUNDS.items()):
        p = pkg_rounds.get(tag)
        if p is None:
            check(f"{tag}: present in the package", False, "missing")
            continue
        for field, mine, theirs in (
            ("width", ours.width, p.width),
            ("depth", ours.depth, p.depth),
            ("n_samples", ours.n_samples, p.n_samples),
            ("flop_budget", ours.flop_budget, p.flop_budget),
            ("lambda", ours.lambda_flops_per_second, p.lambda_flops_per_second),
            ("residual_wall_time_limit_s", ours.residual_wall_time_limit_s,
             p.residual_wall_time_limit_s),
            ("wall_time_limit_s", ours.wall_time_limit_s, p.wall_time_limit_s),
            ("residual_mode", ours.residual_mode, p.residual_mode),
        ):
            check(f"{tag}.{field}", mine == theirs, f"{mine!r} == {theirs!r}")

    # --- the ContestSpec-only limits, for the LIVE round only ---------------
    check("LIVE.limits_verifiable is set for the round the package grades",
          LIVE.limits_verifiable, str(LIVE.limits_verifiable))
    for field, mine in (
        ("memory_limit_mb", LIVE.memory_limit_mb),
        ("setup_timeout_s", LIVE.setup_timeout_s),
        ("predict_timeout_s", LIVE.predict_timeout_s),
        ("wall_time_limit_s", LIVE.wall_time_limit_s),
        ("residual_wall_time_limit_s", LIVE.residual_wall_time_limit_s),
        ("lambda_flops_per_second", LIVE.lambda_flops_per_second),
    ):
        theirs = getattr(ContestSpec, field, "<absent>")
        check(f"ContestSpec.{field}", mine == theirs, f"{mine!r} == {theirs!r}")

    # `width`, `depth` and `flop_budget` deliberately have NO class default on
    # ContestSpec -- they are required per-run arguments. That is a property worth
    # asserting: a required field cannot quietly carry a stale round's value the
    # way a defaulted one can. Their values are checked against
    # `budget.RoundConfig` above, and against a real run's `run_config` below.
    import dataclasses as _dc
    _spec_fields = {f.name: f for f in _dc.fields(ContestSpec)}
    for field in ("width", "depth", "flop_budget"):
        f = _spec_fields.get(field)
        check(f"ContestSpec.{field} is REQUIRED, so it carries no stale default",
              f is not None and f.default is _dc.MISSING
              and f.default_factory is _dc.MISSING,
              "no class default")
    for tag, r in sorted(ROUNDS.items()):
        if r is LIVE:
            continue
        check(f"{tag}: ContestSpec-only limits are NOT claimed verifiable",
              not r.limits_verifiable,
              f"read from whestbench {r.limits_read_from}, recorded not asserted")

    # --- the multiplier, exercised rather than read -------------------------
    B = LIVE.flop_budget
    check("floor: a tiny C is floored at 0.1, by the package",
          wbudget.score_multiplier(1.0, B, failed=False) == LIVE.multiplier_floor,
          f"{wbudget.score_multiplier(1.0, B, failed=False)}")
    check("floor: ...and by us", LIVE.multiplier(1.0, 0.0, failed=False)
          == LIVE.multiplier_floor)
    check("uncapped above: C = 5B gives 5.0, by the package",
          wbudget.score_multiplier(5.0 * B, B, failed=False) == 5.0,
          f"{wbudget.score_multiplier(5.0 * B, B, failed=False)}")
    check("uncapped above: ...and by us",
          LIVE.multiplier(5.0 * B, 0.0, failed=False) == 5.0)
    check("failure is priced at 1.0, NOT the 0.1 floor, by the package",
          wbudget.score_multiplier(1.0, B, failed=True) == LIVE.failure_multiplier,
          f"{wbudget.score_multiplier(1.0, B, failed=True)}")
    check("failure is priced at 1.0 ...and by us",
          LIVE.multiplier(1.0, 0.0, failed=True) == LIVE.failure_multiplier)

    # --- lambda = 0 means C == F, and the deprecated alias does not ---------
    check("package effective_compute default: C == F (lambda is 0)",
          wbudget.effective_compute(1.0e9, 10.0) == 1.0e9,
          f"{wbudget.effective_compute(1.0e9, 10.0):.6e}")
    check("our LIVE.effective_compute agrees",
          LIVE.effective_compute(1.0e9, 10.0) == 1.0e9)
    check("the DEPRECATED alias LAMBDA_FLOPS_PER_SECOND is still 1e11 -- and is "
          "NOT the live rate",
          wbudget.LAMBDA_FLOPS_PER_SECOND == 1e11
          and wbudget.DEFAULT_LAMBDA_FLOPS_PER_SECOND == 0.0,
          f"alias {wbudget.LAMBDA_FLOPS_PER_SECOND:.0e}, "
          f"default {wbudget.DEFAULT_LAMBDA_FLOPS_PER_SECOND:.0e}")
    check("PHASE1's lambda equals the package's Phase-1 rate",
          PHASE1.lambda_flops_per_second == wbudget.PHASE1_LAMBDA_FLOPS_PER_SECOND,
          f"{PHASE1.lambda_flops_per_second:.0e}")

    # --- residual semantics -------------------------------------------------
    check("Phase 2 gates residual and does not price it",
          LIVE.residual_is_gated and not LIVE.residual_is_priced,
          f"cap {LIVE.residual_wall_time_limit_s}s, lambda "
          f"{LIVE.lambda_flops_per_second}")
    check("Phase 1 prices residual and does not gate it",
          PHASE1.residual_is_priced and not PHASE1.residual_is_gated)
    check("headroom is a ratio to the cap, not a cost",
          abs(LIVE.residual_headroom(0.05368) - 0.4 / 0.05368) < 1e-12,
          f"{LIVE.residual_headroom(0.05368):.3f}x at the recorded anchor residual")
    check("an uncapped round has no headroom number to quote",
          PHASE1.residual_headroom(0.05) is None)

    # --- shape lookup refuses to guess --------------------------------------
    check("for_shape(1024, 16) is Phase 2", for_shape(1024, 16) is PHASE2)
    check("for_shape(256, 32) is Phase 1", for_shape(256, 32) is PHASE1)
    try:
        for_shape(512, 4)
        ok_unknown = False
    except KeyError:
        ok_unknown = True
    check("for_shape refuses an unknown shape rather than guessing", ok_unknown)

    # --- against what the RUNNER actually used, not only what it declares ----
    # A default the package states and a value the runner used are two different
    # facts. Where a real `whest run --json` report exists, check the second.
    # Regenerated 2026-09-05 on the PINNED stack (0.16.1 / 0.12.1); the 0.16.0
    # report beside it is retained for audit and is op-for-op identical.
    report = os.path.join(_ROOT, "experiments", "2026-08-24_p2_p0",
                          "official_anchor_1row_0161.json")
    if os.path.exists(report):
        import json
        with open(report, encoding="utf-8-sig") as fh:
            raw = fh.read()
        doc = json.loads(raw[raw.find("{"):])
        rc = doc.get("run_config") or {}
        check("the report was produced by the PINNED grader stack (equality, "
              "not 0.16.x)",
              str(doc.get("whestbench_version", "")) ==
              (_pin.get("whestbench_version") if _pin_ok else object()),
              f"report {doc.get('whestbench_version')}, "
              f"pinned {_pin.get('whestbench_version') if _pin_ok else 'UNKNOWN'}")
        check("the report is the LIVE round's dataset revision",
              str((rc.get("dataset") or {}).get("path", "")).endswith("@" + LIVE.tag),
              str((rc.get("dataset") or {}).get("path")))
        for key, mine in (
            ("width", LIVE.width),
            ("depth", LIVE.depth),
            ("flop_budget", LIVE.flop_budget),
            ("lambda_flops_per_second", LIVE.lambda_flops_per_second),
            ("residual_wall_time_limit_s", LIVE.residual_wall_time_limit_s),
            ("wall_time_limit_s", LIVE.wall_time_limit_s),
            ("setup_timeout_s", LIVE.setup_timeout_s),
        ):
            check(f"run_config.{key} (what the runner USED)", rc.get(key) == mine,
                  f"{mine!r} == {rc.get(key)!r}")
        # ...and the two it does NOT carry, asserted as ABSENT. Without this, a
        # reader sees seven corroborated limits and generalises to "the run_config
        # corroborates the limits" -- which a durable doc did, for exactly the two
        # weakest ones. `memory_limit_mb` is the limit this host cannot test at
        # all, so it is also the one whose provenance matters most to state
        # honestly. If a future whestbench starts publishing them, this check
        # fails and the claim can be UPGRADED deliberately.
        for key in ("memory_limit_mb", "predict_timeout_s"):
            check(f"run_config does NOT carry {key} (class default only)",
                  key not in rc,
                  "absent -- so it is a ContestSpec default, not an observation")
        mult = (doc.get("results") or {}).get("mean_score_multiplier")
        check("the observed multiplier is the floor, so C/B < floor at this "
              "operating point", mult == LIVE.multiplier_floor, f"{mult!r}")
    else:
        check("an official run_config was available to check against", False,
              f"missing {os.path.relpath(report, _ROOT)} -- run the anchor first")

    # --- the frozen dataclass is actually frozen ----------------------------
    try:
        LIVE.flop_budget = 1  # type: ignore[misc]
        frozen = False
    except Exception:
        frozen = True
    check("OfficialRound is frozen (a mutable config is a variable)", frozen)
    check("replace() still produces a modified COPY for what-if analysis",
          replace(LIVE, flop_budget=1).flop_budget == 1
          and LIVE.flop_budget == 2 ** 41)

    # --- the grading server's offset ---------------------------------------
    # These are asserted against the STORED GRADED REPORTS, not against the
    # installed package -- the package cannot see the grading server. WP0B's
    # lesson: a constant no gate scans is a constant nobody has checked.
    # submission id -> (local count UNDER THE GRADER-TWIN ARM, graded count).
    # canary4 was added 2026-09-05 and immediately falsified the "+2**24 always"
    # reading: under the twin arm its local count equals the graded count exactly.
    _GRADED = {
        "328662": (137_600_326_656, 137_617_103_872),
        "328664": (137_600_359_488, 137_617_136_672),
        "329896": (168_669_961_830, 168_669_961_830),
        # 330212 was PREDICTED to be offset-zero by the cast model below, before
        # it was submitted, and graded offset-zero on 2026-09-07 at 10.2x the
        # FLOPs of the row the model was fitted on. Out-of-sample confirmation.
        "330212": (1_714_587_938_406, 1_714_587_938_406),
    }
    check("the Phase-2 server offset is recorded as MEASURED, with a range",
          PHASE2.server_offset_is_measured
          and PHASE2.server_flop_offset_observed_range is not None,
          f"range {PHASE2.server_flop_offset_observed_range}")
    check("graded_flops() never UNDER-estimates a stored graded total",
          all(PHASE2.graded_flops(loc) >= g for loc, g in _GRADED.values()),
          "errors " + ", ".join(
              f"{PHASE2.graded_flops(loc) - g:+,}" for loc, g in _GRADED.values()))
    check("the offset is NOT a constant, and the range says so",
          (PHASE2.server_flop_offset_observed_range[0]
           != PHASE2.server_flop_offset_observed_range[1]),
          "2**24 on 328662, 2**24-32 on 328664, ZERO on 329896 under the "
          "grader-twin arm; the preregistered 'bit-identical constant' reading "
          "was falsified 2026-09-01 and the range widened to [0, 2**24] on "
          "2026-09-05")
    # The zero row carries the design consequence, so it is asserted rather than
    # left in prose: a candidate that performs a GENUINE dtype-changing cast of
    # the grader's weights shows NO offset, and the twin arm predicts its graded
    # bill exactly. `estimator_k3_full.py` is deliberately in that family; v2.6.1
    # and v2.7, whose `astype(fnp.float64)` is a same-dtype no-op on the grader,
    # are the two that show +2**24.
    #
    # THE MODEL HAS NOW BEEN TESTED OUT OF SAMPLE AND HELD. The sentence above
    # was written on 2026-09-05 as a PREDICTION about a file that had not been
    # submitted. On 2026-09-07 that file graded as 330212 with offset exactly 0,
    # at 10.2x the FLOPs of canary4. Both members of the zero branch do a real
    # f64->f32 cast; both members of the +2**24 branch do a same-dtype f64 cast.
    _ZERO_OFFSET = {k: v for k, v in _GRADED.items() if v[0] == v[1]}
    check("the genuine-cast family is predicted EXACTLY by the grader-twin arm "
          "(offset 0), so the offset is not a server constant",
          len(_ZERO_OFFSET) >= 2,
          "329896 canary4 168,669,961,830 and 330212 k3_full "
          "1,714,587,938,406 -- local == graded, and 330212 was predicted "
          "before it was submitted")
    check("every stored graded offset lies inside the recorded range",
          all(PHASE2.server_flop_offset_observed_range[0] <= g - loc
              <= PHASE2.server_flop_offset_observed_range[1]
              for loc, g in _GRADED.values()))
    check("the carried offset is the MAXIMUM observed (conservative direction)",
          PHASE2.server_flop_offset_per_mlp
          == max(g - loc for loc, g in _GRADED.values())
          == PHASE2.server_flop_offset_observed_range[1],
          f"{PHASE2.server_flop_offset_per_mlp:,}")
    check("a round with no measured offset is distinguishable from offset-zero",
          PHASE1.server_flop_offset_per_mlp == 0
          and PHASE1.server_flop_offset_observed_range is None
          and not PHASE1.server_offset_is_measured,
          "Phase 1 never measured; that is NOT measured-and-zero")
    check("meter='server' costs exactly the offset in floor headroom",
          (PHASE2.floor_flops(meter="local") - PHASE2.floor_flops(meter="server"))
          == PHASE2.server_flop_offset_per_mlp,
          f"{PHASE2.floor_flops(meter='server'):,.0f} FLOPs is the local ceiling "
          f"for a floor-pinned candidate")
    check("multiplier() defaults to the LOCAL meter and rejects any other name",
          PHASE2.multiplier(1e11, 0.0, failed=False)
          == PHASE2.multiplier(1e11, 0.0, failed=False, meter="local")
          and _raises(lambda: PHASE2.multiplier(1e11, 0.0, failed=False,
                                                meter="grader")))
    check("the offset cannot rescue a FAILED MLP",
          PHASE2.multiplier(1e9, 0.0, failed=True, meter="server")
          == PHASE2.failure_multiplier)

    # --- the residual law, RECOMPUTED from the graded reports ----------------
    # WP0B's own lesson, applied to this file's other constant: "a constant no
    # gate scans is a constant nobody has checked." Until 2026-09-07 the offset
    # constants were recomputed against `_GRADED` while GRADED_RESIDUAL_S_PER_OP
    # and PHASE2_OP_COUNT_GATE were trusted as prose -- and an independent
    # Skeptic found the prose citing the wrong file, in the wrong weight
    # convention, calling a local number "graded". These checks close that.
    #
    # submission id -> (ops per MLP in the GRADER'S weight convention, report).
    # The op counts are measured by replaying each candidate under
    # `lifecycle_eval.build_weights(..., 'f64_readonly')` plus whestbench's own
    # `materialise_predictions`; that is the ONLY convention whose FLOP total
    # reproduces the graded total, and the graded totals in `_GRADED` above are
    # what pin it. Do NOT substitute an f32 census count here.
    _RESIDUAL = {
        "329896": (2690, "canary4_329896_graded_report.json"),
        "330212": (3249, "k3_full_220d2314_330212_graded_report.json"),
        # 330290 is the OUT-OF-SAMPLE test of the law, at 4.7x the op count of
        # either calibration point. Predicted 0.206 s at the carried 13.4 us/op;
        # measured 0.1692 s median, i.e. 10.99 us/op against 11.19 and 11.20 at
        # 2,690 and 3,249 ops. The per-operation law HELD.
        "330290": (15393, "k3_strassen_L4_330290_graded_report.json"),
    }
    observed = _graded_residual_per_op(_RESIDUAL)
    check("the graded reports for the residual law are present",
          len(observed) == len(_RESIDUAL),
          f"found {sorted(observed)} of {sorted(_RESIDUAL)}")
    if observed:
        meds = {k: v["median"] for k, v in observed.items()}
        worst = max(v["max"] for v in observed.values())
        spread = max(meds.values()) / min(meds.values())
        check("the per-op residual MEDIANS agree across candidates "
              "(this is the whole per-operation law)",
              spread < 1.05,
              " ".join(f"{k}={v * 1e6:.3f}us" for k, v in sorted(meds.items()))
              + f"  spread {100 * (spread - 1):.2f}%")
        check("GRADED_RESIDUAL_S_PER_OP is >= every per-op residual we have "
              "MEASURED (the policy is the maximum observed)",
              GRADED_RESIDUAL_S_PER_OP >= worst,
              f"carried {GRADED_RESIDUAL_S_PER_OP * 1e6:.1f} us"
              f" vs worst measured row {worst * 1e6:.3f} us")
        check("...and >= the maximum INFERRED across the visible board "
              "population, which is 5.4x wider in op count than our own two",
              GRADED_RESIDUAL_S_PER_OP >= 13.31e-6,
              "neilarmstrong4 329639 at 13.31 us/op; Camaro 329762 at 12.17."
              " Both exceed our own worst row (12.096) -- see"
              " local_sim/p2_residual_law.py")
        check("every measured per-op residual lies inside the recorded range",
              all(GRADED_RESIDUAL_S_PER_OP_RANGE[0] <= v["min"]
                  and v["max"] <= GRADED_RESIDUAL_S_PER_OP_RANGE[1]
                  for v in observed.values()),
              f"range {tuple(round(x * 1e6, 1) for x in GRADED_RESIDUAL_S_PER_OP_RANGE)} us")
    # --- the WALL, recomputed from the same reports ------------------------
    # Added 2026-09-08 after 330290 cleared the 120 s cap by 5.3 seconds. The
    # cost model had no wall term at all until then.
    _wall = {}
    for sub, (_ops, fname) in _RESIDUAL.items():
        rows = _find_key(_load_report(fname) or {}, "per_mlp")
        if not isinstance(rows, list):
            continue
        v = sorted(float(r["telemetry"]["participant_predict_wall_s"])
                   for r in rows
                   if isinstance(r, dict) and isinstance(r.get("telemetry"), dict)
                   and r["telemetry"].get("participant_predict_wall_s") is not None)
        if v:
            _wall[sub] = v[-1]
    check("the graded WALL of every recorded submission is inside the cap",
          bool(_wall) and all(w < PHASE2.wall_time_limit_s for w in _wall.values()),
          "  ".join(f"{k}={v:.1f}s" for k, v in sorted(_wall.items()))
          + f"   cap {PHASE2.wall_time_limit_s:.0f}s")
    check("GRADED_WALL_S_WORST_OBSERVED is the maximum actually observed",
          bool(_wall)
          and abs(GRADED_WALL_S_WORST_OBSERVED - max(_wall.values())) < 0.05,
          f"carried {GRADED_WALL_S_WORST_OBSERVED:.2f}s vs"
          f" observed {max(_wall.values()):.2f}s"
          if _wall else "no reports")
    # THE CHECK THIS REPLACES WAS THE REPOSITORY'S OWN NAMED FAILURE MODE, and an
    # independent Skeptic caught it inside the very commit whose knowledge-delta
    # said "a constant no gate scans is a constant nobody has checked". It read:
    #
    #     check("the worst graded wall still leaves a margin -- WARN if under
    #            1.25x", ... >= 1.04, ...)
    #
    # The label advertised 1.25x, the predicate tested 1.04x, and the observed
    # value was 1.046x -- a threshold set 0.6% below the datum that motivated
    # writing it, reporting PASS under a label it did not test. It was also
    # unfalsifiable by construction, because the quantity it compared against is
    # DEFINED as max(_wall.values()).
    #
    # The replacement tests a PROPERTY instead of reporting a STATE: does every
    # graded row survive the contention factor MEASURED on identical bytes graded
    # twice? v2.7's 328664 -> 329889 moved the median wall -0.6% and the MAXIMUM
    # +51.8%, 16/100 rows above 1.15x median. A candidate with less margin than
    # that has an already-observed failure mode, and a failed MLP is zeroed at
    # multiplier 1.0 -- at our operating point a ~3e5 score penalty.
    _surv = {k: PHASE2.wall_time_limit_s / (v * GRADED_WALL_CONTENTION_FACTOR)
             for k, v in _wall.items()}
    # Framed as a SELECTION criterion, not a repository invariant: history is
    # allowed to contain thin rows, but we must always still HOLD a candidate
    # that survives contention, or there is nothing safe to designate. The
    # per-submission margins stay in the detail string so a thin one is visible
    # rather than averaged away.
    check("a contention-surviving nominee still exists "
          f"(margins at the MEASURED {GRADED_WALL_CONTENTION_FACTOR:.2f}x)",
          bool(_surv) and any(m >= 1.0 for m in _surv.values()),
          "  ".join(f"{k}={m:.2f}x" for k, m in sorted(_surv.items()))
          + "   <1.00x = that submission would FAIL rows on a busy server")

    check("the op gate leaves at least a 1.25x margin under the cliff at the "
          "carried per-op figure",
          (PHASE2.residual_wall_time_limit_s
           / (PHASE2_OP_COUNT_GATE * GRADED_RESIDUAL_S_PER_OP)) >= 1.25,
          f"{PHASE2_OP_COUNT_GATE:,} ops x"
          f" {GRADED_RESIDUAL_S_PER_OP * 1e6:.1f} us ="
          f" {PHASE2_OP_COUNT_GATE * GRADED_RESIDUAL_S_PER_OP:.3f} s ->"
          f" {PHASE2.residual_wall_time_limit_s / (PHASE2_OP_COUNT_GATE * GRADED_RESIDUAL_S_PER_OP):.2f}x")

    return checks, all(c[1] for c in checks)


def _load_report(fname: str):
    """The raw graded-report JSON, or None if it is not on disk."""
    import json

    path = os.path.join(_ROOT, "experiments", "run_reports", fname)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:                                          # noqa: BLE001
        return None


def _graded_residual_per_op(spec: Dict[str, Tuple[int, str]]) -> Dict[str, dict]:
    """Per-operation graded residual, recomputed from the raw report JSON.

    Returns `{submission_id: {"min", "median", "max", "ops", "n_rows"}}`, in
    seconds per operation, omitting any submission whose report is absent.
    """
    import json

    out: Dict[str, dict] = {}
    root = os.path.join(_ROOT, "experiments", "run_reports")
    for sub, (ops, fname) in spec.items():
        path = os.path.join(root, fname)
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                doc = json.load(fh)
        except Exception:                                      # noqa: BLE001
            continue
        rows = _find_key(doc, "per_mlp")
        if not isinstance(rows, list) or not rows:
            continue
        vals = sorted(
            float(r["telemetry"]["residual_wall_time_s"]) for r in rows
            if isinstance(r, dict) and isinstance(r.get("telemetry"), dict)
            and r["telemetry"].get("residual_wall_time_s") is not None)
        if not vals:
            continue
        out[sub] = {"min": vals[0] / ops,
                    "median": vals[len(vals) // 2] / ops,
                    "max": vals[-1] / ops,
                    "ops": ops, "n_rows": len(vals)}
    return out


def _find_key(obj, key):
    """First value stored under `key` anywhere in a nested JSON document."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = _find_key(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find_key(v, key)
            if r is not None:
                return r
    return None


def _raises(fn) -> bool:
    """True if `fn()` raises ValueError. A guard that cannot fire is not a guard."""
    try:
        fn()
    except ValueError:
        return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="the round-aware official configuration")
    ap.add_argument("--round", default=None, help="print one round and exit")
    args = ap.parse_args()

    if args.round:
        r = by_tag(args.round)
        for k, v in r.__dict__.items():
            print(f"  {k:<28} {v!r}")
        return 0

    print("official_config -- every value asserted against the installed package")
    print(f"  LIVE round: {LIVE.name} ({LIVE.tag}), AIcrowd round {LIVE.aicrowd_round_id}")
    print()
    try:
        checks, ok = verify()
    except Exception as exc:  # a failure to verify is a failure, not a skip
        print(f"  VERIFY RAISED: {type(exc).__name__}: {exc}")
        return 1
    for name, passed, detail in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {name:<58} {detail}")
    n = sum(c[1] for c in checks)
    print(f"\n  {n}/{len(checks)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
