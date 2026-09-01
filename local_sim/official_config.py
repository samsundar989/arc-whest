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
    server_flop_offset_observed_range=(2 ** 24 - 32, 2 ** 24),
    server_flop_offset_provenance=(
        "MEASURED, NOT DERIVED, AND NOT A CONSTANT. Two graded submissions, "
        "100 MLPs each, every row within a submission identical:\n"
        "  328662 (v2.6.1): local 137,600,326,656 -> graded 137,617,103,872   "
        "offset 16,777,216 = 2**24 exactly\n"
        "  328664 (v2.7)  : local 137,600,359,488 -> graded 137,617,136,672   "
        "offset 16,777,184 = 2**24 - 32\n"
        "So the offset is weakly ESTIMATOR-DEPENDENT. This was preregistered on "
        "2026-09-01 as a bit-identical constant and the preregistered check "
        "FALSIFIED that reading; the claim was corrected rather than kept. "
        "Equivalently: the two files differ by 32,832 counted FLOPs locally but "
        "only 32,800 on the grader -- the server bills the extra calibration "
        "work 32 FLOPs cheaper than the local meter does. "
        "`server_flop_offset_per_mlp` therefore carries the MAXIMUM of the two, "
        "so `graded_flops()` over-estimates rather than under-estimates the "
        "server bill; that is the safe direction for a floor-margin check and "
        "the wrong direction for claiming parity. n = 2 submissions, one round, "
        "one package version, and the variation between them (32 FLOPs) is "
        "1.9e-9 of the offset and 1.5e-11 of the budget. NO GRADED OPERATION "
        "LEDGER EXISTS, so none of this is attributed to an operation and none "
        "of it is a claim of server op-by-op parity. Reports: "
        "experiments/run_reports/submission_3286{62,64}_*_graded_report.json"),
)

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
    check("whestbench is the official Phase-2 stack (0.16.x)",
          wb_ver.startswith("0.16."), wb_ver)
    check("flopscope is the official Phase-2 stack (0.12.x)",
          fs_ver.startswith("0.12."), fs_ver)
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
    report = os.path.join(_ROOT, "experiments", "2026-08-24_p2_p0",
                          "official_anchor_1row.json")
    if os.path.exists(report):
        import json
        with open(report, encoding="utf-8-sig") as fh:
            raw = fh.read()
        doc = json.loads(raw[raw.find("{"):])
        rc = doc.get("run_config") or {}
        check("the report was produced by the official stack",
              str(doc.get("whestbench_version", "")).startswith("0.16."),
              str(doc.get("whestbench_version")))
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
    _GRADED = {  # submission id -> (local official-package count, graded count)
        "328662": (137_600_326_656, 137_617_103_872),
        "328664": (137_600_359_488, 137_617_136_672),
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
          "2**24 on 328662 and 2**24-32 on 328664; the preregistered "
          "'bit-identical constant' reading was falsified 2026-09-01")
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

    return checks, all(c[1] for c in checks)


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
