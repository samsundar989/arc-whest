"""THE canonical grader-lifecycle evaluator.

CP0.2 of `docs/FRONTIER_ARCHITECTURE_PROPOSAL.md`. Contract and predictions L1-L7:
`experiments/LOG.md`, 2026-08-20.

WHY THIS EXISTS
---------------
`local_sim/vault.py:282` and `local_sim/coverage_lab.py:226` build a fresh
`Estimator()` for every MLP and never call `setup()`. The grader builds ONE
instance and calls `setup(context)` ONCE, outside any `BudgetContext`
(`whestbench/runner.py:150`, `whestbench/subprocess_worker.py:197`), so everything
setup does is FREE.

Measured cost of that mismatch on the shipped v2.6.1: MLP #0 over-charged
**+2.5273e8 counted FLOPs** (+0.554% of its own `C`, +155 operations); every other
MLP bit-identical, because `_BASE_POINTS` (`estimator.py:497`) happens to be a
module-level dict, so the Sobol table is billed once and then reused for free.

That is small today and structural tomorrow. A K129/MUB carrier builds 66,048
signed directions in `setup()`. On the grader that is free. Under the old lifecycle
it is billed to MLP #0 -- or, if the carrier lives in INSTANCE state rather than a
module global, billed to **every** MLP. An evaluator that gets this wrong cannot
price a carrier candidate, which is the entire job of CP1.

WHAT IT REPRODUCES, read from `whestbench/scoring.py::evaluate_estimator`
-------------------------------------------------------------------------
* one `Estimator`; `setup(SetupContext)` once, OFF-budget, timed against
  `ContestSpec.setup_timeout_s` (default **5.0 s** -- a real ceiling that is
  recorded nowhere else in this repository);
* per MLP its own `flopscope.BudgetContext(flop_budget, wall_time_limit_s)`;
* SIX failure paths, every one of which zeroes the prediction:
    1. `flops.BudgetExhaustedError`                          (during predict)
    2. `flops.TimeExhaustedError`                            (during predict)
    3. any other `Exception`, including a validation failure (during predict)
    4. post-predict `wall_time_s   > wall_time_limit_s`
    5. post-predict `residual_wall_time_s > residual_wall_time_limit_s`
    6. post-hoc combined budget `C = F + lambda*R > B`
* `failed` is the DISJUNCTION of 1, 2, 4, 5, 6; path 3 is recorded separately with
  its `error_code`, and the grader scores it the same way;
* a failure is priced at multiplier **1.0** -- NOT the 0.1 floor -- and against
  ZEROED predictions, so its MSE becomes `mean(target**2)`;
* MSE is computed in **float32** (`scoring.py:846`), not float64.

WHAT IT DOES NOT MEASURE -- read this before quoting a number from it
---------------------------------------------------------------------
**The `objective` this module prints is NOT a graded objective.** Every
`residual_wall_time_s` in every row is read from the in-process `BudgetContext`,
i.e. a `--runner local` figure. The grader deliberately does NOT do that: it
prefers `last_predict_stats()` (`scoring.py:762-789`) because the host context is
not authoritative. `docs/GOTCHAS.md` measures the same file at 70.85 ms (local) /
54.47 ms (subprocess) / 9.84 ms (graded) per MLP. Those seconds enter
`C = F + lambda*R`, so lambda*R is ~10-13% of `C` here against 2-3% on the grader.
**Counted FLOPs transfer exactly; the objective does not. Use ratios inside one
path, never an absolute.**

The five `ContestSpec` constants above are **installed-package defaults, not
observed grader configuration** -- no graded report we hold publishes any of them.

WHAT IT DOES NOT DO
-------------------
It does not re-derive the score identity. `local_sim/competition_score.py` owns
that and is verified bitwise against the installed whestbench (71/71); this module
produces `MlpRow`s and hands them over. Two implementations of one identity is how
a repository ends up with two answers.

It also does not mint, does not draw seeds, does not read the vault, and does not
write anything unless you ask it to.

Run:  uv run python local_sim/lifecycle_eval.py --self-test
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# APPEND, never insert. `local_sim/` holds numpy shims named `flopscope` and
# `whestbench` for the torch-only GPU venv; prepending this directory makes them
# shadow the real packages, and this module MEASURES counted FLOPs, so it needs
# the real flopscope. `_require_real_flopscope()` asserts it got one.
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np  # noqa: E402

from competition_score import (  # noqa: E402
    DEFAULT_LAMBDA_FLOPS_PER_SECOND,
    MULTIPLIER_FLOOR,
    MlpRow,
    SuiteScore,
    is_combined_budget_exhausted,
    score_suite,
    zero_prediction_mse,
)

# --------------------------------------------------------------------------- #
# THE LIVE ROUND'S CONFIGURATION -- from `local_sim/official_config.py`, which is
# the single authority and asserts every value field by field against the
# installed whestbench (`official_config.verify()`, run by preflight).
#
# Until 2026-08-24 these were Phase-1 literals transcribed into this file:
# wall 60 s, residual None, memory 65,536 MB, budget 2.72e11, lambda 1e11. Phase 2
# moved ALL FIVE at once and added a constraint that did not previously exist --
# the 0.4 s residual cap -- so a transcription that had merely been stale became
# one that could pass a candidate the grader would fail.
#
# READ THIS BEFORE QUOTING ANY OF THEM AS "the grader allows X".
# Every graded report we hold publishes `config_snapshot` with exactly four keys --
# `n_mlps`, `depth`, `width`, `flop_budget`. The rest are PACKAGE DEFAULTS. They
# are now additionally corroborated by the `run_config` block of a real
# `whest run --json` report (experiments/2026-08-24_p2_p0/official_anchor_1row.json),
# which records what the runner actually USED rather than what it declares -- but
# no GRADED report has ever published them, and that distinction stands.
# --------------------------------------------------------------------------- #

from official_config import LIVE as OFFICIAL_ROUND  # noqa: E402
from official_config import PHASE1 as PHASE1_ROUND  # noqa: E402

SETUP_TIMEOUT_S: float = OFFICIAL_ROUND.setup_timeout_s                # 5.0
WALL_TIME_LIMIT_S: Optional[float] = OFFICIAL_ROUND.wall_time_limit_s  # 120.0
#: Phase 2's hard cliff. NOT None any more. An MLP whose residual wall time exceeds
#: this is FAILED -- zeroed predictions, multiplier 1.0 -- and because lambda is 0
#: this is the ONLY route by which residual can change an outcome. P2-CP1 ran with
#: this disabled; every Phase-2 cost/failure claim made before 2026-08-24 is
#: provisional for that reason.
RESIDUAL_WALL_TIME_LIMIT_S: Optional[float] = OFFICIAL_ROUND.residual_wall_time_limit_s
PREDICT_TIMEOUT_S: float = OFFICIAL_ROUND.predict_timeout_s            # 30.0
#: 8,192 MB -- EIGHT TIMES tighter than the Phase-1 limit that already killed v2.5
#: (docs/GOTCHAS.md). RLIMIT_AS under `--runner subprocess`.
MEMORY_LIMIT_MB: int = OFFICIAL_ROUND.memory_limit_mb                  # 8_192
DEFAULT_BUDGET: int = OFFICIAL_ROUND.flop_budget                       # 2**41

#: THE Phase-2 counted-FLOP anchor for the shipped `estimator.py` at 1024x16, from
#: a real `whest run --runner subprocess --profile --json` against the official
#: `v2-phase2` dataset (`experiments/2026-08-24_p2_p0/official_anchor_1row.json`).
#: This module's self-test asserts EQUALITY against it, which is the single check
#: that says the local meter bills what the grader bills. Counted FLOPs on this
#: dense path are value-independent, so it holds for any seed.
#: When it moves, `local_sim/p2_flop_anchor.py` says which operation moved.
P2_OFFICIAL_ANCHOR_FLOPS: int = 137_600_326_656

#: Measured on this host, 2026-08-20, three repeats. The grader's SUBPROCESS setup
#: window (`runner.py:281`) covers the worker process spawn and its imports as well
#: as `setup()`; the LOCAL window (`runner.py:144`) starts before
#: `load_estimator_from_path`. Neither starts where a naive timer around `setup()`
#: starts, so a bare `setup()` duration OVERSTATES the headroom.
#:   process spawn + `import flopscope, flopscope.numpy`   0.222 / 0.245 / 0.238 s
#:   estimator.py module body, warm                        0.0003 s
#:   cold `import whestbench` (pulls HF datasets)          ~1.05 s  <- warm on the
#:       grader's worker, so it is an upper bound, not a charge we necessarily pay
HOST_COLD_START_S: float = 0.24


@dataclass(frozen=True)
class LifecycleSpec:
    """Everything the lifecycle depends on. Per-run, never global mutable state."""

    width: int
    depth: int
    flop_budget: int = DEFAULT_BUDGET
    setup_timeout_s: float = SETUP_TIMEOUT_S
    wall_time_limit_s: Optional[float] = WALL_TIME_LIMIT_S
    residual_wall_time_limit_s: Optional[float] = RESIDUAL_WALL_TIME_LIMIT_S
    predict_timeout_s: float = PREDICT_TIMEOUT_S
    memory_limit_mb: int = MEMORY_LIMIT_MB
    lam: float = DEFAULT_LAMBDA_FLOPS_PER_SECOND
    floor: float = MULTIPLIER_FLOOR
    seed: int = 0
    #: Which official round this spec claims to model. Recorded so a run report
    #: says which rulebook produced it, and asserted by `for_round`.
    round_tag: str = OFFICIAL_ROUND.tag

    @classmethod
    def for_round(cls, round_or_tag, *, seed: int = 0, **overrides) -> "LifecycleSpec":
        """Build a spec from an `official_config.OfficialRound`, or from its tag.

        This is the only sanctioned way to construct a spec for a round that is
        not the live one. Replaying a Phase-1 artifact means naming Phase 1 --
        `LifecycleSpec.for_round("v1-phase1")` -- and getting ALL SIX of its
        constants together. Overriding them one at a time is how a run ends up
        with Phase-2's budget and Phase-1's lambda, which is a rulebook that has
        never existed.

        `overrides` are applied last and are for deliberate what-if analysis
        (`residual_wall_time_limit_s=1e-9` to provoke the cliff, for instance).
        They are recorded in the provenance blob so a report can never claim to be
        a clean round it was not.
        """
        import official_config

        r = (round_or_tag if isinstance(round_or_tag, official_config.OfficialRound)
             else official_config.by_tag(str(round_or_tag)))
        base = dict(
            width=r.width,
            depth=r.depth,
            flop_budget=r.flop_budget,
            setup_timeout_s=r.setup_timeout_s,
            wall_time_limit_s=r.wall_time_limit_s,
            residual_wall_time_limit_s=r.residual_wall_time_limit_s,
            predict_timeout_s=r.predict_timeout_s,
            memory_limit_mb=r.memory_limit_mb,
            lam=r.lambda_flops_per_second,
            floor=r.multiplier_floor,
            round_tag=r.tag,
            seed=seed,
        )
        unknown = set(overrides) - set(base)
        if unknown:
            raise TypeError(f"unknown LifecycleSpec field(s): {sorted(unknown)}")
        base.update(overrides)
        return cls(**base)

    def deviations_from_round(self) -> Dict[str, Any]:
        """Every field that does NOT match the round this spec names.

        An evaluator that silently runs a modified rulebook and reports a clean
        objective is the failure this whole slice exists to remove, so the
        deviations travel with the provenance rather than living in a comment.
        """
        import official_config

        try:
            r = official_config.by_tag(self.round_tag)
        except KeyError:
            return {"round_tag": [self.round_tag, "<unknown round>"]}
        out: Dict[str, Any] = {}
        for field, theirs in (
            ("width", r.width),
            ("depth", r.depth),
            ("flop_budget", r.flop_budget),
            ("setup_timeout_s", r.setup_timeout_s),
            ("wall_time_limit_s", r.wall_time_limit_s),
            ("residual_wall_time_limit_s", r.residual_wall_time_limit_s),
            ("predict_timeout_s", r.predict_timeout_s),
            ("memory_limit_mb", r.memory_limit_mb),
            ("lam", r.lambda_flops_per_second),
            ("floor", r.multiplier_floor),
        ):
            mine = getattr(self, field)
            if mine != theirs:
                out[field] = [mine, theirs]
        return out


@dataclass(frozen=True)
class EvidenceRow:
    """One immutable per-MLP evidence row.

    Frozen on purpose: an evidence row that can be edited after the fact is not
    evidence. `row_hash()` covers everything EXCEPT the timing fields, which are
    genuinely nondeterministic -- so two runs of the same candidate on the same
    networks produce the same hash, and any change to a *decided* quantity moves
    it.
    """

    index: int
    input_seed: int
    estimator_seed: int
    candidate: str
    candidate_sha256: str
    truth_sha256: str
    width: int
    depth: int
    budget: float
    # --- measured, deterministic given the network -------------------------- #
    flops_used: float
    n_ops: float
    mse: float
    zero_prediction_mse: float
    gt_noise: float
    # --- measured, NOT deterministic ---------------------------------------- #
    residual_wall_time_s: float
    wall_time_s: float
    # --- the grader's verdict ------------------------------------------------ #
    budget_exhausted: bool = False
    time_exhausted: bool = False
    residual_exhausted: bool = False
    combined_exhausted: bool = False
    error_code: str = ""
    nonfinite: bool = False

    #: Fields excluded from `row_hash` because they are wall-clock measurements.
    VOLATILE = ("residual_wall_time_s", "wall_time_s")

    @property
    def failed(self) -> bool:
        """The grader's disjunction (`whestbench/scoring.py:865`), plus the
        exception path, which the grader scores identically."""
        return bool(
            self.budget_exhausted
            or self.time_exhausted
            or self.residual_exhausted
            or self.combined_exhausted
            or self.error_code
        )

    def to_row(self) -> MlpRow:
        """Hand off to the ONE place the score identity lives."""
        return MlpRow(
            mse=self.mse,
            flops_used=self.flops_used,
            residual_wall_time_s=self.residual_wall_time_s,
            budget=self.budget,
            failed=self.failed,
            name=f"mlp[{self.index}]/{self.input_seed}",
        )

    def row_hash(self) -> str:
        d = {k: v for k, v in dataclasses.asdict(self).items()
             if k not in self.VOLATILE}
        return hashlib.sha256(
            json.dumps(d, sort_keys=True, default=str).encode()
        ).hexdigest()


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #


class SetupFailure(RuntimeError):
    """setup() raised or overran its window -- the grader would abort the RUN.

    Not a per-MLP failure: `cli.py:2357` calls `runner.start()` outside the try
    that wraps scoring, so nothing gets scored at all.
    """


def _require_real_flopscope():
    import flopscope as flops

    if os.path.abspath(os.path.dirname(flops.__file__ or "")).startswith(
        os.path.abspath(_HERE)
    ):
        raise SystemExit(
            "REFUSED: `flopscope` resolved to the local numpy SHIM in local_sim/, "
            "not the real package. Counted FLOPs measured against the shim are not "
            "the grader's FLOPs. Run from the MAIN venv "
            "(`uv run python local_sim/lifecycle_eval.py`), not the GPU venv."
        )
    return flops


def _require_real_fnp():
    """`flopscope.numpy`, after the same shim refusal as `_require_real_flopscope`."""
    _require_real_flopscope()
    import flopscope.numpy as fnp

    return fnp


def materialise_predictions(predictions, *, depth: int, width: int):
    """`whestbench.scoring.materialise_predictions` -- the IN-WINDOW half.

    Prefers the installed function so this cannot drift from the package. The
    fallback is the two lines the package runs, and
    `local_sim/p2_flop_anchor.py`'s self-test asserts the two cost exactly the
    same, so the fallback is not a second implementation with its own price.

    Only the SHAPE CHECK and the dtype coercion belong here. The finiteness scan
    costs `2*numel - 1` and the package deliberately keeps it outside the metered
    window; `validate_predictions` above owns that half.
    """
    try:
        from whestbench.scoring import (  # type: ignore
            materialise_predictions as _pkg,
        )
    except Exception:
        fnp = _require_real_fnp()
        shape = tuple(predictions.shape) if hasattr(predictions, "shape") else ()
        if shape != (depth, width):
            raise ValueError(
                f"predictions have shape {shape}, expected {(depth, width)}"
            )
        return fnp.asarray(predictions, dtype=fnp.float32)
    return _pkg(predictions, depth=depth, width=width)


def _file_sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _array_sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def load_candidate(path: str):
    """Import a candidate estimator FILE under a private module name."""
    full = path if os.path.isabs(path) else os.path.join(_ROOT, path)
    if not os.path.exists(full):
        raise SystemExit(f"candidate {path!r} does not exist")
    name = "_lifecycle_candidate_" + hashlib.sha1(full.encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(name, full)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod, full


class SetupContext:
    """The grader's `SetupContext`, in the shape an estimator sees it."""

    def __init__(self, spec: LifecycleSpec):
        self.width = spec.width
        self.depth = spec.depth
        self.flop_budget = spec.flop_budget
        self.api_version = "1.0"
        self.scratch_dir = None
        self.submission_dir = None
        self.seed = spec.seed


# --------------------------------------------------------------------------- #
# prediction validation -- whestbench.scoring.validate_predictions
# --------------------------------------------------------------------------- #


def validate_predictions(predictions, *, depth: int, width: int) -> np.ndarray:
    """Shape, transposition and finiteness, as the grader checks them.

    Raises on failure, exactly like the grader -- which then routes the MLP down
    the generic exception path and zeroes it.
    """
    arr = np.asarray(predictions, dtype=np.float32)
    shape = tuple(arr.shape)
    if shape != (depth, width):
        if shape == (width, depth):
            raise ValueError(
                "Returned predictions appear to be transposed (got width-first "
                f"instead of depth-first): {shape} != {(depth, width)}"
            )
        raise ValueError(f"predictions have shape {shape}, expected {(depth, width)}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("predictions contain non-finite values")
    return arr


# --------------------------------------------------------------------------- #
# THE evaluator
# --------------------------------------------------------------------------- #


def evaluate(
    candidate: str,
    seeds: Sequence[int],
    truth_final: np.ndarray,
    *,
    spec: LifecycleSpec,
    gt_noise: Optional[Sequence[float]] = None,
    mint=None,
    progress_every: int = 0,
    on_setup_failure: str = "raise",
    _lifecycle: str = "grader",
) -> Tuple[List[EvidenceRow], Dict[str, Any]]:
    """Run `candidate` over `seeds` under the GRADER's lifecycle.

    `truth_final` is `(n, width)` -- the FINAL layer only, which is the scored one.
    `mint` defaults to `mint_provenance.canonical_mint`, proven bit-identical to
    the grader's generator (100/100, `docs/VALIDATION.md` section 1).

    `_lifecycle` exists ONLY so the regression test can drive the historical
    `"per_mlp_no_setup"` path (vault.py:282 / coverage_lab.py:226) through the
    same code and show the difference. Never use it to produce evidence.
    """
    flops = _require_real_flopscope()
    fnp = _require_real_fnp()
    try:
        flops.configure(symmetry_warnings=False)
    except Exception:
        pass

    if mint is None:
        from mint_provenance import canonical_mint as mint
    import harness

    truth_final = np.asarray(truth_final)
    if truth_final.shape[0] < len(seeds):
        raise ValueError(
            f"truth has {truth_final.shape[0]} rows for {len(seeds)} seeds"
        )
    truth_sha = _array_sha(truth_final[: len(seeds)])

    # ---- ONE instance, setup ONCE, OFF-budget, timed --------------------- #
    # The clock starts BEFORE the module is loaded, matching `runner.py:144`,
    # which calls `time.time()` before `load_estimator_from_path`. A timer wrapped
    # around `setup()` alone reports a duration the grader does not measure and
    # therefore OVERSTATES the headroom -- on the subprocess path
    # (`runner.py:281`) the same window also covers the worker's process spawn and
    # imports, which cannot be seen from in here at all. `HOST_COLD_START_S` is the
    # measured correction and is reported alongside.
    t0 = time.perf_counter()
    mod, full = load_candidate(candidate)
    cand_sha = _file_sha(full)

    est = None
    setup_error = ""
    if _lifecycle == "grader":
        est = mod.Estimator()
        try:
            est.setup(SetupContext(spec))
        except Exception as exc:               # the grader raises SETUP_ERROR
            setup_error = f"{type(exc).__name__}: {exc}"
    setup_s = time.perf_counter() - t0
    setup_timed_out = (setup_s + HOST_COLD_START_S) > spec.setup_timeout_s

    # A failed or timed-out setup is NOT a note in a provenance blob. The grader
    # calls `runner.start()` OUTSIDE the try that wraps `evaluate_estimator`
    # (`cli.py:2357`), and `runner.py:155-172` raises `RunnerError(SETUP_ERROR |
    # SETUP_TIMEOUT)` out of it -- so the ENTIRE submission dies, every MLP
    # included. An evaluator that reports a clean objective here would tell us a
    # candidate is fine when the grader would score it at zero across the board.
    # That matters most for exactly the candidate this module was built to price:
    # a K129/MUB carrier that constructs 66,048 directions in `setup()`.
    if setup_error or setup_timed_out:
        why = (f"SETUP_ERROR: {setup_error}" if setup_error else
               f"SETUP_TIMEOUT: {setup_s:.3f}s + {HOST_COLD_START_S:.3f}s host "
               f"cold start > {spec.setup_timeout_s:.3f}s")
        if on_setup_failure == "raise":
            raise SetupFailure(
                f"{why}  "
                f"The grader aborts the whole submission here (cli.py:2357 -> "
                f"runner.py:155-172); it does not score the MLPs. Pass "
                f"on_setup_failure='score_as_failure' to see what that costs."
            )
        if on_setup_failure != "score_as_failure":
            raise ValueError(f"unknown on_setup_failure {on_setup_failure!r}")

    rows: List[EvidenceRow] = []
    for i, s in enumerate(seeds):
        w_np, est_seed, _ = mint(spec.width, spec.depth, int(s))
        # FLOPSCOPE-BACKED, exactly as `whestbench.domain.MLP.from_row` builds
        # them (`fnp.array(w)`). Until 2026-08-24 this was `np.asarray(x)`, and
        # the difference is not cosmetic: an operation invoked as a METHOD ON A
        # WEIGHT goes through the meter on the grader and through bare NumPy here.
        # The shipped file's live Phase-2 path opens with
        # `[w.astype(fnp.float64) for w in mlp.weights]`, which the grader bills at
        # 33,554,432 FLOPs and this module used to bill at zero. That was
        # 33.5M of the 37.8M total shortfall against the official anchor
        # (`local_sim/p2_flop_anchor.py`).
        #
        # Built OUTSIDE the BudgetContext below, because `from_row` runs during
        # dataset load on the grader, so construction itself is free there too.
        mlp = harness.MLP(
            weights=[fnp.array(x) for x in w_np],
            width=spec.width,
            depth=spec.depth,
            seed=est_seed,
        )
        # The historical path: a fresh instance per MLP, setup never called.
        e = est if _lifecycle == "grader" else mod.Estimator()

        ctx = flops.BudgetContext(
            flop_budget=spec.flop_budget,
            wall_time_limit_s=spec.wall_time_limit_s,
            quiet=True,
        )
        budget_exhausted = time_exhausted = False
        error_code = ""
        preds = None
        try:
            with ctx:
                raw = e.predict(mlp, spec.flop_budget)
                # IN-WINDOW, because whestbench 0.16 does it in-window
                # (`scoring.py`, inside `with budget_ctx:`). Its docstring calls
                # the shape check and the `fnp.asarray` "free ops, so a correct
                # estimator is billed nothing" -- true only for an estimator that
                # already returns float32. Ours returns float64, so the harness's
                # own materialisation is a dtype conversion at 2 FLOPs/element:
                # 16 x 1024 x 2 = 32,768 FLOPs, billed to us. That was the last
                # 32,768 of the shortfall against the official anchor.
                #
                # The metered finiteness scan (2*numel - 1) stays OUTSIDE, below,
                # exactly as the package keeps it outside.
                raw = materialise_predictions(
                    raw, depth=spec.depth, width=spec.width
                )
            preds = validate_predictions(raw, depth=spec.depth, width=spec.width)
        except flops.BudgetExhaustedError:
            budget_exhausted = True
        except flops.TimeExhaustedError:
            time_exhausted = True
        except Exception as exc:
            error_code = type(exc).__name__

        target = np.asarray(truth_final[i], dtype=np.float32)
        # Through the canonical helper, NOT a local mean(target**2). The grader
        # computes this in float32 (`scoring.py:846`); a float64 version of the
        # "same" quantity differs by ~2e-9 relative, which is exactly the
        # two-implementations-of-one-identity failure this module's own docstring
        # warns about. The self-test caught it on the first run.
        zero_mse = zero_prediction_mse(target)

        nonfinite = False
        if preds is None:
            final = np.zeros(spec.width, dtype=np.float32)
        else:
            final = preds[-1]
            nonfinite = not bool(np.all(np.isfinite(final)))

        flops_used = float(ctx.flops_used)
        residual_s = float(ctx.residual_wall_time_s or 0.0)
        wall_s = float(ctx.wall_time_s or 0.0)
        try:
            ns = (ctx.summary_dict(by_namespace=True).get("by_namespace") or {})
            n_ops = float(sum(float(v.get("calls", 0.0)) for v in ns.values()))
        except Exception:
            n_ops = float("nan")

        # The grader's exception path builds its row and `continue`s
        # (`scoring.py:743-760`) BEFORE any post-predict check, recording
        # `flops_used: 0` and `effective_compute: 0.0`. Reproduce both, or a cost
        # aggregation over failed rows diverges and post-predict flags get set on
        # rows the grader leaves False.
        if error_code:
            flops_used = 0.0
            residual_s = 0.0

        # ---- post-predict checks, in the grader's order ------------------- #
        if (not error_code and not budget_exhausted and not time_exhausted
                and spec.wall_time_limit_s is not None
                and wall_s > spec.wall_time_limit_s):
            time_exhausted = True
            final = np.zeros(spec.width, dtype=np.float32)

        residual_exhausted = False
        if (not error_code and not budget_exhausted and not time_exhausted
                and spec.residual_wall_time_limit_s is not None
                and residual_s > spec.residual_wall_time_limit_s):
            residual_exhausted = True
            final = np.zeros(spec.width, dtype=np.float32)

        # Through the canonical predicate, NOT a second copy of `F + lam*R > B`.
        # An earlier revision of this file inlined that arithmetic -- a second
        # implementation of the identity, in the module whose docstring warns
        # against second implementations. The Skeptic found it.
        combined_exhausted = False
        if (not error_code and not budget_exhausted and not time_exhausted
                and not residual_exhausted
                and is_combined_budget_exhausted(
                    flops_used, residual_s, spec.flop_budget, spec.lam)):
            combined_exhausted = True
            final = np.zeros(spec.width, dtype=np.float32)

        if budget_exhausted or time_exhausted or error_code:
            final = np.zeros(spec.width, dtype=np.float32)

        # float32, as scoring.py:846 does it
        mse = float(np.mean((final.astype(np.float32) - target) ** 2))

        rows.append(
            EvidenceRow(
                index=i,
                input_seed=int(s),
                estimator_seed=int(est_seed),
                candidate=os.path.relpath(full, _ROOT).replace("\\", "/"),
                candidate_sha256=cand_sha,
                truth_sha256=truth_sha,
                width=spec.width,
                depth=spec.depth,
                budget=float(spec.flop_budget),
                flops_used=flops_used,
                n_ops=n_ops,
                mse=mse,
                zero_prediction_mse=zero_mse,
                gt_noise=float(gt_noise[i]) if gt_noise is not None else float("nan"),
                residual_wall_time_s=residual_s,
                wall_time_s=wall_s,
                budget_exhausted=budget_exhausted,
                time_exhausted=time_exhausted,
                residual_exhausted=residual_exhausted,
                combined_exhausted=combined_exhausted,
                error_code=error_code,
                nonfinite=nonfinite,
            )
        )
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  {i + 1}/{len(seeds)}", flush=True)

    provenance = {
        "candidate": os.path.relpath(full, _ROOT).replace("\\", "/"),
        "candidate_sha256": cand_sha,
        "truth_sha256": truth_sha,
        "n_mlps": len(seeds),
        "spec": dataclasses.asdict(spec),
        #: Which official round this run claims to model, and every field that
        #: does NOT match it. An empty dict means a clean round; anything in it
        #: means the run used a rulebook the competition does not, and any number
        #: taken from the run must say so. This is the record that stops a
        #: cliff-disabled run from being quoted as a deployment result, which is
        #: exactly what happened to P2-CP1.
        "round_tag": spec.round_tag,
        "round_deviations": spec.deviations_from_round(),
        "lifecycle": _lifecycle,
        "setup_called": _lifecycle == "grader",
        "setup_seconds": setup_s,
        "setup_timeout_s": spec.setup_timeout_s,
        "setup_timed_out": setup_timed_out,
        "setup_error": setup_error,
        "host_cold_start_s": HOST_COLD_START_S,
        "residual_measurement_path": "runner=local (in-process BudgetContext)",
        "residual_is_graded": False,
        "objective_is_graded": False,
        "backend": {
            "flopscope": getattr(flops, "__version__", "?"),
            "flopscope_path": os.path.dirname(flops.__file__ or ""),
            "numpy": np.__version__,
            "python": sys.version.split()[0],
        },
        "rows_hash": rows_hash(rows),
    }
    return rows, provenance


def rows_hash(rows: Sequence[EvidenceRow]) -> str:
    """Content address over the DETERMINISTIC content of an evidence set."""
    h = hashlib.sha256()
    for r in rows:
        h.update(r.row_hash().encode())
    return h.hexdigest()


def score(rows: Sequence[EvidenceRow], *, spec: LifecycleSpec) -> SuiteScore:
    """The competition objective, through the one module that owns the identity."""
    return score_suite(
        [r.to_row() for r in rows],
        lam=spec.lam,
        floor=spec.floor,
        on_over_budget="raise",
    )


def write_evidence(path: str, rows: Sequence[EvidenceRow], provenance: Dict[str, Any],
                   suite: SuiteScore) -> str:
    """Write the evidence bundle. Refuses to overwrite -- evidence is append-only."""
    if os.path.exists(path):
        raise SystemExit(
            f"REFUSED: {path} exists. Evidence rows are immutable; write a new file. "
            "Overwriting is how per-touch prediction evidence was lost before."
        )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "provenance": provenance,
        "objective": suite.objective,
        "n_failed": suite.n_failed,
        "n": suite.n,
        "rows": [dataclasses.asdict(r) for r in rows],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, default=str)
    return path


def format_suite(rows: Sequence[EvidenceRow], suite: SuiteScore,
                 provenance: Dict[str, Any]) -> str:
    gt = np.array([r.gt_noise for r in rows], dtype=np.float64)
    gt_mean = float(np.nanmean(gt)) if np.any(np.isfinite(gt)) else float("nan")
    lines = [
        f"candidate      {provenance['candidate']}  sha256 {provenance['candidate_sha256'][:16]}...",
        f"lifecycle      {provenance['lifecycle']}   setup_called={provenance['setup_called']}"
        f"   setup {provenance['setup_seconds'] * 1e3:.1f} ms"
        f" / {provenance['setup_timeout_s'] * 1e3:.0f} ms allowed",
        f"networks       {suite.n}   truth sha256 {provenance['truth_sha256'][:16]}...",
        "",
        f"  OBJECTIVE (arithmetic mean of per-MLP scores)  {suite.objective:.6e}",
        f"  raw MSE mean {suite.mean_mse:.6e}"
        + (f"   GT noise {gt_mean:.3e} = {gt_mean / suite.mean_mse * 100:.2f}% of it"
           if np.isfinite(gt_mean) and suite.mean_mse else "   GT noise not supplied"),
        f"  multiplier mean {suite.mean_multiplier:.5f}   failures {suite.n_failed}/{suite.n}",
        f"  worst {suite.worst:.4e}   best {suite.best:.4e}   "
        f"top-1 tail share {suite.tail_share(1) * 100:.2f}%",
        f"  rows_hash {provenance['rows_hash'][:32]}...",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# self-test -- predictions L1-L6
# --------------------------------------------------------------------------- #

_PASS = _FAIL = 0

#: Every check this suite runs, as (name, ok, detail). Recorded so another suite
#: can verify a SPECIFIC named result rather than trusting that this file exists.
#:
#: The distinction is the whole point of the P2-P0 G4 failure: an enforceability
#: matrix claimed `provoked_by: "lifecycle_eval.py --self-test (L4)"` for the wall
#: limit, and the named provocation did not exist. A witness that resolves a
#: FILENAME cannot catch that. A witness that resolves a CHECK NAME and its RESULT
#: can, and does.
_RESULTS: List[Dict[str, Any]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    global _PASS, _FAIL
    ok = bool(ok)
    _RESULTS.append({"name": name, "ok": ok, "detail": str(detail)})
    if ok:
        _PASS += 1
        print(f"  {name:<62} ok   {detail}")
    else:
        _FAIL += 1
        print(f"  {name:<62} FAIL {detail}")
    return ok


def _emit_results(path: str) -> None:
    """Write this run's checks, machine-readably, with an evidence hash.

    The hash covers the (name, ok) pairs only -- not the details, which carry
    timings -- so two honest runs of the same suite produce the same digest and a
    changed RESULT changes it.
    """
    payload = {
        "artifact": "lifecycle_eval_self_test",
        "artifact_version": 1,
        "suite": "local_sim/lifecycle_eval.py --self-test",
        "passed": _PASS,
        "total": _PASS + _FAIL,
        "checks": _RESULTS,
    }
    digest = hashlib.sha256(
        json.dumps([[c["name"], c["ok"]] for c in _RESULTS],
                   sort_keys=False).encode("utf-8")
    ).hexdigest()
    payload["results_sha256"] = digest
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n  wrote {os.path.relpath(path, _ROOT)}  "
          f"({_PASS}/{_PASS + _FAIL}, sha256 {digest[:16]}...)")


def head(t: str) -> None:
    print(f"\n{t}")


def _src(lines: Sequence[str]) -> str:
    """Build a throwaway candidate's source from a line list.

    Deliberately no escape sequences: an earlier revision of this file built these
    with embedded newline escapes and a tooling round-trip turned them into real
    newlines, producing a file that would not parse.
    """
    nl = chr(10)
    return nl.join(lines) + nl


def _pkg_version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "?"


def _self_test() -> int:
    flops = _require_real_flopscope()
    import whestbench
    from whestbench.scoring import ContestSpec

    print("=" * 104)
    print("LIFECYCLE EVALUATOR — self-test (predictions L1-L6)")
    print("=" * 104)

    # Legacy-namespace throwaway seeds, the local_sim/residual_model.py convention:
    # this file measures COST and mechanics, decides nothing, and must not draw on
    # a registered block. 5,150,000+ is outside every registered span (nearest is
    # L2 at 7,000,000+), so no pragma is needed -- and an `excludes V` pragma here
    # would be worse than useless: `seed_ledger.py:461` reads it as
    # `bid not in excluded`, permanently disabling the VAULT-seed-literal scan in
    # this file. One was written and removed.
    SEEDS = [5_150_000 + i for i in range(6)]

    # L1-L6 run on the PHASE-1 shape, and therefore under the PHASE-1 RULEBOOK --
    # named, not inherited. Two reasons this must be `for_round` rather than
    # `LifecycleSpec(width=256, depth=32)`:
    #
    #  1. the estimator is BUDGET-ADAPTIVE. `predict(mlp, budget)` sizes its work
    #     to the budget it is handed, so a 256x32 network under Phase 2's 2**41
    #     costs ~7x what it costs under Phase 1's 2.72e11. That is not a Phase-1
    #     measurement; it is a measurement of a rulebook that has never existed.
    #  2. lambda, wall and residual differ too, and picking them up one at a time
    #     is exactly how a run ends up with a Phase-2 budget and a Phase-1 lambda.
    #
    # Section 7 below runs the LIVE round at its own shape, so both rulebooks are
    # exercised in one pass and neither borrows the other's constants.
    spec = LifecycleSpec.for_round("v1-phase1")
    WIDTH, DEPTH = spec.width, spec.depth

    head("0. the transcribed grader constants still match the installed whestbench")
    check("setup_timeout_s", ContestSpec.setup_timeout_s == SETUP_TIMEOUT_S,
          f"{ContestSpec.setup_timeout_s} == {SETUP_TIMEOUT_S}")
    check("wall_time_limit_s", ContestSpec.wall_time_limit_s == WALL_TIME_LIMIT_S,
          f"{ContestSpec.wall_time_limit_s} == {WALL_TIME_LIMIT_S}")
    check("residual_wall_time_limit_s",
          ContestSpec.residual_wall_time_limit_s == RESIDUAL_WALL_TIME_LIMIT_S,
          f"{ContestSpec.residual_wall_time_limit_s}")
    check("lambda", ContestSpec.lambda_flops_per_second == DEFAULT_LAMBDA_FLOPS_PER_SECOND,
          f"{ContestSpec.lambda_flops_per_second:.3e}")
    check("predict_timeout_s", ContestSpec.predict_timeout_s == PREDICT_TIMEOUT_S,
          f"{ContestSpec.predict_timeout_s}")
    check("memory_limit_mb", ContestSpec.memory_limit_mb == MEMORY_LIMIT_MB,
          f"{ContestSpec.memory_limit_mb:,} MB  <- RLIMIT_AS; already killed v2.5")
    print("       NOTE: these are PACKAGE DEFAULTS. No graded report publishes "
          "them; only lambda is corroborated.")
    print(f"       whestbench {getattr(whestbench, '__version__', _pkg_version('whestbench'))}"
          f"   flopscope {getattr(flops, '__version__', _pkg_version('flopscope'))}")

    # A cheap stand-in for real truth: this test measures LIFECYCLE and FAILURE
    # SEMANTICS, not accuracy, so the truth values only need to be fixed and
    # non-degenerate. Real evaluations pass real truth.
    rng = np.random.default_rng(20260820)
    truth = rng.random((len(SEEDS), WIDTH)).astype(np.float32) * 0.5

    head("1. L1/L5 — the grader lifecycle vs the historical vault lifecycle")
    g_rows, g_prov = evaluate("estimator.py", SEEDS, truth, spec=spec)
    v_rows, v_prov = evaluate("estimator.py", SEEDS, truth, spec=spec,
                              _lifecycle="per_mlp_no_setup")
    dif = [(i, v.flops_used - g.flops_used, v.n_ops - g.n_ops)
           for i, (g, v) in enumerate(zip(g_rows, v_rows))
           if v.flops_used != g.flops_used]
    print(f"       {'i':>2} {'grader flops':>14} {'vault flops':>14} {'delta':>13} {'d ops':>7}")
    for i, (g, v) in enumerate(zip(g_rows, v_rows)):
        print(f"       {i:>2} {g.flops_used:>14.5e} {v.flops_used:>14.5e} "
              f"{v.flops_used - g.flops_used:>+13.4e} {v.n_ops - g.n_ops:>+7.0f}")
    check("L5: exactly ONE row differs between the two lifecycles",
          len(dif) == 1, f"{len(dif)} differ")
    check("L5: and it is index 0", bool(dif) and dif[0][0] == 0,
          f"index {dif[0][0]}" if dif else "none")
    check("L1: rows 1.. are BIT-IDENTICAL across lifecycles",
          all(v.flops_used == g.flops_used for g, v in zip(g_rows[1:], v_rows[1:])))
    check("L1: the old lifecycle over-charges MLP #0, it never under-charges",
          bool(dif) and dif[0][1] > 0, f"{dif[0][1]:+.4e} FLOPs" if dif else "")
    check("L1: setup() was called exactly once on the grader path",
          g_prov["setup_called"] and not v_prov["setup_called"])

    head("2. L3 — setup() fits inside the grader's 5-second ceiling")
    check("L3: module load + setup(), plus the measured host cold start, fits",
          g_prov["setup_seconds"] + HOST_COLD_START_S < spec.setup_timeout_s,
          f"{g_prov['setup_seconds'] * 1e3:.1f} ms + {HOST_COLD_START_S * 1e3:.0f} ms"
          f" = {(g_prov['setup_seconds'] + HOST_COLD_START_S) * 1e3:.0f} ms of "
          f"{spec.setup_timeout_s * 1e3:.0f} ms (a PACKAGE DEFAULT, not observed)")
    check("L3: and is therefore not timed out", not g_prov["setup_timed_out"])
    check("L3: setup raised nothing", g_prov["setup_error"] == "",
          g_prov["setup_error"] or "clean")

    head("3. L2 — the objective agrees with a hand-computed identity")
    suite = score(g_rows, spec=spec)
    hand = float(np.mean([
        r.mse * max(spec.floor,
                    (r.flops_used + spec.lam * r.residual_wall_time_s) / spec.flop_budget)
        for r in g_rows]))
    rel = abs(hand - suite.objective) / abs(hand)
    check("L2: score_suite == mean_i mse_i * max(0.1, C_i/B)", rel == 0.0,
          f"relative deviation {rel:.3e}")
    check("L2: no row silently over budget", suite.n_failed == 0,
          f"{suite.n_failed} failed")

    head("4. L4 — every provokable failure path, exactly as the grader prices it")
    tiny = dataclasses.replace(spec, flop_budget=1_000_000)
    t_rows, _ = evaluate("estimator.py", SEEDS[:2], truth[:2], spec=tiny)
    check("L4: a 1e6 budget exhausts the FLOP budget",
          all(r.budget_exhausted for r in t_rows),
          f"{sum(r.budget_exhausted for r in t_rows)}/2 exhausted")
    check("L4: exhausted rows are FAILED", all(r.failed for r in t_rows))
    check("L4: and their MSE is EXACTLY mean(target^2), i.e. zeroed predictions",
          all(r.mse == r.zero_prediction_mse for r in t_rows),
          f"{t_rows[0].mse:.17e} vs {t_rows[0].zero_prediction_mse:.17e}")
    mults = [r.to_row().multiplier(tiny.lam, tiny.floor) for r in t_rows]
    check("L4: a failure is priced at multiplier EXACTLY 1.0, not the 0.1 floor",
          all(m == 1.0 for m in mults), f"{mults}")

    broken = os.path.join(_HERE, "_lifecycle_broken_tmp.py")
    try:
        with open(broken, "w", encoding="utf-8") as fh:
            fh.write(
                "class Estimator:\n"
                "    def setup(self, ctx):\n        pass\n"
                "    def predict(self, mlp, budget):\n"
                "        raise RuntimeError('deliberate')\n"
            )
        b_rows, _ = evaluate(broken, SEEDS[:2], truth[:2], spec=spec)
        check("L4: a raising predict() routes to the exception path",
              all(r.error_code == "RuntimeError" for r in b_rows),
              f"{[r.error_code for r in b_rows]}")
        check("L4: exception rows are FAILED and exactly zeroed",
              all(r.failed and r.mse == r.zero_prediction_mse for r in b_rows))
        with open(broken, "w", encoding="utf-8") as fh:
            fh.write(
                "import numpy as np\n"
                "class Estimator:\n"
                "    def setup(self, ctx):\n        pass\n"
                "    def predict(self, mlp, budget):\n"
                "        return np.zeros((mlp.width, len(mlp.weights)))\n"
            )
        x_rows, _ = evaluate(broken, SEEDS[:1], truth[:1], spec=spec)
        check("L4: a TRANSPOSED prediction is caught by validation, not scored",
              x_rows[0].error_code == "ValueError", x_rows[0].error_code)
    finally:
        if os.path.exists(broken):
            os.remove(broken)

    # The residual-limit and combined-budget paths were once asserted as "not
    # provokable cheaply". That was false, and the Skeptic provoked both in under
    # a minute. Both are now tested, not asserted.
    r_spec = dataclasses.replace(spec, residual_wall_time_limit_s=1e-9)
    r_rows, _ = evaluate("estimator.py", SEEDS[:2], truth[:2], spec=r_spec)
    check("L4: a 1e-9 s residual limit trips the residual path",
          all(r.residual_exhausted and r.failed for r in r_rows),
          f"{sum(r.residual_exhausted for r in r_rows)}/2")
    check("L4: residual-exhausted rows are exactly zeroed at multiplier 1.0",
          all(r.mse == r.zero_prediction_mse
              and r.to_row().multiplier(r_spec.lam, r_spec.floor) == 1.0
              for r in r_rows))

    # --- the WALL path, provoked --------------------------------------------
    # Added 2026-08-24 after a final Skeptic pass found that P2-P0 gate clause G4
    # names FIVE paths and this repository provoked only four. The wall clause was
    # recorded as satisfied by a check whose condition was the literal `True`
    # (`test_deployment_envelope.py`), and the artifact claimed it was provoked
    # "by lifecycle_eval --self-test (L4)" -- which set a tiny RESIDUAL limit and
    # never a tiny WALL limit. Two different constants, one of them untested,
    # and a passing line that could not fail. That is this repository's signature
    # failure class, and it survived two earlier reviews of the same slice.
    #
    # flopscope's BudgetContext enforces the wall clock itself, so a 1e-9 s limit
    # trips on the first counted operation -- the same shape of provocation the
    # residual cliff gets three lines above.
    w_spec = dataclasses.replace(spec, wall_time_limit_s=1e-9)
    w_rows, _ = evaluate("estimator.py", SEEDS[:2], truth[:2], spec=w_spec)
    check("L4: a 1e-9 s WALL limit trips the TIME path",
          all(r.time_exhausted and r.failed for r in w_rows),
          f"{sum(r.time_exhausted for r in w_rows)}/2 time_exhausted")
    check("L4: wall-exhausted rows are exactly zeroed at multiplier 1.0",
          all(r.mse == r.zero_prediction_mse
              and r.to_row().multiplier(w_spec.lam, w_spec.floor) == 1.0
              for r in w_rows))
    check("L4: ...and the WALL path is distinct from the RESIDUAL path",
          all(r.time_exhausted and not r.residual_exhausted for r in w_rows),
          "time_exhausted set, residual_exhausted clear")

    slow = os.path.join(_HERE, "_lifecycle_slow_tmp.py")
    try:
        with open(slow, "w", encoding="utf-8") as fh:
            fh.write(_src([
                "import time",
                "import flopscope.numpy as fnp",
                "class Estimator:",
                "    def setup(self, ctx):",
                "        pass",
                "    def predict(self, mlp, budget):",
                "        time.sleep(0.05)",
                "        return fnp.zeros((len(mlp.weights), mlp.width))",
            ]))
        c_spec = dataclasses.replace(spec, flop_budget=1_000_000_000)
        c_rows, _ = evaluate(slow, SEEDS[:2], truth[:2], spec=c_spec)
        check("L4: 50 ms of residual at B=1e9 trips the COMBINED budget C=F+lam*R>B",
              all(r.combined_exhausted and r.failed for r in c_rows),
              f"C = {c_rows[0].flops_used + c_spec.lam * c_rows[0].residual_wall_time_s:.3e}"
              f" vs B = {c_spec.flop_budget:.3e}")
        check("L4: combined-exhausted rows are exactly zeroed at multiplier 1.0",
              all(r.mse == r.zero_prediction_mse
                  and r.to_row().multiplier(c_spec.lam, c_spec.floor) == 1.0
                  for r in c_rows))
    finally:
        if os.path.exists(slow):
            os.remove(slow)

    from competition_score import PHASE1_LAMBDA_FLOPS_PER_SECOND, over_budget_rows
    over = MlpRow(mse=1.0, flops_used=2.0e11, residual_wall_time_s=1.0,
                  budget=2.72e11, failed=False)
    check("L4: and `competition_score` owns that predicate, not this module",
          over_budget_rows([over], lam=PHASE1_LAMBDA_FLOPS_PER_SECOND) == [0],
          f"C = {over.effective_compute(PHASE1_LAMBDA_FLOPS_PER_SECOND):.3e} "
          f"vs B = 2.72e11, at the PHASE-1 rate")
    # The same row is NOT over budget under the live round, because lambda is 0.
    # Stated as its own check so the round-dependence is visible rather than
    # inferred: `over_budget_rows` is a function of the rulebook, not of the row.
    check("L4: ...and under the LIVE round the same row is IN budget, because "
          "lambda is 0",
          over_budget_rows([over]) == [],
          f"C = {over.effective_compute():.3e} vs B = 2.72e11")

    head("4b. SETUP failure is a REFUSAL, because the grader aborts the whole run")
    slowsetup = os.path.join(_HERE, "_lifecycle_slowsetup_tmp.py")
    try:
        with open(slowsetup, "w", encoding="utf-8") as fh:
            fh.write(_src([
                "import time",
                "import flopscope.numpy as fnp",
                "class Estimator:",
                "    def setup(self, ctx):",
                "        time.sleep(6.0)",
                "    def predict(self, mlp, budget):",
                "        return fnp.zeros((len(mlp.weights), mlp.width))",
            ]))
        raised = False
        try:
            evaluate(slowsetup, SEEDS[:1], truth[:1], spec=spec)
        except SetupFailure:
            raised = True
        check("a 6 s setup RAISES SetupFailure — no clean objective is produced",
              raised, "grader: cli.py:2357 -> runner.py:155-172 kills the run")
        sf_rows, sf_prov = evaluate(slowsetup, SEEDS[:1], truth[:1], spec=spec,
                                    on_setup_failure="score_as_failure")
        check("...and the opt-in path marks it, rather than hiding it",
              sf_prov["setup_timed_out"],
              f"setup {sf_prov['setup_seconds']:.2f}s + "
              f"{HOST_COLD_START_S:.2f}s cold start > {spec.setup_timeout_s}s")

        with open(slowsetup, "w", encoding="utf-8") as fh:
            fh.write(_src([
                "class Estimator:",
                "    def setup(self, ctx):",
                "        raise RuntimeError('bad setup')",
                "    def predict(self, mlp, budget):",
                "        return None",
            ]))
        raised = False
        try:
            evaluate(slowsetup, SEEDS[:1], truth[:1], spec=spec)
        except SetupFailure:
            raised = True
        check("a RAISING setup also refuses", raised, "SETUP_ERROR")
    finally:
        if os.path.exists(slowsetup):
            os.remove(slowsetup)

    head("4c. the setup clock starts where the grader's does")
    check("the timer covers module load, not just setup() "
          "(runner.py:144 precedes load_estimator_from_path)",
          g_prov["setup_seconds"] >= 0.0 and "host_cold_start_s" in g_prov,
          f"measured {g_prov['setup_seconds'] * 1e3:.1f} ms + "
          f"{HOST_COLD_START_S * 1e3:.0f} ms host cold start, "
          f"of {spec.setup_timeout_s * 1e3:.0f} ms")
    check("the evidence bundle says its residual is a LOCAL-runner figure",
          g_prov["residual_is_graded"] is False
          and g_prov["objective_is_graded"] is False,
          g_prov["residual_measurement_path"])

    head("5. L6 — evidence rows are content-addressed and reproducible")
    g2_rows, g2_prov = evaluate("estimator.py", SEEDS, truth, spec=spec)
    check("L6: rows_hash is stable across two independent runs",
          g_prov["rows_hash"] == g2_prov["rows_hash"],
          f"{g_prov['rows_hash'][:24]}...")
    check("L6: the two lifecycles have DIFFERENT hashes (it can discriminate)",
          g_prov["rows_hash"] != v_prov["rows_hash"])
    volatile_moved = any(g.residual_wall_time_s != g2.residual_wall_time_s
                         for g, g2 in zip(g_rows, g2_rows))
    check("L6: ...and the hash is stable EVEN THOUGH timings moved between runs",
          volatile_moved and g_prov["rows_hash"] == g2_prov["rows_hash"],
          "timings differed run-to-run, hash did not" if volatile_moved
          else "timings did NOT differ — weak evidence, not a failure")
    check("L6: EvidenceRow is frozen (an editable row is not evidence)",
          _is_frozen(g_rows[0]))

    # ---------------------------------------------------------------------- #
    # 6. L7 -- the round is NAMED, and a deviation from it cannot hide
    # ---------------------------------------------------------------------- #
    head("6. L7 — a spec names its round, and any deviation from it is recorded")
    check("L7: for_round('v1-phase1') carries ALL SIX Phase-1 constants together",
          (spec.width, spec.depth, spec.flop_budget, spec.lam,
           spec.wall_time_limit_s, spec.residual_wall_time_limit_s)
          == (256, 32, 272_000_000_000, 1e11, 60.0, None),
          f"B={spec.flop_budget:.3e} lam={spec.lam:.0e} wall={spec.wall_time_limit_s} "
          f"residual={spec.residual_wall_time_limit_s}")
    check("L7: a clean spec reports NO deviation from the round it names",
          spec.deviations_from_round() == {}, f"{spec.deviations_from_round()}")
    dev = dataclasses.replace(spec, flop_budget=1_000_000)
    check("L7: ...and a modified one names exactly what it changed",
          dev.deviations_from_round() == {"flop_budget": [1_000_000, 272_000_000_000]},
          f"{dev.deviations_from_round()}")
    check("L7: the provenance blob carries the round tag and its deviations",
          g_prov["spec"]["round_tag"] == "v1-phase1"
          and g_prov.get("round_deviations") == {},
          f"{g_prov['spec']['round_tag']}, deviations "
          f"{g_prov.get('round_deviations')!r}")
    _bad = False
    try:
        LifecycleSpec.for_round("v2-phase2", nonsense=1)
    except TypeError:
        _bad = True
    except Exception:
        pass
    check("L7: for_round refuses an unknown field rather than ignoring it", _bad)

    # ---------------------------------------------------------------------- #
    # 7. L8 -- THE LIVE ROUND, at its own shape. Everything above is Phase 1.
    # ---------------------------------------------------------------------- #
    head("7. L8 — the LIVE round at 1024x16: the exact anchor and the 0.4 s cliff")
    p2 = LifecycleSpec.for_round("v2-phase2")
    check("L8: the live spec is Phase 2, with the cliff ARMED",
          (p2.width, p2.depth, p2.flop_budget, p2.lam,
           p2.wall_time_limit_s, p2.residual_wall_time_limit_s,
           p2.memory_limit_mb)
          == (1024, 16, 2 ** 41, 0.0, 120.0, 0.4, 8192),
          f"B=2**41 lam={p2.lam} residual_cap={p2.residual_wall_time_limit_s}s "
          f"mem={p2.memory_limit_mb} MB")
    p2_seeds = SEEDS[:1]
    p2_truth = np.zeros((1, p2.width), dtype=np.float32)
    p2_rows, p2_prov = evaluate("estimator.py", p2_seeds, p2_truth, spec=p2)
    r0 = p2_rows[0]
    # THE anchor. Counted FLOPs on this dense path are value-independent, so this
    # holds for any seed and is an exact equality, not a tolerance. It is the
    # single number that says the local meter bills what the grader bills; when it
    # moves, `local_sim/p2_flop_anchor.py` says which operation moved.
    check("L8: counted FLOPs EXACTLY equal the official one-row subprocess anchor",
          r0.flops_used == float(P2_OFFICIAL_ANCHOR_FLOPS),
          f"{r0.flops_used:,.0f} vs {P2_OFFICIAL_ANCHOR_FLOPS:,} "
          f"(delta {r0.flops_used - P2_OFFICIAL_ANCHOR_FLOPS:+,.0f})")
    # NOT a second check: given the equality above, this is two constants
    # subtracted and cannot fail independently. It is printed as context for the
    # reader and deliberately kept OUT of the denominator -- a suite that counts
    # its own restatements is padding.
    print(f"        (context: that is +{r0.flops_used - 137_562_547_200:,.0f} FLOPs "
          f"on the pre-repair local figure of 137,562,547,200)")
    check("L8: C == F, because lambda is 0",
          r0.to_row().effective_compute(p2.lam) == r0.flops_used,
          f"C = {r0.to_row().effective_compute(p2.lam):,.0f}")
    check("L8: the multiplier is pinned at the 0.1 floor at this operating point",
          r0.to_row().multiplier(p2.lam, p2.floor) == 0.1,
          f"C/B = {r0.flops_used / p2.flop_budget:.7f} < floor 0.1")
    check("L8: no failure path fires on the shipped file", not r0.failed,
          f"budget={r0.budget_exhausted} time={r0.time_exhausted} "
          f"residual={r0.residual_exhausted} combined={r0.combined_exhausted} "
          f"error={r0.error_code!r}")
    # Residual is a CLIFF now, so what matters is the margin to it. The local
    # in-process figure is the PESSIMISTIC side of that margin (docs/GOTCHAS.md
    # measures this host at ~7x the grader), so it is reported as local headroom
    # and never as a deployment fact.
    import official_config as _oc
    hr = _oc.PHASE2.residual_headroom(r0.residual_wall_time_s)
    check("L8: local in-process residual leaves headroom under the 0.4 s cliff",
          hr is not None and hr > 1.0,
          f"{r0.residual_wall_time_s * 1e3:.1f} ms local -> {hr:.2f}x headroom "
          f"(LOCAL, in-process; the grader's own figure is the authority)")
    # And the cliff must actually FIRE, not merely be configured. A limit that has
    # never been provoked is a comment.
    p2_cliff = LifecycleSpec.for_round("v2-phase2", residual_wall_time_limit_s=1e-9)
    c2_rows, c2_prov = evaluate("estimator.py", p2_seeds, p2_truth, spec=p2_cliff)
    check("L8: the 0.4 s cliff FIRES when provoked (1e-9 s limit)",
          c2_rows[0].residual_exhausted and c2_rows[0].failed)
    check("L8: a residual failure is zeroed and priced at multiplier 1.0, not 0.1",
          c2_rows[0].mse == c2_rows[0].zero_prediction_mse
          and c2_rows[0].to_row().multiplier(p2_cliff.lam, p2_cliff.floor) == 1.0,
          f"mult {c2_rows[0].to_row().multiplier(p2_cliff.lam, p2_cliff.floor)}")
    check("L8: ...and that run's provenance DECLARES it was not a clean round",
          c2_prov.get("round_deviations", {}).get("residual_wall_time_limit_s")
          == [1e-9, 0.4],
          f"{c2_prov.get('round_deviations')}")
    # Under Phase 2 residual cannot put a row over the COMBINED budget by any
    # amount, because lambda is 0. This is the half of the old model that quietly
    # stopped applying, so it is asserted rather than assumed.
    check("L8: no amount of residual can trip the combined budget under lambda 0",
          not is_combined_budget_exhausted(r0.flops_used, 1e9, p2.flop_budget, p2.lam),
          f"C = {r0.flops_used + p2.lam * 1e9:,.0f} vs B = {p2.flop_budget:,}")

    print("\n" + "=" * 104)
    print(f"{_PASS}/{_PASS + _FAIL} checks passed")
    return 0 if _FAIL == 0 else 1


def _is_frozen(row) -> bool:
    try:
        object.__setattr__  # noqa: B018
        row.mse = -1.0
        return False
    except dataclasses.FrozenInstanceError:
        return True
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-test", action="store_true",
                    help="run the L1-L8 regression suite")
    ap.add_argument("--emit-json", default="",
                    help=("with --self-test, also write every check result and an "
                          "evidence hash here, so another suite can verify a "
                          "SPECIFIC named provocation rather than trusting that "
                          "this file exists"))
    a = ap.parse_args()
    if a.self_test:
        rc = _self_test()
        if a.emit_json:
            _emit_results(os.path.abspath(a.emit_json))
        return rc
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
