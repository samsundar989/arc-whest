"""Certify a CANDIDATE estimator file, read-only, without touching the shipped one.

Contract: `experiments/LOG.md`, *CONTRACT WP0C+WP0D* (2026-08-31), committed at
`71bf62b` before this file existed. `GAME_PLAN.md` § WP0.3.

WHY THIS EXISTS
---------------
`scripts/preflight.ps1` hardcodes `estimator.py` in steps 4, 12, 13, 15 and 16;
only `-Archive` is a parameter, and it selects what step 5 compares *against*.
So **a green 16/16 certifies v2.6.1, whatever candidate is under development** —
the defect that made gate item 6 of the Track-0S submission contract
unsatisfiable for v2.7, with two honest exits and no third: promote the
candidate first, or parameterise the gate.

This is the third exit the contract could not see at the time: a **separate
read-only command** that certifies an arbitrary file. It adds no path through
`preflight.ps1`, writes nothing the gate reads, and therefore cannot weaken the
ship-file gate. `preflight.ps1` still certifies `estimator.py`, unchanged.

`--candidate` is REQUIRED and has no default. A default of `estimator.py` would
let a certification of the shipped file be mistaken for a certification of a
candidate, which is precisely the confusion this command exists to remove.

WHAT IT DOES NOT DO
-------------------
* **Sections 1-9 read no truth.** Where the lifecycle evaluator needs a
  reference it is handed **zeros**, so its `mse` column is `mean(prediction^2)`
  — a magnitude, not an error.
* **SECTION 10 DOES READ TRUTH, and the first version of this docstring denied
  it.** The official subprocess smoke runs the organisers' `v2-phase2` mini
  split; its `final_layer_mse_CONFIRM_ONLY` is a real error against real labels.
  That is legitimate **CONFIRM-tier** use — a deployment check — but CONFIRM is
  read-only and may CHOOSE nothing. The report's `truth_free` field is now
  COMPUTED from whether section 10 ran, and `confirm_tier_read` records it
  positively. Use `--skip-smoke` for a genuinely truth-free run.
* **It is not a promotion.** `GAME_PLAN.md` § 6 lists what promotion requires;
  this is one input to it, not a substitute. It does not run `preflight -Full`,
  it does not touch a vault or a seed block, and it does not submit.
* **It cannot enforce 8 GiB on Windows.** Section 9 is *accounting* here and
  says so; `local_sim/linux_memory_gate.py` is the enforcement, and section 9
  reads that gate's report if one exists rather than guessing.

Run:
    uv run python local_sim/certify_candidate.py --candidate estimator.py
    uv run python local_sim/certify_candidate.py --candidate estimator_v2_7.py \
        --json experiments/<slice>/certify_v27.json
    uv run python local_sim/certify_candidate.py --candidate <f> --quick   # skip 1, 10
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import math
import json
import os
import subprocess
import sys
import tempfile
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# local_sim ships numpy shims NAMED flopscope/whestbench: APPEND, never insert.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]

import numpy as np  # noqa: E402

WIDTH, DEPTH = 1024, 16
BUDGET = 2 ** 41
FLOOR = 0.1
#: Seeds OUTSIDE every registered block (`local_sim/seed_blocks.py`). This
#: command measures deployment properties, never accuracy, so it claims no
#: block and reads no truth -- and typing block seeds here would trip
#: `test_seed_ledger.py`'s source scan, as it once did in `test_v27_parity.py`.
SEEDS = (90000000201, 90000000202, 90000000203, 90000000204)

RESIDUAL_CAP_S = 0.4
WALL_CAP_S = 120.0
SETUP_CAP_S = 5.0
MEMORY_CAP_MB = 8192
SINGLE_ARRAY_CAP = 4 * (1 << 30)

#: THE GRADER-TWIN ARM. Mirrors `lifecycle_eval.WEIGHTS_MODES`; declared here so
#: argparse can offer the choices before `lifecycle_eval` is imported.
_TWIN_MODES = ("f32", "f64", "f64_readonly", "f64_listconstructed")
_TWIN_DEFAULT = "f64_readonly"

# Must match ``edge_cases.GROUP_REGISTRY``.  Keeping the consumer's required
# coverage here prevents the producer from certifying an accidental omission in
# its own registry.
EDGE_GROUP_REGISTRY = (
    ("normal", 1), ("degenerate", 3), ("dynamic_range", 4),
    ("seed_extremes", 5), ("shape_generality", 5), ("budget_edges", 5),
    ("immutability", 1), ("layer1", 1), ("permutation", 1),
    ("budget_monotone", 1), ("determinism", 1), ("seed_hygiene", 1),
    ("gt_noise", 1),
)
EDGE_GROUP_RECEIPT_VERSION = 1

_FAILED = []
_SKIPPED = []
_TOTAL = [0]
_SECTIONS = {}
_CURRENT = ["?"]

# --------------------------------------------------------------------------- #
# FOUR KINDS OF EVIDENCE, WHICH ONE VERDICT USED TO CONFLATE
#
# `verdict` is one word over checks from three genuinely different domains:
#
#   MATH      the contract and the arithmetic. Deterministic, host-independent,
#             and a failure here is a failure anywhere.
#   LOCAL     what happened when this Windows box ran it. Real measurements of
#             a real program, and NOT the grading server's seconds -- this host
#             reads ~27x high per operation on residual, and the shim reads
#             ~18% low on wall for rotation-using estimators.
#   SERVER_PREDICTED
#             a FORECAST of deployment risk, from a versioned per-operation
#             model with a stated evidence domain.
#   SERVER_MEASURED
#             telemetry returned by a grading-server run. No current
#             certification section creates such evidence, so this domain is
#             normally NOT_RUN rather than being silently merged with a
#             prediction.
#
# Section 10 prints "DEPLOYMENT AUTHORITY" over a LOCAL subprocess run, and its
# residual check is named "worst GRADED-PATH residual". Two of this
# repository's standing gate failures are that check, and they refuse a
# candidate (330327) that holds a real graded 0/100 row. Whether those gates
# were mis-specified is a question NOBODY HAS RULED ON -- Samuel declined to
# adjudicate it on 2026-09-10 and approved a submission on other grounds. This
# change does not rule on it either.
#
# SO THE CHECK NAMES ARE NOT TOUCHED. Renaming "worst GRADED-PATH residual"
# would silently orphan the two historical FAIL lines that quote it, which is
# the amend-after-the-fact move the record exists to prevent. What changes is
# that every check is now TAGGED with its domain, and the report carries a
# per-domain verdict beside the single one. The single verdict is computed
# exactly as before, so no stored certification changes disposition.
#
# `docs/research/2026-09-10_validation_audit.md` finding 4.

MATH = "MATH"
LOCAL = "LOCAL"
SERVER_PREDICTED = "SERVER_PREDICTED"
SERVER_MEASURED = "SERVER_MEASURED"
# Compatibility for older callers that imported SERVER when it meant the
# forecast domain. New artifacts use the unambiguous spelling above.
SERVER = SERVER_PREDICTED
EVIDENCE_DOMAINS = (MATH, LOCAL, SERVER_PREDICTED, SERVER_MEASURED)

#: Section id -> evidence domain. Section ids are the leading token of the
#: `head()` title, which is how `_CURRENT` already works.
_SECTION_DEFAULT_DOMAIN = {
    "1": MATH,      # imports and the import contract
    "2": MATH,      # setup / shape genericity
    "3": MATH,      # invariants
    "4": MATH,      # determinism
    "5": MATH,      # the grader-twin arms
    "6": SERVER_PREDICTED,  # operation-count planning gate
    "7": LOCAL,     # residual, measured here
    "8": LOCAL,     # wall, measured here
    "9": LOCAL,     # the Linux memory gate: a real kernel, not this server
    "10": LOCAL,    # the official subprocess runner -- still LOCAL seconds
    "R": MATH,      # the read-only self-audit
}


def _domain(section_id: str) -> str:
    """Legacy/default section domain; mixed sections override per check."""
    return _SECTION_DEFAULT_DOMAIN.get(str(section_id), MATH)


def head(t, *, domain=None):
    _CURRENT[0] = t.split(None, 1)[0]
    _CURRENT_DOMAIN[0] = domain or _domain(_CURRENT[0])
    print(f"\n{t}")


#: Every check, with its domain. Fed to `domain_rollup` at the end.
_TAGGED = []
_CURRENT_DOMAIN = [MATH]


def check(name, ok, detail="", *, domain=None):
    _TOTAL[0] += 1
    _SECTIONS.setdefault(_CURRENT[0], []).append((name, bool(ok)))
    _TAGGED.append({"section": _CURRENT[0],
                    "domain": domain or _CURRENT_DOMAIN[0],
                    "name": name, "ok": bool(ok), "skipped": False})
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<62}{detail}")
    if not ok:
        _FAILED.append(f"[{_CURRENT[0]}] {name}")


def skip(name, why, *, domain=None):
    """NOT a pass. Counted separately and named in the summary."""
    _SKIPPED.append(f"[{_CURRENT[0]}] {name} -- {why}")
    _TAGGED.append({"section": _CURRENT[0],
                    "domain": domain or _CURRENT_DOMAIN[0],
                    "name": name, "ok": None, "skipped": True, "why": why})
    print(f"  SKIP  {name:<62}{why}")


def domain_rollup(tagged):
    """Per-domain verdicts. PURE, so it can be tested without a candidate.

    Each domain gets the same three-valued verdict the whole report uses, over
    only its own checks: CERTIFIED with no failures and no skips, REFUSED on
    any failure, INCOMPLETE when something was skipped and nothing failed. A
    domain with no checks at all reports NOT_RUN, which is not a pass.
    """
    out = {}
    for dom in EVIDENCE_DOMAINS:
        mine = [t for t in tagged if t["domain"] == dom]
        failed = [t["name"] for t in mine if t["ok"] is False]
        skipped = [t["name"] for t in mine if t["skipped"]]
        out[dom] = {
            "checks": len([t for t in mine if not t["skipped"]]),
            "failed": failed,
            "skipped": skipped,
            "verdict": ("NOT_RUN" if not mine else
                        "REFUSED" if failed else
                        "INCOMPLETE" if skipped else "CERTIFIED"),
        }
    return out


#: What each domain's verdict is EVIDENCE OF, printed beside it so a reader
#: cannot take a LOCAL pass for a server guarantee.
_DOMAIN_MEANING = {
    MATH: "the contract and the arithmetic. Host-independent; a failure here "
          "is a failure anywhere.",
    LOCAL: "what this host measured. REAL, and NOT the grading server: no "
           "absolute second here transfers (docs/GOTCHAS.md).",
    SERVER_PREDICTED: "a FORECAST from the versioned per-operation model, "
                      "inside its stated evidence domain. Not a measurement, "
                      "and not a guarantee.",
    SERVER_MEASURED: "telemetry observed on the grading server and bound to "
                     "submitted bytes. A missing row is NOT_RUN, never a "
                     "forecast promoted into a measurement.",
}


def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _run(argv, timeout=1800, env=None):
    return subprocess.run(argv, capture_output=True, text=True, cwd=_ROOT,
                          timeout=timeout, errors="replace", env=env)


def _uvpy(*args, timeout=1800):
    return _run([sys.executable] + list(args), timeout=timeout)


def _load(path):
    spec = importlib.util.spec_from_file_location("candidate_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _is_sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def validate_edge_group_receipt(payload, *, group, expected_count,
                                candidate, candidate_sha256,
                                edge_cases_sha256, weights_mode):
    """Strictly validate one allocator-bounded edge-suite worker receipt."""
    errors = []
    if not isinstance(payload, dict):
        return ["receipt is not a JSON object"]
    expected_keys = {
        "artifact", "artifact_version", "candidate",
        "candidate_sha256_before", "candidate_sha256_after",
        "edge_cases_sha256", "group_registry", "world", "group",
        "expected_result_count", "results", "passed", "all_passed",
    }
    if set(payload) != expected_keys:
        errors.append("receipt schema differs")
    if payload.get("artifact") != "edge_cases_group":
        errors.append("artifact identity differs")
    if type(payload.get("artifact_version")) is not int or payload.get(
            "artifact_version") != EDGE_GROUP_RECEIPT_VERSION:
        errors.append("artifact version differs")
    if payload.get("group") != group:
        errors.append("group identity differs")
    if payload.get("candidate") != candidate:
        errors.append("candidate path differs")
    for key in ("candidate_sha256_before", "candidate_sha256_after"):
        value = payload.get(key)
        if not _is_sha256(value) or value != candidate_sha256:
            errors.append(key + " differs")
    value = payload.get("edge_cases_sha256")
    if not _is_sha256(value) or value != edge_cases_sha256:
        errors.append("edge_cases.py hash differs")
    registry = payload.get("group_registry")
    expected_registry = [
        {"group": name, "expected_result_count": count}
        for name, count in EDGE_GROUP_REGISTRY
    ]
    registry_valid = (
        isinstance(registry, list) and len(registry) == len(expected_registry)
        and all(isinstance(row, dict)
                and set(row) == {"group", "expected_result_count"}
                and isinstance(row.get("group"), str)
                and type(row.get("expected_result_count")) is int
                for row in registry)
    )
    if not registry_valid or registry != expected_registry:
        errors.append("group registry differs")

    expected_world = {
        "phase2": True, "real_flopscope": True,
        "weights_mode": weights_mode,
        "width": WIDTH, "depth": DEPTH, "budget": BUDGET,
    }
    world = payload.get("world")
    if not isinstance(world, dict) or set(world) != set(expected_world):
        errors.append("world schema differs")
    else:
        for key, wanted in expected_world.items():
            got = world.get(key)
            if type(got) is not type(wanted) or got != wanted:
                errors.append("world field %s differs" % key)

    count = payload.get("expected_result_count")
    if type(count) is not int or count != expected_count:
        errors.append("expected result count differs")
    results = payload.get("results")
    if (not isinstance(results, list) or len(results) != expected_count
            or not all(type(value) is bool for value in results)):
        errors.append("results are not the exact boolean result vector")
        results = None
    passed = payload.get("passed")
    if type(passed) is not int:
        errors.append("passed count is not an integer")
    elif results is not None and passed != sum(results):
        errors.append("passed count differs from results")
    all_passed = payload.get("all_passed")
    if type(all_passed) is not bool:
        errors.append("all_passed is not a boolean")
    elif results is not None and all_passed != all(results):
        errors.append("all_passed differs from results")
    return errors


class _Ctx:
    def __init__(self, width, depth, budget, seed):
        self.width, self.depth = width, depth
        self.flop_budget, self.seed = budget, seed
        self.n_mlps = len(SEEDS)


# --------------------------------------------------------------------------- #
# the child that runs one predict in a FRESH interpreter
# --------------------------------------------------------------------------- #
FRESH_WORKER = r'''
import hashlib, json, os, sys
REPO, CAND, W, D, B, SEED = sys.argv[1], sys.argv[2], int(sys.argv[3]), \
    int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
LS = os.path.join(REPO, "local_sim")
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != LS]
for p in (REPO, LS):
    if p not in sys.path:
        sys.path.append(p)
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP
from mint_provenance import canonical_mint
try:
    flops.configure(symmetry_warnings=False)
except Exception:
    pass
import importlib.util as iu
spec = iu.spec_from_file_location("cand", CAND)
mod = iu.module_from_spec(spec); spec.loader.exec_module(mod)
class C:
    width, depth, flop_budget, seed, n_mlps = W, D, B, SEED, 1
est = mod.Estimator(); est.setup(C())
w_np, est_seed, _ = canonical_mint(W, D, SEED)
mlp = MLP(weights=[fnp.array(x) for x in w_np], width=W, depth=D, seed=est_seed)
ctx = flops.BudgetContext(flop_budget=B, quiet=True)
with ctx:
    pred = est.predict(mlp, B)
a = np.ascontiguousarray(np.asarray(pred, dtype=np.float64))
print(json.dumps({"sha256": hashlib.sha256(a.tobytes()).hexdigest(),
                  "shape": list(a.shape),
                  "flops_used": float(ctx.flops_used),
                  "pid": os.getpid()}))
'''



# --------------------------------------------------------------------------- #
# The memory report's binding check — extracted so it can be MUTATION-TESTED
#
# Section 9 used to check four things about a `linux_memory_gate` report: that
# the cap applied, that the negative control died, that predict raised nothing,
# and that `status == "PASS"`. It matched the report to the candidate on
# `candidate_sha256` and then asked NOTHING about what the report had actually
# run. Every one of `width`, `depth`, `budget`, `limit_mb`, `seed` and the
# weight representation is a CLI flag on `linux_memory_gate.py`, so a report for
# the right bytes at 64x4 under a 512 MB cap satisfied all four checks and
# certified as "the real 8-GiB enforcement".
#
# That is a reachable acceptance-path defect, and it was found by reading the
# source, not by catching a bad report: no stored report is alleged to be
# misconfigured or forged. `test_linux_memory_binding.py` mutates one field at a
# time and requires each mutation to be REFUSED.
#
# `docs/research/2026-09-10_validation_audit.md` finding 1.


#: Version 4 binds the exact current instrument, distribution metadata and the
#: lifecycle controls. Older reports remain historical evidence, but cannot be
#: upgraded into this gate by a consumer that knows more than their producer did.
MEMORY_REPORT_MIN_VERSION = 5
MEMORY_INSTRUMENT = "local_sim/linux_memory_gate.py"
MEMORY_INSTRUMENT_POLICY = "current-source-sha256"
MEMORY_SUITE_N_MLPS = 100



def _lifecycle_default_mode() -> str:
    """The ONE weight representation a candidate is certified under."""
    try:
        from lifecycle_eval import DEFAULT_WEIGHTS_MODE
        return DEFAULT_WEIGHTS_MODE
    except Exception:
        return "f64_readonly"


def _certified_stack() -> dict:
    """The official package versions THIS host is certifying against.

    Read from the installed packages rather than declared, so a report produced
    against a different stack cannot be bound to this certification by a
    constant somebody forgot to bump.
    """
    out = {}
    for distribution in ("whestbench", "flopscope"):
        try:
            out[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            out[distribution] = ""
    return out


def _memory_instrument_sha256() -> str:
    return _sha(os.path.join(_HERE, "linux_memory_gate.py"))


def consume_memory_report(
    lg,
    *,
    candidate_sha256: str,
    width: int,
    depth: int,
    budget: int,
    cap_mb: int = MEMORY_CAP_MB,
    weights_mode: str = "f64_readonly",
    stack: dict | None = None,
    expected_instrument_sha256: str = "",
):
    """Classify memory evidence without turning absence into a candidate FAIL.

    ``REFUSED`` is reserved for an explicit negative produced by an applicable
    report. Missing legacy bindings, a report for other bytes, or a report for
    another workload are ``INCOMPLETE``. This distinction is essential when old
    artifacts are replayed after the consumer gains new required fields.
    """
    checks = []
    bindings = []
    incomplete = []

    def c(name, ok, detail=""):
        checks.append((name, bool(ok), str(detail)))

    def bind(name, ok, detail=""):
        bindings.append({"name": name, "ok": bool(ok), "detail": str(detail)})
        if not ok:
            incomplete.append(f"{name}: {detail}")

    if not isinstance(lg, dict):
        c("9: the memory report parses", False, f"got {type(lg).__name__}")
        return {"verdict": "REFUSED", "applicable": False, "checks": checks,
                "bindings": bindings, "incomplete": incomplete,
                "artifact_version": 0}

    try:
        ver = int(lg.get("artifact_version", 0) or 0)
    except (TypeError, ValueError):
        ver = 0
    cr = lg.get("candidate_run") if isinstance(lg.get("candidate_run"), dict) else {}
    neg = lg.get("negative_control") if isinstance(lg.get("negative_control"), dict) else {}
    pos = lg.get("positive_control") if isinstance(lg.get("positive_control"), dict) else {}

    got_sha = lg.get("candidate_sha256")
    if not isinstance(got_sha, str) or not got_sha:
        incomplete.append("candidate_sha256 is absent; path-only identity is not binding")
        applicable = None
    elif got_sha != candidate_sha256:
        incomplete.append("report is for different candidate bytes")
        applicable = False
    else:
        applicable = True

    # A report for other or unverified bytes cannot say this candidate failed.
    if applicable is not True:
        return {"verdict": "INCOMPLETE", "applicable": applicable,
                "checks": checks, "bindings": bindings,
                "incomplete": incomplete, "artifact_version": ver}

    # Candidate identity alone is insufficient applicability. Legacy reports
    # often carry enough workload fields to prove that they concern a different
    # run even though they lack the full current binding envelope. Interpret
    # those fields before interpreting an explicit failure: a MemoryError at
    # 64x4 under 512 MiB says nothing about this 1024x16, 8192 MiB workload.
    known_mismatches = []
    got_width, got_depth = lg.get("width"), lg.get("depth")
    if ((got_width is not None and got_width != width) or
            (got_depth is not None and got_depth != depth)):
        known_mismatches.append(
            f"certified shape: {got_width}x{got_depth} vs {width}x{depth}")
    if lg.get("budget") is not None and lg.get("budget") != budget:
        known_mismatches.append(
            f"certified budget: {lg.get('budget')} vs {budget}")
    if lg.get("limit_mb") is not None and lg.get("limit_mb") != cap_mb:
        known_mismatches.append(
            f"full memory cap: {lg.get('limit_mb')} MB vs {cap_mb} MB")
    if (lg.get("weights_mode") is not None and
            lg.get("weights_mode") != weights_mode):
        known_mismatches.append(
            f"deployment weight representation: {lg.get('weights_mode')!r} "
            f"vs {weights_mode!r}")
    if (lg.get("suite_n_mlps") is not None and
            lg.get("suite_n_mlps") != MEMORY_SUITE_N_MLPS):
        known_mismatches.append(
            f"suite lifecycle size: {lg.get('suite_n_mlps')} vs "
            f"{MEMORY_SUITE_N_MLPS}")
    if known_mismatches:
        incomplete.extend(known_mismatches)
        return {"verdict": "INCOMPLETE", "applicable": False,
                "checks": checks, "bindings": bindings,
                "incomplete": incomplete, "artifact_version": ver}

    # Preserve explicit negative evidence before considering whether newer
    # binding fields exist. This is the legacy early-return defect from R2.
    explicit = []
    status = lg.get("status")
    if status not in (None, "PASS"):
        explicit.append(("9: the applicable report's own status is PASS", False,
                         f"status={status!r}"))
    if cr.get("predict_error") not in (None, ""):
        explicit.append(("9: predict completed under the applied cap", False,
                         str(cr.get("predict_error"))))
    if cr.get("setup_raised") not in (None, ""):
        explicit.append(("9: setup completed under the applied cap", False,
                         str(cr.get("setup_raised"))))
    if cr.get("cap_applied") is False:
        explicit.append(("9: the cap was applied in the candidate run", False,
                         str(cr.get("soft_limit_bytes"))))
    if neg.get("survived") is True:
        explicit.append(("9: the negative control died under the cap", False,
                         "the over-cap allocation survived"))
    if pos.get("survived") is False:
        explicit.append(("9: the positive control survived uncapped", False,
                         str(pos.get("error", "it died"))))
    if cr.get("ok") is False:
        explicit.append(("9: the candidate child completed successfully", False,
                         f"stage={cr.get('stage')!r}"))
    checks.extend(explicit)

    if ver < MEMORY_REPORT_MIN_VERSION:
        incomplete.append(
            f"artifact_version {ver} < {MEMORY_REPORT_MIN_VERSION}: missing "
            "current workload, distribution, instrument, lifecycle and literal-launch bindings")
        return {"verdict": "REFUSED" if explicit else "INCOMPLETE",
                "applicable": True, "checks": checks, "bindings": bindings,
                "incomplete": incomplete, "artifact_version": ver}

    bind("artifact kind", lg.get("artifact") == "linux_memory_gate",
         repr(lg.get("artifact")))
    bind("instrument path", lg.get("instrument") == MEMORY_INSTRUMENT,
         repr(lg.get("instrument")))
    bind("instrument verification policy",
         lg.get("instrument_policy") == MEMORY_INSTRUMENT_POLICY,
         repr(lg.get("instrument_policy")))
    bind("instrument sha256",
         bool(expected_instrument_sha256)
         and lg.get("instrument_sha256") == expected_instrument_sha256,
         f"{str(lg.get('instrument_sha256'))[:16]}... vs "
         f"{expected_instrument_sha256[:16]}...")
    bind("certified shape",
         lg.get("width") == width and lg.get("depth") == depth,
         f"{lg.get('width')}x{lg.get('depth')} vs {width}x{depth}")
    bind("certified budget", lg.get("budget") == budget,
         f"{lg.get('budget')} vs {budget}")
    bind("suite lifecycle size", lg.get("suite_n_mlps") == MEMORY_SUITE_N_MLPS
         and cr.get("suite_n_mlps") == MEMORY_SUITE_N_MLPS,
         f"{lg.get('suite_n_mlps')} / child {cr.get('suite_n_mlps')}")
    bind("full memory cap", lg.get("limit_mb") == cap_mb,
         f"{lg.get('limit_mb')} MB vs {cap_mb} MB")
    bind("deployment weight representation",
         lg.get("weights_mode") == weights_mode
         and cr.get("weights_mode") == weights_mode,
         f"{lg.get('weights_mode')} / child {cr.get('weights_mode')}")
    wanted_stack = dict(stack or {})
    got_stack = lg.get("stack") if isinstance(lg.get("stack"), dict) else {}
    bind("distribution version source",
         got_stack.get("source") == "importlib.metadata.version",
         repr(got_stack.get("source")))
    for pkg in ("whestbench", "flopscope"):
        want = wanted_stack.get(pkg, "")
        bind(f"installed distribution {pkg}", bool(want) and got_stack.get(pkg) == want,
             f"{got_stack.get(pkg, '(absent)')!r} vs {want!r}")

    lifecycle = lg.get("lifecycle_controls")
    bind("lifecycle controls are present", isinstance(lifecycle, dict),
         type(lifecycle).__name__)
    if isinstance(lifecycle, dict):
        stateless = lifecycle.get("stateless") or {}
        retaining = lifecycle.get("retaining") or {}
        threshold = int(lifecycle.get("growth_threshold_bytes", 0) or 0)
        bind("stateless lifecycle positive control",
             stateless.get("completed") is True
             and abs(int(stateless.get("growth_bytes", 1 << 60))) < threshold,
             f"growth={stateless.get('growth_bytes')} threshold={threshold}")
        bind("retaining lifecycle negative control",
             retaining.get("completed") is True
             and int(retaining.get("growth_bytes", -1)) >= threshold,
             f"growth={retaining.get('growth_bytes')} threshold={threshold}")
    child_stderr = lg.get("child_stderr")
    bind("all child stderr streams are recorded",
         isinstance(child_stderr, dict)
         and set(child_stderr) == {"negative", "positive",
                                   "lifecycle_stateless", "lifecycle_retaining",
                                   "candidate"},
         repr(sorted(child_stderr) if isinstance(child_stderr, dict)
              else type(child_stderr).__name__))

    if incomplete:
        # Even a FAIL from a differently configured v4 report is not evidence
        # about this workload. Keep it visible in `checks`, but do not promote it
        # to a candidate refusal.
        return {"verdict": "INCOMPLETE", "applicable": False,
                "checks": checks, "bindings": bindings,
                "incomplete": incomplete, "artifact_version": ver}

    c("9: the report is bound to the current instrument and workload", True,
      f"v{ver}, {MEMORY_INSTRUMENT_POLICY}")
    c("9: every child completed without unexpected stderr",
      isinstance(child_stderr, dict) and all(not text for text in child_stderr.values()),
      repr({k: str(v)[-80:] for k, v in (child_stderr or {}).items() if v}))
    c("9: the cap read back exactly", cr.get("cap_applied") is True
      and cr.get("cap_requested_bytes") == cap_mb * 1024 * 1024
      and cr.get("soft_limit_bytes") == cap_mb * 1024 * 1024,
      f"soft={cr.get('soft_limit_bytes')}")
    c("9: the negative control died under the cap", neg.get("survived") is False,
      "an enforcement claim needs a dying over-cap control")
    c("9: the positive control survived uncapped", pos.get("survived") is True,
      "distinguishes the cap from host exhaustion")
    c("9: predict completed under the applied cap",
      cr.get("predict_error") == "" and status == "PASS",
      str(cr.get("predict_error", "")))
    c("9: the deployment arrays are read-only float64",
      weights_mode != "f64_readonly" or
      (cr.get("weight_dtype") == "float64" and cr.get("weight_writeable") is False),
      f"{cr.get('weight_dtype')}, writeable={cr.get('weight_writeable')}")
    c("9: the real packages loaded, not local_sim shims",
      "local_sim" not in str(cr.get("whestbench_file", ""))
      and "local_sim" not in str(cr.get("flopscope_file", "")),
      str(cr.get("whestbench_file", ""))[-46:])
    reps = cr.get("repeat_runs") or []
    c("9: at least three predictions ran in one worker",
      len(reps) >= 3 and cr.get("repeat_error") == "",
      cr.get("repeat_error") or f"{len(reps)} predictions")
    growth = int(cr.get("repeat_vm_size_growth_bytes", 1 << 60))
    c("9: settled address space did not grow across candidate predictions",
      abs(growth) < 64 * (1 << 20), f"{growth / 2**20:+.1f} MB")
    peak = int(cr.get("vm_peak_bytes") or 0)
    c("9: peak address space is under the cap", 0 < peak <= cap_mb * 1024 * 1024,
      f"VmPeak {peak / 2**20:.1f} MB")
    return {"verdict": "REFUSED" if any(not ok for _, ok, _ in checks)
            else "CERTIFIED", "applicable": True, "checks": checks,
            "bindings": bindings, "incomplete": incomplete,
            "artifact_version": ver}


def validate_memory_report(
    lg,
    *,
    candidate_sha256: str,
    width: int,
    depth: int,
    budget: int,
    cap_mb: int = MEMORY_CAP_MB,
    weights_mode: str = "f64_readonly",
    stack: dict | None = None,
    expected_instrument_sha256: str = "",
):
    """Is this memory report evidence about THIS candidate under THIS workload?

    Returns `(checks, degraded)` where `checks` is a list of
    `(name, ok, detail)` triples in the order section 9 prints them, and
    `degraded` is a non-empty string when the report is a pre-version-3 artifact
    that cannot answer the workload questions at all.

    Pure: no filesystem, no subprocess, no globals but the constants above. That
    is the whole point — the mutation battery drives it directly.
    """
    result = consume_memory_report(
        lg, candidate_sha256=candidate_sha256, width=width, depth=depth,
        budget=budget, cap_mb=cap_mb, weights_mode=weights_mode, stack=stack,
        expected_instrument_sha256=expected_instrument_sha256)
    incomplete = "; ".join(result["incomplete"])
    return result["checks"], incomplete


def _json_tail(text):
    for line in reversed((text or "").splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except ValueError:
                continue
    return None


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--candidate", required=True,
                    help="the estimator file to certify. REQUIRED, no default.")
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--depth", type=int, default=DEPTH)
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--json", default="")
    ap.add_argument(
        "--memory-report", default="", metavar="PATH",
        help="explicit Linux memory-gate artifact to consume. Its candidate, "
             "workload, stack and instrument bindings are still verified; "
             "the path itself grants no applicability.")
    ap.add_argument(
        "--memory-only", action="store_true",
        help="run only the production memory-report consumer. Requires "
             "--memory-report; intended for cheap provenance/conformance "
             "checks, not whole-candidate certification.")
    ap.add_argument("--quick", action="store_true",
                    help="skip the network-dependent sections (1 whest validate "
                         "is local, 10 subprocess smoke needs the dataset)")
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--weights-mode", default=None,
                    choices=list(_TWIN_MODES),
                    help="THE GRADER-TWIN ARM for section 5. Default is "
                         "f64_readonly -- what the grading server actually "
                         "hands a candidate. 'f32' is the historical local "
                         "world and is a CONTROL, not evidence.")
    ap.add_argument("--skip-twin", action="store_true",
                    help="skip section 5b (the cross-dtype arm). Only for "
                         "debugging; a certification without it is not one.")
    ap.add_argument("--residual-soak", type=int, default=0, metavar="N",
                    help="run the official subprocess mini split N times and "
                         "pool the residuals into a TAIL (worst / p99 / median) "
                         "instead of the 2-MLP anecdote section 10 gives. "
                         "0 (default) skips it, and a skipped soak is NOT a "
                         "pass. Rung-3 hardening wants N=3. Costs roughly "
                         "N x 100 subprocess MLPs of wall time. READS "
                         "CONFIRM-TIER TRUTH, like section 10.")
    ap.add_argument("--soak-n-mlps", type=int, default=100, metavar="M",
                    help="MLPs per soak repetition (default 100, the full mini "
                         "split). Lower it only to rehearse the plumbing.")
    a = ap.parse_args()
    if a.residual_soak < 0:
        print("--residual-soak must be >= 0")
        return 2

    cand = os.path.abspath(a.candidate)
    if not os.path.exists(cand):
        print(f"no such candidate: {cand}")
        return 2
    rel = os.path.relpath(cand, _ROOT).replace("\\", "/")
    is_ship = os.path.abspath(cand) == os.path.join(_ROOT, "estimator.py")

    sha_before = _sha(cand)
    ship_sha_before = _sha(os.path.join(_ROOT, "estimator.py"))
    git_dirty_before = ((_run(["git", "status", "--porcelain", "--",
                               "estimator.py", "scripts", "local_sim"],
                              timeout=180).stdout) or "").strip()

    # Resolved here because section 3 runs long before section 5 imports
    # lifecycle_eval; the two must agree on which arm is being certified.
    twin_mode_for_edge = a.weights_mode or _TWIN_DEFAULT

    print("candidate certification -- READ-ONLY, TRUTH-FREE")
    print(f"  candidate   {rel}")
    print(f"  sha256      {sha_before}")
    print(f"  shape       {a.width}x{a.depth}   B = {a.budget:,}")
    print("  NOTE: this is NOT preflight and NOT a promotion. The ship-file "
          "gate is untouched;")
    print("        `scripts/preflight.ps1` still certifies estimator.py.")
    if is_ship:
        print("  (the subject IS the shipped file -- a known-good control run)")

    report = {"artifact": "candidate_certification", "artifact_version": 1,
              "candidate": rel, "candidate_sha256": sha_before,
              "width": a.width, "depth": a.depth, "budget": a.budget,
              # COMPUTED, NOT DECLARED. This was hardcoded `True` while
              # section 10 ran `whest run --split mini` against the ORGANISERS'
              # data and stored the resulting `final_layer_mse_CONFIRM_ONLY` in
              # the same report. An independent Skeptic caught the
              # contradiction. The smoke is a deployment check and reading it is
              # legitimate CONFIRM-tier use -- but a report that says
              # `truth_free: true` beside a real MSE is a false label, and a LOG
              # entry that repeats it inherits the falsehood.
              "truth_free": bool(a.quick or a.skip_smoke or a.memory_only),
              "confirm_tier_read": not bool(
                  a.quick or a.skip_smoke or a.memory_only),
              "is_shipped_file": is_ship,
              "sections": {}}

    if a.memory_only:
        if not a.memory_report:
            print("--memory-only requires --memory-report")
            return 2
        lg_path = os.path.abspath(a.memory_report)
        if not os.path.isfile(lg_path):
            print(f"no such memory report: {lg_path}")
            return 2
        try:
            with open(lg_path, encoding="utf-8") as fh:
                lg = json.load(fh)
        except (OSError, ValueError) as exc:
            print(f"invalid memory report JSON: {lg_path}: {exc}")
            return 2
        memory_result = consume_memory_report(
            lg, candidate_sha256=sha_before, width=a.width, depth=a.depth,
            budget=a.budget, cap_mb=MEMORY_CAP_MB,
            weights_mode=_lifecycle_default_mode(), stack=_certified_stack(),
            expected_instrument_sha256=_memory_instrument_sha256())
        report["sections"]["linux_memory_gate"] = {
            "report": os.path.relpath(lg_path, _ROOT),
            "matched_on": "sha256" if lg.get("candidate_sha256") == sha_before
                          else "explicit candidate mismatch",
            "artifact_version": lg.get("artifact_version"),
            "instrument_sha256": lg.get("instrument_sha256", ""),
            "instrument_policy": MEMORY_INSTRUMENT_POLICY,
            "consumer_verdict": memory_result["verdict"],
            "applicable": memory_result["applicable"],
            "binding_checks": memory_result["bindings"],
            "checks": memory_result["checks"],
            "degraded": "; ".join(memory_result["incomplete"]),
        }
        report["status"] = memory_result["verdict"]
        print("memory consumer verdict: %s (applicable=%s)" % (
            memory_result["verdict"], memory_result["applicable"]))
        for name, ok, detail in memory_result["checks"]:
            print("  %s  %s%s" % ("PASS" if ok else "FAIL", name,
                                   "  " + detail if detail else ""))
        for reason in memory_result["incomplete"]:
            print("  INCOMPLETE  " + reason)
        if a.json:
            out = os.path.abspath(a.json)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(report, indent=2, default=str) + "\n")
            print("  report -> " + os.path.relpath(out, _ROOT))
        return 0 if memory_result["verdict"] == "CERTIFIED" else 1

    # ---------------------------------------------------------------- 1 ----
    head("1  the official grader contract", domain=MATH)
    if a.quick:
        skip("1: whest validate", "--quick")
    else:
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        p = _run([os.path.join(os.path.dirname(sys.executable), "whest"),
                  "validate", "--estimator", rel], env=env, timeout=600)
        check("1: `whest validate` accepts the candidate", p.returncode == 0,
              (p.stdout or p.stderr).strip().splitlines()[-1][:60]
              if (p.stdout or p.stderr).strip() else "")
        report["sections"]["validate"] = {"returncode": p.returncode}

    # ---------------------------------------------------------------- 2 ----
    head("2  package and round identity", domain=MATH)
    p = _uvpy(os.path.join(_HERE, "official_config.py"), timeout=900)
    check("2: the LIVE round matches the INSTALLED package, field by field",
          p.returncode == 0, (p.stdout or "").strip().splitlines()[-1][:40])
    import flopscope as _fl
    import whestbench as _wb
    for m, name in ((_wb, "whestbench"), (_fl, "flopscope")):
        check(f"2: {name} is the real package, not local_sim's shim",
              "local_sim" not in os.path.abspath(m.__file__),
              os.path.abspath(m.__file__)[-44:])
    report["sections"]["package_identity"] = {
        "whestbench": _wb.__file__, "flopscope": _fl.__file__,
        "distribution_versions": _certified_stack(),
        "version_source": "importlib.metadata.version",
        "official_config_rc": p.returncode}

    # ---------------------------------------------------------------- 3 ----
    head("3  shape generality, the exact layer-1 closed form, and equivariance",
         domain=MATH)
    # `edge_cases.py` already owns these 30 invariants and already takes a path.
    # Re-implementing them here would give this repository two answers to
    # "is layer 1 right", which is the failure it keeps recording.
    # ...IN THE LIVE WORLD. Until 2026-09-05 this ran the suite's historical
    # defaults: Phase 1's 256x32 shape, Phase 1's 2.72e11 budget, and raw NumPy
    # float32 weights. A Phase-2 candidate was therefore certified on invariants
    # measured at a shape the competition stopped grading on 2026-08-21 and in a
    # dtype the grading server has never handed anyone. `--phase2
    # --real-flopscope` puts it in the world that decides the score.
    edge_path = os.path.join(_HERE, "edge_cases.py")
    edge_sha = _sha(edge_path)
    edge_workers = []
    edge_errors = []
    edge_passed = 0
    edge_observed = 0
    with tempfile.TemporaryDirectory(prefix="certify_edge_groups_") as edge_tmp:
        for edge_group, edge_expected in EDGE_GROUP_REGISTRY:
            edge_receipt_path = os.path.join(edge_tmp, edge_group + ".json")
            edge_argv = [rel, "--phase2", "--real-flopscope",
                         "--weights-mode", twin_mode_for_edge,
                         "--group", edge_group,
                         "--receipt", edge_receipt_path]
            try:
                p = _uvpy(edge_path, *edge_argv, timeout=1800)
                rc = p.returncode
                stdout, stderr = p.stdout or "", p.stderr or ""
            except subprocess.TimeoutExpired as exc:
                rc = -1
                stdout = exc.stdout or ""
                stderr = exc.stderr or ""
                edge_errors.append(edge_group + ": worker timed out")
            except Exception as exc:
                rc = -1
                stdout = ""
                stderr = "%s: %s" % (type(exc).__name__, exc)
                edge_errors.append(edge_group + ": worker launch failed")
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            lines = [line for line in stdout.splitlines() if line.strip()]
            worker_tail = lines[-1] if lines else ""
            payload = None
            if os.path.isfile(edge_receipt_path):
                try:
                    with open(edge_receipt_path, encoding="utf-8") as fh:
                        payload = json.load(fh)
                except (OSError, ValueError) as exc:
                    edge_errors.append(edge_group + ": invalid receipt: " + str(exc))
            else:
                edge_errors.append(edge_group + ": receipt is absent")
            receipt_errors = validate_edge_group_receipt(
                payload, group=edge_group, expected_count=edge_expected,
                candidate=rel, candidate_sha256=sha_before,
                edge_cases_sha256=edge_sha, weights_mode=twin_mode_for_edge)
            edge_errors.extend(edge_group + ": " + err for err in receipt_errors)
            if isinstance(payload, dict):
                results = payload.get("results")
                if (isinstance(results, list)
                        and all(type(value) is bool for value in results)):
                    edge_passed += sum(results)
                    edge_observed += len(results)
            if rc != 0:
                edge_errors.append(edge_group + ": worker rc=%s" % rc)
            edge_workers.append({
                "group": edge_group,
                "expected_result_count": edge_expected,
                "returncode": rc,
                "tail": worker_tail,
                "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
                "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
                "receipt": payload,
                "receipt_errors": receipt_errors,
            })
    edge_total = sum(count for _, count in EDGE_GROUP_REGISTRY)
    edge_ok = not edge_errors and edge_observed == edge_total and edge_passed == edge_total
    tail = "%d/%d checks passed" % (edge_passed, edge_total)
    check("3: the invariant suite passes on THIS candidate, at the LIVE shape "
          "and in the grader's dtype", edge_ok,
          tail[:60] + ("" if not edge_errors else
                       "   " + "; ".join(edge_errors)[:120]))
    report["sections"]["invariants"] = {
        "returncode": 0 if edge_ok else 1, "tail": tail,
        "world": {"phase2": True, "real_flopscope": True,
                   "weights_mode": twin_mode_for_edge},
        "edge_cases_sha256": edge_sha,
        "group_registry": [{"group": group, "expected_result_count": count}
                           for group, count in EDGE_GROUP_REGISTRY],
        "workers": edge_workers,
        "worker_errors": edge_errors,
        # SKEPTIC (2026-09-03): the report recorded only a returncode and a tail
        # line, so it did not say WHICH denominator invariant 9 ran against.
        # `ARC_WHEST_RAW_MSE` is inherited from the environment and can only
        # tighten, but an artifact that cannot say what bar was in force is not
        # a certification of anything in particular.
        "arc_whest_raw_mse": os.environ.get(
            "ARC_WHEST_RAW_MSE", "(unset: the frozen 1.4e-6 decision signal)"),
    }

    # Invariant 9's mutation battery. It exists because that invariant's
    # absolute float32 clause was replaced on 2026-09-03, and an amendment that
    # makes a failing gate pass has to prove it still bites. It was written as a
    # file NOTHING EXECUTED -- an independent Skeptic found it cited as "a gate"
    # while this command and `preflight.ps1` both ran only `edge_cases.py`.
    pb = _uvpy(os.path.join(_HERE, "test_edge_cases_inv9.py"), rel, timeout=1800)
    btail = [l for l in (pb.stdout or "").splitlines() if l.strip()][-1:] or [""]
    check("3b: invariant 9 still rejects deliberate equivariance defects",
          pb.returncode == 0, btail[0][:60])
    report["sections"]["inv9_battery"] = {"returncode": pb.returncode,
                                          "tail": btail[0]}

    # ---------------------------------------------------------------- 4 ----
    head("4  determinism, in-process and in a FRESH worker", domain=MATH)
    fresh = []
    for s in SEEDS[:2]:
        r = _uvpy("-c", FRESH_WORKER, _ROOT, cand, str(a.width), str(a.depth),
                  str(a.budget), str(s), timeout=1800)
        d = _json_tail(r.stdout)
        if d is None:
            check(f"4: fresh worker ran for seed {s}", False,
                  (r.stderr or "")[-70:])
            fresh.append(None)
            continue
        r2 = _uvpy("-c", FRESH_WORKER, _ROOT, cand, str(a.width), str(a.depth),
                   str(a.budget), str(s), timeout=1800)
        d2 = _json_tail(r2.stdout)
        ok = d2 is not None and d["sha256"] == d2["sha256"]
        check(f"4: two FRESH interpreters agree bitwise on seed {s}", ok,
              f"{d['sha256'][:16]} pid {d['pid']} vs "
              f"{(d2 or {}).get('pid')}")
        check(f"4: ...and bill identical FLOPs on seed {s}",
              d2 is not None and d["flops_used"] == d2["flops_used"],
              f"{d['flops_used']:,.0f}")
        check(f"4: the shape is ({a.depth}, {a.width}) on seed {s}",
              d["shape"] == [a.depth, a.width], str(d["shape"]))
        # BOTH members of the comparison, and the seed beside them.
        #
        # `fresh` used to be a list of the FIRST run's dict per seed, so the
        # artifact held two hashes for two DIFFERENT seeds and their inequality
        # was expected -- while the pair that actually proves determinism, the
        # two runs of the SAME seed, was never stored. A reader of the JSON
        # could not tell which comparison had been made. Repaired 2026-09-10
        # (Stage 0 R6, validation audit finding 5).
        fresh.append({
            "seed": int(s),
            "weights_mode": "f32_fnp_array",   # what FRESH_WORKER builds today
            "run_a": d,
            "run_b": d2,
            "hashes_equal": bool(ok),
            "flops_equal": bool(d2 is not None
                                and d["flops_used"] == d2["flops_used"]),
            "distinct_processes": bool(d2 is not None
                                       and d.get("pid") != d2.get("pid")),
        })
    # A pair of runs in two processes cannot see state that only grows across a
    # SEQUENCE of networks in ONE process. The Linux memory gate now runs that
    # sequence (`--repeats`), and section 9 checks its address space did not
    # grow; this records where the boundary is rather than implying determinism
    # covers it.
    report["sections"]["determinism"] = {
        "fresh_worker": fresh,
        "scope": "two fresh interpreters per seed, one prediction each. "
                 "Cross-MLP hidden state inside ONE worker is NOT tested here "
                 "-- section 9's repeat-run evidence covers that.",
        "weights_mode": "FRESH_WORKER builds fnp.array(mint float32); the "
                        "deployment representation is f64_readonly and is "
                        "certified in section 5b. These are different arms.",
    }

    # ------------------------------------------------------------- 5-8 ----
    head("5-8  one metered run: immutability, cost, telemetry, scalar sync",
         domain=MATH)
    import flopscope as flops
    import flopscope.numpy as fnp
    from mint_provenance import canonical_mint
    from p2_weff import MaterializationCounter
    from whestbench import MLP
    from lifecycle_eval import (materialise_predictions, build_weights,
                                WEIGHTS_MODES, DEFAULT_WEIGHTS_MODE)
    import official_config as _oc

    # The arm this certification runs section 5 under. Default = the grader's.
    twin_mode = a.weights_mode or DEFAULT_WEIGHTS_MODE
    assert set(_TWIN_MODES) == set(WEIGHTS_MODES), (
        "certify's argparse choices have drifted from lifecycle_eval.WEIGHTS_MODES"
    )

    try:
        flops.configure(symmetry_warnings=False)
    except Exception:
        pass

    mod = _load(cand)
    est = mod.Estimator()
    t0 = time.perf_counter()
    setup_err = ""
    try:
        est.setup(_Ctx(a.width, a.depth, a.budget, SEEDS[0]))
    except Exception as exc:
        setup_err = f"{type(exc).__name__}: {exc}"
    setup_s = time.perf_counter() - t0
    check("5: setup() raised nothing", setup_err == "",
          setup_err or f"{setup_s:.4f} s")
    check("5: setup is inside the 5 s ceiling (LOCAL, in-process -- the "
          "grader's window is larger)",
          setup_s < SETUP_CAP_S, f"{setup_s:.4f} s of {SETUP_CAP_S} s",
          domain=LOCAL)

    rows = []
    for i, s in enumerate(SEEDS):
        w_np, est_seed, _ = canonical_mint(a.width, a.depth, int(s))
        # THE GRADER-TWIN ARM, not `fnp.array(x)` on a float32 mint. Until
        # 2026-09-05 this line built float32 weights and so did every other
        # weight producer in the repository, which is why a `copy=False` on a
        # float64->float32 cast could survive every local gate and kill five
        # submissions. See lifecycle_eval.WEIGHTS_MODES.
        weights = build_weights(w_np, twin_mode, fnp)
        # sha256 of the grader's arrays BEFORE the call, taken from the
        # flopscope-backed objects the estimator actually receives.
        # At the array's OWN dtype. Casting to float32 to hash would throw away
        # exactly the low bits an in-place f64 mutation would land in.
        pre = [hashlib.sha256(np.ascontiguousarray(
            np.asarray(w)).tobytes()).hexdigest()
            for w in weights]
        mlp = MLP(weights=weights, width=a.width, depth=a.depth, seed=est_seed)

        ctx = flops.BudgetContext(flop_budget=a.budget, quiet=True)
        err = ""
        pred = None
        with MaterializationCounter() as mc:
            try:
                with ctx:
                    pred = est.predict(mlp, a.budget)
                    pred = materialise_predictions(pred, depth=a.depth,
                                                   width=a.width)
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
        post = [hashlib.sha256(np.ascontiguousarray(
            np.asarray(w)).tobytes()).hexdigest()
            for w in weights]

        arr = None if pred is None else np.asarray(pred)
        try:
            ns = (ctx.summary_dict(by_namespace=True).get("by_namespace") or {})
            n_ops = int(sum(float(v.get("calls", 0.0)) for v in ns.values()))
        except Exception:
            n_ops = -1
        rows.append({
            "seed": int(s), "error": err,
            "flops_used": float(ctx.flops_used),
            "n_ops": n_ops,
            "multiplier": max(FLOOR, float(ctx.flops_used) / a.budget),
            "residual_wall_time_s": float(ctx.residual_wall_time_s or 0.0),
            "wall_time_s": float(ctx.wall_time_s or 0.0),
            "materializations": dict(mc.counts),
            "materializations_total": mc.total,
            "weights_unchanged": pre == post,
            "shape": None if arr is None else list(arr.shape),
            "dtype": None if arr is None else str(arr.dtype),
            "finite": None if arr is None else bool(np.all(np.isfinite(arr))),
            "min_value": 0.0 if arr is None else float(np.min(arr)),
            "max_value": 0.0 if arr is None else float(np.max(arr)),
            "n_negative": 0 if arr is None else int(np.sum(np.asarray(arr) < 0)),
            "n_values": 0 if arr is None else int(np.asarray(arr).size),
            "nbytes": None if arr is None else int(arr.nbytes),
            # NOT an error: there is no truth here. This is mean(pred^2), a
            # magnitude that catches an all-zero or exploded return.
            "mean_square_final_row": None if arr is None
            else float(np.mean(np.asarray(arr[-1], dtype=np.float64) ** 2)),
            "weights_mode": twin_mode,
        })

    check(f"5: the arm is the GRADER's dtype ({_TWIN_DEFAULT}), not the "
          f"historical local float32",
          twin_mode == _TWIN_DEFAULT,
          f"ran under {twin_mode!r}"
          + ("" if twin_mode == _TWIN_DEFAULT
             else "  <-- a CONTROL arm; this is not a certification"))
    check("5: no run raised", all(not r["error"] for r in rows),
          "; ".join(r["error"] for r in rows if r["error"])[:60] or "clean")
    check("5: the grader's weight arrays are UNCHANGED after predict",
          all(r["weights_unchanged"] for r in rows), "sha256 before == after")
    check("5: every output has shape (depth, width)",
          all(r["shape"] == [a.depth, a.width] for r in rows),
          str(rows[0]["shape"]))
    check("5: every output is finite", all(r["finite"] for r in rows), "")
    # A post-ReLU mean cannot be negative as a QUANTITY, but a float64 sum that
    # should land on a near-zero neuron can round below it. Measured on the
    # shipped v2.6.1 at 1024x16: 4 of 16,384 entries at -6.1e-17 against row
    # means of ~0.46 -- about one ULP, and 1.3e-16 relative. Demanding exact
    # non-negativity would fail a correct estimator; demanding nothing would
    # miss a sign error. The bar is scaled to the row and sits four orders above
    # the observed noise and many orders below any structural defect.
    worst_neg = min(r["min_value"] for r in rows)
    scale = max(r["max_value"] for r in rows)
    tol = -1e-12 * max(1.0, scale)
    check("5: no output is negative beyond float rounding "
          f"(bar {tol:.1e}, scaled to the row)",
          worst_neg >= tol,
          f"most negative {worst_neg:.3e}"
          + (f", {sum(r['n_negative'] for r in rows)} entr(ies) of "
             f"{sum(r['n_values'] for r in rows)}, "
             f"{abs(worst_neg) / max(scale, 1e-300):.1e} relative"
             if worst_neg < 0 else " -- none"))
    # `nbytes` is None on a row whose predict raised. `0 < None` is a TypeError
    # in Python 3, and a TypeError here ABORTS the certifier -- which is how a
    # candidate that raises on every seed used to skip sections 5b onward
    # entirely, reporting three FAIL lines instead of the arm's verdict. Found
    # by mutation M11 on 2026-09-05.
    check("5: no returned array approaches the 4 GiB single-array cap",
          all(r["nbytes"] is not None and 0 < r["nbytes"] < SINGLE_ARRAY_CAP
              for r in rows),
          f"{rows[0]['nbytes'] / 2**20:.3f} MB" if rows[0]["nbytes"] is not None
          else "no array returned")

    f0 = rows[0]["flops_used"]
    _flops_values = [r["flops_used"] for r in rows]
    check("6: every counted-FLOP total is finite and inside the hard budget",
          all(math.isfinite(v) and 0 < v <= a.budget for v in _flops_values),
          (f"{_flops_values[0]:,.0f}" if len(set(_flops_values)) == 1 else
           "value-dependent: " + ", ".join(f"{v:,.0f}" for v in _flops_values)),
          domain=MATH)
    mult = max(FLOOR, f0 / a.budget)
    check("6: the multiplier is the score's own identity max(0.1, F/B)",
          abs(mult - rows[0]["multiplier"]) < 1e-15,
          f"{mult:.6f}  (C/B = {f0 / a.budget:.7f}"
          f"{', PINNED at the floor' if f0 / a.budget < FLOOR else ''})",
          domain=MATH)
    check("6: the operation ledger is non-empty",
          all(r["n_ops"] > 0 for r in rows), f"{rows[0]['n_ops']:,} ops",
          domain=MATH)

    # --- THE OP COUNT IS THE RESIDUAL GATE --------------------------------
    # Phase 2 does not price residual; it FAILS an MLP above 0.4 s. This host
    # measures residual 8x pessimistically and 2.4x unstably, so the number that
    # decides the outcome cannot come from here. The op count can: it is exact,
    # host-independent, and the graded law is
    # `residual ~ 11.6 us x ops` (official_config.GRADED_RESIDUAL_PROVENANCE,
    # calibrated on 329896 and corroborated by the top-8's telemetry).
    worst_ops = max(r["n_ops"] for r in rows)
    pred_res = _oc.predicted_graded_residual_s(worst_ops)
    pred_hr = _oc.predicted_graded_residual_headroom(worst_ops)
    check(f"7: worst op count is inside the Phase-2 gate "
          f"({_oc.PHASE2_OP_COUNT_GATE:,} ops)",
          worst_ops <= _oc.PHASE2_OP_COUNT_GATE,
          f"{worst_ops:,} ops -> PREDICTED graded residual "
          f"{pred_res * 1000:.1f} ms, {pred_hr:.2f}x under the "
          f"{RESIDUAL_CAP_S} s cliff "
          f"({_oc.op_count_at_cap():,.0f} ops would hit it)",
          domain=SERVER_PREDICTED)

    worst_res = max(r["residual_wall_time_s"] for r in rows)
    worst_wall = max(r["wall_time_s"] for r in rows)
    # The LOCAL residual survives as a RATIO-ONLY instrument: run-to-run spread
    # and worst-vs-median, never a headroom against the cliff. Quoting the local
    # seconds as graded headroom on 2026-09-05 sized a whole work stream against
    # a number 5.9x wrong; the erratum is in experiments/LOG.md.
    med_res = sorted(r["residual_wall_time_s"] for r in rows)[len(rows) // 2]
    check("7: LOCAL residual is stable run to run (a RATIO instrument -- the "
          "cliff is judged on the op count above, never on these seconds)",
          worst_res <= 3.0 * max(med_res, 1e-9),
          f"worst {worst_res * 1000:.1f} ms / median {med_res * 1000:.1f} ms "
          f"= {worst_res / max(med_res, 1e-9):.2f}x spread; this host measures "
          f"{worst_res / max(worst_ops, 1) * 1e6:.0f} us/op against the "
          f"grader's {_oc.GRADED_RESIDUAL_S_PER_OP * 1e6:.1f}",
          domain=LOCAL)
    check("7: worst per-MLP wall is under the installed 120 s limit",
          worst_wall < WALL_CAP_S,
          f"{worst_wall:.3f} s -> {WALL_CAP_S / worst_wall:.0f}x"
          if worst_wall else "0", domain=LOCAL)

    tot_mat = sum(r["materializations_total"] for r in rows)
    check("8: ZERO scalar materializations during predict "
          "(__bool__/__float__/__int__/__index__)",
          tot_mat == 0,
          "counted, not asserted" if tot_mat == 0
          else json.dumps(rows[0]["materializations"]), domain=MATH)
    report["sections"]["metered_runs"] = rows
    report["sections"]["setup_s_local"] = setup_s

    # --------------------------------------------------------------- 5b ----
    # THE CROSS-DTYPE ARM. Section 5 proves the candidate works under the
    # grader's dtype. This proves the dtype is not silently changing the ANSWER,
    # and that whatever FLOPs it costs are ACCOUNTED, not merely observed.
    #
    # The 2026-09-05 defect was invisible because no arm existed at all. An arm
    # that only ran the f64 path would have caught that one; an arm that does
    # not also compare against f32 would miss the opposite defect -- a candidate
    # that runs everything in f64 and quietly doubles its bill.
    if not a.skip_twin:
        head("5b  grader-twin arm: the same candidate across weight dtypes",
             domain=MATH)
        twin_seed = int(SEEDS[0])
        w_np, est_seed, _ = canonical_mint(a.width, a.depth, twin_seed)
        arms = {}
        for mode in WEIGHTS_MODES:
            wts = build_weights(w_np, mode, fnp)
            m = MLP(weights=wts, width=a.width, depth=a.depth, seed=est_seed)
            c = flops.BudgetContext(flop_budget=a.budget, quiet=True)
            e2 = _load(cand).Estimator()
            terr = ""
            try:
                e2.setup(_Ctx(a.width, a.depth, a.budget, twin_seed))
            except Exception as exc:
                terr = f"setup {type(exc).__name__}: {exc}"
            p_ = None
            if not terr:
                try:
                    with c:
                        p_ = e2.predict(m, a.budget)
                        p_ = materialise_predictions(p_, depth=a.depth,
                                                     width=a.width)
                except Exception as exc:
                    terr = f"{type(exc).__name__}: {exc}"
            pa = None if p_ is None else np.asarray(p_, dtype=np.float64)
            try:
                nsx = (c.summary_dict(by_namespace=True).get("by_namespace") or {})
                ops_ = int(sum(float(v.get("calls", 0.0)) for v in nsx.values()))
            except Exception:
                ops_ = -1
            arms[mode] = {
                "error": terr,
                "flops_used": float(c.flops_used),
                "n_ops": ops_,
                "finite": None if pa is None else bool(np.all(np.isfinite(pa))),
                "shape": None if pa is None else list(pa.shape),
                "mean_square_final_row": None if pa is None
                else float(np.mean(pa[-1] ** 2)),
                "_pred": pa,
            }
            print(f"      {mode:<20} {c.flops_used:>18,.0f} FLOPs  "
                  f"{ops_:>7,} ops  " + (terr or "clean"))

        for mode in WEIGHTS_MODES:
            check(f"5b: the {mode} arm raised nothing",
                  arms[mode]["error"] == "", arms[mode]["error"] or "clean")
            check(f"5b: the {mode} arm returned a finite (depth, width) array",
                  arms[mode]["finite"] is True
                  and arms[mode]["shape"] == [a.depth, a.width],
                  f"{arms[mode]['shape']}, finite={arms[mode]['finite']}")

        # --- the answer must not depend on the arm ---------------------------
        # float32 tolerance, because a candidate that legitimately computes in
        # f32 will differ in the last bits depending on where its inputs were
        # rounded. `np.finfo(np.float32).eps` is 1.19e-7; 8x that, relative to
        # the row's own scale, is loose enough for a reordered f32 reduction and
        # far tighter than any real behaviour change.
        F32_TOL = 8.0 * float(np.finfo(np.float32).eps)
        base = arms["f32"]["_pred"]
        for mode in WEIGHTS_MODES:
            if mode == "f32":
                continue
            other = arms[mode]["_pred"]
            if base is None or other is None:
                check(f"5b: {mode} agrees with the f32 control within f32 "
                      f"tolerance", False, "an arm produced no output")
                continue
            scale = max(float(np.max(np.abs(base))), 1e-30)
            # NOT `rel` -- that name belongs to the candidate's repo-relative
            # path for the whole of main(), and shadowing it turned section 9's
            # remediation hint into "--candidate 0.0".
            rel_diff = float(np.max(np.abs(other - base))) / scale
            check(f"5b: {mode} agrees with the f32 control within f32 "
                  f"tolerance ({F32_TOL:.2e} relative)",
                  rel_diff <= F32_TOL,
                  f"max relative difference {rel_diff:.3e}")

        # --- and whatever the dtype costs must be EXPLAINED -------------------
        # A cast of the grader's 16 float64 (n, n) weight matrices is billed at
        # 1 FLOP/element at the f64 rate = 2, i.e. 2 * 16 * n^2. A candidate that
        # casts once pays exactly that; one that casts per layer inside the tower
        # pays a multiple of it; one that never casts pays zero and is running
        # f64 arithmetic end to end, which is a 2x bill everywhere else. All
        # three are legitimate designs and all three must be NAMED.
        quantum = 2 * a.depth * a.width * a.width
        d_f64 = arms["f64"]["flops_used"] - arms["f32"]["flops_used"]
        ratio = d_f64 / quantum if quantum else float("nan")
        near_int = abs(ratio - round(ratio)) <= 0.02
        check("5b: the f64-vs-f32 FLOP delta is an integer multiple of one "
              f"pass over the weights (2*depth*width^2 = {quantum:,})",
              near_int,
              f"delta {d_f64:+,.0f} = {ratio:.4f} passes"
              + ("  (0 passes: the candidate never casts -- it is doing f64 "
                 "arithmetic on the grader)" if abs(ratio) < 0.02 else ""))
        check("5b: the read-only arm bills exactly what the writable f64 arm "
              "does (immutability is free, or the candidate is copying)",
              arms["f64_readonly"]["flops_used"] == arms["f64"]["flops_used"],
              f"{arms['f64_readonly']['flops_used'] - arms['f64']['flops_used']:+,.0f}")
        check("5b: section 5's own bill matches its arm in 5b "
              "(the two sections ran the same candidate the same way)",
              abs(rows[0]["flops_used"] - arms[twin_mode]["flops_used"]) < 1e-9,
              f"{rows[0]['flops_used']:,.0f} vs "
              f"{arms[twin_mode]['flops_used']:,.0f}")

        report["sections"]["grader_twin_arm"] = {
            "seed": twin_seed,
            "certified_under": twin_mode,
            "f32_tolerance": F32_TOL,
            "weight_pass_flops": quantum,
            "arms": {k: {kk: vv for kk, vv in v.items() if kk != "_pred"}
                     for k, v in arms.items()},
        }

    # ---------------------------------------------------------------- 9 ----
    head("9  memory: accounting here, ENFORCEMENT only on Linux", domain=LOCAL)
    peak = _host_memory()
    if peak.get("available"):
        print(f"      peak working set {peak['peak_wset_bytes'] / 2**20:.1f} MB, "
              f"peak commit {peak['peak_pagefile_bytes'] / 2**20:.1f} MB "
              f"({peak['source']})")
        print("      RLIMIT_AS caps ADDRESS SPACE, which exceeds both counters "
              "by an amount")
        print("      no Windows counter reports. These are accounting, and a "
              "headroom computed")
        print("      from them is an UPPER BOUND, never a margin.")
    else:
        # Printing "0.0 MB" for an unavailable counter is the exact failure this
        # gate exists to prevent: reporting a state as though it were a
        # measurement. Say it is unavailable.
        skip("9: host memory accounting", peak.get("why", "unavailable"))

    # WHERE THE ENFORCEMENT REPORT LIVES.
    #
    # Beside the candidate first -- a research candidate under
    # `experiments/<slice>/` keeps its own evidence in that slice -- then the
    # WP0C/D directory for the shipped file. The match is on the report's own
    # `candidate_sha256`, never on the filename and, since 2026-09-05, no longer
    # on the path either: a path is not an identity, so a report written before
    # the file was edited kept passing as "FOR THIS CANDIDATE" for bytes that no
    # longer existed. `artifact_version` 1 reports carry no sha256; they are
    # accepted on the path AND explicitly labelled as unverified bytes, because
    # silently rejecting every historical report would have made this change a
    # different, larger one.
    lg_path = None
    lg_match = ""
    cand_dir = os.path.dirname(cand)
    stem = os.path.splitext(os.path.basename(cand))[0]
    candidates = []
    if a.memory_report:
        explicit = os.path.abspath(a.memory_report)
        if not os.path.isfile(explicit):
            print(f"no such memory report: {explicit}")
            return 2
        candidates.append(explicit)
    else:
        candidates.extend((
            os.path.join(cand_dir, "linux_memory_gate_%s.json" % stem),
            os.path.join(_ROOT, "experiments", "2026-08-31_wp0cd_certify",
                         "linux_memory_gate_%s.json" % stem),
            os.path.join(_ROOT, "experiments", "2026-08-31_wp0cd_certify",
                         "linux_memory_gate_v261.json"),
        ))
    for c in candidates:
        if os.path.exists(c):
            try:
                with open(c, encoding="utf-8") as fh:
                    _d = json.load(fh)
            except (OSError, ValueError):
                if a.memory_report:
                    print(f"invalid memory report JSON: {c}")
                    return 2
                continue
            _lg_sha = _d.get("candidate_sha256")
            if _lg_sha:
                if _lg_sha == sha_before:
                    lg_path, lg_match = c, "sha256"
                    break
                if a.memory_report:
                    lg_path, lg_match = c, "explicit candidate mismatch"
                    break
                continue
            if _d.get("candidate") == rel:
                lg_path, lg_match = c, "path only (artifact_version 1)"
                break
            if a.memory_report:
                lg_path, lg_match = c, "explicit unverified identity"
                break
    if lg_path is None:
        skip("9: the real 8-GiB enforcement",
             "no linux_memory_gate report FOR THIS CANDIDATE. Run: uv run "
             "python local_sim/linux_memory_gate.py --candidate %s --json "
             "<dir>/linux_memory_gate_%s.json" % (rel, stem))
    lg = None
    if lg_path is not None:
        with open(lg_path, encoding="utf-8") as fh:
            lg = json.load(fh)
    if lg is not None:
        check("9: a Linux 8-GiB enforcement report exists FOR THIS CANDIDATE",
              True, f"{os.path.relpath(lg_path, _ROOT)}  matched on {lg_match}")
        check("9: ...and it is matched on the candidate's sha256, not its path",
              lg_match == "sha256",
              f"{sha_before[:16]}..." if lg_match == "sha256"
              else "a pre-2026-09-05 report with no candidate_sha256 -- "
                   "regenerate it with linux_memory_gate.py to certify the "
                   "BYTES rather than the filename")
        memory_result = consume_memory_report(
            lg,
            candidate_sha256=sha_before,
            width=a.width, depth=a.depth, budget=a.budget,
            cap_mb=MEMORY_CAP_MB,
            weights_mode=_lifecycle_default_mode(),
            stack=_certified_stack(),
            expected_instrument_sha256=_memory_instrument_sha256(),
        )
        mem_checks = memory_result["checks"]
        degraded = "; ".join(memory_result["incomplete"])
        for _name, _ok, _detail in mem_checks:
            check(_name, _ok, _detail)
        if degraded:
            skip("9: the STRONG 8-GiB enforcement gate", degraded +
                 ". Regenerate: uv run python local_sim/linux_memory_gate.py "
                 "--candidate %s --json <dir>/linux_memory_gate_%s.json"
                 % (rel, stem))
        # A path is not evidence. Record what was checked, so the artifact can
        # be re-audited without re-running the gate.
        report["sections"]["linux_memory_gate"] = {
            "report": os.path.relpath(lg_path, _ROOT),
            "matched_on": lg_match,
            "artifact_version": lg.get("artifact_version"),
            "instrument_sha256": lg.get("instrument_sha256", ""),
            "instrument_policy": MEMORY_INSTRUMENT_POLICY,
            "consumer_verdict": memory_result["verdict"],
            "applicable": memory_result["applicable"],
            "binding_checks": memory_result["bindings"],
            "bound_to": {"width": a.width, "depth": a.depth, "budget": a.budget,
                         "cap_mb": MEMORY_CAP_MB,
                         "weights_mode": _lifecycle_default_mode(),
                         "stack": _certified_stack(),
                         "instrument_sha256": _memory_instrument_sha256()},
            "degraded": degraded,
            "checks": [{"name": n, "ok": o, "detail": d}
                       for n, o, d in mem_checks],
        }
    report["sections"]["windows_memory_accounting"] = peak

    # --------------------------------------------------------------- 10 ----
    head("10  the official subprocess runner -- DEPLOYMENT AUTHORITY",
         domain=LOCAL)
    if not (a.quick or a.skip_smoke):
        print("      *** THIS SECTION READS CONFIRM-TIER TRUTH. It runs the")
        print("      organisers' v2-phase2 mini split, so the report's")
        print("      `final_layer_mse_CONFIRM_ONLY` is a REAL error against REAL")
        print("      labels. CONFIRM is read-only: it may CHECK a decision made")
        print("      on mints and may CHOOSE nothing. Do not select a basis")
        print("      count, a blend, a rank or a threshold on it, and do not")
        print("      describe a run that includes this section as truth-free.")
    if a.quick or a.skip_smoke:
        skip("10: official Phase-2 subprocess smoke",
             "--quick/--skip-smoke; a skipped deployment check is NOT a pass")
    else:
        smoke = _smoke(rel, timeout=2400)
        report["sections"]["subprocess_smoke"] = smoke
        check("10: the official runner produced a report",
              smoke.get("ok_report") is True, smoke.get("why", "")[:60])
        if smoke.get("ok_report"):
            check("10: zero failed MLPs", smoke["n_failed_mlps"] == 0,
                  f"{smoke['n_failed_mlps']} failed")
            check("10: the live config is the Phase-2 round",
                  smoke["dataset"].endswith("@v2-phase2")
                  and smoke["lambda"] == 0.0
                  and smoke["residual_cap"] == 0.4,
                  f"{smoke['width']}x{smoke['depth']}, lambda "
                  f"{smoke['lambda']}, cap {smoke['residual_cap']} s")
            # `worst_residual` is 0.0 when EVERY smoke row took the exception
            # path (the runner zeroes it there). The headroom division was
            # unguarded, so the certifier died with a ZeroDivisionError before
            # writing its JSON or printing a verdict -- on a host where a 15%
            # per-row failure rate had just been observed, a ~2% event.
            _wr = smoke["worst_residual"]
            check("10: worst GRADED-PATH residual is under the cliff",
                  0.0 < _wr < smoke["residual_cap"],
                  (f"{_wr * 1000:.2f} ms -> "
                   f"{smoke['residual_cap'] / _wr:.1f}x") if _wr > 0 else
                  "0.00 ms -- every smoke row FAILED before it was timed; "
                  "this is not a pass")
            print(f"      counted FLOPs {smoke['flops_per_mlp']:,.0f}   "
                  f"multiplier {smoke['multiplier']:.4f}   "
                  f"C/B {smoke['c_over_b']:.7f}")

    # --------------------------------------------------------------- 11 ----
    head("11  residual TAIL -- the soak, not the anecdote", domain=LOCAL)
    if not a.residual_soak:
        # NOT a skip(). `skip()` is for a check that would otherwise have run
        # and was suppressed -- it downgrades the verdict to INCOMPLETE, which
        # is right for `--quick` and wrong here: the soak is an opt-in
        # DEEPENING, and counting its absence as a skip would flip every
        # existing caller's CERTIFIED to INCOMPLETE for asking nothing new.
        # The report still records that no tail was measured, so a certificate
        # cannot be mistaken for one that carries a p99.
        report["sections"]["residual_soak"] = {"requested": False}
        print("      not requested (--residual-soak 0, the default).")
        print("      Section 10 measured residual on TWO MLPs, ONCE. That is an")
        print("      anecdote, not a tail, and this certificate does NOT carry a")
        print("      p99. Rung-3 hardening requires --residual-soak 3.")
    else:
        print(f"      *** READS CONFIRM-TIER TRUTH, exactly as section 10 does.")
        print(f"      {a.residual_soak} x {a.soak_n_mlps} MLPs on the official "
              f"subprocess runner.")
        soak = _residual_soak(rel, reps=a.residual_soak, n_mlps=a.soak_n_mlps)
        soak["requested"] = True
        report["sections"]["residual_soak"] = soak
        check("11: every soak repetition produced a report",
              soak["ok"], soak.get("why", "")[:70])
        if soak["ok"]:
            cap = soak["residual_cap"]
            check("11: zero failed MLPs across every repetition",
                  soak["n_failed_mlps"] == 0,
                  f"{soak['n_failed_mlps']} failed of {soak['n_rows']} rows"
                  + (f", dominant mode {soak['dominant_failure_mode']!r}"
                     if soak.get("dominant_failure_mode") else ""))
            if soak["n_failed_mlps"]:
                # A REFUSED verdict must carry its reason in the transcript.
                # The first run of this instrument reported 45 failures and no
                # cause, which cost a whole investigation to recover.
                print("      FAILURE ATTRIBUTION -- "
                      f"{soak['n_failed_mlps']} of {soak['n_rows']} rows:")
                for k, v in sorted(soak["failure_breakdown"].items(),
                                   key=lambda kv: -kv[1]):
                    if v:
                        print(f"        {k:32s} {v}")
                seen = {}
                for f in soak["failures"]:
                    seen[f["error_code"]] = seen.get(f["error_code"], 0) + 1
                for code, n in sorted(seen.items(), key=lambda kv: -kv[1]):
                    print(f"        error_code {code!r}: {n}")
                bad = [f for f in soak["failures"] if f["deepest_repo_frame"]]
                if bad:
                    print("      *** A FAILING FRAME IS IN THIS REPOSITORY. "
                          "This is NOT host contention:")
                    print(f"        {bad[0]['deepest_repo_frame']}")
                    print(f"        {bad[0]['traceback_last_line']}")
                else:
                    print("      no failing frame lies in this repository, so "
                          "the fault is harness- or host-side,")
                    print("      not in the candidate's own math. That is a "
                          "LOCATION, not an exoneration.")
                print(f"      residual rows excluded as never-executed: "
                      f"{soak['residual_rows_excluded_failed']}")
            check("11: counted FLOPs are identical on every executed row",
                  soak["flops_executed_constant"] is True,
                  f"{soak['flops_executed_min']:,} .. "
                  f"{soak['flops_executed_max']:,}"
                  if soak["flops_executed_min"] is not None else "no rows")
            check("11: the WORST pooled residual clears the cliff",
                  soak["worst"] < cap,
                  f"worst {soak['worst']*1000:.2f} ms of a {cap*1000:.0f} ms "
                  f"cap -> {soak['headroom_worst']:.2f}x headroom")
            # The number that matters is the WORST, not the mean: the cap is a
            # cliff and one MLP over it FAILS that MLP. The mean is recorded
            # because a mean far below a wild worst is itself a finding.
            print(f"      pooled over {soak['n_rows']} rows from "
                  f"{soak['reps']} run(s):")
            print(f"        median {soak['median']*1000:8.3f} ms")
            print(f"        p99    {soak['p99']*1000:8.3f} ms")
            print(f"        worst  {soak['worst']*1000:8.3f} ms   "
                  f"-> headroom {soak['headroom_worst']:.2f}x")
            print(f"        worst/median dispersion {soak['worst_over_median']:.2f}x")
            print(f"      over {soak['n_rows_executed']} EXECUTED rows of "
                  f"{soak['n_rows']}.")
            if soak["residual_rows_excluded_failed"]:
                # Only the EXCEPTION path zeroes the residual. The four
                # exhaustion paths leave a real -- and possibly over-cap --
                # number, so excluding them makes `worst` LESS conservative and
                # the contaminated figure must be shown next to it.
                print(f"      pooling the {soak['residual_rows_excluded_failed']}"
                      f" failed rows would report median "
                      f"{soak['median_all_rows']*1000:.3f} ms (vs "
                      f"{soak['median']*1000:.3f}) and worst "
                      f"{soak['worst_all_rows']*1000:.3f} ms (vs "
                      f"{soak['worst']*1000:.3f}).")
                print(f"      A row that FAILED ON THE CLIFF keeps its real, "
                      f"over-cap residual, so the")
                print(f"      contaminated worst "
                      f"({soak['headroom_worst_all_rows']:.2f}x headroom) is "
                      f"the CONSERVATIVE one. Check 11's")
                print(f"      zero-failure clause is what guards that, not "
                      f"the headroom number.")
            print(f"      LOCAL HOST ONLY. `AGENTS.md` forbids quoting a local "
                  f"absolute wall time as")
            print(f"      a grader prediction; the two submitted files ran "
                  f"~7x faster on the grader,")
            print(f"      and that is an observation about THOSE files, not a "
                  f"safety factor for this one.")

    # ---------------------------------------------------------------- R ----
    head("R  the repository was not modified", domain=MATH)
    # Taken AFTER the run, deliberately: a hash captured before a write cannot
    # see the write. `p1pre_fault_inject.py` shipped that defect once.
    check("R: the candidate file is byte-identical", _sha(cand) == sha_before,
          sha_before[:16])
    check("R: estimator.py is byte-identical",
          _sha(os.path.join(_ROOT, "estimator.py")) == ship_sha_before,
          ship_sha_before[:16])
    # THIS CHECK'S CONDITION USED TO BE THE LITERAL `True`, with the git output
    # relegated to an "informational:" detail string -- so it printed PASS while
    # its own detail listed modified files. That is `docs/GOTCHAS.md`'s named
    # failure: a check reporting a state rather than testing a property, and an
    # independent Skeptic found it inside a claimed 36/36.
    #
    # The property that is actually this run's business is narrow and is now the
    # condition: THIS RUN wrote nothing. Files already dirty when it started are
    # the caller's business, so they are compared BEFORE and AFTER rather than
    # required to be absent.
    g_after = ((_run(["git", "status", "--porcelain", "--", "estimator.py",
                      "scripts", "local_sim"], timeout=180).stdout) or "").strip()
    check("R: this run changed no tracked source under "
          "estimator.py/scripts/local_sim",
          g_after == git_dirty_before,
          "unchanged since start"
          + (f" (already dirty at start: {len(git_dirty_before.splitlines())} "
             f"file(s), not this run's doing)" if git_dirty_before else "")
          if g_after == git_dirty_before
          else "THIS RUN DIRTIED: "
               + "; ".join(sorted(set(g_after.splitlines())
                                  - set(git_dirty_before.splitlines())))[:70])

    report["checks_total"] = _TOTAL[0]
    report["checks_failed"] = list(_FAILED)
    report["checks_skipped"] = list(_SKIPPED)
    # The per-domain rollup goes in BESIDE the single verdict, never instead
    # of it: one unqualified word over three kinds of evidence is what this
    # repairs, and replacing the word would change what stored reports mean.
    report["checks_by_domain"] = domain_rollup(_TAGGED)
    report["domain_meaning"] = dict(_DOMAIN_MEANING)
    report["checks_tagged"] = _TAGGED
    print()
    print("  EVIDENCE BY DOMAIN -- one verdict over three different questions")
    for _dom in EVIDENCE_DOMAINS:
        _d = report["checks_by_domain"][_dom]
        print(f"    {_dom:<7} {_d['verdict']:<11} "
              f"{_d['checks']} checks, {len(_d['failed'])} failed, "
              f"{len(_d['skipped'])} skipped")
        print(f"            {_DOMAIN_MEANING[_dom]}")
    print()
    report["verdict"] = ("CERTIFIED" if not _FAILED and not _SKIPPED else
                         "INCOMPLETE" if not _FAILED else "REFUSED")
    if a.json:
        d = os.path.dirname(os.path.abspath(a.json))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(report, indent=2, default=str) + "\n")
        print(f"\n  report -> {a.json}")

    print()
    if _FAILED:
        print(f"CANDIDATE REFUSED -- {len(_FAILED)} failed:")
        for f in _FAILED:
            print(f"  - {f}")
        return 1
    if _SKIPPED:
        print(f"CANDIDATE INCOMPLETE -- {_TOTAL[0]} passed, "
              f"{len(_SKIPPED)} NOT CHECKED. A skipped check is not a pass:")
        for s in _SKIPPED:
            print(f"  - {s}")
        return 3
    print(f"CANDIDATE CERTIFIED -- {_TOTAL[0]}/{_TOTAL[0]} checks passed")
    print("  This is one input to promotion, not promotion. See GAME_PLAN.md "
          "section 6.")
    return 0


def _host_memory() -> dict:
    """Peak working set and peak commit for THIS process.

    Delegates to `test_deployment_envelope.peak_memory_bytes()`, which already
    owns this and already carries the lesson: `GetCurrentProcess()` returns the
    pseudo-handle -1, and without an explicit `HANDLE` argtype ctypes passes it
    as a 32-bit int in a 64-bit process, the call fails, and the measurement
    looks merely unavailable rather than wrong. Reimplementing it here reproduced
    that exact failure -- `GetProcessMemoryInfo returned 0 (GetLastError 0)` --
    which is why this function is a delegation and not a second copy.
    """
    try:
        from test_deployment_envelope import peak_memory_bytes
    except Exception as exc:
        return {"available": False,
                "why": f"cannot import peak_memory_bytes: {type(exc).__name__}"}
    try:
        m = peak_memory_bytes()
    except Exception as exc:
        return {"available": False, "why": f"{type(exc).__name__}: {exc}"}
    if not m or not m.get("working_set"):
        return {"available": False,
                "why": "the platform reported no peak working set"}
    return {"available": True,
            "source": "test_deployment_envelope.peak_memory_bytes",
            "peak_wset_bytes": int(m["working_set"]),
            "peak_pagefile_bytes": int(m.get("commit") or 0)}


# The four exhaustion flags plus `error_code`, exactly as
# `whestbench/scoring.py:1036-1044` defines a failed entry. Copied rather than
# imported because `certify_candidate` must import with nothing installed --
# but copied VERBATIM, and `test_certify_candidate.py` asserts the two lists
# still agree against the installed package, so a package change breaks a test
# instead of silently making this module's failure count disagree with the
# runner's.
_FAILURE_FLAGS = ("budget_exhausted", "time_exhausted",
                  "residual_wall_time_exhausted", "combined_budget_exhausted")

# A cap on retained failure records. 45 of 300 rows failed the first time this
# instrument ran; an unbounded list would put a traceback per row into a
# committed artifact.
_MAX_FAILURE_RECORDS = 40


def _row_failed(m):
    """The runner's own definition of a failed per-MLP entry."""
    return bool(m.get("error_code")) or any(bool(m.get(k))
                                            for k in _FAILURE_FLAGS)


def _candidate_frame(tb_lines, candidate):
    """The deepest traceback frame that is the CANDIDATE'S OWN FILE.

    Not "a file under the repository root": the four virtualenvs live under
    that root, so the installed `whestbench` and `flopscope` are under it too,
    and matching the root makes every harness fault look like ours.

    `candidate` may be None (older callers), in which case no frame is claimed
    -- refusing to answer is correct here, and is much better than answering
    "the candidate" for a harness fault.
    """
    if not candidate:
        return None
    target = os.path.normcase(os.path.abspath(candidate))
    for ln in reversed(tb_lines):
        if not ln.strip().startswith("File "):
            continue                       # a source line, not a frame
        i, j = ln.find('"'), ln.rfind('"')
        if i < 0 or j <= i:
            continue
        path = os.path.normcase(os.path.abspath(ln[i + 1:j]))
        parts = path.replace("\\", "/").split("/")
        if any(seg.startswith(".venv") or seg == "site-packages"
               for seg in parts):
            continue                       # an installed package, not ours
        if path == target:
            return ln.strip()
    return None


def _failure_record(rep, m, candidate=None):
    """Why one MLP failed, small enough to live in a committed artifact.

    The traceback's LAST line is kept, not the first: the first line is always
    `Traceback (most recent call last):`, and the last is the exception that
    actually fired.

    `deepest_candidate_frame` answers the only question a failure has to answer
    here: is the fault in the harness/host, or in the candidate's own math?

    THE FIRST VERSION OF THIS GOT IT EXACTLY BACKWARDS, and an independent
    review caught it. It matched the substrings `"estimator"` or `"arc-whest"`
    anywhere in a line. But `.venv/` lives INSIDE
    `C:/Users/samsu/Documents/arc-whest/`, so every absolute path
    `traceback.format_exc()` emits for a `whestbench` or `flopscope` frame
    contains `arc-whest` -- and `scoring.py`'s own source line
    `raw_predictions = estimator.predict(...)` contains `estimator`. A pure
    harness fault was therefore reported as a candidate fault, loudly. The one
    test guarding that direction passed only because its fixture used a
    RELATIVE path, which `format_exc()` never produces.

    So the match is now on the candidate file's own absolute path, with every
    virtualenv prefix excluded explicitly. `candidate` is threaded in from
    `main()`, which already holds it.
    """
    err = m.get("error")
    if isinstance(err, dict):
        err = err.get("message", "")
    tb = (m.get("traceback") or "").strip().splitlines()
    repo_frame = _candidate_frame(tb, candidate)
    return {
        "rep": rep,
        "mlp_index": m.get("mlp_index"),
        "mlp_name": m.get("mlp_name"),
        "error_code": m.get("error_code"),
        "error": str(err)[:200] if err else None,
        "flags": [k for k in _FAILURE_FLAGS if m.get(k)],
        "traceback_last_line": tb[-1][:200] if tb else None,
        "deepest_candidate_frame": repo_frame[:200] if repo_frame else None,
        "residual_wall_time_s": m.get("residual_wall_time_s"),
        "flops_used": m.get("flops_used"),
    }


def _residual_soak(rel, reps, n_mlps, timeout=5400):
    """Pool per-MLP residuals across `reps` official subprocess runs.

    Section 10 measures residual on two MLPs, once. Residual under Phase 2 is a
    CLIFF, not a price: one MLP over 0.4 s fails that MLP outright. What decides
    safety is therefore the TAIL, and a two-row sample has none. This runs the
    official runner repeatedly and reports the pooled worst / p99 / median.

    Nothing here is fitted, selected, or averaged into a score; the split's
    labels are read by the runner and deliberately not returned, because a soak
    is a residual instrument and must not become an accuracy one.

    TWO THINGS THIS FUNCTION USED TO GET WRONG, both found on its first real
    run (45 failed MLPs of 300 on the shipped v2.6.1 canary, `d22d0cb`):

    1. It counted failures and discarded the REASON. The runner already
       computes a five-way `failure_breakdown` and stamps every failed row with
       an `error_code` and a traceback; none of it was read, so a REFUSED
       certificate said only that something went wrong.
    2. It pooled the residual of rows THAT NEVER EXECUTED. `scoring.py`'s
       exception path appends a failed row carrying
       `"residual_wall_time_s": 0.0`. Those zeros sort to the bottom of the
       pooled sample, so a "median" at rank 150/300 with 45 failures is really
       the 41st percentile of executed work. `worst` is a max and is immune,
       which is exactly why the defect survives the number a reader checks.

    So the tail is now computed over EXECUTED rows only, and the contaminated
    figures are kept beside it as `*_all_rows` -- the correction is shown, not
    silently applied.
    """
    exe = os.path.join(os.path.dirname(sys.executable), "whest")
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    reports = []
    for i in range(reps):
        cmd = [exe, "run", "--estimator", rel,
               "--dataset", "hf://aicrowd/arc-whestbench-public-2026@v2-phase2",
               "--split", "mini", "--runner", "subprocess",
               "--n-mlps", str(n_mlps), "--json"]
        try:
            p = _run(cmd, timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            return {"ok": False, "why": f"rep {i+1} timed out after {timeout}s",
                    "reps_completed": i}
        j = (p.stdout or "").find("{")
        if j < 0:
            return {"ok": False, "reps_completed": i,
                    "why": f"rep {i+1}: {(p.stderr or p.stdout or '')[-160:]}"}
        try:
            d = json.loads(p.stdout[j:])
        except ValueError as exc:
            return {"ok": False, "reps_completed": i,
                    "why": f"rep {i+1}: unparseable report: {exc}"}
        if not d.get("results"):
            err = d.get("error") or {}
            return {"ok": False, "reps_completed": i,
                    "why": (f"rep {i+1} REFUSED at {err.get('stage')}: "
                            f"{err.get('code')} {err.get('message','')[:70]}")}
        reports.append(d)
    return _pool_residual(reports, reps, n_mlps, rel)


def _pool_residual(reports, reps, n_mlps, candidate=None):
    """The arithmetic half of the soak, separated so it can be TESTED.

    It was fused with the subprocess loop, which meant the only way to exercise
    the exclusion arithmetic was a three-hour run against the organisers' data
    -- so in practice it was never exercised at all, and shipped with the
    never-executed rows in the sample.

    Takes parsed official reports, returns the pooled residual tail. Pure: no
    subprocess, no clock, no filesystem.
    """
    res, res_all, per_rep, failures, fl = [], [], [], [], []
    failed, cap, breakdown = 0, None, {}
    for i, d in enumerate(reports):
        r = d["results"]
        cap = d["run_config"]["residual_wall_time_limit_s"]
        rows_all = [float(m["residual_wall_time_s"]) for m in r["per_mlp"]]
        # A row is EXECUTED unless the runner marked it failed. `_row_failed`
        # applies the runner's own definition rather than a second one of ours,
        # so the two can never drift apart.
        rows = [float(m["residual_wall_time_s"]) for m in r["per_mlp"]
                if not _row_failed(m)]
        # Per-row counted FLOPs across EXECUTED rows. CONTRACT WP8's
        # prediction P5 was "counted FLOPs on every executed row remain
        # exactly 137,600,326,656" -- and the soak stored no such thing, so
        # the prediction was unanswerable from its own artifact. Section 10's
        # figure is a max over TWO smoke rows, which is not "every executed
        # row". A spread of zero is the claim; recording min and max is what
        # makes it checkable.
        fl += [int(m.get("flops_used", 0)) for m in r["per_mlp"]
               if not _row_failed(m)]
        res.extend(rows)
        res_all.extend(rows_all)
        failed += int(r["n_failed_mlps"])
        for k, v in (r.get("failure_breakdown") or {}).items():
            breakdown[k] = breakdown.get(k, 0) + int(v)
        for m in r["per_mlp"]:
            if not _row_failed(m) or len(failures) >= _MAX_FAILURE_RECORDS:
                continue
            failures.append(_failure_record(i + 1, m, candidate))
        per_rep.append({
            "rep": i + 1, "n_rows": len(rows_all), "n_executed": len(rows),
            "n_failed": int(r["n_failed_mlps"]),
            "worst": max(rows) if rows else None,
            "median": _pctile(rows, 50) if rows else None})
    if not res_all:
        return {"ok": False, "why": "no residual rows were produced",
                "reps_completed": 0}
    if not res:
        # Every row failed. There is no tail of executed work to report, and
        # inventing one from zeros would be worse than saying so.
        return {"ok": False, "reps_completed": reps,
                "why": f"all {len(res_all)} rows failed; no executed residual",
                "n_failed_mlps": failed, "failure_breakdown": breakdown,
                "failures": failures}
    worst, med = max(res), _pctile(res, 50)
    med_all = _pctile(res_all, 50)
    return {
        "ok": True, "reps": reps, "n_mlps_per_rep": n_mlps,
        "n_rows": len(res_all), "n_rows_executed": len(res),
        "residual_rows_excluded_failed": len(res_all) - len(res),
        "n_failed_mlps": failed,
        "failure_breakdown": breakdown,
        "dominant_failure_mode": (max(breakdown.items(), key=lambda kv: kv[1])[0]
                                  if failed and breakdown else None),
        "failures": failures,
        "failures_truncated": failed > len(failures),
        "residual_cap": cap,
        # EXECUTED ROWS ONLY. A failed row carries residual 0.0 from
        # scoring.py's exception path; pooling it understates every percentile
        # and leaves the max untouched, which is the worst of both.
        "worst": worst, "p99": _pctile(res, 99), "median": med,
        "worst_over_median": (worst / med) if med > 0 else float("inf"),
        # Per VALIDATION.md 2b-ii: residual is reported as HEADROOM against the
        # cliff, quoted on the WORST observed row, never as a cost.
        "headroom_worst": (cap / worst) if worst > 0 else float("inf"),
        # The contaminated figures, kept so the correction is visible and a
        # reader can see how much the zeros were moving the tail.
        # The contaminated WORST matters more than the contaminated median, and
        # the first version of this kept the median and dropped the worst --
        # exactly backwards. The four exhaustion paths do NOT zero the
        # residual: `residual_wall_time_exhausted` is set on a row whose
        # residual is ABOVE the cap. So excluding failed rows makes `worst`
        # LESS conservative, and without this field a reader sees a comfortable
        # headroom with no counterpart beside it.
        "flops_executed_min": (min(fl) if fl else None),
        "flops_executed_max": (max(fl) if fl else None),
        "flops_executed_constant": (len(set(fl)) == 1 if fl else None),
        "worst_all_rows": max(res_all),
        "headroom_worst_all_rows": ((cap / max(res_all)) if max(res_all) > 0
                                    else float("inf")),
        "median_all_rows": med_all,
        "p99_all_rows": _pctile(res_all, 99),
        "median_bias_from_failed_rows": (med / med_all) if med_all > 0 else None,
        "per_rep": per_rep,
        "meter_caveat": ("LOCAL HOST. Not a grader prediction. The two graded "
                         "K2 submissions ran ~7x faster on the grading server "
                         "than on this host, and AGENTS.md refuses that as a "
                         "general safety factor for a different candidate."),
    }


def _pctile(xs, q):
    """Nearest-rank percentile: rank = ceil(q/100 * N), value = s[rank-1].

    No numpy -- this module must import anywhere. Written with `round(...)` the
    first time, which put p99 of 1..100 at 100 and p1 at 2; the unit test caught
    it. An off-by-one in a tail statistic is exactly the kind of thing that
    reads plausibly and reports the wrong cliff margin.
    """
    s = sorted(xs)
    if not s:
        raise ValueError("empty sample")
    k = max(0, min(len(s) - 1, math.ceil((q / 100.0) * len(s)) - 1))
    return s[k]


def _smoke(rel, timeout=2400):
    """`whest run --runner subprocess` on two organiser MLPs."""
    exe = os.path.join(os.path.dirname(sys.executable), "whest")
    cmd = [exe, "run", "--estimator", rel,
           "--dataset", "hf://aicrowd/arc-whestbench-public-2026@v2-phase2",
           "--split", "mini", "--runner", "subprocess", "--n-mlps", "2", "--json"]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    try:
        p = _run(cmd, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {"ok_report": False, "why": f"timed out after {timeout}s"}
    i = (p.stdout or "").find("{")
    if i < 0:
        return {"ok_report": False,
                "why": (p.stderr or p.stdout or "")[-200:]}
    try:
        d = json.loads(p.stdout[i:])
    except ValueError as exc:
        return {"ok_report": False, "why": f"unparseable report: {exc}"}
    if not d.get("results"):
        err = d.get("error") or {}
        return {"ok_report": False,
                "why": f"REFUSED at {err.get('stage')}: {err.get('code')} "
                       f"{err.get('message', '')[:80]}"}
    r, cfg = d["results"], d["run_config"]
    return {
        "ok_report": True,
        "n_failed_mlps": r["n_failed_mlps"],
        # Same attribution as the soak. A two-row smoke that fails silently is
        # exactly as unhelpful as a 300-row one that does.
        "failure_breakdown": r.get("failure_breakdown") or {},
        "failures": [_failure_record(1, m, rel) for m in r["per_mlp"]
                     if _row_failed(m)],
        "dataset": str(cfg["dataset"]["path"]),
        "width": cfg["width"], "depth": cfg["depth"],
        "lambda": cfg["lambda_flops_per_second"],
        "residual_cap": cfg["residual_wall_time_limit_s"],
        "wall_limit": cfg["wall_time_limit_s"],
        "worst_residual": max(m["residual_wall_time_s"] for m in r["per_mlp"]),
        "flops_per_mlp": max(m.get("flops_used", 0) for m in r["per_mlp"]),
        "multiplier": r["mean_score_multiplier"],
        "c_over_b": r["mean_compute_utilization"],
        # Recorded, never read as accuracy: the organisers' own labels are
        # CONFIRM-tier and nothing here may be fitted or selected on them.
        "final_layer_mse_CONFIRM_ONLY": r["final_layer_mse"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
