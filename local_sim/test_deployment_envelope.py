"""The deployment envelope: what fails a Phase-2 MLP, and what this host can prove.

P2-P0 (Phase 0 of `docs/GAME_PLAN.md`). Contract gate G4: `experiments/LOG.md`,
2026-08-24.

WHY THIS FILE EXISTS
--------------------
`local_sim/lifecycle_eval.py` models the six FAILURE paths and provokes every one
of them it can reach in-process. Three parts of the live envelope it cannot reach
at all, and until today the repository had no record of which was which:

    the 8,192 MB memory limit      -- RLIMIT_AS, set by the subprocess worker
    worker death and restart       -- a process that dies mid-suite
    the predict-response timeout   -- a runner-side wait, not an in-process clock

An evaluator that models three of six paths and says nothing about the other three
does not have a gap; it has a *hidden* gap, and a hidden gap reads as coverage.

THE FINDING THIS FILE EXISTS TO RECORD
--------------------------------------
**Windows cannot enforce the memory limit by any route.** `--runner local`
enforces nothing (`runner.py:163-167` logs "advisory in --runner local"), and
`--runner subprocess` enforces it with `resource.setrlimit(RLIMIT_AS, ...)`
(`subprocess_worker.py:172-183`) -- and the `resource` module **does not exist on
Windows**. The worker catches the failure and prints a warning; the run then
proceeds with NO memory cap at all.

**So this file reports differently on the two platforms, and that is the point.**
Where `RLIMIT_AS` exists it PROVOKES the limit and proves the path fires; where it
does not, it says NOT PROVABLE and points at the Linux evidence. A gate clause that
asks for a provocation is not satisfied by documenting why the provocation is
impossible -- the documentation was honest and it was still not the check.

Provoked on Ubuntu 24.04 / WSL2 under the same official stack, 2026-08-24
(`experiments/2026-08-24_p2_p0/linux/`): the worker reported
`RLIMIT_AS soft = 8,589,934,592` (exactly 8 GiB, so the cap really was applied), a
256 MiB control mapping SUCCEEDED, a 9 GiB anonymous reservation was REFUSED with
`OSError: [Errno 12] Cannot allocate memory` without a byte being touched, that MLP
was scored through the error path, and **the next two MLPs ran and were scored**.

So a Phase-2 candidate can be memory-safe on this laptop by construction, because
nothing here is measuring, and still be killed on the grader. The Phase-1 limit
was 65,536 MB and it already killed v2.5 (`docs/GOTCHAS.md`). The Phase-2 limit is
**8,192 MB -- eight times tighter** -- and it applies to a shape with sixteen times
the weight bytes. Memory is therefore a DESIGN constraint to be accounted for, not
a test to be passed, and this file provides the accounting instead of the test.

WHAT IT ACTUALLY MEASURES
-------------------------
1. **The enforceability matrix.** For each envelope constraint: is it enforced
   here, and by what. Derived from the platform and the installed package, never
   asserted from memory. This is the part that must never quietly become "all
   green".
2. **Worker death and restart, provoked for real** through the official
   `whest run --runner subprocess`, using a sacrificial estimator that hard-exits
   on one MLP. The grader restarts the worker and scopes the failure to that MLP
   (`runner.py:359-380`); a suite that scored every remaining MLP as failed would
   turn one bad network into a whole-submission loss.
3. **A peak-memory accounting number** at the live shape, with its limitation
   stated and its direction stated correctly. RLIMIT_AS caps *virtual address
   space*; the counters here measure *resident* and *committed* memory, both of
   which are smaller. So the measured peak is a **lower bound on the charged
   quantity**, and the headroom derived from it is therefore an **UPPER bound** --
   the most optimistic reading, never a safety margin. (An earlier revision of
   this file called that headroom a lower bound. It is not, and inverting the
   sign of the one constraint this host cannot test is the worst place to do it.)
4. **The timeout arithmetic**, which is not what it looks like: the runner waits
   `max(predict_timeout_s, wall_time_limit_s + grace)`, so under Phase 2's 120 s
   wall the 30 s `predict_timeout_s` is NOT the binding constraint.

WHAT P1-PRE CHANGED, 2026-08-25 (D1-D3 of the Phase-0 accepting review)
-----------------------------------------------------------------------
The review that closed Phase 0 carried three defects out of this file. All three
were the same shape -- a mechanism that could report success without having done
anything -- and they are closed here:

* **D1.** The matrix loop had one surviving branch whose condition was the literal
  `True`: a constraint whose witness named "this file" was counted as enforced
  without any check being consulted. It was the ONLY line covering
  `worker_death_restart`, and with `--skip-worker-death` it printed green while
  section 6 was skipped and the same constraint was reported NOT PROVABLE thirty
  lines later. This suite now EMITS its own results (`own_results()`) exactly as
  `lifecycle_eval.py --self-test --emit-json` does, and its self-provoked
  constraints are verified against those results in **section 7**, which runs
  AFTER the provocations rather than before them. When a provocation did not run,
  its witness cannot resolve the checks it names and the constraint is reported
  NOT PROVABLE -- never green.
* **D2.** `verify_witness` was handed a results object and ignored the `suite`
  the witness named, so every witness was adjudicated against the lifecycle
  artifact whatever it claimed. It now resolves the name through a
  `WitnessResolver` -- a registry of suite name -> results source -- and refuses
  an unknown suite, a registered suite with no results, and an artifact whose own
  `suite` field disagrees with the name that resolved it.
* **D3.** `--no-write` regenerated the lifecycle artifact twice and then printed
  that nothing had been touched. It now arms a module-level READ-ONLY LATCH:
  regeneration and every artifact write pass through one choke point that RAISES
  while the latch is armed, so the mode is incapable of writing rather than
  merely not asked to. The claim itself is printed only after the run has hashed
  the tracked evidence before and after and compared `git status --porcelain`.
  (The latch governs TRACKED evidence. Sections 3 and 6 still write throwaway
  candidates under `local_sim/_envelope_*_tmp*`, which is gitignored and removed
  in a `finally` -- which is precisely why `git status --porcelain` is a valid
  witness for the claim.)

`local_sim/test_envelope_readonly.py` proves the D3 claim and the D1 skip path
from OUTSIDE this file, by different logic (child process, sha256, git), so this
suite is not the only thing certifying its own read-only mode.

Run:  uv run python local_sim/test_deployment_envelope.py
      uv run python local_sim/test_envelope_readonly.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# APPEND, never insert -- local_sim/ holds numpy shims named flopscope/whestbench.
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np  # noqa: E402

import official_config  # noqa: E402

OUT_DIR = os.path.join(_ROOT, "experiments", "2026-08-24_p2_p0")
OUT = os.path.join(OUT_DIR, "deployment_envelope.json")

#: This suite's identity, as a witness may name it. It is the key under which
#: `own_results()` is registered with a `WitnessResolver`, and it is the value
#: `own_results()` stamps into its own payload -- so a witness naming this suite
#: and a resolver handing back somebody else's artifact is a SUITE MISMATCH and
#: not a silent success.
SELF_SUITE = "local_sim/test_deployment_envelope.py"

#: The other suite whose provocations this matrix cites.
LIFECYCLE_SUITE = "local_sim/lifecycle_eval.py --self-test"

_PASS = 0
_FAIL = 0
_UNPROVABLE: list = []
#: Checks this INVOCATION declined to perform -- see `declined()`. Never a pass,
#: and deliberately not mixed with the envelope's own unprovable constraints.
_DECLINED: list = []
#: Every check this run has performed, in order, machine-readably. This is what
#: turns "provoked by this file" from an assertion into something a witness can
#: be verified against -- the D1 repair. Same shape as `lifecycle_eval._RESULTS`.
_RESULTS: list = []
#: Which self-provoked sections ACTUALLY RAN. Set by the sections themselves, so
#: section 7 distinguishes "the provocation ran and its witness failed" (a
#: FAILURE) from "the provocation never ran" (NOT PROVABLE) without reading the
#: command line. A flag is an intention; this is an observation.
#:
#: A section MUST record one or the other. An ABSENT entry is a failure in
#: section 7, not a gap -- otherwise forgetting a bookkeeping line silently
#: downgrades a real failure into "not provable here", which is the softer
#: report and therefore the dangerous default.
_SECTION_RAN: dict = {}


def check(name, ok, detail=""):
    global _PASS, _FAIL
    ok = bool(ok)
    _RESULTS.append({"name": name, "ok": ok, "detail": str(detail)})
    if ok:
        _PASS += 1
    else:
        _FAIL += 1
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<62} {detail}")
    return ok


def unprovable(name, why):
    """Record an envelope CONSTRAINT this run cannot test. Never a pass.

    Reserved for the deployment envelope's own constraints -- the things the
    grader enforces and this host may not be able to. The closing block says so
    in those words, so anything else reported here would make that sentence
    false about at least one line.
    """
    _UNPROVABLE.append({"constraint": name, "why": why})
    print(f"  ----  {name:<62} NOT PROVABLE HERE: {why}")


def declined(name, why):
    """Record a CHECK this invocation cannot perform. Never a pass either.

    Different from `unprovable` and kept in a different list on purpose: a check
    the `--no-write` mode declines to run is not a gap in the evaluator's
    coverage of the grader's envelope, and filing it under a heading that reads
    "they are enforced only on the grader" would be a small lie of exactly the
    kind this slice exists to remove.
    """
    _DECLINED.append({"check": name, "why": why})
    print(f"  ~~~~  {name:<62} NOT PERFORMED IN THIS RUN: {why}")


def pending(name, why):
    """Record a constraint whose provocation has not run YET, in this run.

    Neither a pass nor a gap: section 7 adjudicates every one of these, and it
    asserts that it adjudicated all of them, so a deferred constraint cannot
    disappear between the matrix and the closure.
    """
    print(f"  ....  {name:<62} DEFERRED to section 7: {why}")


def own_results() -> dict:
    """This run's checks so far, in `lifecycle_eval._emit_results` form.

    The digest covers the `(name, ok)` pairs only -- not the details, which carry
    timings -- so two honest runs agree and a changed RESULT changes it.
    """
    checks = list(_RESULTS)
    payload = {
        "artifact": "deployment_envelope_self_results",
        "artifact_version": 1,
        "suite": SELF_SUITE,
        "passed": _PASS,
        "total": _PASS + _FAIL,
        "checks": checks,
    }
    payload["results_sha256"] = hashlib.sha256(
        json.dumps([[c["name"], c["ok"]] for c in checks],
                   sort_keys=False).encode("utf-8")).hexdigest()
    return payload


def head(t):
    print(f"\n{t}\n{'-' * len(t)}")


# --------------------------------------------------------------------------- #
# 0. the READ-ONLY LATCH -- D3
# --------------------------------------------------------------------------- #
# `--no-write` used to be a promise made in a print statement. An independent
# Skeptic followed the documented read-only protocol, and had to `git checkout`
# the artifact this file had silently regenerated -- twice -- immediately after
# reading "the artifact on disk was NOT touched".
#
# So the mode is no longer a request. It is a latch, and every path in this
# module that could touch TRACKED evidence asks the latch first and RAISES if it
# is armed. Being unable to write is a property of the process; not having been
# asked to write is a property of the caller.

class ReadOnlyRefusal(RuntimeError):
    """Raised when the read-only latch is armed and something tried to write."""


_READ_ONLY = False


def arm_read_only(value: bool = True) -> bool:
    """Arm (or disarm) the latch. Returns the PREVIOUS state.

    The return value exists so the negative tests can arm the latch, watch a
    write refuse, and restore -- a guard nobody has watched fire is a guard
    nobody has tested, which is the defect this whole slice repairs.
    """
    global _READ_ONLY
    prev = _READ_ONLY
    _READ_ONLY = bool(value)
    return prev


def read_only_armed() -> bool:
    return _READ_ONLY


def refuse_if_read_only(what: str) -> None:
    if _READ_ONLY:
        raise ReadOnlyRefusal(
            f"--no-write is armed: refusing to {what}. A read-only run that "
            f"rewrites committed evidence is the defect this latch closes.")


def write_artifact(path: str, payload: dict) -> None:
    """The ONLY place this module writes an evidence artifact."""
    refuse_if_read_only(f"write {path}")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _sha256_file(path: str) -> "str | None":
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return None


def _git_porcelain() -> "str | None":
    """`git status --porcelain`, or None where git cannot answer.

    Gitignored scratch does not appear here, which is what makes this a fair
    witness for the read-only claim: sections 3 and 6 write throwaway candidates
    under `local_sim/_envelope_*_tmp*`, and `.gitignore` covers them.
    """
    try:
        p = subprocess.run(["git", "status", "--porcelain"], cwd=_ROOT,
                           capture_output=True, text=True, timeout=120)
    except Exception:
        return None
    return p.stdout if p.returncode == 0 else None


# --------------------------------------------------------------------------- #
# check names, shared between the checks that PRODUCE them and the witnesses
# that CONSUME them
# --------------------------------------------------------------------------- #
# A witness naming a check by a string literal typed a second time is one
# rename away from `CHECK NOT FOUND`, and one careless "fix" away from a witness
# quietly pointing at nothing. They are constants, once.

WD_SURVIVED = "the suite SURVIVED a worker that hard-exited mid-run"
WD_EXACTLY_ONE = "EXACTLY one MLP failed -- not zero, not several"
WD_INDEX = "...and it is MLP 0, the one that killed the worker"
WD_CLASSIFIED = "...classified as an ERROR, not as a budget/time exhaustion"
WD_LATER_CLEAN = "...and EVERY later MLP is clean -- no traceback, no flag"
WD_LATER_SCORED = "...and every one was GENUINELY SCORED, not zeroed"

#: The envelope constraints this matrix is required to carry, pinned by name.
#: Without this, a constraint DELETED from `enforceability()` disappears from the
#: matrix, from `_UNPROVABLE`, from `_DECLINED` and from the artifact in silence:
#: every other accounting line in this file is a traversal of
#: `enf["constraints"]`, so removing an entry shrinks the checker along with the
#: checked. Section 7's `expected_self` has the same blind spot by construction.
#: A second independent Skeptic named it, and this list is the only thing here
#: that is not derived from the dict it is checking.
#:
#: Adding a constraint is a deliberate edit to this tuple. That is the point.
REQUIRED_CONSTRAINTS = (
    "flop_budget",
    "wall_time_limit_s",
    "residual_wall_time_limit_s",
    "setup_timeout_s",
    "memory_limit_mb",
    "worker_death_restart",
    "predict_response_timeout_s",
)

MEM_CAP_APPLIED = "the worker actually APPLIED the 8 GiB cap"
MEM_CONTROL = "the 256 MiB control mapping SUCCEEDED"
MEM_REFUSED = "the 9 GiB reservation was REFUSED by the limit"
MEM_UNTOUCHED = "...and it was refused, never filled -- no page was touched"
MEM_ONE_FAILED = "exactly ONE MLP failed, through the error path"
MEM_LATER_SCORED = "...and every LATER MLP ran and was scored"


# --------------------------------------------------------------------------- #
# 1. the enforceability matrix
# --------------------------------------------------------------------------- #

def _rlimit_available() -> bool:
    try:
        import resource  # noqa: F401
    except Exception:
        return False
    import resource

    return hasattr(resource, "RLIMIT_AS")


def _rlimit_usable_in_child() -> dict:
    """Can a CHILD interpreter apply the memory cap? `{"usable": bool, "why": str}`.

    Three versions of this question have been asked here and only this one can
    come out either way:

      1. `hasattr(resource, "RLIMIT_AS")` -- what `_rlimit_available()` asks, and
         what `enforced_locally` is assigned from. Comparing the matrix against
         it is an identity.
      2. an in-process `getrlimit(RLIMIT_AS)` -- better, and still agrees with
         (1) on every platform either Skeptic or I could name. An identity with
         extra steps.
      3. **this one.** `subprocess_worker.py` calls `setrlimit` in a SEPARATE
         interpreter, so whether the cap can be applied is a fact about the
         child, not about us. A child with a different interpreter, a different
         environment, or a sandbox that blocks the module answers differently
         from its parent -- and if it ever does, the matrix is claiming
         enforcement the grader's worker would not get.
    """
    src = ("import json,sys\n"
           "try:\n"
           "    import resource\n"
           "    resource.getrlimit(resource.RLIMIT_AS)\n"
           "    print(json.dumps({'usable': True, 'why': 'getrlimit(RLIMIT_AS) "
           "succeeded in the child'}))\n"
           "except Exception as e:\n"
           "    print(json.dumps({'usable': False, 'why': type(e).__name__ + ': '"
           " + str(e)[:60]}))\n")
    try:
        p = subprocess.run([sys.executable, "-c", src], capture_output=True,
                           text=True, timeout=60)
        return json.loads(p.stdout.strip().splitlines()[-1])
    except Exception as exc:  # the probe itself failed: say so, never assume
        return {"usable": None, "why": f"probe failed: {type(exc).__name__}"}


#: Where the lifecycle suite's machine-readable results live. Regenerated on
#: demand; never trusted stale.
LIFECYCLE_RESULTS = os.path.join(
    _ROOT, "experiments", "2026-08-24_p2_p0", "lifecycle_self_test.json")

#: The TRACKED evidence a `--no-write` run must leave untouched, hashed before
#: and after. Both were rewritten by the old `--no-write`; `LIFECYCLE_RESULTS`
#: twice per run.
READ_ONLY_WITNESS_PATHS = (OUT, LIFECYCLE_RESULTS)


def regenerate_lifecycle_results(path: str) -> None:
    """Re-run the lifecycle suite so `path` is this session's results, not last week's.

    Regeneration exists because a STALE results file is exactly the failure this
    witness mechanism was built to prevent: an earlier matrix "verified" the wall
    provocation against a file that existed and contained no such check.

    It is also the single most expensive thing this suite does (~49 s), and it is
    a write to TRACKED evidence -- so it asks the latch first.

    **Consequence, and `--no-write` is therefore the WEAKER mode.** Under the
    latch this does not run, so that mode adjudicates its witnesses against
    whatever `lifecycle_self_test.json` is committed on disk -- which is exactly
    the staleness the paragraph above says the witness mechanism exists to
    prevent. It is the right trade (a read-only audit that rewrites committed
    evidence is worse), the gate runs the WRITING mode, and a second independent
    Skeptic was right that nowhere said so. Now it does, here and in `--help`.
    """
    refuse_if_read_only(f"regenerate {os.path.relpath(path, _ROOT)} by re-running "
                        f"the lifecycle suite")
    subprocess.run(
        [sys.executable, os.path.join(_HERE, "lifecycle_eval.py"),
         "--self-test", "--emit-json", path],
        capture_output=True, text=True, cwd=_ROOT,
        env=dict(os.environ, PYTHONIOENCODING="utf-8"))


def _read_json(path: str) -> "dict | None":
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _mtime(path: str):
    try:
        return os.stat(path).st_mtime_ns
    except OSError:
        return None


class JsonSuite:
    """A suite whose results live in a file, optionally regenerated first."""

    def __init__(self, path: str, regenerate=None):
        self.path = path
        self.regenerate = regenerate


class LiveSuite:
    """A suite whose results exist in THIS process -- e.g. our own checks.

    Never cached: `own_results()` grows as the run proceeds, and a resolver that
    froze it at first use would adjudicate section 6's witness against a snapshot
    taken before section 6 ran. That is the D1 bug wearing a different hat.
    """

    def __init__(self, get):
        self.get = get


def default_registry() -> dict:
    """Every suite a witness in this file is allowed to name."""
    return {
        LIFECYCLE_SUITE: JsonSuite(LIFECYCLE_RESULTS, regenerate_lifecycle_results),
        SELF_SUITE: LiveSuite(own_results),
    }


class WitnessResolver:
    """Maps a witness's `suite` NAME to that suite's emitted results. D2.

    The mechanism it replaces did not do this. `_load_suite_results` always
    loaded the lifecycle artifact, whatever the witness claimed, so every witness
    was adjudicated against one suite's results and the `suite` field was
    decoration. Today's failure mode was a false NEGATIVE, so nothing was
    over-claimed -- but the docstring promised a resolving mechanism the code did
    not implement, and a witness that names one suite and is checked against
    another is not a witness.

    Four outcomes, three of them refusals:

      OK             the named suite produced results and they ARE that suite's
      UNKNOWN_SUITE  nothing is registered under that name
      MISSING        registered, but it produced nothing to verify against
      MISMATCH       the results resolved, and they declare a DIFFERENT suite
    """

    def __init__(self, registry=None, allow_regenerate: bool = True):
        self._registry = dict(default_registry() if registry is None else registry)
        self._allow_regenerate = bool(allow_regenerate)
        self._cache: dict = {}
        #: suite name -> did resolving it actually rewrite its file? Read by the
        #: caller, so "--no-write regenerated nothing" is a measurement and its
        #: mirror image ("the normal run DID regenerate") keeps it non-vacuous.
        self.regenerated: dict = {}

    def known(self) -> list:
        return sorted(self._registry)

    def resolve(self, suite) -> "tuple[str, dict | None]":
        src = self._registry.get(suite) if isinstance(suite, str) else None
        if src is None:
            return "UNKNOWN_SUITE", None
        if isinstance(src, LiveSuite):
            results = src.get()
        else:
            if suite not in self._cache:
                self._cache[suite] = self._load(suite, src)
            results = self._cache[suite]
        if results is None:
            return "MISSING", None
        declared = results.get("suite")
        if declared is not None and declared != suite:
            return "MISMATCH", results
        return "OK", results

    def _load(self, suite: str, src: JsonSuite) -> "dict | None":
        before = _mtime(src.path)
        # The latch is consulted HERE and not only at construction, so arming it
        # after a resolver exists still stops the write.
        if self._allow_regenerate and not read_only_armed() and src.regenerate:
            src.regenerate(src.path)
        self.regenerated[suite] = (_mtime(src.path) != before)
        return _read_json(src.path)


def verify_witness(witness, resolver: "WitnessResolver") -> "tuple[bool, str]":
    """Does `witness` name checks that ACTUALLY RAN and ACTUALLY PASSED?

    This is the heart of P2-P0 gate clause G4, and it replaces two weaker
    mechanisms that both failed in this slice:

      1. `check(name, True, ...)` -- a literal condition. It counted the wall
         limit as enforced while nothing provoked it.
      2. a filename resolver -- it confirmed `lifecycle_eval.py` exists, which
         was true and meaningless, because the file contained no wall provocation.

    A witness must now be a STRUCTURED claim:

        {"suite": <a results artifact>,
         "checks": [<exact check names that must be present and True>],
         "expect_detail": {<check name>: <substring its detail must contain>}}

    and every named check must be present in that suite's emitted results with
    `ok == True`. A witness that is prose, absent, or names a check that does not
    exist FAILS -- it does not pass unadjudicated.

    The `suite` is RESOLVED (D2): the name is looked up in the resolver's
    registry and the results that come back must declare themselves to be that
    same suite. Previously the field was read only to build error messages.

    `resolver` is REQUIRED and has no default. A default-constructed one would
    re-run the lifecycle suite on its first resolve -- a ~49 s write to tracked
    evidence hiding behind an omitted argument, which is the shape of the defect
    this slice is closing, one level down.

    Returns `(ok, reason)`.
    """
    if witness is None:
        return False, "NO WITNESS: nothing names a provocation"
    if not isinstance(witness, dict):
        return False, (f"UNADJUDICABLE WITNESS: {witness!r} is prose, not a "
                       "structured claim. A sentence is not evidence.")
    names = witness.get("checks") or []
    if not names:
        return False, "WITNESS NAMES NO CHECKS"
    suite = witness.get("suite")
    if not isinstance(suite, str) or not suite:
        return False, (f"WITNESS NAMES NO SUITE: {suite!r} -- a list of check "
                       "names with nothing to look them up in")
    status, results = resolver.resolve(suite)
    if status == "UNKNOWN_SUITE":
        return False, (f"UNKNOWN SUITE: {suite!r} is not a registered results "
                       f"source; known: {resolver.known()}")
    if status == "MISSING":
        return False, (f"SUITE RESULTS MISSING: {suite!r} produced "
                       "nothing to verify against")
    if status == "MISMATCH":
        return False, (f"SUITE MISMATCH: {suite!r} resolved to results emitted by "
                       f"{(results or {}).get('suite')!r} -- the witness and the "
                       "evidence are about different suites")
    by_name = {c["name"]: c for c in results.get("checks", [])}
    missing = [n for n in names if n not in by_name]
    if missing:
        return False, (f"CHECK NOT FOUND in {witness.get('suite')}: "
                       f"{missing[0]!r} -- the witness names a provocation the "
                       "suite does not contain")
    failed = [n for n in names if not by_name[n]["ok"]]
    if failed:
        return False, f"CHECK PRESENT BUT FAILED: {failed[0]!r}"
    for n, needle in (witness.get("expect_detail") or {}).items():
        if n not in by_name:
            return False, f"CHECK NOT FOUND for detail assertion: {n!r}"
        if needle not in by_name[n]["detail"]:
            return False, (f"DETAIL MISMATCH on {n!r}: expected {needle!r}, "
                           f"got {by_name[n]['detail']!r}")
    return True, (f"{len(names)} named check(s) verified in "
                  f"{witness.get('suite')} "
                  f"[{(results.get('results_sha256') or '')[:12]}]")


def _record_witness(c: dict, w, ok: bool, why: str,
                    resolver: "WitnessResolver") -> None:
    """Embed the VERIFIED RESULTS in the artifact, not just the claim.

    A committed evidence file that says `provoked_by: <name>` is a pointer; one
    that carries the named check, its `ok` flag and its detail is evidence. The
    pre-P2-P0 artifact was the former and its pointer was false.

    The rows are taken from **the suite the witness named**, resolved here the
    same way `verify_witness` resolves it. An earlier revision of this function
    read them from a fixed `lifecycle_results` instead, so a witness naming any
    third suite would have written an empty `witness_results` beside a
    `witness_verified: true` -- D2's own shape, surviving on the artifact path
    after being fixed on the verification path. Found by an independent Skeptic.
    """
    rows: list = []
    suite = w.get("suite") if isinstance(w, dict) else None
    if isinstance(suite, str):
        status, results = resolver.resolve(suite)
        if status == "OK":
            by_name = {x["name"]: x for x in (results or {}).get("checks", [])}
            rows = [by_name[n] for n in (w.get("checks") or []) if n in by_name]
    c["witness_verified"] = ok
    c["witness_reason"] = why
    c["witness_suite"] = suite
    c["witness_results"] = rows


def enforceability() -> dict:
    """Which envelope constraints this host can actually enforce, and how."""
    r = official_config.LIVE
    rl = _rlimit_available()
    return {
        "platform": sys.platform,
        "rlimit_as_available": rl,
        "constraints": {
            "flop_budget": {
                "value": r.flop_budget,
                "enforced_locally": True,
                "by": "flopscope.BudgetContext raises BudgetExhaustedError",
                "provoked_by": {
                    "suite": "local_sim/lifecycle_eval.py --self-test",
                    "checks": [
                        "L4: a 1e6 budget exhausts the FLOP budget",
                        "L4: exhausted rows are FAILED",
                        "L4: a failure is priced at multiplier EXACTLY 1.0, not the 0.1 floor",
                    ],
                    "expect_detail": {
                        "L4: a 1e6 budget exhausts the FLOP budget": "2/2 exhausted",
                    },
                },
            },
            "wall_time_limit_s": {
                "value": r.wall_time_limit_s,
                "enforced_locally": True,
                "by": "flopscope.BudgetContext wall clock -> TimeExhaustedError",
                # This entry used to read `provoked_by: "lifecycle_eval.py
                # --self-test (L4)"` while L4 provoked only the RESIDUAL limit --
                # a false provenance claim that a filename check could not catch,
                # because the filename was correct. It is now a STRUCTURED claim
                # naming the exact checks, and `verify_witness` reads that suite's
                # emitted results and requires each to be present and True.
                "provoked_by": {
                    "suite": "local_sim/lifecycle_eval.py --self-test",
                    "checks": [
                        "L4: a 1e-9 s WALL limit trips the TIME path",
                        "L4: wall-exhausted rows are exactly zeroed at multiplier 1.0",
                        "L4: ...and the WALL path is distinct from the RESIDUAL path",
                    ],
                    "expect_detail": {
                        "L4: a 1e-9 s WALL limit trips the TIME path":
                            "2/2 time_exhausted",
                        "L4: ...and the WALL path is distinct from the RESIDUAL path":
                            "residual_exhausted clear",
                    },
                },
            },
            "residual_wall_time_limit_s": {
                "value": r.residual_wall_time_limit_s,
                "enforced_locally": True,
                "by": "post-predict comparison against BudgetContext residual",
                "provoked_by": {
                    "suite": "local_sim/lifecycle_eval.py --self-test",
                    "checks": [
                        "L4: a 1e-9 s residual limit trips the residual path",
                        "L8: the 0.4 s cliff FIRES when provoked (1e-9 s limit)",
                        "L8: a residual failure is zeroed and priced at multiplier 1.0, not 0.1",
                    ],
                },
                "note": ("the LOCAL residual is a different instrument from the "
                         "graded one; the cliff is modelled faithfully, the "
                         "seconds are not transferable"),
            },
            "setup_timeout_s": {
                "value": r.setup_timeout_s,
                "enforced_locally": True,
                "by": "timed module load + setup(), plus HOST_COLD_START_S",
                "provoked_by": {
                    "suite": "local_sim/lifecycle_eval.py --self-test",
                    "checks": [
                        "a 6 s setup RAISES SetupFailure \u2014 no clean objective is produced",
                        "a RAISING setup also refuses",
                    ],
                },
            },
            "memory_limit_mb": {
                "value": r.memory_limit_mb,
                "enforced_locally": bool(rl),
                "by": ("resource.setrlimit(RLIMIT_AS) in the subprocess worker"
                       if rl else "NOTHING -- the `resource` module does not exist "
                                  "on this platform, so subprocess_worker.py's "
                                  "setrlimit call is caught and skipped, and "
                                  "--runner local is advisory by design"),
                # Where RLIMIT_AS exists this file provokes the limit itself, so
                # the witness is STRUCTURED and names section 5's own checks --
                # it used to read "this file, section 5", which is prose, and
                # prose took the literal-True branch that D1 removed. Where
                # RLIMIT_AS does NOT exist, `enforced_locally` is False and the
                # constraint is reported NOT PROVABLE, so the descriptive string
                # below is never adjudicated as a witness.
                "provoked_by": ({
                    "suite": SELF_SUITE,
                    "checks": [MEM_CAP_APPLIED, MEM_CONTROL, MEM_REFUSED,
                               MEM_UNTOUCHED, MEM_ONE_FAILED, MEM_LATER_SCORED],
                    "expect_detail": {MEM_REFUSED: "Cannot allocate memory"},
                } if rl else
                    "experiments/2026-08-24_p2_p0/linux/ -- provoked on "
                    "Ubuntu 24.04 under the same official stack, since "
                    "it cannot be provoked on this platform"),
                "accounting_instead": ("peak working set and peak commit, measured "
                                       "below; both are LOWER bounds on the "
                                       "RLIMIT_AS charge, so the headroom they "
                                       "imply is an UPPER bound"),
            },
            "worker_death_restart": {
                "value": None,
                "enforced_locally": True,
                "by": "whest run --runner subprocess restarts a dead worker",
                # D1. This entry used to be `{"suite": "this file, section 6"}`,
                # which the matrix loop matched with `startswith("this file")`
                # and then passed with the literal `True` -- the ONLY line
                # covering this constraint, and green even when section 6 was
                # skipped. It now names section 6's actual checks against THIS
                # suite's emitted results, and section 7 verifies them AFTER
                # section 6 has had its chance to produce them.
                "provoked_by": {
                    "suite": SELF_SUITE,
                    "checks": [WD_SURVIVED, WD_EXACTLY_ONE, WD_INDEX,
                               WD_CLASSIFIED, WD_LATER_CLEAN, WD_LATER_SCORED],
                    "expect_detail": {
                        WD_EXACTLY_ONE: "n_failed_mlps = 1",
                        WD_INDEX: "failed indices [0]",
                    },
                },
            },
            "predict_response_timeout_s": {
                "value": r.predict_timeout_s,
                "enforced_locally": False,
                "by": ("runner-side wait of max(predict_timeout_s, "
                       "wall_time_limit_s + grace); under Phase 2's 120 s wall "
                       "the 30 s predict_timeout is NOT the binding constraint"),
                "provoked_by": None,
            },
        },
    }


# --------------------------------------------------------------------------- #
# 2. peak memory, measured -- and labelled with what it is not
# --------------------------------------------------------------------------- #

def peak_memory_bytes() -> dict:
    """Peak memory of THIS process: `{"working_set": ..., "commit": ...}`.

    Two numbers, because they are not the same thing and only one of them is a
    fair analogue of what the grader caps:

      * **working set** -- RESIDENT memory. What `ru_maxrss` reports on POSIX.
      * **commit** (`PeakPagefileUsage`) -- private committed bytes, the closest
        Windows counter to the process's own address-space charge, and therefore
        the closer analogue of what `RLIMIT_AS` bounds.

    `RLIMIT_AS` caps VIRTUAL ADDRESS SPACE, which includes mapped files and every
    reservation, so it is normally larger than either of these. Both numbers here
    are LOWER BOUNDS on what the grader's limit would see and must be quoted as
    such. Missing keys mean the platform would not report them.
    """
    if sys.platform != "win32":
        try:
            import resource

            # ru_maxrss is KiB on Linux, bytes on macOS.
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            v = int(rss * 1024) if sys.platform.startswith("linux") else int(rss)
            return {"working_set": v}
        except Exception:
            return {}
    import ctypes
    from ctypes import wintypes

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    fn = getattr(ctypes.windll.kernel32, "K32GetProcessMemoryInfo", None)
    if fn is None:  # pre-Win8 it lives in psapi.dll
        try:
            fn = ctypes.WinDLL("psapi").GetProcessMemoryInfo
        except Exception:
            return {}
    # argtypes/restype are NOT optional here. GetCurrentProcess() returns the
    # pseudo-handle -1; without an explicit HANDLE argtype ctypes passes it as a
    # 32-bit int on a 64-bit process and the call fails silently, returning None
    # for a measurement that looks merely unavailable rather than wrong.
    fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
                   wintypes.DWORD]
    fn.restype = wintypes.BOOL
    ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    if not fn(handle, ctypes.byref(counters), counters.cb):
        return {}
    return {
        "working_set": int(counters.PeakWorkingSetSize),
        "commit": int(counters.PeakPagefileUsage),
    }


def measure_live_shape_memory() -> dict:
    """Run one `predict` at the live shape in a FRESH process and report its peak.

    A fresh process is the point: measuring in an interpreter that has already run
    a test suite reports the suite's peak, not the candidate's.
    """
    src = textwrap.dedent(
        """
        import json, os, sys
        _ROOT = sys.argv[1]
        sys.path = [p for p in sys.path
                    if os.path.abspath(p or ".") != os.path.join(_ROOT, "local_sim")]
        for p in (os.path.join(_ROOT, "local_sim"), _ROOT):
            if p not in sys.path:
                sys.path.append(p)
        import numpy as np
        import flopscope as flops
        import flopscope.numpy as fnp
        from test_deployment_envelope import peak_memory_bytes
        from mint_provenance import canonical_mint
        import official_config, harness, importlib.util as u

        r = official_config.LIVE
        base = peak_memory_bytes()
        w, seed, _ = canonical_mint(r.width, r.depth, 5_150_900)
        mlp = harness.MLP(weights=[fnp.array(x) for x in w],
                          width=r.width, depth=r.depth, seed=seed)
        spec = u.spec_from_file_location("cand", os.path.join(_ROOT, "estimator.py"))
        mod = u.module_from_spec(spec); spec.loader.exec_module(mod)
        est = mod.Estimator()
        class C:
            width, depth, flop_budget, seed = r.width, r.depth, r.flop_budget, 0
        try:
            est.setup(C())
        except Exception:
            pass
        with flops.BudgetContext(flop_budget=r.flop_budget, quiet=True) as ctx:
            est.predict(mlp, r.flop_budget)
        print(json.dumps({"baseline": base,
                          "peak": peak_memory_bytes(),
                          "weight_bytes": int(sum(x.nbytes for x in w))}))
        """
    )
    tmp = os.path.join(_HERE, "_envelope_mem_tmp.py")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(src)
        p = subprocess.run([sys.executable, tmp, _ROOT], capture_output=True,
                           text=True, cwd=_ROOT)
        if p.returncode != 0:
            return {"error": p.stderr[-2000:]}
        return json.loads(p.stdout.strip().splitlines()[-1])
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# --------------------------------------------------------------------------- #
# 3. worker death and restart, through the OFFICIAL subprocess runner
# --------------------------------------------------------------------------- #

_SACRIFICIAL = '''"""A deliberately fatal estimator: hard-exits the worker EXACTLY ONCE.

Used by local_sim/test_deployment_envelope.py to provoke the grader's worker
restart path. `os._exit` skips every finally/atexit hook, which is the closest
reachable analogue of an OOM kill or a segfault.

The "exactly once" is the whole difficulty, and getting it wrong makes the runner
look broken. A module-global call counter does NOT work: the replacement worker
imports this file fresh, its counter starts at 0 again, and it kills itself on ITS
first MLP too -- so every MLP in the suite dies and the runner appears not to
recover at all. The state has to outlive the process, so it lives in a SENTINEL
FILE whose path arrives by environment variable.
"""
import os

import flopscope.numpy as fnp
from whestbench.sdk import BaseEstimator

_SENTINEL = os.environ.get("ARC_WHEST_SACRIFICE_SENTINEL", "")


class Estimator(BaseEstimator):
    """Must SUBCLASS BaseEstimator -- the loader discovers candidates by subclass,
    not by name, and refuses a file with none.
    """

    def setup(self, context):
        pass

    def predict(self, mlp, budget):
        if _SENTINEL and not os.path.exists(_SENTINEL):
            with open(_SENTINEL, "w", encoding="utf-8") as fh:
                fh.write(str(int(mlp.seed)))
            os._exit(9)
        return fnp.zeros((len(mlp.weights), mlp.width), dtype=fnp.float32)
'''


#: The sacrificial estimator that provokes RLIMIT_AS. Kept as a repository FILE
#: rather than an embedded string so the Linux transcript and this suite run the
#: byte-identical thing -- two copies of a probe drift, and then the transcript
#: stops being evidence about the code that ships.
MEMORY_SACRIFICE = os.path.join(
    _ROOT, "experiments", "2026-08-24_p2_p0", "linux", "sacrificial_memory.py")

#: Recorded on Ubuntu 24.04 / WSL2, whestbench 0.16.0 / flopscope 0.12.0, 2026-08-24.
#: Cited on platforms that cannot run the provocation themselves.
LINUX_EVIDENCE = os.path.join(
    _ROOT, "experiments", "2026-08-24_p2_p0", "linux",
    "rlimit_as_worker_report.json")


def provoke_memory_limit(n_mlps: int = 3, request_gib: float = 9.0) -> dict:
    """Reserve `request_gib` of ADDRESS SPACE under the 8 GiB cap, and never touch it.

    `RLIMIT_AS` bounds virtual address space, so the kernel refuses the mapping
    outright -- no page is faulted in and no memory is filled. Requesting address
    space is precisely what the limit governs, which makes this the cheapest
    honest way to exercise it.

    Returns the runner's report plus the worker's own view of its limit, written
    before either mapping is attempted so the cap is proven applied rather than
    assumed.
    """
    sentinel = os.path.join(_HERE, "_envelope_mem_sentinel_tmp")
    report = os.path.join(_HERE, "_envelope_mem_report_tmp.json")
    for f in (sentinel, report):
        if os.path.exists(f):
            os.remove(f)
    env = dict(
        os.environ,
        PYTHONIOENCODING="utf-8",
        ARC_WHEST_MEM_SENTINEL=sentinel,
        ARC_WHEST_MEM_REPORT=report,
        ARC_WHEST_MEM_REQUEST=str(int(request_gib * 1024 ** 3)),
        ARC_WHEST_MEM_CONTROL=str(256 * 1024 ** 2),
    )
    try:
        p = subprocess.run(
            [_whest_exe(), "run", "--estimator", MEMORY_SACRIFICE,
             "--dataset", "hf://aicrowd/arc-whestbench-public-2026@v2-phase2",
             "--split", "mini", "--runner", "subprocess",
             "--n-mlps", str(n_mlps), "--json"],
            capture_output=True, text=True, cwd=_ROOT, env=env)
        out: dict = {}
        i = p.stdout.find("{")
        if i < 0:
            return {"error": (p.stderr or p.stdout)[-2000:]}
        doc = json.loads(p.stdout[i:])
        if not doc.get("results"):
            return {"refused": doc.get("error") or doc}
        out["report"] = doc
        if os.path.exists(report):
            with open(report, encoding="utf-8") as fh:
                out["worker"] = json.load(fh)
        return out
    finally:
        for f in (sentinel, report):
            if os.path.exists(f):
                os.remove(f)


def _whest_exe() -> str:
    """The `whest` console script beside the running interpreter."""
    d = os.path.dirname(sys.executable)
    for name in ("whest.exe", "whest"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return "whest"


def provoke_worker_death(n_mlps: int = 3) -> dict:
    """Run the official subprocess runner over a candidate that kills its worker."""
    cand = os.path.join(_HERE, "_envelope_sacrificial_tmp.py")
    sentinel = os.path.join(_HERE, "_envelope_sacrifice_sentinel_tmp")
    try:
        if os.path.exists(sentinel):
            os.remove(sentinel)
        with open(cand, "w", encoding="utf-8") as fh:
            fh.write(_SACRIFICIAL)
        env = dict(os.environ, PYTHONIOENCODING="utf-8",
                   ARC_WHEST_SACRIFICE_SENTINEL=sentinel)
        # The `whest` console script, not `python -m whestbench.cli`: the module
        # has no __main__ guard, so `-m` exits 0 having done nothing, which reads
        # as an empty report rather than as an error.
        p = subprocess.run(
            [_whest_exe(), "run",
             "--estimator", cand,
             "--dataset", "hf://aicrowd/arc-whestbench-public-2026@v2-phase2",
             "--split", "mini", "--runner", "subprocess",
             "--n-mlps", str(n_mlps), "--json"],
            capture_output=True, text=True, cwd=_ROOT, env=env,
        )
        out = p.stdout
        i = out.find("{")
        if i < 0:
            return {"error": (p.stderr or out)[-3000:]}
        doc = json.loads(out[i:])
        if not doc.get("results"):
            # The runner REFUSED the whole run -- typically SETUP_TIMEOUT, which
            # is an environment condition and not an estimator defect. Return it
            # as a refusal rather than a report, because the caller reading
            # `doc["results"]` would otherwise die with a KeyError and report an
            # environment hiccup as a crashed test suite. That happened once.
            return {"refused": doc.get("error") or doc}
        return {"report": doc}
    finally:
        for p in (cand, sentinel):
            if os.path.exists(p):
                os.remove(p)


# --------------------------------------------------------------------------- #

def _flipped_lifecycle(results: dict, name: str) -> dict:
    """A copy of `results` with one check's `ok` turned False. Test fixture."""
    import copy

    out = copy.deepcopy(results)
    for c in out.get("checks", []):
        if c["name"] == name:
            c["ok"] = False
    return out


def read_only_negative_tests() -> None:
    """Prove the read-only latch REFUSES, rather than merely not being asked. D3.

    Both write paths in this module are exercised with the latch armed, and each
    must raise AND leave no file behind. This runs in both modes: in `--no-write`
    the latch is already armed and these simply re-arm it.

    It writes to a system temp directory, never into the repository, so the test
    of the read-only mode is itself read-only with respect to the worktree.
    """
    head("0b. NEGATIVE TESTS -- the read-only latch refuses both write paths")
    tmpdir = tempfile.mkdtemp(prefix="arc_whest_ro_")
    try:
        target = os.path.join(tmpdir, "would_have_been_written.json")
        prev = arm_read_only(True)
        try:
            raised = ""
            try:
                regenerate_lifecycle_results(target)
            except ReadOnlyRefusal as exc:
                raised = str(exc)
            check("REFUSED: regenerating a suite's results while the latch is armed",
                  bool(raised) and not os.path.exists(target),
                  raised[:90] or "NO REFUSAL RAISED")

            raised = ""
            try:
                write_artifact(target, {"this": "must not reach disk"})
            except ReadOnlyRefusal as exc:
                raised = str(exc)
            check("REFUSED: writing an evidence artifact while the latch is armed",
                  bool(raised) and not os.path.exists(target),
                  raised[:90] or "NO REFUSAL RAISED")

            # ...and the latch is not a one-way door that quietly disables the
            # writer forever: with it DOWN the same call writes. Without this the
            # two refusals above could pass on a writer that is simply broken.
            #
            # In a `--no-write` run the latch was armed BEFORE this function, and
            # lowering it mid-run -- even to write into a system temp directory --
            # would make the read-only claim depend on a promise to put it back.
            # So that run reports the control as NOT PROVABLE and the writing
            # run performs it. An absent check is declared, never assumed.
            if prev:
                declined(
                    "the writer's positive control",
                    "the latch was armed before this suite started (--no-write); "
                    "lowering it here would make the read-only claim conditional "
                    "on restoring it. Exercised in the writing mode instead.")
            else:
                arm_read_only(False)
                write_artifact(target, {"this": "is the positive control"})
                check("...and with the latch DOWN the same writer DOES write",
                      os.path.exists(target),
                      f"wrote {os.path.basename(target)} to a system temp dir")
        finally:
            arm_read_only(prev)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def witness_negative_tests(resolver: "WitnessResolver") -> int:
    """Prove `verify_witness` REFUSES every way a witness can be worthless.

    A guard nobody has watched fail is a guard nobody has tested. Both mechanisms
    this one replaces passed happily while the wall-time path was provoked
    nowhere, so this suite exists to make the refusals visible and permanent.

    The first case is the exact witness that shipped in
    `deployment_envelope.json` and was false: it named a real file and a real
    suite, and the check it pointed at did not exist.

    P1-PRE (D2): every case now goes through a **`WitnessResolver`**. The
    previous version reached the `SUITE RESULTS MISSING` and `PRESENT BUT FAILED`
    branches by handing `verify_witness` a doctored results object behind the
    resolver's back, which tested the branch and not the mechanism -- and the
    mechanism was the part that did not exist. Where a case needs unusual
    results, it supplies them by REGISTERING a source under a suite name, which
    is the same public path production code uses.
    """
    global _PASS, _FAIL
    head("0. NEGATIVE TESTS -- every way a witness can be worthless, refused")

    status, results = resolver.resolve(LIFECYCLE_SUITE)
    if status != "OK":
        check("the lifecycle results are available to test against", False,
              f"{status}: could not produce them")
        return 1

    real = "L4: a 1e-9 s WALL limit trips the TIME path"

    #: A suite name that is registered but whose artifact does not exist, so
    #: MISSING is reached through `resolve()` and not by passing None past it.
    ABSENT = os.path.join(tempfile.gettempdir(), "arc_whest_absent_results.json")
    missing_resolver = WitnessResolver(
        registry={LIFECYCLE_SUITE: JsonSuite(ABSENT, regenerate=None)})
    #: The lifecycle artifact registered under somebody ELSE's name: it resolves,
    #: and then declares a different suite. That is a MISMATCH, and before P1-PRE
    #: it was invisible because the `suite` field was never read.
    IMPOSTOR = "local_sim/impostor_suite.py"
    mismatch_resolver = WitnessResolver(
        registry={IMPOSTOR: JsonSuite(LIFECYCLE_RESULTS, regenerate=None)})
    failed_resolver = WitnessResolver(
        registry={LIFECYCLE_SUITE:
                  LiveSuite(lambda: _flipped_lifecycle(results, real))})
    #: THIS suite's own results, but from a run in which section 6 never
    #: happened. This is the D1 repair's own falsifier: it is exactly the state
    #: `--skip-worker-death` produces, and the worker-restart witness must refuse
    #: it. If this case ever passes, the self-witness has become the literal
    #: `True` again.
    empty_self_resolver = WitnessResolver(
        registry={SELF_SUITE: LiveSuite(
            lambda: {"suite": SELF_SUITE, "checks": [], "results_sha256": ""})})

    worker_witness = enforceability()["constraints"]["worker_death_restart"]["provoked_by"]

    cases = [
        (
            "the ORIGINAL FALSE witness -- prose naming a real file, no check",
            "local_sim/lifecycle_eval.py --self-test (L4)",
            "UNADJUDICABLE", resolver,
        ),
        (
            "a missing witness (None)",
            None,
            "NO WITNESS", resolver,
        ),
        (
            "a witness naming NO checks",
            {"suite": LIFECYCLE_SUITE, "checks": []},
            "NAMES NO CHECKS", resolver,
        ),
        (
            "a witness naming checks but NO suite to look them up in",
            {"checks": [real]},
            "NAMES NO SUITE", resolver,
        ),
        (
            "a witness naming a suite NOBODY REGISTERED",
            {"suite": "local_sim/does_not_exist.py", "checks": [real]},
            "UNKNOWN SUITE", resolver,
        ),
        (
            "a REGISTERED suite whose results file does not exist",
            {"suite": LIFECYCLE_SUITE, "checks": [real]},
            "SUITE RESULTS MISSING", missing_resolver,
        ),
        (
            "results that resolve but DECLARE A DIFFERENT SUITE",
            {"suite": IMPOSTOR, "checks": [real]},
            "SUITE MISMATCH", mismatch_resolver,
        ),
        (
            "a WRONG check identifier inside a real suite",
            {"suite": LIFECYCLE_SUITE,
             "checks": ["L4: a 1e-9 s WALL limit trips the TIME path (typo)"]},
            "CHECK NOT FOUND", resolver,
        ),
        (
            "a real check whose RESULT is false",
            {"suite": LIFECYCLE_SUITE, "checks": [real]},
            "PRESENT BUT FAILED", failed_resolver,
        ),
        (
            "a real check whose DETAIL contradicts the claim",
            {"suite": LIFECYCLE_SUITE, "checks": [real],
             "expect_detail": {real: "7/7 time_exhausted"}},
            "DETAIL MISMATCH", resolver,
        ),
        (
            "unadjudicable prose presented as an enforcement witness",
            "it is enforced, trust me",
            "UNADJUDICABLE", resolver,
        ),
        (
            "THE SELF-WITNESS, against a run in which section 6 never ran",
            worker_witness,
            "CHECK NOT FOUND", empty_self_resolver,
        ),
    ]

    for label, witness, expect, res in cases:
        ok, why = verify_witness(witness, res)
        check(f"REFUSED: {label}", (not ok) and expect in why,
              why[:96] if why else "")

    # ...and the positive controls, so this suite cannot pass by refusing
    # everything. Two, because there are now two KINDS of source: a suite whose
    # results are a file on disk, and one whose results are this process.
    ok, why = verify_witness(
        {"suite": LIFECYCLE_SUITE, "checks": [real],
         "expect_detail": {real: "2/2 time_exhausted"}}, resolver)
    check("ACCEPTED: the real wall witness, with its real detail", ok, why[:96])

    ok, why = verify_witness(
        {"suite": SELF_SUITE,
         "checks": ["REFUSED: a missing witness (None)"],
         "expect_detail": {"REFUSED: a missing witness (None)": "NO WITNESS"}},
        resolver)
    check("ACCEPTED: a LIVE witness against this suite's own emitted results",
          ok, why[:96])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--skip-worker-death", action="store_true",
                    help="skip section 3 (it runs the official runner, ~30 s)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--no-write", action="store_true",
                    help=("run the checks and write NOTHING. An auditor re-running "
                          "this file should not have to `git checkout` the "
                          "committed artifact afterwards -- which is exactly what "
                          "an independent Skeptic had to do, and it is the kind of "
                          "side effect that makes a read-only protocol untrue. "
                          "NOTE: this is the WEAKER mode. Suppressing the write "
                          "also suppresses REGENERATION, so witnesses are "
                          "adjudicated against the lifecycle results committed on "
                          "disk rather than against this session's. The gate runs "
                          "the writing mode."))
    args = ap.parse_args()

    r = official_config.LIVE
    ro_before: dict = {}
    ro_git_before = None
    if args.no_write:
        # ARM FIRST, before anything can run. The latch is what makes the mode
        # incapable of writing; the snapshot below is what makes the CLAIM
        # verifiable rather than asserted.
        arm_read_only(True)
        ro_before = {p: _sha256_file(p) for p in READ_ONLY_WITNESS_PATHS}
        ro_git_before = _git_porcelain()

    print("=" * 96)
    print("DEPLOYMENT ENVELOPE -- what fails a Phase-2 MLP, and what this host can prove")
    print("=" * 96)

    resolver = WitnessResolver()
    witness_negative_tests(resolver)
    read_only_negative_tests()

    head("1. the enforceability matrix")
    enf = enforceability()
    print(f"  platform {enf['platform']}   RLIMIT_AS available: "
          f"{enf['rlimit_as_available']}")
    # A CONDITION, not the literal `True`. This loop used to read
    #     check(f"{name} is enforced here", True, str(c["value"]))
    # which cannot fail, and it counted FIVE constraints as passing -- including
    # `wall_time_limit_s`, which at the time was provoked NOWHERE in the
    # repository, while the artifact recorded `provoked_by:
    # "local_sim/lifecycle_eval.py --self-test (L4)"`. L4 set a tiny RESIDUAL
    # limit and never a tiny WALL limit: two different constants, one untested,
    # and a green line asserting otherwise. Found by a final Skeptic pass after
    # two earlier reviews of the same slice missed it, and it is this
    # repository's signature failure class (docs/GOTCHAS.md).
    #
    # What is checked now is the thing that was being assumed: a constraint
    # claiming local enforcement must NAME a provocation, and that named
    # provocation must resolve, through the registry, to a check that RAN and
    # PASSED in the suite the witness names.
    #
    # D1: there is no longer an exemption for constraints this file provokes
    # itself. Their sections have not run yet at this point, so they are
    # DEFERRED to section 7 -- deferred, not passed, and section 7 asserts that
    # it adjudicated every one of them.
    _, lifecycle_results = resolver.resolve(LIFECYCLE_SUITE)
    enf["witness_suite_sha256"] = (lifecycle_results or {}).get("results_sha256")
    deferred: list = []
    for name, c in enf["constraints"].items():
        if not c["enforced_locally"]:
            unprovable(name, c["by"])
            continue
        w = c.get("provoked_by")
        if isinstance(w, dict) and w.get("suite") == SELF_SUITE:
            deferred.append((name, c))
            pending(name, "provoked by this suite, which has not run it yet")
            continue
        ok, why = verify_witness(w, resolver)
        _record_witness(c, w, ok, why, resolver)
        check(f"{name}: its named provocation RAN and PASSED", ok,
              f"{c['value']}  <- {why}")
    # The matrix carries every constraint it is REQUIRED to carry. This is the
    # one line here whose right-hand side is not derived from the dict on its
    # left, so it is the only one that survives a constraint being deleted.
    present = set(enf["constraints"])
    check("the matrix carries every REQUIRED envelope constraint",
          present >= set(REQUIRED_CONSTRAINTS),
          f"{len(present)} present; missing "
          f"{sorted(set(REQUIRED_CONSTRAINTS) - present) or 'none'}"
          + (f"; unpinned extras {sorted(present - set(REQUIRED_CONSTRAINTS))}"
             if present - set(REQUIRED_CONSTRAINTS) else ""))
    # The matrix's memory claim, checked against a probe run in a CHILD process.
    #
    # It used to read `enforced_locally == _rlimit_available()`, an identity --
    # `enforced_locally` is assigned `bool(rl)` from that same call. Replacing it
    # with an in-process `getrlimit` call was better and still had no reachable
    # falsifier: on either platform both answers agree, as a second Skeptic
    # pointed out. The question that CAN come out either way is the one the
    # grader actually asks -- can the WORKER apply the cap? -- because the worker
    # is a separate interpreter, and `subprocess_worker.py` calls `setrlimit`
    # there, not here.
    probe = _rlimit_usable_in_child()
    claim = enf["constraints"]["memory_limit_mb"]["enforced_locally"]
    check("the matrix's memory claim matches a probe run in a CHILD interpreter",
          claim == probe["usable"],
          f"matrix says {claim}; child says {probe['usable']} ({probe['why']})")
    # NOT `len(_UNPROVABLE) > 0`, and no longer `bool(unenforced) and ...`:
    # `predict_response_timeout_s` is hard-coded `enforced_locally: False`, so
    # BOTH of those can never be false. That vacuous conjunct was carried into
    # the replacement and a second Skeptic caught it there. What is left is the
    # half that can fail: a matrix loop that skipped a constraint leaves it in
    # neither block, and the pinned list above is what proves the constraint was
    # there to skip.
    unenforced = {n for n, c in enf["constraints"].items()
                  if not c["enforced_locally"]}
    reported = {u["constraint"] for u in _UNPROVABLE}
    check("every constraint the matrix marks unenforceable was REPORTED as such",
          unenforced <= reported,
          f"{sorted(unenforced)} marked; missing from the report: "
          f"{sorted(unenforced - reported) or 'none'}")
    # H6. The two halves of the read-only claim, and they are mutually
    # exclusive: exactly one fires per run. Without the writing-mode half,
    # "--no-write changed nothing" could be true because nothing ever writes.
    if read_only_armed():
        check("--no-write: resolving the lifecycle suite regenerated NOTHING",
              resolver.regenerated.get(LIFECYCLE_SUITE) is False,
              f"regenerated={resolver.regenerated}")
    else:
        check("the lifecycle results WERE regenerated for this run "
              "(so the --no-write refusal is not vacuous)",
              resolver.regenerated.get(LIFECYCLE_SUITE) is True,
              f"regenerated={resolver.regenerated}")

    head("2. the timeout arithmetic is not what it looks like")
    from whestbench.runner import ResourceLimits, _predict_response_timeout_s
    lim = ResourceLimits(
        flop_budget=r.flop_budget,
        predict_timeout_s=r.predict_timeout_s,
        memory_limit_mb=r.memory_limit_mb,
        setup_timeout_s=r.setup_timeout_s,
        wall_time_limit_s=r.wall_time_limit_s,
        residual_wall_time_limit_s=r.residual_wall_time_limit_s,
    )
    eff = _predict_response_timeout_s(lim)
    check("the runner waits max(predict_timeout, wall + grace), not predict_timeout",
          eff > r.predict_timeout_s,
          f"{eff:.1f} s effective vs predict_timeout_s {r.predict_timeout_s:.1f} s")
    check("so under Phase 2 the 120 s WALL binds, not the 30 s predict timeout",
          eff >= r.wall_time_limit_s,
          f"wall {r.wall_time_limit_s:.0f} s")

    head("3. peak memory at the live shape -- accounting, not enforcement")
    mem = measure_live_shape_memory()
    mem_report: dict = {"measurement": mem, "limit_mb": r.memory_limit_mb}
    if "error" in mem or not mem.get("peak"):
        check("a peak-memory measurement was obtained", False,
              str(mem.get("error", mem))[:200])
    else:
        MB = 1024 * 1024
        peak = mem["peak"]
        ws_mb = peak["working_set"] / MB
        # `commit` is the closer analogue of what RLIMIT_AS bounds; fall back to
        # working set where the platform does not report it.
        binding_mb = peak.get("commit", peak["working_set"]) / MB
        weights_mb = mem["weight_bytes"] / MB
        # SIGN, and an independent Skeptic caught it pointing the wrong way.
        # `peak_commit` is a LOWER bound on the quantity RLIMIT_AS charges (that
        # bounds VIRTUAL ADDRESS SPACE, which is larger). Dividing a fixed limit by
        # a LOWER bound on the denominator therefore yields an UPPER bound on the
        # headroom -- the most OPTIMISTIC reading available, not the floor.
        # Labelling it a lower bound inverted the safety direction of the one
        # constraint this host cannot test at all.
        headroom_ub = r.memory_limit_mb / binding_mb
        mem_report.update({"peak_working_set_mb": ws_mb,
                           "peak_commit_mb": binding_mb,
                           "weights_mb": weights_mb,
                           "headroom_upper_bound": headroom_ub})
        print(f"  weights alone            {weights_mb:>10.1f} MB "
              f"({r.depth} x {r.width}x{r.width} float32)")
        print(f"  peak WORKING SET         {ws_mb:>10.1f} MB   (resident)")
        print(f"  peak COMMIT              {binding_mb:>10.1f} MB   "
              f"(a LOWER bound on the RLIMIT_AS charge)")
        print(f"  Phase-2 limit            {r.memory_limit_mb:>10.1f} MB")
        check("the shipped file's peak memory fits the Phase-2 limit",
              binding_mb < r.memory_limit_mb,
              f"at most {headroom_ub:.1f}x headroom -- an UPPER BOUND, i.e. the "
              f"most optimistic reading. RLIMIT_AS caps VIRTUAL address space, "
              f"which exceeds both counters above by an unmeasured amount")
        check("...and the peak is at least as large as the weights alone",
              binding_mb > weights_mb,
              f"{binding_mb:.1f} MB > {weights_mb:.1f} MB (a peak below the "
              f"weights would mean the probe never materialised them)")

    # ---------------------------------------------------------------- #
    # 5. the memory limit -- PROVOKED where the limit exists
    # ---------------------------------------------------------------- #
    mem_gate: dict = {}
    head("5. the 8 GiB memory limit -- provoked, or explicitly not provable here")
    if not _rlimit_available():
        _SECTION_RAN["memory_limit_mb"] = False
        unprovable(
            "memory_limit_mb (provocation)",
            "no RLIMIT_AS on this platform; provoked instead on Ubuntu 24.04 "
            "under the same official stack -- see "
            "experiments/2026-08-24_p2_p0/linux/")
        if os.path.exists(LINUX_EVIDENCE):
            with open(LINUX_EVIDENCE, encoding="utf-8") as fh:
                lin = json.load(fh)
            mem_gate = {"provoked_here": False, "linux_evidence": lin}
            print(f"      Linux worker saw RLIMIT_AS soft = "
                  f"{lin.get('rlimit_as_soft'):,} "
                  f"({lin.get('rlimit_as_soft_gib')} GiB)")
            print(f"      control {lin.get('control_mmap')}   "
                  f"9 GiB request -> {lin.get('provocation_mmap')}")
            check("the Linux evidence shows the cap was actually APPLIED",
                  lin.get("rlimit_as_soft") == r.memory_limit_mb * 1024 * 1024,
                  f"{lin.get('rlimit_as_soft'):,} == "
                  f"{r.memory_limit_mb * 1024 * 1024:,}")
            check("...the control mapping SUCCEEDED, so the mechanism works",
                  lin.get("control_mmap") == "OK", str(lin.get("control_mmap")))
            check("...and the over-limit reservation was REFUSED",
                  bool(lin.get("provocation_refused")),
                  str(lin.get("provocation_mmap")))
        else:
            mem_gate = {"provoked_here": False, "linux_evidence": None}
            check("Linux evidence for the memory path is present", False,
                  f"missing {os.path.relpath(LINUX_EVIDENCE, _ROOT)}")
    else:
        # The section RAN. Section 7 reads this to tell "the provocation ran and
        # its witness failed" (a FAILURE) from "the provocation never ran" (NOT
        # PROVABLE) -- from an observation, not from the command line.
        _SECTION_RAN["memory_limit_mb"] = True
        d = provoke_memory_limit()
        if "error" in d:
            check("the official runner produced a report", False,
                  d["error"][:300])
            mem_gate = {"provoked_here": True, "error": d["error"][:2000]}
        elif "refused" in d:
            mem_gate = {"provoked_here": True, "refused": d["refused"]}
            check("the official runner accepted the run", False,
                  f"REFUSED: {(d['refused'] or {}).get('code', '?')} -- an "
                  f"environment condition; re-run on a quiet host")
        else:
            res = d["report"]["results"]
            rows = res["per_mlp"]
            w = d.get("worker") or {}
            mem_gate = {"provoked_here": True, "worker": w,
                        "n_failed_mlps": res["n_failed_mlps"],
                        "failure_breakdown": res.get("failure_breakdown")}
            print(f"      worker saw RLIMIT_AS soft = {w.get('rlimit_as_soft'):,} "
                  f"({w.get('rlimit_as_soft_gib')} GiB)")
            # These six names are the CONSTANTS the memory constraint's witness
            # names. Section 7 verifies that witness against this suite's own
            # emitted results, so a renamed check here becomes a CHECK NOT FOUND
            # there rather than a witness quietly pointing at nothing.
            check(MEM_CAP_APPLIED,
                  w.get("rlimit_as_soft") == r.memory_limit_mb * 1024 * 1024,
                  f"{w.get('rlimit_as_soft'):,} == "
                  f"{r.memory_limit_mb * 1024 * 1024:,}")
            check(MEM_CONTROL,
                  w.get("control_mmap") == "OK", str(w.get("control_mmap")))
            check(MEM_REFUSED,
                  bool(w.get("provocation_refused")),
                  str(w.get("provocation_mmap")))
            check(MEM_UNTOUCHED,
                  "Cannot allocate memory" in str(w.get("provocation_mmap"))
                  or "MemoryError" in str(w.get("provocation_mmap")),
                  "address space refused at mmap time")
            check(MEM_ONE_FAILED,
                  res["n_failed_mlps"] == 1
                  and (res.get("failure_breakdown") or {}).get("error") == 1,
                  f"n_failed_mlps={res['n_failed_mlps']} "
                  f"breakdown={res.get('failure_breakdown')}")
            later = rows[1:]
            check(MEM_LATER_SCORED,
                  len(later) >= 1 and all(not m["traceback"] for m in later)
                  and all(m["final_layer_mse"] > 0 for m in later),
                  f"{len(later)} later MLPs, mse "
                  f"{[round(m['final_layer_mse'], 4) for m in later]}")

    death: dict = {"skipped": True}
    if args.skip_worker_death:
        head("6. worker death and restart -- SKIPPED by request")
        _SECTION_RAN["worker_death_restart"] = False
        unprovable("worker_death_restart", "--skip-worker-death was passed")
    else:
        head("6. worker death and restart, through the OFFICIAL subprocess runner")
        _SECTION_RAN["worker_death_restart"] = True
        d = provoke_worker_death(3)
        if "error" in d:
            check("the official runner produced a report", False, d["error"][:300])
            death = {"error": d["error"][:2000]}
        elif "refused" in d:
            code = (d["refused"] or {}).get("code", "?")
            death = {"refused": d["refused"]}
            check("the official runner accepted the run", False,
                  f"REFUSED at {(d['refused'] or {}).get('stage', '?')}: {code}")
            if code == "SETUP_TIMEOUT":
                print("      SETUP_TIMEOUT is an ENVIRONMENT condition, not an")
                print("      estimator defect: this host's `import whestbench`")
                print("      alone is 1.3 s quiet and 4.7-9.1 s loaded against a")
                print("      5 s ceiling, while our own setup is 0.6 ms. Re-run on")
                print("      a quiet machine. It is reported as a FAILURE and not")
                print("      a skip because a gate that cannot verify deployment")
                print("      must not certify it.")
        else:
            res = d["report"]["results"]
            rows = res["per_mlp"]
            failed = [i for i, x in enumerate(rows)
                      if x["budget_exhausted"] or x["time_exhausted"]
                      or x["residual_wall_time_exhausted"]
                      or x["combined_budget_exhausted"] or x["traceback"]]
            death = {
                "skipped": False,
                "n_mlps": len(rows),
                "n_failed_mlps": res["n_failed_mlps"],
                "failed_indices": failed,
                "failure_breakdown": res.get("failure_breakdown"),
            }
            # These six names are the CONSTANTS `worker_death_restart`'s witness
            # names, and section 7 resolves that witness against this suite's own
            # emitted results. Before P1-PRE this constraint's only matrix line
            # was `check(..., True, ...)` -- green whether or not any of the six
            # had run, and green with `--skip-worker-death`, when none of them
            # had.
            check(WD_SURVIVED,
                  len(rows) == 3, f"{len(rows)} rows returned")
            # EXACTLY one, at a KNOWN index, with a KNOWN classification.
            # `<= 1` also passes when the sacrifice never fired and nothing
            # failed at all -- which would mean the restart path was never
            # exercised while the check went green. That is the same shape of
            # defect as the literal-True check this suite already lost G4 to.
            check(WD_EXACTLY_ONE,
                  res["n_failed_mlps"] == 1,
                  f"n_failed_mlps = {res['n_failed_mlps']}, indices {failed}")
            check(WD_INDEX,
                  failed == [0], f"failed indices {failed}")
            check(WD_CLASSIFIED,
                  (res.get("failure_breakdown") or {}).get("error") == 1
                  and all((res.get("failure_breakdown") or {}).get(k) == 0
                          for k in ("budget_exhausted", "time_exhausted",
                                    "residual_wall_time_exhausted",
                                    "combined_budget_exhausted")),
                  f"{res.get('failure_breakdown')}")
            # "flops_used" is a weak witness here -- the sacrificial estimator
            # returns fnp.zeros, which costs nothing, so 0 is the CORRECT count and
            # proves nothing. What proves the row ran is that it carries a real
            # MSE and no failure flag of any kind.
            later = rows[1:]
            check(WD_LATER_CLEAN,
                  len(later) == 2 and all(
                      (not m["traceback"]) and not any(
                          m[k] for k in ("budget_exhausted", "time_exhausted",
                                         "residual_wall_time_exhausted",
                                         "combined_budget_exhausted"))
                      for m in later),
                  f"{len(later)} later MLPs, all clean")
            check(WD_LATER_SCORED,
                  all(m["final_layer_mse"] > 0.0 for m in later)
                  and len({round(m["final_layer_mse"], 9) for m in later}) == len(later),
                  f"mse {[round(m['final_layer_mse'], 4) for m in later]} -- "
                  f"distinct, so each was scored against its own network")

    # ---------------------------------------------------------------- #
    # 7. witness closure -- D1
    # ---------------------------------------------------------------- #
    # The constraints this suite provokes ITSELF, verified against this suite's
    # OWN emitted results, and only now that those results exist. The branch
    # this replaces was `check(name, True, ...)`: it ran BEFORE the provocations
    # and could not have consulted them even in principle.
    #
    # The risk of a suite verifying its own output is that it becomes
    # self-certifying. Two things stop that, and both are executable: the
    # witness names SPECIFIC checks that exist only if the section ran (section
    # 0 watches that refusal fire against an empty result set), and a section
    # that did not run is reported NOT PROVABLE, never green.
    head("7. witness closure -- the provocations THIS suite performed")
    adjudicated: set = set()
    for name, c in deferred:
        w = c.get("provoked_by")
        ok, why = verify_witness(w, resolver)
        _record_witness(c, w, ok, why, resolver)
        # Written so that the GREEN line's condition is `ok` and never a
        # literal. `check(name, True, ...)` inside an `if ok:` branch is
        # semantically identical and visually identical to the defect this
        # section replaces, and the next reader greps for the shape.
        if not ok and name not in _SECTION_RAN:
            # Its section never recorded whether it ran, so this refusal cannot
            # be classified -- and the SOFTER classification is the gap. A
            # failure downgraded to a gap because a bookkeeping line was
            # forgotten is the same defect family as the one being fixed, so the
            # missing record is itself a FAILURE.
            check(f"{name}: its section RECORDED whether it ran", False,
                  f"no entry in _SECTION_RAN, so this refusal cannot be "
                  f"classified as gap-or-failure -- {why}")
        elif not ok and not _SECTION_RAN[name]:
            # The section did not run: a gap, not a failure.
            unprovable(f"{name} (witness)", why)
        else:
            # It ran. Green if the witness holds, RED if it does not -- AND the
            # section must have recorded that it ran.
            #
            # The `_SECTION_RAN` conjunct is not redundant with `ok`. Without it
            # the green line's only protection is that the witness names check
            # names section 6 alone emits -- safety BY CONSTRUCTION, which is
            # what D1 was. Repoint a witness at a check that always runs and the
            # old defect's exact output line comes back. Injection G does that
            # and this conjunct is what catches it. A second independent Skeptic
            # found the hole; it is the same defect one level along.
            check(f"{name}: its named provocation RAN and PASSED in THIS suite",
                  ok and _SECTION_RAN.get(name) is True,
                  f"{why}" if _SECTION_RAN.get(name) is True else
                  f"the witness holds but its section did not run -- "
                  f"_SECTION_RAN[{name!r}]={_SECTION_RAN.get(name)!r}: {why}")
        adjudicated.add(name)
    # RECOMPUTED FROM THE MATRIX, not counted off the same list that drove the
    # loop. The previous spelling was `adjudicated == len(deferred)` with
    # `adjudicated` incremented once per iteration over `deferred` -- an
    # identity, and the only guard against the very failure it named: a
    # constraint dropped in the matrix loop shrinks BOTH sides and the line
    # still prints green on `0 of 0`. Instance nine of this repository's
    # signature failure, written inside the section that closes instance eight,
    # and found by an independent Skeptic.
    #
    # This set comes from `enf["constraints"]` and the loop's from `deferred`,
    # so they CAN disagree, and injection E makes them.
    expected_self = {n for n, c in enf["constraints"].items()
                     if c["enforced_locally"]
                     and isinstance(c.get("provoked_by"), dict)
                     and c["provoked_by"].get("suite") == SELF_SUITE}
    check("every SELF-PROVOKED constraint in the matrix reached this section",
          bool(expected_self) and adjudicated == expected_self,
          f"adjudicated {sorted(adjudicated)}; the matrix declares "
          f"{sorted(expected_self)}; missing "
          f"{sorted(expected_self - adjudicated) or 'none'}")
    enf["self_results_sha256"] = own_results()["results_sha256"]

    # ---------------------------------------------------------------- #
    # 8. the --no-write claim, VERIFIED before it is printed -- D3
    # ---------------------------------------------------------------- #
    if args.no_write:
        head("8. --no-write: the read-only claim, measured rather than announced")
        after = {p: _sha256_file(p) for p in READ_ONLY_WITNESS_PATHS}
        moved = [os.path.relpath(p, _ROOT) for p in READ_ONLY_WITNESS_PATHS
                 if ro_before.get(p) != after.get(p)]
        for p in READ_ONLY_WITNESS_PATHS:
            print(f"      {os.path.relpath(p, _ROOT)}\n"
                  f"        before {str(ro_before.get(p))[:16]}   "
                  f"after {str(after.get(p))[:16]}")
        check("--no-write: every tracked evidence artifact has the SAME sha256",
              not moved,
              "unchanged: " + ", ".join(
                  os.path.relpath(p, _ROOT) for p in READ_ONLY_WITNESS_PATHS)
              if not moved else f"MOVED: {moved}")
        git_after = _git_porcelain()
        if ro_git_before is None or git_after is None:
            declined("--no-write worktree cleanliness",
                     "`git status --porcelain` is unavailable here, so the "
                     "hash comparison above is the whole of the evidence")
        else:
            check("--no-write: `git status --porcelain` is byte-identical",
                  ro_git_before == git_after,
                  f"{len(ro_git_before.splitlines())} line(s) before, "
                  f"{len(git_after.splitlines())} after")

    print("\n" + "=" * 96)
    print(f"{_PASS}/{_PASS + _FAIL} checks passed; "
          f"{len(_UNPROVABLE)} constraint(s) NOT PROVABLE on this host; "
          f"{len(_DECLINED)} check(s) not performed in this run")
    if _UNPROVABLE:
        print("\nNOT PROVABLE HERE -- these are gaps in the local evaluator, and a")
        print("candidate that passes everything above has NOT been shown to satisfy")
        print("them. They are enforced only on the grader:")
        for u in _UNPROVABLE:
            print(f"  - {u['constraint']}: {u['why']}")
    if _DECLINED:
        print("\nNOT PERFORMED IN THIS RUN -- these are checks this INVOCATION")
        print("declined, not gaps in the evaluator's coverage of the grader. They")
        print("are named rather than skipped quietly, and they are not passes:")
        for d in _DECLINED:
            print(f"  - {d['check']}: {d['why']}")

    if args.no_write:
        # The claim is CONDITIONAL on the checks above. The old unconditional
        # line was printed by a run that had just rewritten a committed
        # artifact twice.
        if _FAIL == 0:
            print("\n  --no-write: VERIFIED -- no tracked evidence file moved and "
                  "the worktree is unchanged.")
        else:
            print("\n  --no-write: NOT VERIFIED. The claim is WITHHELD -- see the "
                  "failures above.")
        return 0 if _FAIL == 0 else 1

    before_out = _mtime(args.out)
    write_artifact(args.out, {
        "artifact": "deployment_envelope",
        "artifact_version": 1,
        "round": r.tag,
        "enforceability": enf,
        "unprovable_here": _UNPROVABLE,
        "not_performed_in_this_run": _DECLINED,
        "predict_response_timeout_s": eff,
        "memory": mem_report,
        "memory_limit_provocation": mem_gate,
        "worker_death": death,
        "checks_passed": _PASS,
        "checks_total": _PASS + _FAIL,
    })
    print(f"\n  wrote {os.path.relpath(args.out, _ROOT)}")
    # The other half of H6: this mode really does write, so "--no-write changed
    # nothing" is a fact about the latch and not about a writer that never fires.
    if _mtime(args.out) == before_out:
        print("  WARNING: the artifact's mtime did not move")
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
