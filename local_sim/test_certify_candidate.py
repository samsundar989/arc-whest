"""Every certification section must be able to FAIL. Proven by mutation.

Contract: `experiments/LOG.md`, *CONTRACT WP0C+WP0D* (2026-08-31), committed at
`71bf62b` before this file existed. Prediction **C4**: every deliberate mutation
is caught **by its named section**, not merely by a non-zero exit.

WHY THE ORACLE IS A NAMED CHECK AND NOT AN EXIT CODE
----------------------------------------------------
`local_sim/p1pre_fault_inject.py` learned this the hard way and wrote it down:
`caught = returncode != 0` counts any red run as a catch, *including* a run that
died on a `SyntaxError` the injection itself introduced — which happened, and
was recorded as evidence. Two injections were later found broken, one failing
open and one failing closed, and only reading the FAIL lines told them apart.

So every mutation below declares the exact check name it expects to see fail. A
red run for any other reason is a **HARNESS FAILURE**, which is a finding about
this suite rather than about the thing it audits.

WHAT IS MUTATED
---------------
**Not `estimator.py`.** Each mutation is a synthetic candidate file written into
a system temp directory: a copy of the shipped file with one behaviour changed,
or a minimal hand-written estimator that is wrong in exactly one way. The
repository is opened read-only and its hashes are checked afterwards.

Sections 1 (`whest validate`) and 10 (subprocess smoke) are network- and
dataset-bound, so the mutations run with `--skip-smoke`; the certifier reports
those as SKIP, which is not a pass, and says so in its own summary.

Run: uv run python local_sim/test_certify_candidate.py
     uv run python local_sim/test_certify_candidate.py --only M3,M7
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]

import certify_candidate as cc  # noqa: E402  (after the sys.path fix)

SHIP = os.path.join(_ROOT, "estimator.py")

_FAILED = []
_TOTAL = [0]
_CHECKS = []

RECEIPT_VERSION = 1

UNIT_CHECK_NAMES = (
    "U1: _FAILURE_FLAGS is the installed whestbench's own flag_keys",
    "U2: the tail is taken over EXECUTED rows only",
    "U2: ...so the median is of executed work",
    "U2: ...and the CONTAMINATED median is recorded beside it",
    "U2: the worst is UNCHANGED by the zeros -- why the defect hid",
    "U2: headroom is quoted on the worst EXECUTED row",
    "U3: the dominant failure mode is named",
    "U3: every failed row keeps its error_code",
    "U3: the traceback's LAST line is kept, not its first",
    "U3: a harness fault UNDER the repo root claims no candidate frame",
    "U4: a fault inside the candidate names the candidate's frame",
    "U4b: ...and it is not claimed for a different candidate file",
    "U4c: an installed-package frame under the repo root is EXCLUDED",
    "U5: a cliff overrun is attributed to the CLIFF, not to 'error'",
    "U5: ...and its flags survive into the record",
    "U5: an over-cliff row does not masquerade as the tail",
    "U8: an over-cap failed row is retained in worst_all_rows",
    "U6: an all-failed run is REFUSED, not reported as 0 ms residual",
    "U7: retained failure records are capped",
    "U9: a LOCAL-only failure leaves MATH certified",
    "U9: ...and SERVER certified",
    "U9: ...while LOCAL is REFUSED, naming the check",
    "U9: the historical check NAME is unchanged, so old FAIL lines match",
    "U9: a domain with no checks reports NOT_RUN, not CERTIFIED",
    "U9: a skipped-only domain is INCOMPLETE, not CERTIFIED",
    "U9: real head/check emission separates all four evidence domains",
    "U9: measured server evidence is not a server forecast",
    "U9: installed distribution metadata recovers both package versions",
    "U10: every stored certification's verdict is reproduced exactly",
)


def check(name, ok, detail=""):
    _TOTAL[0] += 1
    _CHECKS.append({"name": str(name), "ok": bool(ok)})
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<64}{detail}")
    if not ok:
        _FAILED.append(name)


def _sha(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _write_json_exclusive(path, payload):
    """Write one complete receipt without replacing prior evidence."""
    out = os.path.abspath(path)
    parent = os.path.dirname(out)
    if not os.path.isdir(parent):
        raise ValueError(f"receipt parent directory does not exist: {parent}")
    with open(out, "x", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _distribution_bytes(name):
    """Hash every file declared by an installed distribution's RECORD."""
    dist = importlib.metadata.distribution(name)
    declared = dist.files
    if not declared:
        raise RuntimeError(f"installed distribution {name!r} has no file manifest")
    files = []
    for item in sorted(declared, key=lambda p: str(p).replace("\\", "/")):
        rel = str(item).replace("\\", "/")
        path = os.fspath(dist.locate_file(item))
        if not os.path.isfile(path):
            raise RuntimeError(f"installed distribution file is absent: {name}:{rel}")
        files.append({"path": rel, "size": os.path.getsize(path),
                      "sha256": _sha(path)})
    raw = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return {
        "version": dist.version,
        "file_count": len(files),
        "total_bytes": sum(row["size"] for row in files),
        "manifest_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "files": files,
    }


def receipt_provenance():
    """Exact identities needed to combine independently executed shards."""
    return {
        "test_certify_candidate_sha256": _sha(__file__),
        "certify_candidate_sha256": _sha(os.path.join(_HERE, "certify_candidate.py")),
        "estimator_sha256": _sha(SHIP),
        "distributions": {name: importlib.metadata.version(name)
                          for name in ("whestbench", "flopscope")},
        "distribution_bytes": {name: _distribution_bytes(name)
                               for name in ("whestbench", "flopscope")},
    }


def validate_workerq_job_id(value, environ_value=""):
    """Validate the explicitly declared workerq identity for a receipt.

    This binds the declaration to WORKERQ_JOB_ID when workerq supplies it.  The
    receipt remains a record, not an authentication mechanism; raw workerq
    metadata and logs are the execution authority.
    """
    if not isinstance(value, str) or not value.isdecimal() or int(value) <= 0:
        raise ValueError("--workerq-job-id must be a positive decimal job id")
    if environ_value and value != environ_value:
        raise ValueError("--workerq-job-id does not match WORKERQ_JOB_ID")
    return value


# --------------------------------------------------------------------------- #
# the mutations
# --------------------------------------------------------------------------- #
# Each entry is (tag, title, builder, expected FAIL substring, section).
#
# `builder(tmpdir) -> path` writes the mutant. Builders that start from the
# shipped file do a textual substitution and assert it matched exactly once, so
# an injection that silently stops applying is caught before it is graded.

def _ship_text():
    with open(SHIP, encoding="utf-8") as fh:
        return fh.read()


def _write(tmp, name, text):
    p = os.path.join(tmp, name)
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return p


def _mutate_ship(tmp, name, find, repl):
    t = _ship_text()
    if t.count(find) != 1:
        raise AssertionError(
            f"{name}: the injection target appears {t.count(find)} times, "
            "expected exactly 1 -- the mutation no longer tests what it claims")
    return _write(tmp, name, t.replace(find, repl, 1))


#: Each mutant WRAPS the shipped file and changes exactly one behaviour.
#:
#: The first version of this suite used a hand-written minimal estimator as the
#: base. It failed `edge_cases.py`'s 30 invariants for reasons unrelated to any
#: injection, so every "the battery goes RED" was ambiguous and the positive
#: control could not pass. Wrapping the shipped file makes the injected defect
#: the ONLY difference from a known-good subject -- which is what a mutation
#: test is for.
WRAP = """
import importlib.util as _iu

_spec = _iu.spec_from_file_location("_base_est", r"%s")
_base = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_base)

import flopscope.numpy as fnp    # noqa: F401  (used by injected defects)


class Estimator:
    \"\"\"The shipped estimator, wrapped. %s\"\"\"

    def __init__(self):
        self._inner = _base.Estimator()

    def setup(self, context):
        %s
        return self._inner.setup(context)

    def predict(self, mlp, flop_budget):
        %s
        out = self._inner.predict(mlp, flop_budget)
        %s
        return out
"""


def _indent(lines, pad=8):
    """A multi-line injected body, indented into the wrapper's method scope."""
    return ("\n" + " " * pad).join(lines)


def _wrap(tmp, name, note, setup_body="pass", pre_body="pass", post_body="pass"):
    return _write(tmp, name, WRAP % (SHIP, note, setup_body, pre_body, post_body))


MUTATIONS = []


def register(tag, title, builder, expect, section, extra=()):
    MUTATIONS.append((tag, title, builder, expect, section, tuple(extra)))


def build_ok(tmp):
    """The positive control: the shipped file, wrapped, with no defect."""
    return _wrap(tmp, "cand_ok.py", "no defect -- the positive control")


# ---- M1: non-deterministic output (section 4) -----------------------------
register(
    "M1", "output depends on wall-clock, not on mlp.seed",
    lambda tmp: _wrap(tmp, "cand_m1.py", "NON-DETERMINISTIC",
                      post_body="out = out * (1.0 + 1e-9 * "
                                "(__import__('time').time() % 2.0))"),
    "FRESH interpreters agree bitwise", "4")

# ---- M2: wrong shape ------------------------------------------------------
register(
    "M2", "returns (depth-1, width)",
    lambda tmp: _wrap(tmp, "cand_m2.py", "WRONG SHAPE", post_body="out = out[:-1]"),
    "shape", "3/4/5")

# ---- M3: non-finite -------------------------------------------------------
register(
    "M3", "returns NaN in the first row",
    lambda tmp: _wrap(tmp, "cand_m3.py", "NON-FINITE",
                      post_body="out = out / fnp.array("
                                "[0.0] + [1.0] * (out.shape[0] - 1))[:, None]"),
    "finite", "5")

# ---- M4: structurally negative --------------------------------------------
register(
    "M4", "returns a large negative value, not float noise",
    lambda tmp: _wrap(tmp, "cand_m4.py", "NEGATIVE", post_body="out = out - 0.05"),
    "negative beyond float rounding", "5")

# ---- M5: mutates the grader's arrays ---------------------------------------
register(
    "M5", "writes into mlp.weights before predicting",
    lambda tmp: _wrap(
        tmp, "cand_m5.py", "MUTATES INPUTS",
        pre_body=_indent([
            "import numpy as _np",
            "_a = _np.asarray(mlp.weights[0])",
            "try:",
            "    _a.setflags(write=True)",
            "except Exception:",
            "    pass",
            "_a[0, 0] = _np.float32(0.0)",
        ])),
    "UNCHANGED after predict", "5")

# ---- M6: a scalar materialization inside predict ---------------------------
register(
    "M6", "calls float() on a flopscope value inside predict",
    lambda tmp: _wrap(tmp, "cand_m6.py", "SCALAR SYNC",
                      post_body=_indent(["if float(fnp.sum(out)) > -1.0:",
                                         "    out = out * 1.0"])),
    "ZERO scalar materializations", "8")

# ---- M7: setup() raises ----------------------------------------------------
register(
    "M7", "setup() raises",
    lambda tmp: _wrap(tmp, "cand_m7.py", "SETUP RAISES",
                      setup_body="raise RuntimeError('deliberate setup failure')"),
    "setup() raised nothing", "5")

# ---- M8: predict raises ----------------------------------------------------
register(
    "M8", "predict() raises",
    lambda tmp: _wrap(tmp, "cand_m8.py", "PREDICT RAISES",
                      pre_body="raise RuntimeError('deliberate predict failure')"),
    "no run raised", "5")

# ---- M9: exceeds the budget it is given ------------------------------------
# Driven by a TINY --budget rather than by a huge injected matmul: the shipped
# path spends ~1.376e11 FLOPs whatever the budget, so a 1e6 budget exhausts
# immediately and cheaply. The first version added 64 matmuls (~4.1e11) against
# a 2.2e12 budget and did not exhaust at all -- an injection that never reached
# its target, which is exactly the HARNESS FAILURE class this suite grades for.
register(
    "M9", "spends more than the budget it was handed",
    lambda tmp: _wrap(tmp, "cand_m9.py", "OVER BUDGET (via a tiny --budget)"),
    "no run raised", "5", extra=("--budget", "1000000"))

# ---- M10: a wrong layer 1 that is finite, positive and plausible ------------
register(
    "M10", "layer 1 scaled by 1/sqrt(2) -- finite, positive, and wrong",
    lambda tmp: _wrap(tmp, "cand_m10.py", "WRONG LAYER 1",
                      post_body="out = out * fnp.array("
                                "[0.7071067811865476] + [1.0] * (out.shape[0] - 1)"
                                ")[:, None]"),
    "invariant suite passes on THIS candidate", "3")


# ---- M11/M12: THE GRADER-TWIN ARM (section 5b) -----------------------------
# On 2026-09-05 seven submission slots went to a candidate that opened with
#
#     w = [x if x.dtype == dtype else x.astype(dtype, copy=False)
#          for x in mlp.weights]
#
# Every local weight producer in this repository built float32, so the `if`
# short-circuited, the `else` branch NEVER RAN offline, and the grader -- whose
# weights are float64 -- took a path no gate had ever executed. Section 5b now
# runs the candidate under all four weight dtypes.
#
# What these two mutants DO and DO NOT prove, stated plainly because the
# distinction is the whole value of the arm:
#
#   They prove the arm EXECUTES the dtype-guarded branch and reports both ways
#   it can go wrong -- raising, and silently answering differently.
#
#   They do NOT reproduce submission 329887's own crash. `copy=False` behaves
#   identically to `copy=True` in-process on a flopscope float64 array (measured
#   2026-09-05: both return a writeable float32 FlopscopeArray, and both bill
#   2 FLOP/element), and canary4's own source comment records the same negative
#   result under `--runner server`. That crash remains unreproduced offline; the
#   arm's claim is narrower and true -- a candidate whose BEHAVIOUR depends on
#   the weight dtype can no longer reach a submission slot unexercised.
#
# Numbered M11/M12 because M9 (over budget) and M10 (wrong layer 1) were already
# taken; the handoff that requested this called it "M9".
register(
    "M11", "a dtype-guarded branch that RAISES on the grader's float64 weights",
    lambda tmp: _wrap(
        tmp, "cand_m11.py", "RAISES ONLY ON THE GRADER'S DTYPE",
        pre_body=_indent([
            "import numpy as _np",
            "if any(_np.asarray(x).dtype != _np.float32 for x in mlp.weights):",
            "    raise RuntimeError('deliberate: float64 weights take this path')",
        ])),
    "the f64 arm raised nothing", "5b")

register(
    "M12", "a dtype-guarded branch that silently ANSWERS DIFFERENTLY on float64",
    lambda tmp: _wrap(
        tmp, "cand_m12.py", "WRONG ONLY ON THE GRADER'S DTYPE",
        pre_body=_indent([
            "import numpy as _np",
            "self._twin_scale = 1.0",
            "if any(_np.asarray(x).dtype != _np.float32 for x in mlp.weights):",
            "    self._twin_scale = 1.01",
        ]),
        post_body="out = out * self._twin_scale"),
    "agrees with the f32 control", "5b")


def _certify(cand, tmp_json, timeout=7200, extra=()):
    argv = [sys.executable, os.path.join(_HERE, "certify_candidate.py"),
            "--candidate", cand, "--skip-smoke", "--quick",
            "--json", tmp_json] + list(extra)
    return subprocess.run(argv, capture_output=True, text=True, cwd=_ROOT,
                          timeout=timeout, errors="replace")


def _failed_lines(stdout):
    return [l.strip() for l in (stdout or "").splitlines()
            if l.strip().startswith("FAIL ")]


def mutation_registry():
    """Machine-readable registry in the same order the monolith executes."""
    return [
        {"tag": tag, "title": title, "expected_fail": expect,
         "section": section, "extra_args": list(extra)}
        for tag, title, _builder, expect, section, extra in MUTATIONS
    ]


def shard_selection(only_text="", *, skip_positive=False,
                    positive_only=False):
    """Validate a shard request before a candidate or temporary file exists."""
    requested = [t.strip() for t in only_text.split(",") if t.strip()]
    if len(requested) != len(set(requested)):
        raise ValueError("--only contains a duplicate mutation tag")
    known = [m["tag"] for m in mutation_registry()]
    unknown = sorted(set(requested) - set(known))
    if unknown:
        raise ValueError("unknown mutation tag(s): " + ", ".join(unknown))
    if skip_positive and positive_only:
        raise ValueError("--skip-positive and --positive-only are mutually exclusive")
    if skip_positive and not requested:
        raise ValueError("--skip-positive requires a non-empty --only selection")
    if positive_only and requested:
        raise ValueError("--positive-only cannot be combined with --only")
    selected = known if not requested else [tag for tag in known if tag in requested]
    if positive_only:
        selected = []
    return {
        "include_positive": not skip_positive,
        "mutation_tags": selected,
    }



# ---------------------------------------------------------------------------
# U  the residual soak's arithmetic, tested without a three-hour run
# ---------------------------------------------------------------------------
# Added 2026-09-01 under CONTRACT WP8. The soak shipped with two defects that
# survived review because the only way to exercise them was an official run
# against the organisers' data: it discarded the REASON a row failed, and it
# pooled rows that never executed (`scoring.py`'s exception path stamps a
# failed row with `residual_wall_time_s = 0.0`). `_pool_residual` is now pure,
# so both are checkable in milliseconds.

# A REAL `traceback.format_exc()` shape: absolute paths, and the installed
# package living UNDER the repository root because `.venv/` is inside it. The
# first version of this fixture used the relative path "whestbench/runner.py",
# which `format_exc()` never emits -- and that is the only reason the broken
# substring heuristic passed U3. The source line is included too, because
# `scoring.py`'s own line reads `raw_predictions = estimator.predict(...)` and
# contains the word `estimator`.
_TB_HOST = (
    "Traceback (most recent call last):\n"
    '  File "C:/Users/samsu/Documents/arc-whest/.venv/lib/site-packages/'
    'whestbench/scoring.py", line 716, in score_estimator\n'
    "    raw_predictions = estimator.predict(mlp, budget)\n"
    '  File "C:/Users/samsu/Documents/arc-whest/.venv/lib/site-packages/'
    'whestbench/runner.py", line 451, in _call\n'
    "RunnerError: worker died"
)
_TB_OURS = (
    "Traceback (most recent call last):\n"
    '  File "C:/Users/samsu/Documents/arc-whest/estimator.py", line 900, in predict\n'
    "ValueError: shapes do not align"
)


def _row(i, resid, failed_how=None):
    """One per-MLP entry shaped like the official runner's."""
    m = {"mlp_index": i, "mlp_name": "mlp_%d" % i,
         "residual_wall_time_s": resid, "flops_used": 137600326656,
         "budget_exhausted": False, "time_exhausted": False,
         "residual_wall_time_exhausted": False,
         "combined_budget_exhausted": False}
    if failed_how == "error":
        # Exactly the shape scoring.py:790-850 appends: residual ZEROED.
        m.update({"error_code": "RunnerError", "error": "worker died",
                  "residual_wall_time_s": 0.0, "flops_used": 0,
                  "traceback": _TB_HOST})
    elif failed_how:
        m[failed_how] = True
    return m


def _report(rows, breakdown, cap=0.4):
    n_failed = sum(1 for m in rows if m.get("error_code")
                   or any(m.get(k) for k in cc._FAILURE_FLAGS))
    return {"run_config": {"residual_wall_time_limit_s": cap},
            "results": {"per_mlp": rows, "n_failed_mlps": n_failed,
                        "failure_breakdown": breakdown}}


def unit_tests():
    print("U  the soak's arithmetic (pure, no official run)")

    # -- U1: our failure definition is the runner's, still ------------------
    # Copied constants rot. This asserts the copy against the installed
    # package, so a whestbench change breaks a test instead of silently making
    # the soak's failure count disagree with the runner's own.
    # THE FIRST VERSION OF THIS TEST WAS INVERTED, and an independent review
    # caught it. It built the "package" tuple by FILTERING a hardcoded
    # four-name literal in this file, so `pkg` was a subset of those four by
    # construction: a whestbench release ADDING a fifth flag -- the only drift
    # that would make our failure count disagree with the runner's, which is
    # the stated purpose -- left `pkg` equal to the four and the test PASSED.
    # It also scored its own skip as a pass, which `docs/VALIDATION.md` forbids
    # by name. Now it PARSES the package's actual `flag_keys` tuple, and an
    # unreadable package is a FAIL, not a green skip.
    import ast as _ast
    import inspect
    try:
        from whestbench import scoring as _sc
        src = inspect.getsource(_sc)
        i = src.find("flag_keys = (")
        j = src.find(")", i)
        pkg = tuple(_ast.literal_eval(src[i + len("flag_keys = "):j + 1]))
        readable, why = True, ""
    except Exception as exc:
        pkg, readable = (), False
        why = "%s: %s" % (type(exc).__name__, str(exc)[:40])
    check("U1: _FAILURE_FLAGS is the installed whestbench's own flag_keys",
          readable and pkg == cc._FAILURE_FLAGS,
          ("package %s vs ours %s" % (pkg, cc._FAILURE_FLAGS)) if readable
          else "COULD NOT READ THE PACKAGE -- not a pass: " + why)

    # -- U2: the exclusion arithmetic, on the shape that actually bit -------
    # 15 rows, 5 of them failed-with-zero -- the committed 45/300 in miniature.
    # Executed residuals 10..100 ms; the zeros must not enter the sample.
    rows = ([_row(i, 0.010 * (i + 1)) for i in range(10)]
            + [_row(10 + i, 0.0, "error") for i in range(5)])
    rep = _report(rows, {"error": 5, "budget_exhausted": 0,
                         "time_exhausted": 0,
                         "residual_wall_time_exhausted": 0,
                         "combined_budget_exhausted": 0})
    out = cc._pool_residual([rep], reps=1, n_mlps=15)

    check("U2: the tail is taken over EXECUTED rows only",
          out["n_rows"] == 15 and out["n_rows_executed"] == 10
          and out["residual_rows_excluded_failed"] == 5,
          "%d/%d executed" % (out["n_rows_executed"], out["n_rows"]))
    # median of 10..100 ms at nearest-rank p50 is the 5th value, 50 ms.
    check("U2: ...so the median is of executed work",
          abs(out["median"] - 0.050) < 1e-12,
          "%.1f ms" % (out["median"] * 1000))
    # Pooling the zeros would put p50 of 15 rows at rank 8 -> 30 ms. That is
    # the defect, reproduced, and it is a 1.67x understatement.
    check("U2: ...and the CONTAMINATED median is recorded beside it",
          abs(out["median_all_rows"] - 0.030) < 1e-12,
          "%.1f ms, bias %.3fx" % (out["median_all_rows"] * 1000,
                                   out["median_bias_from_failed_rows"]))
    check("U2: the worst is UNCHANGED by the zeros -- why the defect hid",
          abs(out["worst"] - 0.100) < 1e-12, "%.1f ms" % (out["worst"] * 1000))
    check("U2: headroom is quoted on the worst EXECUTED row",
          abs(out["headroom_worst"] - 4.0) < 1e-9,
          "%.2fx" % out["headroom_worst"])

    # -- U3: the failure is ATTRIBUTED, not just counted --------------------
    check("U3: the dominant failure mode is named",
          out["dominant_failure_mode"] == "error",
          repr(out["dominant_failure_mode"]))
    check("U3: every failed row keeps its error_code",
          len(out["failures"]) == 5
          and all(f["error_code"] == "RunnerError" for f in out["failures"]),
          "%d records" % len(out["failures"]))
    if not out["failures"]:
        # Without this, a mutation that stops recording failures kills the
        # suite with IndexError three lines down -- and this file's own
        # docstring calls a red run for another reason a HARNESS FAILURE, not
        # a catch. An independent review found exactly that (mutation MU3).
        out["failures"] = [{"error_code": None, "traceback_last_line": None,
                            "deepest_candidate_frame": None, "flags": []}]
    check("U3: the traceback's LAST line is kept, not its first",
          len(out["failures"]) > 0
          and out["failures"][0]["traceback_last_line"] == "RunnerError: worker died",
          repr(out["failures"][0]["traceback_last_line"] if out["failures"]
               else "NO RECORDS")[:50])
    # The claim that would have been FALSE before the repair: this traceback
    # has no participant code in it at all, yet every frame's absolute path
    # contains "arc-whest" and one source line contains "estimator".
    check("U3: a harness fault UNDER the repo root claims no candidate frame",
          out["failures"][0]["deepest_candidate_frame"] is None,
          repr(out["failures"][0]["deepest_candidate_frame"]))

    # -- U4: a fault in the CANDIDATE is distinguishable from a host fault --
    # The whole point of the attribution: P2 of the WP8 contract turns on
    # whether the failing frame is ours. It must not read as host contention.
    bad = _row(0, 0.0, "error")
    # Keep the fixture tied to the checkout under test.  workerq and other
    # frozen runners relocate the repository, so a hard-coded developer path
    # would turn this positive attribution control into a false negative.
    checkout = _ROOT.replace("\\", "/")
    bad["traceback"] = _TB_OURS.replace(
        "C:/Users/samsu/Documents/arc-whest", checkout)
    cand = os.path.join(_ROOT, "estimator.py")
    rec = cc._failure_record(1, bad, cand)
    check("U4: a fault inside the candidate names the candidate's frame",
          rec["deepest_candidate_frame"] is not None
          and "estimator.py" in rec["deepest_candidate_frame"],
          (rec["deepest_candidate_frame"] or "none")[-44:])
    # U4b: the SAME traceback, asked about a DIFFERENT candidate, must not be
    # claimed. Without this, a match on any repository file would still pass.
    other = cc._failure_record(1, bad, os.path.join(_ROOT, "estimator_v1.py"))
    check("U4b: ...and it is not claimed for a different candidate file",
          other["deepest_candidate_frame"] is None,
          repr(other["deepest_candidate_frame"]))
    # U4c: the real host traceback, run through the real helper, against a
    # real candidate path. This is the case the broken heuristic failed.
    host_tb = _TB_HOST.replace(
        "C:/Users/samsu/Documents/arc-whest", checkout)
    host = cc._candidate_frame(host_tb.splitlines(), cand)
    check("U4c: an installed-package frame under the repo root is EXCLUDED",
          host is None, repr(host))

    # -- U5: an exhaustion failure is NOT reported as an error --------------
    # If the cliff is ever the mechanism, the soak has to say so -- that is the
    # branch of the WP8 falsifier that would invalidate every residual claim
    # in this repository.
    rows5 = ([_row(i, 0.05) for i in range(8)]
             + [_row(8, 0.41, "residual_wall_time_exhausted"),
                _row(9, 0.42, "residual_wall_time_exhausted")])
    out5 = cc._pool_residual(
        [_report(rows5, {"error": 0, "budget_exhausted": 0,
                         "time_exhausted": 0,
                         "residual_wall_time_exhausted": 2,
                         "combined_budget_exhausted": 0})],
        reps=1, n_mlps=10)
    check("U5: a cliff overrun is attributed to the CLIFF, not to 'error'",
          out5["dominant_failure_mode"] == "residual_wall_time_exhausted"
          and out5["n_failed_mlps"] == 2,
          repr(out5["dominant_failure_mode"]))
    check("U5: ...and its flags survive into the record",
          len(out5["failures"]) > 0
          and out5["failures"][0]["flags"] == ["residual_wall_time_exhausted"],
          str([f["flags"] for f in out5["failures"][:2]]) or "NO RECORDS")
    # An over-cliff row is excluded from the tail like any other failure, so
    # the reported worst is 50 ms, not 420. That is deliberate, and it is why
    # check 11 tests `n_failed_mlps == 0` SEPARATELY from the headroom: a soak
    # that quietly reported 8x headroom while two rows blew the cap would be
    # the worst possible false green.
    check("U5: an over-cliff row does not masquerade as the tail",
          abs(out5["worst"] - 0.05) < 1e-12 and out5["n_failed_mlps"] == 2,
          "worst %.0f ms, %d failed" % (out5["worst"] * 1000,
                                        out5["n_failed_mlps"]))

    # -- U8: the CONTAMINATED worst is kept, and it is the conservative one --
    # An independent review found that the repair kept `median_all_rows` and
    # `p99_all_rows` but dropped the contaminated WORST -- exactly backwards.
    # Only the exception path zeroes the residual; the four exhaustion paths
    # leave a real, possibly OVER-CAP number. So excluding failed rows makes
    # `worst` LESS conservative, and out5 is the proof: two rows blew a 0.4 s
    # cap and the executed-only worst is 50 ms.
    check("U8: an over-cap failed row is retained in worst_all_rows",
          abs(out5["worst_all_rows"] - 0.42) < 1e-12
          and out5["headroom_worst_all_rows"] < 1.0,
          "worst_all_rows %.0f ms, headroom %.3fx (executed-only says %.0f ms,"
          " %.1fx)" % (out5["worst_all_rows"] * 1000,
                       out5["headroom_worst_all_rows"],
                       out5["worst"] * 1000, out5["headroom_worst"]))

    # -- U6: all rows failed -> refuse, do not invent a tail from zeros -----
    out6 = cc._pool_residual(
        [_report([_row(i, 0.0, "error") for i in range(4)], {"error": 4})],
        reps=1, n_mlps=4)
    check("U6: an all-failed run is REFUSED, not reported as 0 ms residual",
          out6["ok"] is False and "no executed residual" in out6["why"],
          out6.get("why", "")[:44])

    # -- U7: the failure-record list is bounded ----------------------------
    many = [_row(i, 0.0, "error") for i in range(200)]
    out7 = cc._pool_residual([_report(many, {"error": 200})], reps=1,
                             n_mlps=200)
    check("U7: retained failure records are capped",
          len(out7["failures"]) == cc._MAX_FAILURE_RECORDS,
          "%d of 200" % len(out7["failures"]))

    # -- U9: the three evidence domains, and the verdict that does NOT move -
    #
    # Stage 0 R5, validation audit finding 4. One word over three different
    # questions is what this repairs. The rollup is pure, so it is tested here
    # rather than behind a candidate run.
    tagged = [
        {"section": "1", "domain": cc.MATH, "name": "imports", "ok": True,
         "skipped": False},
        {"section": "5", "domain": cc.MATH, "name": "grader twin", "ok": True,
         "skipped": False},
        {"section": "6", "domain": cc.SERVER, "name": "op-count gate",
         "ok": True, "skipped": False},
        # The two standing historical failures are BOTH of this shape.
        {"section": "10", "domain": cc.LOCAL,
         "name": "10: worst GRADED-PATH residual is under the cliff",
         "ok": False, "skipped": False},
    ]
    roll = cc.domain_rollup(tagged)
    check("U9: a LOCAL-only failure leaves MATH certified",
          roll[cc.MATH]["verdict"] == "CERTIFIED", roll[cc.MATH]["verdict"])
    check("U9: ...and SERVER certified",
          roll[cc.SERVER]["verdict"] == "CERTIFIED", roll[cc.SERVER]["verdict"])
    check("U9: ...while LOCAL is REFUSED, naming the check",
          roll[cc.LOCAL]["verdict"] == "REFUSED"
          and "GRADED-PATH" in roll[cc.LOCAL]["failed"][0],
          roll[cc.LOCAL]["failed"][0][:46])
    check("U9: the historical check NAME is unchanged, so old FAIL lines match",
          "10: worst GRADED-PATH residual is under the cliff"
          in roll[cc.LOCAL]["failed"],
          "renaming it would orphan two standing gate failures")

    # A domain nobody ran is NOT_RUN, which VALIDATION.md forbids scoring as a
    # pass.
    roll2 = cc.domain_rollup([t for t in tagged if t["domain"] != cc.SERVER])
    check("U9: a domain with no checks reports NOT_RUN, not CERTIFIED",
          roll2[cc.SERVER]["verdict"] == "NOT_RUN",
          roll2[cc.SERVER]["verdict"])
    roll3 = cc.domain_rollup(
        [{"section": "6", "domain": cc.SERVER, "name": "x", "ok": None,
          "skipped": True, "why": "not run"}])
    check("U9: a skipped-only domain is INCOMPLETE, not CERTIFIED",
          roll3[cc.SERVER]["verdict"] == "INCOMPLETE",
          roll3[cc.SERVER]["verdict"])

    # Drive the production emitter itself. The real section id is `5-8`, so a
    # section lookup cannot distinguish these checks; each mixed check must
    # carry its domain at creation. Section 11 similarly has no old map entry.
    start = len(cc._TAGGED)
    old_section, old_domain = cc._CURRENT[0], cc._CURRENT_DOMAIN[0]
    old_total = cc._TOTAL[0]
    old_sections = {k: list(v) for k, v in cc._SECTIONS.items()}
    try:
        cc.head("5-8  integration fixture", domain=cc.MATH)
        cc.check("5: mathematical contract", True)
        cc.check("7: local runtime observation", True, domain=cc.LOCAL)
        cc.check("7: predicted server residual", True,
                 domain=cc.SERVER_PREDICTED)
        cc.check("S: measured grading-server row", True,
                 domain=cc.SERVER_MEASURED)
        cc.head("11  integration fixture", domain=cc.LOCAL)
        cc.check("11: residual tail", True)
        emitted = cc._TAGGED[start:]
    finally:
        del cc._TAGGED[start:]
        cc._CURRENT[0], cc._CURRENT_DOMAIN[0] = old_section, old_domain
        cc._TOTAL[0] = old_total
        cc._SECTIONS.clear()
        cc._SECTIONS.update(old_sections)
    got_domains = [row["domain"] for row in emitted]
    check("U9: real head/check emission separates all four evidence domains",
          got_domains == [cc.MATH, cc.LOCAL, cc.SERVER_PREDICTED,
                          cc.SERVER_MEASURED, cc.LOCAL],
          repr(got_domains))
    check("U9: measured server evidence is not a server forecast",
          cc.SERVER_MEASURED != cc.SERVER_PREDICTED,
          f"{cc.SERVER_MEASURED} vs {cc.SERVER_PREDICTED}")

    stack = cc._certified_stack()
    check("U9: installed distribution metadata recovers both package versions",
          stack == {"whestbench": importlib.metadata.version("whestbench"),
                    "flopscope": importlib.metadata.version("flopscope")}
          and all(stack.values()), repr(stack))

    # -- U10: prediction 9 -- no STORED certification changes disposition ---
    #
    # The single verdict is still computed from `_FAILED`/`_SKIPPED` alone.
    # Re-derive each stored report's verdict from the failures and skips IT
    # recorded, and require the stored word back.
    import glob as _glob
    import json as _json
    seen, agreed, disagreed = 0, 0, []
    for path in _glob.glob(os.path.join(cc._ROOT, "experiments", "*",
                                        "certify_*.json")):
        try:
            with open(path, encoding="utf-8") as fh:
                rep = _json.load(fh)
        except (OSError, ValueError):
            continue
        if "verdict" not in rep:
            continue
        seen += 1
        f, k = rep.get("checks_failed") or [], rep.get("checks_skipped") or []
        want = ("CERTIFIED" if not f and not k else
                "INCOMPLETE" if not f else "REFUSED")
        if want == rep["verdict"]:
            agreed += 1
        else:
            disagreed.append((os.path.basename(path), rep["verdict"], want))
    check("U10: every stored certification's verdict is reproduced exactly",
          seen > 0 and not disagreed,
          "%d reports, %d agree%s" % (seen, agreed,
                                      "" if not disagreed else
                                      "; DISAGREE %s" % (disagreed[:2],)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", default="", help="comma-separated tags, e.g. M3,M7")
    ap.add_argument("--skip-positive", action="store_true",
                    help="omit B from a non-empty --only mutation shard")
    ap.add_argument("--positive-only", action="store_true",
                    help="run B and no mutations")
    ap.add_argument("--receipt", default="", metavar="PATH",
                    help="exclusive-create machine-readable shard receipt")
    ap.add_argument("--workerq-job-id", default="", metavar="ID",
                    help="required explicit workerq job id when writing a receipt")
    a = ap.parse_args()
    try:
        selection = shard_selection(
            a.only, skip_positive=a.skip_positive,
            positive_only=a.positive_only)
    except ValueError as exc:
        ap.error(str(exc))
    if a.receipt:
        try:
            validate_workerq_job_id(
                a.workerq_job_id, os.environ.get("WORKERQ_JOB_ID", ""))
        except ValueError as exc:
            ap.error(str(exc))
        receipt_path = os.path.abspath(a.receipt)
        if os.path.exists(receipt_path):
            ap.error("receipt already exists; evidence is never overwritten")
        if not os.path.isdir(os.path.dirname(receipt_path)):
            ap.error("receipt parent directory must already exist")
    elif a.workerq_job_id:
        ap.error("--workerq-job-id is only valid with --receipt")

    provenance = receipt_provenance() if a.receipt else None
    registry = mutation_registry()
    registry_sha = hashlib.sha256(json.dumps(
        registry, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    selected_tags = set(selection["mutation_tags"])
    cases = []

    print("WP0C - every certification section must be able to FAIL\n")
    unit_tests()
    unit_check_count = len(_CHECKS)
    if tuple(row.get("name") for row in _CHECKS) != UNIT_CHECK_NAMES:
        check("H: unit-check prefix matches the registered names", False,
              f"expected {len(UNIT_CHECK_NAMES)}, observed {unit_check_count}")
    print()
    ship_before = _sha(SHIP)
    tmp = tempfile.mkdtemp(prefix="wp0c_mutants_")
    try:
        # ---- the positive control ----------------------------------------
        if selection["include_positive"]:
            print("B  a CORRECT minimal candidate is not rejected by the battery")
            ok_path = build_ok(tmp)
            r = _certify(ok_path, os.path.join(tmp, "ok.json"))
            fails = _failed_lines(r.stdout)
            # rc 3 = INCOMPLETE (sections 1 and 10 skipped by --quick), which is the
            # certifier refusing to call a skip a pass. rc 1 would be a real refusal.
            not_refused = r.returncode in (0, 3)
            no_fail_lines = not fails
            skips_named = r.returncode == 3 and "NOT CHECKED" in (r.stdout or "")
            check("B: the correct minimal candidate is not REFUSED", not_refused,
                  f"rc={r.returncode}, {len(fails)} FAIL line(s)"
                  + ("" if not fails else " -- " + fails[0][:60]))
            check("B: ...and it produced no FAIL line at all", no_fail_lines,
                  "; ".join(fails[:2])[:88] or "clean")
            check("B: the certifier REFUSES to call its skips a pass",
                  skips_named, f"rc={r.returncode}")
            cases.append({
                "tag": "B", "kind": "positive",
                "candidate_sha256": _sha(ok_path),
                "returncode": r.returncode, "fail_lines": fails,
                "stdout": r.stdout or "", "stderr": r.stderr or "",
                "stdout_sha256": hashlib.sha256(
                    (r.stdout or "").encode("utf-8")).hexdigest(),
                "stderr_sha256": hashlib.sha256(
                    (r.stderr or "").encode("utf-8")).hexdigest(),
                "not_refused": not_refused, "no_fail_lines": no_fail_lines,
                "skips_named": skips_named,
            })

        # ---- the mutations ------------------------------------------------
        print("\nM  each mutation must fail its NAMED check")
        for tag, title, builder, expect, section, extra in MUTATIONS:
            if tag not in selected_tags:
                continue
            try:
                path = builder(tmp)
            except AssertionError as exc:
                check(f"M[{tag}]: the injection applies", False, str(exc)[:70])
                cases.append({"tag": tag, "kind": "mutation",
                              "title": title, "section": section,
                              "expected_fail": expect,
                              "extra_args": list(extra),
                              "injection_applied": False,
                              "error": str(exc)})
                continue
            r = _certify(path, os.path.join(tmp, f"{tag}.json"), extra=extra)
            fails = _failed_lines(r.stdout)
            named = any(expect in f for f in fails)
            red = r.returncode == 1
            check(f"M[{tag}] (§{section}): {title} -- the battery goes RED",
                  red,
                  f"rc={r.returncode}, {len(fails)} FAIL line(s)")
            check(f"M[{tag}]: ...and the NAMED check '{expect[:26]}' is one of them",
                  named,
                  "caught" if named else
                  "HARNESS FAILURE -- red for another reason: "
                  + ("; ".join(f[6:] for f in fails[:2])[:80] or
                     "(no FAIL line at all)"))
            cases.append({
                "tag": tag, "kind": "mutation", "title": title,
                "section": section, "expected_fail": expect,
                "extra_args": list(extra), "injection_applied": True,
                "candidate_sha256": _sha(path),
                "returncode": r.returncode, "battery_red": red,
                "named_failure_matched": named, "fail_lines": fails,
                "stdout": r.stdout or "", "stderr": r.stderr or "",
                "stdout_sha256": hashlib.sha256(
                    (r.stdout or "").encode("utf-8")).hexdigest(),
                "stderr_sha256": hashlib.sha256(
                    (r.stderr or "").encode("utf-8")).hexdigest(),
            })
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\nR  the repository was not modified")
    check("R: estimator.py sha256 unchanged", _sha(SHIP) == ship_before,
          ship_before[:16])
    check("R: the mutant directory is gone", not os.path.exists(tmp), tmp)
    if a.receipt:
        provenance_after = receipt_provenance()
        check("R: receipt provenance unchanged across execution",
              provenance_after == provenance,
              "test, certifier, estimator, and installed package bytes")

    if a.receipt:
        payload = {
            "artifact": "certification_mutation_shard",
            "artifact_version": RECEIPT_VERSION,
            "provenance": provenance,
            "mutation_registry": registry,
            "mutation_registry_sha256": registry_sha,
            "selection": selection,
            "cases": cases,
            "unit_check_count": unit_check_count,
            "unit_checks_passed": all(c["ok"] for c in _CHECKS[:unit_check_count]),
            "checks": _CHECKS,
            "overall_pass": not _FAILED,
            "workerq_job_id": a.workerq_job_id,
        }
        _write_json_exclusive(a.receipt, payload)
        print("  receipt -> " + os.path.abspath(a.receipt))

    print()
    if _FAILED:
        print(f"FAILED {len(_FAILED)}: " + "; ".join(_FAILED))
        raise SystemExit(1)
    print(f"{_TOTAL[0]}/{_TOTAL[0]} checks passed")


if __name__ == "__main__":
    main()
