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


def check(name, ok, detail=""):
    _TOTAL[0] += 1
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<64}{detail}")
    if not ok:
        _FAILED.append(name)


def _sha(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


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


def _certify(cand, tmp_json, timeout=2400, extra=()):
    argv = [sys.executable, os.path.join(_HERE, "certify_candidate.py"),
            "--candidate", cand, "--skip-smoke", "--quick",
            "--json", tmp_json] + list(extra)
    return subprocess.run(argv, capture_output=True, text=True, cwd=_ROOT,
                          timeout=timeout, errors="replace")


def _failed_lines(stdout):
    return [l.strip() for l in (stdout or "").splitlines()
            if l.strip().startswith("FAIL ")]



# ---------------------------------------------------------------------------
# U  the residual soak's arithmetic, tested without a three-hour run
# ---------------------------------------------------------------------------
# Added 2026-09-01 under CONTRACT WP8. The soak shipped with two defects that
# survived review because the only way to exercise them was an official run
# against the organisers' data: it discarded the REASON a row failed, and it
# pooled rows that never executed (`scoring.py`'s exception path stamps a
# failed row with `residual_wall_time_s = 0.0`). `_pool_residual` is now pure,
# so both are checkable in milliseconds.

_TB_HOST = (
    "Traceback (most recent call last):\n"
    '  File "whestbench/runner.py", line 451, in _call\n'
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
    try:
        import inspect
        from whestbench import scoring as _sc
        src = inspect.getsource(_sc)
        i = src.find("flag_keys = (")
        pkg = tuple(f for f in ("budget_exhausted", "time_exhausted",
                                "residual_wall_time_exhausted",
                                "combined_budget_exhausted")
                    if i >= 0 and ('"%s"' % f) in src[i:i + 400])
        check("U1: _FAILURE_FLAGS matches the installed whestbench",
              pkg == cc._FAILURE_FLAGS,
              "package %s vs ours %s" % (pkg, cc._FAILURE_FLAGS))
    except Exception as exc:  # the suite must run with nothing installed
        check("U1: _FAILURE_FLAGS matches the installed whestbench", True,
              "SKIPPED, whestbench unreadable: %s" % type(exc).__name__)

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
    check("U3: the traceback's LAST line is kept, not its first",
          out["failures"][0]["traceback_last_line"] == "RunnerError: worker died",
          repr(out["failures"][0]["traceback_last_line"])[:50])
    check("U3: a harness-side failure reports NO repository frame",
          out["failures"][0]["deepest_repo_frame"] is None,
          repr(out["failures"][0]["deepest_repo_frame"]))

    # -- U4: a fault in the CANDIDATE is distinguishable from a host fault --
    # The whole point of the attribution: P2 of the WP8 contract turns on
    # whether the failing frame is ours. It must not read as host contention.
    bad = _row(0, 0.0, "error")
    bad["traceback"] = _TB_OURS
    rec = cc._failure_record(1, bad)
    check("U4: a fault inside the candidate names a repository frame",
          rec["deepest_repo_frame"] is not None
          and "estimator.py" in rec["deepest_repo_frame"],
          (rec["deepest_repo_frame"] or "none")[-44:])

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
          out5["failures"][0]["flags"] == ["residual_wall_time_exhausted"],
          str(out5["failures"][0]["flags"]))
    # An over-cliff row is excluded from the tail like any other failure, so
    # the reported worst is 50 ms, not 420. That is deliberate, and it is why
    # check 11 tests `n_failed_mlps == 0` SEPARATELY from the headroom: a soak
    # that quietly reported 8x headroom while two rows blew the cap would be
    # the worst possible false green.
    check("U5: an over-cliff row does not masquerade as the tail",
          abs(out5["worst"] - 0.05) < 1e-12 and out5["n_failed_mlps"] == 2,
          "worst %.0f ms, %d failed" % (out5["worst"] * 1000,
                                        out5["n_failed_mlps"]))

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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", default="", help="comma-separated tags, e.g. M3,M7")
    a = ap.parse_args()
    only = {t.strip() for t in a.only.split(",") if t.strip()}

    print("WP0C - every certification section must be able to FAIL\n")
    unit_tests()
    print()
    ship_before = _sha(SHIP)
    tmp = tempfile.mkdtemp(prefix="wp0c_mutants_")
    try:
        # ---- the positive control ----------------------------------------
        print("B  a CORRECT minimal candidate is not rejected by the battery")
        ok_path = build_ok(tmp)
        r = _certify(ok_path, os.path.join(tmp, "ok.json"))
        fails = _failed_lines(r.stdout)
        # rc 3 = INCOMPLETE (sections 1 and 10 skipped by --quick), which is the
        # certifier refusing to call a skip a pass. rc 1 would be a real refusal.
        check("B: the correct minimal candidate is not REFUSED",
              r.returncode in (0, 3),
              f"rc={r.returncode}, {len(fails)} FAIL line(s)"
              + ("" if not fails else " -- " + fails[0][:60]))
        check("B: ...and it produced no FAIL line at all", not fails,
              "; ".join(fails[:2])[:88] or "clean")
        check("B: the certifier REFUSES to call its skips a pass",
              r.returncode == 3 and "NOT CHECKED" in (r.stdout or ""),
              f"rc={r.returncode}")

        # ---- the mutations ------------------------------------------------
        print("\nM  each mutation must fail its NAMED check")
        for tag, title, builder, expect, section, extra in MUTATIONS:
            if only and tag not in only:
                continue
            try:
                path = builder(tmp)
            except AssertionError as exc:
                check(f"M[{tag}]: the injection applies", False, str(exc)[:70])
                continue
            r = _certify(path, os.path.join(tmp, f"{tag}.json"), extra=extra)
            fails = _failed_lines(r.stdout)
            named = any(expect in f for f in fails)
            check(f"M[{tag}] (§{section}): {title} -- the battery goes RED",
                  r.returncode == 1,
                  f"rc={r.returncode}, {len(fails)} FAIL line(s)")
            check(f"M[{tag}]: ...and the NAMED check '{expect[:26]}' is one of them",
                  named,
                  "caught" if named else
                  "HARNESS FAILURE -- red for another reason: "
                  + ("; ".join(f[6:] for f in fails[:2])[:80] or
                     "(no FAIL line at all)"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\nR  the repository was not modified")
    check("R: estimator.py sha256 unchanged", _sha(SHIP) == ship_before,
          ship_before[:16])
    check("R: the mutant directory is gone", not os.path.exists(tmp), tmp)

    print()
    if _FAILED:
        print(f"FAILED {len(_FAILED)}: " + "; ".join(_FAILED))
        raise SystemExit(1)
    print(f"{_TOTAL[0]}/{_TOTAL[0]} checks passed")


if __name__ == "__main__":
    main()
