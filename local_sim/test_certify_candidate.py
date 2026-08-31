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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", default="", help="comma-separated tags, e.g. M3,M7")
    a = ap.parse_args()
    only = {t.strip() for t in a.only.split(",") if t.strip()}

    print("WP0C - every certification section must be able to FAIL\n")
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
