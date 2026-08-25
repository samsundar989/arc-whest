"""Is `--no-write` actually read-only? Proved from OUTSIDE the suite that claims it.

P1-PRE, defect **D3** of the Phase-0 accepting review
(`experiments/LOG.md`, 2026-08-24, P2-P0 GATE CLOSED).

WHY THIS FILE EXISTS, AND WHY IT IS NOT INSIDE THE SUITE
--------------------------------------------------------
`local_sim/test_deployment_envelope.py --no-write` printed

    --no-write: the artifact on disk was NOT touched.

while that same run had regenerated the committed
`experiments/2026-08-24_p2_p0/lifecycle_self_test.json` **twice**. An independent
Skeptic followed the documented read-only protocol and had to `git checkout` the
artifact afterwards. That was the second review to file the complaint.

The suite now verifies its own claim before printing it. That is necessary and it
is not sufficient: a file that grades its own read-only mode is exactly the kind
of self-certification this repository keeps losing checks to. So the proof lives
here, in a different process, by different logic -- **a child run, sha256, and
`git status --porcelain`** -- and it is the gate, not the suite, that runs both.

WHAT IT PROVES
--------------
1. **Structurally.** Every *direct* write call in the suite -- `open()` in a
   write mode, and the named `os` / `shutil` / `pathlib` write primitives --
   lives in a function that is either the guarded artifact writer or a writer of
   gitignored scratch.

   **This scan does not see every way a process can write**, and an earlier
   revision of this docstring claimed it did ("fails this check by
   construction"). It cannot see a write performed by a CHILD PROCESS, which is
   exactly how D3 happened: `regenerate_lifecycle_results` rewrites a committed
   artifact and contains no write call at all, only a `subprocess.run`. Those
   paths are covered by the latch (2) and the end-to-end run (3), not by this
   scan. The scan's value is narrow and real: a newly *added* direct write
   outside the allow-list turns it red, which reading a diff would not.
1b. **No check in any SCANNED file passes on a LITERAL TRUTHY CONSTANT.** Every
   `check(...)` is parsed -- callee matched as a name or an attribute, condition
   read from the second positional argument or an `ok=` keyword -- and a literal
   truthy condition fails this suite.

   **That is the whole of the claim, and it is narrower than it sounds.** The
   scan does not see a tautology built from a comparison (`a == a`), a folded
   expression (`1 == 1`, `not False`), a name bound to a constant, or an aliased
   callee. Every recent real instance in this repository is of the FIRST kind:
   `adjudicated == len(deferred)`, `enforced_locally == _rlimit_available()`,
   `len(_UNPROVABLE) > 0`. A previous revision of this file was recorded in the
   LOG as "the answer to every future instance of the family"; an independent
   Skeptic measured it at three spellings out of eleven, and that claim is
   retracted in `experiments/LOG.md`.

   So section 1b measures its own coverage: it asserts the five spellings it
   claims, asserts that an honest `check(..., False, ...)` is spared, and
   **reports** the six documented blind spots -- printing each, and printing a
   NOTICE if one has become caught.

   The blind spots are reported and NOT asserted, and that direction was itself a
   defect once. Asserting "these six are still blind" turned preflight `-Full`
   RED the moment anyone widened the scanner to catch `ast.Compare` tautologies
   -- the improvement this docstring, the LOG retraction and three reviews all
   say is needed -- and the only way past a red mandatory gate would have been to
   edit the gate. `docs/GOTCHAS.md` (CP0.1, 2026-08-20) records that trap by
   name. An improvement is a notice; only a regression in the five claimed
   spellings, or a probe that failed to run at all, is a failure.

   Compare-shaped tautologies remain a human-review obligation, and the question
   that finds them is not a grep: *could these two sides ever have disagreed,
   and are they computed from different places?*
2. **Mechanically.** With the latch down the writer writes; with it armed the
   same call raises and leaves no file. A guard nobody has watched refuse is a
   guard nobody has tested.
3. **End to end.** A real `--no-write --skip-worker-death` child leaves both
   tracked evidence artifacts byte-identical and `git status --porcelain`
   unmoved -- and, because the same run skips section 6, it also proves the D1
   repair: `worker_death_restart` is reported NOT PROVABLE and appears on **no**
   green line.

`--skip-worker-death` is deliberate. Section 6 drives the official subprocess
runner, which is load-sensitive and can be refused with `SETUP_TIMEOUT` on a busy
host; a read-only proof must not be able to fail for that reason. Skipping it is
also what makes this run cover the D1 skip path, so the choice costs nothing.

Run:  uv run python local_sim/test_envelope_readonly.py
"""

from __future__ import annotations

import ast
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# APPEND, never insert -- local_sim/ holds numpy shims named flopscope/whestbench.
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

import test_deployment_envelope as env  # noqa: E402

SUITE = os.path.join(_HERE, "test_deployment_envelope.py")

#: Functions in the suite that are ALLOWED to write directly, and why.
#: `write_artifact` is the guarded evidence writer. The other two write throwaway
#: candidates under `local_sim/_envelope_*_tmp*`, which `.gitignore` covers and
#: which are removed in a `finally` -- that is what makes `git status
#: --porcelain` a fair witness for the read-only claim.
ALLOWED_WRITERS = {
    "write_artifact",
    "measure_live_shape_memory",
    "provoke_worker_death",
}

#: Direct write primitives this scan recognises. Not exhaustive of "ways a
#: process can write" -- see the docstring -- but exhaustive of the ones a new
#: edit is likely to reach for.
_WRITE_CALLS = {
    "replace", "rename", "renames",             # os
    "copy", "copy2", "copyfile", "copytree", "move",   # shutil
    "write_text", "write_bytes",                # pathlib.Path
    "mkstemp", "NamedTemporaryFile",            # tempfile
}

#: The files this suite holds to the no-constant-condition rule.
#:
#: `lifecycle_eval.py` was added 2026-08-25 on a third Skeptic's point: FOUR of
#: the seven envelope witnesses resolve against its emitted results, so a check
#: that cannot go red there is a witness that cannot go red here. It scanned
#: clean when added. (It is also preflight step 11 in its own right.)
SCANNED = ("test_deployment_envelope.py", "test_envelope_readonly.py",
           "lifecycle_eval.py")

#: **Pinned literals, and the only reason they exist is that everything else in
#: this section is a traversal of the lists it audits.**
#:
#: A fourth independent Skeptic measured the previous guard across seven
#: mutations: `len(newly) + len(still) == len(KNOWN_BLIND)` is a partition
#: identity, so silently deleting entries from `KNOWN_BLIND` passed green while
#: the printed coverage rose from 5/11 to 5/6. That is `adjudicated ==
#: len(deferred)` a second time, in the file whose own fix for that class was to
#: pin a literal elsewhere.
#:
#: These are compared against the measured lengths, so the two sides come from
#: different places. Changing the scan's coverage means editing a number here on
#: purpose. Injection J deletes a `KNOWN_BLIND` entry and turns this red.
COVERAGE_CLAIM = (5, 6)          # (spellings CAUGHT, spellings KNOWN-BLIND)
REQUIRED_SCANNED = ("test_deployment_envelope.py", "test_envelope_readonly.py",
                    "lifecycle_eval.py")

_PASS = 0
_FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> bool:
    global _PASS, _FAIL
    ok = bool(ok)
    if ok:
        _PASS += 1
    else:
        _FAIL += 1
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<64} {detail}")
    return ok


def head(t: str) -> None:
    print(f"\n{t}\n{'-' * len(t)}")


def sha256(path: str) -> "str | None":
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return None


def porcelain() -> "str | None":
    p = subprocess.run(["git", "status", "--porcelain"], cwd=_ROOT,
                       capture_output=True, text=True)
    return p.stdout if p.returncode == 0 else None


# --------------------------------------------------------------------------- #
# 1. structural: where can this module write at all?
# --------------------------------------------------------------------------- #

def _write_sites(path: str) -> dict:
    """`{enclosing function name: [line, ...]}` for every DIRECT write call.

    Write-mode `open()` plus `_WRITE_CALLS`. Blind to child-process writes by
    construction -- see this module's docstring, and note that the write which
    caused D3 is one of them.
    """
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    sites: dict = {}
    for fn in [n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name in _WRITE_CALLS:
                sites.setdefault(fn.name, []).append(node.lineno)
                continue
            if name != "open":
                continue
            mode = ""
            if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                mode = str(node.args[1].value)
            for kw in node.keywords:
                if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                    mode = str(kw.value.value)
            if any(ch in mode for ch in ("w", "a", "x", "+")):
                sites.setdefault(fn.name, []).append(node.lineno)
    return sites


def _constant_conditions(path: str) -> list:
    """Every `check(...)` whose condition is a literal truthy constant: `(line, src)`.

    A literal FALSE condition is a deliberate failure report and is fine -- an
    error branch calling `check(name, False, detail)` is how the suite records
    that something went wrong. A literal TRUTHY condition is the defect: it
    prints green whatever happened, and this repository has now lost NINE checks
    to it, the ninth written inside the section repairing the eighth.

    **What this does NOT catch, stated because the first version of this scan was
    recorded as "the answer to every future instance of the family" and a second
    independent Skeptic measured it at three spellings out of eleven:**

      * a condition that is constant only after folding -- `1 == 1`,
        `True and True`, `not False`;
      * a NAME bound to a constant -- `ok = True; check(..., ok, ...)`;
        `check(..., bool(1), ...)`;
      * **and, the important omission: `ast.Compare` conditions whose two sides
        cannot differ.** Instance nine (`adjudicated == len(deferred)`) and the
        two survivors found beside it (`enforced_locally == _rlimit_available()`,
        `len(_UNPROVABLE) > 0`) are ALL of this shape. Every recent real instance
        in this repository is in the set this scan cannot see.

    So the honest scope is: **no `check()` in either scanned file passes on a
    literal truthy constant.** Compare-shaped tautologies remain a human-review
    obligation, and the question that finds them is not a grep -- it is
    *could these two sides ever have disagreed, and are they computed from
    different places?*

    Fixed here after that measurement: the callee is matched as a bare name
    (`check(...)`) **or** an attribute (`env.check(...)`, the spelling an
    importing file uses), and the condition is read from the second positional
    argument **or** from an `ok=` keyword.
    """
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if callee != "check":
            continue
        cond = node.args[1] if len(node.args) >= 2 else None
        for kw in node.keywords:
            if kw.arg == "ok":
                cond = kw.value
        if isinstance(cond, ast.Constant) and bool(cond.value):
            out.append((node.lineno, ast.unparse(cond)))
    return out


def structural_tests() -> None:
    head("1. STRUCTURAL -- every write-mode open() in the suite, and where it lives")
    sites = _write_sites(SUITE)
    for fn, lines in sorted(sites.items()):
        print(f"      {fn:<34} lines {lines}")
    unexpected = {k: v for k, v in sites.items() if k not in ALLOWED_WRITERS}
    check("every write-mode open() is in a function on the allow-list",
          not unexpected,
          f"{len(sites)} writer(s): {sorted(sites)}" if not unexpected
          else f"UNEXPECTED: {unexpected}")
    check("...and the guarded evidence writer is one of them",
          "write_artifact" in sites,
          "write_artifact is where tracked evidence is written")
    # Non-vacuity: the extractor must actually find things, or the check above
    # passes on an empty set. This repository has lost checks to exactly that.
    check("...and the extractor is not returning an empty set",
          len(sites) >= 3, f"found {len(sites)} functions that write")
    # The suite contains only `open()` writers, so the primitives added to
    # `_WRITE_CALLS` are unexercised by the scan above -- a widening nobody has
    # watched fire. A probe file exercises each one.
    probe = os.path.join(tempfile.gettempdir(), "arc_whest_write_probe.py")
    try:
        with open(probe, "w", encoding="utf-8") as fh:
            fh.write("import os, shutil, tempfile\n"
                     "from pathlib import Path\n"
                     "def w():\n"
                     "    open('a', 'w')\n"
                     "    os.replace('a', 'b')\n"
                     "    shutil.copyfile('a', 'b')\n"
                     "    Path('a').write_text('x')\n"
                     "    tempfile.mkstemp()\n"
                     "def r():\n"
                     "    open('a')\n"
                     "    open('a', 'r')\n"
                     "    shutil.rmtree('a')\n")
        found = _write_sites(probe)
        check("...and every widened write primitive is actually detected",
              sorted(found) == ["w"] and len(found.get("w", [])) == 5,
              f"{ {k: len(v) for k, v in found.items()} } -- 5 writes in w(), "
              f"and r() (reads plus a delete) is not a writer")
    finally:
        if os.path.exists(probe):
            os.remove(probe)

    head("1b. STRUCTURAL -- no check() in any SCANNED file can pass on a constant")
    offenders = {}
    for base in SCANNED:
        found = _constant_conditions(os.path.join(_HERE, base))
        if found:
            offenders[base] = found
        print(f"      {base:<34} {len(found)} literal-truthy condition(s)")
    # The check NAME used to be `f"...in any of {len(SCANNED)} files"` -- derived
    # from the tuple it audits, so dropping a file changed the label and passed.
    # `REQUIRED_SCANNED` is the literal; the two sides now come from different
    # places, and removing `lifecycle_eval.py` (added one commit ago on a
    # Skeptic's point) turns this red instead of relabelling it.
    check("every REQUIRED file is scanned",
          set(SCANNED) == set(REQUIRED_SCANNED),
          f"scanned {sorted(SCANNED)}; missing "
          f"{sorted(set(REQUIRED_SCANNED) - set(SCANNED)) or 'none'}; "
          f"unpinned {sorted(set(SCANNED) - set(REQUIRED_SCANNED)) or 'none'}")
    check("no check() has a literal truthy condition, in any scanned file",
          not offenders, "clean" if not offenders else f"{offenders}")
    # Non-vacuity, and it must MEASURE COVERAGE rather than confirm the one
    # spelling the scanner already handles. The first version of this probe fed
    # it `check('...', True, 'x')` and asserted the hit -- which proved the
    # scanner parses, not that it covers, and the LOG then described it as "the
    # answer to every future instance of the family". A second Skeptic measured
    # the real figure at 3 of 11. These are the eleven.
    MUST_CATCH = [                      # every spelling this scan is claimed to see
        ("positional True", "check('a', True, 'd')"),
        ("positional 1", "check('a', 1, 'd')"),
        ("positional truthy str", "check('a', 'yes', 'd')"),
        ("ok= keyword", "check('a', ok=True, detail='d')"),
        ("attribute callee", "env.check('a', True, 'd')"),
    ]
    MUST_NOT_CATCH = [                  # honest failure reports, never flagged
        ("positional False", "check('a', False, 'd')"),
        ("ok= False keyword", "check('a', ok=False, detail='d')"),
    ]
    KNOWN_BLIND = [                     # documented gaps -- see the docstring
        ("folded compare", "check('a', 1 == 1, 'd')"),
        ("folded boolop", "check('a', True and True, 'd')"),
        ("folded unary", "check('a', not False, 'd')"),
        ("name bound to True", "check('a', ok, 'd')"),
        ("truthy call", "check('a', bool(1), 'd')"),
        ("aliased callee", "c('a', True, 'd')"),
    ]
    probe = os.path.join(tempfile.gettempdir(), "arc_whest_const_probe.py")

    def _flags(snippet: str) -> bool:
        with open(probe, "w", encoding="utf-8") as fh:
            fh.write("def f():\n    " + snippet + "\n")
        return bool(_constant_conditions(probe))

    try:
        missed = [n for n, s in MUST_CATCH if not _flags(s)]
        check("the scanner catches every spelling it CLAIMS to catch",
              not missed, f"{len(MUST_CATCH)} spellings; missed {missed or 'none'}")
        wrong = [n for n, s in MUST_NOT_CATCH if _flags(s)]
        check("...and spares an honest check(..., False, ...) failure report",
              not wrong, f"wrongly flagged {wrong or 'none'}")
        # The gaps are MEASURED AND REPORTED -- never asserted.
        #
        # The previous spelling was `check(..., not now_caught, ...)`: it turned
        # preflight -Full RED the moment somebody widened the scanner to catch
        # `ast.Compare` tautologies, which is the improvement the retraction in
        # `experiments/LOG.md`, both prior reviews and this file's own docstring
        # all say is needed. The only way past a red gate would have been to edit
        # the gate, and `docs/GOTCHAS.md` (CP0.1, 2026-08-20) records that exact
        # trap by name. A third independent Skeptic caught it wired into the
        # mandatory tier.
        #
        # So an improvement is now a NOTICE, and only a REGRESSION is a failure.
        newly = [n for n, s in KNOWN_BLIND if _flags(s)]
        still = [n for n, s in KNOWN_BLIND if not _flags(s)]
        for n in still:
            print(f"      blind: {n}")
        if newly:
            print(f"      NOTICE: the scanner now catches {newly} -- widen the "
                  f"KNOWN_BLIND list, this docstring and the LOG scope claim to "
                  f"match. This is an improvement, not a failure.")
        # The coverage figure, against a PINNED literal.
        #
        # What stood here was `len(newly) + len(still) == len(KNOWN_BLIND)` --
        # a partition of `KNOWN_BLIND` compared against `KNOWN_BLIND`, so both
        # sides moved together and deleting documented blind spots passed green
        # with the printed coverage improving. Measured across seven mutations by
        # a fourth independent Skeptic; only emptying the list outright failed.
        # The comment claimed it would catch a probe that stopped writing and a
        # KNOWN_BLIND entry that stopped parsing: the first passes, the second
        # raises an uncaught SyntaxError before the check runs. Neither landed.
        measured = (len(MUST_CATCH), len(KNOWN_BLIND))
        check("the scan's coverage matches the PINNED claim",
              measured == COVERAGE_CLAIM,
              f"measured {measured[0]} caught / {measured[1]} blind "
              f"= {measured[0]}/{sum(measured)}; pinned "
              f"{COVERAGE_CLAIM[0]}/{sum(COVERAGE_CLAIM)}"
              + ("" if measured == COVERAGE_CLAIM else
                 " -- edit COVERAGE_CLAIM deliberately, or restore the lists"))
        check("...and every documented spelling was reached by the probe",
              len(newly) + len(still) == len(KNOWN_BLIND),
              f"{len(still)} still blind, {len(newly)} newly caught")
    finally:
        if os.path.exists(probe):
            os.remove(probe)


# --------------------------------------------------------------------------- #
# 2. mechanical: the latch refuses, and the writer is not merely broken
# --------------------------------------------------------------------------- #

def latch_tests() -> None:
    head("2. MECHANICAL -- the latch refuses both write paths, and the writer works")
    tmpdir = tempfile.mkdtemp(prefix="arc_whest_ro_ext_")
    target = os.path.join(tmpdir, "target.json")
    prev = env.arm_read_only(False)
    try:
        # POSITIVE CONTROL FIRST. If the writer were broken, every refusal below
        # would pass for the wrong reason.
        env.write_artifact(target, {"control": True})
        check("with the latch DOWN, the evidence writer writes",
              os.path.exists(target), os.path.basename(target))
        os.remove(target)

        env.arm_read_only(True)
        raised = ""
        try:
            env.write_artifact(target, {"must": "not reach disk"})
        except env.ReadOnlyRefusal as exc:
            raised = str(exc)
        check("with the latch ARMED, the same call RAISES and writes nothing",
              bool(raised) and not os.path.exists(target),
              raised[:80] or "NO REFUSAL RAISED")

        raised = ""
        try:
            env.regenerate_lifecycle_results(target)
        except env.ReadOnlyRefusal as exc:
            raised = str(exc)
        check("with the latch ARMED, suite REGENERATION raises and writes nothing",
              bool(raised) and not os.path.exists(target),
              raised[:80] or "NO REFUSAL RAISED")

        # The resolver consults the latch at resolve time, not only at
        # construction -- so a resolver built before `--no-write` was armed
        # cannot regenerate afterwards.
        r = env.WitnessResolver(
            registry={"probe": env.JsonSuite(target, env.regenerate_lifecycle_results)},
            allow_regenerate=True)
        status, _ = r.resolve("probe")
        check("a resolver built BEFORE the latch was armed still does not regenerate",
              status == "MISSING" and not os.path.exists(target)
              and r.regenerated.get("probe") is False,
              f"status={status} regenerated={r.regenerated}")
    finally:
        env.arm_read_only(prev)
        # rmtree, not rmdir: a build in which the latch has stopped refusing
        # leaves files here, and this cleanup must not itself raise while the
        # checks above are busy reporting exactly that.
        shutil.rmtree(tmpdir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# 3. end to end: a real --no-write child, judged by hashes and by git
# --------------------------------------------------------------------------- #

def _closing_blocks(out: str) -> "tuple[list, list]":
    """The suite's two closing lists: envelope gaps, and checks it did not run.

    They mean different things and the suite prints them under different
    headings; this splits them so a line filed under the wrong one is visible.
    """
    gaps, not_run, where = [], [], None
    for ln in out.splitlines():
        if ln.startswith("NOT PROVABLE HERE"):
            where = gaps
        elif ln.startswith("NOT PERFORMED IN THIS RUN"):
            where = not_run
        elif where is not None and ln.startswith("  - "):
            where.append(ln[4:])
        elif where is not None and ln.strip() and not ln.startswith(("  ", "them",
                                                                    "candidate",
                                                                    "declined",
                                                                    "are named")):
            where = None
    return gaps, not_run

def child_tests() -> int:
    head("3. END TO END -- a real `--no-write --skip-worker-death` child run")

    watched = list(env.READ_ONLY_WITNESS_PATHS)
    for p in watched:
        rel = os.path.relpath(p, _ROOT).replace(os.sep, "/")
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel],
                                 cwd=_ROOT, capture_output=True, text=True)
        check(f"watched path is TRACKED by git: {rel}", tracked.returncode == 0,
              "an 'unchanged' claim about an untracked file proves nothing")

    before = {p: sha256(p) for p in watched}
    git_before = porcelain()
    check("a git worktree status was obtained before the run",
          git_before is not None, "git status --porcelain")

    p = subprocess.run(
        [sys.executable, SUITE, "--no-write", "--skip-worker-death"],
        cwd=_ROOT, capture_output=True, text=True,
        env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    out = p.stdout + p.stderr

    after = {q: sha256(q) for q in watched}
    git_after = porcelain()

    check("the child exited 0", p.returncode == 0,
          f"exit {p.returncode}" + ("" if p.returncode == 0 else f"\n{out[-1500:]}"))

    for q in watched:
        rel = os.path.relpath(q, _ROOT).replace(os.sep, "/")
        check(f"sha256 UNCHANGED across the run: {rel}",
              before[q] is not None and before[q] == after[q],
              f"{str(before[q])[:16]} -> {str(after[q])[:16]}")

    check("`git status --porcelain` is byte-identical across the run",
          git_before is not None and git_before == git_after,
          f"{len((git_before or '').splitlines())} line(s) before, "
          f"{len((git_after or '').splitlines())} after")

    # --- and the D1 skip path, observed in the same child ------------------- #
    # What is forbidden is a green line CLAIMING this constraint was enforced or
    # provoked -- either one whose check NAME is about it (the field starts at
    # column 8), or one whose text asserts it in words. A green line that merely
    # LISTS the name in a detail is fine and there is now one: section 7's
    # closure prints the set it adjudicated, and `worker_death_restart` is in
    # that set precisely because it was adjudicated -- as NOT PROVABLE. The
    # first version of this check matched any mention and went red on it.
    green = [ln for ln in out.splitlines()
             if re.match(r"\s*PASS\s+worker_death_restart\b", ln)
             or (re.match(r"\s*PASS\b", ln) and "worker_death_restart" in ln
                 and ("RAN and PASSED" in ln or "is enforced" in ln))]
    check("NO green line CLAIMS worker_death_restart when section 6 is skipped",
          not green, "0 such lines" if not green else f"{len(green)}: {green[:2]}")
    check("...and it is reported NOT PROVABLE instead",
          re.search(r"^\s*-\s*worker_death_restart", out, re.M) is not None,
          "present in the NOT PROVABLE block")
    check("...through the WITNESS, not only through the skipped section",
          "worker_death_restart (witness)" in out,
          "section 7 adjudicated the deferred witness and refused it")
    check("...and the refusal names the check the witness could not find",
          "CHECK NOT FOUND" in out and env.WD_SURVIVED in out,
          f"names {env.WD_SURVIVED!r}")

    check("the child verified its own read-only claim before printing it",
          "--no-write: VERIFIED" in out,
          "the claim is now conditional on the checks above it")
    # A check this INVOCATION declined is not a gap in the evaluator's coverage
    # of the grader's envelope, and must not be filed under a heading that says
    # it is. The writer's positive control is the one such case in this mode.
    gaps, not_run = _closing_blocks(out)
    # NON-VACUITY FIRST. `not any(x in gaps)` is satisfied by an empty `gaps`,
    # so a parser that silently found nothing would make the check below pass
    # for the worst possible reason. This host always reports at least the three
    # envelope constraints it cannot enforce, so an empty block means the parser
    # broke, not that the report is clean.
    check("both closing blocks were parsed and neither is empty",
          len(gaps) >= 3 and len(not_run) >= 1,
          f"{len(gaps)} NOT PROVABLE, {len(not_run)} NOT PERFORMED")
    check("...the writer's positive control is filed as NOT PERFORMED",
          any("writer's positive control" in ln for ln in not_run),
          f"{len(not_run)} line(s) in that block")
    check("...and NOT as a gap in the evaluator's coverage of the grader",
          gaps and not any("writer's positive control" in ln for ln in gaps),
          f"{len(gaps)} envelope constraint(s) reported NOT PROVABLE")
    check("...and it regenerated nothing, by its own measurement",
          re.search(r"PASS\s+--no-write: resolving the lifecycle suite "
                    r"regenerated NOTHING", out) is not None,
          "the mirror of the writing run's non-vacuity check")
    return 0


def main() -> int:
    print("=" * 100)
    print("READ-ONLY PROOF -- is `test_deployment_envelope.py --no-write` actually "
          "read-only?")
    print("=" * 100)
    structural_tests()
    latch_tests()
    child_tests()
    print("\n" + "=" * 100)
    print(f"{_PASS}/{_PASS + _FAIL} checks passed")
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
