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
1. **Structurally.** Every write-mode `open()` in the suite lives in a function
   that is either the guarded artifact writer or a writer of gitignored scratch.
   A new unguarded write to tracked evidence fails this check by construction,
   which reading the diff would not.
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

#: Functions in the suite that are ALLOWED to open a file for writing, and why.
#: `write_artifact` is the guarded evidence writer. The other two write throwaway
#: candidates under `local_sim/_envelope_*_tmp*`, which `.gitignore` covers and
#: which are removed in a `finally` -- that is what makes `git status
#: --porcelain` a fair witness for the read-only claim.
ALLOWED_WRITERS = {
    "write_artifact",
    "measure_live_shape_memory",
    "provoke_worker_death",
}

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
    """`{enclosing function name: [line, ...]}` for every write-mode `open()`."""
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    sites: dict = {}
    for fn in [n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
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
    green = [ln for ln in out.splitlines()
             if re.match(r"\s*PASS\b", ln) and "worker_death_restart" in ln]
    check("NO green line mentions worker_death_restart when section 6 is skipped",
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
