"""Break each P1-PRE repair on purpose, and prove the checks that guard it go RED.

P1-PRE (`experiments/LOG.md`, 2026-08-25). Companion to
`local_sim/test_deployment_envelope.py` and `local_sim/test_envelope_readonly.py`.

WHY THIS IS IN THE REPOSITORY
-----------------------------
A check nobody has watched fail is a check nobody has tested, and that sentence is
the whole subject of P1-PRE: the slice exists because a `check(name, True, ...)`
sat green for a week inside the enforceability matrix while the constraint it
named was provoked nowhere.

So the slice's central evidence is not that its suites pass. It is that they FAIL
when the repairs are removed. That evidence was first produced by a driver living
in a scratch directory outside the tree -- which an independent Skeptic correctly
filed as a continuity defect: `AGENTS.md` requires that no session depend on
another session's context, and a transcript nobody can regenerate is a claim, not
a reproduction. The driver is therefore committed here.

WHAT IT DOES
------------
For each injection: apply an exact source substitution to
`test_deployment_envelope.py`, run the suite the repair is guarded by, record
whether it went red, then restore the file from a byte-for-byte backup and
`git checkout` any tracked artifact the broken build touched. It asserts the
restoration by sha256 before exiting, and it refuses to start on a dirty copy of
the file it is about to edit.

**It is not part of the gate.** `preflight.ps1` runs the two suites; this runs
their falsification, deliberately writes a broken build to disk for a few seconds
at a time, and is invoked by hand when the guarantees are re-argued:

    uv run python local_sim/p1pre_fault_inject.py
    uv run python local_sim/p1pre_fault_inject.py --only A,E

Exit code 0 means every injection was CAUGHT. A non-zero exit means an injection
went unnoticed, which is the interesting outcome: it names a guarantee that has
stopped being enforced.

INJECTIONS
----------
A  D1  -- section 7 reports an UNRUN provocation as green again
B  D2  -- the resolver ignores the suite it was given (pre-P1-PRE behaviour)
C  D3  -- the read-only latch stops refusing
D      -- a section forgets to record whether it ran
E      -- a self-provoked constraint never reaches section 7

E exists because the check that was supposed to catch it could not: it read
`adjudicated == len(deferred)` with `adjudicated` incremented once per iteration
over `deferred`, so both sides moved together and `0 of 0` printed green. That was
instance NINE of this repository's signature failure, written inside the section
that closes instance eight, and found by an independent Skeptic rather than by
this harness -- which is exactly why the harness now carries E.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

SUITE = os.path.join(_HERE, "test_deployment_envelope.py")
READONLY = os.path.join(_HERE, "test_envelope_readonly.py")
OUT = os.path.join(_ROOT, "experiments", "2026-08-25_p1_pre", "fault_injection.txt")
BACKUP = SUITE + ".fault_inject_backup"

#: Tracked artifacts a broken build can rewrite. Restored after every injection.
TRACKED = (
    "experiments/2026-08-24_p2_p0/lifecycle_self_test.json",
    "experiments/2026-08-24_p2_p0/deployment_envelope.json",
)

#: (tag, what it breaks, [(exact source to find, what to replace it with)], argv)
INJECTIONS = [
    (
        "A",
        "D1 -- section 7 reports an UNRUN provocation as green again",
        [("""        elif not ok and not _SECTION_RAN[name]:
            # The section did not run: a gap, not a failure.
            unprovable(f"{name} (witness)", why)""",
          """        elif not ok and not _SECTION_RAN[name]:
            check(f"{name}: its named provocation RAN and PASSED in THIS suite",
                  True, why)""")],
        [READONLY],
    ),
    (
        "B",
        "D2 -- the resolver ignores the suite it was given (pre-P1-PRE behaviour)",
        [("""        src = self._registry.get(suite) if isinstance(suite, str) else None
        if src is None:
            return "UNKNOWN_SUITE", None""",
          """        src = (self._registry.get(suite) if isinstance(suite, str) else None)
        if src is None:
            src = self._registry.get(LIFECYCLE_SUITE)
        if src is None:
            return "UNKNOWN_SUITE", None
        suite = LIFECYCLE_SUITE""")],
        [SUITE, "--no-write", "--skip-worker-death"],
    ),
    (
        "C",
        "D3 -- the read-only latch stops refusing",
        [("""def refuse_if_read_only(what: str) -> None:
    if _READ_ONLY:""",
          """def refuse_if_read_only(what: str) -> None:
    if False:"""),
         ("""        if self._allow_regenerate and not read_only_armed() and src.regenerate:""",
          """        if self._allow_regenerate and src.regenerate:""")],
        [READONLY],
    ),
    (
        "D",
        "a section forgets to record whether it ran",
        [("""        _SECTION_RAN["worker_death_restart"] = False
        unprovable("worker_death_restart", "--skip-worker-death was passed")""",
          """        unprovable("worker_death_restart", "--skip-worker-death was passed")""")],
        [SUITE, "--no-write", "--skip-worker-death"],
    ),
    (
        "E",
        "a self-provoked constraint never reaches section 7",
        [("""        if isinstance(w, dict) and w.get("suite") == SELF_SUITE:
            deferred.append((name, c))
            pending(name, "provoked by this suite, which has not run it yet")
            continue""",
          """        if isinstance(w, dict) and w.get("suite") == SELF_SUITE:
            continue""")],
        [SUITE, "--no-write", "--skip-worker-death"],
    ),
    (
        "F",
        "the literal-True defect returns in a spelling the scan used to miss",
        # `ok=` keyword form. Before the second Skeptic pass the scanner read
        # `args[1]` only, so this exact spelling went GREEN through a check whose
        # whole purpose is to forbid it.
        #
        # The replacement must be VALID PYTHON. A first version spliced
        # `ok=True, detail=(` in front of the existing positional detail
        # strings, which is a syntax error -- the child then died on import and
        # the harness recorded CAUGHT on a non-zero exit that had nothing to do
        # with the scan. An injection has to reach its target to mean anything,
        # and "it went red" is not evidence that it went red for the reason
        # claimed.
        [("""    check("the matrix carries every REQUIRED envelope constraint",
          present >= set(REQUIRED_CONSTRAINTS),
          f"{len(present)} present; missing "
          f"{sorted(set(REQUIRED_CONSTRAINTS) - present) or 'none'}"
          + (f"; unpinned extras {sorted(present - set(REQUIRED_CONSTRAINTS))}"
             if present - set(REQUIRED_CONSTRAINTS) else ""))""",
          """    check("the matrix carries every REQUIRED envelope constraint", ok=True,
          detail="injection F: a literal truthy condition, ok= keyword form")""")],
        [READONLY],
    ),
    (
        "G",
        "a witness is repointed at a check that ALWAYS runs (D1's shape returning)",
        # The green branch's safety used to be entirely that the witness named
        # `WD_*` strings only section 6 emits -- safety by construction, which is
        # what D1 was. Repointing the witness reproduces D1's exact output line
        # unless the green condition also consults _SECTION_RAN.
        # BOTH `checks` and `expect_detail` must be repointed. The first version
        # of this injection replaced only `checks`, and `expect_detail` still
        # named WD_EXACTLY_ONE -- so `verify_witness` refused on the detail
        # assertion, `ok` came back False, and the run took the `unprovable`
        # branch without ever reaching the green branch under test. It reported
        # UNCAUGHT, correctly: the injection had not landed. An injection that
        # does not reach its target is a broken test, and it looks exactly like
        # a broken guard until you check which.
        [("""                    "checks": [WD_SURVIVED, WD_EXACTLY_ONE, WD_INDEX,
                               WD_CLASSIFIED, WD_LATER_CLEAN, WD_LATER_SCORED],
                    "expect_detail": {
                        WD_EXACTLY_ONE: "n_failed_mlps = 1",
                        WD_INDEX: "failed indices [0]",
                    },""",
          """                    "checks": ["the runner waits max(predict_timeout, wall "
                               "+ grace), not predict_timeout"],""")],
        [SUITE, "--no-write", "--skip-worker-death"],
    ),
    (
        "H",
        "an envelope constraint is DELETED from the matrix entirely",
        # Every accounting line in the suite is a traversal of
        # `enf["constraints"]`, so deleting an entry shrinks the checker along
        # with the checked. Only the pinned REQUIRED_CONSTRAINTS list notices.
        [("""            "predict_response_timeout_s": {""",
          """            "deleted_by_injection_h": {""")],
        [SUITE, "--no-write", "--skip-worker-death"],
    ),
    (
        "I",
        "an envelope constraint is ADDED to the matrix without being pinned",
        # H covers deletion. Addition was silent until 2026-08-25: the check read
        # `present >= set(REQUIRED_CONSTRAINTS)`, a superset test, while the
        # tuple's comment claimed "adding a constraint is a deliberate edit to
        # this tuple". A third independent Skeptic measured both directions and
        # found this one green. The comparison is `==` now, and this is the test
        # that says so.
        [("""            "predict_response_timeout_s": {""",
          """            "added_by_injection_i": {
                "value": None,
                "enforced_locally": False,
                "by": "injection I: an unpinned constraint",
                "provoked_by": None,
            },
            "predict_response_timeout_s": {""")],
        [SUITE, "--no-write", "--skip-worker-death"],
    ),
]


def _sha(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=_ROOT, capture_output=True, text=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", default="",
                    help="comma-separated injection tags, e.g. A,E")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    # A dirty suite would be restored to the WRONG bytes, silently reverting
    # someone's uncommitted work. Refuse instead.
    rel = os.path.relpath(SUITE, _ROOT).replace(os.sep, "/")
    dirty = _git("status", "--porcelain", "--", rel).stdout.strip()
    if dirty:
        print(f"REFUSING: {rel} has uncommitted changes ({dirty!r}).")
        print("This harness restores it from a backup of its CURRENT bytes, but a "
              "crash mid-run would leave a broken build in your tree. Commit or "
              "stash first.")
        return 2

    wanted = {t.strip().upper() for t in args.only.split(",") if t.strip()}
    plan = [i for i in INJECTIONS if not wanted or i[0] in wanted]
    if wanted and len(plan) != len(wanted):
        print(f"unknown tag(s): {sorted(wanted - {i[0] for i in plan})}")
        return 2

    original = _sha(SUITE)
    shutil.copyfile(SUITE, BACKUP)
    lines: list = []
    uncaught: list = []
    try:
        for tag, title, subs, argv in plan:
            with open(SUITE, encoding="utf-8") as fh:
                src = fh.read()
            for old, new in subs:
                if old not in src:
                    print(f"injection {tag}: ANCHOR NOT FOUND -- the code it "
                          f"breaks has moved. Update this file.\n{old[:160]}")
                    return 3
                src = src.replace(old, new, 1)
            with open(SUITE, "w", encoding="utf-8") as fh:
                fh.write(src)

            p = subprocess.run([sys.executable, *argv], cwd=_ROOT,
                               capture_output=True, text=True,
                               env=dict(os.environ, PYTHONIOENCODING="utf-8"))
            out = p.stdout + p.stderr
            fails = [ln.strip() for ln in out.splitlines()
                     if ln.strip().startswith("FAIL")]
            caught = p.returncode != 0
            if not caught:
                uncaught.append(tag)

            lines.append("=" * 100)
            lines.append(f"INJECTION {tag} -- {title}")
            lines.append(f"command: {' '.join(os.path.basename(a) for a in argv)}")
            verdict = ("CAUGHT" if caught else
                       "UNCAUGHT -- the defect was reintroduced and NOTHING "
                       "noticed")
            lines.append(f"exit code: {p.returncode}   ({verdict})")
            lines.append(f"FAIL lines: {len(fails)}")
            lines.extend("  " + ln for ln in fails)
            lines.append("")
            lines.append("--- last 12 lines of the run ---")
            lines.extend("  " + ln for ln in out.splitlines()[-12:])
            lines.append("")

            shutil.copyfile(BACKUP, SUITE)
            _git("checkout", "--", *TRACKED)
    finally:
        shutil.copyfile(BACKUP, SUITE)
        os.remove(BACKUP)

    if _sha(SUITE) != original:
        print("FATAL: the suite was not restored byte-for-byte")
        return 4

    lines.append("=" * 100)
    lines.append(f"suite restored byte-for-byte: sha256 {original}")
    lines.append(f"injections run: {[i[0] for i in plan]}   "
                 f"uncaught: {uncaught or 'none'}")
    text = "\n".join(lines)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text)
    print(f"\n  wrote {os.path.relpath(args.out, _ROOT)}")
    if uncaught:
        print(f"\n  {len(uncaught)} injection(s) UNCAUGHT: {uncaught}. A guarantee "
              f"this slice claims is no longer enforced.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
