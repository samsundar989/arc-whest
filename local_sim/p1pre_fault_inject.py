"""Break each P1-PRE guarantee on purpose, in a scratch mirror, and grade the REASON.

P1-PRE (`experiments/LOG.md`, 2026-08-25). Companion to
`local_sim/test_deployment_envelope.py` and `local_sim/test_envelope_readonly.py`.

WHY THIS IS IN THE REPOSITORY
-----------------------------
A check nobody has watched fail is a check nobody has tested, and that is the
whole subject of P1-PRE: the slice exists because a `check(name, True, ...)` sat
green for a week while the constraint it named was provoked nowhere. So the
slice's evidence is not that its suites pass. It is that they FAIL when the
guarantees are removed — which means the harness must live where the next session
can re-run it.

TWO THINGS THIS FILE GOT WRONG, BOTH FOUND BY INDEPENDENT REVIEW
----------------------------------------------------------------
**1. It mutated the repository.** Earlier revisions edited the real source in
place, restored it from a backup, and ran `git checkout --` over two tracked
evidence artifacts after *every* injection — whether or not that injection had
touched them. Its refuse-on-dirty guard covered only the files it edited, so
uncommitted work in `lifecycle_self_test.json` or `deployment_envelope.json` was
silently discarded. That produced no false green; it destroyed evidence, and a
tool that requires a clean tree in order not to eat your work is a tool with a
loaded footgun.

**Now: nothing under the repository is written, moved or restored, ever** — and
that sentence took two attempts to make true. A scratch MIRROR is built in a
system temp directory (`local_sim/`, `estimator.py` and the P2-P0 evidence tree,
copied out) and made into a throwaway git repo so the suites' own `git status`
checks have something real to talk to. Mutations are written into the mirror. The
repository is opened read-only. There is no `git checkout` in this file, and
unrelated dirty work in your tree is untouched.

**The first attempt asserted that sentence while violating it.** The transcript's
default `--out` was a *tracked* file, the in-repo refusal carved that one path
out, and the "REPOSITORY UNTOUCHED" measurement ran *before* the write over a
protected list that omitted it — so the tool's one repository write was invisible
to its own proof that it made none. A sixth independent Skeptic measured all
three together. Fixed by making the rule containment rather than tracking status
(**any** path inside the repository is refused, tracked or not), defaulting the
transcript to a temp directory, adding the old tracked transcript to the
protected set, and taking the measurement **after** the write. If you want the
transcript as committed evidence, copy it in yourself — deliberately, as a human
act, which is the distinction that was missing.

**2. Its verdict was a state, not a property.** `caught = returncode != 0` counts
any red run as a catch — including a run that died on a `SyntaxError` the
injection itself introduced, which happened, and was recorded as evidence. Two
injections were later found broken: one failing open (it reported UNCAUGHT
without ever reaching its target) and one failing closed (a syntax error scored
as a catch). Only reading the FAIL lines distinguished them.

**Now every injection declares the check it expects to fail**, and a run that
goes red for any other reason is reported `HARNESS FAILURE`, not `CAUGHT`. An
injection that does not reach its target is a broken test, and it looks exactly
like a broken guard until something grades the reason.

USAGE
-----
    uv run python local_sim/p1pre_fault_inject.py
    uv run python local_sim/p1pre_fault_inject.py --only L,M
    uv run python local_sim/p1pre_fault_inject.py --out <path outside the repo>

Exit 0 means every injection was CAUGHT BY ITS NAMED GUARD. Non-zero means either
a guarantee is no longer enforced (`UNCAUGHT`) or an injection stopped testing
what it claims (`HARNESS FAILURE`). Both are findings. `--out` refuses any path
that resolves inside the repository -- tracked or not, and through a
symlink or junction. Containment is the test; tracking status is not.

**It is not part of the gate.** `preflight.ps1` runs the two suites; this runs
their falsification, and is invoked by hand when the guarantees are re-argued.
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

SUITE_BASE = "test_deployment_envelope.py"
READONLY_BASE = "test_envelope_readonly.py"
#: The default transcript path, and it is OUTSIDE the repository.
#:
#: It used to default to the tracked `experiments/.../fault_injection.txt`, and
#: the in-repo refusal below carved that one path out — so the tool's one and
#: only repository write was the one its own refusal permitted, and the module
#: docstring said "nothing under the repository is written, moved or restored,
#: ever". A sixth independent Skeptic measured the carve-out and the false
#: universal together. Writing outside the tree by default is what makes the
#: sentence true instead of qualified; copy the transcript into `experiments/`
#: deliberately if you want it as committed evidence.
OUT = os.path.join(tempfile.gettempdir(), "arc_whest_p1pre_fault_injection.txt")

#: The transcript this tool USED to write. It is tracked, and it is now in the
#: protected set below precisely because it was the file the old carve-out
#: allowed the tool to overwrite.
LEGACY_OUT = os.path.join(_ROOT, "experiments", "2026-08-25_p1_pre",
                          "fault_injection.txt")

#: What the mirror needs so both suites run exactly as they do in the repository.
MIRROR_TREES = ("local_sim", os.path.join("experiments", "2026-08-24_p2_p0"))
MIRROR_FILES = ("estimator.py",)

#: (tag, title, target basename, [(find, replace)], argv-in-mirror, expected FAIL
#:  substring). The last field is the oracle: the guard that must be the one to
#:  fire. Nothing here is graded on an exit code alone.
INJECTIONS = [
    (
        "A", "D1 -- section 7 reports an UNRUN provocation as green again",
        SUITE_BASE,
        [("""        elif not ok and not _SECTION_RAN[name]:
            # The section did not run: a gap, not a failure.
            unprovable(f"{name} (witness)", why)""",
          """        elif not ok and not _SECTION_RAN[name]:
            check(f"{name}: its named provocation RAN and PASSED in THIS suite",
                  True, why)""")],
        [READONLY_BASE],
        "NO green line CLAIMS worker_death_restart",
    ),
    (
        "B", "D2 -- the resolver ignores the suite it was given",
        SUITE_BASE,
        [("""        src = self._registry.get(suite) if isinstance(suite, str) else None
        if src is None:
            return "UNKNOWN_SUITE", None""",
          """        src = (self._registry.get(suite) if isinstance(suite, str) else None)
        if src is None:
            src = self._registry.get(LIFECYCLE_SUITE)
        if src is None:
            return "UNKNOWN_SUITE", None
        suite = LIFECYCLE_SUITE""")],
        [SUITE_BASE, "--no-write", "--skip-worker-death"],
        "REFUSED: a witness naming a suite NOBODY REGISTERED",
    ),
    (
        "C", "D3 -- the read-only latch stops refusing",
        SUITE_BASE,
        [("""def refuse_if_read_only(what: str) -> None:
    if _READ_ONLY:""",
          """def refuse_if_read_only(what: str) -> None:
    if False:"""),
         ("""        if self._allow_regenerate and not read_only_armed() and src.regenerate:""",
          """        if self._allow_regenerate and src.regenerate:""")],
        [READONLY_BASE],
        "with the latch ARMED, the same call RAISES and writes nothing",
    ),
    (
        "D", "a section forgets to record whether it ran",
        SUITE_BASE,
        [("""        _SECTION_RAN["worker_death_restart"] = False
        unprovable("worker_death_restart", "--skip-worker-death was passed")""",
          """        unprovable("worker_death_restart", "--skip-worker-death was passed")""")],
        [SUITE_BASE, "--no-write", "--skip-worker-death"],
        "its section RECORDED whether it ran",
    ),
    (
        "E", "a self-provoked constraint never reaches section 7",
        SUITE_BASE,
        [("""        if isinstance(w, dict) and w.get("suite") == SELF_SUITE:
            deferred.append((name, c))
            pending(name, "provoked by this suite, which has not run it yet")
            continue""",
          """        if isinstance(w, dict) and w.get("suite") == SELF_SUITE:
            continue""")],
        [SUITE_BASE, "--no-write", "--skip-worker-death"],
        "every SELF-PROVOKED constraint in the matrix reached this section",
    ),
    (
        "F", "the literal-True defect returns in `ok=` keyword form",
        SUITE_BASE,
        [("""    check("the matrix carries EXACTLY the required envelope constraints",
          present == required,
          f"{len(present)} present; missing {sorted(required - present) or 'none'}; "
          f"unpinned {sorted(present - required) or 'none'}")""",
          """    check("the matrix carries EXACTLY the required envelope constraints",
          ok=True,
          detail="injection F: a literal truthy condition, ok= keyword form")""")],
        [READONLY_BASE],
        "no check() has a literal truthy condition",
    ),
    (
        "G", "a witness is repointed at a check that ALWAYS runs",
        SUITE_BASE,
        [("""                    "checks": [WD_SURVIVED, WD_EXACTLY_ONE, WD_INDEX,
                               WD_CLASSIFIED, WD_LATER_CLEAN, WD_LATER_SCORED],
                    "expect_detail": {
                        WD_EXACTLY_ONE: "n_failed_mlps = 1",
                        WD_INDEX: "failed indices [0]",
                    },""",
          """                    "checks": ["the runner waits max(predict_timeout, wall "
                               "+ grace), not predict_timeout"],""")],
        [SUITE_BASE, "--no-write", "--skip-worker-death"],
        "emits the witness it is PINNED to",
    ),
    (
        "H", "an envelope constraint is DELETED from the matrix",
        SUITE_BASE,
        [("""            "predict_response_timeout_s": {""",
          """            "deleted_by_injection_h": {""")],
        [SUITE_BASE, "--no-write", "--skip-worker-death"],
        "the matrix carries EXACTLY the required envelope constraints",
    ),
    (
        "I", "an envelope constraint is ADDED without being pinned",
        SUITE_BASE,
        [("""            "predict_response_timeout_s": {""",
          """            "added_by_injection_i": {
                "value": None,
                "enforced_locally": False,
                "by": "injection I: an unpinned constraint",
                "provoked_by": None,
            },
            "predict_response_timeout_s": {""")],
        [SUITE_BASE, "--no-write", "--skip-worker-death"],
        "the matrix carries EXACTLY the required envelope constraints",
    ),
    (
        "J", "a documented blind spot is deleted, improving the coverage figure",
        READONLY_BASE,
        # RESTORED 2026-08-25. J was dropped when the injection set was rebuilt
        # for the terminal pass, while a docstring in `test_envelope_readonly.py`
        # went on citing it as the evidence that `COVERAGE_CLAIM` can fail --
        # a deleted regression test with a live reference to it. The sixth
        # independent Skeptic re-ran J against HEAD and it still fires, so the
        # guarantee was never lost; only its watched-failing evidence was.
        [("""        ("truthy call", "check('a', bool(1), 'd')"),
        ("aliased callee", "c('a', True, 'd')"),""",
          """        ("truthy call", "check('a', bool(1), 'd')"),""")],
        [READONLY_BASE],
        "the scan's coverage matches the PINNED claim",
    ),
    (
        "K", "a constraint's ENFORCEMENT claim is flipped, relocating it silently",
        SUITE_BASE,
        [("""            "flop_budget": {
                "value": r.flop_budget,
                "enforced_locally": True,""",
          """            "flop_budget": {
                "value": r.flop_budget,
                "enforced_locally": False,""")],
        [SUITE_BASE, "--no-write", "--skip-worker-death"],
        "makes the enforcement claim it is pinned to",
    ),
    (
        "L", "M4 -- the WALL witness is repointed at the RESIDUAL check",
        SUITE_BASE,
        # THE review-5 mutation. At 7e81600 this was green, 36/36, zero FAIL
        # lines: it reproduced the original P2-P0 false-provenance defect --
        # a constraint claiming a provocation that provokes a DIFFERENT constant
        # -- inside the mechanism built to close it. Structured witnesses refuse
        # prose and refuse names that do not exist; only the semantic pin refuses
        # a name that exists and is wrong.
        [("""                    "checks": [
                        "L4: a 1e-9 s WALL limit trips the TIME path",
                        "L4: wall-exhausted rows are exactly zeroed at multiplier 1.0",
                        "L4: ...and the WALL path is distinct from the RESIDUAL path",
                    ],
                    "expect_detail": {
                        "L4: a 1e-9 s WALL limit trips the TIME path":
                            "2/2 time_exhausted",
                        "L4: ...and the WALL path is distinct from the RESIDUAL path":
                            "residual_exhausted clear",
                    },""",
          """                    "checks": [
                        "L4: a 1e-9 s residual limit trips the residual path",
                    ],""")],
        [SUITE_BASE, "--no-write", "--skip-worker-death"],
        "emits the witness it is PINNED to",
    ),
    (
        "M", "M3a -- flop_budget is relocated into the platform exemption",
        SUITE_BASE,
        [("""    "flop_budget": True,
    "wall_time_limit_s": True,""",
          """    "wall_time_limit_s": True,"""),
         ("""PLATFORM_DEPENDENT_CONSTRAINTS = ("memory_limit_mb",)""",
          """PLATFORM_DEPENDENT_CONSTRAINTS = ("memory_limit_mb", "flop_budget")""")],
        [SUITE_BASE, "--no-write", "--skip-worker-death"],
        "only the permitted constraint has a PLATFORM-DEPENDENT classification",
    ),
    (
        "N", "M1 -- a REQUIRED file is never actually opened by the scanner",
        READONLY_BASE,
        [("""    for base in SCANNED:
        found = _constant_conditions(os.path.join(_HERE, base), opened=opened)""",
          """    for base in SCANNED[:2]:
        found = _constant_conditions(os.path.join(_HERE, base), opened=opened)""")],
        [READONLY_BASE],
        "every REQUIRED file was ACTUALLY OPENED by the scanner",
    ),
    (
        "O", "M2 -- a probe spelling is substituted for a duplicate of another",
        READONLY_BASE,
        [("""        ("ok= keyword", "check('a', ok=True, detail='d')"),""",
          """        ("ok= keyword", "check('a', True, 'd')"),""")],
        [READONLY_BASE],
        "every probe snippet is DISTINCT",
    ),
]


def _sha(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _git(*args: str, cwd: str = _ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def _tracked_evidence() -> list:
    """Repository paths whose bytes must be identical before and after.

    `LEGACY_OUT` is here because it is the file the old carve-out let this tool
    overwrite: a protected list that omits the one path the tool actually wrote
    is a proof that cannot see its own counterexample.
    """
    return [
        os.path.join(_HERE, SUITE_BASE),
        os.path.join(_HERE, READONLY_BASE),
        os.path.join(_ROOT, "experiments", "2026-08-24_p2_p0",
                     "lifecycle_self_test.json"),
        os.path.join(_ROOT, "experiments", "2026-08-24_p2_p0",
                     "deployment_envelope.json"),
        LEGACY_OUT,
    ]


def build_mirror(dst: str) -> None:
    """Copy what the suites touch into `dst` and make it a throwaway git repo.

    A git repo because both suites legitimately run `git status --porcelain` --
    that is how the read-only proof detects a write. Giving them a real one in
    scratch is what lets the mutations run without the repository being involved
    at all.
    """
    for tree in MIRROR_TREES:
        shutil.copytree(os.path.join(_ROOT, tree), os.path.join(dst, tree))
    for f in MIRROR_FILES:
        shutil.copyfile(os.path.join(_ROOT, f), os.path.join(dst, f))
    _git("init", "-q", cwd=dst)
    _git("add", "-A", cwd=dst)
    _git("-c", "user.email=harness@local", "-c", "user.name=harness",
         "commit", "-q", "-m", "mirror", cwd=dst)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", default="",
                    help="comma-separated injection tags, e.g. L,M")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    # ANY --out inside the repository is refused -- tracked, untracked, existing
    # or not. The previous rule refused only TRACKED paths and then exempted this
    # tool's own tracked default, which is how a tool that must not write the
    # repository came to write it on every invocation. There is no carve-out now:
    # the test is containment, not tracking status.
    # realpath, NOT abspath: `abspath` does not resolve reparse points, so a
    # Windows junction or a symlink aiming into the repository slipped through
    # containment. With a gitignored target -- `experiments/vault/vault_v1.npz`
    # is the spelling that matters -- the write landed inside the tree, `git
    # status` saw nothing, and the tail line printed `inside the repository:
    # False`. A false green on the path that would destroy the vault array.
    # Found by the accepting review and closed here rather than carried.
    out_abs = os.path.realpath(args.out)
    rel = os.path.relpath(out_abs, _ROOT)
    if not rel.startswith(".." + os.sep) and rel != "..":
        print(f"REFUSING: --out {rel.replace(os.sep, '/')} resolves INSIDE the "
              f"repository. This harness does not write the repository at all; "
              f"point it elsewhere and copy the transcript in yourself if you "
              f"want it as committed evidence.")
        print(f"          default: {OUT}")
        return 2

    wanted = {t.strip().upper() for t in args.only.split(",") if t.strip()}
    plan = [i for i in INJECTIONS if not wanted or i[0] in wanted]
    if wanted and len(plan) != len(wanted):
        print(f"unknown tag(s): {sorted(wanted - {i[0] for i in plan})}")
        return 2

    # The repository, read-only, before and after. Nothing below writes to it.
    before_status = _git("status", "--porcelain").stdout
    before_hash = {p: _sha(p) for p in _tracked_evidence()}

    mirror = tempfile.mkdtemp(prefix="arc_whest_mirror_")
    lines: list = []
    uncaught: list = []
    harness_failures: list = []
    try:
        build_mirror(mirror)
        m_local = os.path.join(mirror, "local_sim")
        pristine = {b: open(os.path.join(_HERE, b), encoding="utf-8").read()
                    for b in (SUITE_BASE, READONLY_BASE)}

        for tag, title, target, subs, argv, expect in plan:
            src = pristine[target]
            for old, new in subs:
                if old not in src:
                    print(f"injection {tag}: ANCHOR NOT FOUND in {target} -- the "
                          f"code it breaks has moved. Update this file.\n"
                          f"{old[:160]}")
                    return 3
                src = src.replace(old, new, 1)
            with open(os.path.join(m_local, target), "w", encoding="utf-8") as fh:
                fh.write(src)

            cmd = [sys.executable, os.path.join(m_local, argv[0])] + list(argv[1:])
            p = subprocess.run(cmd, cwd=mirror, capture_output=True, text=True,
                               env=dict(os.environ, PYTHONIOENCODING="utf-8"))
            out = p.stdout + p.stderr
            fails = [ln.strip() for ln in out.splitlines()
                     if ln.strip().startswith("FAIL")]
            on_target = [ln for ln in fails if expect in ln]

            # THE ORACLE. Red is not enough; the NAMED guard must be the one that
            # fired. A red run with no on-target failure is a broken injection.
            if not fails and p.returncode == 0:
                verdict = "UNCAUGHT -- the defect was reintroduced and NOTHING noticed"
                uncaught.append(tag)
            elif not on_target:
                verdict = (f"HARNESS FAILURE -- it went red, but not for its own "
                           f"reason; expected a FAIL containing {expect!r}")
                harness_failures.append(tag)
            else:
                verdict = "CAUGHT by its named guard"

            lines.append("=" * 100)
            lines.append(f"INJECTION {tag} -- {title}")
            lines.append(f"edits:    {target}   (in a scratch mirror, never the repo)")
            lines.append(f"command:  {' '.join(os.path.basename(a) for a in argv)}")
            lines.append(f"expects:  a FAIL containing {expect!r}")
            lines.append(f"exit {p.returncode}   ({verdict})")
            lines.append(f"FAIL lines: {len(fails)}, on target: {len(on_target)}")
            lines.extend("  " + ln for ln in fails)
            lines.append("")

            # Restore the mirror from the repository's own bytes, read-only.
            with open(os.path.join(m_local, target), "w", encoding="utf-8") as fh:
                fh.write(pristine[target])
            for tree in (os.path.join("experiments", "2026-08-24_p2_p0"),):
                shutil.rmtree(os.path.join(mirror, tree))
                shutil.copytree(os.path.join(_ROOT, tree),
                                os.path.join(mirror, tree))
    finally:
        shutil.rmtree(mirror, ignore_errors=True)

    lines.append("=" * 100)
    lines.append(f"injections run: {[i[0] for i in plan]}")
    lines.append(f"  uncaught:         {uncaught or 'none'}")
    lines.append(f"  harness failures: {harness_failures or 'none'}")

    # THE TRANSCRIPT IS WRITTEN FIRST, AND THE MEASUREMENT COMES AFTER IT.
    #
    # It used to be the other way round: the "REPOSITORY UNTOUCHED" block was
    # computed before this write, over a protected list that omitted the file
    # being written. So the tool's single repository write was structurally
    # invisible to the tool's own proof that it performed none -- a measurement
    # taken before the event it certifies absent. Measuring afterwards is what
    # makes it a measurement.
    text = "\n".join(lines)
    os.makedirs(os.path.dirname(out_abs) or ".", exist_ok=True)
    with open(out_abs, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")

    after_status = _git("status", "--porcelain").stdout
    after_hash = {p: _sha(p) for p in _tracked_evidence()}
    moved = [os.path.relpath(p, _ROOT) for p in before_hash
             if before_hash[p] != after_hash[p]]

    tail = ["=" * 100,
            "REPOSITORY UNTOUCHED -- measured AFTER the transcript was written:",
            f"  transcript path: {out_abs}",
            f"  inside the repository: "
            f"{not os.path.relpath(out_abs, _ROOT).startswith('..')}",
            f"  git status --porcelain identical:  {before_status == after_status}"
            f"   ({len(before_status.splitlines())} line(s) before, "
            f"{len(after_status.splitlines())} after)",
            f"  protected files with a moved sha256: {moved or 'none'}"]
    for p in sorted(before_hash):
        tail.append(f"    {os.path.relpath(p, _ROOT)}  {before_hash[p][:16]}")
    with open(out_abs, "a", encoding="utf-8") as fh:
        fh.write("\n".join(tail) + "\n")
    text = text + "\n" + "\n".join(tail)
    print(text)
    print(f"\n  wrote {out_abs}")

    rc = 0
    if before_status != after_status or moved:
        print("\n  FATAL: the repository changed. This harness must not be able "
              "to do that.")
        rc = 4
    if uncaught:
        print(f"\n  {len(uncaught)} UNCAUGHT: {uncaught}. A guarantee this slice "
              f"claims is no longer enforced.")
        rc = rc or 1
    if harness_failures:
        print(f"\n  {len(harness_failures)} HARNESS FAILURE(S): {harness_failures}. "
              f"Those injections no longer test what they claim.")
        rc = rc or 5
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
