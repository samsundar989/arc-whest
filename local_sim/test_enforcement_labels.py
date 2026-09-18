"""The enforcement layer's STATED coverage, made executable.

Contract: `experiments/LOG.md`, *CONTRACT WP0B* (2026-08-31), committed at
`b39b9d0` before this file existed.

WHY THIS EXISTS
---------------
`scripts/preflight.ps1` and `docs/VALIDATION.md` §7 say how much each gate step
checks — "9. seed-ledger refusals - 60/60", "11. lifecycle evaluator - 56/56".
On 2026-08-31 three of those numbers were wrong (44 vs 60, 54 vs 56, 71 vs 98)
and `VALIDATION.md` attached step 9's count to step **7**, which reports no
count at all. Nothing noticed; an audit found it by hand.

`VALIDATION.md` already said *"a count in a runtime label is a snapshot and goes
stale on the next test added"* — and then carried four of them. **Prose cannot
enforce prose.** This suite parses the counts out of both documents, runs the
suite each one names, and fails on a disagreement. A stale label is now a red
test.

WHAT IT DOES NOT DO
-------------------
It does not judge whether a suite's checks are any *good*. A suite that grew
from 44 to 60 vacuous checks would satisfy section A and mean nothing. That is
what section C is for: it neuters two real guards in a scratch mirror and
requires the **named** check to be the one that fails. A red run for any other
reason is a HARNESS FAILURE, not a catch — the distinction
`local_sim/p1pre_fault_inject.py` was forced to learn after an injection died on
its own `SyntaxError` and was recorded as evidence.

READ-ONLY — WITH ONE EXCEPTION, NAMED
-------------------------------------
This suite writes nothing itself. The mirror is built in a system temp directory
and removed; the two documents are opened for reading and mutated only as
strings.

**But it RUNS other suites, and one of them writes.** `test_seed_ledger.py`
creates an isolated `local_sim/_ledger_scan_*/` to prove the source scan catches a real
seed literal in a real file, and removes it in a `finally`. That is a write
under the repository, and the first version of this docstring said there were
none — while section R hashed only the two documents, so its own proof could not
see the counterexample. An independent Skeptic measured it. Section R now checks
that path too, which is the difference between a claim and a check.

Run:  uv run python local_sim/test_enforcement_labels.py
      uv run python local_sim/test_enforcement_labels.py --full     # + 27/27, 48/48
      uv run python local_sim/test_enforcement_labels.py --labels-only
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# local_sim ships shims NAMED flopscope/whestbench: APPEND, never insert.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]

PREFLIGHT = os.path.join(_ROOT, "scripts", "preflight.ps1")
VALIDATION = os.path.join(_ROOT, "docs", "VALIDATION.md")

_FAILED = []
_TOTAL = [0]


def check(name, ok, detail=""):
    _TOTAL[0] += 1
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<66}{detail}")
    if not ok:
        _FAILED.append(name)


# --------------------------------------------------------------------------- #
# the suites, and where each one's count is claimed
# --------------------------------------------------------------------------- #
#: (key, argv, tail-pattern that captures the suite's OWN reported count, slow?)
#:
#: The pattern reads the suite's final line rather than a count we maintain
#: here. A number typed in this file would be one more snapshot to go stale --
#: which is the defect, not the fix.
SUITES = {
    "official_config": (["local_sim/official_config.py"],
                        r"^\s*(\d+)/(\d+)\s*$", False),
    "edge_cases": (["local_sim/edge_cases.py"],
                   r"^(\d+)/(\d+) checks passed", False),
    "competition_score": (["local_sim/test_competition_score.py"],
                          r"^(\d+)/(\d+) checks passed", False),
    "seed_ledger_refusals": (["local_sim/test_seed_ledger.py"],
                             r"^(\d+)/(\d+) checks passed", False),
    "lifecycle": (["local_sim/lifecycle_eval.py", "--self-test"],
                  r"^(\d+)/(\d+) checks passed", False),
    "shape_parity": (["local_sim/test_shape_parity.py"],
                     r"^(\d+)/(\d+) shapes passed", True),
    "carriers": (["local_sim/test_carriers.py"],
                 r"^(\d+)/(\d+) checks passed", True),
}

#: Where each suite's count is CLAIMED. `(document, regex capturing the count)`.
#: Every regex is anchored on surrounding words that identify the step, so a
#: count moving to a different step (the `VALIDATION.md` §7 defect) is caught
#: rather than matched by accident.
CLAIMS = [
    ("edge_cases", PREFLIGHT, r"3\. invariant suite\s+- (\d+)/(\d+) on the SHIPPED file", None),
    ("competition_score", PREFLIGHT, r"8\. canonical scorer\s+- (\d+)/(\d+), the competition identity", None),
    ("competition_score", PREFLIGHT, r"every gate downstream reads its numbers\. (\d+)/(\d+) asserts", None),
    ("seed_ledger_refusals", PREFLIGHT, r"9\. seed-ledger refusals - (\d+)/(\d+) negative tests", None),
    ("lifecycle", PREFLIGHT, r"11\. lifecycle evaluator - (\d+)/(\d+)\.", None),
    ("shape_parity", PREFLIGHT, r"13\. shape parity\s+- (\d+)/(\d+)\.", None),
    ("carriers", PREFLIGHT, r"14\. carrier strength\s+- (\d+)/(\d+) design-strength", None),
    # The count here is followed by a comma, not a paren: "(73/73, asserted
    # against the *installed* package)". Requiring `\)` matched nothing and the
    # suite reported official_config as UNCLAIMED.
    ("official_config", VALIDATION, r"official stack \+ round configuration \((\d+)/(\d+)[,)]", None),
    ("edge_cases", VALIDATION, r"invariant suite \((\d+)/(\d+)\)", None),
    ("competition_score", VALIDATION, r"canonical scorer \+ the binding gate \((\d+)/(\d+)\)", None),
    ("seed_ledger_refusals", VALIDATION, r"seed-ledger \*refusals\* \((\d+)/(\d+)\)", None),
    ("lifecycle", VALIDATION, r"lifecycle evaluator \((\d+)/(\d+)\)", None),
    ("edge_cases", VALIDATION, r"edge_cases\.py\s+# 2\. invariants, (\d+)/(\d+)", None),
    # Found by the exhaustive scan below, not by the hand-list: this one was
    # TRUE but unclaimed, so nothing would have noticed it going stale.
    ("edge_cases", VALIDATION, r"\*\*(\d+)/(\d+) passing\*\* as of 2026-08-06", None),
    ("seed_ledger_refusals", VALIDATION,
     r"drives every refusal and\s*\n?asserts it fires: \*\*(\d+)/(\d+)\*\*", None),
    ("seed_ledger_refusals", PREFLIGHT,
     r"These are (\d+) negative tests", "single"),
]

#: Claims that must NOT exist: a count attached to a step that reports none.
#: `VALIDATION.md` §7 carried `seed ledger (60/60)` on step 7 for days; step 7
#: is `seed_ledger.py --check --scan`, which prints CONSISTENT and no count.
FORBIDDEN_CLAIMS = [
    (VALIDATION, r"\*\*7\*\* seed ledger \(\d+/\d+\)",
     "step 7 (consumption + source scan) reports no count; a number there is "
     "step 9's, mis-attached"),
    (PREFLIGHT, r"7\. seed ledger\s+- \d+/\d+",
     "step 7 reports no count"),
]

# --------------------------------------------------------------------------- #
# THE EXHAUSTIVE SCAN — added after an independent Skeptic showed the hand-list
# above is structurally blind
# --------------------------------------------------------------------------- #
#
# `CLAIMS` is a hand-maintained list of anchored patterns. That means a count
# it does not happen to name is INVISIBLE, and `section_a` silently `continue`s
# on a pattern that stops matching. The Skeptic demonstrated both with two live
# counterexamples that had been false since WP0B's own repair:
#
#   * `scripts/preflight.ps1:397` -- "These are 44 negative tests" (prose, so no
#     `N/N` to anchor on) against a suite that reports 60;
#   * `docs/VALIDATION.md:202` -- "asserts it fires: **39/39**" for the same
#     suite, outside section 7 where the hand-list looks.
#
# So the header of `preflight.ps1` claiming this suite "parses every N/N in this
# docstring" was itself an overstatement. It is true now: every occurrence of a
# count in either document must be either MATCHED by a `CLAIMS` entry or listed
# in `ALLOWED_COUNTS` with a written reason. There is no third option, and a new
# unexplained count is a red test.
_COUNT_RE = re.compile(
    r"\d+/\d+|\b\d+ (?:negative tests|checks|invariants|shapes|tests)\b")

#: Occurrences that are NOT a suite's pass count. Each carries the reason it is
#: exempt; "it is inconvenient" is not one.
ALLOWED_COUNTS = [
    (PREFLIGHT, r"GREEN 12/16", "step denominator, not a suite count"),
    (PREFLIGHT, r'never "12/12"', "the anti-pattern this gate names on purpose"),
    (PREFLIGHT, r'report "12/12"', "the same anti-pattern, in the accounting note"),
    (PREFLIGHT, r"GREEN 16/16 at HEAD", "step denominator in a transcript example"),
    (PREFLIGHT, r"steps 15/16 regenerate", "a step RANGE, not a count"),
    (PREFLIGHT, r"7 of 17 shapes", "the v2.6 odd-width defect, a historical fact"),
    (VALIDATION, r"\| \*\*20/20\*\* \| 20/20 \| 0/20 \|",
     "the mint-provenance table: 20 seeds per shape, not a suite count"),
    (VALIDATION, r"6/6 lossless float32 round-trips",
     "a float round-trip count, not a suite"),
    (VALIDATION, r"\| 0/20 \|", "residual rows over the 0.4 s cap, per 20 MLPs"),
    (VALIDATION, r"was 24/24", "explicitly labelled as the historical value"),
    (VALIDATION, r"GREEN 12/16", "step denominator"),
    (VALIDATION, r"11/16, 5", "step denominator with the vault absent"),
    (VALIDATION, r"`9/11` as `9/9`", "the first build's accounting bug, quoted"),
    (VALIDATION, r"step 9's `60/60` to step \*\*7\*\*",
     "the WP0B defect, quoted in its own correction note"),
    # The note describing this very scan quotes the counts it was built to
    # catch. Each exemption is anchored on its surrounding prose, so it cannot
    # drift onto a live claim elsewhere in the file.
    (VALIDATION, r"count \(`\d+ negative tests`\)",
     "an EXAMPLE of the prose form the scan matches, inside its own note"),
    (VALIDATION, r'\*"\d+ negative tests"\*, prose, no `N/N` to anchor on',
     "the stale value quoted in the correction that fixed it"),
    (VALIDATION, r'fires: \*\*39/39\*\*"\*, both against a suite',
     "the stale value quoted in the correction that fixed it"),
]


def _run(argv, cwd=_ROOT, env=None):
    return subprocess.run([sys.executable] + argv, capture_output=True,
                          text=True, cwd=cwd, env=env, errors="replace")


def _receipt_identity(key, ledger_path=None):
    """Conservative closure for a count receipt, including runtime identity.

    A receipt only represents a suite executed earlier in the SAME preflight.
    """
    paths = [Path(PREFLIGHT), Path(VALIDATION), Path(_ROOT) / "estimator.py",
             Path(_ROOT) / "pyproject.toml", Path(_ROOT) / "uv.lock",
             Path(ledger_path) if ledger_path is not None else
             Path(_ROOT) / "experiments" / "seed_ledger.jsonl"]
    paths.extend(p for p in Path(_HERE).glob("*.py")
                 if not p.name.startswith("_lifecycle_"))
    paths.extend(p for p in (Path(_ROOT) / "scripts").rglob("*")
                 if p.is_file() and p.suffix in (".py", ".ps1"))
    payload = {}
    for p in paths:
        if p.is_file():
            name = ("experiments/seed_ledger.jsonl" if ledger_path is not None
                    and p == Path(ledger_path) else
                    str(p.relative_to(_ROOT)).replace("\\", "/"))
            payload[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    versions = {}
    for name in ("numpy", "flopscope", "whestbench"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    payload.update({"suite": key, "python": sys.version,
                    "executable": sys.executable, "platform": platform.platform(),
                    "packages": versions})
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _count(key, stdout):
    pat = SUITES[key][1]
    for line in reversed(stdout.splitlines()):
        m = re.match(pat, line.strip())
        if m:
            return int(m.group(1)), int(m.group(2))
    return None, None


def _receipt_path(key):
    directory = os.environ.get("ARC_WHEST_LABEL_RECEIPTS")
    return Path(directory) / (key + ".json") if directory else None


def measure(key, cwd=_ROOT):
    """Read same-preflight receipt, or execute a standalone suite."""
    receipt = _receipt_path(key) if cwd == _ROOT else None
    if receipt is not None:
        try:
            saved = json.loads(receipt.read_text(encoding="utf-8"))
            identity = _receipt_identity(key)
            proof = hashlib.sha256(f"{identity}:{saved.get('passed')}:{saved.get('total')}:0".encode()).hexdigest()
            if saved.get("identity") == identity and saved.get("proof") == proof \
                    and saved.get("rc") == 0 \
                    and type(saved.get("passed")) is int \
                    and saved["passed"] == saved.get("total"):
                print(f"  same-run receipt: {key} exact source/configuration match")
                return saved["passed"], saved["total"], 0
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError(f"missing/invalid same-run receipt for {key}: {exc}") from exc
        raise ValueError(f"tampered or stale same-run receipt for {key}")
    argv, pat, _ = SUITES[key]
    argv = [os.path.join(cwd, a) if a.endswith(".py") else a for a in argv]
    p = _run(argv, cwd=cwd)
    passed, total = _count(key, p.stdout or "")
    return passed, total, p.returncode


def emit_receipt(key):
    """Run the real suite once, relay its output, then publish its count."""
    receipt = _receipt_path(key)
    if receipt is None:
        raise ValueError("ARC_WHEST_LABEL_RECEIPTS is required")
    argv = SUITES[key][0]
    p = _run(argv)
    sys.stdout.write(p.stdout or "")
    sys.stderr.write(p.stderr or "")
    passed, total = _count(key, p.stdout or "")
    if p.returncode != 0 or passed is None or passed != total:
        return p.returncode or 1
    identity = _receipt_identity(key)
    proof = hashlib.sha256(f"{identity}:{passed}:{total}:0".encode()).hexdigest()
    receipt.parent.mkdir(parents=True, exist_ok=True)
    tmp = receipt.with_name(receipt.name + "." + str(os.getpid()) + ".tmp")
    tmp.write_text(json.dumps({"identity": identity, "passed": passed,
                               "total": total, "rc": 0,
                               "proof": proof}), encoding="utf-8")
    os.replace(tmp, receipt)
    return 0


# --------------------------------------------------------------------------- #
# A. the documented counts, against what the suites actually report
# --------------------------------------------------------------------------- #

def section_a(run_slow: bool):
    print("A  every documented N/N matches the suite it names")
    docs = {PREFLIGHT: open(PREFLIGHT, encoding="utf-8").read(),
            VALIDATION: open(VALIDATION, encoding="utf-8").read()}

    measured = {}
    for key, (_, _, slow) in SUITES.items():
        if slow and not run_slow:
            continue
        passed, total, rc = measure(key)
        measured[key] = (passed, total, rc)
        check(f"A: suite '{key}' runs green and reports a count",
              rc == 0 and passed is not None and passed == total,
              f"{passed}/{total}, rc={rc}")

    seen = set()
    for key, doc, pat, kind in CLAIMS:
        m = re.search(pat, docs[doc], re.M)
        # A CLAIM THAT STOPS MATCHING USED TO BE INVISIBLE. `continue` here
        # silently dropped it, and the only guard was "is this key claimed
        # somewhere" -- so for a key with two claims, one regex breaking left no
        # trace. A non-matching pattern is now a failure of THIS suite.
        if m is None:
            check(f"A: the claim pattern for {key} in "
                  f"{os.path.basename(doc)} still matches", False,
                  f"pattern {pat[:44]!r} matched nothing -- the label is "
                  "unguarded, not absent")
            continue
        if m.lastindex is None:
            continue                      # a claim with no numeric group
        seen.add(key)
        # A prose claim ("These are 60 negative tests") carries one number, not
        # a pair. It is compared against the suite's PASSED count.
        claimed = ((int(m.group(1)), int(m.group(1))) if kind == "single"
                   else (int(m.group(1)), int(m.group(2))))
        if key not in measured:
            check(f"A: {os.path.basename(doc)} claims {key} = "
                  f"{claimed[0]}/{claimed[1]} (not measured this run)",
                  True, "slow suite; run with --full to check it")
            continue
        passed, total, _ = measured[key]
        check(f"A: {os.path.basename(doc)} claims {key} = "
              f"{claimed[0]}/{claimed[1]}",
              claimed == (passed, total),
              f"suite reports {passed}/{total}"
              + ("" if claimed == (passed, total) else "   <-- STALE LABEL"))

    # ---- the exhaustive scan: no count may be invisible to this suite ----
    for doc in (PREFLIGHT, VALIDATION):
        text = docs[doc]
        covered = []
        for _k, d, pat, _ in CLAIMS:
            if d != doc:
                continue
            for m in re.finditer(pat, text, re.M):
                covered.append((m.start(), m.end()))
        for pat, _why in [(p, w) for d, p, w in ALLOWED_COUNTS if d == doc]:
            for m in re.finditer(pat, text, re.M):
                covered.append((m.start(), m.end()))
        orphans = []
        for m in _COUNT_RE.finditer(text):
            if not any(s <= m.start() and m.end() <= e for s, e in covered):
                line = text.count("\n", 0, m.start()) + 1
                orphans.append(f"{os.path.basename(doc)}:{line} {m.group(0)!r}")
        check(f"A: every count in {os.path.basename(doc)} is claimed or "
              "explicitly exempt",
              not orphans,
              "; ".join(orphans[:4]) if orphans else
              f"{len(covered)} spans cover every occurrence")

    for doc, pat, why in FORBIDDEN_CLAIMS:
        check(f"A: {os.path.basename(doc)} carries no count where none exists",
              re.search(pat, docs[doc], re.M) is None, why)

    # Every non-slow suite must be claimed SOMEWHERE, or the checker is blind
    # to it and a future label could appear unguarded.
    for key, (_, _, slow) in SUITES.items():
        if slow and not run_slow:
            continue
        check(f"A: '{key}' is claimed by at least one document",
              key in seen, "otherwise the label is unguarded")
    return measured


# --------------------------------------------------------------------------- #
# B. the checker itself must be able to fail
# --------------------------------------------------------------------------- #

def section_b(measured):
    print("\nB  the label checker is not vacuous -- mutate a label, it must complain")
    docs = {PREFLIGHT: open(PREFLIGHT, encoding="utf-8").read(),
            VALIDATION: open(VALIDATION, encoding="utf-8").read()}

    mutations = [
        (PREFLIGHT, r"9\. seed-ledger refusals - (\d+)/(\d+) negative tests",
         "seed_ledger_refusals", "preflight step 9"),
        (PREFLIGHT, r"11\. lifecycle evaluator - (\d+)/(\d+)\.",
         "lifecycle", "preflight step 11"),
        (VALIDATION, r"canonical scorer \+ the binding gate \((\d+)/(\d+)\)",
         "competition_score", "VALIDATION step 8"),
        (VALIDATION, r"seed-ledger \*refusals\* \((\d+)/(\d+)\)",
         "seed_ledger_refusals", "VALIDATION step 9"),
    ]
    for doc, pat, key, label in mutations:
        text = docs[doc]
        m = re.search(pat, text, re.M)
        if m is None:
            check(f"B: {label} -- the claim is present to mutate", False,
                  "pattern did not match; the checker would silently skip it")
            continue
        passed, total, _ = measured[key]
        if passed is None:
            # Section A already failed this suite. Crashing here with a
            # TypeError turns one named failure into a traceback that hides
            # every remaining check -- which is how a red run stops being
            # readable. Report and continue.
            check(f"B: a wrong count at {label} is DETECTED", False,
                  f"cannot mutate: suite '{key}' reported no count in section A")
            continue
        bad = str(passed + 1)
        mutated = text[:m.start(1)] + bad + text[m.end(1):m.start(2)] \
            + bad + text[m.end(2):]
        m2 = re.search(pat, mutated, re.M)
        detected = m2 is not None and (int(m2.group(1)), int(m2.group(2))) \
            != (passed, total)
        check(f"B: a wrong count at {label} is DETECTED",
              detected,
              f"{passed}/{total} -> {bad}/{bad}, mismatch seen: {detected}")

    # And the forbidden-claim guard must fire when the defect is reintroduced.
    reintroduced = docs[VALIDATION].replace(
        "**7** seed ledger consumption + source scan",
        "**7** seed ledger (60/60)", 1)
    fired = re.search(FORBIDDEN_CLAIMS[0][1], reintroduced, re.M) is not None
    check("B: reintroducing the step-7 mis-attachment is DETECTED", fired,
          "the exact 2026-08-31 defect, put back on purpose")
    check("B: ...and the mutation actually changed the document",
          reintroduced != docs[VALIDATION],
          "a no-op mutation would make the check above vacuous")


# --------------------------------------------------------------------------- #
# C. the guards whose stated coverage moved must still FIRE
# --------------------------------------------------------------------------- #
#: (tag, title, target file relative to the mirror, [(find, replace)], suite key,
#:  the NAMED check that must fail). The last field is the oracle: an exit code
#:  alone is not a catch.
GUARD_INJECTIONS = [
    (
        "S1",
        "step 9 -- the SINGLE_USE second-claim refusal is neutered",
        os.path.join("local_sim", "seed_ledger.py"),
        [('    if tier == "SINGLE_USE" and mine:',
          '    if False and mine:')],
        "seed_ledger_refusals",
        "a SECOND claim on D17 is refused even on disjoint seeds",
    ),
    (
        "L1",
        "step 11 -- the residual-cliff failure path stops firing",
        os.path.join("local_sim", "lifecycle_eval.py"),
        [("                and residual_s > spec.residual_wall_time_limit_s):",
          "                and False):")],
        "lifecycle",
        "L4: a 1e-9 s residual limit trips the residual path",
    ),
]

#: What a mirror needs so both suites run exactly as they do in the repository.
MIRROR_TREES = ("local_sim", "scripts")
#: `lifecycle_eval.py --self-test` loads the SHIPPED file (it refuses with
#: "candidate 'estimator.py' does not exist" without it), and
#: `test_seed_ledger.py` reads the real append-only ledger.
MIRROR_FILES = (os.path.join("experiments", "seed_ledger.jsonl"),
                "estimator.py")


def build_mirror(dst):
    for tree in MIRROR_TREES:
        shutil.copytree(os.path.join(_ROOT, tree), os.path.join(dst, tree),
                        ignore=shutil.ignore_patterns("__pycache__"))
    for f in MIRROR_FILES:
        d = os.path.join(dst, os.path.dirname(f))
        os.makedirs(d, exist_ok=True)
        shutil.copyfile(os.path.join(_ROOT, f), os.path.join(dst, f))


def _failed_check_names(stdout):
    """The names of checks the suite reported as failures.

    Both suites print `  <name>  <status>  <detail>` with status `ok` or
    `FAIL`/`fail`. Reading the FAIL lines is what distinguishes a caught guard
    from a suite that died for an unrelated reason.
    """
    out = []
    for line in (stdout or "").splitlines():
        s = line.strip()
        # test_seed_ledger.py prints "FAILED", lifecycle_eval.py prints "FAIL".
        # `\bFAIL\b` alone matches only the second, which is how the first
        # injection reported "0 FAIL line(s)" against a suite that had one.
        if re.search(r"\bFAIL(ED)?\b", s):
            out.append(s)
    return out


def section_c():
    print("\nC  the two guards whose stated coverage moved still FIRE when removed")
    tmp = tempfile.mkdtemp(prefix="wp0b_guard_mirror_")
    try:
        build_mirror(tmp)

        # P5: the mirror must reproduce the repository's own counts, or a
        # mutation's red is not attributable to the mutation.
        baseline = {}
        for key in ("seed_ledger_refusals", "lifecycle"):
            passed, total, rc = measure(key, cwd=tmp)
            repo_passed, repo_total, _ = measure(key, cwd=_ROOT)
            baseline[key] = (passed, total)
            check(f"C: the UNMUTATED mirror reproduces '{key}' exactly",
                  rc == 0 and (passed, total) == (repo_passed, repo_total),
                  f"mirror {passed}/{total} vs repo {repo_passed}/{repo_total}")

        for tag, title, target, edits, key, expect in GUARD_INJECTIONS:
            path = os.path.join(tmp, target)
            original = open(path, encoding="utf-8").read()
            mutated = original
            applied = 0
            for find, repl in edits:
                if mutated.count(find) == 1:
                    mutated = mutated.replace(find, repl, 1)
                    applied += 1
            check(f"C[{tag}]: the injection has a unique target to edit",
                  applied == len(edits),
                  f"{applied}/{len(edits)} edits applied in {target}")
            if applied != len(edits):
                continue
            open(path, "w", encoding="utf-8").write(mutated)
            try:
                argv, pat, _ = SUITES[key]
                argv = [os.path.join(tmp, a) if a.endswith(".py") else a
                        for a in argv]
                p = _run(argv, cwd=tmp)
                fails = _failed_check_names(p.stdout)
                named = any(expect in f for f in fails)
                check(f"C[{tag}]: {title} -- the suite goes RED",
                      p.returncode != 0,
                      f"rc={p.returncode}, {len(fails)} FAIL line(s)")
                check(f"C[{tag}]: ...and the NAMED guard is the one that fails",
                      named,
                      (f"'{expect[:44]}...'" if named
                       else "HARNESS FAILURE -- red for some other reason: "
                            + "; ".join(fails[:2])[:110]))
            finally:
                open(path, "w", encoding="utf-8").write(original)

            # And the restore must return the mirror to green, so the next
            # injection starts from a known state rather than from damage.
            passed, total, rc = measure(key, cwd=tmp)
            check(f"C[{tag}]: the mirror is restored to green after the injection",
                  rc == 0 and (passed, total) == baseline[key],
                  f"{passed}/{total}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    check("C: the scratch mirror is removed", not os.path.exists(tmp), tmp)


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--full", action="store_true",
                    help="also run shape parity (~480 s) and carriers (~130 s)")
    ap.add_argument("--labels-only", action="store_true",
                    help="skip the guard-mutation mirror (sections C)")
    ap.add_argument("--emit-receipt", choices=SUITES,
                    help="preflight: run this suite once and issue a same-run count receipt")
    a = ap.parse_args()
    if a.emit_receipt:
        raise SystemExit(emit_receipt(a.emit_receipt))

    print("WP0B - the enforcement layer's stated coverage, under test\n")
    import hashlib
    before = {p: hashlib.sha256(open(p, "rb").read()).hexdigest()
              for p in (PREFLIGHT, VALIDATION)}

    measured = section_a(a.full)
    section_b(measured)
    if not a.labels_only:
        section_c()

    print("\nR  the two documents were opened read-only, and the one repository")
    print("   write any child makes is left clean")
    for p, h in before.items():
        now = hashlib.sha256(open(p, "rb").read()).hexdigest()
        check(f"R: {os.path.basename(p)} sha256 unchanged", now == h, h[:16])
    # `test_seed_ledger.py` writes here to prove the source scan catches a real
    # literal in a real file, and removes it in a `finally`. Checking the
    # residue is what lets the docstring above make a claim it can support.
    scan_tmp = list(Path(_HERE).glob("_ledger_scan_*/"))
    check("R: no local_sim/_ledger_scan_*/ left by a child",
          not scan_tmp,
          "the one repository write this suite's children make, cleaned up")

    print()
    if _FAILED:
        print(f"FAILED {len(_FAILED)}: " + "; ".join(_FAILED))
        raise SystemExit(1)
    print(f"{_TOTAL[0]}/{_TOTAL[0]} checks passed")


if __name__ == "__main__":
    main()
