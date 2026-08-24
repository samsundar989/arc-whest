#!/usr/bin/env python
"""Is this harness still swappable between agents?

The portability audit (2026-08-07) established one property: Samuel can switch the
driving agent at any point without changing his methodology, losing durable state, or
remembering a different workflow.  That property decays silently -- one Claude-only
instruction, one workflow that exists on one side, one adapter that grew a rule of its
own -- so it is checked mechanically here rather than believed.

    uv run python scripts/verify_portability.py
    uv run python scripts/verify_portability.py --verbose

Exit 0 = the harness is portable.  Exit 1 = a named check failed.

Run this after any edit under ai/workflows/, .claude/, .agents/, .codex/ or scripts/.
It touches no seed block, no vault, and no shipped file, so it is ROUTINE to run --
but *editing* what it checks is IMPORTANT-class work (AGENTS.md § Work classification).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from guard_paths import ALLOW, ASK, DENY, decide  # noqa: E402

# The six workflow commands and the slice-contract primitive were REMOVED on
# 2026-08-21 along with ai/workflows/ and both skills directories. What they
# enforced is now Rules 1 and 2 in AGENTS.md, and those are prose that no checker
# can verify -- so this file no longer pretends to. What it still checks is what
# is still mechanical: both agents load the SAME Skeptic brief, run the SAME
# enforcement implementation, and carry no project behaviour of their own.
_REMOVED_WORKFLOWS = ["session-start", "next-slice", "review-slice",
             "cold-start", "routine", "session-close"]
_REMOVED_INTERNAL = ["slice-contract"]
_REMOVED = _REMOVED_WORKFLOWS + _REMOVED_INTERNAL

#: The canonical Skeptic brief, shared by both agents. The one piece of the old
#: workflow layer that was kept, because it is a role rather than a ceremony.
SKEPTIC_BRIEF = "docs/SKEPTIC.md"

CANON = REPO / "docs"

# An adapter is allowed to be this long. Past it, it has started carrying project
# behaviour that belongs in ai/workflows/ where both agents can see it.
ADAPTER_MAX_LINES = 60

# The canonical layer must name no product and no model -- roles only, so the
# workflow survives a change of vendor or model generation.
#
# Path references are exempt: `.claude/skills/...` and `.codex/agents/...` are where
# the adapters physically live, and a canonical file is allowed to say so. The
# lookbehind/lookahead exclude a name that sits inside a path; what is caught is a
# bare product name used as part of an instruction.
BANNED = re.compile(
    r"(?<![./\w-])(claude|codex|gpt-?\d|opus|sonnet|haiku|fable|gemini|anthropic|"
    r"openai|copilot|cursor)(?![./\w-])",
    re.IGNORECASE,
)

# Repo-relative file references worth existence-checking. Requires a directory
# separator so bare names and prose words cannot match.
FILE_REF = re.compile(r"`([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+\.(?:md|py|ps1|json|jsonl))`")

results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    results.append((bool(ok), name, detail))
    return bool(ok)


def read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def frontmatter_name(text: str) -> str | None:
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.S)
    if not m:
        return None
    m2 = re.search(r"^name:\s*(\S+)\s*$", m.group(1), re.M)
    return m2.group(1) if m2 else None


# --- 1. portable sources of truth ---------------------------------------------

def check_sources_of_truth() -> None:
    agents = REPO / "AGENTS.md"
    if not check(agents.exists(), "AGENTS.md exists"):
        return
    text = read(agents)
    for heading in ["## Objective", "## Hard rules", "## Measurement law",
                    "## How work is done", "## Work classification", "## Continuity",
                    "## Enforcement", "## Capabilities", "## Agent adapters"]:
        check(heading in text, "AGENTS.md has '%s'" % heading)
    # Instruction discovery degrades with size; CLAUDE.md was 9.5 KB and worked.
    kb = len(text.encode("utf-8")) / 1024
    check(kb < 20, "AGENTS.md is concise enough for discovery", "%.1f KB (budget 20)" % kb)

    for doc in ["docs/CURRENT_STATE.md", "docs/HUMAN_MODEL.md", "docs/VALIDATION.md",
                "docs/GOTCHAS.md", "docs/GAME_PLAN.md", "experiments/LOG.md",
                "scripts/preflight.ps1"]:
        check((REPO / doc).exists(), "durable source present: %s" % doc)

    claude_md = REPO / "CLAUDE.md"
    if check(claude_md.exists(), "CLAUDE.md exists"):
        ct = read(claude_md)
        check("@AGENTS.md" in ct, "CLAUDE.md imports AGENTS.md",
              "so the shared contract is always in context")
        check(len(ct.splitlines()) < 90, "CLAUDE.md is thin",
              "%d lines" % len(ct.splitlines()))


# --- 2. the canonical brief, and the absence of the removed layer -------------

def check_canonical() -> None:
    p = REPO / SKEPTIC_BRIEF
    if check(p.exists(), "canonical Skeptic brief: %s" % SKEPTIC_BRIEF):
        text = read(p)
        hits = sorted({m.group(0).lower() for m in BANNED.finditer(text)})
        check(not hits, "%s is agent-neutral" % SKEPTIC_BRIEF,
              "found product/model names: %s" % ", ".join(hits) if hits else "")

    # The removal must be COMPLETE. A half-removed layer -- a skills directory on
    # one agent and not the other, or a dangling ai/workflows/ -- would recreate
    # exactly the asymmetry this checker exists to prevent.
    for gone in (REPO / "ai", REPO / ".claude" / "skills", REPO / ".agents"):
        check(not gone.exists(), "removed layer stays removed: %s"
              % gone.relative_to(REPO).as_posix())
    for name in _REMOVED:
        check(not (REPO / "docs" / ("%s.md" % name)).exists(),
              "removed workflow did not reappear in docs/: %s" % name)


# --- 3/4. adapter parity -------------------------------------------------------

def check_adapters() -> None:
    """There is no skills layer any more, so the only adapters are the subagents.

    This function is kept (rather than deleted) so the section numbering in the
    output still matches the report both agents have been reading, and so that a
    future reintroduction of a skills layer has an obvious place to be checked.
    """
    check(True, "no skills layer to compare (removed 2026-08-21)")


# --- 5. subagent parity --------------------------------------------------------

def check_subagents() -> None:
    for label, p in (("claude", REPO / ".claude" / "agents" / "skeptic.md"),
                     ("codex", REPO / ".codex" / "agents" / "skeptic.md")):
        if not check(p.exists(), "%s skeptic agent exists" % label):
            continue
        text = read(p)
        check(SKEPTIC_BRIEF in text,
              "%s skeptic points at canonical brief" % label,
              "expected a reference to %s" % SKEPTIC_BRIEF)
        check(len(text.splitlines()) <= ADAPTER_MAX_LINES,
              "%s skeptic adapter is thin" % label,
              "%d lines" % len(text.splitlines()))


# --- 6. hook parity ------------------------------------------------------------

def check_hooks() -> None:
    shared = REPO / "scripts" / "guard_paths.py"
    check(shared.exists(), "shared enforcement implementation exists")

    for label, p in (("claude", REPO / ".claude" / "hooks" / "guard_hook.py"),
                     ("codex", REPO / ".codex" / "hooks" / "guard_hook.py")):
        if not check(p.exists(), "%s hook adapter exists" % label):
            continue
        text = read(p)
        check("guard_paths" in text, "%s hook adapter uses the shared policy" % label)
        check(len(text.splitlines()) <= 70, "%s hook adapter is thin" % label,
              "%d lines" % len(text.splitlines()))

    for label, p in (("claude", REPO / ".claude" / "hooks" / "guard-protected-files.sh"),
                     ("codex", REPO / ".codex" / "hooks" / "guard-protected-files.sh")):
        check(p.exists(), "%s hook shim exists" % label)

    settings = REPO / ".claude" / "settings.json"
    if check(settings.exists(), "claude settings.json exists"):
        check("guard-protected-files.sh" in read(settings),
              "claude PreToolUse hook is registered")

    check((REPO / ".codex" / "hooks" / "README.md").exists(),
          "codex hook wiring is documented")

    # The agent-independent backstop.
    installed = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "install_git_hooks.py"), "--check"],
        cwd=REPO, capture_output=True, text=True,
    )
    check(installed.returncode == 0, "git pre-commit backstop installed and current",
          installed.stdout.strip().splitlines()[-1] if installed.stdout.strip() else "")


# --- 7. the policy itself behaves the same for everyone ------------------------

CASES = [
    (dict(path="docs/HUMAN_MODEL.md"), DENY, "HUMAN_MODEL write denied"),
    (dict(path="./docs/HUMAN_MODEL.md"), DENY, "HUMAN_MODEL (relative) denied"),
    (dict(path=r"C:\Users\samsu\Documents\arc-whest\docs\HUMAN_MODEL.md"), DENY,
     "HUMAN_MODEL (absolute windows) denied"),
    (dict(path="experiments/vault/manifest.json"), DENY, "vault manifest denied"),
    (dict(path="experiments/vault/touch_log.jsonl"), DENY, "vault touch log denied"),
    (dict(command="echo x > docs/HUMAN_MODEL.md"), DENY, "shell write to HUMAN_MODEL denied"),
    (dict(command="Set-Content -Path docs/HUMAN_MODEL.md -Value x"), DENY,
     "powershell write to HUMAN_MODEL denied"),
    (dict(command="sed -i 's/a/b/' docs/HUMAN_MODEL.md"), DENY, "in-place sed denied"),
    (dict(command="git restore docs/HUMAN_MODEL.md"), DENY, "git restore of HUMAN_MODEL denied"),
    (dict(command="echo y > experiments/vault/manifest.json"), DENY,
     "shell write to vault manifest denied"),
    # The guard must not cry wolf: writing elsewhere while *discussing* a protected
    # file is ordinary work (every LOG entry about the harness does it). A guard
    # that blocks honest work gets disabled, which costs the invariant outright.
    (dict(command="cat >> experiments/LOG.md <<'E'\nnever edit docs/HUMAN_MODEL.md\nE"),
     ALLOW, "LOG entry discussing HUMAN_MODEL allowed"),
    (dict(command="rg HUMAN_MODEL docs/"), ALLOW, "searching for HUMAN_MODEL allowed"),
    (dict(command="cat docs/HUMAN_MODEL.md"), ALLOW, "reading HUMAN_MODEL allowed"),
    (dict(command="uv run whest submit --estimator estimator.py"), ASK, "submit asks"),
    (dict(command="python local_sim/vault.py evaluate --label v"), ASK, "vault touch asks"),
    (dict(path="docs/CURRENT_STATE.md"), ALLOW, "ordinary doc allowed"),
    (dict(path="estimator.py"), ALLOW, "estimator edit allowed (gated by protocol, not hook)"),
    (dict(command="git status --short"), ALLOW, "read-only git allowed"),
    (dict(command="uv run python local_sim/vault.py verify"), ALLOW, "vault verify allowed"),
    (dict(command="python scripts/human_model_state.py"), ALLOW,
     "reading HUMAN_MODEL state allowed"),
]


def check_policy() -> None:
    for kwargs, expected, label in CASES:
        got = decide(**kwargs).verdict
        check(got == expected, "policy: %s" % label, "expected %s, got %s" % (expected, got))

    # Both agent adapters must return the same verdict for the same input.
    for kwargs, expected, label in CASES:
        args = []
        if kwargs.get("path"):
            args += ["--path", kwargs["path"]]
        if kwargs.get("command"):
            args += ["--command", kwargs["command"]]
        out = subprocess.run(
            [sys.executable, str(REPO / ".codex" / "hooks" / "guard_hook.py")] + args,
            cwd=REPO, capture_output=True, text=True,
        )
        want = {ALLOW: 0, DENY: 1, ASK: 2}[expected]
        check(out.returncode == want, "codex adapter agrees: %s" % label,
              "expected exit %d, got %d" % (want, out.returncode))


# --- 8. session-start's branch is mechanical and works -------------------------

def check_human_model_detection() -> None:
    out = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "human_model_state.py"), "--json"],
        cwd=REPO, capture_output=True, text=True,
    )
    if not check(out.returncode == 0, "human_model_state.py runs", out.stderr.strip()):
        return
    import json
    data = json.loads(out.stdout)
    check(data["verdict"] in ("BLANK", "PARTIAL", "STALE", "FILLED"),
          "human_model_state.py returns a valid verdict", data["verdict"])
    check(isinstance(data.get("first_orientation_mode"), bool),
          "the staleness branch is machine-readable",
          "verdict=%s -> first_orientation_mode=%s"
          % (data["verdict"], data["first_orientation_mode"]))
    # There is no longer a session-start workflow to consume this. The check that
    # survives is the one that matters: the verdict comes from a SCRIPT reading
    # git history, not from an agent's impression of how stale the file looks.
    check("scripts/human_model_state.py" in read(REPO / "AGENTS.md"),
          "AGENTS.md points at the mechanical human-model check")


# --- 9. no dangling file references -------------------------------------------

def check_references() -> None:
    scanned = [REPO / SKEPTIC_BRIEF]
    scanned += [REPO / "AGENTS.md", REPO / "CLAUDE.md",
                REPO / ".claude" / "agents" / "skeptic.md",
                REPO / ".codex" / "agents" / "skeptic.md"]
    missing: list[str] = []
    for p in scanned:
        if not p.exists():
            continue
        for m in FILE_REF.finditer(read(p)):
            ref = m.group(1)
            if not (REPO / ref).exists():
                missing.append("%s -> %s" % (p.relative_to(REPO), ref))
    check(not missing, "no dangling file references in the harness layer",
          "; ".join(sorted(set(missing))))


# --- 10. (retired) the workflow chain -----------------------------------------
# This section checked that session-start handed off to next-slice, next-slice to
# review-slice, and so on -- that the human control surface was a loop rather than
# a set of disconnected commands. The commands were removed on 2026-08-21, so
# there is no chain left to close. What replaced them is prose (AGENTS.md Rules 1
# and 2), and prose is not mechanically checkable; pretending otherwise would be
# exactly the "check that reports a state rather than testing a property" this
# repository has lost five checks to.


def check_chain() -> None:
    check(True, "workflow chain retired with the commands (2026-08-21)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--verbose", action="store_true", help="print passing checks too")
    args = ap.parse_args(argv)

    for fn in (check_sources_of_truth, check_canonical, check_adapters,
               check_subagents, check_hooks, check_policy,
               check_human_model_detection, check_references, check_chain):
        fn()

    failed = [r for r in results if not r[0]]
    for ok, name, detail in results:
        if ok and not args.verbose:
            continue
        print("%s  %s%s" % ("ok  " if ok else "FAIL", name,
                            ("  -- " + detail) if detail else ""))

    print("\n%d checks, %d failed" % (len(results), len(failed)))
    if failed:
        print("\nThe harness is NOT agent-portable as it stands. Fix the failures "
              "above before\nswitching drivers -- a broken control surface is "
              "discovered at the worst moment.")
        return 1
    print("PORTABLE -- Claude and Codex load the same Skeptic brief over the same\n"
          "canonical semantics, the same enforcement, and the same durable state.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
