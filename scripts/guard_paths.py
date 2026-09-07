#!/usr/bin/env python
"""Shared protected-path / irreversible-action guard for arc-whest.

THIS FILE IS THE ONLY IMPLEMENTATION OF THESE INVARIANTS.  Every agent hook is a
thin adapter around it:

    .claude/hooks/guard-protected-files.sh   Claude Code PreToolUse adapter
    .codex/hooks/guard-protected-files.sh    Codex pre-tool adapter
    scripts/git-hooks/pre-commit             agent-independent backstop

The invariants, and why each is mechanical rather than prose:

  docs/HUMAN_MODEL.md
      Samuel's own beliefs.  An agent that "corrects" this file destroys the only
      record of whether the project's model of itself is any good.  Agents may
      read it, quiz against it, and propose changes in chat -- never write it.

  experiments/vault/manifest.json
  experiments/vault/touch_log.jsonl
      The seal on the 500-network holdout.  Its whole value is that there is no
      quiet way to run against it: the counter and the log must only ever be moved
      together, by local_sim/vault.py.

  `whest submit`
      Outward-facing, consumes a daily slot, and a failed MLP scores as zeros.
      Never without Samuel's explicit approval.

  `vault.py evaluate`
      Spends one of three permitted holdout evaluations and increments a counter
      that cannot be decremented.

Interfaces
----------
    python scripts/guard_paths.py --stdin-payload [--format json|text]
        Read an agent hook payload (JSON) on stdin, extract file paths and shell
        commands from any of the common field names, and decide.

    python scripts/guard_paths.py --path P [--command C]
        Decide directly.  Either argument may be omitted.

    python scripts/guard_paths.py --check-paths P [P ...]
        Batch path check, for the git pre-commit backstop.

    from guard_paths import decide          # -> Decision(verdict, reason, rule)

Exit codes (adapters map these onto their agent's contract):
    0   allow
    1   deny   -- the action must not happen
    2   ask    -- the action is legitimate but irreversible; require confirmation

An agent whose hook contract only understands zero / non-zero gets the safe
behaviour: both deny and ask block.

No third-party imports.  Runs under any Python 3.9+, including a bare system
interpreter with no virtualenv active, because hooks must not depend on `uv`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional

ALLOW, DENY, ASK = "allow", "deny", "ask"
EXIT_CODE = {ALLOW: 0, DENY: 1, ASK: 2}

# --- the protected set --------------------------------------------------------
# Matched on the normalised tail of a path, so an absolute Windows path, a
# repo-relative POSIX path and a `./`-prefixed path all hit the same rule.

HUMAN_MODEL_REASON = (
    "docs/HUMAN_MODEL.md is Samuel-owned and write-protected by the project "
    "harness, under every agent. Agents may read it, quiz against it, and PROPOSE "
    "corrections in chat -- but must not edit it. If a belief in it is contradicted "
    "by evidence, say so and cite the evidence; let Samuel make the edit."
)

VAULT_SEAL_REASON = (
    "The vault seal is write-protected. manifest.json and touch_log.jsonl may only "
    "be modified by local_sim/vault.py, which increments the touch counter and "
    "appends a log line together. Hand-editing either one breaks the guarantee that "
    "the holdout baseline was not chosen by re-rolling. Permitted touches: "
    "baseline, one per estimator change, designation day."
)

PROTECTED_PATHS = [
    # (rule name, tail pattern, reason)
    ("human-model", re.compile(r"(^|[\\/])docs[\\/]human_model\.md$"), HUMAN_MODEL_REASON),
    ("human-model", re.compile(r"^human_model\.md$"), HUMAN_MODEL_REASON),
    ("vault-seal", re.compile(r"(^|[\\/])vault[\\/]manifest\.json$"), VAULT_SEAL_REASON),
    ("vault-seal", re.compile(r"(^|[\\/])vault[\\/]touch_log\.jsonl$"), VAULT_SEAL_REASON),
]

# Bare filenames that, if they appear anywhere in a shell command together with a
# write-ish verb, mean that command is about to rewrite a protected file.
# `manifest.json` alone is far too common a filename to match on, so the vault one
# is matched path-qualified; the other two names are distinctive enough bare.
PROTECTED_TOKENS = [
    ("human-model", re.compile(r"human_model\.md", re.IGNORECASE), HUMAN_MODEL_REASON),
    ("vault-seal", re.compile(r"vault[\\/]manifest\.json", re.IGNORECASE), VAULT_SEAL_REASON),
    ("vault-seal", re.compile(r"touch_log\.jsonl", re.IGNORECASE), VAULT_SEAL_REASON),
]

# Write indicators in a shell command.  Broad, but every entry has to be a verb
# that actually mutates a file: a bare `del` or `apply` also appears in ordinary
# Python and prose, and a guard that cries wolf gets worked around.
WRITE_INDICATORS = re.compile(
    r"(>>?|\btee\b|\bSet-Content\b|\bAdd-Content\b|\bOut-File\b|\bNew-Item\b"
    r"|\bsed\s+-i|\bperl\s+-i|\btruncate\b|\bcp\s|\bmv\s|\bCopy-Item\b"
    r"|\bMove-Item\b|\brm\s|\bRemove-Item\b|\bClear-Content\b"
    r"|\bgit\s+checkout\s+--|\bgit\s+restore\b|\bgit\s+apply\b)",
    re.IGNORECASE,
)

# A protected name must be the write verb's TARGET, not merely present somewhere in
# a long command.  "mentions it anywhere" was the first cut and it denied
# `cat >> experiments/LOG.md <<EOF ... HUMAN_MODEL.md ...` -- a LOG entry that
# legitimately discusses the file.  A guard that blocks honest work gets disabled,
# so the target is resolved by tokens instead: after the verb, skip flags and take
# the next few arguments.  Covers `> P`, `Set-Content -Path P`, `cp a P`,
# `Remove-Item -Force -Path P`, `sed -i 's/x/y/' P`.
# A redirection takes exactly one target; a verb may be preceded by flags.
REDIRECTION = re.compile(r"^(>>?|tee)$", re.IGNORECASE)
VERB_TARGET_TOKENS = 4


def _write_targets(command: str) -> list[str]:
    targets: list[str] = []
    for m in WRITE_INDICATORS.finditer(command):
        # Stop at a statement boundary or a comment, so the next command's
        # arguments -- or a comment that merely names the file -- are not swept in.
        tail = re.split(r"[;\n|&#]", command[m.end():], maxsplit=1)[0]
        n = 1 if REDIRECTION.match(m.group(0).strip()) else VERB_TARGET_TOKENS
        targets.extend(tail.split()[:n])
    return targets

# --- irreversible commands ----------------------------------------------------

SUBMIT_REASON = (
    "SUBMISSION. This is outward-facing and cannot be undone: it consumes a daily "
    "submission slot and a failed MLP scores as zeros. Project rule: never submit "
    "without Samuel's explicit approval, and only after `whest validate` plus a "
    "clean mini run with n_failed_mlps=0. Confirm the file, the version, and that "
    "Samuel has approved this specific submission."
)

VAULT_TOUCH_REASON = (
    "VAULT TOUCH. This spends one of three permitted evaluations of the "
    "500-network sealed holdout and increments a counter that cannot be "
    "decremented. Permitted reasons: the baseline, one per estimator change, "
    "designation day. If this is to browse or to re-check a number already "
    "measured, cancel."
)

def _slot_ledger_line() -> str:
    """Today's slot usage, read from `experiments/submissions.jsonl`.

    Ten submissions per team per UTC day. On 2026-09-05 seven of them went to
    one defect and the running count lived only in a chat transcript -- state
    that `AGENTS.md` says must live in the repository, because the driving agent
    changes and chat history does not transfer. Putting it in the confirmation
    prompt puts it in front of whoever approves, at the moment they approve.

    Never raises and never blocks: a guard that can fail closed on a missing
    optional file is a worse guard than one that says it does not know.
    """
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        local_sim = os.path.join(os.path.dirname(here), "local_sim")
        if local_sim not in sys.path:
            sys.path.append(local_sim)
        import submission_ledger  # noqa: PLC0415
        return "  " + submission_ledger.status_line()
    except Exception as exc:  # pragma: no cover - advisory only
        return (f"  SLOT LEDGER unavailable ({type(exc).__name__}); slot usage "
                f"is UNKNOWN, not zero.")


CONFIRM_COMMANDS = [
    ("submit", re.compile(r"whest\s+submit", re.IGNORECASE), SUBMIT_REASON),
    ("vault-touch", re.compile(r"vault\.py\s+evaluate", re.IGNORECASE), VAULT_TOUCH_REASON),
]


@dataclass
class Decision:
    verdict: str          # allow | deny | ask
    reason: str = ""
    rule: str = ""

    @property
    def exit_code(self) -> int:
        return EXIT_CODE[self.verdict]

    def as_json(self) -> str:
        return json.dumps(
            {"decision": self.verdict, "rule": self.rule, "reason": self.reason}
        )


def _normalise(path: str) -> str:
    return path.strip().strip('"').strip("'").replace("\\", "/").lstrip("./").lower()


def check_path(path: str) -> Optional[Decision]:
    """Deny if `path` names a file no agent may write."""
    if not path:
        return None
    norm = _normalise(path)
    for rule, pattern, reason in PROTECTED_PATHS:
        if pattern.search(norm):
            return Decision(DENY, reason, rule)
    return None


def check_command(command: str) -> Optional[Decision]:
    """Deny writes to protected files via the shell; ask before irreversible acts."""
    if not command:
        return None

    # A shell command that rewrites a protected file is the same invariant as a
    # direct Edit, and the previous prose-plus-file_path guard did not catch it.
    targets = _write_targets(command)
    if targets:
        joined = " ".join(targets)
        for rule, token, reason in PROTECTED_TOKENS:
            if token.search(joined):
                return Decision(
                    DENY,
                    reason + " (Blocked at the shell: this command writes to a "
                    "protected path.)",
                    rule,
                )

    for rule, pattern, reason in CONFIRM_COMMANDS:
        if pattern.search(command):
            if rule == "submit":
                reason = reason + chr(10) + _slot_ledger_line()
            return Decision(ASK, reason, rule)
    return None


def decide(path: Optional[str] = None, command: Optional[str] = None) -> Decision:
    """The whole policy. Deny outranks ask; ask outranks allow."""
    results = [r for r in (check_path(path or ""), check_command(command or "")) if r]
    for verdict in (DENY, ASK):
        for r in results:
            if r.verdict == verdict:
                return r
    return Decision(ALLOW)


# --- payload extraction -------------------------------------------------------
# Tolerant by design: different agents nest the tool arguments differently and the
# guard must not silently pass because it did not recognise a field name.

PATH_KEYS = {"file_path", "filepath", "path", "file", "target_file", "notebook_path"}
COMMAND_KEYS = {"command", "cmd", "shell_command", "script"}


def _walk(obj, paths: List[str], commands: List[str]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            k = key.lower()
            if isinstance(value, str):
                if k in PATH_KEYS:
                    paths.append(value)
                elif k in COMMAND_KEYS:
                    commands.append(value)
            _walk(value, paths, commands)
    elif isinstance(obj, list):
        for item in obj:
            _walk(item, paths, commands)


def extract(payload_text: str) -> tuple[List[str], List[str]]:
    paths: List[str] = []
    commands: List[str] = []
    try:
        _walk(json.loads(payload_text), paths, commands)
    except (ValueError, TypeError):
        # Not JSON, or truncated. Fall back to a regex sweep so a malformed
        # payload cannot become a silent allow.
        for key in PATH_KEYS:
            paths += re.findall(r'"%s"\s*:\s*"([^"]*)"' % key, payload_text)
        for key in COMMAND_KEYS:
            commands += re.findall(r'"%s"\s*:\s*"(.*?)"\s*[,}]' % key, payload_text, re.S)
    return paths, commands


def decide_payload(payload_text: str) -> Decision:
    paths, commands = extract(payload_text)
    results = [decide(path=p) for p in paths] + [decide(command=c) for c in commands]
    for verdict in (DENY, ASK):
        for r in results:
            if r.verdict == verdict:
                return r
    return Decision(ALLOW)


def check_many(paths: Iterable[str]) -> List[Decision]:
    out = []
    for p in paths:
        d = check_path(p)
        if d:
            out.append(Decision(d.verdict, "%s -- %s" % (p, d.reason), d.rule))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stdin-payload", action="store_true",
                    help="read an agent hook payload as JSON on stdin")
    ap.add_argument("--path", default=None)
    ap.add_argument("--command", default=None)
    ap.add_argument("--check-paths", nargs="*", default=None,
                    help="batch path check (git pre-commit backstop)")
    ap.add_argument("--format", choices=("json", "text"), default="json")
    args = ap.parse_args(argv)

    if args.check_paths is not None:
        hits = check_many(args.check_paths)
        if not hits:
            return 0
        for h in hits:
            sys.stderr.write("BLOCKED [%s] %s\n\n" % (h.rule, h.reason))
        return 1

    if args.stdin_payload:
        decision = decide_payload(sys.stdin.read())
    else:
        decision = decide(path=args.path, command=args.command)

    if args.format == "json":
        sys.stdout.write(decision.as_json() + "\n")
    else:
        sys.stdout.write("%s\n" % decision.verdict.upper())
        if decision.reason:
            sys.stdout.write(decision.reason + "\n")
    return decision.exit_code


if __name__ == "__main__":
    sys.exit(main())
