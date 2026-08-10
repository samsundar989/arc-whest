#!/usr/bin/env python
"""Agent-independent backstop: no protected file changes quietly.

The per-agent hooks (.claude/hooks/, .codex/hooks/) stop a protected write at the
moment it is attempted.  They are the fast path, but each depends on that agent's
hook system being wired, enabled, and schema-compatible.  This one does not depend
on any agent at all: it runs inside git, so it holds for a future agent nobody has
configured yet, for a hook that silently stopped firing, and for a plain text
editor.

It does not FORBID -- Samuel must be able to commit his own HUMAN_MODEL.md, and a
legitimate `vault.py evaluate` really does move the seal files.  It forbids
committing them *quietly*:

    ARC_WHEST_PROTECTED_OK=1 git commit ...

is the conscious override.  An agent that sets that variable to get past this hook
has stopped being a collaborator, and that is now visible in the shell history.

Additionally, when the vault seal is staged, the touch counter must still equal the
touch-log line count -- the same assertion as scripts/preflight.ps1 check 5, applied
at the moment the change would become permanent.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from guard_paths import check_path  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OVERRIDE = "ARC_WHEST_PROTECTED_OK"


def staged_files() -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMRT"],
        cwd=REPO, capture_output=True, text=True,
    )
    if out.returncode != 0:
        return []
    return [l.strip() for l in out.stdout.splitlines() if l.strip()]


def vault_seal_consistent() -> tuple[bool, str]:
    man = REPO / "experiments" / "vault" / "manifest.json"
    log = REPO / "experiments" / "vault" / "touch_log.jsonl"
    if not man.exists():
        return True, "no vault manifest"
    try:
        m = json.loads(man.read_text(encoding="utf-8"))
    except ValueError as exc:
        return False, "manifest.json is not valid JSON (%s)" % exc
    counter = m.get("touches", m.get("touch_count"))
    lines = [l for l in log.read_text(encoding="utf-8").splitlines() if l.strip()] \
        if log.exists() else []
    if counter is None:
        return False, "manifest has no touch counter"
    if counter != len(lines):
        return False, "touch counter %s != %d log entries" % (counter, len(lines))
    return True, "counter %s == %d log entries" % (counter, len(lines))


def main() -> int:
    hits = [(f, d) for f, d in ((f, check_path(f)) for f in staged_files()) if d]
    if not hits:
        return 0

    rules = {d.rule for _, d in hits}
    override = os.environ.get(OVERRIDE, "").strip() not in ("", "0", "false", "False")

    sys.stderr.write("\narc-whest pre-commit guard -- protected paths are staged:\n\n")
    for f, d in hits:
        sys.stderr.write("  [%s] %s\n" % (d.rule, f))
    sys.stderr.write("\n")
    for rule in sorted(rules):
        reason = next(d.reason for _, d in hits if d.rule == rule)
        sys.stderr.write("%s\n\n" % reason)

    if "vault-seal" in rules:
        ok, detail = vault_seal_consistent()
        sys.stderr.write("  vault seal check: %s -- %s\n\n" % ("OK" if ok else "FAILED", detail))
        if not ok:
            sys.stderr.write(
                "  REFUSING even with the override: the seal is internally "
                "inconsistent, which is the exact condition it exists to detect.\n"
                "  Run: .venv-gpu\\Scripts\\python.exe local_sim/vault.py verify\n\n"
            )
            return 1

    if override:
        sys.stderr.write("  %s is set -- allowing this commit.\n\n" % OVERRIDE)
        return 0

    sys.stderr.write(
        "  Commit refused. If this is Samuel deliberately committing his own\n"
        "  HUMAN_MODEL.md, or a legitimate logged vault touch, re-run with:\n\n"
        "      %s=1 git commit ...\n\n"
        "  If an agent produced this change, revert the file instead.\n\n" % OVERRIDE
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
