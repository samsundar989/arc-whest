#!/usr/bin/env python
"""Claude Code PreToolUse adapter around scripts/guard_paths.py.

THIN BY DESIGN.  The policy -- which paths are protected, which commands are
irreversible, and what the reasons say -- lives in scripts/guard_paths.py and is
shared with the Codex adapter and the git pre-commit backstop.  This file only
translates the shared decision into Claude Code's hook JSON.

Reads the hook payload on stdin; always exits 0, because Claude Code takes the
verdict from the JSON body rather than from the exit code.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from guard_paths import ALLOW, decide_payload  # noqa: E402


def main() -> int:
    decision = decide_payload(sys.stdin.read())
    if decision.verdict == ALLOW:
        return 0
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": decision.verdict,      # "deny" | "ask"
            "permissionDecisionReason": decision.reason,
        }
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
