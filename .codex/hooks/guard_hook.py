#!/usr/bin/env python
"""Codex pre-tool adapter around scripts/guard_paths.py.

THIN BY DESIGN.  The policy -- which paths are protected, which commands are
irreversible, and what the reasons say -- lives in scripts/guard_paths.py and is
shared with the Claude adapter and the git pre-commit backstop.  This file only
translates the shared decision into the contract a Codex hook is expected to obey.

Because the exact hook payload and response schema varies across Codex versions,
this adapter is written to satisfy every common contract at once:

  * it accepts a JSON payload on stdin with the tool arguments nested anywhere,
    and also accepts `--path` / `--command` if the caller passes them as flags;
  * it writes a machine-readable decision to stdout;
  * it writes the human-readable reason to stderr, which is what a hook runner
    surfaces when it blocks;
  * it exits 0 to allow and NON-ZERO to block.

An agent that reads the JSON gets the deny/ask distinction.  An agent that only
reads the exit code gets the safe behaviour: both deny and ask block, and the
reason is on stderr for the model to read and act on.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from guard_paths import ALLOW, ASK, decide, decide_payload  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=None)
    ap.add_argument("--command", default=None)
    args, _unknown = ap.parse_known_args()

    if args.path or args.command:
        decision = decide(path=args.path, command=args.command)
    else:
        payload = sys.stdin.read() if not sys.stdin.isatty() else ""
        decision = decide_payload(payload)

    print(json.dumps({
        "decision": decision.verdict,
        "rule": decision.rule,
        "reason": decision.reason,
    }))

    if decision.verdict == ALLOW:
        return 0

    label = "CONFIRMATION REQUIRED" if decision.verdict == ASK else "BLOCKED"
    sys.stderr.write("arc-whest guard: %s [%s]\n%s\n" % (label, decision.rule, decision.reason))
    return 2 if decision.verdict == ASK else 1


if __name__ == "__main__":
    sys.exit(main())
