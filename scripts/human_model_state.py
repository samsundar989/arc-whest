#!/usr/bin/env python
"""Is docs/HUMAN_MODEL.md blank, partial, stale, or filled?

`session-start` branches into FIRST-ORIENTATION MODE on this answer.  That branch
must not depend on an agent eyeballing the file, because two agents will eyeball it
differently and the human workflow is supposed to be identical under both.  So the
test is mechanical and lives here.

    python scripts/human_model_state.py            human-readable
    python scripts/human_model_state.py --json     machine-readable

Verdicts
    BLANK    no answer slot has been filled
    PARTIAL  some slots filled, but fewer than the FILLED threshold
    STALE    filled, but docs/CURRENT_STATE.md has moved since it was last touched,
             so at least one stated belief may have been superseded.  This is a
             prompt to *check*, not a proof: confirm by reading both files.
    FILLED   filled and no newer CURRENT_STATE commits

Exit code is 0 for every verdict (this is a report, not a gate) and 1 only if the
file is missing.  No third-party imports; `git` is optional.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HUMAN_MODEL = REPO / "docs" / "HUMAN_MODEL.md"
CURRENT_STATE = REPO / "docs" / "CURRENT_STATE.md"

# Answers live either in a blockquote slot (`>` with text after it) or as items of
# a numbered list. Empty template slots are a bare `>` or a bare `1.`.
ANSWER_QUOTE = re.compile(r"^>\s*\S")
ANSWER_ITEM = re.compile(r"^\d+\.\s+\S")
PROMPT_LINE = re.compile(r"^\*.*\*\s*$")          # the italic question under a heading
SECTION = re.compile(r"^##\s+(?!#)(.*)$")
COLD_START_HEADING = "cold-start check"

# The file has 10 answerable sections. Below this many, the human model is not yet
# usable as a comprehension baseline and orientation is the right move.
FILLED_THRESHOLD = 7


def parse(text: str) -> tuple[list[str], list[str]]:
    """Return (filled section titles, empty section titles)."""
    filled: list[str] = []
    empty: list[str] = []
    current: str | None = None
    has_answer = False

    def flush() -> None:
        if current is None:
            return
        (filled if has_answer else empty).append(current)

    for raw in text.splitlines():
        line = raw.strip()
        m = SECTION.match(line)
        if m:
            title = m.group(1).strip()
            if COLD_START_HEADING in title.lower():
                flush()
                return filled, empty          # the quiz below is not an answer slot
            flush()
            current, has_answer = title, False
            continue
        if current is None or PROMPT_LINE.match(line):
            continue
        if ANSWER_QUOTE.match(line) or ANSWER_ITEM.match(line):
            has_answer = True

    flush()
    return filled, empty


def newer_current_state_commits() -> list[str]:
    """CURRENT_STATE commits since HUMAN_MODEL was last committed. [] if unknown."""
    def run(args: list[str]) -> str | None:
        try:
            out = subprocess.run(
                ["git"] + args, cwd=REPO, capture_output=True, text=True, timeout=15
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    last_hm = run(["log", "-1", "--format=%H", "--", "docs/HUMAN_MODEL.md"])
    if not last_hm:
        return []
    since = run(
        ["log", "--oneline", "%s..HEAD" % last_hm, "--", "docs/CURRENT_STATE.md"]
    )
    return [l for l in (since or "").splitlines() if l.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if not HUMAN_MODEL.exists():
        sys.stderr.write("MISSING: %s\n" % HUMAN_MODEL)
        return 1

    filled, empty = parse(HUMAN_MODEL.read_text(encoding="utf-8", errors="replace"))
    total = len(filled) + len(empty)
    drift = newer_current_state_commits()

    if not filled:
        verdict, why = "BLANK", "no answer slot has been filled"
    elif len(filled) < FILLED_THRESHOLD:
        verdict, why = "PARTIAL", "%d of %d slots filled (threshold %d)" % (
            len(filled), total, FILLED_THRESHOLD)
    elif drift:
        verdict, why = "STALE", (
            "%d slots filled, but docs/CURRENT_STATE.md changed in %d commit(s) "
            "since docs/HUMAN_MODEL.md was last committed" % (len(filled), len(drift))
        )
    else:
        verdict, why = "FILLED", "%d of %d slots filled" % (len(filled), total)

    payload = {
        "verdict": verdict,
        "reason": why,
        "filled": filled,
        "empty": empty,
        "current_state_commits_since": drift,
        "first_orientation_mode": verdict in ("BLANK", "PARTIAL", "STALE"),
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print("HUMAN_MODEL state: %s" % verdict)
    print("  %s" % why)
    if empty:
        print("  unfilled slots: %s" % ", ".join(empty))
    if drift:
        print("  CURRENT_STATE has moved since:")
        for line in drift:
            print("    %s" % line)
        print("  -> confirm staleness by reading both files; this is a prompt, not a proof.")
    print(
        "  session-start action: %s"
        % ("FIRST-ORIENTATION MODE" if payload["first_orientation_mode"]
           else "control panel")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
