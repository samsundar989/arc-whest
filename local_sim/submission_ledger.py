"""The submission slot ledger: 10 per team per UTC day, counted rather than recalled.

AIcrowd allows ten submissions per team per UTC day in Phase 2 (the first-week
amnesty is over). On 2026-09-05 seven of them went to one defect, and the count
of how many were left existed only in a chat transcript -- which is exactly the
kind of state `AGENTS.md` says must live in the repository, because the driving
agent changes and chat history does not transfer.

So the ledger is a file: `experiments/submissions.jsonl`, one JSON object per
line, append-only, newest last. `scripts/guard_paths.py` reads it and puts
today's remaining slots into the confirmation prompt that `whest submit` raises,
so the number is in front of whoever approves, at the moment they approve.

    python local_sim/submission_ledger.py                 # today's usage
    python local_sim/submission_ledger.py --all           # the whole ledger
    python local_sim/submission_ledger.py --record ...    # append one row

Fields (all optional except `submitted_at_utc` and `artifact`):

    submitted_at_utc  ISO-8601 Z. The UTC DAY of this stamp is what the cap counts.
    artifact          the file submitted, repo-relative
    sha256            of the exact bytes submitted
    submission_id     AIcrowd's id, once known
    result            free text: GRADED / FAILED / pending, plus the score
    approved_by       who approved this specific submission
    note              why it was spent

This module never submits anything and never blocks anything. It counts.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

LEDGER_PATH = os.path.join(_ROOT, "experiments", "submissions.jsonl")

#: Phase 2, after the first-week amnesty. Forum post 18197.
DAILY_SLOTS = 10


def _utc_day(stamp: str) -> str:
    return (stamp or "")[:10]


def today_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")


def load(path: str = LEDGER_PATH) -> List[Dict[str, Any]]:
    """Every recorded submission, oldest first. A missing ledger is empty."""
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    rows.sort(key=lambda r: str(r.get("submitted_at_utc", "")))
    return rows


def used_on(day: Optional[str] = None, path: str = LEDGER_PATH) -> int:
    day = day or today_utc()
    return sum(1 for r in load(path)
               if _utc_day(str(r.get("submitted_at_utc", ""))) == day)


def remaining(day: Optional[str] = None, path: str = LEDGER_PATH) -> int:
    return max(0, DAILY_SLOTS - used_on(day, path))


def status_line(day: Optional[str] = None, path: str = LEDGER_PATH) -> str:
    """One line for a human about to approve a submission.

    Deliberately says when the ledger is EMPTY rather than reporting ten free
    slots as a fact: an unrecorded submission is indistinguishable here from one
    that never happened, and the prompt must not launder that into confidence.
    """
    day = day or today_utc()
    rows = load(path)
    if not rows:
        return (f"SLOT LEDGER: {LEDGER_PATH} is empty or missing, so slot usage "
                f"for {day} is UNKNOWN, not zero. The cap is "
                f"{DAILY_SLOTS}/UTC-day.")
    u = used_on(day, path)
    todays = [r for r in rows if _utc_day(str(r.get("submitted_at_utc"))) == day]
    ids = ", ".join(str(r.get("submission_id") or r.get("artifact") or "?")
                    for r in todays) or "none"
    return (f"SLOT LEDGER: {u}/{DAILY_SLOTS} used on {day} (UTC), "
            f"{DAILY_SLOTS - u} left. Today: {ids}.")


def record(*, artifact: str, sha256: str = "", submission_id: str = "",
           result: str = "pending", approved_by: str = "", note: str = "",
           submitted_at_utc: str = "", path: str = LEDGER_PATH) -> Dict[str, Any]:
    """Append one row. Never rewrites; the ledger is append-only."""
    row = {
        "submitted_at_utc": submitted_at_utc or _dt.datetime.now(
            _dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "artifact": artifact,
        "sha256": sha256,
        "submission_id": submission_id,
        "result": result,
        "approved_by": approved_by,
        "note": note,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(row, sort_keys=False) + "\n")
    return row


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="print the whole ledger")
    ap.add_argument("--day", default="", help="UTC day, default today")
    ap.add_argument("--record", action="store_true", help="append one row")
    ap.add_argument("--artifact", default="")
    ap.add_argument("--sha256", default="")
    ap.add_argument("--submission-id", default="")
    ap.add_argument("--result", default="pending")
    ap.add_argument("--approved-by", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--at", default="", help="ISO-8601 Z stamp (for backfill)")
    a = ap.parse_args(argv)

    if a.record:
        if not a.artifact:
            ap.error("--record needs --artifact")
        row = record(artifact=a.artifact, sha256=a.sha256,
                     submission_id=a.submission_id, result=a.result,
                     approved_by=a.approved_by, note=a.note,
                     submitted_at_utc=a.at)
        print("recorded:", json.dumps(row))

    day = a.day or today_utc()
    if a.all:
        for r in load():
            print(f"  {r.get('submitted_at_utc','?'):<21} "
                  f"{str(r.get('submission_id') or '-'):>8}  "
                  f"{str(r.get('artifact') or '-'):<34} {r.get('result','')}")
        print()
    print(status_line(day))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
