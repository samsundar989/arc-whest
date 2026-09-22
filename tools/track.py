"""Track a submission's grading state WITHOUT submitting anything.

    python tools/track.py 331816 [--wait] [--interval 120] [--ledger]

**Why this exists.** `whest` has no status command, and when `whest submit
--watch` hits its `--watch-timeout` it prints "Next: track grading with `whest
submit --watch`". Following that tip re-runs *submit*, which uploads the
artifact again, creates a SECOND submission and consumes another of the ten
daily slots. This does a plain authenticated GET on the submission instead
(`AIcrowdClient.get_submission_status`, the same call the `--watch` loop uses),
so it is read-only and can be run as often as you like.

`--ledger` updates the pending row for that submission id in
`experiments/submissions.jsonl` once grading reaches a terminal state, so the
durable record does not depend on anyone watching the terminal.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "experiments" / "submissions.jsonl"

TERMINAL = {"graded", "failed"}

#: The fields worth printing and recording. The serializer returns a lot more.
FIELDS = ("id", "grading_status_cd", "score", "score_secondary",
          "grading_message", "created_at", "updated_at")


def client():
    from whestbench.aicrowd_client import AIcrowdClient
    from whestbench.aicrowd_config import resolve_api_key
    return AIcrowdClient(api_key=resolve_api_key(None))


def status(cl, sub_id: int) -> dict:
    d = cl.get_submission_status(int(sub_id))
    return {k: d.get(k) for k in FIELDS} | {"_raw_keys": sorted(d.keys())}


def update_ledger(sub_id: int, st: dict) -> bool:
    """Rewrite the pending row for `sub_id` with the terminal result.

    Returns whether a row was found and changed. Only ever touches the row whose
    submission_id matches, and only its `result`/`standing` fields.
    """
    if not LEDGER.exists():
        return False
    lines = [l for l in LEDGER.read_text(encoding="utf-8").splitlines() if l.strip()]
    changed = False
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except Exception:
            continue
        if str(row.get("submission_id")) != str(sub_id):
            continue
        row["result"] = (f"{st['grading_status_cd'].upper()} "
                         f"score={st['score']} secondary={st['score_secondary']}")
        row["standing"] = ("graded clean" if st["grading_status_cd"] == "graded"
                           else "FAILED on the server")
        row["graded_at_utc"] = st.get("updated_at")
        row["grading_message"] = (st.get("grading_message") or "")[:500]
        lines[i] = json.dumps(row)
        changed = True
    if changed:
        LEDGER.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return changed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("submission_id", type=int)
    ap.add_argument("--wait", action="store_true",
                    help="poll until a terminal state instead of printing once")
    ap.add_argument("--interval", type=int, default=120, help="seconds between polls")
    ap.add_argument("--max-hours", type=float, default=6.0)
    ap.add_argument("--ledger", action="store_true",
                    help="write the terminal result into experiments/submissions.jsonl")
    a = ap.parse_args(argv)

    cl = client()
    deadline = time.time() + a.max_hours * 3600
    while True:
        st = status(cl, a.submission_id)
        stamp = time.strftime("%H:%M:%S")
        print(f"{stamp}  {a.submission_id}  {st['grading_status_cd']}  "
              f"score={st['score']}  secondary={st['score_secondary']}", flush=True)
        if str(st["grading_status_cd"]) in TERMINAL:
            print(json.dumps({k: st[k] for k in FIELDS}, indent=1, default=str))
            if a.ledger:
                print("ledger updated" if update_ledger(a.submission_id, st)
                      else "no matching ledger row")
            return 0 if st["grading_status_cd"] == "graded" else 2
        if not a.wait:
            return 1
        if time.time() > deadline:
            print(f"still {st['grading_status_cd']} after {a.max_hours} h — giving up "
                  f"on the poll, not on the submission")
            return 3
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
