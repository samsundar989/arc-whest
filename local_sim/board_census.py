"""
BOARD-HYGIENE CENSUS — what is our rank among teams whose numbers can be real?

**PRIVATE. Never the forum, never a public write-up. Names never leave this repo.**

Rank on the public board is not the prize (that is a private re-run), but it is
the only external calibration we have, and it is only useful if we read it
honestly. RESEARCH_NOTES 2g established that the scored suite is fixed, that every
submission returns exact per-MLP MSEs for 50 fixed networks, and that **257
submissions over-determine those references** (D1). A team with that many entries
has the *capacity* to stop estimating and start solving. This script screens for
the signature that capacity leaves behind.

It is a screen, not an accusation, and it is written to be falsifiable:

  reconstruction-capable   entries >= 257            (D1's arithmetic)
  step-change              a >=3x improvement between consecutive personal bests
                           AFTER already being below 1e-6 — real methods improve
                           in compounding small steps once they are in the regime;
                           a late 3x is a different kind of event
  sub-frontier             score < 1e-7, i.e. better than the best *published*
                           method (1.551e-7, thread 18106) by more than 1.5x

A team is classified PROBING-SIGNATURE only when it is sub-frontier AND
(reconstruction-capable OR has a late step-change). Sub-frontier alone is not
enough — someone may simply be better than the published state of the art, which
is exactly what we are trying to be. Capacity alone is not enough either: 900
honest submissions is a work ethic, not a method.

The headline number this produces is **our rank among plausible-honest teams**.

Run:
    uv run python local_sim/board_census.py
    uv run python local_sim/board_census.py --round 1429 --json out.json
"""

import argparse
import glob
import json
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAPS = os.path.join(_ROOT, "experiments", "board_snapshots")

# Participants known to have published a Phase-1 method write-up (RESEARCH_NOTES 3).
# A write-up is evidence FOR a real method: it exposes the mechanism to scrutiny.
WRITEUP_AUTHORS = {
    "evaaaz": "18053 RQMC + Rao-Blackwell (4.10e-7)",
    "pscamillo": "18063 Gaussian-closure scale bias",
    "radiant-allomancer": "18085 variance/bias budget (4.47e-7)",
    "natasha_stewart": "18106 structure-aware hybrid (1.551e-7)",
    "mliston": "18108 wall-time channel fix proposal",
    "Kerensa": "18099 flopscope bypass report",
}
ENTRIES_THRESHOLD = 257      # D1: submissions that over-determine the scored set
STEP_RATIO = 3.0             # a "late step-change"
LATE_BELOW = 1e-6            # ... once already in the real regime
FRONTIER = 1e-7              # best published honest method is 1.551e-7


def latest_snapshot(pattern="*board.json"):
    fs = sorted(glob.glob(os.path.join(SNAPS, pattern)))
    if not fs:
        raise SystemExit("no board snapshots; run board_snapshot.py first")
    return fs[-1]


def biggest_late_jump(history):
    """(ratio, date) of the largest improvement between consecutive personal bests
    that happened after the team was already below LATE_BELOW."""
    best = (1.0, None)
    for a, b in zip(history, history[1:]):
        va, vb = a.get("value"), b.get("value")
        if not va or not vb or vb <= 0:
            continue
        if va > LATE_BELOW:
            continue
        r = va / vb
        if r > best[0]:
            best = (r, b.get("date"))
    return best


def classify(node):
    hist = node.get("scoreHistory") or []
    entries = node.get("entries") or 0
    score = node.get("score")
    jump, jump_date = biggest_late_jump(hist)
    capable = entries >= ENTRIES_THRESHOLD
    stepped = jump >= STEP_RATIO
    sub = score is not None and score < FRONTIER
    probing = sub and (capable or stepped)
    return {
        "rank": node.get("rank"),
        "name": (node.get("participant") or {}).get("name") or "(anon)",
        "score": score,
        "secondary": node.get("scoreSecondary"),
        "entries": entries,
        "late_jump": jump,
        "late_jump_date": jump_date,
        "capable": capable,
        "stepped": stepped,
        "sub_frontier": sub,
        "probing": probing,
        "is_self": bool(node.get("viewerIsSelf")),
        "writeup": WRITEUP_AUTHORS.get((node.get("participant") or {}).get("name")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default="1428")
    ap.add_argument("--snapshot", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("--show", type=int, default=25)
    ap.add_argument("--me", default="neopok919",
                    help="our display name; public snapshots are unauthenticated "
                         "so viewerIsSelf is never set (RESEARCH_NOTES 2g V1)")
    a = ap.parse_args()

    path = a.snapshot or latest_snapshot()
    d = json.load(open(path))
    boards = [b for b in d["boards"] if b.get("challengeRoundId") == a.round]
    if not boards or not boards[0]["entries"]["nodes"]:
        raise SystemExit(f"round {a.round} has no entries in {os.path.basename(path)}")
    nodes = boards[0]["entries"]["nodes"]
    rows = [classify(n) for n in nodes]
    for r in rows:
        if r["name"] == a.me:
            r["is_self"] = True
    rows.sort(key=lambda r: (r["score"] is None, r["score"]))

    us = next((r for r in rows if r["is_self"]), None)
    honest = [r for r in rows if not r["probing"]]
    probing = [r for r in rows if r["probing"]]

    print("=" * 108)
    print(f"BOARD-HYGIENE CENSUS — round {a.round}   snapshot {d['fetched_utc'][:19]}Z")
    print(f"PRIVATE: screen, not accusation. Criteria: entries>={ENTRIES_THRESHOLD}, "
          f"late jump>={STEP_RATIO}x below {LATE_BELOW:g}, sub-frontier<{FRONTIER:g}")
    print("=" * 108)
    print(f"{'rk':>3} {'participant':<24} {'score':>11} {'entries':>8} {'late jump':>10} "
          f"{'cap':>4} {'step':>5} {'sub':>4} {'class':<10} writeup")
    print("-" * 108)
    for r in rows[:a.show]:
        cls = "PROBING?" if r["probing"] else "honest"
        mark = " <== US" if r["is_self"] else ""
        print(f"{r['rank']:>3} {r['name'][:23]:<24} {r['score']:>11.4g} {r['entries']:>8} "
              f"{r['late_jump']:>9.2f}x {str(r['capable'])[0]:>4} {str(r['stepped'])[0]:>5} "
              f"{str(r['sub_frontier'])[0]:>4} {cls:<10} "
              f"{(r['writeup'] or '')[:26]}{mark}")
    if us and us["rank"] > a.show:
        print("  ...")
        print(f"{us['rank']:>3} {us['name'][:23]:<24} {us['score']:>11.4g} {us['entries']:>8} "
              f"{us['late_jump']:>9.2f}x {str(us['capable'])[0]:>4} {str(us['stepped'])[0]:>5} "
              f"{str(us['sub_frontier'])[0]:>4} {'honest':<10} <== US")

    print("-" * 108)
    print(f"  total entries on board          {len(rows)}")
    print(f"  probing-signature               {len(probing)}")
    print(f"  plausible-honest                {len(honest)}")
    if us:
        hrank = sum(1 for r in honest if r["score"] <= us["score"])
        print()
        print(f"  our raw rank                    #{us['rank']} of {len(rows)}")
        print(f"  >> OUR RANK AMONG PLAUSIBLE-HONEST TEAMS:  #{hrank} of {len(honest)}  <<")
        print(f"     our score {us['score']:.4g}   entries {us['entries']}   "
              f"largest late jump {us['late_jump']:.2f}x")
        ahead = [r for r in honest if r["score"] < us["score"]]
        if ahead:
            print(f"\n  plausible-honest teams ahead of us ({len(ahead)}):")
            for r in ahead:
                print(f"    #{r['rank']:>3} {r['name'][:22]:<23} {r['score']:.4g}  "
                      f"entries {r['entries']:>4}  jump {r['late_jump']:.2f}x  "
                      f"{r['writeup'] or ''}")
    else:
        print("  (our own entry not flagged viewerIsSelf in this snapshot)")

    print("\n  ANCHORS")
    print(f"    reference plain-MC (sampling_mse)   6.4695e-07")
    print(f"    best published honest method        1.5510e-07  (18106)")
    print(f"    reconstruction zone                 < {FRONTIER:g}")
    if us:
        print(f"    us                                  {us['score']:.4e}")

    if a.json:
        json.dump({"snapshot": os.path.basename(path), "fetched": d["fetched_utc"],
                   "round": a.round, "rows": rows,
                   "n_probing": len(probing), "n_honest": len(honest),
                   "us": us}, open(a.json, "w"), indent=2)
        print(f"\nsaved {a.json}")


if __name__ == "__main__":
    main()
