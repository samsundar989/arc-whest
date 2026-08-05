"""
Forensic read of an archived leaderboard snapshot. PRIVATE — nothing here is for
the forum.

Three questions, in order of how much they constrain the others:

1. **What are the two score columns?** Verified against our own graded report
   rather than assumed: if `score` == our adjusted and `scoreSecondary` == our
   final-layer MSE, then `score/scoreSecondary` is the multiplier max(0.1, C/B)
   each entry actually ran at, and every downstream inference is on solid ground.

2. **What per-sample physics does each entry imply?** Above the multiplier floor
   the adjusted score is `V*c/B` — per-sample variance times per-sample cost,
   with the sample count cancelling. So `V*c = adjusted * B` is an
   implementation-independent invariant we can compare across teams and against
   our own measured value. At the floor the identity becomes an inequality
   (adjusted = 0.1*MSE >= 0.1*V*c_min/C_used), reported separately.

3. **When did each entry arrive?** `scoreHistory` is a dated series per team, so
   a board-collapse timeline is reconstructable without any private data.

Run:
    uv run python local_sim/board_forensics.py <snapshot.json> [--round 1428]
"""

import argparse
import datetime as dt
import glob
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

B = 2.72e11
# our own graded facts, for the column check and the physics reference
OURS = {
    "name": "neopok919",  # our AIcrowd display name on the board
    "adjusted": 2.7194e-7,
    "mse": 1.3748e-6,
    "multiplier": 0.19898,
    "cost_per_sample": 3.13e6,  # v2.4, pruned, measured
    "V": 1.6079e-6 * 16384,  # per-sample variance from the mini split
}
# strongest PUBLISHED phase-1 method (thread 18106, structure-aware hybrid)
PUBLISHED_BEST_ADJ = 1.551e-7


def _name(nd):
    for key in ("participant", "team", "submitter"):
        v = nd.get(key)
        if isinstance(v, dict) and v.get("name"):
            return str(v["name"])
        if isinstance(v, str) and v:
            return v
    return "?"


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load(path):
    if path is None:
        cands = sorted(glob.glob(os.path.join(_ROOT, "experiments", "board_snapshots", "*_board.json")))
        if not cands:
            raise SystemExit("no snapshots in experiments/board_snapshots/")
        path = cands[-1]
    with open(path, encoding="utf-8") as fh:
        return path, json.load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot", nargs="?")
    ap.add_argument("--round", type=int, default=1428)
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    path, snap = load(args.snapshot)
    board = next((b for b in snap["boards"] if int(b.get("challengeRoundId") or -1) == args.round), None)
    if board is None:
        raise SystemExit(f"round {args.round} not in {path}")
    nodes = ((board.get("entries") or {}).get("nodes")) or []
    print(f"snapshot {os.path.basename(path)}  fetched {snap['fetched_utc']}")
    print(f"round {args.round} '{board.get('name')}': {len(nodes)} entries")
    print(f"column titles: score='{board.get('scoreTitle')}'  secondary='{board.get('scoreSecondaryTitle')}'\n")

    # ---- 1. column identity, checked against our own graded report ----------
    us = next((nd for nd in nodes if OURS["name"] in json.dumps(nd, default=str).lower() or nd.get("viewerIsSelf")), None)
    print("=" * 78)
    print("1. WHAT THE COLUMNS ARE (checked against our own graded report)")
    print("=" * 78)
    if us is None:
        print("  our row not found in this board — column identity UNVERIFIED")
    else:
        s, s2 = _f(us.get("score")), _f(us.get("scoreSecondary"))
        print(f"  our row: rank {us.get('rank')}  score={s}  scoreSecondary={s2}  entries={us.get('entries')}")
        print(f"  graded : adjusted={OURS['adjusted']}  final_layer_mse={OURS['mse']}  multiplier={OURS['multiplier']}")
        ok_s = s is not None and abs(s - OURS["adjusted"]) / OURS["adjusted"] < 0.02
        ok_2 = s2 is not None and abs(s2 - OURS["mse"]) / OURS["mse"] < 0.02
        print(f"  => score IS the adjusted score:      {ok_s}")
        print(f"  => scoreSecondary IS final-layer MSE: {ok_2}")
        if ok_s and ok_2:
            print(f"  => score/scoreSecondary = multiplier max(0.1, C/B). Ours: {s / s2:.5f} vs graded {OURS['multiplier']:.5f}")

    # ---- 2. implied physics --------------------------------------------------
    print("\n" + "=" * 78)
    print("2. IMPLIED PER-SAMPLE PHYSICS   (V*c = adjusted * B, invariant in N)")
    print("=" * 78)
    ours_vc = OURS["adjusted"] * B
    pub_vc = PUBLISHED_BEST_ADJ * B
    print(f"  ours (v2.4, measured)                V*c = {ours_vc:.3e}   (V={OURS['V']:.3e}, c={OURS['cost_per_sample']:.2e})")
    print(f"  best PUBLISHED method (18106)        V*c = {pub_vc:.3e}")
    print(f"  flag threshold: >3x below the best published => V*c < {pub_vc / 3:.3e}\n")
    print(f"  {'rank':>4} {'team':<26} {'adjusted':>11} {'MSE':>11} {'mult':>7} {'entries':>8} {'V*c':>10} {'vs pub':>8}")
    print("  " + "-" * 92)
    flagged = []
    for nd in nodes[: args.top]:
        s, s2 = _f(nd.get("score")), _f(nd.get("scoreSecondary"))
        if s is None:
            continue
        vc = s * B
        mult = (s / s2) if s2 else float("nan")
        ratio = pub_vc / vc if vc else float("inf")
        mark = ""
        if ratio > 3:
            mark = "  <== FLAG"
            flagged.append((nd, vc, ratio))
        print(
            f"  {nd.get('rank'):>4} {_name(nd)[:25]:<26} {s:>11.4g} {s2 if s2 else float('nan'):>11.4g} "
            f"{mult:>7.3f} {nd.get('entries', '?'):>8} {vc:>10.3g} {ratio:>7.1f}x{mark}"
        )

    # ---- 2b. what the multiplier implies about sample counts ----------------
    print("\n  Sample-count reading of the leaders (assumes a sampling estimator at our")
    print("  per-sample cost; C/B = mult above the floor):")
    print(f"  {'rank':>4} {'team':<26} {'C/B':>7} {'N implied':>12} {'V implied':>11} {'vs our V':>9}")
    print("  " + "-" * 74)
    for nd in nodes[:10]:
        s, s2 = _f(nd.get("score")), _f(nd.get("scoreSecondary"))
        if not (s and s2):
            continue
        mult = s / s2
        cb = max(mult, 0.1)
        n_impl = cb * B / OURS["cost_per_sample"]
        v_impl = s2 * n_impl
        print(
            f"  {nd.get('rank'):>4} {_name(nd)[:25]:<26} {cb:>7.3f} {n_impl:>12,.0f} {v_impl:>11.3e} "
            f"{OURS['V'] / v_impl:>8.1f}x"
        )

    # ---- 3. timeline ---------------------------------------------------------
    print("\n" + "=" * 78)
    print("3. BOARD-COLLAPSE TIMELINE (from scoreHistory)")
    print("=" * 78)
    events = []
    for nd in nodes[: args.top]:
        for h in nd.get("scoreHistory") or []:
            v = _f(h.get("value"))
            if v is not None:
                events.append((h.get("date"), _name(nd), v))
    events.sort(key=lambda e: e[0] or "")
    best = float("inf")
    print("  record-setting improvements among the top entries:")
    for date, who, v in events:
        if v < best:
            best = v
            print(f"    {date}  {who[:28]:<30} {v:.4g}")

    print("\n  entry counts vs rank (top 30):")
    ent = [(nd.get("rank"), _name(nd), nd.get("entries"), _f(nd.get("score"))) for nd in nodes[: args.top]]
    have = [e for e in ent if isinstance(e[2], (int, float))]
    if have:
        import statistics

        med = statistics.median([e[2] for e in have])
        print(f"    median entries in top {args.top}: {med:.0f}")
        for r, who, n, s in sorted(have, key=lambda e: -(e[2] or 0))[:8]:
            print(f"      rank {r:>3}  {who[:26]:<28} {n:>6} entries   score {s:.4g}")

    if flagged:
        print("\n" + "=" * 78)
        print("FLAGGED (implied V*c more than 3x below the best published method)")
        print("=" * 78)
        for nd, vc, ratio in flagged:
            hist = nd.get("scoreHistory") or []
            first = hist[0].get("date") if hist else "?"
            last = hist[-1].get("date") if hist else "?"
            print(
                f"  rank {nd.get('rank')} {_name(nd)}: V*c={vc:.3g} ({ratio:.0f}x below published best), "
                f"{nd.get('entries')} entries, history {first} -> {last} ({len(hist)} points)"
            )
    print(f"\n(generated {dt.datetime.now(dt.timezone.utc).isoformat()} — PRIVATE, not for the forum)")


if __name__ == "__main__":
    main()
