"""
Archive a full leaderboard snapshot to experiments/board_snapshots/.

Why this exists
---------------
Two uses, both time-sensitive:

1. **Phase 1 close (Aug 7 23:59 UTC).** The final board is evidence for the
   write-up's forensics section, and it stops being retrievable once the round
   rolls over and the site starts showing Phase 2. Archive it before then.
2. **The entries-vs-score time series.** A single final snapshot cannot show
   whether entries appear or move late in a round; a sequence of dated snapshots
   can. Run this daily during Phase 2 and the diff is free.

Everything is written twice: the raw node payload as JSON (so a field we did not
think to render is still there later) and a top-N markdown table for reading.

Run:
    uv run python local_sim/board_snapshot.py                 # all rounds, top 30
    uv run python local_sim/board_snapshot.py --top 50
    uv run python local_sim/board_snapshot.py --probe         # show available fields
    uv run python local_sim/board_snapshot.py --diff a.json b.json
"""

import argparse
import datetime as dt
import html
import json
import os
import re
import sys

import httpx

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]

from whestbench import aicrowd_config  # noqa: E402

SLUG = "arc-white-box-estimation-challenge-2026"
OUT = os.path.join(_ROOT, "experiments", "board_snapshots")
US = "sundararaman"


def _props(page):
    out = []
    for m in re.finditer(r'data-react-props="(.*?)"(?=\s|>)', page, re.S):
        try:
            out.append(json.loads(html.unescape(m.group(1))))
        except Exception:
            pass
    return out


def fetch_boards():
    tok = aicrowd_config.resolve_api_key(None)
    page = httpx.get(
        f"https://www.aicrowd.com/challenges/{SLUG}/leaderboards",
        headers={"Authorization": f"Token {tok}"},
        timeout=90,
    ).text
    boards = []
    for p in _props(page):
        lbs = (((p.get("props") or {}).get("data") or {}).get("challenge") or {}).get("leaderboards")
        for lb in lbs or []:
            boards.append(lb)
    if not boards:
        raise SystemExit("no leaderboard payload found (login expired? page layout changed?)")
    return boards


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _name(node):
    for key in ("participant", "team", "submitter"):
        v = node.get(key)
        if isinstance(v, dict) and v.get("name"):
            return str(v["name"])
        if isinstance(v, str) and v:
            return v
    return "?"


def render(boards, top):
    lines = []
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines.append(f"# Leaderboard snapshot — {stamp}\n")
    for lb in boards:
        nodes = ((lb.get("entries") or {}).get("nodes")) or []
        rnd = lb.get("challengeRoundId")
        lines.append(f"\n## leaderboard {lb.get('id')} — round {rnd} — '{lb.get('name')}'")
        lines.append(f"\n*{len(nodes)} entries visible*\n")
        if not nodes:
            lines.append("(empty)\n")
            continue
        keys = set(nodes[0].keys())
        sec = "score_secondary" if "score_secondary" in keys else ("scoreSecondary" if "scoreSecondary" in keys else None)
        ent = next((k for k in ("entries", "submissionCount", "submissions_count", "entriesCount") if k in keys), None)
        lines.append(f"| rank | team | adjusted (score) | {sec or 'secondary'} | {ent or 'entries'} | implied C/B |")
        lines.append("|---:|---|---:|---:|---:|---:|")
        for nd in nodes[:top]:
            s = _num(nd.get("score"))
            s2 = _num(nd.get(sec)) if sec else None
            # adjusted = mse * max(0.1, C/B); if the secondary column is the raw MSE
            # then score/secondary recovers the multiplier the entry ran at.
            mult = (s / s2) if (s and s2) else None
            lines.append(
                f"| {nd.get('rank')} | {_name(nd)[:32]} | {s if s is not None else '?'} | "
                f"{s2 if s2 is not None else '?'} | {nd.get(ent) if ent else '?'} | "
                f"{f'{mult:.4f}' if mult else '?'} |"
            )
        for nd in nodes:
            if US in json.dumps(nd, default=str).lower() or nd.get("viewerIsSelf"):
                lines.append(f"\n**us: rank {nd.get('rank')} — score {nd.get('score')}**\n")
    return "\n".join(lines) + "\n"


def diff(path_a, path_b, top=30):
    a = json.load(open(path_a, encoding="utf-8"))
    b = json.load(open(path_b, encoding="utf-8"))

    def flat(snap):
        out = {}
        for lb in snap["boards"]:
            for nd in ((lb.get("entries") or {}).get("nodes")) or []:
                out[(lb.get("challengeRoundId"), _name(nd))] = nd
        return out

    fa, fb = flat(a), flat(b)
    print(f"{a['fetched_utc']}  ->  {b['fetched_utc']}")
    print(f"{'team':<34} {'was':>12} {'now':>12} {'change':>10}")
    print("-" * 72)
    for k in sorted(set(fa) | set(fb), key=lambda k: _num(fb.get(k, fa.get(k, {})).get("score")) or 9e9):
        na, nb = fa.get(k), fb.get(k)
        sa = _num(na.get("score")) if na else None
        sb = _num(nb.get("score")) if nb else None
        if sa == sb:
            continue
        tag = "NEW" if sa is None else ("GONE" if sb is None else f"{sa / sb:.2f}x")
        print(f"{str(k[1])[:33]:<34} {sa if sa is not None else '-':>12} {sb if sb is not None else '-':>12} {tag:>10}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--label", default="")
    ap.add_argument("--probe", action="store_true", help="print the fields each entry node carries")
    ap.add_argument("--diff", nargs=2, metavar=("OLD", "NEW"))
    args = ap.parse_args()

    if args.diff:
        diff(*args.diff, top=args.top)
        return

    boards = fetch_boards()
    if args.probe:
        for lb in boards:
            nodes = ((lb.get("entries") or {}).get("nodes")) or []
            print(f"leaderboard {lb.get('id')} round {lb.get('challengeRoundId')} '{lb.get('name')}': "
                  f"{len(nodes)} entries; board keys = {sorted(lb.keys())}")
            if nodes:
                print(f"  node keys = {sorted(nodes[0].keys())}")
                print(f"  first node = {json.dumps(nodes[0], indent=2, default=str)[:1200]}")
        return

    os.makedirs(OUT, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc)
    stamp = now.strftime("%Y-%m-%dT%H%MZ")
    label = f"_{args.label}" if args.label else ""
    payload = {"fetched_utc": now.isoformat(), "slug": SLUG, "boards": boards}
    jpath = os.path.join(OUT, f"{stamp}{label}_board.json")
    mpath = os.path.join(OUT, f"{stamp}{label}_board.md")
    with open(jpath, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    with open(mpath, "w", encoding="utf-8") as fh:
        fh.write(render(boards, args.top))
    print(f"saved {jpath}")
    print(f"saved {mpath}")
    for lb in boards:
        nodes = ((lb.get("entries") or {}).get("nodes")) or []
        head = f"top={nodes[0].get('score')}" if nodes else "empty"
        print(f"  round {lb.get('challengeRoundId')} '{lb.get('name')}': {len(nodes)} entries, {head}")


if __name__ == "__main__":
    main()
