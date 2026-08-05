"""
Phase 2 opening checklist — run this on Aug 8 when round 1429 goes live.

Phase 2 (round 1429) opens 2026-08-08 00:00 UTC and runs to 09-19 23:59. This
script does the read-only reconnaissance in one shot so the only manual decision
left is whether to submit:

  1. Which rounds are live, and is 1429 actually open?
  2. Does a `v1-phase2` dataset revision exist? Any new HF tags at all?
  3. Fresh leaderboard snapshot for every round.
  4. Reminder of what to verify from the first Phase 2 graded report.

It does NOT submit anything — submission stays a deliberate act.

Run: uv run python local_sim/phase2_checklist.py
"""

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
REPO = "aicrowd/arc-whestbench-public-2026"


def _page(url, tok):
    return httpx.get(url, headers={"Authorization": f"Token {tok}"}, timeout=90).text


def _props(page):
    out = []
    for m in re.finditer(r'data-react-props="(.*?)"(?=\s|>)', page, re.S):
        try:
            out.append(json.loads(html.unescape(m.group(1))))
        except Exception:
            pass
    return out


def rounds(tok):
    print("=" * 72)
    print("1. ROUNDS")
    print("=" * 72)
    page = _page(f"https://www.aicrowd.com/challenges/{SLUG}", tok)
    seen = {}
    for p in _props(page):
        for m in re.finditer(
            r'\{"id":\s*"(\d+)",\s*"name":\s*"([^"]+)"[^{}]*?"startDttm":\s*"([^"]*)",\s*"endDttm":\s*"([^"]*)"[^{}]*?\}',
            json.dumps(p, default=str),
        ):
            seen[m.group(1)] = m.groups()
    if not seen:
        for p in _props(page):
            for m in re.finditer(r'\{"id":\s*"(\d+)",\s*"name":\s*"(Phase[^"]*)"[^{}]*\}', json.dumps(p, default=str)):
                seen.setdefault(m.group(1), (m.group(1), m.group(2), "?", "?"))
    for rid, (_, name, start, end) in sorted(seen.items()):
        live = '"active": true' in json.dumps(_props(page), default=str) and rid in json.dumps(_props(page))
        print(f"  round {rid}  {name:<12} {start} -> {end}")
    blob = json.dumps(_props(page), default=str)
    for rid in seen:
        for state in ("LIVE", "UPCOMING", "COMPLETED"):
            if f'"id": "{rid}"' in blob and state in blob:
                pass
    m = re.findall(r'\{"id":\s*"(\d+)",\s*"name":\s*"([^"]+)",\s*"active":\s*(true|false),\s*"state":\s*"([^"]+)"\}', blob)
    for rid, name, active, state in m:
        flag = "  <== SUBMIT HERE" if state == "LIVE" else ""
        print(f"  state: round {rid} {name:<10} active={active:<5} {state}{flag}")
    return seen


def dataset_tags():
    print("\n" + "=" * 72)
    print("2. DATASET REVISIONS")
    print("=" * 72)
    from huggingface_hub import HfApi

    refs = HfApi().list_repo_refs(REPO, repo_type="dataset")
    tags = [t.name for t in refs.tags]
    print(f"  tags: {tags}")
    if any("phase2" in t for t in tags):
        print("  *** v1-phase2 EXISTS -- switch local runs to it and re-measure the error-vs-N curve ***")
    else:
        print("  no phase-2 tag yet; keep using v1-phase1 for local runs")
    return tags


def leaderboards(tok):
    print("\n" + "=" * 72)
    print("3. LEADERBOARDS")
    print("=" * 72)
    page = _page(f"https://www.aicrowd.com/challenges/{SLUG}/leaderboards", tok)
    for p in _props(page):
        lbs = (((p.get("props") or {}).get("data") or {}).get("challenge") or {}).get("leaderboards")
        if not lbs:
            continue
        for lb in lbs:
            nodes = ((lb.get("entries") or {}).get("nodes")) or []
            print(f"\n  leaderboard {lb.get('id')} (round {lb.get('challengeRoundId')}) '{lb.get('name')}': {len(nodes)} entries")
            for nd in nodes[:5]:
                who = (nd.get("participant") or {}).get("name")
                print(f"    #{nd.get('rank'):<4} {str(who)[:28]:<28} {nd.get('score')}")
            for nd in nodes:
                if nd.get("viewerIsSelf") or "sundararaman" in json.dumps(nd, default=str).lower():
                    print(f"    >>> US: rank {nd.get('rank')}  score {nd.get('score')}")


def main():
    tok = aicrowd_config.resolve_api_key(None)
    rounds(tok)
    dataset_tags()
    leaderboards(tok)
    print("\n" + "=" * 72)
    print("4. AFTER THE FIRST PHASE-2 SUBMISSION")
    print("=" * 72)
    print("""  uv run python local_sim/fetch_graded_report.py <submission_id> 2026-08-08_phase2
  and record from it into RESEARCH_NOTES section 1:
    * config_snapshot: width / depth / n_mlps / flop_budget
    * lambda_derived and residual_share_of_billed (per-MLP telemetry)
    * multiplier_floor_check  (score / score_secondary; 0.1 => floor unchanged)
    * runtime_environment (whestbench / flopscope versions)
  If width or depth changed:
    1) refit calibration for the new shape, 2) re-measure error-vs-N at the new
    budget, 3) refit any fusion scalars -- IN THAT ORDER. estimator_v1_1 is
    shape-generic (Sobol table covers width <= 256) and stays the safe fallback.""")


if __name__ == "__main__":
    main()
