"""Phase-2 reconnaissance -- re-poll the round, the dataset revisions and the boards.

Phase 2 is round **1429** and the published dataset revision is **`v2-phase2`**
(width **1024**, depth **16**, B = 2**41, multiplier floor 0.1). Dataset
publication and round activation are SEPARATE events -- the files went public
before the round opened -- so this script reports each separately and never
infers one from the other.

  1. Which rounds exist, and is 1429 actually LIVE?
  2. Which dataset revisions exist, and does `v2-phase2` still point at the
     commit the manifest was built against? (drift there is a re-bake, and it
     invalidates every shape-dependent result)
  3. Fresh leaderboard snapshot for every round.
  4. What is still unknown, and what would settle it.

It does NOT submit anything -- submission stays a deliberate act.

The authoritative Phase-2 constants live in
`experiments/2026-08-21_phase2_manifest/phase2_manifest.json`, built by
`local_sim/phase2_manifest.py`. This script READS that manifest and cross-checks
it against the live tag list; it never restates the constants from memory.

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
PHASE2_TAG = "v2-phase2"
_MANIFEST = os.path.join(
    _ROOT, "experiments", "2026-08-21_phase2_manifest", "phase2_manifest.json"
)


def _manifest():
    """The official Phase-2 constants, or None if the manifest was never built."""
    try:
        with open(_MANIFEST, encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None


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
    commits = {t.name: t.target_commit for t in refs.tags}
    print(f"  tags: {tags}")

    if PHASE2_TAG not in tags:
        print(f"  *** {PHASE2_TAG} IS GONE from the tag list -- a re-tag or a")
        print("      withdrawal. STOP and re-read the dataset card before trusting")
        print("      any Phase-2 number in this repository. ***")
        return tags

    print(f"  {PHASE2_TAG}: commit {commits.get(PHASE2_TAG)}")
    man = _manifest()
    if man is None:
        print("  !! no Phase-2 manifest on disk. Build it first:")
        print("       uv run python local_sim/phase2_manifest.py")
        return tags

    d, sh, b = man["dataset"], man["shape"], man["budget"]
    print(
        f"  manifest:  {sh['width']} x {sh['depth']}, "
        f"B = {b['flop_budget_per_mlp']:,} ({b['flop_budget_expr']}), "
        f"floor {b['multiplier_floor']}, "
        f"{b['effective_naive_samples']:,} naive samples"
    )
    if d["commit"] != commits.get(PHASE2_TAG):
        print("  *** COMMIT DRIFT -- the manifest was built against")
        print(f"      {d['commit']}")
        print(f"      but {PHASE2_TAG} now points at")
        print(f"      {commits.get(PHASE2_TAG)}")
        print("      The dataset was re-baked or re-tagged. Rebuild the manifest")
        print("      and re-check every shape-dependent result. ***")
    else:
        print("  commit matches the manifest -- no re-bake since it was built.")

    par = man["package_parity"]
    print(
        f"  package parity: {par['status']}"
        f"   (bake stack whestbench {par['bake_stack']['whestbench']}"
        f" / flopscope {par['bake_stack']['flopscope']};"
        f" installed {par['installed_here']['whestbench']}"
        f" / {par['installed_here']['flopscope']})"
    )
    for bl in par["blockers"]:
        print(f"    - {bl}")

    print()
    print("  MIGRATION STATE -- Phase 2 is 1024 x 16, NOT 256 x 32:")
    print("    * Phase-1 seed blocks (D16-D19) and the 500-network vault are")
    print("      256x32 allocations. They CANNOT be reused at the new shape.")
    print("    * `_PRUNE_TAU = -2.5` and `_BASIS_RANK = 32` are Phase-1 constants")
    print("      of a width-256 sampling architecture. Do NOT refit them as the")
    print("      first Phase-2 action -- establish which family owns the frontier")
    print("      first (docs/PHASE2_FRONTIER_ARCHITECTURE.md, P2-CP1).")
    print("    * The CP1.1 Kerdock/MUB-129 carrier is a width-256 object. Its")
    print("      width-1024 analogue is 1,050,624 points, ~16 budgets. Dead as a")
    print("      Phase-2 backbone.")
    print("    * `estimator.py` (v2.6.1) is shape-generic and RUNS at 1024x16, but")
    print("      takes the analytic-only fallback for width > 256. A score from it")
    print("      is telemetry, not a competitive Phase-2 entry.")
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
    print("4. STILL UNKNOWN -- only a graded Phase-2 report settles these")
    print("=" * 72)
    man = _manifest()
    if man is not None:
        u = man["unknowns"]
        print(f"  lambda (residual price)  {u['lambda_residual_price']['status']}")
        for k in ("setup_timeout_s", "wall_time_limit_s", "predict_timeout_s",
                  "memory_limit_mb"):
            print(f"  {k:<24} package default {u[k]['package_default']}, "
                  f"observed {u[k]['observed']}")
        print(f"  {'n scored public MLPs':<24} Phase 1 = "
              f"{u['n_scored_public_mlps']['phase1']}, Phase 2 = "
              f"{u['n_scored_public_mlps']['phase2']}")
    print("""
  Once a Phase-2 submission exists:
    uv run python local_sim/fetch_graded_report.py <submission_id> 2026-08-21_phase2
  and record from it into RESEARCH_NOTES section 1:
    * config_snapshot: width / depth / n_mlps / flop_budget
    * lambda_derived and residual_share_of_billed (per-MLP telemetry)
    * multiplier_floor_check  (score / score_secondary; 0.1 => floor unchanged)
    * runtime_environment (whestbench / flopscope versions) -- the ONLY thing
      that resolves the manifest's provenance-vs-legacy version split
  Then rebuild the manifest so those unknowns become observed values.""")


if __name__ == "__main__":
    main()
