"""Phase-2 reconnaissance -- re-poll the round, the dataset revisions and the boards.

Phase 2 is round **1429** and the published dataset revision is **`v2-phase2`**
(width **1024**, depth **16**, B = 2**41, multiplier floor 0.1). Dataset
publication and round activation are SEPARATE events -- the files went public
before the round opened -- so this script reports each separately and never
infers one from the other.

  1. Which rounds exist, and is 1429 actually LIVE?
  2. Which dataset revisions exist, and does `v2-phase2` still point at the
     commit the manifest was built against? A moved tag is NOT automatically a
     re-bake: shape-dependent evidence depends on the `data/*.parquet` SOURCE
     objects only, so the verdict comes from `local_sim/p2_tag_drift.py`, which
     compares all 70 of them. An UNMEASURED move is still treated as
     invalidating.
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

for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

from p2_tag_drift import (  # noqa: E402
    PINNED, SOURCE_EQUIVALENT, classify_live_commit,
)
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
    # A MOVED TAG IS NOT AUTOMATICALLY A RE-BAKE, and until 2026-08-31 this
    # branch said it was. The classification lives in `local_sim/p2_tag_drift.py`
    # (one shared policy, offline-testable), over TWO durable content-addressed
    # comparisons: the 70 `data/*.parquet` source objects, and the FULL 226-object
    # tree.
    #
    # The full tree is not optional. An earlier version of this comment said
    # every shape-dependent result is a function of the source parquet "never of
    # the repo's file count, its prepared-split artifacts, or `metadata.json`'s
    # split index". **That is false and is retracted**: whestbench 0.16.0 PREFERS
    # a `prepared/<split>` Arrow tree over the parquet path whenever
    # `metadata.json` advertises `prepared_splits` (`dataset.py:479-491`,
    # `dataset_io.py:61-68`), `whest run` is non-streaming by default, and this
    # very tag move added that tree. So the loader-visible input set is
    # `data/**` + `metadata.json` + `prepared/**`, and a source-only comparison
    # licenses nothing -- an absent loader record is now an integrity violation.
    #
    # An UNMEASURED move still raises the original alarm. That is what keeps
    # SOURCE_EQUIVALENT falsifiable: an arbitrary commit cannot obtain it.
    live = commits.get(PHASE2_TAG)
    verdict = classify_live_commit(live or "", d["commit"])
    for line in verdict.report_lines():
        print(line)
    if verdict.status == PINNED:
        print("  commit matches the manifest -- no re-bake since it was built.")
    elif verdict.status == SOURCE_EQUIVALENT:
        print("  *** THE EXECUTABLE PIN IS STALE (bookkeeping), and stored")
        print("      Phase-2 mint/CONFIRM bytes REMAIN VALID (evidence). Repin")
        print("      with:  uv run python local_sim/phase2_manifest.py"
              " --allow-rebake ***")
    else:
        print("  *** COMMIT DRIFT -- the manifest was built against")
        print(f"      {d['commit']}")
        print(f"      but {PHASE2_TAG} now points at")
        print(f"      {live}")
        print("      This move is NOT covered by a source comparison. Treat it")
        print("      as a re-bake: rebuild the manifest, capture a fresh")
        print("      read-only tree comparison, and re-check every")
        print("      shape-dependent result. ***")

    lin = d.get("commit_lineage")
    if lin:
        print(f"  lineage: record {str(lin.get('record_old_commit'))[:12]} -> "
              f"{str(lin.get('record_new_commit'))[:12]}, manifest pinned at "
              f"{str(lin.get('manifest_pin'))[:12]} "
              f"({lin['status']}, {lin['source_objects']} source objects, "
              f"{lin['source_objects_changed']} changed)")
        ldr = lin.get("loader")
        if ldr:
            print(f"  loader:  {ldr['objects_compared']} objects compared; "
                  f"route changed = {ldr['loader_path_changed']}; "
                  f"functional check {ldr['functional_check']}")
        else:
            print("  !! no LOADER record in the lineage. data/ is NOT the whole")
            print("     input set: whestbench prefers prepared/<split> Arrow when")
            print("     metadata.json advertises prepared_splits.")
    else:
        print("  !! the manifest carries no dataset.commit_lineage. Refresh it:")
        print("       uv run python local_sim/phase2_manifest.py --refresh-lineage")

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
    print("      first (docs/GAME_PLAN.md; the old frontier packet is archived).")
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
    print("4. RESOLVED FROM THE BAKE STACK, and what a graded report still owes us")
    print("=" * 72)
    man = _manifest()
    if man is not None:
        # Key renamed from "unknowns" on 2026-08-24 (P2-P0): installing the bake
        # stack resolved every scalar in it. The old name is still read, so an
        # un-rebuilt manifest reports honestly instead of raising.
        u = man.get("resolved_from_package") or man.get("unknowns") or {}
        if "resolved_from_package" not in man:
            print("  *** this manifest predates the P2-P0 repair; rebuild it with")
            print("      uv run python local_sim/phase2_manifest.py ***")
        print(f"  lambda (residual price)  {u['lambda_residual_price']['status']}")
        for k in ("setup_timeout_s", "wall_time_limit_s", "predict_timeout_s",
                  "memory_limit_mb", "residual_wall_time_limit_s"):
            rec = u.get(k) or {}
            print(f"  {k:<28} package default {rec.get('package_default')}, "
                  f"observed {rec.get('observed')}")
        print(f"  {'n scored public MLPs':<28} Phase 1 = "
              f"{u['n_scored_public_mlps']['phase1']}, Phase 2 = "
              f"{u['n_scored_public_mlps']['phase2']}   <- STILL UNKNOWN")
        print()
        print("  These are PACKAGE facts, not GRADED facts. No graded report we")
        print("  hold publishes any limit above, and a graded Phase-2 report is")
        print("  still the only thing that confirms the runtime we are scored on.")
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
