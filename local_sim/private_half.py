"""
PUBLIC-PRIVATE DIVERGENCE SELF-CHECK.

Thread 18118 says a Phase-1 evaluation runs 50 public + 50 private MLPs, and
that "public-private divergence" is watched. Our graded reports confirm the
mechanism from the participant side: `results.per_mlp` carries **100** entries,
`results.public_scores` carries **50** (indices 0-49), and `aggregates` has a
`public` key and no sibling. So MLPs 50-99 execute inside our submission and are
scored somewhere we cannot see.

That makes the honest answer to "what is our public-private divergence?" a
NOT VISIBLE — and this script's first job is to establish that properly rather
than assume it, by searching every key of every report for a private score.

Its second job is to extract what *is* visible. The private half's **compute
telemetry** is reported in full: FLOPs, wall time, residual, exhaustion flags.
That is not an accuracy divergence, but it is a behavioural one, and it answers a
question that matters for the private re-run: does the estimator behave the same
on draws it has never been tuned against? A pruning threshold that quietly
prunes less on unseen networks would show up here as a cost divergence long
before it showed up as a score.

Run: uv run python local_sim/private_half.py
"""

import glob
import json
import os
import statistics as st

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS = os.path.join(_ROOT, "experiments", "run_reports")


def find_private_keys(obj, path="", hits=None):
    """Recursively hunt for anything that could be a private-half score."""
    if hits is None:
        hits = []
    needles = ("private", "hidden", "holdout", "secret")
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else k
            if any(nd in k.lower() for nd in needles):
                hits.append((p, type(v).__name__))
            find_private_keys(v, p, hits)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:3]):
            find_private_keys(v, f"{path}[{i}]", hits)
    return hits


def analyse(path):
    d = json.load(open(path))
    ev = d.get("evaluation") or {}
    res = ev.get("results") or {}
    per = res.get("per_mlp") or []
    pub = res.get("public_scores") or []
    if not per:
        return None

    pub_idx = {s["mlp_index"] for s in pub}
    priv = [m for m in per if m["mlp_index"] not in pub_idx]
    pubm = [m for m in per if m["mlp_index"] in pub_idx]

    def tel(ms, key):
        return [m["telemetry"][key] for m in ms if m.get("telemetry") and key in m["telemetry"]]

    out = {
        "file": os.path.basename(path),
        "submission": ev.get("submission_id"),
        "n_per_mlp": len(per),
        "n_public": len(pub),
        "n_private": len(priv),
        "private_score_keys": find_private_keys(d),
        "aggregate_keys": sorted((res.get("aggregates") or {}).keys()),
        "states_private": sorted({m.get("state") for m in priv}),
        "errors_private": sum(1 for m in priv if m.get("error_code")),
    }
    for key in ("effective_compute", "flops_used", "residual_wall_time_s", "wall_time_s"):
        a, b = tel(pubm, key), tel(priv, key)
        if a and b:
            out[key] = (st.mean(a), st.mean(b), st.mean(b) / st.mean(a) if st.mean(a) else float("nan"))
    return out


def main():
    files = sorted(glob.glob(os.path.join(REPORTS, "*graded_report.json")))
    print("=" * 100)
    print("PUBLIC-PRIVATE DIVERGENCE SELF-CHECK")
    print("=" * 100)
    if not files:
        print("no graded reports found")
        return

    rows = [r for r in (analyse(f) for f in files) if r]

    print("\n1. IS ANY PRIVATE-HALF SCORE VISIBLE?")
    print("-" * 100)
    any_hit = False
    for r in rows:
        hits = [h for h in r["private_score_keys"]]
        any_hit |= bool(hits)
        print(f"  {r['file'][:52]:<54} per_mlp={r['n_per_mlp']:>4}  scored={r['n_public']:>3}  "
              f"unscored={r['n_private']:>3}  aggregates={r['aggregate_keys']}")
        if hits:
            for p, t in hits:
                print(f"      !! candidate private field: {p} ({t})")
    print()
    print(f"  VERDICT: private-half scores are "
          f"{'VISIBLE — investigate above' if any_hit else 'NOT VISIBLE'}.")
    print("  Every report exposes exactly the 50 public MLPs' scores and a single")
    print("  `public` aggregate. The other 50 run inside our submission and are")
    print("  scored out of sight, so we cannot compute an accuracy divergence.")

    print("\n2. WHAT IS VISIBLE: DID THE ESTIMATOR BEHAVE THE SAME ON THE HIDDEN 50?")
    print("-" * 100)
    print(f"  {'submission':<12} {'private state':<14} {'err':>4} "
          f"{'eff.compute pub':>16} {'eff.compute priv':>17} {'ratio':>8} {'residual ratio':>15}")
    print("-" * 100)
    for r in rows:
        ec = r.get("effective_compute")
        rw = r.get("residual_wall_time_s")
        if not ec:
            continue
        print(f"  {str(r['submission'])[:11]:<12} {','.join(r['states_private'])[:13]:<14} "
              f"{r['errors_private']:>4} {ec[0]:>16.4e} {ec[1]:>17.4e} {ec[2]:>8.4f} "
              f"{(rw[2] if rw else float('nan')):>15.4f}")
    print("-" * 100)
    print("  A ratio near 1.000 says the estimator spent the same compute per MLP on")
    print("  networks it has never been measured against — the cost half of")
    print("  'no public-private divergence'. The accuracy half is not observable.")
    print()
    print("  For the write-up's integrity section, state it exactly that way: the")
    print("  cost ratio is measured and reported; the score ratio is NOT VISIBLE and")
    print("  is not claimed.")


if __name__ == "__main__":
    main()
