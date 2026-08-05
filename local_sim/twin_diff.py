"""
Twin test: diff two graded reports of the SAME estimator, per MLP.

The question is whether a round's evaluation suite is fixed across submissions or
redrawn each time. It matters for two reasons: a fixed suite makes a single
graded run a much weaker signal than it looks (we would be re-measuring the same
100 draws, not sampling new ones), and it changes how much weight to put on
grader-vs-local differences when deciding whether a change helped.

Our estimator is deterministic given `mlp.seed`, so:

  identical per-MLP mse AND identical mlp ids  => same MLPs, same references
  identical ids, mse differs slightly          => same MLPs, references redrawn
  ids differ                                   => suite redrawn per submission

wall time and residual will differ regardless (machine noise); flops must not.

Run:
    uv run python local_sim/fetch_graded_report.py <id_a> twin_a
    uv run python local_sim/fetch_graded_report.py <id_b> twin_b
    uv run python local_sim/twin_diff.py <report_a.json> <report_b.json>
"""

import glob
import json
import os
import sys


def suite_survey():
    """Compare the MLP identities across every graded report we hold, and against
    the public splits. Answers the fixed-suite question with no submission spend."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths = sorted(glob.glob(os.path.join(root, "experiments", "run_reports", "*_graded_report.json")))
    if not paths:
        raise SystemExit("no graded reports")
    names = {}
    for p in paths:
        r = json.load(open(p, encoding="utf-8"))
        names[os.path.basename(p)] = [row["mlp_name"] for row in r["evaluation"]["results"]["per_mlp"]]
    first = names[os.path.basename(paths[0])]
    print(f"{len(paths)} graded reports\n")
    print(f"{'report':<46} {'n':>4} {'same order':>11} {'same set':>9}")
    print("-" * 74)
    for k, v in names.items():
        print(f"{k[:45]:<46} {len(v):>4} {str(v == first):>11} {str(set(v) == set(first)):>9}")
    same = all(v == first for v in names.values())
    print(f"\nSUITE IDENTITY: {'FIXED across all submissions' if same else 'VARIES between submissions'}")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        import real_data as rd

        for split in ("mini", "full"):
            pub = [rd.get(i, split=split)[2]["mlp_name"] for i in range(rd.n_mlps(split))]
            print(f"  overlap with public {split:<5}: {len(set(first) & set(pub))}/{len(first)}")
    except Exception as e:  # offline is fine; the identity answer above stands
        print(f"  (public-split comparison skipped: {type(e).__name__})")

    # what the per-submission feedback channel actually contains
    r = json.load(open(paths[-1], encoding="utf-8"))
    ps = r["evaluation"]["results"].get("public_scores") or []
    if ps:
        k = sorted(ps[0].get("scores", {}).keys())
        npl = len(ps[0]["scores"].get("per_layer_mse") or [])
        print(f"\nFEEDBACK CHANNEL per submission: {len(ps)} per-MLP rows, fields {k}, "
              f"per_layer_mse length {npl}")
        print(f"  => {len(ps)} exact per-MLP final-layer MSEs (and {len(ps) * npl} per-layer values) "
              f"are returned for every submission.")


def per_mlp(rep):
    ev = rep["evaluation"]
    rows = {}
    for r in ev.get("results", {}).get("per_mlp", []) or []:
        key = r.get("mlp_id", r.get("mlp_name", len(rows)))
        rows[key] = r
    return rows, (ev.get("results", {}).get("aggregates", {}) or {}).get("public", {})


def num(r, *path):
    cur = r
    for p in path:
        cur = (cur or {}).get(p) if isinstance(cur, dict) else None
    return cur


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "--all":
        suite_survey()
        return
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    a = json.load(open(sys.argv[1], encoding="utf-8"))
    b = json.load(open(sys.argv[2], encoding="utf-8"))
    ra, agg_a = per_mlp(a)
    rb, agg_b = per_mlp(b)

    print(f"A: submission {a.get('id')}  round {a.get('round')}")
    print(f"B: submission {b.get('id')}  round {b.get('round')}\n")

    print("aggregate fields:")
    keys = sorted(set(agg_a) | set(agg_b))
    for k in keys:
        va, vb = agg_a.get(k), agg_b.get(k)
        same = "SAME" if va == vb else "DIFF"
        print(f"  {k:<40} {str(va)[:22]:>24} {str(vb)[:22]:>24}  {same}")

    ids_a, ids_b = set(ra), set(rb)
    print(f"\nMLP ids: {len(ids_a)} vs {len(ids_b)}; shared {len(ids_a & ids_b)}")
    if ids_a != ids_b:
        print("  => SUITE IS REDRAWN (or ids are not stable identifiers)")
    shared = sorted(ids_a & ids_b)
    if not shared:
        return

    exact_mse = exact_flops = 0
    wall_ratio = []
    print(f"\n{'mlp':<26} {'mse A':>13} {'mse B':>13} {'rel diff':>11} {'flops same':>11} {'wall A/B':>9}")
    print("-" * 90)
    for k in shared[:15]:
        ma = num(ra[k], "final_layer_mse") or num(ra[k], "metrics", "final_layer_mse")
        mb = num(rb[k], "final_layer_mse") or num(rb[k], "metrics", "final_layer_mse")
        fa = num(ra[k], "telemetry", "flops_used")
        fb = num(rb[k], "telemetry", "flops_used")
        wa = num(ra[k], "telemetry", "residual_wall_time_s")
        wb = num(rb[k], "telemetry", "residual_wall_time_s")
        rel = abs(ma - mb) / ma if (ma and mb) else float("nan")
        print(f"{str(k)[:25]:<26} {ma if ma is not None else '?':>13} {mb if mb is not None else '?':>13} "
              f"{rel:>11.2e} {str(fa == fb):>11} {(wa / wb if wa and wb else float('nan')):>9.3f}")
    for k in shared:
        ma = num(ra[k], "final_layer_mse") or num(ra[k], "metrics", "final_layer_mse")
        mb = num(rb[k], "final_layer_mse") or num(rb[k], "metrics", "final_layer_mse")
        fa = num(ra[k], "telemetry", "flops_used")
        fb = num(rb[k], "telemetry", "flops_used")
        wa = num(ra[k], "telemetry", "residual_wall_time_s")
        wb = num(rb[k], "telemetry", "residual_wall_time_s")
        exact_mse += int(ma == mb)
        exact_flops += int(fa == fb)
        if wa and wb:
            wall_ratio.append(wa / wb)

    n = len(shared)
    print(f"\nover all {n} shared MLPs:")
    print(f"  bit-identical final_layer_mse : {exact_mse}/{n}")
    print(f"  identical flops_used          : {exact_flops}/{n}")
    if wall_ratio:
        wr = sorted(wall_ratio)
        print(f"  residual wall ratio A/B       : median {wr[n // 2]:.3f} "
              f"[{wr[0]:.3f}, {wr[-1]:.3f}]  (machine noise, expected to differ)")
    print("\nVERDICT")
    if exact_mse == n and ids_a == ids_b:
        print("  FIXED SUITE — same MLPs and same reference values on both runs.")
        print("  Consequence: one graded run is a re-measurement, not a fresh sample;")
        print("  treat repeated graded scores as ONE observation for decision purposes.")
    elif ids_a == ids_b:
        print("  Same MLPs, but the scored values moved => the references carry")
        print("  run-to-run noise. Size it from the rel-diff column before reading")
        print("  any grader-vs-local difference as a real effect.")
    else:
        print("  Suite differs between runs => each graded run is a fresh sample.")


if __name__ == "__main__":
    main()
