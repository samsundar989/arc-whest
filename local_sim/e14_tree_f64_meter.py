"""Count complete installed-flopscope operations for F21 and full-f64 E14 tree.

Run only through workerq at production shape.  This uses the existing
grader-weight/materialisation twin in p2_flop_anchor and does not read labels.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import p2_flop_anchor as anchor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/e14_tree_f64/ledger.json"
CANDIDATE = "estimator_e14_tree_f64.py"
BASELINE = "estimator_k3_strassen_L4_f21.py"


def sha(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def run_one(path: str, weights, estimator_seed: int) -> dict:
    """Grader-shaped in-window ledger with a diagnostic cap above 2**41."""
    import flopscope as flops
    import flopscope.numpy as fnp

    anchor._require_real_flopscope(flops)
    anchor._stub_whestbench()
    flops.configure(symmetry_warnings=False)
    mlp = __import__("whestbench").MLP(
        weights=weights, width=anchor.WIDTH, depth=anchor.DEPTH,
        seed=estimator_seed)
    mod = anchor._load_candidate(path)
    est = mod.Estimator()

    class Context:
        width = anchor.WIDTH
        depth = anchor.DEPTH
        flop_budget = anchor.BUDGET
        seed = 0

    est.setup(Context())
    # The estimator receives the LIVE budget; only the diagnostic meter's
    # ceiling is raised so an over-budget arm still yields its complete bill.
    with flops.BudgetContext(flop_budget=10**13, quiet=True) as ctx:
        pred = est.predict(mlp, anchor.BUDGET)
        anchor.materialise_like_grader(pred, anchor.DEPTH, anchor.WIDTH)
        total = float(ctx.flops_used)
        ns = ctx.summary_dict(by_namespace=True).get("by_namespace") or {}
    ops = {}
    for blob in ns.values():
        for op, rec in (blob.get("operations") or {}).items():
            entry = ops.setdefault(op, {"flops": 0.0, "calls": 0.0})
            entry["flops"] += float(rec.get("flop_cost", 0.0))
            entry["calls"] += float(rec.get("calls", 0.0))
    return {"total": total, "ops": ops,
            "last_run_repr": repr(getattr(est, "last_run", None))}


def main() -> None:
    source_set = (CANDIDATE, BASELINE,
                  "local_sim/e14_tree_f64_meter.py",
                  "local_sim/p2_flop_anchor.py",
                  "local_sim/lifecycle_eval.py",
                  "local_sim/mint_provenance.py")
    before = {p: sha(p) for p in source_set}
    from lifecycle_eval import build_weights
    import flopscope.numpy as fnp

    raw_weights, estimator_seed = anchor._mint_weights()
    weights = build_weights(raw_weights, "f64_readonly", fnp)
    # The same immutable grader-style weights are reused sequentially.
    baseline = run_one(BASELINE, weights, estimator_seed)
    candidate = run_one(CANDIDATE, weights, estimator_seed)
    if baseline["total"] != 928_457_637_478:
        raise RuntimeError("F21 anchor drift: " + str(baseline["total"]))
    after = {p: sha(p) for p in before}
    if before != after:
        raise RuntimeError("meter source changed during evaluation")
    names = sorted(set(baseline["ops"]) | set(candidate["ops"]))
    delta = {
        name: {
            "baseline_flops": baseline["ops"].get(name, {}).get("flops", 0),
            "candidate_flops": candidate["ops"].get(name, {}).get("flops", 0),
            "baseline_calls": baseline["ops"].get(name, {}).get("calls", 0),
            "candidate_calls": candidate["ops"].get(name, {}).get("calls", 0),
        }
        for name in names
    }
    result = {"source_sha256": before, "fixture": "canonical_mint PROBE_SEED, 1024x16; no truth",
              "instrument": "installed flopscope 0.12.1 BudgetContext with diagnostic 1e13 ceiling, live 2**41 passed into predict; reused immutable f64_readonly weights and in-window grader materialisation",
              "baseline": baseline, "candidate": candidate, "operation_comparison": delta,
              "delta_flops": candidate["total"]-baseline["total"],
              "ratio": candidate["total"]/baseline["total"]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"baseline": baseline["total"],
                      "candidate": candidate["total"],
                      "ratio": result["ratio"], "out": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
