"""Count complete installed-flopscope operations for F21 and the E14 tree arm.

Run only through workerq at production shape.  This uses the existing
grader-weight/materialisation twin in p2_flop_anchor and does not read labels.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import p2_flop_anchor as anchor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/e14_tree_meter/ledger.json"
CANDIDATE = "estimator_e14_tree.py"
BASELINE = "estimator_k3_strassen_L4_f21.py"


def sha(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def main() -> None:
    before = {p: sha(p) for p in (CANDIDATE, BASELINE,
                                "local_sim/e14_tree_meter.py",
                                "local_sim/p2_flop_anchor.py")}
    baseline = anchor._run_arm(BASELINE, "grader", input_weight_dtype="f64_readonly")
    candidate = anchor._run_arm(CANDIDATE, "grader", input_weight_dtype="f64_readonly")
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
              "instrument": "installed flopscope 0.12.1 BudgetContext, f64_readonly weights and in-window grader materialisation",
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
