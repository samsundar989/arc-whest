"""PRE-REGISTERED verdict for the Skeptic follow-up P11 read (2026-09-17). Written before data.

Contract: `experiments/LOG.md`, *CONTRACT (Rule 1): Skeptic follow-ups F — kappa_iiij-joint closure and a4 shared bundles*.
One `p2_design_score.py` artifact, control v32, candidates:
  v35_Q3 (same-run comparator), v35_Q3q, v32_p4d5, v37_a4r256_p4d5, v37_a4r384_p4d5.
Score per network = mse * max(0.1, flops/2**41), paired on seed, studentised t.

C1 QUESTION (closure):
  F1  Q3   -> Q3q   decide(gate=1.00)   does kappa_iiij help the BUDGET-3 closure?
  F2  v32  -> Q3q   decide(gate=1.00) and its gain g2:
        F2 passes            -> "C1 lane closure WRONG: kappa_iiij-consistent closure beats v32" (reopen only by new contract)
        g2 <= 0.80           -> "C1 stands as worded" (mixed-order inconsistency does not explain the harm)
        otherwise            -> "partial: kappa_iiij recovers part (per F1); closure still not supported on this base"
C3 QUESTION (shared coordinates on the adjusted objective):
  F3  v32_p4d5 -> v37_a4r256_p4d5  decide(gate=1.00)
  F4  v32_p4d5 -> v37_a4r384_p4d5  decide(gate=1.00)
  COST BASE: lowest mean score among {a4r256, a4r384} bundles passing F3/F4; else v32_p4d5.
  F5  v32 -> COST BASE  decide(gate=1.15)   reported as the base-advancement statistic (design tier).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]

import numpy as np  # noqa: E402

B = 2 ** 41
NAMES = {"v32": "estimator_k3_v32.py", "Q3": "estimator_k3_v35_Q3.py", "Q3q": "estimator_k3_v35_Q3q.py",
         "p4d5": "estimator_k3_v32_p4d5.py", "a4r256": "estimator_k3_v37_a4r256_p4d5.py",
         "a4r384": "estimator_k3_v37_a4r384_p4d5.py"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    import competition_score as cs
    d = json.load(open(a.score, encoding="utf-8"))
    by = {}
    for arm in d["arms"].values():
        base = os.path.basename(arm["path"])
        key = next((k for k, v in NAMES.items() if v == base), None)
        if key is None:
            continue
        rows = sorted(arm["rows"], key=lambda r: int(r["seed"]))
        if any(r["failed"] or r["nonfinite"] for r in rows):
            raise SystemExit(f"{key}: failed rows")
        by[key] = {"seeds": [int(r["seed"]) for r in rows], "sha256": arm.get("candidate_sha256"),
                   "score": np.array([float(r["mse"]) * max(0.1, float(r["flops_used"]) / B) for r in rows]),
                   "mse": np.array([float(r["mse"]) for r in rows]),
                   "flops": np.array([float(r["flops_used"]) for r in rows])}
    if set(by) != set(NAMES):
        raise SystemExit(f"arms missing: {sorted(set(NAMES) - set(by))}")
    for k in by:
        if by[k]["seeds"] != by["v32"]["seeds"]:
            raise SystemExit(f"{k}: different networks")
    out = {"artifact": "s8_decide", "tier": "DESIGN (P11, reused)",
           "arms": {k: {"sha256": v["sha256"], "raw": float(v["mse"].mean()), "c_over_b": float(v["flops"].mean() / B),
                        "score": float(v["score"].mean())} for k, v in by.items()}, "comparisons": {}}

    def cmp(tag, c, x, gate):
        v = cs.decide(by[c]["score"], by[x]["score"], gate=gate, interval="t")
        rec = {"control": c, "candidate": x, "gate": gate, "passed": v.passed, "gain": v.gain,
               "raw_gain": float(by[c]["mse"].mean() / by[x]["mse"].mean()), "t95": list(v.ci),
               "better": v.n_candidate_better, "reasons": list(v.reasons)}
        out["comparisons"][tag] = rec
        print(f"{tag} {c:>6} -> {x:<6} gate {gate:.2f} {'PASS' if v.passed else 'fail'}  gain {v.gain:.4f}x  "
              f"raw {rec['raw_gain']:.4f}x  t95 [{v.ci[0]:+.2e}, {v.ci[1]:+.2e}]  better {v.n_candidate_better}/{v.n}")
        return rec

    f1 = cmp("F1", "Q3", "Q3q", 1.00)
    f2 = cmp("F2", "v32", "Q3q", 1.00)
    if f2["passed"]:
        c1 = "C1 lane closure WRONG: kappa_iiij-consistent closure beats v32"
    elif f2["gain"] <= 0.80:
        c1 = "C1 stands as worded"
    else:
        c1 = "partial: kappa_iiij recovers part (per F1); closure still not supported on this base"
    f3 = cmp("F3", "p4d5", "a4r256", 1.00)
    f4 = cmp("F4", "p4d5", "a4r384", 1.00)
    passing = [(out["arms"][k]["score"], k) for k, f in (("a4r256", f3), ("a4r384", f4)) if f["passed"]]
    base = min(passing)[1] if passing else "p4d5"
    f5 = cmp("F5", "v32", base, 1.15)
    out["C1"] = c1
    out["cost_base"] = {"arm": base, "file": NAMES[base]}
    print("\nC1:", c1, "\nCOST BASE:", base, NAMES[base])
    for k, v in out["arms"].items():
        print(f"  {k:<7} raw {v['raw']:.4e}  C/B {v['c_over_b']:.4f}  score {v['score']:.4e}")
    if a.out:
        json.dump(out, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
