"""The PRE-REGISTERED verdict for the handoff-09-17 step-2 P11 screen. Written before the data.

Contract: `experiments/LOG.md`, 2026-09-17, *CONTRACT (Rule 1): step 2 coherent-base screen*.
Input: the `p2_design_score.py` artifact (control v32, candidates v35_R, v35_Q,
v35_P, v35_QP; P11 DESIGN, 24 networks, caps off).

Per-network score is `mse * max(0.1, flops / 2**41)` exactly as the artifact's rows
carry them (observed, averaged two-replica labels). All comparisons pair on SEED.

REGISTERED COMPARISONS (control first):
  C1  v32 vs v35_Q    decide(gate=1.15, interval="t")   isolated closure, BASE-ADVANCEMENT gate
  C2  v32 vs v35_QP   decide(gate=1.15, interval="t")   combined coherent base, BASE-ADVANCEMENT gate
  C3  v32 vs v35_P    decide(gate=1.00, interval="t")   protected feed: supported improvement (t95 > 0)
  C4  v32 vs v35_R    decide(gate=1.00, interval="t")   readout-only DIAGNOSTIC (attribution; never a base)
  C5  v35_Q vs v35_QP decide(gate=1.00, interval="t")   does protection add on top of the closure?
  C6  v35_P vs v35_QP decide(gate=1.00, interval="t")   does the closure add on top of protection?

BASE FOR THE COST WORK (step 3), registered rule, first match wins:
  1. C2 passes and C5 passes                -> v35_QP   ("base advanced, combined")
  2. C1 passes                              -> v35_Q    ("base advanced, closure")
  3. C2 passes                              -> v35_QP   ("base advanced, combined; protection not separately supported")
  4. C3 passes                              -> v35_P    ("no base advancement; secondary supported gain")
  5. otherwise                              -> v32      ("no supported change")
R (C4) can never be selected: it changes only the final readout of v32's trajectory
and exists to attribute. Two 1.15 gates are tested on the same control; this is
DESIGN tier (P11 is exposed and reused), and any release claim needs fresh DECIDE
evidence, so the multiplicity is recorded rather than corrected.

Run: uv run python local_sim/s6_decide.py --score experiments/2026-09-17_step2/score.json
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
ARMS = {"v32": "estimator_k3_v32.py", "R": "estimator_k3_v35_R.py", "Q": "estimator_k3_v35_Q.py",
        "P": "estimator_k3_v35_P.py", "QP": "estimator_k3_v35_QP.py"}
COMPARISONS = [("C1", "v32", "Q", 1.15), ("C2", "v32", "QP", 1.15), ("C3", "v32", "P", 1.00),
               ("C4", "v32", "R", 1.00), ("C5", "Q", "QP", 1.00), ("C6", "P", "QP", 1.00)]


def arm_rows(score):
    by = {}
    for key, arm in score["arms"].items():
        base = os.path.basename(arm["path"])
        name = next((k for k, v in ARMS.items() if v == base), None)
        if name is None:
            continue
        rows = sorted(arm["rows"], key=lambda r: int(r["seed"]))
        if any(r["failed"] or r["nonfinite"] for r in rows):
            raise SystemExit(f"{name}: failed or non-finite rows; the registered verdict requires 0 failures")
        by[name] = {"seeds": [int(r["seed"]) for r in rows],
                    "score": np.array([float(r["mse"]) * max(0.1, float(r["flops_used"]) / B) for r in rows]),
                    "mse": np.array([float(r["mse"]) for r in rows]),
                    "gt": np.array([float(r.get("gt_noise", float("nan"))) for r in rows]),
                    "flops": np.array([float(r["flops_used"]) for r in rows]),
                    "sha256": arm.get("candidate_sha256")}
    missing = set(ARMS) - set(by)
    if missing:
        raise SystemExit(f"arms missing from the artifact: {sorted(missing)}")
    seeds = by["v32"]["seeds"]
    for k, v in by.items():
        if v["seeds"] != seeds:
            raise SystemExit(f"{k}: networks differ from the control's")
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    import competition_score as cs
    with open(a.score, encoding="utf-8") as fh:
        score = json.load(fh)
    by = arm_rows(score)
    out = {"artifact": "s6_decide", "score_artifact": a.score, "tier": "DESIGN (P11, exposed, reused)",
           "arms": {k: {"sha256": v["sha256"], "raw_mse": float(v["mse"].mean()),
                        "c_over_b": float(v["flops"].mean() / B), "score": float(v["score"].mean())}
                    for k, v in by.items()},
           "comparisons": {}}
    passed = {}
    for tag, ctl, cand, gate in COMPARISONS:
        v = cs.decide(by[ctl]["score"], by[cand]["score"], gate=gate, interval="t")
        passed[tag] = bool(v.passed)
        raw_gain = float(by[ctl]["mse"].mean() / by[cand]["mse"].mean())
        out["comparisons"][tag] = {"control": ctl, "candidate": cand, "gate": gate, "passed": v.passed,
                                   "gain": v.gain, "t95_diff": list(v.ci), "raw_gain": raw_gain,
                                   "geometric_gain": v.geometric_gain,
                                   "n_candidate_better": v.n_candidate_better, "n": v.n,
                                   "reasons": list(v.reasons)}
        print(f"{tag} {ctl:>4} vs {cand:<3} gate {gate:.2f}  {'PASS' if v.passed else 'fail'}  "
              f"gain {v.gain:.4f}x  raw {raw_gain:.4f}x  t95 [{v.ci[0]:+.3e}, {v.ci[1]:+.3e}]  "
              f"better {v.n_candidate_better}/{v.n}")
    if passed["C2"] and passed["C5"]:
        base, why = "QP", "base advanced, combined"
    elif passed["C1"]:
        base, why = "Q", "base advanced, closure"
    elif passed["C2"]:
        base, why = "QP", "base advanced, combined; protection not separately supported"
    elif passed["C3"]:
        base, why = "P", "no base advancement; secondary supported gain"
    else:
        base, why = "v32", "no supported change"
    out["base_for_cost_work"] = {"arm": base, "file": ARMS[base], "rule": why}
    print(f"\nBASE FOR STEP 3: {base} ({ARMS[base]}) -- {why}")
    for k, v in out["arms"].items():
        print(f"  {k:<4} raw {v['raw_mse']:.4e}  C/B {v['c_over_b']:.5f}  score {v['score']:.5e}")
    if a.out:
        with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(out, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
