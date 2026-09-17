"""PRE-REGISTERED verdict for the step-3 shared old-lineage accuracy screen. Written before data.

Contract: `experiments/LOG.md`, 2026-09-17, *CONTRACT (Rule 1): step 3 shared-coordinate accuracy screen*.
Artifact: `p2_design_score.py` on P11 (24 nets, caps off), control v32.

Per network score = mse * max(0.1, flops / 2**41) from the artifact rows, paired on seed.

REGISTERED:
  S-arms: v36_v32_a3r256, v36_v32_a2r256, v36_v32_a3r192, v36_v32_a2r192
    each: decide(v32, arm, gate=1.15, interval="t")          (base-advancement gate)
          raw_ratio = mean mse(arm) / mean mse(v32)            (intermediate target: <= 1.05)
          c_over_b  = mean flops / 2**41                        (intermediate target: <= 0.20)
  P4 parity arm v32_p4d5: raw_ratio in [0.99, 1.01] -> "P4 raw parity HOLDS";
          decide(v32, p4d5, gate=1.00) reported.

BASE FOR THE RELEASE BUNDLE, first match:
  1. arms passing the 1.15 gate AND raw_ratio <= 1.05 AND C/B <= 0.20 -> the lowest mean score among them
     ("intermediate target met")
  2. arms passing the 1.15 gate                                   -> the lowest mean score among them
     ("score gate met, intermediate target missed")
  3. otherwise                                                    -> v32 ("no supported shared configuration")
Four gates on one control at DESIGN tier: multiplicity recorded; the release claim needs fresh DECIDE.
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
CONTROL = "estimator_k3_v32.py"
SHARED = ["estimator_k3_v36_v32_a3r256.py", "estimator_k3_v36_v32_a2r256.py",
          "estimator_k3_v36_v32_a3r192.py", "estimator_k3_v36_v32_a2r192.py"]
P4 = "estimator_k3_v32_p4d5.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    import competition_score as cs
    d = json.load(open(a.score, encoding="utf-8"))
    by = {}
    for arm in d["arms"].values():
        rows = sorted(arm["rows"], key=lambda r: int(r["seed"]))
        if any(r["failed"] or r["nonfinite"] for r in rows):
            raise SystemExit(f"{arm['path']}: failed rows; the registered verdict requires 0 failures")
        by[os.path.basename(arm["path"])] = {
            "seeds": [int(r["seed"]) for r in rows],
            "score": np.array([float(r["mse"]) * max(0.1, float(r["flops_used"]) / B) for r in rows]),
            "mse": np.array([float(r["mse"]) for r in rows]),
            "flops": np.array([float(r["flops_used"]) for r in rows]),
            "sha256": arm.get("candidate_sha256")}
    for need in [CONTROL, P4] + SHARED:
        if need not in by:
            raise SystemExit(f"missing arm {need}")
        if by[need]["seeds"] != by[CONTROL]["seeds"]:
            raise SystemExit(f"{need}: different networks")
    ctl = by[CONTROL]
    out = {"artifact": "s7_decide", "tier": "DESIGN (P11, reused)", "arms": {}, "base": None}
    eligible1, eligible2 = [], []
    for name in SHARED + [P4]:
        v = by[name]
        gate = 1.00 if name == P4 else 1.15
        vd = cs.decide(ctl["score"], v["score"], gate=gate, interval="t")
        raw_ratio = float(v["mse"].mean() / ctl["mse"].mean())
        cb = float(v["flops"].mean() / B)
        rec = {"sha256": v["sha256"], "gate": gate, "passed": vd.passed, "gain": vd.gain,
               "t95": list(vd.ci), "better": vd.n_candidate_better, "raw_ratio": raw_ratio,
               "c_over_b": cb, "score": float(v["score"].mean()), "reasons": list(vd.reasons)}
        if name == P4:
            rec["p4_raw_parity_holds"] = bool(0.99 <= raw_ratio <= 1.01)
        else:
            rec["intermediate_target_met"] = bool(vd.passed and raw_ratio <= 1.05 and cb <= 0.20)
            if rec["intermediate_target_met"]:
                eligible1.append((rec["score"], name))
            if vd.passed:
                eligible2.append((rec["score"], name))
        out["arms"][name] = rec
        print(f"{name:<34} gate {gate:.2f} {'PASS' if vd.passed else 'fail'}  adj {vd.gain:.4f}x  "
              f"raw ratio {raw_ratio:.4f}  C/B {cb:.4f}  t95 [{vd.ci[0]:+.2e}, {vd.ci[1]:+.2e}]  "
              f"better {vd.n_candidate_better}/{vd.n}")
    if eligible1:
        out["base"] = {"file": min(eligible1)[1], "rule": "intermediate target met"}
    elif eligible2:
        out["base"] = {"file": min(eligible2)[1], "rule": "score gate met, intermediate target missed"}
    else:
        out["base"] = {"file": CONTROL, "rule": "no supported shared configuration"}
    print("\nBASE:", out["base"], "\nP4 raw parity holds:", out["arms"][P4]["p4_raw_parity_holds"],
          "\ncontrol raw %.4e C/B %.4f score %.4e" % (ctl["mse"].mean(), ctl["flops"].mean() / B, ctl["score"].mean()))
    if a.out:
        json.dump(out, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
