"""PRE-REGISTERED verdict for the 2026-09-17b reciprocal program-swap read. Written before data.

Contract: `experiments/LOG.md`, *CONTRACT (Rule 1): slice 2026-09-17b*. One
`p2_design_score.py` artifact, control frozen v32, candidates: native v17, chain 3
(`estimator_k3_v38_v17prog.py`), chain 3 readout-only, chain 4
(`_diag/estimator_v17_v32prog.py`), native v29.

Score per network = mse * max(0.1, flops / 2**41), paired on the INPUT SEED
(the network's identity), studentised-t interval. Phase 2 sets lambda = 0, so
C = F and the residual is not a cost.

STAGE 1 IS ATTRIBUTION, NOT PROMOTION. Nothing here promotes anything; the
verdicts below choose which ONE conditional branch the slice takes next.

  A1  v32 -> chain3     ATTRIBUTION gate: RAW gain >= 1.10x and t95 on the raw
                        difference strictly above zero. "Does the complete
                        reference program help OUR chain?"
  A2  v17 -> chain4     ATTRIBUTION gate: RAW gain <= 0.92x and t95 on the raw
                        difference strictly below zero. "Does deleting it cost
                        THEIR chain the same thing?"
  A3  reciprocity:      A1 and A2 both hold AND
                        |log(raw gain A1) + log(raw gain A2)| < 0.35.

  BRANCH (declared before data, exactly one is taken):
    A3 holds                      -> "(a) program explains the gap: build our own
                                     coherent implementation of those families"
    A2 holds and A1 fails         -> "(b) one 2x2 of program x coherent
                                     lambda/feed package"
    neither A1 nor A2 holds       -> "(c) H1 FALSIFIED: test the remaining
                                     source-approximation package reciprocally,
                                     choosing the branch from v29's measured
                                     gain beyond v17 on this panel"
    A1 holds and A2 fails         -> "(a') asymmetric: the program helps us but
                                     its absence does not hurt them -- proceed to
                                     stage 2 and record the asymmetry as an open
                                     question, do NOT open a second branch"

  Reported beside every comparison, never as a gate here:
    * ADJUSTED gain (raw x the cost multiplier) -- the economics;
    * chain3 vs v17 raw (prediction 4: chain3 stays >= 5% worse than v17);
    * readout-only share (prediction 5);
    * C/B of every arm.

Run:  uv run python local_sim/s9_decide.py --score <score.json> --out <verdict.json>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]

import numpy as np  # noqa: E402

B = 2 ** 41

NAMES = {
    "v32": "estimator_k3_v32.py",
    "v17": "estimator_v17.py",
    "chain3": "estimator_k3_v38_v17prog.py",
    "chain3_last": "estimator_k3_v38_v17prog_last.py",
    "chain4": "estimator_v17_v32prog.py",
    "v29": "estimator_v29.py",
}

A1_RAW_GATE = 1.10
A2_RAW_GATE = 0.92
A3_LOG_TOL = 0.35


def _t95(d):
    """Studentised-t 95% interval of the mean of `d`.

    THE t-quantile comes from `competition_score._t975`, the one implementation
    this repository has and the one `decide` itself uses -- a second copy of a
    quantile table is exactly how two numbers for one statistic appear.
    """
    import competition_score as cs
    d = np.asarray(d, dtype=np.float64)
    n = d.size
    if n < 2:
        return (float("nan"), float("nan"))
    m = float(np.mean(d))
    se = float(np.std(d, ddof=1)) / math.sqrt(n)
    tq = cs._t975(n - 1)
    return (m - tq * se, m + tq * se)


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
        n_failed = sum(1 for r in rows if r["failed"] or r["nonfinite"])
        by[key] = {
            "seeds": [int(r["seed"]) for r in rows],
            "sha256": arm.get("candidate_sha256"),
            "path": arm["path"],
            "n_failed": n_failed,
            "score": np.array([float(r["mse"]) * max(0.1, float(r["flops_used"]) / B)
                               for r in rows]),
            "mse": np.array([float(r["mse"]) for r in rows]),
            "flops": np.array([float(r["flops_used"]) for r in rows]),
        }
    missing = sorted(set(NAMES) - set(by))
    if missing:
        raise SystemExit(f"arms missing from {a.score}: {missing}")
    for k in by:
        if by[k]["seeds"] != by["v32"]["seeds"]:
            raise SystemExit(f"{k}: different networks from the control")
        if by[k]["n_failed"]:
            print(f"  *** {k}: {by[k]['n_failed']} FAILED rows -- read the flags "
                  "before trusting any number below")

    out = {"artifact": "s9_decide", "version": 1,
           "tier": "DESIGN (P12, fresh panel)",
           "contract": "experiments/LOG.md 2026-09-17b",
           "score_artifact": os.path.relpath(a.score, _ROOT).replace("\\", "/"),
           "gates": {"A1_raw": A1_RAW_GATE, "A2_raw": A2_RAW_GATE,
                     "A3_log_tol": A3_LOG_TOL},
           "arms": {k: {"path": v["path"], "sha256": v["sha256"],
                        "raw": float(v["mse"].mean()),
                        "c_over_b": float(v["flops"].mean() / B),
                        "score": float(v["score"].mean()),
                        "n_failed": v["n_failed"]}
                    for k, v in by.items()},
           "comparisons": {}}

    def cmp(tag, c, x):
        draw = by[c]["mse"] - by[x]["mse"]
        ci_raw = _t95(draw)
        v = cs.decide(by[c]["score"], by[x]["score"], gate=1.0, interval="t")
        rec = {"control": c, "candidate": x,
               "raw_gain": float(by[c]["mse"].mean() / by[x]["mse"].mean()),
               "raw_t95": [float(ci_raw[0]), float(ci_raw[1])],
               "raw_better": int((draw > 0).sum()),
               "adjusted_gain": float(v.gain),
               "adjusted_t95": [float(v.ci[0]), float(v.ci[1])],
               "adjusted_better": int(v.n_candidate_better), "n": int(v.n)}
        out["comparisons"][tag] = rec
        print(f"{tag} {c:>7} -> {x:<12} raw {rec['raw_gain']:.4f}x "
              f"t95 [{ci_raw[0]:+.3e}, {ci_raw[1]:+.3e}] {rec['raw_better']}/{rec['n']}"
              f"   adjusted {v.gain:.4f}x t95 [{v.ci[0]:+.2e}, {v.ci[1]:+.2e}]")
        return rec

    a1 = cmp("A1", "v32", "chain3")
    a2 = cmp("A2", "v17", "chain4")
    # control=chain3 so this reads "how much better v17 is than chain 3"
    cmp("D_chain3_vs_v17", "chain3", "v17")
    cmp("D_chain3_last", "v32", "chain3_last")
    cmp("D_v32_vs_v17", "v32", "v17")
    cmp("D_v32_vs_v29", "v32", "v29")
    cmp("D_v17_vs_v29", "v17", "v29")

    a1_pass = bool(a1["raw_gain"] >= A1_RAW_GATE and a1["raw_t95"][0] > 0.0)
    a2_pass = bool(a2["raw_gain"] <= A2_RAW_GATE and a2["raw_t95"][1] < 0.0)
    recip = abs(math.log(a1["raw_gain"]) + math.log(a2["raw_gain"]))
    a3_pass = bool(a1_pass and a2_pass and recip < A3_LOG_TOL)

    if a3_pass:
        branch = ("(a) program explains the gap: build our own coherent "
                  "implementation of the implicated families")
    elif a2_pass and not a1_pass:
        branch = ("(b) one preregistered 2x2 of program x coherent lambda/feed "
                  "package, backend/rank/storage fixed")
    elif a1_pass and not a2_pass:
        branch = ("(a') asymmetric: the program helps our chain but its absence "
                  "does not cost theirs -- proceed to stage 2 and record the "
                  "asymmetry as an open question; do NOT open a second branch")
    else:
        branch = ("(c) H1 FALSIFIED on both directions: test the remaining "
                  "source-approximation package (sketch rank, protected feed) "
                  "reciprocally, choosing the ONE branch from v29's measured "
                  "gain beyond v17 on this panel")

    # prediction checks, reported not gated
    # DIRECTION. `cmp(tag, c, x)` reports control/candidate, so
    # `D_chain3_vs_v17` with control=chain3 is v17_better_by = v17 over chain3:
    # a value ABOVE 1 means v17 is ahead. Prediction 4 says chain 3 stays at
    # least 5% WORSE than native v17, which is therefore `>= 1.05`, not `<= 0.95`.
    # Caught by a synthetic dry-run of this module before the real artifact
    # existed; the A1/A2/A3 GATES are untouched and this is a reported
    # diagnostic, not a threshold.
    p4 = out["comparisons"]["D_chain3_vs_v17"]["raw_gain"]
    last_share = (abs(math.log(out["comparisons"]["D_chain3_last"]["raw_gain"]))
                  / max(abs(math.log(a1["raw_gain"])), 1e-12))
    out["verdict"] = {
        "A1_passed": a1_pass, "A2_passed": a2_pass, "A3_passed": a3_pass,
        "reciprocity_log_gap": float(recip), "branch": branch,
        "H1_falsified": bool(not a1_pass and not a2_pass),
        "prediction_4_chain3_at_least_5pct_worse_than_v17": bool(p4 >= 1.05),
        "chain3_over_v17_raw": float(1.0 / p4),
        "prediction_5_readout_share": float(last_share),
    }
    print()
    print(f"  A1 chain3 raw >= {A1_RAW_GATE}: {'PASS' if a1_pass else 'fail'}")
    print(f"  A2 chain4 raw <= {A2_RAW_GATE}: {'PASS' if a2_pass else 'fail'}")
    print(f"  A3 reciprocity |log sum| = {recip:.4f} < {A3_LOG_TOL}: "
          f"{'PASS' if a3_pass else 'fail'}")
    print(f"  BRANCH: {branch}")
    print(f"  prediction 4 (chain3 stays >=5% worse than v17 raw): "
          f"{out['verdict']['prediction_4_chain3_at_least_5pct_worse_than_v17']} "
          f"(v17 is {p4:.4f}x better; chain3/v17 raw {1.0 / p4:.4f}x)")
    print(f"  prediction 5 (readout share < 0.3): {last_share:.3f}")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(out, fh, indent=1)
        print("\n  ->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
