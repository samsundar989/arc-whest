"""PRE-REGISTERED verdict for the 2026-09-17b stage-3 consumed-query basis screen.

Contract: `experiments/LOG.md`, *CONTRACT (Rule 1): slice 2026-09-17b*, stage-3
screen. Control is the SAME CHAIN spelled with the old diagonal-surrogate
objective (`..._qoff.py`, proved bitwise identical to frozen
`estimator_k3_v37_a4r256_p4d5.py`); candidates are `qE` and the preregistered
`qEd`. Age 4, rank 256, power passes, initialiser, P4, closure and lambda are
identical in all three, which is what makes the comparison isolate the objective.

THE SCREEN AS REGISTERED, all three conditions, none added afterwards:

  S1  lower OFFICIAL-PRECISION adjusted score with `decide(gate=1.0)`;
  S2  no new failures relative to the control;
  S3  reduced ACTUAL post-projection query error -- the full recomputed E, D3
      and D21 after projection plus the whole next-layer and output
      displacement, measured label-free by `s10_parity.py --query-error` at
      1024x16 BEFORE this screen ran.

S3 is already in hand (jobs 1658/1659/1660) and its artifacts are named on the
command line so the verdict carries them rather than asserting them.

THE CHEAPER-SETTING BRANCH, also registered before the measurement:
  if the screen passes, ONE cheaper setting is allowed -- default age3/rank192,
  or age2/rank192 ONLY IF the label-free criterion
      retained_query_energy(age2, r192) >= retained_query_energy(age3, r192)
  selects it. `--cheap-a3` / `--cheap-a2` take the two query-error artifacts and
  evaluate that criterion here rather than by eye.

Run:  uv run python local_sim/s12_decide.py --score <score.json> \\
          --qe-a4 <query_error_1024.json> --qe-a3 <..._a3r192.json> \\
          --qe-a2 <..._a2r192.json> --out <verdict.json>
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
GATE = 1.0

NAMES = {"qoff": "estimator_k3_v39_v37_a4r256_p4d5_qoff.py",
         "qE": "estimator_k3_v39_v37_a4r256_p4d5_qE.py",
         "qEd": "estimator_k3_v39_v37_a4r256_p4d5_qEd.py"}


def _qe(path):
    """(arm -> final-row displacement vs uncompressed) from a query-error artifact."""
    d = json.load(open(path, encoding="utf-8"))
    return {t: {"final_row": a["final_row_rel_vs_uncompressed"],
                "all_rows": a["all_rows_rel_vs_uncompressed"],
                "c_over_b": a["c_over_b"],
                "D3": float(np.mean([r["D3"] for r in a["per_layer"]
                                     if r.get("D3") is not None])),
                "D21": float(np.mean([r["D21"] for r in a["per_layer"]
                                      if r.get("D21") is not None]))}
            for t, a in d["arms"].items()}, d.get("overrides", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", required=True)
    ap.add_argument("--qe-a4", required=True)
    ap.add_argument("--qe-a3", default="")
    ap.add_argument("--qe-a2", default="")
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
        by[key] = {"seeds": [int(r["seed"]) for r in rows],
                   "sha256": arm.get("candidate_sha256"), "path": arm["path"],
                   "n_failed": sum(1 for r in rows if r["failed"] or r["nonfinite"]),
                   "score": np.array([float(r["mse"]) * max(0.1, float(r["flops_used"]) / B)
                                      for r in rows]),
                   "mse": np.array([float(r["mse"]) for r in rows]),
                   "flops": np.array([float(r["flops_used"]) for r in rows])}
    missing = sorted(set(NAMES) - set(by))
    if missing:
        raise SystemExit(f"arms missing from {a.score}: {missing}")

    # the OFFICIAL-PRECISION adjusted estimand, which is the registered selector
    comp = d.get("comparisons", {})
    official = {}
    for lab, rec in comp.items():
        na = rec.get("noise_aware") or {}
        path = os.path.basename(d["arms"][lab]["path"])
        key = next((k for k, v in NAMES.items() if v == path), lab)
        official[key] = {"observed_gain": rec["arithmetic_gain"],
                         "observed_t95": rec["t_ci"],
                         "true_function_gain": na.get("true_function_gain"),
                         "forecast_gain": na.get("forecast_gain"),
                         "noise_bias_share": na.get("noise_bias_share"),
                         "n_negative_corrected": na.get("n_negative_corrected")}

    qe4, _ = _qe(a.qe_a4)
    out = {"artifact": "s12_decide", "version": 1, "tier": "DESIGN (P12)",
           "contract": "experiments/LOG.md 2026-09-17b stage 3",
           "gate": GATE,
           "score_artifact": os.path.relpath(a.score, _ROOT).replace("\\", "/"),
           "query_error_artifact": os.path.relpath(a.qe_a4, _ROOT).replace("\\", "/"),
           "arms": {k: {"path": v["path"], "sha256": v["sha256"],
                        "raw": float(v["mse"].mean()),
                        "c_over_b": float(v["flops"].mean() / B),
                        "score": float(v["score"].mean()),
                        "n_failed": v["n_failed"],
                        "query_error": qe4.get(k)} for k, v in by.items()},
           "official_precision": official, "comparisons": {}}

    verdicts = {}
    for cand in ("qE", "qEd"):
        v = cs.decide(by["qoff"]["score"], by[cand]["score"], gate=GATE, interval="t")
        q_ok = (qe4[cand]["final_row"] < qe4["qoff"]["final_row"]
                and qe4[cand]["D21"] < qe4["qoff"]["D21"]
                and qe4[cand]["D3"] < qe4["qoff"]["D3"])
        f_ok = by[cand]["n_failed"] <= by["qoff"]["n_failed"]
        passed = bool(v.passed and q_ok and f_ok)
        verdicts[cand] = {
            "S1_adjusted_decide": bool(v.passed), "S1_gain": float(v.gain),
            "S1_t95": [float(x) for x in v.ci],
            "S2_no_new_failures": bool(f_ok),
            "S3_query_error_reduced": bool(q_ok),
            "S3_detail": {"final_row": [qe4["qoff"]["final_row"], qe4[cand]["final_row"]],
                          "D3": [qe4["qoff"]["D3"], qe4[cand]["D3"]],
                          "D21": [qe4["qoff"]["D21"], qe4[cand]["D21"]]},
            "SCREEN_PASSED": passed,
        }
        print(f"  {cand:<4} adjusted {v.gain:.4f}x t95 [{v.ci[0]:+.2e}, {v.ci[1]:+.2e}] "
              f"{'PASS' if v.passed else 'fail'}   query-error reduced {q_ok}   "
              f"no new failures {f_ok}   -> SCREEN {'PASSED' if passed else 'FAILED'}")
    out["comparisons"] = verdicts

    # the cheaper-setting branch selector, registered before the measurement
    branch = None
    if a.qe_a3 and a.qe_a2:
        q3, o3 = _qe(a.qe_a3)
        q2, o2 = _qe(a.qe_a2)
        arm = "qE" if verdicts["qE"]["SCREEN_PASSED"] else "qoff"
        # retained query energy, as registered: 1 - the relative displacement of
        # the consumed contraction. Higher is more retained.
        e3 = 1.0 - q3[arm]["all_rows"]
        e2 = 1.0 - q2[arm]["all_rows"]
        pick = "age2/rank192" if e2 >= e3 else "age3/rank192"
        branch = {"measured_on": arm, "retained_a3r192": e3, "retained_a2r192": e2,
                  "displacement_a3r192": q3[arm]["all_rows"],
                  "displacement_a2r192": q2[arm]["all_rows"],
                  "c_over_b_a3r192": q3[arm]["c_over_b"],
                  "c_over_b_a2r192": q2[arm]["c_over_b"],
                  "selected": pick,
                  "eligible": bool(verdicts["qE"]["SCREEN_PASSED"]
                                   or verdicts["qEd"]["SCREEN_PASSED"])}
        print(f"\n  cheaper-setting criterion on {arm}: retained a3r192 {e3:.6f} "
              f"vs a2r192 {e2:.6f}  ->  {pick}"
              + ("" if branch["eligible"] else
                 "   (NOT ELIGIBLE: the fixed-rank screen did not pass, so no "
                 "cheaper setting is taken)"))
    out["cheaper_setting_branch"] = branch

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(out, fh, indent=1)
        print("\n  ->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
