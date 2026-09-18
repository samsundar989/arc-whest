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

Historical jobs 1658/1659/1660 did not record E or retained query energy and
cannot complete S3 or the cheaper age2 selector. Supply fresh label-free
artifacts from the repaired producer; old artifacts remain valid diagnostics.

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
import dataclasses
import hashlib
import json
import math
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
    """Read measured displacement; absent contraction evidence stays absent."""
    d = json.load(open(path, encoding="utf-8"))
    depth = int(d.get("network", {}).get("depth", 0))
    if depth < 3:
        raise ValueError("query artifact needs a complete network depth >= 3")
    expected = {"E": list(range(1, depth - 1)),
                "D21": list(range(1, depth - 1)),
                "D3": list(range(1, depth)),
                "mu_out": list(range(depth)),
                "cov_out": list(range(depth - 1))}
    def collect(a):
        metrics = {"final_row": a.get("final_row_rel_vs_uncompressed"),
                   "all_rows": a.get("all_rows_rel_vs_uncompressed"),
                   "c_over_b": a.get("c_over_b"),
                   "retained_query_energy": a.get("retained_query_energy"),
                   "path": a.get("path"), "sha256": a.get("sha256")}
        rows = a.get("per_layer", [])
        metrics["layers"] = [r.get("layer") for r in rows]
        metrics["metric_layers"] = {}
        metrics["missing_layers"] = []
        if metrics["layers"] != list(range(depth)):
            metrics["missing_layers"].append("per_layer")
        for key in ("E", "D3", "D21", "mu_out", "cov_out"):
            present = [(r.get("layer"), r[key]) for r in rows if r.get(key) is not None]
            vals = [v for _, v in present]
            metrics["metric_layers"][key] = [l for l, _ in present]
            if metrics["metric_layers"][key] != expected[key]:
                metrics["missing_layers"].append(key)
            metrics[key] = (float(np.mean(vals)) if vals and all(
                math.isfinite(float(v)) for v in vals) else None)
        return metrics
    return {t: collect(a) for t, a in d["arms"].items()}, d.get("overrides", {})


def _qe_network(path):
    return json.load(open(path, encoding="utf-8")).get("network", {})


def _check_query_binding(qe, by):
    for key, scored in by.items():
        measured = qe.get(key)
        if measured is None:
            raise ValueError(f"query evidence lacks arm {key}")
        path = scored["path"].replace("\\", "/")
        measured_path = (measured["path"] or "").replace("\\", "/")
        if path != NAMES[key] or measured_path != path:
            raise ValueError(f"query evidence path differs for {key}")
        source = os.path.join(_ROOT, NAMES[key])
        actual = hashlib.sha256(open(source, "rb").read()).hexdigest()
        if not scored["sha256"] or scored["sha256"] != measured["sha256"] or actual != scored["sha256"]:
            raise ValueError(f"query evidence source SHA differs for {key}")


def _rows(arm, cs):
    rows = arm["rows"]
    if not rows:
        raise ValueError("empty arm")
    parsed = []
    for raw in rows:
        if "seed" not in raw or "index" not in raw:
            raise ValueError("network seed and index required on every row")
        seed, index = int(raw["seed"]), int(raw["index"])
        if index < 0:
            raise ValueError("negative network index")
        mse, flops, noise = (float(raw[k]) for k in ("mse", "flops_used", "gt_noise"))
        if not all(map(math.isfinite, (mse, flops, noise))) or min(mse, flops, noise) < 0:
            raise ValueError("invalid score, cost, or noise floor")
        failed = bool(raw["failed"] or raw["nonfinite"] or raw.get("error_code"))
        parsed.append(cs.MlpRow(mse=mse, flops_used=flops,
                                residual_wall_time_s=float(raw.get("residual_wall_time_s", 0)),
                                budget=B, failed=failed,
                                name=f"seed:{seed}", gt_noise=noise))
    indices = [int(r["index"]) for r in rows]
    if sorted(indices) != list(range(len(rows))):
        raise ValueError("arm has missing or duplicate network indices")
    return parsed


def _scores(rows, local_samples, cs):
    observed = cs.score_suite(rows, lam=0).scores
    official = cs.score_suite([
        dataclasses.replace(r, mse=r.mse - r.gt_noise +
                            r.gt_noise * local_samples / 1e9) for r in rows], lam=0).scores
    return observed, official


def _query_gate(qe, cand):
    control, candidate = qe.get("qoff"), qe.get(cand)
    keys = ("E", "D3", "D21", "mu_out", "cov_out", "final_row")
    if control is None or candidate is None:
        return False, {"missing": ["arm"]}
    missing = [key for key in keys if control.get(key) is None or candidate.get(key) is None]
    missing.extend(control["missing_layers"] + candidate["missing_layers"])
    if not control["layers"] or control["layers"] != candidate["layers"]:
        missing.append("paired complete layers")
    for key in keys[:-1]:
        if control["metric_layers"].get(key) != candidate["metric_layers"].get(key):
            missing.append(f"{key} paired layers")
    detail = {key: [control.get(key), candidate.get(key)] for key in keys}
    detail["missing"] = missing
    return not missing and all(candidate[k] < control[k] for k in keys), detail


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
    if int(d.get("width", 0)) != 1024 or int(d.get("depth", 0)) != 16:
        raise ValueError("stage-3 score artifact must explicitly identify live 1024x16 shape")
    local_samples = 2 * int(d["n_samples_per_replica"])
    if local_samples <= 0:
        raise ValueError("positive truth sample count required")
    by = {}
    for arm in d["arms"].values():
        base = os.path.basename(arm["path"])
        key = next((k for k, v in NAMES.items() if v == base), None)
        if key is None:
            continue
        if key in by:
            raise ValueError(f"duplicate arm {key}")
        rows = _rows(arm, cs)
        by[key] = {"rows": rows, "sha256": arm.get("candidate_sha256"),
                   "path": arm["path"]}
    missing = sorted(set(NAMES) - set(by))
    if missing:
        raise SystemExit(f"arms missing from {a.score}: {missing}")

    for cand in ("qE", "qEd"):
        control_rows, candidate_rows, _ = cs.align_rows_by_identity(
            by["qoff"]["rows"], by[cand]["rows"], require_ids=True)
        # Row order may differ, but a shared panel's index-to-seed mapping may not.
        original = next(v["rows"] for v in d["arms"].values()
                        if os.path.basename(v["path"]) == NAMES["qoff"])
        other = next(v["rows"] for v in d["arms"].values()
                     if os.path.basename(v["path"]) == NAMES[cand])
        if {int(r["index"]): int(r["seed"]) for r in original} != {
                int(r["index"]): int(r["seed"]) for r in other}:
            raise ValueError(f"{cand} index-to-seed map differs from qoff")
        by["qoff"]["rows"] = control_rows
        by[cand]["rows"] = candidate_rows
        cs.noise_corrected_compare(control_rows, candidate_rows,
                                   local_samples=local_samples, require_ids=True)
    for arm in by.values():
        arm["observed"], arm["official"] = _scores(arm["rows"], local_samples, cs)

    qe4, overrides4 = _qe(a.qe_a4)
    if overrides4 != {"age": 4, "rank": 256}:
        raise ValueError("age4 query artifact has wrong override setting")
    _check_query_binding(qe4, by)
    qe_network = _qe_network(a.qe_a4)
    for dim in ("width", "depth"):
        if int(qe_network[dim]) != int(d[dim]):
            raise ValueError(f"query artifact {dim} differs from score artifact")
    out = {"artifact": "s12_decide", "version": 2, "tier": "DESIGN (P12)",
           "contract": "experiments/LOG.md 2026-09-17b stage 3",
           "gate": GATE,
           "score_artifact": os.path.relpath(a.score, _ROOT).replace("\\", "/"),
           "query_error_artifact": os.path.relpath(a.qe_a4, _ROOT).replace("\\", "/"),
           "arms": {k: {"path": v["path"], "sha256": v["sha256"],
                        "raw": float(np.mean([r.mse for r in v["rows"]])),
                        "c_over_b": float(np.mean([r.flops_used for r in v["rows"]]) / B),
                        "observed_score": float(v["observed"].mean()),
                        "official_precision_score": float(v["official"].mean()),
                        "n_failed": sum(r.failed for r in v["rows"]),
                        "query_error": qe4.get(k)} for k, v in by.items()},
           "comparisons": {}}

    verdicts = {}
    for cand in ("qE", "qEd"):
        v = cs.decide(by["qoff"]["official"], by[cand]["official"], gate=GATE, interval="t")
        observed = cs.paired_compare(by["qoff"]["observed"], by[cand]["observed"])
        q_ok, q_detail = _query_gate(qe4, cand)
        old_fail = {r.name for r in by["qoff"]["rows"] if r.failed}
        new_fail = {r.name for r in by[cand]["rows"] if r.failed} - old_fail
        f_ok = not new_fail
        passed = bool(v.passed and q_ok and f_ok)
        verdicts[cand] = {
            "S1_adjusted_decide": bool(v.passed), "S1_gain": float(v.gain),
            "S1_t95": [float(x) for x in v.ci],
            "observed_gain": float(observed.arithmetic_gain),
            "observed_t95": [float(x) for x in observed.t_ci_difference],
            "S2_no_new_failures": bool(f_ok),
            "S2_new_failed_identities": sorted(new_fail),
            "S3_query_error_reduced": bool(q_ok),
            "S3_detail": q_detail,
            "SCREEN_PASSED": passed,
        }
        print(f"  {cand:<4} adjusted {v.gain:.4f}x t95 [{v.ci[0]:+.2e}, {v.ci[1]:+.2e}] "
              f"{'PASS' if v.passed else 'fail'}   query-error reduced {q_ok}   "
              f"no new failures {f_ok}   -> SCREEN {'PASSED' if passed else 'FAILED'}")
    out["comparisons"] = verdicts

    # the cheaper-setting branch selector, registered before the measurement
    branch = None
    if a.qe_a3 and a.qe_a2:
        if _qe_network(a.qe_a3) != qe_network or _qe_network(a.qe_a2) != qe_network:
            raise ValueError("cheaper query artifacts use different networks")
        q3, o3 = _qe(a.qe_a3)
        q2, o2 = _qe(a.qe_a2)
        if o3 != {"age": 3, "rank": 192} or o2 != {"age": 2, "rank": 192}:
            raise ValueError("cheaper query artifacts have wrong age/rank settings")
        _check_query_binding(q3, by)
        _check_query_binding(q2, by)
        arm = next((k for k in ("qE", "qEd") if verdicts[k]["SCREEN_PASSED"]), None)
        e3 = q3.get(arm, {}).get("retained_query_energy") if arm else None
        e2 = q2.get(arm, {}).get("retained_query_energy") if arm else None
        pick = ("age2/rank192" if e2 >= e3 else "age3/rank192") if (
            e2 is not None and e3 is not None) else "age3/rank192 (default; energy unmeasured)"
        branch = {"measured_on": arm, "retained_a3r192": e3, "retained_a2r192": e2,
                  "displacement_a3r192": q3.get(arm, {}).get("all_rows") if arm else None,
                  "displacement_a2r192": q2.get(arm, {}).get("all_rows") if arm else None,
                  "c_over_b_a3r192": q3.get(arm, {}).get("c_over_b") if arm else None,
                  "c_over_b_a2r192": q2.get(arm, {}).get("c_over_b") if arm else None,
                  "selected": pick,
                  "eligible": bool(arm),
                  "energy_evidence_complete": e2 is not None and e3 is not None}
        print(f"\n  cheaper-setting criterion on {arm}: retained a3r192 {e3} "
              f"vs a2r192 {e2}  ->  {pick}"
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
