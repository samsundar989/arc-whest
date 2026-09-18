"""Read-only September17b P12 synthesis from the two completed score JSONs.

All score arithmetic and intervals delegate to local_sim.competition_score.
This script does not run an estimator, choose a new gate, or read truth arrays.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "local_sim"))
from local_sim import competition_score as cs


S1 = ROOT / "experiments/2026-09-17b_stage1/score.json"
SB = ROOT / "experiments/2026-09-17b_stage3/score.json"
OUT = ROOT / "experiments/2026-09-17b_stage3/synthesis.json"
BUDGET = 2**41
LOCAL_SAMPLES = 2**25


def load(path: Path) -> dict:
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["block"] == "P12" and doc["n"] == 24
    assert (doc["width"], doc["depth"]) == (1024, 16)
    assert int(doc["n_samples_per_replica"]) * 2 == LOCAL_SAMPLES
    for arm in doc["arms"].values():
        source = ROOT / arm["path"]
        assert source.is_file(), source
        assert hashlib.sha256(source.read_bytes()).hexdigest() == arm["candidate_sha256"]
        rows = arm["rows"]
        assert len(rows) == doc["n"]
        assert sorted(int(r["index"]) for r in rows) == list(range(doc["n"]))
    return doc


def mlp_rows(arm: dict) -> list[cs.MlpRow]:
    out = []
    for r in sorted(arm["rows"], key=lambda x: int(x["index"])):
        failed = bool(r["failed"] or r["nonfinite"] or r.get("error_code"))
        out.append(cs.MlpRow(
            mse=float(r["mse"]), flops_used=float(r["flops_used"]),
            budget=BUDGET, failed=failed, name=f"seed:{int(r['seed'])}",
            gt_noise=float(r["gt_noise"]),
        ))
    return out


def pair(a: dict, b: dict, gate: float | None = None) -> dict:
    ra, rb = mlp_rows(a), mlp_rows(b)
    ra, rb, _ = cs.align_rows_by_identity(ra, rb, require_ids=True)
    # The panel has a fixed index-to-network map, including across the two jobs.
    ia = {int(r["index"]): int(r["seed"]) for r in a["rows"]}
    ib = {int(r["index"]): int(r["seed"]) for r in b["rows"]}
    assert ia == ib
    assert all(x.gt_noise == y.gt_noise for x, y in zip(ra, rb))
    raw = cs.paired_compare([x.mse for x in ra], [x.mse for x in rb])
    noise = cs.noise_corrected_compare(
        ra, rb, local_samples=LOCAL_SAMPLES, lam=0, require_ids=True)

    def fields(x: cs.PairedResult) -> dict:
        return {
            "control_mean": x.mean_control,
            "candidate_mean": x.mean_candidate,
            "difference_control_minus_candidate": x.mean_difference,
            "t95_difference": list(x.t_ci_difference),
            "arithmetic_gain": x.arithmetic_gain,
            "geometric_diagnostic": x.geometric_gain,
            "n_candidate_better": x.n_candidate_better,
        }

    result = {
        "control": a["path"], "candidate": b["path"],
        "raw_mse": fields(raw),
        "observed_adjusted": fields(noise.observed),
        "true_function_adjusted": fields(noise.true_function),
        "official_precision_adjusted": fields(noise.forecast),
        "observed_noise_bias_difference": noise.noise_bias,
        "mean_truth_noise_floor": sum(x.gt_noise for x in ra) / len(ra),
        "new_failed_identities": sorted(
            {x.name for x in rb if x.failed} - {x.name for x in ra if x.failed}),
    }
    if gate is not None:
        official_rows = [
            dataclasses.replace(r, mse=r.mse - r.gt_noise +
                                r.gt_noise * LOCAL_SAMPLES / 1e9)
            for r in ra
        ]
        candidate_rows = [
            dataclasses.replace(r, mse=r.mse - r.gt_noise +
                                r.gt_noise * LOCAL_SAMPLES / 1e9)
            for r in rb
        ]
        v = cs.decide(cs.score_suite(official_rows, lam=0).scores,
                      cs.score_suite(candidate_rows, lam=0).scores,
                      gate=gate, interval="t")
        result["registered_official_score_gate"] = {
            "gate": gate, "passed": bool(v.passed),
            "gain": v.gain, "t95_difference": list(v.ci),
            "reasons": list(v.reasons),
        }
    return result


def arm_summary(arm: dict) -> dict:
    rows = mlp_rows(arm)
    official = [dataclasses.replace(
        r, mse=r.mse - r.gt_noise + r.gt_noise * LOCAL_SAMPLES / 1e9)
        for r in rows]
    obs = cs.score_suite(rows, lam=0)
    forecast = cs.score_suite(official, lam=0)
    return {
        "path": arm["path"], "sha256": arm["candidate_sha256"],
        "observed_raw_mse": sum(r.mse for r in rows) / len(rows),
        "observed_adjusted": obs.objective,
        "official_precision_adjusted": forecast.objective,
        "mean_counted_flops": sum(r.flops_used for r in rows) / len(rows),
        "mean_cost_over_budget": sum(r.flops_used for r in rows) / len(rows) / BUDGET,
        "n_failed": sum(r.failed for r in rows),
        "noise_floor_mean": sum(r.gt_noise for r in rows) / len(rows),
    }


def main() -> None:
    s1, sb = load(S1), load(SB)
    assert s1["truth_sha256"] == sb["truth_sha256"]
    names = {
        "v32": s1["arms"]["control"],
        "chain3": s1["arms"]["cand1"],
        "pairq": s1["arms"]["cand6"],
        "v32_p4": s1["arms"]["cand7"],
        "qoff": sb["arms"]["cand0"],
        "qE": sb["arms"]["cand1"],
        "qEd": sb["arms"]["cand2"],
        "pairq_gram": sb["arms"]["cand3"],
        "pairq_qE": sb["arms"]["cand4"],
        "p3": sb["arms"]["cand5"],
        "pairq_p3": sb["arms"]["cand6"],
        "v32_lam504": sb["arms"]["cand7"],
        "chain3_lam504": sb["arms"]["cand8"],
    }
    assert sb["arms"]["control"]["candidate_sha256"] == names["v32_p4"]["candidate_sha256"]
    cases = [
        ("stage2_pairq", "v32_p4", "pairq", 1.10),
        ("stage2_pairq_p3", "v32_p4", "pairq_p3", 1.10),
        ("stage3_qE_score_only", "qoff", "qE", 1.0),
        ("stage3_qEd_score_only", "qoff", "qEd", 1.0),
        ("composed_storage", "pairq_gram", "pairq_qE", None),
        ("p3_isolated", "v32", "p3", None),
        ("p3_on_pairq", "pairq", "pairq_p3", None),
        ("lambda_on_v32", "v32", "v32_lam504", None),
        ("lambda_on_chain3", "chain3", "chain3_lam504", None),
        ("program_native_lambda", "v32", "chain3", None),
        ("program_ref_lambda", "v32_lam504", "chain3_lam504", None),
    ]
    output = {
        "artifact": "2026-09-17b P12 completed-batch synthesis",
        "tier": "DESIGN", "binding_release_verdict": False,
        "stage3_query_evidence_complete": False,
        "stage3_limitation": "Historical query artifacts lack E and true retained energy; score-only subgates cannot pass the complete S3 screen.",
        "truth_sha256": s1["truth_sha256"],
        "score_files": {str(S1.relative_to(ROOT)): hashlib.sha256(S1.read_bytes()).hexdigest(),
                        str(SB.relative_to(ROOT)): hashlib.sha256(SB.read_bytes()).hexdigest()},
        "arms": {name: arm_summary(arm) for name, arm in names.items()},
        "comparisons": {label: pair(names[a], names[b], gate)
                        for label, a, b, gate in cases},
    }
    OUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(OUT)
    for label, result in output["comparisons"].items():
        p = result["official_precision_adjusted"]
        print(f"{label:25s} {p['arithmetic_gain']:.4f}x "
              f"diff {p['difference_control_minus_candidate']:+.3e} "
              f"t95 [{p['t95_difference'][0]:+.3e},{p['t95_difference'][1]:+.3e}]")


if __name__ == "__main__":
    main()
