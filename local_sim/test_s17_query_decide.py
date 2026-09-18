"""Tiny synthetic query-reader checks; no P12 labels are opened."""
import copy
import dataclasses
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import competition_score as cs
import lifecycle_eval as le
import s17_query_decide as m


def fixture(tmp):
    exact = m.ROOT / "estimator_k3_v43_complete_c00.py"
    approx = m.ROOT / "estimator_k3_v43_complete_c10.py"
    paths = (exact, approx)
    seeds = list(m.p12.SEEDS)
    relpred = Path(tmp).relative_to(m.ROOT).as_posix()
    spec = dataclasses.asdict(le.LifecycleSpec(
        width=1024, depth=16, flop_budget=m.B, lam=0.,
        residual_wall_time_limit_s=None, wall_time_limit_s=None,
        predict_timeout_s=1e9, setup_timeout_s=1e9))
    man = {"run_complete": True, "tier": "DESIGN", "block": "P12",
           "n": 24, "width": 1024, "depth": 16,
           "truth": "experiments/2026-09-17b_p12/truth_P12.npz",
           "truth_sha256": m.p12.TRUTH_SHA,
           "n_samples_per_replica": 2**24, "truth_replicas": 2,
           "ordered_network_seeds": seeds, "control": str(exact),
           "candidate": str(approx), "config": {
               "accuracy_only": True, "local_seconds_caps": "off",
               "official_samples": 1e9, "gate": 1.05, "ts": "registered",
               "weights_mode": le.DEFAULT_WEIGHTS_MODE, "spec": spec}, "arms": {}}
    score = {k: man[k] for k in ("tier", "block", "n", "width", "depth", "truth",
                                "truth_sha256", "n_samples_per_replica", "control", "candidate")}
    score.update(accuracy_only=True, local_seconds_caps="off",
                 prediction_sidecars=relpred, arms={})
    loaded = {}
    for label, path in zip(m.LABELS, paths):
        sha = m._sha(path)
        # Observed candidate wins through its cheap multiplier; after label
        # precision conversion the same true error makes it lose.
        mse, flops = ((.011, .2*m.B) if label == "control" else (.020, .1*m.B))
        raw = [{"index": i, "seed": seed, "mse": mse,
                "flops_used": flops, "gt_noise": .01,
                "residual_wall_time_s": 0., "failed": False,
                "nonfinite": False, "error_code": None}
               for i, seed in enumerate(seeds)]
        full = [dataclasses.asdict(le.EvidenceRow(
            index=i, input_seed=seed, estimator_seed=1,
            candidate=str(path), candidate_sha256=sha,
            truth_sha256=m.p12.TRUTH_SHA, width=1024, depth=16,
            budget=m.B, flops_used=flops, n_ops=100, mse=mse,
            zero_prediction_mse=1., gt_noise=.01,
            residual_wall_time_s=0., wall_time_s=0.,
            error_code=None, nonfinite=False)) for i, seed in enumerate(seeds)]
        rh = le.rows_hash([le.EvidenceRow(**r) for r in full])
        score["arms"][label] = {"path": str(path), "candidate_sha256": sha,
                                "rows_hash": rh, "n_failed": 0,
                                "objective": cs.score_suite(m._scored_rows(raw, seeds), lam=0.).objective,
                                "rows": raw}
        man["arms"][label] = {"path": str(path), "candidate_sha256": sha,
                              "rows_hash": rh, "n_written": 24,
                              "evidence_rows": full}
        loaded[label] = {"seeds": seeds,
                         "rows": [{"failed": False} for _ in seeds]}
    return score, {"manifest": man, "arms": loaded}, paths


def main():
    with tempfile.TemporaryDirectory(dir=m.ROOT) as tmp:
        directory = Path(tmp)
        (directory / m.ds.MANIFEST).write_text("{}", encoding="utf-8")
        score, side, paths = fixture(tmp)
        score_path = directory / "score.json"

        def run(data, verified):
            score_path.write_text(json.dumps(data), encoding="utf-8")
            with patch.object(m.ds, "verify", return_value=verified):
                return m.synthesize(score_path, directory, str(paths[0]), m._sha(paths[0]),
                                    str(paths[1]), m._sha(paths[1]))

        out = run(score, side)
        assert out["observed"]["gain"] > 1.05
        assert out["official"]["gain"] < 1.0
        assert not out["QUERY_SCORE_PASSED"]
        s = copy.deepcopy(score); v = copy.deepcopy(side)
        changed = s["arms"]["candidate"]
        changed["rows"][0]["failed"] = True
        v["manifest"]["arms"]["candidate"]["evidence_rows"][0]["budget_exhausted"] = True
        v["arms"]["candidate"]["rows"][0]["failed"] = True
        full = v["manifest"]["arms"]["candidate"]["evidence_rows"]
        rh = le.rows_hash([le.EvidenceRow(**r) for r in full])
        changed["rows_hash"] = v["manifest"]["arms"]["candidate"]["rows_hash"] = rh
        changed["n_failed"] = 1
        changed["objective"] = cs.score_suite(m._scored_rows(changed["rows"], list(m.p12.SEEDS)), lam=0.).objective
        fail = run(s, v)
        assert fail["new_failed_identities"] == [f"seed:{m.p12.SEEDS[0]}"]
        assert not fail["QUERY_SCORE_PASSED"]
        cases = []
        for field, val in (("n_samples_per_replica", 1e9), ("width", 512),
                           ("truth_sha256", "0"*64), ("local_seconds_caps", "official")):
            s = copy.deepcopy(score); s[field] = val; cases.append((field, s, side))
        for field, val in (("gate", 1.0), ("official_samples", 1e8),
                           ("local_seconds_caps", "official")):
            v = copy.deepcopy(side); v["manifest"]["config"][field] = val
            cases.append(("config "+field, score, v))
        s = copy.deepcopy(score); s["arms"]["candidate"]["rows"][0]["seed"] += 1
        cases.append(("seed", s, side))
        s = copy.deepcopy(score); s["arms"]["candidate"]["rows"][0]["mse"] += .001
        cases.append(("sidecar row", s, side))
        s = copy.deepcopy(score); s["arms"]["candidate"]["candidate_sha256"] = "0"*64
        cases.append(("source role", s, side))
        s = copy.deepcopy(score); s["arms"]["control"], s["arms"]["candidate"] = (
            s["arms"]["candidate"], s["arms"]["control"])
        cases.append(("role swap", s, side))
        for name, s, v in cases:
            try:
                run(s, v)
            except (ValueError, KeyError):
                pass
            else:
                raise AssertionError(f"accepted {name}")
        print(f"query reader synthetic observed/official split and {len(cases)} refusals passed")


if __name__ == "__main__":
    main()
