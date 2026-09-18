"""Tiny synthetic birth-reader checks; no P14 labels are opened."""
import copy
import dataclasses
import json
import os
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch

import competition_score as cs
import lifecycle_eval as le
import s18_birth_decide as m


TRUTH_SHA = "a" * 64  # synthetic identity; no truth file is opened


def fixture(tmp, root=None):
    root = Path(root) if root is not None else m.ROOT
    exact = root / m.CONTROL_NAME
    approx = root / m.CANDIDATE_NAME
    paths = (exact, approx)
    seeds = list(m.SEEDS)
    relpred = os.path.relpath(tmp, root).replace("\\", "/")
    spec = dataclasses.asdict(le.LifecycleSpec(
        width=1024, depth=16, flop_budget=m.B, lam=0.,
        residual_wall_time_limit_s=None, wall_time_limit_s=None,
        predict_timeout_s=1e9, setup_timeout_s=1e9))
    man = {"run_complete": True, "tier": "DESIGN", "block": "P14",
           "n": 24, "width": 1024, "depth": 16,
           "truth": "experiments/2026-09-18_p14_birth/truth_P14.npz",
           "truth_sha256": TRUTH_SHA,
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
            truth_sha256=TRUTH_SHA, width=1024, depth=16,
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
                return m.synthesize(score_path, directory, TRUTH_SHA)

        out = run(score, side)
        try:
            m.synthesize(score_path, directory, "0" * 64)
        except ValueError as exc:
            assert "identity" in str(exc)
        else:
            raise AssertionError("accepted different frozen truth hash")
        assert out["observed"]["gain"] > 1.05
        assert out["official"]["gain"] < 1.0
        assert not out["BIRTH_SCORE_PASSED"]
        windows_truth = copy.deepcopy(score)
        windows_truth["truth"] = score["truth"].replace("/", "\\")
        assert run(windows_truth, side)["BIRTH_SCORE_PASSED"] is False
        wrong_truth = copy.deepcopy(windows_truth)
        wrong_truth["truth"] = wrong_truth["truth"].replace("truth_P14.npz", "truth_P13.npz")
        try:
            run(wrong_truth, side)
        except ValueError as exc:
            assert "provenance" in str(exc)
        else:
            raise AssertionError("accepted different Windows-spelled truth target")
        s = copy.deepcopy(score); v = copy.deepcopy(side)
        changed = s["arms"]["candidate"]
        changed["rows"][0]["failed"] = True
        v["manifest"]["arms"]["candidate"]["evidence_rows"][0]["budget_exhausted"] = True
        v["arms"]["candidate"]["rows"][0]["failed"] = True
        full = v["manifest"]["arms"]["candidate"]["evidence_rows"]
        rh = le.rows_hash([le.EvidenceRow(**r) for r in full])
        changed["rows_hash"] = v["manifest"]["arms"]["candidate"]["rows_hash"] = rh
        changed["n_failed"] = 1
        changed["objective"] = cs.score_suite(m._scored_rows(changed["rows"], list(m.SEEDS)), lam=0.).objective
        fail = run(s, v)
        assert fail["new_failed_identities"] == [f"seed:{m.SEEDS[0]}"]
        assert not fail["BIRTH_SCORE_PASSED"]
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
        snapshot = directory / "snapshot"
        snapshot.mkdir()
        output = directory / "external-output" / "predictions"
        output.mkdir(parents=True)
        (output / m.ds.MANIFEST).write_text("{}", encoding="utf-8")
        for path in paths:
            shutil.copyfile(path, snapshot / path.name)
        snap_score, snap_side, snap_paths = fixture(output, snapshot)
        snap_path = output.parent / "score.json"

        def snap_run(data, verified=snap_side):
            snap_path.write_text(json.dumps(data), encoding="utf-8")
            with patch.object(m.ds, "verify", return_value=verified) as verify:
                result = m.synthesize(snap_path, output, TRUTH_SHA, run_root=snapshot)
                assert verify.call_args.kwargs == {
                    "root": str(snapshot.resolve()), "check_sources": True,
                    "require_present_sources": True, "require_full_row_binding": True}
                return result

        assert snap_run(snap_score)["BIRTH_SCORE_PASSED"] is False
        wrong = copy.deepcopy(snap_score)
        wrong["prediction_sidecars"] = "predictions"
        try:
            snap_run(wrong)
        except ValueError as exc:
            assert "prediction directory" in str(exc)
        else:
            raise AssertionError("accepted wrong snapshot-relative sidecar path")
        wrong = copy.deepcopy(snap_score)
        wrong["arms"]["candidate"]["path"] = str(paths[1])
        try:
            snap_run(wrong)
        except ValueError as exc:
            assert "source or sidecar hash" in str(exc)
        else:
            raise AssertionError("accepted checkout source instead of snapshot source")
        wrong = copy.deepcopy(snap_score)
        wrong["control"], wrong["candidate"] = wrong["candidate"], wrong["control"]
        wrong_side = copy.deepcopy(snap_side)
        wrong_side["manifest"]["control"] = wrong["control"]
        wrong_side["manifest"]["candidate"] = wrong["candidate"]
        try:
            snap_run(wrong, wrong_side)
        except ValueError as exc:
            assert "source roles differ" in str(exc)
        else:
            raise AssertionError("accepted snapshot role swap")
        (snapshot / paths[0].name).write_bytes(b"tampered source")
        try:
            snap_run(snap_score)
        except ValueError as exc:
            assert "source bytes differ" in str(exc)
        else:
            raise AssertionError("accepted changed snapshot source")
        print(f"birth reader synthetic observed/official split and {len(cases)+4} refusals passed")


if __name__ == "__main__":
    main()
