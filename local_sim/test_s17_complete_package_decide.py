"""Synthetic provenance-tamper checks; no P12 truth is opened."""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch

import competition_score as cs
import lifecycle_eval as le
import s15_own_fit_driver as p12
import s17_complete_package_decide as decision


def fixture(directory, root=None):
    root = Path(root) if root is not None else decision.ROOT
    sources = {label: str(root / decision.ROLE_PATHS[label])
               for label in decision.LABELS}
    hashes = {label: hashlib.sha256(Path(path).read_bytes()).hexdigest()
              for label, path in sources.items()}
    seeds = list(p12.SEEDS)
    labels = ("control", "cand0", "cand1", "cand2", "cand3", "cand4")
    relpred = os.path.relpath(directory, root).replace("\\", "/")
    spec = dataclasses.asdict(le.LifecycleSpec(
        width=1024, depth=16, flop_budget=2**41, lam=0.,
        residual_wall_time_limit_s=None, wall_time_limit_s=None,
        predict_timeout_s=1e9, setup_timeout_s=1e9))
    man = {"run_complete": True, "block": "P12", "tier": "DESIGN",
           "n": 24, "width": 1024, "depth": 16,
           "truth": "experiments/2026-09-17b_p12/truth_P12.npz",
           "truth_sha256": p12.TRUTH_SHA, "n_samples_per_replica": 2**24,
           "truth_replicas": 2, "ordered_network_seeds": seeds,
           "control": sources["C00"],
           "candidate": ",".join(sources[label] for label in decision.LABELS[1:]),
           "config": {"accuracy_only": True, "local_seconds_caps": "off",
                      "official_samples": 1e9, "gate": 1.0, "ts": "registered",
                      "weights_mode": le.DEFAULT_WEIGHTS_MODE,
                      "spec": spec}, "arms": {}}
    score = {"tier": "DESIGN", "block": "P12", "n": 24,
             "width": 1024, "depth": 16, "accuracy_only": True,
             "local_seconds_caps": "off", "truth": man["truth"],
             "truth_sha256": p12.TRUTH_SHA,
             "n_samples_per_replica": 2**24,
             "prediction_sidecars": relpred, "control": sources["C00"],
             "candidate": man["candidate"], "arms": {}}
    loaded = {}
    for k, label in enumerate(labels):
        role = decision.LABELS[k]
        raw = [{"index": i, "seed": seed, "mse": 3e-8*(1+.02*k+.001*i),
                "flops_used": 6e11, "gt_noise": 2e-9,
                "residual_wall_time_s": 1., "failed": False,
                "nonfinite": False, "error_code": None}
               for i, seed in enumerate(seeds)]
        rr = decision._rows(raw, seeds)
        score["arms"][label] = {
            "path": sources[role], "candidate_sha256": hashes[role],
            "rows_hash": f"hash-{label}", "n_failed": 0,
            "objective": cs.score_suite(rr, lam=0.).objective,
            "rows": raw}
        man["arms"][label] = {"candidate_sha256": hashes[role],
                               "rows_hash": f"hash-{label}",
                               "n_written": 24, "n_failed": 0}
        loaded[label] = {"seeds": seeds,
                         "rows": [{"seed": r["seed"], "mse": r["mse"],
                                   "flops_used": r["flops_used"],
                                   "gt_noise": r["gt_noise"],
                                   "failed": False} for r in raw]}
    assert hashes == decision.ROLE_SHA256, "reviewed role bytes changed"
    return score, {"manifest": man, "arms": loaded}, sources


def main():
    with tempfile.TemporaryDirectory(dir=decision.ROOT) as tmp:
        directory = Path(tmp)
        (directory / "predictions_manifest.json").write_text("{}", encoding="utf-8")
        score, side, sources = fixture(directory)
        path = directory / "score.json"

        def run(s, verified):
            path.write_text(json.dumps(s), encoding="utf-8")
            with patch.object(decision.ds, "verify", return_value=verified):
                return decision.synthesize(path, directory, sources)

        result = run(score, side)
        assert result["selection"]["incumbent"] == "C00"
        cases = []
        for field, value in (("n_samples_per_replica", 1e9),
                             ("width", 512), ("depth", 8), ("n", 23),
                             ("tier", "DECIDE"), ("local_seconds_caps", "official")):
            s = copy.deepcopy(score); s[field] = value
            cases.append((field, s, side))
        for field, value in (("width", 512), ("depth", 8),
                             ("n_samples_per_replica", 1e9),
                             ("tier", "DECIDE")):
            v = copy.deepcopy(side); v["manifest"][field] = value
            cases.append(("manifest "+field, score, v))
        for field, value in (("local_seconds_caps", "official"),
                             ("accuracy_only", False), ("official_samples", 1e8),
                             ("gate", 1.05)):
            v = copy.deepcopy(side); v["manifest"]["config"][field] = value
            cases.append(("config "+field, score, v))
        s = copy.deepcopy(score); s["arms"]["cand2"]["rows_hash"] = "forged"
        cases.append(("rows_hash", s, side))
        s = copy.deepcopy(score); v = copy.deepcopy(side)
        for key in ("cand0", "cand1"):
            other = "cand1" if key == "cand0" else "cand0"
            s["arms"][key] = copy.deepcopy(score["arms"][other])
            v["manifest"]["arms"][key] = copy.deepcopy(side["manifest"]["arms"][other])
            v["arms"][key] = copy.deepcopy(side["arms"][other])
        s["candidate"] = ",".join(s["arms"][key]["path"] for key in
                                  ("cand0", "cand1", "cand2", "cand3", "cand4"))
        v["manifest"]["candidate"] = s["candidate"]
        cases.append(("C10/C01 semantic swap", s, v))
        s = copy.deepcopy(score); s["arms"]["cand3"] = copy.deepcopy(score["arms"]["cand4"])
        s["arms"]["cand4"] = copy.deepcopy(score["arms"]["cand3"])
        v = copy.deepcopy(side)
        v["manifest"]["arms"]["cand3"], v["manifest"]["arms"]["cand4"] = (
            v["manifest"]["arms"]["cand4"], v["manifest"]["arms"]["cand3"])
        v["arms"]["cand3"], v["arms"]["cand4"] = v["arms"]["cand4"], v["arms"]["cand3"]
        s["candidate"] = ",".join(s["arms"][key]["path"] for key in
                                  ("cand0", "cand1", "cand2", "cand3", "cand4"))
        v["manifest"]["candidate"] = s["candidate"]
        cases.append(("v31/sentinel semantic swap", s, v))
        for name, s, v in cases:
            try:
                run(s, v)
            except ValueError:
                pass
            else:
                raise AssertionError(f"accepted tampered {name}")
        snapshot = directory / "snapshot"
        snapshot.mkdir()
        output = directory / "external-output" / "predictions"
        output.mkdir(parents=True)
        (output / "predictions_manifest.json").write_text("{}", encoding="utf-8")
        for role in decision.LABELS:
            shutil.copyfile(sources[role], snapshot / decision.ROLE_PATHS[role])
        snap_score, snap_side, snap_sources = fixture(output, snapshot)
        snap_path = output.parent / "score.json"
        snap_path.write_text(json.dumps(snap_score), encoding="utf-8")
        with patch.object(decision.ds, "verify", return_value=snap_side) as verify:
            decision.synthesize(snap_path, output, snap_sources, run_root=snapshot)
            assert verify.call_args.kwargs["root"] == str(snapshot.resolve())
        bad_path = copy.deepcopy(snap_score)
        bad_path["prediction_sidecars"] = "predictions"
        snap_path.write_text(json.dumps(bad_path), encoding="utf-8")
        with patch.object(decision.ds, "verify", return_value=snap_side):
            try:
                decision.synthesize(snap_path, output, snap_sources, run_root=snapshot)
            except ValueError as exc:
                assert "prediction directory" in str(exc)
            else:
                raise AssertionError("accepted wrong snapshot-relative prediction path")
        snap_path.write_text(json.dumps(snap_score), encoding="utf-8")
        (snapshot / decision.ROLE_PATHS["C00"]).write_bytes(b"tampered source")
        with patch.object(decision.ds, "verify", return_value=snap_side):
            try:
                decision.synthesize(snap_path, output, snap_sources, run_root=snapshot)
            except ValueError as exc:
                assert "source path or bytes differ" in str(exc)
            else:
                raise AssertionError("accepted tampered snapshot source")
        print(f"valid fixture and {len(cases)} provenance tamper refusals passed")


if __name__ == "__main__":
    main()
