"""Sacrificial four-arm report through the real lock-first lab entrypoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np


_ROOT = Path(__file__).resolve().parent.parent
_SOURCE = _ROOT / "local_sim"
_BASE_COMMIT = "8236e25f2f4a04296c2661736026a447c7fa66a5"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _refresh_fingerprint(path):
    raw = path.read_bytes()
    marker = re.compile(rb'(?m)^_SELF_FINGERPRINT = "[0-9a-f]{64}"')
    zeroed, count = marker.subn(
        b'_SELF_FINGERPRINT = "' + b'0' * 64 + b'"', raw)
    assert count == 1
    digest = hashlib.sha256(zeroed.replace(b"\r\n", b"\n")).hexdigest()
    path.write_bytes(marker.sub(
        b'_SELF_FINGERPRINT = "' + digest.encode() + b'"', raw))


_TEST_HOOKS = r'''

# Sacrificial fixture hooks. The production lab methods under test are unchanged.
import contextlib as _fixture_contextlib
import truth_reads as _fixture_truth_reads
_fixture_rows = []

def _fixture_status(block):
    return {"block": block, "tier": "DESIGN", "reads": list(_fixture_rows)}

def _fixture_load(block, artifact, **kw):
    with open(artifact, "rb") as fh:
        truth_sha = hashlib.sha256(fh.read()).hexdigest()
    binding = {
        "block": block,
        "artifact": os.path.relpath(artifact, _ROOT).replace("\\", "/"),
        "artifact_sha256": truth_sha,
        "consumer": kw["consumer"],
        "instrument_sha256": hashlib.sha256(open(kw["instrument"], "rb").read()).hexdigest(),
        "purpose": kw["purpose"],
        "ts": "2026-09-12T00:00:00Z",
        "reservation": "sacrificial-lifecycle-fixture",
        "result_artifact": os.path.relpath(kw["result_artifact"], _ROOT).replace("\\", "/"),
        "note": kw["note"],
        "backfilled": False,
        "timestamp_source": "automatic_runtime",
        "receipt_created_utc": "2026-09-12T00:00:00Z",
    }
    _fixture_rows.append(dict(binding, status="reserved", receipt_completed_utc="",
                              elapsed_seconds=None))
    _fixture_rows.append(dict(binding, status="completed",
                              receipt_completed_utc="2026-09-12T00:00:01Z",
                              elapsed_seconds=1.0))
    with np.load(artifact) as z:
        return {key: z[key] for key in z.files}

@_fixture_contextlib.contextmanager
def _fixture_hold(path):
    with open(path, encoding="utf-8") as fh:
        receipt = json.load(fh)
    arm_paths = [k4_freezer._repo_path(row["path"], "fixture arm")
                 for row in receipt["arms"]]
    with k4_freezer._stable_files([path] + arm_paths):
        snapshot = {p: hashlib.sha256(open(p, "rb").read()).hexdigest()
                    for p in [path] + arm_paths}
        _fixture_revalidate = lambda: all(
            hashlib.sha256(open(p, "rb").read()).hexdigest() == h
            for p, h in snapshot.items())
        if any(snapshot[p] != row["sha256"]
               for p, row in zip(arm_paths, receipt["arms"])):
            raise k4_freezer.FreezeRefused("sacrificial arm source swapped")
        held = {"receipt": receipt, "receipt_path": path,
                "receipt_sha256": snapshot[path],
                "fit_receipt_path": os.path.join(_ROOT, "fit_fixture.json"),
                "fit_receipt_sha256": hashlib.sha256(open(os.path.join(
                    _ROOT, "fit_fixture.json"), "rb").read()).hexdigest(),
                "k4_lam": 1.0 / 3.0, "_fixture_revalidate": _fixture_revalidate}
        yield held
        if not _fixture_revalidate():
            raise k4_freezer.FreezeRefused("sacrificial arm changed while held")

def _fixture_revalidate_held(held):
    if not held["_fixture_revalidate"]():
        raise k4_freezer.FreezeRefused("sacrificial arm changed before publication")

_fixture_truth_reads.status = _fixture_status
_fixture_truth_reads.load_truth = _fixture_load
k4_freezer.hold_freeze_receipt = _fixture_hold
k4_freezer.revalidate_held_freeze = _fixture_revalidate_held
load_manifest = lambda: {
    "shape": {"width": 2, "depth": 2},
    "budget": {"flop_budget_per_mlp": 1000000, "multiplier_floor": 0.1},
    "package_parity": {"installed_here": {"flopscope": "fixture"},
                       "status": "SYNTHETIC", "limitation_stamp": ""},
}
_cost_under_012 = lambda *args: None
'''


def _build(root, *, baseline_source=None):
    local = root / "local_sim"
    local.mkdir()
    for name in (
            "p2_locked_bootstrap.py", "p2_lab.py", "lifecycle_eval.py",
            "competition_score.py", "p2_freeze_k4_arms.py", "phase2_manifest.py",
            "p2_tag_drift.py", "official_config.py", "harness.py",
            "mint_provenance.py", "seed_ledger.py", "seed_blocks.py",
            "truth_reads.py", "p2_k4_fit.py", "p2_k4_fit_producer.py",
            "p2_build_k4_arms.py", "p2_compact_candidate_template.py"):
        shutil.copyfile(_SOURCE / name, local / name)
    manifest = root / "experiments" / "2026-08-21_phase2_manifest"
    manifest.mkdir(parents=True)
    (manifest / "phase2_manifest.json").write_text("{}\n", encoding="utf-8")
    target = local / "p2_lab.py"
    if baseline_source is not None:
        target.write_bytes(baseline_source)
    with target.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(_TEST_HOOKS)
    _refresh_fingerprint(target)
    arm_dir = root / "arms"
    arm_dir.mkdir()
    rows = []
    for index, name in enumerate(("exact", "coherent", "consumers_only", "births_only")):
        path = arm_dir / ("arm_%s.py" % name)
        path.write_text(
            "# sacrificial arm %s\n" % name +
            "import flopscope.numpy as fnp\n"
            "class Estimator:\n"
            "    def setup(self, context): pass\n"
            "    def predict(self, mlp, budget):\n"
            "        return fnp.zeros((len(mlp.weights), mlp.width), dtype=fnp.float32)\n",
            encoding="utf-8")
        rows.append({"index": index, "name": name,
                     "path": path.relative_to(root).as_posix(),
                     "sha256": _sha(path)})
    source_bindings = {}
    for key, name in (("truth_reads_interface", "truth_reads.py"),
                      ("arm_freezer", "p2_freeze_k4_arms.py")):
        path = local / name
        source_bindings[key] = {
            "path": path.relative_to(root).as_posix(), "sha256": _sha(path)}
    receipt = root / "freeze_fixture.json"
    receipt.write_text(json.dumps({"arms": rows,
                                   "source_bindings": source_bindings}), encoding="utf-8")
    (root / "fit_fixture.json").write_text("{}\n", encoding="utf-8")
    truth = root / "truth_P01.npz"
    np.savez(truth, seeds=np.asarray([101, 202], dtype=np.int64),
             means_r1=np.zeros((2, 2, 2), dtype=np.float64),
             means_r2=np.full((2, 2, 2), 0.2, dtype=np.float64),
             gt_noise=np.asarray([0.02, 0.02]), n_samples=np.asarray(10),
             probe=np.asarray(1))
    return target, receipt, truth


def _invoke(target, receipt, truth, out):
    command = [sys.executable, str(target), "--truth", str(truth),
               "--freeze-receipt", str(receipt), "--out", str(out),
               "--control", "exact", "--competition-cost-lambda", "0",
               "--residual-cap", "0", "--depth-subset", "2"]
    return subprocess.run(command, capture_output=True, text=True,
                          cwd=target.parent.parent)


def run_probe(*, baseline=False):
    with tempfile.TemporaryDirectory(prefix="p2-locked-lifecycle-", dir=_ROOT) as td:
        root = Path(td)
        baseline_source = None
        if baseline:
            baseline_source = subprocess.check_output(
                ["git", "show", _BASE_COMMIT + ":local_sim/p2_lab.py"], cwd=_ROOT)
        target, receipt, truth = _build(root, baseline_source=baseline_source)
        out = root / "report.json"
        positive = _invoke(target, receipt, truth, out)
        report = json.loads(out.read_text(encoding="utf-8")) if out.exists() else None
        expected = {row["name"].replace("_", "-"): row["sha256"]
                    for row in json.loads(receipt.read_text(encoding="utf-8"))["arms"]}
        positive_ok = (positive.returncode == 0 and report is not None and
                       report["status"] == "complete" and
                       report["artifact"] == "p2_cp1_ladder" and
                       report["artifact_version"] == 3 and
                       report["truth"]["n_networks"] == 2 and
                       set(report["outcomes"]) == {
                           "build", "kill", "mechanism", "promotion"} and
                       report["outcomes"]["build"]["status"] == "PASS" and
                       report["k4_freeze"]["protected_truth"]["reservation_receipt"][
                           "result_artifact"] == "report.json" and
                       set(report["arms"]) == set(expected) and
                       len(set(expected.values())) == 4 and
                       all(len(report["arms"][name]["per_mlp_score"]) == 2
                           for name in expected) and
                       all(report["arms"][name]["candidate_sha256"] == sha
                           for name, sha in expected.items()))
        original_output_sha = _sha(out) if out.exists() else None
        again = _invoke(target, receipt, truth, out)
        preexisting_ok = (again.returncode == 2 and out.exists() and
                          _sha(out) == original_output_sha and
                          "Evidence is immutable" in again.stdout)
        direct_out = root / "direct_bootstrap_report.json"
        direct = subprocess.run(
            [sys.executable, str(target.parent / "p2_locked_bootstrap.py"),
             "--target", str(target), "--expected-target-sha", _sha(target),
             "--", "--truth", str(truth), "--freeze-receipt", str(receipt),
             "--out", str(direct_out), "--control", "exact",
             "--competition-cost-lambda", "0", "--residual-cap", "0",
             "--depth-subset", "2"],
            capture_output=True, text=True, cwd=root)
        direct_ok = (direct.returncode == 2 and not direct_out.exists() and
                     "separate parent-owned publication stage" in direct.stdout)
        swapped = root / "swapped_report.json"
        with (root / "arms" / "arm_exact.py").open("ab") as fh:
            fh.write(b"\n# swapped after receipt\n")
        negative = _invoke(target, receipt, truth, swapped)
        negative_ok = (negative.returncode == 2 and not swapped.exists() and
                       "sacrificial arm source swapped" in negative.stdout)
        (root / "arms" / "arm_exact.py").unlink()
        missing_out = root / "missing_report.json"
        missing = _invoke(target, receipt, truth, missing_out)
        missing_ok = missing.returncode == 2 and not missing_out.exists()
        reproduced = (baseline and positive.returncode == 2 and report is None and
                      "depth lifecycle for exact did not execute its frozen candidate bytes"
                      in positive.stdout and
                      "loaded repo-local module _lifecycle_candidate_" in positive.stdout)
        return {
            "pass": ((reproduced and negative_ok and missing_ok) if baseline else
                     (positive_ok and negative_ok and missing_ok and
                      preexisting_ok and direct_ok)),
            "baseline_reproduced": bool(reproduced),
            "positive": {"pass": positive_ok, "returncode": positive.returncode,
                         "stdout_tail": positive.stdout[-4000:],
                         "stderr_tail": positive.stderr[-4000:],
                         "report_arms": sorted(report["arms"]) if report else [],
                         "reported_hashes": {name: report["arms"][name]["candidate_sha256"]
                                             for name in report["arms"]} if report else {}},
            "source_swap": {"pass": negative_ok, "returncode": negative.returncode,
                            "stdout_tail": negative.stdout[-2000:],
                            "stderr_tail": negative.stderr[-2000:],
                            "publication_absent": not swapped.exists()},
            "missing_arm": {"pass": missing_ok, "returncode": missing.returncode,
                            "publication_absent": not missing_out.exists()},
            "preexisting_output": {"pass": preexisting_ok, "returncode": again.returncode,
                                   "original_sha_unchanged": (out.exists() and
                                                              _sha(out) == original_output_sha)},
            "direct_bootstrap_without_parent_stage": {
                "pass": direct_ok, "returncode": direct.returncode,
                "publication_absent": not direct_out.exists()},
        }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args(argv)
    evidence = run_probe(baseline=args.baseline)
    if args.output:
        path = Path(args.output)
        with path.open("x", encoding="utf-8", newline="\n") as fh:
            json.dump(evidence, fh, indent=2, sort_keys=True)
            fh.write("\n")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0 if evidence["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
