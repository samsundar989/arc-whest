"""Truth-free/sacrificial tests for the P07-freeze to P01-lab bridge."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile

import numpy as np

import lifecycle_eval as le
import p2_freeze_k4_arms as freezer
import p2_lab as lab
import test_p2_freeze_k4_arms as freeze_test
import truth_reads as tr


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@contextlib.contextmanager
def patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


def _manifest():
    return {"shape": {"width": 2, "depth": 2},
            "budget": {"flop_budget_per_mlp": 100,
                       "multiplier_floor": 0.1},
            "package_parity": {"installed_here": {"flopscope": "test"},
                               "status": "SYNTHETIC", "limitation_stamp": ""}}


def _truth(path):
    means1 = np.zeros((2, 2, 2), dtype=np.float64)
    means2 = np.full((2, 2, 2), 0.2, dtype=np.float64)
    np.savez(path, seeds=np.asarray([101, 202], dtype=np.int64),
             means_r1=means1, means_r2=means2,
             gt_noise=np.asarray([0.02, 0.02]), n_samples=np.asarray(10),
             probe=np.asarray(0))


def _row(name, index, seed, source_hash):
    mse = {"exact": .05, "coherent": .04,
           "consumers-only": .045, "births-only": .047}[name]
    flops = {"exact": 20, "coherent": 60,
             "consumers-only": 35, "births-only": 45}[name]
    return le.EvidenceRow(
        index=index, input_seed=seed, estimator_seed=seed + 1,
        candidate=freezer._relative(lab._arm_path(name)),
        candidate_sha256=source_hash, truth_sha256="f" * 64,
        width=2, depth=2, budget=100.0, flops_used=float(flops), n_ops=1.0,
        mse=mse + index * .001, zero_prediction_mse=.01, gt_noise=.01,
        residual_wall_time_s=0.0, wall_time_s=0.0)


def _build(root):
    fit_path, _ = freeze_test._fixture(root, selected=1)
    freeze_dir = os.path.join(root, "freeze")
    freezer.freeze(type("A", (), {"fit_receipt": fit_path,
                                   "out_dir": freeze_dir})())
    truth = os.path.join(root, "truth_P01.npz")
    _truth(truth)
    return os.path.join(freeze_dir, "freeze_receipt.json"), truth


def _run(root, *, runner=None, competition_lambda="0", tier="DESIGN"):
    freeze_receipt, truth = _build(root)
    out = os.path.join(root, "report.json")
    calls = []
    status_rows = []

    def fake_status(block):
        return {"block": block, "tier": tier, "reads": list(status_rows)}

    def fake_load(block, artifact, **kw):
        calls.append((block, artifact, kw))
        base = {"block": block,
                "artifact": freezer._relative(artifact),
                "artifact_sha256": hashlib.sha256(open(artifact, "rb").read()).hexdigest(),
                "consumer": kw["consumer"],
                "instrument_sha256": hashlib.sha256(open(kw["instrument"], "rb").read()).hexdigest(),
                "purpose": kw["purpose"], "ts": "2026-09-11T00:00:00Z",
                "reservation": "p01-test-reservation",
                "result_artifact": kw["result_artifact"], "note": kw["note"],
                "backfilled": False, "timestamp_source": "automatic_runtime",
                "receipt_created_utc": "2026-09-11T00:00:00Z"}
        status_rows.append(dict(base, status="reserved", receipt_completed_utc="",
                                elapsed_seconds=None))
        status_rows.append(dict(base, status="completed",
                                receipt_completed_utc="2026-09-11T00:00:01Z",
                                elapsed_seconds=1.0))
        with np.load(artifact) as z:
            return {name: z[name] for name in z.files}

    def fake_run(name, seeds, truth_final, spec, gt_noise):
        source_hash = hashlib.sha256(open(lab._arm_path(name), "rb").read()).hexdigest()
        rows = [_row(name, i, int(seed), source_hash)
                for i, seed in enumerate(seeds)]
        return rows, {"setup_seconds": 0.0, "candidate_sha256": source_hash}

    def fake_predict(name, seeds, spec):
        values = np.zeros((len(seeds), spec.depth, spec.width), dtype=np.float64)
        source_hash = hashlib.sha256(open(lab._arm_path(name), "rb").read()).hexdigest()
        return values[:, -1], values, source_hash

    args = ["--truth", truth, "--out", out, "--freeze-receipt", freeze_receipt,
            "--control", "exact", "--competition-cost-lambda", competition_lambda,
            "--residual-cap", "0", "--official-samples", "1000"]
    with patched(lab, "load_manifest", _manifest), \
            patched(lab, "_run_arm", runner or fake_run), \
            patched(lab, "_predict_all", fake_predict), \
            patched(lab, "_cost_under_012", lambda *args: None), \
            patched(tr, "load_truth", fake_load), \
            patched(tr, "status", fake_status):
        rc = lab.main(args)
    return rc, out, calls, freeze_receipt


def run_checks():
    rows = []

    def check(name, condition, detail):
        rows.append({"check": name, "pass": bool(condition), "detail": detail})
        print("%-62s %s  %s" % (name, "PASS" if condition else "FAIL", detail))

    original_ledger = freezer._TRUTH_READS_LEDGER
    try:
        with tempfile.TemporaryDirectory(prefix="p2-lab-freeze-positive-", dir=_ROOT) as root:
            rc, out, calls, freeze_receipt = _run(root)
            report = json.load(open(out, encoding="utf-8"))
            frozen = json.load(open(freeze_receipt, encoding="utf-8"))
            check("receipt-driven P01 lab writes one complete report", rc == 0 and len(calls) == 1,
                  "one sacrificial truth load, four mocked lifecycle arms")
            check("report separates K4 and competition lambdas",
                  report["k4_freeze"]["k4_lam"] == 1.0 / 6.0 and
                  report["competition_cost_lambda"] == 0.0 and
                  "lambda_used" not in report,
                  "k4_lam=1/6; competition_cost_lambda=0")
            check("report binds fit, freeze, arms, and lab instruments",
                  report["k4_freeze"]["freeze_receipt"]["sha256"] ==
                  hashlib.sha256(open(freeze_receipt, "rb").read()).hexdigest() and
                  set(report["k4_freeze"]["frozen_arm_hashes"]) == set(lab.MECHANISM_ARMS) and
                  set(report["k4_freeze"]["instrument_bindings"]) == {
                      "p2_lab", "lifecycle_eval", "competition_score", "harness",
                      "mint_provenance", "phase2_manifest", "phase2_manifest_data",
                      "official_config", "seed_ledger", "seed_blocks",
                      "truth_reads_interface", "arm_freezer"} and
                  report["k4_freeze"]["protected_truth"]["data_role"] == "exposed-design" and
                  report["k4_freeze"]["protected_truth"]["block"] == "P01" and
                  report["k4_freeze"]["protected_truth"]["reservation_receipt"]["status"] == "completed",
                  "exact bridge and instrument provenance")
            expected = {row["name"].replace("_", "-"): row["sha256"]
                        for row in frozen["arms"]}
            check("executed lifecycle hashes equal frozen hashes",
                  all(report["arms"][name]["candidate_sha256"] == expected[name]
                      for name in lab.MECHANISM_ARMS),
                  "all four scored paths")

        with tempfile.TemporaryDirectory(prefix="p2-lab-freeze-missing-", dir=_ROOT) as root:
            truth = os.path.join(root, "truth_P01.npz")
            _truth(truth)
            out = os.path.join(root, "report.json")
            calls = []
            with patched(tr, "load_truth", lambda *args, **kw: calls.append(True)):
                rc = lab.main(["--truth", truth, "--out", out,
                               "--control", "exact"])
            check("missing freeze receipt refuses before P01 truth",
                  rc == 2 and not calls and not os.path.exists(out),
                  "required bridge has no compatibility bypass")

        with tempfile.TemporaryDirectory(prefix="p2-lab-freeze-tampered-", dir=_ROOT) as root:
            freeze_receipt, truth = _build(root)
            frozen = json.load(open(freeze_receipt, encoding="utf-8"))
            arm = freezer._repo_path(frozen["arms"][0]["path"], "test arm")
            with open(arm, "ab") as fh:
                fh.write(b"\n# mixed stale arm\n")
            out = os.path.join(root, "report.json")
            calls = []
            with patched(tr, "load_truth", lambda *args, **kw: calls.append(True)):
                rc = lab.main(["--truth", truth, "--out", out,
                               "--freeze-receipt", freeze_receipt,
                               "--control", "exact"])
            check("tampered frozen arm refuses before P01 truth",
                  rc == 2 and not calls and not os.path.exists(out),
                  "lab invokes strict freeze validation first")

        with tempfile.TemporaryDirectory(prefix="p2-lab-freeze-cost-", dir=_ROOT) as root:
            rc, out, calls, _ = _run(root, competition_lambda="1")
            check("nonzero competition lambda refuses before truth",
                  rc == 2 and not calls and not os.path.exists(out),
                  "Phase-2 cost model remains C=F")

        with tempfile.TemporaryDirectory(prefix="p2-lab-freeze-role-", dir=_ROOT) as root:
            rc, out, calls, _ = _run(root, tier="DECISION")
            check("P01 must remain exposed DESIGN before truth",
                  rc == 2 and not calls and not os.path.exists(out),
                  "the report cannot relabel P01 as fresh decision data")

        with tempfile.TemporaryDirectory(prefix="p2-lab-freeze-hash-", dir=_ROOT) as root:
            def wrong_hash(name, seeds, truth_final, spec, gt_noise):
                rows = [_row(name, i, int(seed), "0" * 64)
                        for i, seed in enumerate(seeds)]
                return rows, {"setup_seconds": 0.0, "candidate_sha256": "0" * 64}
            rc, out, calls, _ = _run(root, runner=wrong_hash)
            check("forged lifecycle hash refuses without a report",
                  rc == 2 and len(calls) == 1 and not os.path.exists(out),
                  "truth read counts, invalid report does not publish")

        with tempfile.TemporaryDirectory(prefix="p2-lab-freeze-closing-", dir=_ROOT) as root:
            freeze_receipt, truth = _build(root)
            out = os.path.join(root, "report.json")
            real_revalidate = freezer.revalidate_held_freeze
            calls = {"n": 0}
            status_rows = []

            def fail_after_publication(result):
                calls["n"] += 1
                if calls["n"] == 2:
                    raise freezer.FreezeRefused("injected closing mutation")
                return real_revalidate(result)

            def fake_load(block, artifact, **kw):
                base = {"block": block, "artifact": freezer._relative(artifact),
                        "artifact_sha256": hashlib.sha256(open(artifact, "rb").read()).hexdigest(),
                        "consumer": kw["consumer"],
                        "instrument_sha256": hashlib.sha256(open(kw["instrument"], "rb").read()).hexdigest(),
                        "purpose": kw["purpose"], "ts": "2026-09-11T00:00:00Z",
                        "reservation": "p01-closing-reservation",
                        "result_artifact": kw["result_artifact"], "note": kw["note"],
                        "backfilled": False, "timestamp_source": "automatic_runtime",
                        "receipt_created_utc": "2026-09-11T00:00:00Z"}
                status_rows.append(dict(base, status="reserved", receipt_completed_utc="",
                                        elapsed_seconds=None))
                status_rows.append(dict(base, status="completed",
                                        receipt_completed_utc="2026-09-11T00:00:01Z",
                                        elapsed_seconds=1.0))
                with np.load(artifact) as z:
                    return {name: z[name] for name in z.files}

            def fake_status(block):
                return {"block": block, "tier": "DESIGN", "reads": list(status_rows)}

            def fake_run(name, seeds, truth_final, spec, gt_noise):
                h = hashlib.sha256(open(lab._arm_path(name), "rb").read()).hexdigest()
                return [_row(name, i, int(seed), h) for i, seed in enumerate(seeds)], {
                    "setup_seconds": 0.0, "candidate_sha256": h}

            def fake_predict(name, seeds, spec):
                values = np.zeros((len(seeds), spec.depth, spec.width))
                h = hashlib.sha256(open(lab._arm_path(name), "rb").read()).hexdigest()
                return values[:, -1], values, h

            args = ["--truth", truth, "--out", out, "--freeze-receipt", freeze_receipt,
                    "--control", "exact", "--residual-cap", "0"]
            with patched(lab, "load_manifest", _manifest), \
                    patched(lab, "_run_arm", fake_run), \
                    patched(lab, "_predict_all", fake_predict), \
                    patched(lab, "_cost_under_012", lambda *args: None), \
                    patched(tr, "load_truth", fake_load), \
                    patched(tr, "status", fake_status), \
                    patched(freezer, "revalidate_held_freeze", fail_after_publication):
                rc = lab.main(args)
            check("closing mutation removes the just-written report",
                  rc == 2 and calls["n"] == 2 and not os.path.exists(out),
                  "exclusive report is not left accepted")
    finally:
        freezer._TRUTH_READS_LEDGER = original_ledger
    return rows


def main(argv=None):
    rows = run_checks()
    passed = sum(row["pass"] for row in rows)
    print("\n%d/%d checks passed" % (passed, len(rows)))
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
