"""Truth-free/sacrificial tests for the P07-freeze to P01-lab bridge."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

import lifecycle_eval as le
import p2_freeze_k4_arms as freezer
import p2_lab as lab
import test_p2_freeze_k4_arms as freeze_test
import truth_reads as tr

freezer._ALLOW_IN_PROCESS_TEST = True
lab._ALLOW_IN_PROCESS_TEST = True


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


def _run(root, *, runner=None, predictor=None, instrument_revalidator=None,
         competition_lambda="0", tier="DESIGN"):
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
    revalidator = instrument_revalidator or lab._revalidate_instruments
    with patched(lab, "load_manifest", _manifest), \
            patched(lab, "_run_arm", runner or fake_run), \
            patched(lab, "_predict_all", predictor or fake_predict), \
            patched(lab, "_revalidate_instruments", revalidator), \
            patched(lab, "_cost_under_012", lambda *args: None), \
            patched(tr, "load_truth", fake_load), \
            patched(tr, "status", fake_status):
        rc = lab.main(args)
    return rc, out, calls, freeze_receipt


def _write_attempt_denied(path):
    if os.name == "nt":
        code = ("import sys\n"
                "p=sys.argv[1]\n"
                "try:\n"
                " f=open(p,'ab'); f.write(b'X'); f.close()\n"
                "except OSError:\n"
                " raise SystemExit(7)\n"
                "raise SystemExit(0)\n")
    else:
        code = ("import fcntl,sys\n"
                "f=open(sys.argv[1],'rb')\n"
                "try:\n"
                " fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
                "except OSError:\n"
                " raise SystemExit(7)\n"
                "raise SystemExit(0)\n")
    return subprocess.run([sys.executable, "-c", code, path],
                          capture_output=True).returncode == 7


def _loaded_dependency_ab_uses_lock_first_entrypoints(root):
    """Prove imported entry refuses while locked entrypoints execute B."""
    local = os.path.join(root, "local_sim")
    manifest_dir = os.path.join(
        root, "experiments", "2026-08-21_phase2_manifest")
    os.makedirs(local)
    os.makedirs(manifest_dir)
    copied_lab = os.path.join(local, "p2_lab.py")
    shutil.copyfile(lab.__file__, copied_lab)
    copied_bootstrap = os.path.join(local, "p2_locked_bootstrap.py")
    shutil.copyfile(lab._BOOTSTRAP, copied_bootstrap)
    marker = os.path.join(root, "loaded_marker.txt")
    success_marker = os.path.join(root, "success_marker.txt")
    dynamic_marker = os.path.join(root, "dynamic_marker.txt")
    dynamic_success = os.path.join(root, "dynamic_success.txt")
    lifecycle = os.path.join(local, "lifecycle_eval.py")
    success_code = ("\nwith open(%r, 'w', encoding='utf-8') as _success_fh:\n"
                    "    _success_fh.write('published')\n" % success_marker)
    dynamic_code = (
        "\n_original_main = main\n"
        "def main(argv=None):\n"
        "    import dynamic_unregistered\n"
        "    with open(%r, 'w', encoding='utf-8') as _dynamic_success_fh:\n"
        "        _dynamic_success_fh.write('published')\n"
        "    return 0\n" % dynamic_success)

    def lifecycle_text(value):
        return ("from pathlib import Path\n"
                f"with Path({marker!r}).open('a', encoding='utf-8') as fh:\n"
                f"    fh.write({value!r})\n")

    with open(lifecycle, "w", encoding="utf-8") as fh:
        fh.write(lifecycle_text("A"))
    with open(os.path.join(local, "dynamic_unregistered.py"), "w",
              encoding="utf-8") as fh:
        fh.write("with open(%r, 'w', encoding='utf-8') as _fh:\n"
                 "    _fh.write('executed')\n" % dynamic_marker)
    with open(os.path.join(local, "competition_score.py"), "w", encoding="utf-8") as fh:
        fh.write("# synthetic scorer\n")
    with open(os.path.join(local, "p2_freeze_k4_arms.py"), "w", encoding="utf-8") as fh:
        fh.write("class FreezeRefused(ValueError): pass\n"
                 "def _assert_loaded_source(expected=None): return expected or 'ok'\n")
    with open(os.path.join(local, "phase2_manifest.py"), "w", encoding="utf-8") as fh:
        fh.write("def load(): return {}\n")
    for name in ("p2_tag_drift.py", "official_config.py", "harness.py",
                 "mint_provenance.py", "seed_ledger.py", "seed_blocks.py",
                 "truth_reads.py", "p2_k4_fit.py", "p2_k4_fit_producer.py",
                 "p2_build_k4_arms.py", "p2_compact_candidate_template.py"):
        with open(os.path.join(local, name), "w", encoding="utf-8") as fh:
            fh.write("# locked synthetic dependency\n")
    with open(os.path.join(manifest_dir, "phase2_manifest.json"), "w",
              encoding="utf-8") as fh:
        fh.write("{}\n")

    parent = os.path.join(root, "exercise.py")
    with open(parent, "w", encoding="utf-8") as fh:
        fh.write(
            "import hashlib, importlib.util, json, os, pathlib, re, subprocess, sys\n"
            f"path={copied_lab!r}\n"
            "original_target_raw=pathlib.Path(path).read_bytes()\n"
            f"lifecycle={lifecycle!r}\n"
            "stamp=pathlib.Path(lifecycle).stat()\n"
            "spec=importlib.util.spec_from_file_location('_copied_lab_a',path)\n"
            "module=importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(module)\n"
            f"pathlib.Path({lifecycle!r}).write_text({lifecycle_text('B')!r},encoding='utf-8')\n"
            "os.utime(lifecycle,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))\n"
            "bindings=module._binding_snapshot(module._cli_source_paths())\n"
            "os.environ[module._CLI_BINDINGS_ENV]=json.dumps(bindings,sort_keys=True,separators=(',',':'))\n"
            f"pathlib.Path({marker!r}).unlink(missing_ok=True)\n"
            "try:\n"
            " rc=module.cli_entry(['--help'])\n"
            "except SystemExit as exc:\n"
            " rc=exc.code\n"
            f"seen_imported=pathlib.Path({marker!r}).read_text(encoding='utf-8') if pathlib.Path({marker!r}).exists() else 'missing'\n"
            f"pathlib.Path({marker!r}).unlink(missing_ok=True)\n"
            "target_sha=hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()\n"
            "direct=subprocess.run([sys.executable,module._BOOTSTRAP,'--target',path,'--expected-target-sha',target_sha,'--','--help'],cwd="
            f"{root!r})\n"
            f"seen_direct=pathlib.Path({marker!r}).read_text(encoding='utf-8') if pathlib.Path({marker!r}).exists() else 'missing'\n"
            f"pathlib.Path({marker!r}).unlink(missing_ok=True)\n"
            "real=subprocess.run([sys.executable,path,'--help'],cwd="
            f"{root!r})\n"
            f"seen_real=pathlib.Path({marker!r}).read_text(encoding='utf-8') if pathlib.Path({marker!r}).exists() else 'missing'\n"
            f"pathlib.Path({marker!r}).unlink(missing_ok=True)\n"
            "target_raw=pathlib.Path(path).read_bytes()\n"
            "target_raw,count=re.subn(rb'(?m)^    \"lifecycle_eval\": \"local_sim/lifecycle_eval.py\",\\r?\\n',b'',target_raw)\n"
            "assert count==1\n"
            f"target_raw+={success_code!r}.encode('utf-8')\n"
            "fingerprint=re.compile(rb'(?m)^_SELF_FINGERPRINT = \"[0-9a-f]{64}\"')\n"
            "zeroed,count=fingerprint.subn(b'_SELF_FINGERPRINT = \"'+b'0'*64+b'\"',target_raw)\n"
            "assert count==1\n"
            "target_fingerprint=hashlib.sha256(zeroed.replace(b'\\r\\n',b'\\n')).hexdigest().encode('ascii')\n"
            "target_raw=fingerprint.sub(b'_SELF_FINGERPRINT = \"'+target_fingerprint+b'\"',target_raw)\n"
            "pathlib.Path(path).write_bytes(target_raw)\n"
            "missing_sha=hashlib.sha256(target_raw).hexdigest()\n"
            "direct_missing=subprocess.run([sys.executable,module._BOOTSTRAP,'--target',path,'--expected-target-sha',missing_sha,'--','--help'],cwd="
            f"{root!r},capture_output=True,text=True)\n"
            f"direct_missing_clean=not pathlib.Path({marker!r}).exists() and not pathlib.Path({success_marker!r}).exists()\n"
            "public_missing=subprocess.run([sys.executable,path,'--help'],cwd="
            f"{root!r},capture_output=True,text=True)\n"
            f"public_missing_clean=not pathlib.Path({marker!r}).exists() and not pathlib.Path({success_marker!r}).exists()\n"
            f"dynamic_raw=original_target_raw+{dynamic_code!r}.encode('utf-8')\n"
            "zeroed,count=fingerprint.subn(b'_SELF_FINGERPRINT = \"'+b'0'*64+b'\"',dynamic_raw)\n"
            "assert count==1\n"
            "dynamic_fingerprint=hashlib.sha256(zeroed.replace(b'\\r\\n',b'\\n')).hexdigest().encode('ascii')\n"
            "dynamic_raw=fingerprint.sub(b'_SELF_FINGERPRINT = \"'+dynamic_fingerprint+b'\"',dynamic_raw)\n"
            "pathlib.Path(path).write_bytes(dynamic_raw)\n"
            "dynamic_sha=hashlib.sha256(dynamic_raw).hexdigest()\n"
            "direct_dynamic=subprocess.run([sys.executable,module._BOOTSTRAP,'--target',path,'--expected-target-sha',dynamic_sha,'--','--help'],cwd="
            f"{root!r},capture_output=True,text=True)\n"
            f"direct_dynamic_clean=not pathlib.Path({dynamic_marker!r}).exists() and not pathlib.Path({dynamic_success!r}).exists()\n"
            "public_dynamic=subprocess.run([sys.executable,path,'--help'],cwd="
            f"{root!r},capture_output=True,text=True)\n"
            f"public_dynamic_clean=not pathlib.Path({dynamic_marker!r}).exists() and not pathlib.Path({dynamic_success!r}).exists()\n"
            f"pathlib.Path({marker!r}).unlink(missing_ok=True)\n"
            "pathlib.Path(path).write_bytes(pathlib.Path(path).read_bytes()+b'\\n# target version B\\n')\n"
            "mismatch=subprocess.run([sys.executable,module._BOOTSTRAP,'--target',path,'--expected-target-sha',dynamic_sha,'--','--help'],cwd="
            f"{root!r})\n"
            f"seen_mismatch=pathlib.Path({marker!r}).exists()\n"
            "missing_text=direct_missing.stdout+public_missing.stdout\n"
            "dynamic_text=direct_dynamic.stdout+public_dynamic.stdout\n"
            "raise SystemExit(0 if rc==2 and direct.returncode==0 and real.returncode==0 and direct_missing.returncode==2 and public_missing.returncode==2 and direct_dynamic.returncode==2 and public_dynamic.returncode==2 and mismatch.returncode==2 and seen_imported=='missing' and seen_direct=='B' and seen_real=='B' and direct_missing_clean and public_missing_clean and direct_dynamic_clean and public_dynamic_clean and not seen_mismatch and missing_text.count('outside the locked registry')>=2 and dynamic_text.count('outside the locked registry')>=2 else 9)\n")
    result = subprocess.run([sys.executable, parent], cwd=root,
                            capture_output=True, text=True)
    if result.returncode != 0:
        print("dependency A/B fixture stdout:", result.stdout)
        print("dependency A/B fixture stderr:", result.stderr)
    return result.returncode == 0


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
                      "mint_provenance", "phase2_manifest", "p2_tag_drift",
                      "phase2_manifest_data",
                      "official_config", "seed_ledger", "seed_blocks",
                      "truth_reads_interface", "arm_freezer", "locked_bootstrap"} and
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

        with tempfile.TemporaryDirectory(prefix="p2-lab-real-locks-", dir=_ROOT) as root:
            attempts = {}
            arm_path = os.path.join(root, "freeze", "arm_exact.py")
            real_revalidator = lab._revalidate_instruments

            def attempt(stage):
                if stage not in attempts:
                    attempts[stage] = _write_attempt_denied(arm_path)

            def locked_run(name, seeds, truth_final, spec, gt_noise):
                attempt("lifecycle")
                source_hash = hashlib.sha256(open(lab._arm_path(name), "rb").read()).hexdigest()
                return ([_row(name, i, int(seed), source_hash)
                         for i, seed in enumerate(seeds)],
                        {"setup_seconds": 0.0, "candidate_sha256": source_hash})

            def locked_predict(name, seeds, spec):
                attempt("depth")
                values = np.zeros((len(seeds), spec.depth, spec.width), dtype=np.float64)
                source_hash = hashlib.sha256(open(lab._arm_path(name), "rb").read()).hexdigest()
                return values[:, -1], values, source_hash

            def locked_publish(bindings):
                attempt("publication")
                return real_revalidator(bindings)

            rc, out, _calls, _receipt = _run(
                root, runner=locked_run, predictor=locked_predict,
                instrument_revalidator=locked_publish)
            check("real writes are denied through lifecycle, depth, and publication",
                  rc == 0 and attempts == {"lifecycle": True, "depth": True,
                                           "publication": True} and os.path.isfile(out),
                  "subprocess write/exclusive-lock attempts cannot change a held arm")

        with tempfile.TemporaryDirectory(prefix="p2-lab-loaded-a-disk-b-", dir=_ROOT) as root:
            copied = os.path.join(root, "lab_copy.py")
            shutil.copyfile(lab.__file__, copied)
            spec = importlib.util.spec_from_file_location("_lab_loaded_a", copied)
            loaded_a = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(loaded_a)
            positive = bool(loaded_a._assert_loaded_source())
            with open(copied, "ab") as fh:
                fh.write(b"\n# disk version B\n")
            try:
                loaded_a._assert_loaded_source()
                refused = False
            except loaded_a.k4_freezer.FreezeRefused as exc:
                refused = "does not match the locked disk snapshot" in str(exc)
            check("loaded lab A cannot bind changed disk B", positive and refused,
                  "self-normalized fingerprint refuses the pre-lock A-to-B race")

        with patched(lab, "_ALLOW_IN_PROCESS_TEST", False):
            guarded = lab.main([])
        check("in-process production bypass is refused", guarded == 2,
              "only the lock-first bootstrap or an explicit test override may run")

        with tempfile.TemporaryDirectory(prefix="p2-lab-dependency-a-b-") as root:
            child_used_b = _loaded_dependency_ab_uses_lock_first_entrypoints(root)
        check("forged env cannot run cached dependency A instead of disk B", child_used_b,
              "locked paths use B; missing top-level/dynamic imports refuse cleanly")

        check("lab exposes no bearer-authorized private child mode",
              not hasattr(lab, "_LOCKED_CHILD_ARG") and
              not hasattr(lab, "_locked_child_entry"),
              "public and direct bootstrap calls both lock before target import")

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
