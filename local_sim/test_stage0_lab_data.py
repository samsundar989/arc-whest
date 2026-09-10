"""Focused Stage-0 0C/0D/0E production-path fixtures.

All arrays, arm sources, registers and ledgers created here are sacrificial and
live in a temporary directory. No research truth, seed block or vault is read.

Run: uv run python local_sim/test_stage0_lab_data.py
"""

import contextlib
import hashlib
import json
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np  # noqa: E402

import lifecycle_eval as le  # noqa: E402
import p2_corrector as corrector  # noqa: E402
import p2_lab as lab  # noqa: E402
import truth_reads as tr  # noqa: E402


RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"  {name:<76} {'ok' if ok else 'FAILED'}  {detail}")


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


@contextlib.contextmanager
def patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


def _sandbox():
    td = tempfile.TemporaryDirectory(prefix="stage0_lab_data_")
    root = td.name
    arms = os.path.join(root, "arms")
    os.makedirs(arms)
    for name in lab.MECHANISM_ARMS:
        with open(os.path.join(arms, f"arm_{name.replace('-', '_')}.py"),
                  "w", encoding="utf-8") as fh:
            fh.write("class Estimator:\n    pass\n")
    truth = os.path.join(root, "truth_P01.npz")
    means1 = np.zeros((2, 2, 2), dtype=np.float64)
    means2 = np.full((2, 2, 2), 0.2, dtype=np.float64)
    np.savez(
        truth, seeds=np.asarray([101, 202], dtype=np.int64),
        means_r1=means1, means_r2=means2,
        allvar_r1=np.ones_like(means1), allvar_r2=np.ones_like(means2),
        gt_noise=np.asarray([0.02, 0.02]), n_samples=np.asarray(10),
        width=np.asarray(2), depth=np.asarray(2), probe=np.asarray(0),
    )
    return td, arms, truth


def _manifest():
    return {
        "shape": {"width": 2, "depth": 2},
        "budget": {"flop_budget_per_mlp": 100,
                   "multiplier_floor": 0.1},
        "package_parity": {"installed_here": {"flopscope": "test"},
                           "status": "SYNTHETIC", "limitation_stamp": ""},
    }


def _row(name, i, seed, mse, flops, noise):
    return le.EvidenceRow(
        index=i, input_seed=seed, estimator_seed=seed + 1,
        candidate=f"arm_{name}.py", candidate_sha256=(name[0] * 64),
        truth_sha256=("f" * 64), width=2, depth=2, budget=100.0,
        flops_used=float(flops), n_ops=1.0, mse=float(mse),
        zero_prediction_mse=0.01, gt_noise=float(noise),
        residual_wall_time_s=0.0, wall_time_s=0.0)


def test_validation_precedes_truth():
    print("\n1. request validation precedes protected truth access")
    td, arms, truth = _sandbox()
    with td:
        calls = []

        def forbidden(*args, **kwargs):
            calls.append((args, kwargs))
            raise AssertionError("truth loader must not run")

        base = ["--truth", truth, "--out", os.path.join(td.name, "out.json"),
                "--arms", ",".join(lab.MECHANISM_ARMS),
                "--control", "exact", "--lam", "0"]
        with patched(lab, "ARMS_DIR", arms), patched(lab, "load_manifest", _manifest), \
                patched(tr, "load_truth", forbidden):
            rc_control = lab.main(base[:-3] + ["wrong", "--lam", "0"])
            existing = os.path.join(td.name, "exists.json")
            with open(existing, "w", encoding="utf-8") as fh:
                fh.write("keep")
            rc_output = lab.main([
                "--truth", truth, "--out", existing, "--arms",
                ",".join(lab.MECHANISM_ARMS), "--control", "exact",
                "--lam", "0"])
            bad_arm = os.path.join(arms, "arm_births_only.py")
            with open(bad_arm, "w", encoding="utf-8") as fh:
                fh.write("def broken(\n")
            rc_parse = lab.main(base)
        check("invalid control is refused", rc_control == 2)
        check("existing output is refused", rc_output == 2)
        check("a generated arm that does not parse is refused", rc_parse == 2)
        check("all three refusals occur before truth reservation", not calls,
              f"loader calls={len(calls)}")


def test_complete_report_and_receipt():
    print("\n2. real lab entry point writes one complete, noise-aware report")
    td, arms, truth = _sandbox()
    with td:
        out = os.path.join(td.name, "report.json")
        reads = os.path.join(td.name, "reads.jsonl")
        ledger = os.path.join(td.name, "ledger.jsonl")
        open(ledger, "w").close()
        original_load = tr.load_truth

        def sacrificial_load(block, artifact, **kw):
            kw.update(path=reads, ledger_path=ledger)
            return original_load(block, artifact, **kw)

        flops = {"exact": 20, "coherent": 60,
                 "consumers-only": 35, "births-only": 45}
        mse = {"exact": 0.05, "coherent": 0.04,
               "consumers-only": 0.045, "births-only": 0.047}

        def fake_run(name, seeds, truth_final, spec, gt_noise):
            check(f"{name} receives half the per-replica noise floor",
                  np.allclose(gt_noise, [0.01, 0.01]))
            rows = [_row(name, i, int(seed), mse[name] + i * 0.001,
                         flops[name], gt_noise[i])
                    for i, seed in enumerate(seeds)]
            return rows, {"setup_seconds": 0.0,
                          "candidate_sha256": name[0] * 64}

        def fake_predict(name, seeds, spec):
            a = np.zeros((len(seeds), spec.depth, spec.width), dtype=np.float64)
            return a[:, -1], a

        args = ["--truth", truth, "--out", out, "--arms",
                ",".join(lab.MECHANISM_ARMS), "--control", "exact",
                "--lam", "0", "--residual-cap", "0",
                "--official-samples", "1000"]
        with patched(lab, "ARMS_DIR", arms), patched(lab, "load_manifest", _manifest), \
                patched(lab, "_run_arm", fake_run), \
                patched(lab, "_predict_all", fake_predict), \
                patched(lab, "_cost_under_012", lambda *x: None), \
                patched(tr, "load_truth", sacrificial_load):
            rc = lab.main(args)
            n_lines_after_first = len(tr.reads(reads))
            rc_again = lab.main(args)
        report = json.load(open(out, encoding="utf-8"))
        check("the production main() succeeds", rc == 0)
        check("the report identifies itself as complete", report["status"] == "complete")
        check("all four registered arm names survive into the report",
              set(report["arms"]) == set(lab.MECHANISM_ARMS))
        check("the saved average-truth floor is half the per-replica floor",
              report["truth"]["noise_floor_of_two_replica_average_mean"] == 0.01)
        q = report["arms"]["coherent"]["noise_aware_vs_control"]
        check("unequal multipliers produce the predicted score-noise term",
              abs(q["noise_bias"] - ((0.2 - 0.6) * 0.01)) < 1e-15,
              str(q["noise_bias"]))
        check("observed/corrected/official-precision comparisons are distinct",
              len({q[k]["mean_difference"] for k in
                   ("observed", "true_function", "official_precision_forecast")}) == 3)
        check("MLP identities and counted multipliers are retained",
              report["arms"]["coherent"]["per_mlp_ids"] ==
              ["seed:101", "seed:202"] and
              report["arms"]["coherent"]["per_mlp_counted_multiplier"] ==
              [0.6, 0.6])
        recs = tr.reads(reads)
        check("automatic receipts distinguish runtime timestamps",
              recs[0].timestamp_source == "automatic_runtime" and
              bool(recs[0].receipt_created_utc) and
              recs[-1].elapsed_seconds is not None)
        check("the second run refuses the existing output before another read",
              rc_again == 2 and len(tr.reads(reads)) == n_lines_after_first)


def test_corrector_and_history():
    print("\n3. corrector uses the shared loader and live history is unchanged")
    before = _sha(tr.READS_PATH)
    p01, p08 = tr.spent("P01"), tr.spent("P08")
    td, _arms, truth = _sandbox()
    with td:
        seen = {}

        def fake_load(block, artifact, **kw):
            seen.update(block=block, artifact=artifact, kw=kw)
            with np.load(artifact) as z:
                return {k: z[k] for k in z.files}

        with patched(tr, "load_truth", fake_load):
            z = corrector.load_truth(
                truth, purpose="synthetic corrector", result_artifact="x.json")
        check("p2_corrector routes through truth_reads.load_truth",
              seen.get("block") == "P01" and "means_r1" in z)
        check("new corrector runs request an automatic receipt",
              seen["kw"]["ts"] == "")
    hits = tr.audit_direct_loads()
    check("the direct-load audit no longer finds p2_corrector",
          not any(h["file"].endswith("p2_corrector.py") for h in hits))
    check("P01's three reads and P08's exhausted three reads are preserved",
          p01 == 3 and p08 == 3 and tr.spent("P01") == p01 and
          tr.spent("P08") == p08 and _sha(tr.READS_PATH) == before,
          f"P01={p01}, P08={p08}")


def test_historical_baseline():
    print("\n4. a fixed pre-registry baseline participates in the cap")
    td, _arms, truth = _sandbox()
    with td:
        reads = os.path.join(td.name, "reads.jsonl")
        ledger = os.path.join(td.name, "ledger.jsonl")
        open(ledger, "w").close()
        try:
            with tr.reserve("P08", truth, consumer="test", purpose="must refuse",
                            path=reads, ledger_path=ledger, prior_spent=1):
                pass
            refused = False
        except tr.ReadRefused:
            refused = True
        check("prior_spent=1 exhausts P08 before yielding labels", refused)
        check("a refusal does not manufacture a registry line",
              tr.reads(reads) == [])
        with tr.reserve("P01", truth, consumer="historical replay",
                        purpose="receipt distinction", ts="2000-01-01T00:00:00Z",
                        path=reads, ledger_path=ledger):
            pass
        recs = tr.reads(reads)
        check("a supplied historical event time is distinct from receipt time",
              recs[0].timestamp_source == "supplied_historical" and
              recs[0].ts == "2000-01-01T00:00:00Z" and
              recs[0].receipt_created_utc != recs[0].ts)


def main():
    print("=" * 100)
    print("STAGE 0 LAB / NOISE REPORT / PROTECTED ACCESS")
    print("=" * 100)
    for fn in (test_validation_precedes_truth, test_complete_report_and_receipt,
               test_corrector_and_history, test_historical_baseline):
        fn()
    ok = sum(RESULTS)
    print(f"\n{ok}/{len(RESULTS)} checks passed")
    return 0 if ok == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
