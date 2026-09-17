"""Tests for the DESIGN prediction sidecars (handoff 2026-09-17b stage 0).

What must be caught, per the handoff: *"tests should catch wrong source/config/
cache identity and malformed or incomplete sidecars."*

  1  a real end-to-end scorer run writes one complete sidecar per arm, and every
     stored prediction is the array that arm was SCORED on (bitwise);
  2  a truncated stream is refused (a partial prediction is not a prediction);
  3  an index with the wrong number of lines is refused;
  4  a tampered prediction whose hash no longer matches its index line is refused;
  5  a candidate file edited after the run is refused by `verify` (wrong SOURCE
     identity), and so is a truth file edited after the run;
  6  an incomplete run (fewer rows than the manifest's n) is refused;
  7  a second run into the same directory refuses rather than appending;
  8  the exact prediction-change identity holds to float64 round-off.

Run:  uv run python local_sim/test_design_sidecars.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]

import numpy as np  # noqa: E402

import design_sidecar as ds  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + name + (f"   {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)
    return ok


def raises(name, fn, needle=""):
    try:
        fn()
    except Exception as exc:
        return check(name, needle.lower() in str(exc).lower() or not needle,
                     f"{type(exc).__name__}: {str(exc)[:80]}")
    return check(name, False, "no exception raised")


_EST = '''"""tiny deterministic estimator for the sidecar tests"""
import flopscope as flops
import flopscope.numpy as fnp


class Estimator:
    def setup(self, context):
        self.k = %s

    def predict(self, mlp, budget):
        n, d = mlp.width, len(mlp.weights)
        rng = fnp.random.default_rng(mlp.seed)
        base = rng.standard_normal((d, n)).astype(fnp.float32) * fnp.float32(0.01)
        return base + fnp.float32(self.k)
'''


def _write_truth(path, seeds, width, depth, rng):
    n = len(seeds)
    m1 = rng.standard_normal((n, depth, width)).astype(np.float32) * 0.01 + 0.5
    m2 = m1 + rng.standard_normal((n, depth, width)).astype(np.float32) * 0.001
    np.savez(path, seeds=np.array(seeds, dtype=np.int64), width=width, depth=depth,
             n_samples=float(2 ** 12), means_r1=m1, means_r2=m2,
             gt_noise=np.full(n, 1e-8), mint="mint_provenance.canonical_mint")


def main():
    import p2_design_score as pds
    import seed_ledger as sl
    import truth_reads as tr

    tmp = tempfile.mkdtemp(prefix="sidecar_test_")
    # The scorer registers its look in the real truth-read register. A TEST must
    # not spend a read of anything or write a line into the project's register,
    # so the block is an INFRA test id and the register is redirected to `tmp`
    # for the duration. Both are restored in `finally`.
    _orig_reserve, _orig_tier = tr.reserve, dict(sl.TIERS)
    sl.TIERS["INFRA_TEST"] = "INFRA"
    _reads = os.path.join(tmp, "truth_reads.jsonl")

    def _reserve(block, artifact, **kw):
        kw.setdefault("path", _reads)
        return _orig_reserve(block, artifact, **kw)

    tr.reserve = _reserve
    try:
        width, depth, n = 8, 3, 4
        seeds = [90_000_000_500 + i for i in range(n)]
        rng = np.random.default_rng(7)
        truth = os.path.join(tmp, "truth_TEST.npz")
        _write_truth(truth, seeds, width, depth, rng)

        ctrl = os.path.join(tmp, "arm_ctrl.py")
        cand = os.path.join(tmp, "arm_cand.py")
        open(ctrl, "w", encoding="utf-8").write(_EST % "0.50")
        open(cand, "w", encoding="utf-8").write(_EST % "0.55")

        preds_dir = os.path.join(tmp, "preds")
        out = os.path.join(tmp, "score.json")
        # the register resolves a block from the path; give it one explicitly
        rc = pds.main(["--control", ctrl, "--candidate", cand, "--truth", truth,
                       "--n", str(n), "--ts", "2026-09-17T00:00:00Z",
                       "--block", "INFRA_TEST", "--out", out,
                       "--pred-out", preds_dir])
        check("scorer returns 0", rc == 0)

        # --- 1  complete, and bitwise what was scored ---------------------- #
        v = ds.verify(preds_dir, root=_ROOT)
        check("manifest verifies", v["manifest"]["n"] == n)
        check("both arms present", set(v["arms"]) == {"control", "candidate"})
        got = v["arms"]["candidate"]
        check("row count", got["n"] == n, f"{got['n']}")
        check("seeds match", got["seeds"] == seeds)
        # independent re-evaluation of the same arm must reproduce the array
        import lifecycle_eval as le
        z = dict(np.load(truth))
        y = ((z["means_r1"][:n, -1, :].astype(np.float64)
              + z["means_r2"][:n, -1, :].astype(np.float64)) / 2.0)
        spec = le.LifecycleSpec(width=width, depth=depth, flop_budget=2 ** 41,
                                lam=0.0, residual_wall_time_limit_s=None,
                                wall_time_limit_s=None, predict_timeout_s=1e9,
                                setup_timeout_s=1e9)
        po = []
        le.evaluate(cand, seeds, y, spec=spec, gt_noise=[5e-9] * n,
                    predictions_out=po)
        check("stored predictions are bitwise the scored arrays",
              all(np.array_equal(got["predictions"][i], po[i]) for i in range(n)))
        check("manifest records package identity",
              v["manifest"]["packages"]["flopscope"] != "unknown")
        check("manifest records truth sha",
              v["manifest"]["truth_sha256"] == ds.sha_file(truth))
        check("manifest records the config the run used",
              v["manifest"]["config"]["accuracy_only"] is True
              and v["manifest"]["config"]["local_seconds_caps"] == "off")

        # --- 8  the exact identity ----------------------------------------- #
        dec = ds.prediction_change(v["arms"]["control"]["predictions"],
                                   v["arms"]["candidate"]["predictions"], y)
        check("prediction-change identity holds",
              dec["identity_residual"] < 1e-12, f"{dec['identity_residual']:.2e}")
        check("decomposition reproduces the scored MSE",
              float(np.max(np.abs(dec["mse_new"] - got["mse"]))) < 1e-9,
              f"{float(np.max(np.abs(dec['mse_new'] - got['mse']))):.2e}")

        # --- 7  a second run into the same directory refuses ---------------- #
        raises("second run into the same sidecar directory refuses",
               lambda: ds.ArmWriter(preds_dir, "candidate", width=width,
                                    candidate="x", candidate_sha256="y"),
               "already exists")

        # --- 2  truncated stream -------------------------------------------- #
        d2 = os.path.join(tmp, "trunc")
        shutil.copytree(preds_dir, d2)
        sp = ds.stream_path(d2, "candidate")
        raw = open(sp, "rb").read()
        open(sp, "wb").write(raw[:-7])
        raises("truncated stream refused",
               lambda: ds.verify(d2, root=_ROOT), "whole number")

        # --- 6  incomplete (whole rows, but too few) ------------------------ #
        d6 = os.path.join(tmp, "short")
        shutil.copytree(preds_dir, d6)
        sp = ds.stream_path(d6, "candidate")
        raw = open(sp, "rb").read()
        open(sp, "wb").write(raw[: 4 * width * (n - 1)])
        lines = open(ds.index_path(d6, "candidate"), encoding="utf-8").read().splitlines()
        open(ds.index_path(d6, "candidate"), "w", encoding="utf-8").write(
            "\n".join(lines[: n - 1]) + "\n")
        raises("incomplete sidecar refused",
               lambda: ds.verify(d6, root=_ROOT), "expected")

        # --- 3  index/stream disagreement ----------------------------------- #
        d3 = os.path.join(tmp, "idx")
        shutil.copytree(preds_dir, d3)
        with open(ds.index_path(d3, "candidate"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"arm": "candidate", "i": n, "index": n,
                                 "seed": 1, "sha256": "0" * 64, "mse": 0.0,
                                 "flops_used": 0.0, "n_ops": 0.0,
                                 "gt_noise": 0.0, "failed": False,
                                 "error_code": "", "nonfinite": False,
                                 "candidate_sha256": "z"}) + "\n")
        raises("index/stream disagreement refused",
               lambda: ds.verify(d3, root=_ROOT), "disagree")

        # --- 4  tampered prediction ----------------------------------------- #
        d4 = os.path.join(tmp, "tamper")
        shutil.copytree(preds_dir, d4)
        sp = ds.stream_path(d4, "candidate")
        arr = np.frombuffer(open(sp, "rb").read(), dtype="<f4").reshape(n, width).copy()
        arr[2, 3] = np.float32(arr[2, 3] + 1.0)
        open(sp, "wb").write(arr.tobytes(order="C"))
        raises("tampered prediction refused",
               lambda: ds.verify(d4, root=_ROOT), "index recorded")

        # --- 5  wrong source identity ---------------------------------------- #
        d5 = os.path.join(tmp, "src")
        shutil.copytree(preds_dir, d5)
        man = json.load(open(os.path.join(d5, ds.MANIFEST), encoding="utf-8"))
        cand_rel = os.path.join(tmp, "arm_cand_moved.py")
        shutil.copy(cand, cand_rel)
        open(cand_rel, "a", encoding="utf-8").write("\n# edited after the run\n")
        man["arms"]["candidate"]["path"] = os.path.relpath(cand_rel, _ROOT)
        json.dump(man, open(os.path.join(d5, ds.MANIFEST), "w", encoding="utf-8"))
        raises("candidate edited after the run refused",
               lambda: ds.verify(d5, root=_ROOT), "hashes")

        d5b = os.path.join(tmp, "truthid")
        shutil.copytree(preds_dir, d5b)
        man = json.load(open(os.path.join(d5b, ds.MANIFEST), encoding="utf-8"))
        moved = os.path.join(tmp, "truth_TEST2.npz")
        shutil.copy(truth, moved)
        with open(moved, "ab") as fh:
            fh.write(b"\0")
        man["truth"] = os.path.relpath(moved, _ROOT)
        json.dump(man, open(os.path.join(d5b, ds.MANIFEST), "w", encoding="utf-8"))
        raises("truth edited after the run refused",
               lambda: ds.verify(d5b, root=_ROOT), "hashes")

        # --- manifest missing a required identity field ---------------------- #
        d9 = os.path.join(tmp, "nopkg")
        shutil.copytree(preds_dir, d9)
        man = json.load(open(os.path.join(d9, ds.MANIFEST), encoding="utf-8"))
        del man["packages"]
        json.dump(man, open(os.path.join(d9, ds.MANIFEST), "w", encoding="utf-8"))
        raises("manifest missing package identity refused",
               lambda: ds.verify(d9, root=_ROOT), "missing")
    finally:
        tr.reserve = _orig_reserve
        sl.TIERS.clear()
        sl.TIERS.update(_orig_tier)
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if FAILS:
        print(f"FAILED {len(FAILS)}: {FAILS}")
        return 1
    print("all sidecar tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
