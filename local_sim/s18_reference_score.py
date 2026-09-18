"""One-arm, offline DESIGN truth scorer for the frozen C10 storage references.

This is diagnostic evidence, never a promotion or deployment certificate. Run one
arm per invocation; a fresh output directory is mandatory. No public/P13 truth.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:] = [str(ROOT)] + [p for p in sys.path if Path(p or ".").resolve() != HERE]

import numpy as np  # noqa: E402
import flopscope as flops  # noqa: E402
import flopscope.numpy as fnp  # noqa: E402
from whestbench import MLP  # noqa: E402

sys.path.append(str(HERE))
import competition_score as cs  # noqa: E402
import design_sidecar as ds  # noqa: E402
import lifecycle_eval as le  # noqa: E402
from mint_provenance import canonical_mint  # noqa: E402
from p2_weff import MaterializationCounter  # noqa: E402
import s18_same_closure_reference as ref  # noqa: E402
sys.path.append(str(HERE))  # the generator deliberately removes the shim directory
import truth_reads as tr  # noqa: E402
from seed_blocks import seeds as registered_seeds  # noqa: E402

ARMS = ("C10", *ref.SWITCHES)
DIAGNOSTIC_BUDGET = 2**55
OFFICIAL_BUDGET = 2**41
ARTIFACT = "s18_one_arm_reference_score_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _version(name):
    return importlib.metadata.version(name)


def _truth(z, block):
    if block.upper() == "P13" or block.lower() in ("public", "vault"):
        raise ValueError("protected/public truth is outside this diagnostic")
    required = ("seeds", "means_r1", "means_r2", "var_r1", "var_r2",
                "gt_noise", "n_samples", "width", "depth", "producer_manifest_json")
    if any(k not in z for k in required) or "probe" in z:
        raise ValueError("incomplete or probe truth schema")
    width, depth = int(z["width"]), int(z["depth"])
    seeds = np.asarray(z["seeds"])
    n = len(seeds)
    if n < 1 or width < 2 or depth < 2 or len(set(map(int, seeds))) != n:
        raise ValueError("invalid truth shape or seed order")
    if seeds.dtype.kind not in "iu" or int(z["n_samples"]) <= 0:
        raise ValueError("invalid truth seeds or sample count")
    shape = (n, depth, width)
    if any(z[k].shape != shape or z[k].dtype != np.float32
           for k in ("means_r1", "means_r2")):
        raise ValueError("two float32 truth replicas required")
    if any(z[k].shape != (n, width) for k in ("var_r1", "var_r2")):
        raise ValueError("truth variance shape mismatch")
    if z["gt_noise"].shape != (n,):
        raise ValueError("truth noise shape mismatch")
    if any(not np.isfinite(z[k]).all() for k in required[1:6]):
        raise ValueError("nonfinite truth")
    provenance = json.loads(str(z["producer_manifest_json"].item()))
    if provenance.get("schema") != "p2_truth_math_e14_v1" or provenance.get("width") != width or provenance.get("depth") != depth:
        raise ValueError("truth producer provenance mismatch")
    return seeds.astype(np.int64), width, depth, provenance


def _source(arm):
    if arm == "C10":
        if sha(ref.CONTROL) != ref.CONTROL_SHA:
            raise ValueError("frozen C10 source changed")
        return le.load_candidate(str(ref.CONTROL))[0], ref.CONTROL_SHA
    if arm not in ref.SWITCHES:
        raise ValueError("unknown reference arm")
    source = ref.build(arm)
    return ref.load(arm), hashlib.sha256(source.encode("utf-8")).hexdigest()


def _p15_contract(seeds, samples):
    expected = registered_seeds("P15")[:24]
    if list(map(int, seeds)) != expected or int(samples) != 2**24:
        raise ValueError("P15 truth must be the registered first 24 seeds and 2^24 samples per replica")


def _operations(ctx):
    namespaces = ctx.summary_dict(by_namespace=True).get("by_namespace") or {}
    ops = {}
    for bucket in namespaces.values():
        for name, rec in (bucket.get("operations") or {}).items():
            item = ops.setdefault(name, {"flops": 0, "calls": 0})
            item["flops"] += int(rec.get("flop_cost", 0))
            item["calls"] += int(rec.get("calls", 0))
    total = int(ctx.flops_used)
    if total <= 0 or sum(x["flops"] for x in ops.values()) != total or sum(x["calls"] for x in ops.values()) <= 0:
        raise ValueError("operation ledger does not conserve counted FLOPs/calls")
    return ops


def _identity(arm, truth_path, truth_sha, block, seeds, width, depth, provenance, source_sha):
    dependencies = {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p)
                    for p in (HERE / "lifecycle_eval.py", HERE / "mint_provenance.py",
                              HERE / "competition_score.py", HERE / "design_sidecar.py",
                              HERE / "p2_weff.py", HERE / "s18_same_closure_reference.py",
                              HERE / "s18_reference_score.py", HERE / "p2_truth.py")}
    return {"artifact": ARTIFACT, "tier": "DESIGN", "arm": arm,
            "truth": str(Path(truth_path).resolve()), "truth_sha256": truth_sha,
            "block": block, "ordered_network_seeds": [int(s) for s in seeds],
            "n": len(seeds), "width": width, "depth": depth,
            "n_samples_per_replica": None, "truth_replicas": 2,
            "truth_provenance": provenance, "control_sha256": ref.CONTROL_SHA,
            "source_sha256": source_sha, "generator_sha256": sha(ref.__file__),
            "dependency_sha256": dependencies,
            "packages": {x: _version(x) for x in ("flopscope", "whestbench", "numpy")},
            "numerical_environment": {"platform": platform.platform(),
                                      "python": sys.version, "executable": sys.executable,
                                      "numpy_build": str(np.__config__.CONFIG),
                                      "threads": {k: os.environ.get(k) for k in
                                                  ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}},
            "config": {"input_weight_dtype": "f64_readonly", "diagnostic_budget": DIAGNOSTIC_BUDGET,
                       "official_budget": OFFICIAL_BUDGET, "lambda": 0.0,
                       "local_seconds_caps": "off", "complete_endogenous": True,
                       "deployable_cost_claim": False}}


def score_loaded(arm, truth_path, out_dir, *, block, synthetic_fixture=False):
    """Score an already-authorized DESIGN truth read. Caller owns read reservation."""
    out = Path(out_dir).resolve()
    if out.exists():
        raise FileExistsError(out)
    truth_path = Path(truth_path).resolve()
    if block.upper() == "P13" or "public" in str(truth_path).lower():
        raise ValueError("P13/public truth refused")
    source, source_sha = _source(arm)
    with np.load(truth_path, allow_pickle=False) as archive:
        z = {k: archive[k] for k in archive.files}
    seeds, width, depth, provenance = _truth(z, block)
    if block.upper() == "P15":
        _p15_contract(seeds, z["n_samples"])
    if not synthetic_fixture and (width, depth) != (1024, 16):
        raise ValueError("reference diagnostic requires complete Phase-2 shape")
    truth_sha = sha(truth_path)
    side = json.loads(Path(str(truth_path) + ".producer.json").read_text(encoding="utf-8"))
    if side != {"artifact_sha256": truth_sha, "math_provenance": provenance}:
        raise ValueError("truth producer sidecar mismatch")
    if provenance["mint_sha256"] != sha(HERE / "mint_provenance.py") or provenance["producer_sha256"] != sha(HERE / "p2_truth.py"):
        raise ValueError("truth mint/producer source mismatch")
    manifest = _identity(arm, truth_path, truth_sha, block, seeds, width, depth, provenance, source_sha)
    manifest["n_samples_per_replica"] = int(z["n_samples"])
    flops.configure(symmetry_warnings=False)
    estimator = source.Estimator()
    setup = type("Context", (), {"width": width, "depth": depth,
                                  "flop_budget": DIAGNOSTIC_BUDGET, "seed": 0})()
    estimator.setup(setup)
    out.mkdir(parents=True)
    writer = ds.ArmWriter(str(out), arm, width=width, candidate=arm, candidate_sha256=source_sha)
    rows = []
    manifest["run_complete"] = False
    ds.write_manifest(str(out), manifest)
    for i, seed in enumerate(seeds):
        weights_np, estimator_seed, _ = canonical_mint(width, depth, int(seed))
        mlp = MLP(weights=le.build_weights(weights_np, "f64_readonly", fnp),
                  width=width, depth=depth, seed=int(estimator_seed))
        with MaterializationCounter() as scalar:
            with flops.BudgetContext(flop_budget=DIAGNOSTIC_BUDGET, quiet=True) as ctx:
                if arm == "C10":
                    pred = estimator.predict(mlp, DIAGNOSTIC_BUDGET)
                else:
                    pred = ref.offline_predict(source, mlp, diagnostic_budget=DIAGNOSTIC_BUDGET, estimator=estimator)
                scored = le.materialise_predictions(pred, depth=depth, width=width)
        final = le.validate_predictions(scored, depth=depth, width=width)[-1].copy()
        if estimator.last_run["mode"] != "full" or estimator.last_run["generations"] is not None:
            raise ValueError("capped or fallback output")
        ops = _operations(ctx)
        target = (z["means_r1"][i, -1].astype(np.float64) + z["means_r2"][i, -1].astype(np.float64)) / 2
        target = target.astype(np.float32)
        mse = float(np.mean((final - target) ** 2))
        noise = float(z["gt_noise"][i]) / 2
        row = {"index": i, "input_seed": int(seed), "estimator_seed": int(estimator_seed),
               "source_sha256": source_sha, "truth_sha256": truth_sha,
               "mse": mse, "gt_noise_averaged": noise, "flops_used": int(ctx.flops_used),
               "operation_calls": sum(x["calls"] for x in ops.values()), "operations": ops,
               "scalar_materializations": dict(scalar.counts), "scalar_materializations_total": scalar.total,
               "last_run": estimator.last_run, "prediction_sha256": hashlib.sha256(final.tobytes()).hexdigest(),
               "failed": False}
        # ArmWriter stores the exact float32 final vector and fsyncs stream/index.
        # The sidecar contract expects dataclasses.asdict; keep a compact local row type.
        @dataclasses.dataclass
        class SideRow:
            index: int; input_seed: int; mse: float; flops_used: int
            n_ops: int; gt_noise: float; failed: bool; error_code: str; nonfinite: bool
        writer.append(SideRow(i, int(seed), mse, int(ctx.flops_used),
                              row["operation_calls"], noise, False, "", False), final)
        with open(out / "rows.jsonl", "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
            fh.flush(); os.fsync(fh.fileno())
        rows.append(row)
        manifest["completed_rows"] = len(rows)
        ds.write_manifest(str(out), manifest)
    manifest["rows_sha256"] = sha(out / "rows.jsonl")
    manifest["predictions_sha256"] = sha(ds.stream_path(str(out), arm))
    manifest["prediction_index_sha256"] = sha(ds.index_path(str(out), arm))
    manifest["mean_raw_mse"] = float(np.mean([r["mse"] for r in rows]))
    manifest["mean_truth_noise_floor"] = float(np.mean([r["gt_noise_averaged"] for r in rows]))
    # Cost forecast only. The high diagnostic budget is never scored as official.
    manifest["official_precision_score_forecast"] = float(np.mean([
        cs.mlp_score(r["mse"], r["flops_used"], 0.0, OFFICIAL_BUDGET,
                     failed=False, lam=0.0) for r in rows]))
    manifest["run_complete"] = True
    ds.write_manifest(str(out), manifest)
    return verify(out)


def verify(directory):
    out = Path(directory)
    m = json.loads((out / ds.MANIFEST).read_text(encoding="utf-8"))
    if m.get("artifact") != ds.ARTIFACT or m.get("run_complete") is not True or m.get("completed_rows") != m.get("n"):
        raise ValueError("incomplete reference manifest")
    if m.get("arm") not in ARMS or m.get("tier") != "DESIGN" or m["block"].upper() == "P13":
        raise ValueError("invalid reference identity")
    if sha(m["truth"]) != m["truth_sha256"] or sha(ref.CONTROL) != m["control_sha256"]:
        raise ValueError("truth/control bytes changed")
    _, src_sha = _source(m["arm"])
    if src_sha != m["source_sha256"] or sha(ref.__file__) != m["generator_sha256"]:
        raise ValueError("reference source changed")
    for rel, digest in m["dependency_sha256"].items():
        if sha(ROOT / rel) != digest:
            raise ValueError("dependency changed: " + rel)
    for key, filename in (("rows_sha256", "rows.jsonl"),
                          ("predictions_sha256", Path(ds.stream_path(str(out), m["arm"])).name),
                          ("prediction_index_sha256", Path(ds.index_path(str(out), m["arm"])).name)):
        if sha(out / filename) != m[key]:
            raise ValueError("sidecar changed: " + filename)
    loaded = ds.load_arm(str(out), m["arm"], width=m["width"], expect_n=m["n"])
    rows = [json.loads(line) for line in (out / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(rows) != m["n"] or [r["input_seed"] for r in rows] != m["ordered_network_seeds"]:
        raise ValueError("row/seed identity mismatch")
    for i, (row, idx) in enumerate(zip(rows, loaded["rows"])):
        if (row["failed"] or row["index"] != i or row["input_seed"] != idx["seed"]
                or row["source_sha256"] != m["source_sha256"]
                or row["truth_sha256"] != m["truth_sha256"]
                or row["prediction_sha256"] != idx["sha256"] or row["mse"] != idx["mse"]):
            raise ValueError("row/prediction binding mismatch")
        if row["flops_used"] != sum(v["flops"] for v in row["operations"].values()) or row["operation_calls"] != sum(v["calls"] for v in row["operations"].values()):
            raise ValueError("operation ledger mismatch")
    if (m["mean_raw_mse"] != float(np.mean([r["mse"] for r in rows]))
            or m["mean_truth_noise_floor"] != float(np.mean([r["gt_noise_averaged"] for r in rows]))
            or m["official_precision_score_forecast"] != float(np.mean([
                cs.mlp_score(r["mse"], r["flops_used"], 0.0, OFFICIAL_BUDGET,
                             failed=False, lam=0.0) for r in rows]))):
        raise ValueError("manifest summary mismatch")
    return m


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--block", required=True)
    ap.add_argument("--ts", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.block.upper() == "P13" or "public" in a.truth.lower():
        ap.error("P13/public truth refused")
    if Path(a.out).exists():
        ap.error("output directory already exists")
    block = tr.resolve_block(a.truth, a.block)
    if block.upper() == "P13":
        ap.error("P13 truth refused")
    if tr.tier_of(block) != "DESIGN":
        ap.error("reference scorer requires DESIGN truth")
    with tr.reserve(block, a.truth, consumer="local_sim/s18_reference_score.py",
                    purpose="offline same-closure one-arm DESIGN diagnostic " + a.arm,
                    ts=a.ts, instrument=str(Path(__file__).resolve()), result_artifact=a.out) as truth_path:
        result = score_loaded(a.arm, truth_path, a.out, block=block)
    print(json.dumps({k: result[k] for k in ("arm", "n", "mean_raw_mse", "mean_truth_noise_floor", "official_precision_score_forecast")}, indent=2))


if __name__ == "__main__":
    main()
