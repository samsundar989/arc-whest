"""Read-only P12 DESIGN verdict for corrected pairq query objectives.

The September-18 corrected-storage contract fixes old qE -> qEc as PRIMARY.
Old qEd -> qEdc is diagnostic only. This reader never opens truth or executes
an estimator. Query artifacts must come from s10_parity --query-error with the
same qoff source and network. Historical query artifacts without E are refused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != HERE] + [ROOT, HERE]

import competition_score as cs  # noqa: E402
import design_sidecar as ds  # noqa: E402
import lifecycle_eval as le  # noqa: E402
import s12_decide as s12  # noqa: E402
from seed_blocks import seeds  # noqa: E402

PATHS = {
    "qoff": "estimator_k3_v41_pairq_a4r256_gram_p4d5.py",
    "old_qE": "estimator_k3_v41_pairq_a4r256_qE_p4d5.py",
    "qEc": "estimator_k3_v42_pairq_a4r256_qEc_p4d5.py",
    "old_qEd": "estimator_k3_v42_pairq_a4r256_qEd_old_p4d5.py",
    "qEdc": "estimator_k3_v42_pairq_a4r256_qEdc_p4d5.py",
}
ORDER = seeds("P12")[:24]
P12_TRUTH_SHA256 = "92a3228edee16080795c4a1ea0124b8c7bd1f7f7ecf7c84ad078edc4eca756b2"
METRICS = ("E", "D3", "D21", "mu_out", "cov_out", "final_row")


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _sha(path):
    with open(path, "rb") as fh:
        h = hashlib.sha256()
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
        return h.hexdigest()


def _source(label, path, digest):
    if path != PATHS[label]:
        raise ValueError(f"{label} source path differs from frozen contract")
    if not digest or digest != _sha(os.path.join(ROOT, path)):
        raise ValueError(f"{label} source SHA differs from current frozen bytes")


def _rows(arm):
    raw = arm["rows"]
    if len(raw) != len(ORDER):
        raise ValueError("P12 comparison requires exactly 24 rows")
    if [int(r["index"]) for r in raw] != list(range(len(ORDER))):
        raise ValueError("P12 network indices are missing, duplicated or reordered")
    if [int(r["seed"]) for r in raw] != ORDER:
        raise ValueError("P12 canonical ordered identities differ")
    if any(not math.isfinite(float(r.get("residual_wall_time_s", 0))) or
           float(r.get("residual_wall_time_s", 0)) < 0 for r in raw):
        raise ValueError("invalid residual measurement")
    return s12._rows(arm, cs)


def _score_arms(data):
    if data.get("block") != "P12" or data.get("tier") != "DESIGN":
        raise ValueError("score artifact must identify P12 DESIGN")
    if (data.get("n"), data.get("width"), data.get("depth"),
            data.get("n_samples_per_replica")) != (24, 1024, 16, 2**24):
        raise ValueError("score artifact has wrong panel, shape or truth precision")
    if data.get("truth_sha256") != P12_TRUTH_SHA256:
        raise ValueError("score artifact has wrong P12 truth identity")
    if data.get("accuracy_only") is not True or data.get("local_seconds_caps") != "off":
        raise ValueError("score artifact is not the registered caps-off accuracy protocol")
    found = {}
    for arm in data["arms"].values():
        path = arm["path"]
        label = next((k for k, v in PATHS.items() if path == v), None)
        if label is None:
            continue
        if label in found:
            raise ValueError(f"duplicate score arm {label}")
        _source(label, path, arm.get("candidate_sha256"))
        found[label] = _rows(arm)
    missing = set(PATHS) - {"qoff"} - set(found)
    if missing:
        raise ValueError(f"missing score arms {sorted(missing)}")
    for old, new in (("old_qE", "qEc"), ("old_qEd", "qEdc")):
        cs.noise_corrected_compare(found[old], found[new],
                                   local_samples=2 * 2**24, require_ids=True)
    for name in ("qEc", "old_qEd", "qEdc"):
        cs.noise_corrected_compare(found["old_qE"], found[name],
                                   local_samples=2 * 2**24, require_ids=True)
    return found


def _bind_sidecar(data, directory, source_root=None):
    source_root = ROOT if source_root is None else os.path.realpath(source_root)
    verified = ds.verify(directory, check_sources=True,
                         require_present_sources=True, require_full_row_binding=True,
                         root=source_root)
    man = verified["manifest"]
    if man.get("run_complete") is not True:
        raise ValueError("prediction sidecar is incomplete")
    for key in ("block", "n", "width", "depth", "truth_sha256", "n_samples_per_replica"):
        if man.get(key) != data.get(key):
            raise ValueError(f"prediction sidecar {key} differs from score artifact")
    if man.get("ordered_network_seeds") != ORDER:
        raise ValueError("prediction sidecar has wrong ordered P12 networks")
    cfg = man.get("config", {})
    if cfg.get("accuracy_only") is not True or cfg.get("local_seconds_caps") != "off":
        raise ValueError("prediction sidecar has wrong caps-off protocol")
    for arm_label, arm in data["arms"].items():
        if arm["path"] not in PATHS.values():
            continue
        info = man["arms"].get(arm_label)
        if info is None or info.get("path") != arm["path"] or (
                info.get("candidate_sha256") != arm.get("candidate_sha256")):
            raise ValueError(f"sidecar source differs for {arm_label}")
        if not arm.get("rows_hash") or arm["rows_hash"] != info.get("rows_hash"):
            raise ValueError(f"sidecar rows hash differs for {arm_label}")
        full = info.get("evidence_rows")
        if full is None or len(full) != len(ORDER) or (
                le.rows_hash([le.EvidenceRow(**r) for r in full]) != arm["rows_hash"]):
            raise ValueError(f"sidecar full rows differ for {arm_label}")
        for compact, row in zip(arm["rows"], full):
            for short, long in (("index", "index"), ("seed", "input_seed"),
                                ("mse", "mse"), ("flops_used", "flops_used"),
                                ("gt_noise", "gt_noise"), ("nonfinite", "nonfinite"),
                                ("error_code", "error_code")):
                if compact.get(short) != row.get(long):
                    raise ValueError(f"score row differs from fsynced sidecar: {arm_label} {short}")
            failed = bool(row.get("budget_exhausted") or row.get("time_exhausted")
                          or row.get("residual_exhausted") or row.get("combined_exhausted")
                          or row.get("error_code"))
            if compact.get("failed") != failed:
                raise ValueError(f"score failure differs from fsynced sidecar: {arm_label}")


def _query(path, expected):
    raw = _read(path)
    if raw.get("artifact") != "s10_parity.query_error" or raw.get("label_free") is not True:
        raise ValueError("query evidence has wrong artifact type")
    if raw.get("network", {}).get("width") != 1024 or raw.get("network", {}).get("depth") != 16:
        raise ValueError("query evidence has wrong shape")
    if raw.get("overrides") != {"age": 4, "rank": 256}:
        raise ValueError("query evidence has wrong fixed storage setting")
    metric, _ = s12._qe(path)
    for arm, label in (("qoff", "qoff"), ("qE", expected[0]), ("qEd", expected[1])):
        source = raw["arms"][arm]
        _source(label, source.get("path"), source.get("sha256"))
        if not math.isfinite(float(source["flops"])) or float(source["flops"]) < 0:
            raise ValueError(f"{arm} missing metered FLOP bill")
        m = metric[arm]
        if m["missing_layers"] or m["layers"] != list(range(16)):
            raise ValueError(f"{arm} missing complete query layers: {m['missing_layers']}")
        if any(m.get(k) is None or not math.isfinite(m[k]) for k in METRICS):
            raise ValueError(f"{arm} missing finite E/continuation evidence")
    if not math.isfinite(float(raw["uncompressed_flops"])):
        raise ValueError("missing uncompressed FLOP bill")
    return raw, metric


def _query_compare(old, new):
    detail = {key: [old[key], new[key]] for key in METRICS}
    # E must improve; all other registered continuations may hold but not worsen.
    passed = new["E"] < old["E"] and all(new[k] <= old[k] for k in METRICS[1:])
    return bool(passed), detail


def decide(score_path, query_old_path, query_corrected_path, sidecar_dir,
           source_root=None):
    data = _read(score_path)
    if not sidecar_dir:
        raise ValueError("production decision requires verified prediction sidecar")
    _bind_sidecar(data, sidecar_dir, source_root)
    by = _score_arms(data)
    old_raw, old = _query(query_old_path, ("old_qE", "old_qEd"))
    new_raw, new = _query(query_corrected_path, ("qEc", "qEdc"))
    if old_raw["network"] != new_raw["network"]:
        raise ValueError("query audits use different networks")
    if old_raw["uncompressed_flops"] != new_raw["uncompressed_flops"]:
        raise ValueError("query audits use different uncompressed controls")
    if old_raw["arms"]["qoff"] != new_raw["arms"]["qoff"]:
        raise ValueError("query audits use different qoff reference")

    comparisons = {}
    for label, control, candidate in (("primary", "old_qE", "qEc"),
                                       ("secondary_diagnostic", "old_qEd", "qEdc")):
        c, k = by[control], by[candidate]
        observed_c, official_c = s12._scores(c, 2 * 2**24, cs)
        observed_k, official_k = s12._scores(k, 2 * 2**24, cs)
        observed = cs.paired_compare(observed_c, observed_k)
        official = cs.paired_compare(official_c, official_k)
        v = cs.decide(official_c, official_k, gate=1.0, interval="t") if label == "primary" else None
        old_metric = old["qE" if label == "primary" else "qEd"]
        new_metric = new["qE" if label == "primary" else "qEd"]
        query_ok, query_detail = _query_compare(old_metric, new_metric)
        failed_c = {r.name for r in c if r.failed}
        new_failed = sorted({r.name for r in k if r.failed} - failed_c)
        comparisons[label] = {
            "control": control, "candidate": candidate,
            "observed_gain": float(observed.arithmetic_gain),
            "control_raw_mse": float(sum(r.mse for r in c) / len(c)),
            "candidate_raw_mse": float(sum(r.mse for r in k) / len(k)),
            "control_observed_score": float(observed.mean_control),
            "candidate_observed_score": float(observed.mean_candidate),
            "observed_difference": float(observed.mean_difference),
            "observed_t95": list(observed.t_ci_difference),
            "official_precision_gain": float(official.arithmetic_gain),
            "control_official_precision_score": float(official.mean_control),
            "candidate_official_precision_score": float(official.mean_candidate),
            "official_precision_difference": float(official.mean_difference),
            "official_precision_t95": list(official.t_ci_difference),
            "truth_noise_floor": float(sum(r.gt_noise for r in c) / len(c)),
            "control_failures": sorted(failed_c), "new_failed_identities": new_failed,
            "query_condition": query_ok, "query_detail": query_detail,
            "control_flops_mean": float(sum(r.flops_used for r in c) / len(c)),
            "candidate_flops_mean": float(sum(r.flops_used for r in k) / len(k)),
            "query_audit_flops": [
                old_raw["arms"]["qE" if label == "primary" else "qEd"]["flops"],
                new_raw["arms"]["qE" if label == "primary" else "qEd"]["flops"]],
            "score_decide": None if v is None else {
                "passed": bool(v.passed), "gain": float(v.gain), "ci": list(v.ci),
                "reasons": list(v.reasons)},
        }
    primary = comparisons["primary"]
    primary["PRIMARY_SCREEN_PASSED"] = bool(
        primary["score_decide"]["passed"] and not primary["new_failed_identities"]
        and primary["query_condition"])
    return {"artifact": "s16_corrected_storage_decide", "tier": "DESIGN (P12)",
            "contract": "experiments/LOG.md 2026-09-18 corrected-storage discriminator",
            "gate": 1.0, "score_artifact": score_path,
            "query_old_artifact": query_old_path,
            "query_corrected_artifact": query_corrected_path,
            "truth_sha256": data["truth_sha256"], "comparisons": comparisons,
            "PRIMARY_SCREEN_PASSED": primary["PRIMARY_SCREEN_PASSED"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--score", required=True)
    p.add_argument("--query-old", required=True)
    p.add_argument("--query-corrected", required=True)
    p.add_argument("--sidecar", required=True,
                   help="complete prediction sidecar directory for all scored arms")
    p.add_argument("--source-root", default=None,
                   help="original workerq snapshot root for sidecar source verification; "
                        "defaults to the live repository")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    result = decide(a.score, a.query_old, a.query_corrected, a.sidecar,
                    a.source_root)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")
    print("PRIMARY", "PASS" if result["PRIMARY_SCREEN_PASSED"] else "FAIL",
          "official gain", result["comparisons"]["primary"]["official_precision_gain"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
