"""Exact S-column feed and zero/diagonal transport cost equivalence fixture."""

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import p2_k4_integrated_first_source as optimized
from test_p2_k4_integrated_first_source_n8 import (
    direct_birth, direct_carry, expand_source, raw_fourth_cumulant)


HERE = Path(__file__).resolve()
OLD_COMMIT = "5861747"
OLD_SHA = "284b550f5f1fde7840a7a84923a394bee069a3454f0c780601fe696457b8b9c8"
N8_INPUT_SHA = "50e78f1851bf4432447c9436db147063bf241b060e5560ca144cf29a2b5aebcc"
LIMIT = 1e-4


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(checks, name, passed, **details):
    checks.append({"name": name, "pass": bool(passed), **details})


def compare(checks, name, new, reference, limit=LIMIT):
    new = np.asarray(new, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    difference = np.abs(new - reference)
    relative = difference / np.maximum(1.0, np.abs(reference))
    error = float(np.max(relative)) if relative.size else 0.0
    check(checks, name, np.isfinite(relative).all() and error <= limit,
          material_relative_error=error, limit=limit)


def baseline_bytes():
    data = subprocess.check_output([
        "git", "show", OLD_COMMIT + ":local_sim/p2_k4_integrated_first_source.py"])
    if hashlib.sha256(data).hexdigest() != OLD_SHA:
        raise AssertionError("padded baseline Git bytes changed")
    return data


def load_baseline(data, folder):
    path = Path(folder) / "p2_k4_integrated_padded_baseline.py"
    path.write_bytes(data)
    name = "p2_k4_integrated_padded_baseline"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def n4_birth_identity(checks):
    n, S = 4, 3
    weights, raw, p, input_sha = optimized._integrated_inputs(n, 16, S)
    centered = raw - (raw * p[None, :]).sum(axis=1)[:, None]
    v = weights[0] @ centered
    w1 = np.array([0.7, 0.9, 0.55, 0.8])
    w2 = np.array([0.2, -0.35, 0.4, 0.15])
    covariance = (v * p[None, :]) @ v.T
    a = w2 * np.diag(covariance)
    z = 1.5 * ((w2[:, None] * v * v) - a[:, None]) * p[None, :]
    pbase = w1[:, None] * v
    zero_s, one_s = np.zeros(S), np.ones(S)
    zero_n, one_n = np.zeros(n), np.ones(n)
    skinny = optimized.Source(np.zeros((n, S)), pbase, z, np.eye(S),
                              zero_s, zero_s, zero_s, one_s, one_s,
                              None, None, None)
    padded = optimized.Source(np.zeros((n, n)),
                              np.concatenate((pbase, np.zeros((n, n - S))), axis=1),
                              z, np.eye(n)[:, :S], zero_n, zero_n, zero_n,
                              one_n, one_n, None, None, None)
    negative = optimized.Source(np.diag(w2), w1[:, None] * covariance,
                                np.zeros((n, 0)), np.zeros((n, 0)),
                                zero_n, -one_n, zero_n, one_n, one_n,
                                None, None, None)
    birth = direct_birth(raw_fourth_cumulant(v, p), w1, w2)
    compare(checks, "n4_skinny_vs_padded_birth",
            expand_source(skinny), expand_source(padded), 1e-8)
    compare(checks, "n4_two_legs_vs_direct_birth",
            expand_source(skinny) + expand_source(negative), birth, 1e-8)
    # Transport the full tensor through two deterministic full-width maps.
    represented = expand_source(skinny) + expand_source(negative)
    for later, W in enumerate(weights[1:3], 1):
        birth = direct_carry(birth, W)
        represented = direct_carry(represented, W)
        compare(checks, f"n4_two_leg_direct_carry_{later}",
                represented, birth, 1e-8)
    return input_sha


def collect_prevariance(module):
    seen = []
    original = module.Meter.run

    def observe(self, family, fn, live_bytes=0, scratch_bytes=0):
        out = original(self, family, fn, live_bytes, scratch_bytes)
        if family == "backbone_variance_extract":
            seen.append(np.asarray(out, dtype=np.float64).copy())
        return out

    module.Meter.run = observe
    try:
        raw = module.integrated_metered_program(
            n=8, depth=16, source_rank=6, rank=8, recent=4,
            ordinary_residual=8, capture_tiny=True)
    finally:
        module.Meter.run = original
    return raw, seen


def evaluate():
    checks = []
    # The historical source is loaded from a temporary path; initialize the
    # official package from the repo-root source before that module runs.
    optimized._real_flopscope()
    source_sha = sha(HERE.with_name("p2_k4_integrated_first_source.py"))
    old_bytes = baseline_bytes()
    n4_sha = n4_birth_identity(checks)
    with tempfile.TemporaryDirectory(prefix="integrated_padded_") as folder:
        old = load_baseline(old_bytes, folder)
        baseline, base_variance = collect_prevariance(old)
        revised, new_variance = collect_prevariance(optimized)
        repeat = optimized.integrated_metered_program(
            n=8, depth=16, source_rank=6, rank=8, recent=4,
            ordinary_residual=8, capture_tiny=False)
    old_trace = baseline.pop("_trace")
    baseline.pop("_inputs")
    new_trace = revised.pop("_trace")
    revised.pop("_inputs")
    check(checks, "same_frozen_n8_input",
          baseline["input_sha256"] == revised["input_sha256"] == N8_INPUT_SHA,
          old=baseline["input_sha256"], new=revised["input_sha256"])
    check(checks, "same_installed_packages",
          baseline["installed_packages"] == revised["installed_packages"])
    check(checks, "both_context_ledgers_reconcile",
          sum(baseline["families"].values()) == baseline["flops"] and
          sum(revised["families"].values()) == revised["flops"])
    fixed = {"k4_uses": 15, "k4_full": 14, "k4_final": 1,
             "feed_births": 14, "ordinary_births": 15,
             "joins": 28, "join_ordinary": 10,
             "join_feed_positive": 9, "join_feed_negative": 9,
             "source_D3_uses": 330, "source_D21_uses": 287,
             "feed_positive_D3_uses": 105, "feed_negative_D3_uses": 105,
             "feed_positive_D21_uses": 91, "feed_negative_D21_uses": 91}
    for key, expected in fixed.items():
        check(checks, "schedule_" + key,
              baseline["schedule"].get(key) ==
              revised["schedule"].get(key) == expected,
              old=baseline["schedule"].get(key),
              new=revised["schedule"].get(key), expected=expected)
    check(checks, "no_padded_positive_families",
          all(name not in revised["families"] for name in (
              "feed_birth_pad_zeros", "feed_birth_Ppad", "feed_birth_selector",
              "insert_feed_positive_A", "insert_feed_negative_A",
              "young_feed_positive_A", "joiner_wick_positive_A")))
    check(checks, "zero_positive_and_diagonal_negative_birth_shortcuts",
          revised["family_calls"].get("insert_feed_positive_A_zero") == 14
          and revised["family_calls"].get("insert_feed_negative_A_column_scale") == 14
          and revised["family_calls"].get("feed_birth_positive_identity") == 14)
    check(checks, "no_old_direct_feed_families",
          not any(name.startswith("dense_feed_D21") for name in revised["families"]))
    check(checks, "one_weight_cast_and_shared_C_gram_per_layer",
          revised["family_calls"].get("cast_weight_f32") == 16
          and revised["family_calls"].get("dense_k4_C_gram") == 14)
    check(checks, "final_mean_shortcut",
          new_trace[-1]["D21"] is None and new_trace[-1]["D22"] is None
          and new_trace[-1]["D31"] is None)
    for layer, (oldrow, newrow) in enumerate(zip(old_trace, new_trace)):
        for field in ("V", "p", "D4", "D3", "base", "mu"):
            compare(checks, f"L{layer}_{field}_padded_equivalence",
                    newrow[field], oldrow[field])
        if layer < 15:
            for field in ("D22", "D31", "D21", "K11", "C", "var_out"):
                compare(checks, f"L{layer}_{field}_padded_equivalence",
                        newrow[field], oldrow[field])
        old_sources = {(kind, source.age): source
                       for kind, source in oldrow["sources"]}
        new_sources = {(kind, source.age): source
                       for kind, source in newrow["sources"]}
        check(checks, f"L{layer}_same_source_age_keys",
              old_sources.keys() == new_sources.keys())
        for key, source in new_sources.items():
            if key[0] == "feed_positive":
                check(checks, f"L{layer}_positive_{key[1]}_S_columns",
                      source.A.shape == (8, 6) and source.P.shape == (8, 6)
                      and source.Z.shape == (8, 6)
                      and source.L.shape == (6, 6))
            if key[0] == "feed_negative":
                check(checks, f"L{layer}_negative_{key[1]}_n_columns",
                      source.A.shape == (8, 8) and source.P.shape == (8, 8))
            compare(checks, f"L{layer}_{key[0]}_age{key[1]}_dense_tensor",
                    expand_source(source), expand_source(old_sources[key]))
        compare(checks, f"L{layer}_preactivation_variance_equivalence",
                new_variance[layer], base_variance[layer])
    check(checks, "n8_negative_variance_failure_remains_visible",
          min(new_variance[3]) < 0 and min(base_variance[3]) < 0,
          old_L3_min=float(np.min(base_variance[3])),
          new_L3_min=float(np.min(new_variance[3])))
    check(checks, "deterministic_revised_meter",
          all(repeat[key] == revised[key] for key in (
              "input_sha256", "source_sha256", "installed_packages", "flops",
              "operations", "families", "family_calls", "schedule", "lifecycle")))
    old_bytes_by_layer = [row["resident_array_bytes"] for row in baseline["layers"]]
    new_bytes_by_layer = [row["resident_array_bytes"] for row in revised["layers"]]
    check(checks, "positive_storage_reduction_visible",
          len(old_bytes_by_layer) == len(new_bytes_by_layer) == 16
          and any(new < old for new, old in zip(new_bytes_by_layer, old_bytes_by_layer)),
          old_max=max(old_bytes_by_layer), new_max=max(new_bytes_by_layer))
    check(checks, "counted_FLOPs_decrease",
          revised["flops"] < baseline["flops"],
          old=baseline["flops"], new=revised["flops"],
          saved=baseline["flops"] - revised["flops"])
    return {"kind": "exact skinny positive K4-feed Source shortcut equivalence",
            "contract_commit": "5588b4c", "old_commit": OLD_COMMIT,
            "old_source_sha256": OLD_SHA, "source_sha256": sha(HERE),
            "optimized_source_sha256": source_sha,
            "n4_input_sha256": n4_sha, "n8_input_sha256": N8_INPUT_SHA,
            "checks": checks, "n_checks": len(checks),
            "n_failed": sum(not row["pass"] for row in checks),
            "pass": all(row["pass"] for row in checks),
            "max_material_relative_error": max(
                (row["material_relative_error"] for row in checks
                 if "material_relative_error" in row), default=0.0),
            "n8_L3_min_prevariance_old": float(np.min(base_variance[3])),
            "n8_L3_min_prevariance_new": float(np.min(new_variance[3])),
            "old_flops": baseline["flops"],
            "new_flops": revised["flops"],
            "saved_flops": baseline["flops"] - revised["flops"],
            "old_peak_resident_bytes": max(old_bytes_by_layer),
            "new_peak_resident_bytes": max(new_bytes_by_layer),
            "old_memory_ledger": baseline["memory"],
            "new_memory_ledger": revised["memory"],
            "old_families": baseline["families"],
            "new_families": revised["families"],
            "old_schedule": baseline["schedule"],
            "new_schedule": revised["schedule"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    args = parser.parse_args()
    target = Path(args.json)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = evaluate()
    except Exception as exc:
        result = {"kind": "exact skinny positive K4-feed Source shortcut equivalence",
                  "contract_commit": "5588b4c", "source_sha256": sha(HERE),
                  "optimized_source_sha256": sha(HERE.with_name(
                      "p2_k4_integrated_first_source.py")),
                  "pass": False, "error_type": type(exc).__name__,
                  "error": str(exc)}
    data = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    target.write_bytes(data)
    receipt = {"source_sha256": sha(HERE),
               "optimized_source_sha256": sha(HERE.with_name(
                   "p2_k4_integrated_first_source.py")),
               "result_sha256": hashlib.sha256(data).hexdigest(),
               "pass": result["pass"]}
    target.with_name("shortcut_receipt.json").write_bytes(
        (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print("%d/%d checks passed" %
          (result.get("n_checks", 0) - result.get("n_failed", 0),
           result.get("n_checks", 0)))
    if not result["pass"]:
        print("FAIL: " + result.get("error", repr(
            [row for row in result.get("checks", []) if not row["pass"]])))
        return 1
    print("tiny counted FLOPs saved", result["saved_flops"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
