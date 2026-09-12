"""Fresh n8 stability adjudication for the frozen integrated first-source meter.

The reference expands full tensors in float64 and enumerates Taylor factorials.
It does not call the instrument's compact slice or consumer formulas. The n4
strict-gate failure remains separate and unchanged.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import p2_k4_integrated_first_source as instrument


HERE = Path(__file__).resolve()
ROOT = HERE.parent.parent
EVIDENCE = ROOT / "experiments/2026-09-11_stage2_compact_p07/integrated_first_source"
CONTRACT = "9e5ec41"
SOURCE_SHA = "284b550f5f1fde7840a7a84923a394bee069a3454f0c780601fe696457b8b9c8"
INPUT_SHA = "50e78f1851bf4432447c9436db147063bf241b060e5560ca144cf29a2b5aebcc"
LIMIT = 1e-4


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(checks, name, condition, **details):
    checks.append({"name": name, "pass": bool(condition), **details})


def compare(checks, name, got, reference):
    got = np.asarray(got, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    difference = np.abs(got - reference)
    relative = difference / np.maximum(1.0, np.abs(reference))
    error = float(np.max(relative)) if relative.size else 0.0
    check(checks, name, np.isfinite(relative).all() and error <= LIMIT,
          material_relative_error=error, limit=LIMIT)


def raw_fourth_cumulant(nodes, probability):
    """Enumerate the supplied centered atom law, including all Wick pairings."""
    nodes = np.asarray(nodes, dtype=np.float64)
    probability = np.asarray(probability, dtype=np.float64)
    n = nodes.shape[0]
    covariance = np.zeros((n, n), dtype=np.float64)
    moment4 = np.zeros((n, n, n, n), dtype=np.float64)
    for atom in range(nodes.shape[1]):
        column = nodes[:, atom]
        covariance += probability[atom] * np.einsum("i,j->ij", column, column)
        moment4 += probability[atom] * np.einsum(
            "i,j,k,l->ijkl", column, column, column, column)
    cumulant = np.empty_like(moment4)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for ell in range(n):
                    cumulant[i, j, k, ell] = (
                        moment4[i, j, k, ell]
                        - covariance[i, j] * covariance[k, ell]
                        - covariance[i, k] * covariance[j, ell]
                        - covariance[i, ell] * covariance[j, k])
    return cumulant


def direct_birth(cumulant, derivative1, derivative2):
    """Contract one K4 leg with ReLU's second derivative at all three sites."""
    n = cumulant.shape[0]
    out = np.empty((n, n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                out[i, j, k] = 0.5 * (
                    derivative2[i] * derivative1[j] * derivative1[k]
                    * cumulant[i, i, j, k]
                    + derivative1[i] * derivative2[j] * derivative1[k]
                    * cumulant[i, j, j, k]
                    + derivative1[i] * derivative1[j] * derivative2[k]
                    * cumulant[i, j, k, k])
    return out


def direct_carry(tensor, map_matrix):
    """Dense three-leg transport, independent of Source factors and Gram QR."""
    return np.einsum("ia,jb,kc,abc->ijk", map_matrix, map_matrix,
                     map_matrix, tensor, optimize=True)


def expand_source(source):
    """Direct index expansion of a Source's represented cubic, no slice API."""
    A = np.asarray(source.A, dtype=np.float64)
    P = np.asarray(source.P, dtype=np.float64)
    Z = np.asarray(source.Z, dtype=np.float64)
    L = np.asarray(source.L, dtype=np.float64)
    scalar = np.asarray(source.s, dtype=np.float64)
    extra = np.asarray(source.e, dtype=np.float64)
    weight2 = np.asarray(source.w2, dtype=np.float64)
    M = (P * scalar[None, :] + 3.0 * A * extra[None, :] + Z @ L.T)
    n, count = A.shape
    out = np.zeros((n, n, n), dtype=np.float64)
    for h in range(count):
        a = A[:, h]
        p = P[:, h]
        m = M[:, h]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    out[i, j, k] += weight2[h] * (
                        p[i] * a[j] * a[k]
                        + a[i] * p[j] * a[k]
                        + a[i] * a[j] * p[k])
                    out[i, j, k] += (
                        m[i] * p[j] * p[k]
                        + p[i] * m[j] * p[k]
                        + p[i] * p[j] * m[k]) / 3.0
    return out


def audit_meter(captured):
    original = instrument.Meter.run
    calls = {}

    def observed(self, family, fn, live_bytes=0, scratch_bytes=0):
        out = original(self, family, fn, live_bytes, scratch_bytes)
        call = calls.get(family, 0) + 1
        calls[family] = call
        array = np.asarray(out)
        if array.size and np.issubdtype(array.dtype, np.number):
            finite = bool(np.isfinite(array).all())
            if not finite and captured["first_nonfinite"] is None:
                captured["first_nonfinite"] = {"family": family, "call": call}
            if family == "backbone_variance_extract":
                captured["preactivation_variance_min"].append(
                    float(np.min(array)))
            elif family == "nonlinear_gaussian_mean":
                captured["gaussian_means"].append(np.asarray(array, dtype=np.float64).copy())
            elif family == "nonlinear_mean_k4":
                captured["k4_mean_corrections"].append(
                    np.asarray(array, dtype=np.float64).copy())
        return out

    instrument.Meter.run = observed
    return original


def evaluate():
    checks = []
    source_path = HERE.with_name("p2_k4_integrated_first_source.py")
    source_sha = sha(source_path)
    check(checks, "frozen_instrument_source", source_sha == SOURCE_SHA,
          got=source_sha, expected=SOURCE_SHA)
    weights, _, _, generated_input_sha = instrument._integrated_inputs(8, 16, 6)
    check(checks, "frozen_n8_generator_input", generated_input_sha == INPUT_SHA,
          got=generated_input_sha, expected=INPUT_SHA)
    prior_manifest = json.loads((EVIDENCE / "input_manifest.json").read_text(encoding="utf-8"))
    current_packages = instrument._integrated_package_manifest()
    check(checks, "frozen_package_manifest",
          current_packages == prior_manifest["installed_packages"],
          got=current_packages, expected=prior_manifest["installed_packages"])
    captured = {"first_nonfinite": None, "preactivation_variance_min": [],
                "gaussian_means": [], "k4_mean_corrections": []}
    original = audit_meter(captured)
    try:
        raw = instrument.integrated_metered_program(
            n=8, depth=16, source_rank=6, rank=8, recent=4,
            ordinary_residual=8, capture_tiny=True)
    finally:
        instrument.Meter.run = original
    traces = raw.pop("_trace")
    raw.pop("_inputs")
    check(checks, "instrument_input_identity", raw["input_sha256"] == INPUT_SHA,
          got=raw["input_sha256"], expected=INPUT_SHA)
    check(checks, "instrument_source_identity", raw["source_sha256"] == SOURCE_SHA,
          got=raw["source_sha256"], expected=SOURCE_SHA)
    check(checks, "instrument_package_identity",
          raw["installed_packages"] == current_packages)
    check(checks, "all_meter_intermediates_finite",
          captured["first_nonfinite"] is None,
          first_nonfinite=captured["first_nonfinite"])
    variance = captured["preactivation_variance_min"]
    check(checks, "all_preactivation_variances_positive_and_usable",
          len(variance) == 16 and all(math.isfinite(v) and v > 1e-6 for v in variance),
          minimum=min(variance) if variance else None,
          per_layer=variance, limit_strictly_greater_than=1e-6)
    correction_ratio = [
        float(np.max(np.abs(c) / np.maximum(1.0, np.abs(b))))
        for c, b in zip(captured["k4_mean_corrections"],
                        captured["gaussian_means"])]
    check(checks, "all_k4_mean_corrections_bounded",
          len(correction_ratio) == 16 and
          all(math.isfinite(r) and r < 100 for r in correction_ratio),
          maximum=max(correction_ratio) if correction_ratio else None,
          per_layer=correction_ratio, limit_strictly_less_than=100)

    direct_feeds = {}
    mutant_gap = 0.0
    for layer, row in enumerate(traces):
        n = row["V"].shape[0]
        bank4 = (np.zeros((n, n, n, n), dtype=np.float64) if layer == 0
                 else raw_fourth_cumulant(row["V"], row["p"]))
        i = np.arange(n)
        compare(checks, f"L{layer}_D4", row["D4"], bank4[i, i, i, i])
        if layer < 15:
            compare(checks, f"L{layer}_D22", row["D22"], bank4[i[:, None], i[:, None],
                                                                 i[None, :], i[None, :]])
            d31 = bank4[i[:, None], i[:, None], i[:, None], i[None, :]]
            compare(checks, f"L{layer}_D31_forward", row["D31"], d31)
            compare(checks, f"L{layer}_D31_reverse", row["D31"].T, d31.T)
        ordinary = np.zeros((n, n, n), dtype=np.float64)
        for kind, source in row["sources"]:
            if kind == "ordinary":
                ordinary += expand_source(source)
        feed = np.zeros_like(ordinary)
        for birth in range(1, layer):
            if birth not in direct_feeds:
                birth_row = traces[birth]
                birth_k4 = raw_fourth_cumulant(birth_row["V"], birth_row["p"])
                direct_feeds[birth] = direct_birth(
                    birth_k4, birth_row["w1"], birth_row["w2"])
            carried = direct_feeds[birth]
            for later in range(birth + 1, layer + 1):
                transform = np.asarray(weights[later], dtype=np.float32).astype(np.float64)
                if later > birth + 1:
                    transform = transform * traces[later - 1]["w1"][None, :]
                carried = direct_carry(carried, transform)
            feed += carried
            if layer == birth + 1:
                positive = [source for kind, source in row["sources"]
                            if kind == "feed_positive" and source.age == 0]
                if len(positive) == 1:
                    mutant_gap = max(mutant_gap, float(np.max(np.abs(
                        carried - expand_source(positive[0])))))
        full3 = ordinary + feed
        direct_d3 = full3[i, i, i]
        compare(checks, f"L{layer}_D3", row["D3"], direct_d3)
        direct_mean = (np.asarray(row["base"], dtype=np.float64)
                       + direct_d3 * row["w3"] / math.factorial(3)
                       + bank4[i, i, i, i] * row["w4"] / math.factorial(4))
        compare(checks, f"L{layer}_mean_factorial", row["mu"], direct_mean)
        if layer < 15:
            d21 = full3[i[:, None], i[:, None], i[None, :]]
            compare(checks, f"L{layer}_D21_oriented", row["D21"], d21)
            want_cov = np.empty((n, n), dtype=np.float64)
            derivative = {1: row["w1"], 2: row["w2"], 3: row["w3"]}
            for left_index in range(n):
                for right_index in range(n):
                    co = row["C_off"][left_index, right_index]
                    value = (co * derivative[1][left_index] * derivative[1][right_index]
                             + co * co * derivative[2][left_index]
                             * derivative[2][right_index] / 2.0)
                    for degree, tensor in ((3, full3), (4, bank4)):
                        for left_degree in range(1, degree):
                            right_degree = degree - left_degree
                            if left_degree > 3 or right_degree > 3:
                                continue
                            indices = ((left_index,) * left_degree
                                       + (right_index,) * right_degree)
                            value += (derivative[left_degree][left_index]
                                      * derivative[right_degree][right_index]
                                      * tensor[indices]
                                      / (math.factorial(left_degree)
                                         * math.factorial(right_degree)))
                    want_cov[left_index, right_index] = value
            compare(checks, f"L{layer}_covariance_factorial", row["K11"], want_cov)
            assembled = np.diag(row["var_out"]) + want_cov * (
                np.ones((n, n)) - np.eye(n))
            compare(checks, f"L{layer}_covariance_assembled", row["C"], assembled)
        counts = {kind: sum(k == kind for k, _ in row["sources"])
                  for kind in ("ordinary", "feed_positive", "feed_negative")}
        ordinary_ages = sorted(source.age for kind, source in row["sources"]
                               if kind == "ordinary")
        positive_ages = sorted(source.age for kind, source in row["sources"]
                               if kind == "feed_positive")
        negative_ages = sorted(source.age for kind, source in row["sources"]
                               if kind == "feed_negative")
        check(checks, f"L{layer}_source_ages_counts",
              counts == {"ordinary": layer, "feed_positive": max(0, layer - 1),
                         "feed_negative": max(0, layer - 1)} and
              ordinary_ages == list(range(layer)) and
              positive_ages == list(range(max(0, layer - 1))) and
              negative_ages == list(range(max(0, layer - 1))),
              counts=counts, ordinary_ages=ordinary_ages,
              positive_ages=positive_ages, negative_ages=negative_ages)
    check(checks, "negative_feed_drop_discrimination", mutant_gap > 1e-6,
          max_absolute_tensor_gap=mutant_gap,
          limit_strictly_greater_than=1e-6)
    fixed = {"k4_uses": 15, "k4_full": 14, "k4_final": 1,
             "feed_births": 14, "ordinary_births": 15,
             "joins": 28, "join_ordinary": 10,
             "join_feed_positive": 9, "join_feed_negative": 9,
             "source_D3_uses": 330, "source_D21_uses": 287,
             "ordinary_D3_uses": 120, "ordinary_D21_uses": 105,
             "feed_positive_D3_uses": 105, "feed_negative_D3_uses": 105,
             "feed_positive_D21_uses": 91, "feed_negative_D21_uses": 91}
    for key, expected in fixed.items():
        check(checks, "schedule_" + key, raw["schedule"].get(key) == expected,
              got=raw["schedule"].get(key), expected=expected)
    forbidden = [key for key in raw["families"] if
                 key.startswith("dense_feed_D21") or key in (
                     "dense_shared_A_transport", "dense_feed_P_transport",
                     "dense_feed_a_transport", "dense_feed_Q_reconstruct")]
    check(checks, "no_old_direct_feed_families", not forbidden, forbidden=forbidden)
    check(checks, "one_weight_cast_per_layer",
          raw["family_calls"].get("cast_weight_f32") == 16,
          got=raw["family_calls"].get("cast_weight_f32"))
    check(checks, "fourteen_dense_C_grams",
          raw["family_calls"].get("dense_k4_C_gram") == 14,
          got=raw["family_calls"].get("dense_k4_C_gram"))
    check(checks, "family_context_flop_parity",
          sum(raw["families"].values()) == raw["flops"],
          family_sum=sum(raw["families"].values()), context=raw["flops"])
    check(checks, "final_shortcut_D3_only",
          traces[-1]["D21"] is None and traces[-1]["D22"] is None
          and traces[-1]["D31"] is None and raw["layers"][-1]["D21"] is False)
    # Repeating the complete tiny path tests output and ledger determinism.
    again = instrument.integrated_metered_program(
        n=8, depth=16, source_rank=6, rank=8, recent=4,
        ordinary_residual=8, capture_tiny=False)
    check(checks, "deterministic_complete_meter",
          all(again[key] == raw[key] for key in (
              "input_sha256", "source_sha256", "installed_packages", "flops",
              "operations", "families", "family_calls", "schedule", "lifecycle")))
    return {"kind": "fresh n8 integrated first-source numerical adjudication",
            "contract_commit": CONTRACT, "instrument_sha256": source_sha,
            "source_sha256": sha(HERE), "input_sha256": generated_input_sha,
            "installed_packages": current_packages,
            "checks": checks, "n_checks": len(checks),
            "n_failed": sum(not item["pass"] for item in checks),
            "pass": all(item["pass"] for item in checks),
            "max_material_relative_error": max((item["material_relative_error"]
                for item in checks if "material_relative_error" in item), default=0.0),
            "first_nonfinite": captured["first_nonfinite"],
            "meter": raw}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    args = parser.parse_args()
    target = Path(args.json)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = evaluate()
    except Exception as exc:
        result = {"kind": "fresh n8 integrated first-source numerical adjudication",
                  "contract_commit": CONTRACT, "source_sha256": sha(HERE),
                  "instrument_sha256": sha(HERE.with_name(
                      "p2_k4_integrated_first_source.py")),
                  "pass": False, "error_type": type(exc).__name__,
                  "error": str(exc)}
    data = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    target.write_bytes(data)
    receipt = {"source_sha256": sha(HERE),
               "result_sha256": hashlib.sha256(data).hexdigest(),
               "pass": result["pass"]}
    target.with_name("n8_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("%d/%d checks passed" %
          (result.get("n_checks", 0) - result.get("n_failed", 0),
           result.get("n_checks", 0)))
    if not result["pass"]:
        print("FAIL: " + result.get("error", repr(
            [c for c in result.get("checks", []) if not c["pass"]])))
        return 1
    print("n8 integrated FLOPs %d" % result["meter"]["flops"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
