"""Descriptive same-input decomposition of the first n8 negative prevariance.

This is not an adjudication gate. It leaves the frozen 186/187 FAIL intact.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "local_sim"))
import p2_k4_integrated_first_source as instrument  # noqa: E402
from test_p2_k4_integrated_first_source_n8 import expand_source  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def report():
    result_path = HERE.with_name("n8_result.json")
    frozen = json.loads(result_path.read_text(encoding="utf-8"))
    if frozen["pass"] or frozen["n_failed"] != 1:
        raise AssertionError("unexpected frozen n8 verdict")
    raw = instrument.integrated_metered_program(
        n=8, depth=16, source_rank=6, rank=8, recent=4,
        ordinary_residual=8, capture_tiny=True)
    traces = raw["_trace"]
    weights = raw["_inputs"][0]
    if (raw["input_sha256"] != frozen["input_sha256"] or
            raw["source_sha256"] != frozen["instrument_sha256"] or
            raw["lifecycle"]["output_sha256"] !=
            frozen["meter"]["lifecycle"]["output_sha256"]):
        raise AssertionError("same-input diagnostic did not reproduce frozen meter")
    layer = 2  # C at L2 is transported into the first negative prevar at L3.
    row = traces[layer]
    n = row["C"].shape[0]
    w1 = np.asarray(row["w1"], dtype=np.float64)
    w2 = np.asarray(row["w2"], dtype=np.float64)
    w3 = np.asarray(row["w3"], dtype=np.float64)
    co = np.asarray(row["C_off"], dtype=np.float64)
    off = np.ones((n, n), dtype=np.float64) - np.eye(n)
    parts = {"gaussian_diagonal": np.diag(np.asarray(row["var_out"], dtype=np.float64)),
             "gaussian_offdiagonal": off * (
                 co * w1[:, None] * w1[None, :]
                 + 0.5 * (co * co) * w2[:, None] * w2[None, :])}
    for kind in ("ordinary", "feed_positive", "feed_negative"):
        tensor = np.zeros((n, n, n), dtype=np.float64)
        for source_kind, source in row["sources"]:
            if source_kind == kind:
                tensor += expand_source(source)
        index = np.arange(n)
        d21 = tensor[index[:, None], index[:, None], index[None, :]]
        parts["k3_" + kind] = off * 0.5 * (
            d21 * w2[:, None] * w1[None, :]
            + d21.T * w1[:, None] * w2[None, :])
    d22 = np.asarray(row["D22"], dtype=np.float64)
    d31 = np.asarray(row["D31"], dtype=np.float64)
    parts["k4_bank"] = off * (
        d22 * w2[:, None] * w2[None, :] / 4.0
        + (d31 * w3[:, None] * w1[None, :]
           + d31.T * w1[:, None] * w3[None, :]) / 6.0)
    reconstructed = sum(parts.values())
    C2 = np.asarray(row["C"], dtype=np.float64)
    W3 = np.asarray(weights[3], dtype=np.float32).astype(np.float64)
    propagated = {name: np.diag(W3 @ matrix @ W3.T)
                  for name, matrix in parts.items()}
    prevar3 = sum(propagated.values())
    negative = np.flatnonzero(prevar3 <= 0)
    without = {}
    for omitted in ("k3_ordinary", "k3_feed_positive", "k3_feed_negative",
                    "k4_bank"):
        without[omitted] = prevar3 - propagated[omitted]
    include_groups = {
        "gaussian_only": propagated["gaussian_diagonal"]
                         + propagated["gaussian_offdiagonal"],
        "gaussian_plus_ordinary_k3": propagated["gaussian_diagonal"]
                         + propagated["gaussian_offdiagonal"]
                         + propagated["k3_ordinary"],
        "gaussian_plus_all_k3": prevar3 - propagated["k4_bank"],
        "gaussian_plus_k4_only": propagated["gaussian_diagonal"]
                         + propagated["gaussian_offdiagonal"]
                         + propagated["k4_bank"],
    }
    return {
        "kind": "descriptive same-input n8 L3 prevariance decomposition",
        "gate_status": "original n8 186/187 FAIL unchanged; no new acceptance gate",
        "diagnostic_source_sha256": sha(HERE),
        "frozen_test_sha256": frozen["source_sha256"],
        "instrument_sha256": raw["source_sha256"],
        "input_sha256": raw["input_sha256"],
        "frozen_result_sha256": sha(result_path),
        "l2_covariance_reconstruction_max_abs": float(np.max(np.abs(reconstructed - C2))),
        "l2_covariance_min_eigenvalue": float(np.linalg.eigvalsh((C2 + C2.T) / 2).min()),
        "l3_reconstructed_prevariance": prevar3.tolist(),
        "l3_negative_neurons": negative.tolist(),
        "l3_component_diagonals": {k: v.tolist() for k, v in propagated.items()},
        "l3_without_component": {k: v.tolist() for k, v in without.items()},
        "l3_component_group_sums": {k: v.tolist() for k, v in include_groups.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    args = parser.parse_args()
    value = report()
    output = Path(args.json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print("L2 C reconstruction max abs", value["l2_covariance_reconstruction_max_abs"])
    print("L2 C min eigenvalue", value["l2_covariance_min_eigenvalue"])
    print("L3 negative indices", value["l3_negative_neurons"])
    for name, numbers in value["l3_component_diagonals"].items():
        print(name, [round(numbers[i], 7) for i in value["l3_negative_neurons"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
