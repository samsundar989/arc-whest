"""E14 truth math tests; --pilot is heavy and must run through workerq.

No truth files, seed ledgers, estimator predictions or protected data are read.
The independent reference follows installed whestbench/simulation.py and
_simulation_torch.py: f32 forward, f64 sums/squared sums, f32 mean export.
All-layer pre-export-centered variance is an explicit diagnostic extension.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
import tempfile
import unittest

# Keep even the CPU reference bounded to one BLAS thread before importing it.
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import torch
import p2_truth as producer
from mint_provenance import canonical_mint

torch.set_num_threads(1)

RAW_SCALE = 2e-8
SCORE_BOUND_LIMIT = 1e-10
MOMENT_ATOL_RTOL = 2e-6


def chunks(inputs, chunk):
    for start in range(0, len(inputs), chunk):
        yield inputs[start:start + chunk]


def reference(weights, inputs, chunk):
    sums = np.zeros((len(weights), len(weights[0])), dtype=np.float64)
    squares = np.zeros_like(sums)
    for batch in chunks(inputs, chunk):
        h = batch
        for layer, weight in enumerate(weights):
            h = np.maximum(np.matmul(h, weight), np.float32(0))
            if h.dtype != np.float32:
                raise AssertionError("reference forward dtype drift")
            h64 = h.astype(np.float64)
            sums[layer] += np.sum(h64, axis=0)
            squares[layer] += np.sum(h64 * h64, axis=0)
    means = sums / len(inputs)
    second = squares / len(inputs)
    variances = np.maximum(second - means * means, 0)
    exported = means.astype(np.float32)
    official_avg_variance = float(np.mean(second[-1] - exported[-1].astype(np.float64)**2))
    return means, variances, official_avg_variance


def compare(weights, inputs, chunk, device):
    expected_means, expected_vars, official_var = reference(weights, inputs, chunk)
    means, final_var, variances = producer._truth_one(
        weights, 123, len(inputs), device, chunk, input_chunks=chunks(inputs, chunk))
    delta = means.astype(np.float32).astype(np.float64) - expected_means.astype(np.float32).astype(np.float64)
    disagreement = float(np.mean(delta[-1]**2))
    bound = 2 * np.sqrt(RAW_SCALE * disagreement) + disagreement
    mean_norm = np.max(np.abs(means - expected_means) / (1 + np.abs(expected_means)), axis=1)
    var_norm = np.max(np.abs(variances - expected_vars) / (1 + np.abs(expected_vars)), axis=1)
    return {
        "chunk": chunk, "per_layer_mean_normalized_max": mean_norm.tolist(),
        "per_layer_variance_normalized_max": var_norm.tolist(),
        "final_export_arithmetic_disagreement_squared": disagreement,
        "conditional_raw_score_error_bound": float(bound),
        "reference_official_avg_variance_export_centered": official_var,
        "reference_avg_variance_pre_export_centered": float(expected_vars[-1].mean()),
        "production_avg_variance_pre_export_centered": float(final_var.mean()),
        "last_variance_identity": bool(np.array_equal(final_var, variances[-1])),
        "passed": bool(np.all(mean_norm <= MOMENT_ATOL_RTOL)
                       and np.all(var_norm <= MOMENT_ATOL_RTOL)
                       and bound <= SCORE_BOUND_LIMIT
                       and np.array_equal(final_var, variances[-1])),
    }


class TruthMathTests(unittest.TestCase):
    def setUp(self):
        producer._pin_truth_math()
        self.weights = [np.asarray([[1, -.25], [.5, 1]], dtype=np.float32)] * 3
        self.inputs = np.asarray([[1, -2], [-1, 3], [0, 1], [2, 2], [-2, -2]], dtype=np.float32)

    def tearDown(self):
        producer._pin_truth_math()

    def test_actual_producer_common_inputs_and_ragged_chunk(self):
        for chunk in (2, 5):
            self.assertTrue(compare(self.weights, self.inputs, chunk, "cpu")["passed"])

    def test_default_rng_route_matches_shared_inputs(self):
        gen = torch.Generator(device="cpu").manual_seed(123)
        shared = [torch.randn((n, 2), generator=gen, dtype=torch.float32).numpy() for n in (2, 2, 1)]
        random = producer._truth_one(self.weights, 123, 5, "cpu", 2)
        injected = producer._truth_one(self.weights, 123, 5, "cpu", 2, input_chunks=shared)
        for a, b in zip(random, injected):
            np.testing.assert_array_equal(a, b)

    def test_mid_forward_backend_mutation_refused(self):
        def mutated():
            torch.backends.cuda.matmul.allow_tf32 = True
            yield self.inputs
        with self.assertRaisesRegex(RuntimeError, "policy changed"):
            producer._truth_one(self.weights, 1, 5, "cpu", 5, input_chunks=mutated())

    def test_autocast_refused(self):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            with self.assertRaisesRegex(RuntimeError, "autocast"):
                producer._truth_one(self.weights, 1, 5, "cpu", 5)

    def test_changed_bindings_refused(self):
        original = producer.truth_provenance("cpu", 5, width=2, depth=3)
        for field in ("producer_sha256", "torch", "device", "chunk_size"):
            changed = copy.deepcopy(original)
            changed[field] = "injected mutation"
            with self.assertRaisesRegex(RuntimeError, "binding changed"):
                producer._truth_one(self.weights, 1, 5, "cpu", 5, expected_provenance=changed)

    def test_export_and_sidecar_binding(self):
        provenance = producer.truth_provenance("cpu", 5, width=2, depth=3)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "synthetic.npz")
            np.savez(path, producer_manifest_json=np.asarray(json.dumps(provenance)))
            producer._write_truth_sidecar(path, provenance)
            record = json.loads(Path(path + ".producer.json").read_text())
            self.assertEqual(record["artifact_sha256"], hashlib.sha256(Path(path).read_bytes()).hexdigest())
            with np.load(path, allow_pickle=False) as saved:
                self.assertEqual(json.loads(str(saved["producer_manifest_json"])), record["math_provenance"])
            with self.assertRaises(FileExistsError):
                producer._write_truth_sidecar(path, provenance)

    def test_bad_input_shape_and_dtype_refused(self):
        for inputs in (self.inputs.astype(np.float64), self.inputs[:2]):
            with self.assertRaisesRegex(ValueError, "shape/dtype"):
                producer._truth_one(self.weights, 1, 5, "cpu", 5, input_chunks=[inputs])


def pilot(output):
    path = Path(output)
    if path.exists():
        raise RuntimeError("REFUSED: immutable pilot artifact already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA pilot requires CUDA; no silent CPU fallback")
    weights, _, _ = canonical_mint(1024, 16, 1_000_051)
    inputs = np.random.default_rng(140914).standard_normal((65537, 1024), dtype=np.float32)
    producer._pin_truth_math()
    evidence = Path(__file__).resolve().parents[1] / "experiments/2026-09-14_e14"
    refs = {name: hashlib.sha256((evidence / name).read_bytes()).hexdigest()
            for name in ("truth_official_simulation.py", "truth_official_simulation_torch.py")}
    record = {
        "schema": "truth_common_input_e14_v1", "scope": "arithmetic pilot; no MC truth/accuracy score",
        "probe_seed": 1_000_051, "input_fixture_seed": 140914, "samples": 65537,
        "input_sha256": hashlib.sha256(inputs.tobytes()).hexdigest(),
        "weight_sha256": hashlib.sha256(b"".join(w.tobytes() for w in weights)).hexdigest(),
        "test_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "official_reference_source_sha256": refs,
        "reference": "independent NumPy f32 forward/f64 reductions/f32 export; all-layer variance extension",
        "reference_numpy_config": str(np.show_config(mode="dicts")),
        "thresholds": {"moment_normalized_max": MOMENT_ATOL_RTOL,
                       "conditional_raw_error_scale": RAW_SCALE,
                       "score_error_bound_max": SCORE_BOUND_LIMIT},
        "bound_scope": "Cauchy bound conditional on reference residual MSE <=2e-8 and multiplier <=1, on these inputs only; not a population bias guarantee",
        "sampling_noise": "common inputs eliminate independent-sample noise from the arithmetic difference; no ground-truth MSE measured",
        "rows": [],
    }
    for chunk in (65536, 8192):
        row = compare(weights, inputs, chunk, "cuda")
        row["provenance"] = producer.truth_provenance("cuda", chunk)
        record["rows"].append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    record["passed"] = all(row["passed"] for row in record["rows"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(record, f, indent=2, sort_keys=True)
        f.write("\n")
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--out", default="experiments/2026-09-14_e14/truth_pilot.json")
    args = parser.parse_args()
    if args.pilot:
        raise SystemExit(pilot(args.out))
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TruthMathTests)
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
