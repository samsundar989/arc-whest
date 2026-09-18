"""Synthetic, label-safe checks for the one-arm offline reference scorer."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s18_reference_score as scorer  # noqa: E402
import design_sidecar as ds  # noqa: E402


def fixture(path, *, n=1, width=8, depth=6):
    provenance = {"schema": "p2_truth_math_e14_v1", "width": width,
                  "depth": depth, "mint_sha256": scorer.sha(HERE / "mint_provenance.py"),
                  "producer_sha256": scorer.sha(HERE / "p2_truth.py")}
    a = np.full((n, depth, width), 0.3, dtype=np.float32)
    b = np.full((n, depth, width), 0.5, dtype=np.float32)
    np.savez_compressed(path, seeds=np.arange(998_001, 998_001 + n, dtype=np.int64),
                        means_r1=a, means_r2=b,
                        var_r1=np.ones((n, width), dtype=np.float64),
                        var_r2=np.ones((n, width), dtype=np.float64),
                        gt_noise=np.full(n, 0.25, dtype=np.float64),
                        n_samples=np.int64(8), width=np.int64(width), depth=np.int64(depth),
                        producer_manifest_json=np.asarray(json.dumps(provenance, sort_keys=True)))
    Path(str(path) + ".producer.json").write_text(json.dumps({
        "artifact_sha256": scorer.sha(path), "math_provenance": provenance}), encoding="utf-8")


class TestReferenceScore(unittest.TestCase):
    def test_tiny_complete_and_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            truth = Path(tmp) / "truth_P99.npz"
            out = Path(tmp) / "score"
            fixture(truth)
            m = scorer.score_loaded("unshared", truth, out, block="P99", synthetic_fixture=True)
            self.assertEqual((m["arm"], m["n"], m["run_complete"]), ("unshared", 1, True))
            pred = ds.load_arm(str(out), "unshared", width=8, expect_n=1)["predictions"][0]
            target = np.full(8, 0.4, dtype=np.float32)
            self.assertEqual(m["mean_raw_mse"], float(np.mean((pred - target) ** 2)))
            self.assertEqual(m["mean_truth_noise_floor"], 0.125)
            row = json.loads((out / "rows.jsonl").read_text().strip())
            self.assertEqual(row["prediction_sha256"], hashlib.sha256(pred.tobytes()).hexdigest())
            self.assertEqual(row["flops_used"], sum(x["flops"] for x in row["operations"].values()))
            self.assertGreater(row["operation_calls"], 0)
            with self.assertRaises(FileExistsError):
                scorer.score_loaded("unshared", truth, out, block="P99", synthetic_fixture=True)
            p = Path(ds.stream_path(str(out), "unshared"))
            p.write_bytes(p.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                scorer.verify(out)

    def test_refusals_before_any_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            truth = Path(tmp) / "truth_P99.npz"
            fixture(truth)
            with self.assertRaises(ValueError):
                scorer.score_loaded("unknown", truth, Path(tmp) / "bad", block="P99")
            with self.assertRaises(ValueError):
                scorer.score_loaded("C10", truth, Path(tmp) / "p13", block="P13")
            side = Path(str(truth) + ".producer.json")
            record = json.loads(side.read_text())
            record["artifact_sha256"] = "0" * 64
            side.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                scorer.score_loaded("C10", truth, Path(tmp) / "changed", block="P99", synthetic_fixture=True)

    def test_capped_result_cannot_be_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            truth = Path(tmp) / "truth_P99.npz"
            out = Path(tmp) / "capped"
            fixture(truth)
            with patch.object(scorer.ref, "offline_predict", side_effect=RuntimeError("capped reference")):
                with self.assertRaisesRegex(RuntimeError, "capped"):
                    scorer.score_loaded("unshared", truth, out, block="P99", synthetic_fixture=True)
            with self.assertRaises(ValueError):
                scorer.verify(out)

    def test_changed_source_identity_refused(self):
        with patch.object(scorer.ref, "CONTROL_SHA", "0" * 64):
            with self.assertRaises(ValueError):
                scorer._source("C10")


if __name__ == "__main__":
    unittest.main()
