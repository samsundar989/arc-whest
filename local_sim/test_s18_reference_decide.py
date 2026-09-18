"""Synthetic-only checks of the P15 read-only paired reader."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s18_reference_decide as reader  # noqa: E402
import s18_reference_score as scorer  # noqa: E402
from test_s18_reference_score import fixture  # noqa: E402


class TestReferenceDecide(unittest.TestCase):
    def test_fixed_cost_gate_and_noise_cancellation(self):
        control = [{"mse": x, "gt_noise_averaged": .2} for x in (2., 3., 4., 5.)]
        candidate = [{"mse": x, "gt_noise_averaged": .2} for x in (1., 2., 3., 4.)]
        got = reader._comparison(control, candidate, binding=True)
        self.assertTrue(got["isolated_accuracy_gate"]["passed"])
        self.assertAlmostEqual(got["paired_raw_difference"], .1)
        self.assertAlmostEqual(got["noise_corrected_true_function_difference"], .1)
        self.assertAlmostEqual(got["mean_midpoint_truth_noise_floor_fixed_cost"], .02)
        self.assertNotIn("isolated_accuracy_gate", reader._comparison(control, candidate, binding=False))
        candidate[0]["gt_noise_averaged"] = .3
        with self.assertRaisesRegex(ValueError, "different truth noise"):
            reader._comparison(control, candidate, binding=True)

    def test_real_tiny_four_arm_sidecars_and_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            truth = Path(tmp) / "truth_P99.npz"
            fixture(truth, n=2)
            paths = [Path(tmp) / name for name in reader.ROLES]
            for name, path in zip(reader.ROLES.values(), paths):
                scorer.score_loaded(name, truth, path, block="P99", synthetic_fixture=True)
            with (patch.object(reader, "N", 2), patch.object(reader, "WIDTH", 8),
                  patch.object(reader, "DEPTH", 6), patch.object(reader, "SAMPLES", 8),
                  patch.object(reader, "P15_SEEDS", [998_001, 998_002]),
                  patch.object(reader.scorer, "verify", wraps=scorer.verify) as verify):
                # Fixture provenance uses P99. Permit only this synthetic block here.
                original = reader.scorer.verify
                def synthetic_verify(p):
                    m = original(p)
                    m["block"] = "P15"
                    return m
                with patch.object(reader.scorer, "verify", side_effect=synthetic_verify):
                    out = reader.synthesize(*paths)
                    self.assertEqual(out["n"], 2)
                    self.assertEqual(len(out["comparisons"]), 3)
                    self.assertNotIn("isolated_accuracy_gate", out["comparisons"]["combined"])
                    # Changing a saved prediction must refuse even if rows remain intact.
                    stream = paths[1] / "preds_unshared.f32"
                    stream.write_bytes(stream.read_bytes()[:-1])
                    with self.assertRaises(ValueError):
                        reader.synthesize(*paths)


if __name__ == "__main__":
    unittest.main()
