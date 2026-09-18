"""Small, label-free checks for the continuation audit's refusal and binding."""
from __future__ import annotations

import unittest

import numpy as np

import s17_query_continuation as audit


class ContinuationAuditTest(unittest.TestCase):
    def test_displacement_and_hashes_bind_every_layer(self) -> None:
        exact = np.array([[1.0, 2.0], [3.0, 4.0]])
        approx = exact.copy()
        approx[1, 0] += 0.25
        report = audit.describe_predictions(exact, approx)
        self.assertEqual(report["shape"], [2, 2])
        self.assertEqual(report["layers"][0]["displacement_l2"], 0.0)
        self.assertAlmostEqual(report["layers"][1]["displacement_l2"], 0.25)
        self.assertEqual(report["layers"][0]["exact_sha256"],
                         report["layers"][0]["approx_sha256"])
        self.assertNotEqual(report["layers"][1]["exact_sha256"],
                            report["layers"][1]["approx_sha256"])
        self.assertEqual(report, audit.describe_predictions(exact, approx))

    def test_nonfinite_and_shape_refused(self) -> None:
        a = np.ones((2, 2))
        with self.assertRaisesRegex(ValueError, "shape"):
            audit.describe_predictions(a, a[:1])
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            audit.describe_predictions(a, np.array([[1.0, np.nan], [1.0, 1.0]]))


if __name__ == "__main__":
    unittest.main()
