"""Small, label-free checks for the continuation audit's refusal and binding."""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

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

    def test_stack_drift_refused(self) -> None:
        with patch.object(audit.importlib.metadata, "version", side_effect=["0.0", "0.16.1"]):
            with self.assertRaisesRegex(RuntimeError, "stack drift"):
                audit._pinned_stack()

    def test_branch_trace_binds_layer_and_refuses_missing_layer(self) -> None:
        with TemporaryDirectory() as td:
            source = Path(td) / "query.py"
            source.write_text("\n".join([
                "old = _mm(_mm(LL, FF.T), tw.Q.T)",
                "old = _mm(_mm(LL, FF.T), tw.Q.T)",
                "old = _mm(U, _mm(_mm(_mm(U.T, LL), FF.T), tw.Q.T))",
            ]), encoding="utf-8")
            trace = audit.QueryTrace(source)
            layer = SimpleNamespace(f_code=SimpleNamespace(co_name="_layer", co_filename=str(source)),
                                    f_locals={"layer_index": 4}, f_back=None)
            query = SimpleNamespace(f_code=SimpleNamespace(co_name="_shared_contract_lean",
                                                         co_filename=str(source)),
                                    f_lineno=3, f_back=layer)
            trace.trace(query, "line", None)
            self.assertEqual(trace.summary()["first_approximate_layer"], 4)
            layer.f_locals.clear()
            with self.assertRaisesRegex(RuntimeError, "layer index"):
                trace.trace(query, "line", None)

    def test_no_approximate_branch_refused(self) -> None:
        trace = object.__new__(audit.QueryTrace)
        trace.events = [{"layer": 2, "branch": "rank_fallback"},
                        {"layer": 3, "branch": "deficient_fallback"}]
        with self.assertRaisesRegex(RuntimeError, "no approximate"):
            trace.summary()

    def test_displacement_before_first_approximate_branch_refused(self) -> None:
        exact = np.ones((4, 2))
        approx = exact.copy()
        approx[1, 0] += 0.01
        with self.assertRaisesRegex(RuntimeError, "before first"):
            audit.validate_prebranch_identity(exact, approx, 2)
        approx[1, 0] = 1.0
        approx[2, 0] += 0.01
        audit.validate_prebranch_identity(exact, approx, 2)

    def test_exclusive_output_refuses_overwrite(self) -> None:
        with TemporaryDirectory() as td:
            out = Path(td) / "report.json"
            audit.write_exclusive(out, {"a": 1})
            with self.assertRaises(FileExistsError):
                audit.write_exclusive(out, {"a": 2})
            self.assertEqual(out.read_text(encoding="utf-8"), '{\n  "a": 1\n}\n')
            self.assertEqual(list(Path(td).glob("*.tmp")), [])

    def test_changed_source_refused_after_predictions(self) -> None:
        source_records = [("exact.py", "a"), ("approx.py", "b"),
                          ("exact.py", "changed"), ("approx.py", "b")]
        with patch.object(audit, "_pinned_stack", return_value={"pin": "fixed"}), \
             patch.object(audit, "_source", side_effect=source_records), \
             patch.object(audit.anchor, "_mint_weights", return_value=([np.ones((2, 2))], 3)), \
             patch.object(audit, "QueryTrace"), \
             patch.object(audit, "_predict", return_value=(np.ones((2, 2)), 10.0)):
            with self.assertRaisesRegex(RuntimeError, "source changed"):
                audit.run(Path("exact.py"), Path("approx.py"))


if __name__ == "__main__":
    unittest.main()
