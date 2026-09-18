"""Tiny, label-free tests for the source-bound complete-package fit."""
import hashlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s17_complete_fit as fit
import s17_query_cost as query

ROOT = Path(__file__).resolve().parents[1]


class CompleteFit(unittest.TestCase):
    def source(self, name):
        path = ROOT / name
        raw = path.read_bytes()
        src, center, has_controls = fit.selected(raw)
        return path, raw, src, center, has_controls

    def test_exact_selected_default_and_control_patching(self):
        for name, expected in (
                ("estimator_k3_v43_complete_c00.py", fit.old.X0),
                ("estimator_k3_v43_complete_c11.py", (.625, 1.225, -.1, 1.))):
            path, raw, src, center, has_controls = self.source(name)
            self.assertEqual(center, expected)
            self.assertEqual(fit.render(src, center, has_controls, center), raw)
            trial = fit.nearby(center, 0, 1, .25)
            changed = fit.render(src, center, has_controls, trial).decode()
            self.assertIn("_OWN_THETA_FAM", changed)
            self.assertIn("_P3_ON", changed)
            self.assertEqual(fit.plan(path, hashlib.sha256(raw).hexdigest())["scheduled_evaluations"], 19)
            with self.assertRaises(ValueError):
                fit.plan(path, "0" * 64)

    def test_centered_schedule_and_validation_order(self):
        center = (.625, 1.225, -.1, 1.)
        calls = []
        def evaluate(split, point, index):
            calls.append((split, point, index))
            return 1.0
        chosen, _, _, _, history = fit.coordinate_fit(evaluate, center)
        self.assertEqual(chosen, center)
        self.assertEqual(len(history), 19)
        self.assertEqual([c[0] for c in calls], ["fit"] * 17 + ["validation"] * 2)
        self.assertEqual([c[2] for c in calls], list(range(19)))
        self.assertEqual(calls[0][1], center)
        self.assertEqual(calls[-2][1], center)
        self.assertEqual(calls[-1][1], chosen)
        self.assertAlmostEqual(fit.objective(1., 1., center, center), 1.)
        self.assertGreater(fit.objective(1., 1., (.75, 1.225, -.1, 1.), center), 1.)

    def test_malformed_controls_fail_closed(self):
        _, raw, _, _, _ = self.source("estimator_k3_v43_complete_c11.py")
        broken = raw.replace(b"_OWN_T_LAM = -0.1\n", b"")
        with self.assertRaises(ValueError):
            fit.selected(broken)
        with self.assertRaises(ValueError):
            fit.selected(b"not a complete package")

    def test_query_wrapper_survives_trial(self):
        _, raw, _, _, _ = self.source("estimator_k3_v43_complete_c11.py")
        wrapped = query.build(raw.decode("utf-8"), rank=32).encode("utf-8")
        src, center, has_controls = fit.selected(wrapped)
        self.assertEqual(fit.render(src, center, has_controls, center), wrapped)
        trial = fit.render(src, center, has_controls,
                           fit.nearby(center, 0, 1, .25)).decode("utf-8")
        self.assertIn("_OLD_QUERY_RANK = 32", trial)
        self.assertIn("_OWN_THETA_FAM = 0.875", trial)


if __name__ == "__main__":
    unittest.main()
