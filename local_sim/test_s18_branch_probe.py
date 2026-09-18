"""Focused label-free branch and refusal checks."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s18_branch_probe as probe


class BranchProbeTest(unittest.TestCase):
    def test_verified_join_and_birth_boundaries(self):
        join = probe.run(branch_layer=4)
        p2 = probe.run(branch_layer=0)
        self.assertEqual(set(join["arms"]), {"control", "unshared"})
        self.assertEqual(set(p2["arms"]), {"control", "full_p2", "unshared_full_p2"})
        for result in (join, p2):
            self.assertEqual({a["incoming_sha256"] for a in result["arms"].values()},
                             {result["incoming_sha256"]})
            for arm in result["arms"].values():
                self.assertGreater(arm["ledger"]["flops"], 0)
                self.assertGreater(arm["ledger"]["calls"], 0)
        self.assertEqual(join["arms"]["unshared"]["joins"], [])
        self.assertEqual(join["arms"]["control"]["joins"][0]["layer"], 4)
        self.assertEqual(p2["arms"]["full_p2"]["birth_ranks"][0]["rank"], 33)
        self.assertEqual(p2["arms"]["control"]["birth_ranks"][0]["rank"], 32)

    def test_refuse_wrong_boundary_and_changed_source(self):
        with self.assertRaisesRegex(ValueError, "branch"):
            probe.run(branch_layer=5)
        with patch.object(probe, "GENERATOR_SHA", "0" * 64):
            with self.assertRaisesRegex(RuntimeError, "generator changed"):
                probe.run()
        with patch.object(probe, "CONTROL_SHA", "0" * 64):
            with self.assertRaisesRegex(RuntimeError, "C10 source changed"):
                probe.run()

    def test_refuse_incomplete_ledger_and_nonfinite_state(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            probe._assert_ledger_conserved(3, {"add": {"flops": 2, "calls": 1}})
        import numpy as np
        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            probe.state_hash((np.array([float("nan")]),))

    def test_nested_cache_mutation_and_unsupported_type_refused(self):
        import numpy as np

        class TinyTower:
            __slots__ = ("_p2", "_hub")

            def __init__(self):
                self._p2 = (np.array([1.0], dtype=np.float32),
                            [None, {"cache": np.array([[2.0]], dtype=np.float64)}])
                self._hub = {"z": (3, True)}

        tower = TinyTower()
        state = (np.array([0.0], dtype=np.float32), tower)
        expected = probe.state_hash(state)
        self.assertEqual(probe.state_hash(state), expected)
        probe._assert_same_state(state, expected)
        tower._p2[1][1]["cache"][0, 0] = 4.0
        with self.assertRaisesRegex(RuntimeError, "incoming state changed"):
            probe._assert_same_state(state, expected)
        tower._hub = object()
        with self.assertRaisesRegex(TypeError, "unsupported state type"):
            probe.state_hash(state)


if __name__ == "__main__":
    unittest.main()
