"""Tiny CPU-only regression tests; never launch WSL or evaluate an estimator.

    python -m unittest discover -s tools -p test_ev_memory_gate.py
"""
import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ev


class MemoryGateEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.candidate = self.root / "candidate.py"
        self.candidate.write_text("# tiny candidate fixture\n", encoding="utf-8")
        self.sha = ev.sha256_file(self.candidate)
        self.output = self.root / "memory.json"
        self.good = {"status": "PASS", "candidate_sha256": self.sha}
        self.old_bytes = json.dumps(self.good).encode()
        self.output.write_bytes(self.old_bytes)

    def run_gate(self, *, rc=0, payload=None, raw=None, mutate=False):
        def fake_run(args, **kwargs):
            fresh = Path(args[args.index("--json") + 1])
            self.assertNotEqual(fresh, self.output)
            self.assertFalse(fresh.exists())
            if payload is not None:
                fresh.write_text(json.dumps(payload), encoding="utf-8")
            if raw is not None:
                fresh.write_text(raw, encoding="utf-8")
            if mutate:
                self.candidate.write_text("# changed during gate\n", encoding="utf-8")
            return subprocess.CompletedProcess(args, rc, "", "mock failure" if rc else "")

        with patch.object(ev.subprocess, "run", side_effect=fake_run):
            return ev.memory_gate(self.candidate, out=self.output)

    def assert_old_preserved(self):
        self.assertEqual(self.output.read_bytes(), self.old_bytes)

    def test_nonzero_exit_rejects_old_pass(self):
        self.assertEqual(self.run_gate(rc=2)["status"], "FAIL")
        self.assert_old_preserved()

    def test_nonzero_exit_rejects_even_new_pass(self):
        self.assertEqual(self.run_gate(rc=2, payload=self.good)["status"], "FAIL")
        self.assert_old_preserved()

    def test_success_without_fresh_json_rejects_old_pass(self):
        self.assertEqual(self.run_gate()["status"], "UNAVAILABLE")
        self.assert_old_preserved()

    def test_wrong_hash_rejected(self):
        bad = dict(self.good, candidate_sha256="0" * 64)
        self.assertEqual(self.run_gate(payload=bad)["status"], "FAIL")
        self.assert_old_preserved()

    def test_malformed_json_rejected(self):
        self.assertEqual(self.run_gate(raw="{")["status"], "FAIL")
        self.assert_old_preserved()

    def test_changed_candidate_rejected(self):
        self.assertEqual(self.run_gate(payload=self.good, mutate=True)["status"], "FAIL")
        self.assert_old_preserved()

    def test_fresh_matching_success_saved(self):
        self.assertEqual(self.run_gate(payload=self.good), self.good)
        self.assertEqual(json.loads(self.output.read_text()), self.good)

    def test_launch_error_rejected(self):
        with patch.object(ev.subprocess, "run", side_effect=OSError("cannot launch")):
            result = ev.memory_gate(self.candidate, out=self.output)
        self.assertEqual(result["status"], "UNAVAILABLE")
        self.assert_old_preserved()


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.row = {
            "projected_public": 1e-9, "ops": 23000, "setup_s": .01,
            "materializations": 0, "failed_nets": [], "panel": "dev", "n": 100,
            "baseline_name": "incumbent", "paired": {"n": 100},
            "projected_server_wall_s": 90, "projected_server_wall_noisy_s": 100,
            "candidate_sha256": "a" * 64,
            "memory": {"status": "PASS", "candidate_sha256": "a" * 64},
        }

    def memory_passes(self, row):
        return next(ok for name, _, _, ok in ev.gate_table(row, 2e-9)
                    if name.startswith("WSL RLIMIT_AS"))

    def test_matching_hash_passes(self):
        self.assertTrue(self.memory_passes(self.row))

    def test_wrong_or_missing_memory_hash_fails(self):
        for digest in (None, "b" * 64):
            with self.subTest(digest=digest):
                row = copy.deepcopy(self.row)
                row["memory"]["candidate_sha256"] = digest
                self.assertFalse(self.memory_passes(row))

    def test_missing_candidate_hash_fails_even_if_memory_passed(self):
        self.row.pop("candidate_sha256")
        self.assertFalse(self.memory_passes(self.row))

    def test_failed_memory_status_fails_even_if_hash_matches(self):
        self.row["memory"]["status"] = "FAIL"
        self.assertFalse(self.memory_passes(self.row))

    def test_incomplete_panel_pair_or_memory_fails(self):
        for field, value in (("n", 99), ("panel", "val"), ("paired", None), ("memory", None)):
            with self.subTest(field=field):
                row = copy.deepcopy(self.row)
                row[field] = value
                hard = [ok for name, _, _, ok in ev.gate_table(row, 2e-9)
                        if not name.startswith("projected server wall")]
                self.assertFalse(all(hard))


if __name__ == "__main__":
    unittest.main()
