"""Tiny reusable-input E14 journal tests, including the actual F21 scorer path.

Run with one BLAS thread: uv run python local_sim/test_evidence_journal.py
No protected truth is loaded, no seed block is claimed, no candidate is promoted.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

_HERE = Path(__file__).resolve().parent
sys.path = [p for p in sys.path if Path(p or ".").resolve() != _HERE]
sys.path.append(str(_HERE))

import numpy as np
import evidence_journal as ej
import f21_score as scorer
import lifecycle_eval as le


def _fixture(root, mode="ok"):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    paths = {}
    for arm in ("L4", "fused_L4", "L2"):
        path = root / (arm + ".py")
        setup = "raise RuntimeError('synthetic setup failure')" if mode == "setup" and arm == "fused_L4" else "pass"
        predict = "raise ArithmeticError('synthetic row failure')" if mode == "row" and arm == "fused_L4" else "return fnp.zeros((len(mlp.weights), mlp.width), dtype=fnp.float32)"
        path.write_text("import flopscope.numpy as fnp\nclass Estimator:\n"
                        f"    def setup(self, context): {setup}\n"
                        f"    def predict(self, mlp, budget): {predict}\n", encoding="utf-8")
        paths[arm] = (path, ej.sha_file(path))
    truth_file = root / "synthetic-truth.txt"
    truth_file.write_text("reusable tiny synthetic inputs, no protected truth", encoding="utf-8")
    truth_prov = {"files": [{"path": str(truth_file), "sha256": ej.sha_file(truth_file)}],
                  "local_samples_total": 2048}
    seeds = [0, 1, 2, 3]
    truth = np.asarray([[.1, .2], [.3, .1], [.15, .12], [.07, .1]], dtype=np.float32)
    noise = np.full(4, 1e-6)
    return paths, truth_prov, seeds, truth, noise


def _journal(root, fixture):
    paths, prov, seeds, truth, noise = fixture
    return ej.EvidenceJournal(
        Path(root) / "result.json",
        candidates={a: {"path": str(p), "sha256": h} for a, (p, h) in paths.items()},
        instrument_files=[ej.__file__, scorer.__file__, le.__file__],
        truth_files=prov["files"], expected_per_arm=4,
        config={"test_only": True, "no_protected_truth": True})


def _valid_row(fixture, arm='L4', index=0):
    paths, prov, seeds, truth, noise = fixture
    return dataclasses.asdict(le.EvidenceRow(
        index=index, input_seed=seeds[index], estimator_seed=0,
        candidate=str(paths[arm][0]), candidate_sha256=paths[arm][1],
        truth_sha256=le._array_sha(truth), width=2, depth=1, budget=1e8,
        flops_used=0., n_ops=0., mse=.01, zero_prediction_mse=.01,
        gt_noise=1e-6, residual_wall_time_s=0., wall_time_s=0.))


def run_synthetic(root, mode="ok"):
    fixture = _fixture(root, mode)
    paths, prov, seeds, truth, noise = fixture
    tiny_spec = le.LifecycleSpec.for_round("v2-phase2", width=2, depth=1,
                                        residual_wall_time_limit_s=None,
                                        wall_time_limit_s=None, setup_timeout_s=30)
    args = argparse.Namespace(tier="design")
    with _journal(root, fixture) as journal:
        original_append, original_event = journal.append_row, journal.event
        if mode == "mid_arm":
            def append(arm, row):
                original_append(arm, row)
                if arm == "L4" and row.index == 1:
                    os._exit(73)
            journal.append_row = append
        if mode == "after_arm":
            def event(category, **details):
                original_event(category, **details)
                if category == "arm_completed":
                    os._exit(73)
            journal.event = event
        if mode == "publication":
            def publish(payload):
                journal.stage = "publication"
                journal.event("publication_started")
                journal.assert_complete()
                ej._atomic_new(journal.output, payload)
                os._exit(73)
            journal.publish = publish
        with mock.patch.object(scorer, "_labels", return_value=(seeds, truth, noise, prov)), \
             mock.patch.object(le.LifecycleSpec, "for_round", return_value=tiny_spec), \
             mock.patch.object(scorer.truth_reads, "load_truth", side_effect=AssertionError("protected read forbidden")):
            scorer._run(args, Path(root), Path(root)/"unused_reads", Path(root)/"unused_ledger",
                        paths, journal.output, journal)


class JournalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="e14-journal-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_production_scorer_complete_and_report_hash_refusal(self):
        with contextlib.redirect_stdout(io.StringIO()):
            run_synthetic(self.root)
        manifest = ej.verify_complete(str(self.root / "result.json") + ".journal")
        self.assertEqual(manifest["completed_rows"], 12)
        self.assertFalse(manifest["failed_rows"])
        report = json.loads((self.root / "result.json").read_text())
        self.assertFalse(report["verdict"]["promotion"])
        (self.root / "result.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "report hash"):
            ej.verify_complete(str(self.root / "result.json") + ".journal")

    def test_setup_and_row_failure_keep_actual_prior_observations(self):
        for mode, count in (("setup", 4), ("row", 8)):
            root = self.root / mode
            with self.assertRaises((le.SetupFailure, ValueError)), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                run_synthetic(root, mode)
            directory = Path(str(root / "result.json") + ".journal")
            manifest = ej._read(directory / "manifest.json")
            self.assertEqual(manifest["status"], "FAILED")
            self.assertEqual(manifest["completed_rows"], count)
            if mode == "setup":
                self.assertIn("setup_failure", [e["category"] for e in manifest["events"]])
            else:
                self.assertEqual(len(manifest["failed_rows"]), 4)
                self.assertIn("estimator_error:ArithmeticError", manifest["failed_rows"][0]["failure_categories"])
                self.assertGreater(manifest["failed_rows"][0]["row"]["zero_prediction_mse"], 0)
            with self.assertRaisesRegex(ValueError, "non-COMPLETE"):
                ej.verify_complete(directory)
            self.assertFalse((root / "result.json").exists())

    def test_abrupt_death_mid_arm_after_arm_during_publication(self):
        for mode, expected in (("mid_arm", 2), ("after_arm", 4), ("publication", 12)):
            root = self.root / mode
            process = subprocess.run([sys.executable, __file__, "--child", str(root), mode],
                                     env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                                          "MKL_NUM_THREADS": "1"}, capture_output=True, text=True, timeout=45)
            self.assertEqual(process.returncode, 73, process.stderr)
            directory = Path(str(root / "result.json") + ".journal")
            terminal = directory / "manifest.json"
            deadline = time.monotonic() + 15
            while not terminal.exists() and time.monotonic() < deadline:
                time.sleep(.025)
            self.assertTrue(terminal.exists())
            manifest = ej._read(terminal)
            self.assertEqual(manifest["status"], "INTERRUPTED")
            self.assertEqual(manifest["completed_rows"], expected)
            with self.assertRaisesRegex(ValueError, "non-COMPLETE"):
                ej.verify_complete(directory)

    def test_exclusive_output_reservation_before_any_second_run(self):
        fixture = _fixture(self.root)
        with _journal(self.root, fixture):
            with self.assertRaises(FileExistsError):
                _journal(self.root, fixture)

    def test_duplicate_incomplete_and_binding_mismatches(self):
        fixture = _fixture(self.root)
        paths, prov, seeds, truth, noise = fixture
        row = _valid_row(fixture)
        with _journal(self.root, fixture) as journal:
            journal.bind(seeds, le._array_sha(truth), prov)
            journal.append_row("L4", row)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                journal.assert_complete()
            with self.assertRaisesRegex(ValueError, "duplicate"):
                journal.append_row("L4", row)
            self.assertEqual(ej._scan(journal.directory)[1][-1]["data"]["category"], "rejected_row")
            for field in ("candidate_sha256", "truth_sha256"):
                bad = {**row, "index": 1, "input_seed": 1, field: "0" * 64}
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    journal.append_row("L4", bad)

    def test_chain_torn_tail_and_manifest_mutations_refused(self):
        with contextlib.redirect_stdout(io.StringIO()):
            run_synthetic(self.root)
        directory = Path(str(self.root / "result.json") + ".journal")
        terminal_path = directory / "manifest.json"
        original_terminal = terminal_path.read_bytes()
        terminal = ej._read(terminal_path)
        terminal["completed_rows"] = 11
        terminal_path.write_bytes(ej._bytes(terminal))
        with self.assertRaisesRegex(ValueError, "manifest"):
            ej.verify_complete(directory)
        terminal_path.write_bytes(original_terminal)
        row_path = directory / "rows.jsonl"
        original_rows = row_path.read_bytes()
        row_path.write_bytes(original_rows + b'{"partial":')
        with self.assertRaisesRegex(ValueError, "torn"):
            ej.verify_complete(directory)
        row_path.write_bytes(original_rows.replace(b'"input_seed":0', b'"input_seed":9', 1))
        with self.assertRaisesRegex(ValueError, "hash/chain"):
            ej.verify_complete(directory)

    def test_instrument_changed_before_publication_refused(self):
        fixture = _fixture(self.root)
        with _journal(self.root, fixture) as journal:
            # Avoid real source mutation: inject mismatch in the in-memory pinned
            # reference used by publication, after complete synthetic rows.
            paths, prov, seeds, truth, noise = fixture
            journal.bind(seeds, le._array_sha(truth), prov)
            for arm in paths:
                for i, seed in enumerate(seeds):
                    journal.append_row(arm, _valid_row(fixture, arm, i))
            instrument = next(iter(journal.header["instrument_hashes"]))
            journal.header["instrument_hashes"][instrument] = "0" * 64
            with self.assertRaisesRegex(ValueError, "instrument hash"):
                journal.publish({"promotion": True})
            self.assertFalse(journal.output.exists())

    def test_valid_chain_duplicate_and_truth_binding_mutation_refused(self):
        fixture = _fixture(self.root)
        paths, prov, seeds, truth, noise = fixture
        with _journal(self.root, fixture) as journal:
            journal.bind(seeds, le._array_sha(truth), prov)
            row = _valid_row(fixture)
            journal.append_row("L4", row)
            # Corrupt writer bypasses the API but computes a valid chain.
            journal._append("row", {"arm": "L4", "row": row, "failure_categories": []})
            with self.assertRaisesRegex(ValueError, "duplicate"):
                ej._validate_complete(journal.directory)
            binding = ej._read(journal.directory / "binding.json")
            binding["truth_sha256"] = "0" * 64
            (journal.directory / "binding.json").write_bytes(ej._bytes(binding))
            with self.assertRaisesRegex(ValueError, "truth hash"):
                ej._validate_complete(journal.directory)

    def test_pending_truth_hash_binds_without_reading_truth_bytes(self):
        fixture = _fixture(self.root)
        paths, prov, seeds, truth, noise = fixture
        pending = {**prov, "files": [{**prov["files"][0], "sha256": None}]}
        pending_fixture = (paths, pending, seeds, truth, noise)
        truth_path = Path(prov["files"][0]["path"])
        truth_path.unlink()  # reservation/binding must work without opening it
        with _journal(self.root, pending_fixture) as journal:
            journal.bind(seeds, le._array_sha(truth), prov)
            self.assertEqual(journal.binding["truth_provenance"]["files"][0]["sha256"],
                             prov["files"][0]["sha256"])

    def test_complete_row_schema_mutations_refused(self):
        fixture = _fixture(self.root)
        paths, prov, seeds, truth, noise = fixture
        for index, mutation in enumerate(('missing_flags', 'identity_only', 'tagged_nan', 'int_flag', 'nan', 'negative')):
            root = self.root / mutation
            changed_fixture = _fixture(root)
            with _journal(root, changed_fixture) as journal:
                paths, prov, seeds, truth, noise = changed_fixture
                journal.bind(seeds, le._array_sha(truth), prov)
                for arm in paths:
                    for i in range(4):
                        row = _valid_row(changed_fixture, arm, i)
                        if i == 0 and arm == 'L4':
                            if mutation == 'missing_flags': del row['nonfinite']
                            elif mutation == 'identity_only': row = {k: row[k] for k in ('index','input_seed','candidate_sha256','truth_sha256')}
                            elif mutation == 'tagged_nan': row['mse'] = {'nonfinite_float': 'nan'}
                            elif mutation == 'int_flag': row['time_exhausted'] = 0
                            elif mutation == 'nan': row['mse'] = float('nan')
                            elif mutation == 'negative': row['flops_used'] = -1
                        # Simulate a corrupted writer with a valid hash chain.
                        journal._append('row', {'arm': arm, 'row': row, 'failure_categories': []})
                with self.assertRaises(ValueError): journal.assert_complete()

    def test_decide_checks_spent_read_before_any_truth_hash(self):
        args = argparse.Namespace(tier='decide', truth='synthetic-no-file.npz', out=str(self.root/'out.json'))
        final = argparse.Namespace(block='P10', artifact=args.truth, artifact_sha256='1'*64)
        with mock.patch.object(scorer.seed_ledger, 'finalizations', return_value=[final]), \
             mock.patch.object(scorer, '_sha', side_effect=AssertionError('premature truth hash')), \
             mock.patch.object(scorer, '_read', side_effect=PermissionError('SPENT')) as read:
            with self.assertRaisesRegex(PermissionError, 'SPENT'):
                scorer._labels(args, self.root, self.root/'reads', self.root/'ledger')
            read.assert_called_once()


def write_artifacts(directory):
    """Persist synthetic-only example journals for independent raw inspection."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    results = []
    for mode in ("ok", "setup", "row", "mid_arm", "after_arm", "publication"):
        root = directory / ("journal_" + mode)
        proc = subprocess.run([sys.executable, __file__, "--child", str(root), mode],
                              capture_output=True, text=True, timeout=45)
        terminal = Path(str(root / "result.json") + ".journal") / "manifest.json"
        deadline = time.monotonic() + 15
        while not terminal.exists() and time.monotonic() < deadline:
            time.sleep(.025)
        manifest = ej._read(terminal)
        results.append({"scenario": mode, "exit_code": proc.returncode,
                        "status": manifest["status"], "completed_rows": manifest["completed_rows"],
                        "manifest": str(terminal), "manifest_sha256": ej.sha_file(terminal),
                        "stderr": proc.stderr})
    ej._atomic_new(directory / "journal_synthetic_results.json", {
        "test_only": True, "protected_reads": 0,
        "fixture": "width2 depth1, four reusable synthetic rows, three zero-predictor arms",
        "cases": results,
        "limitations": ["machine/guardian death can leave no terminal and is refused",
                        "process-crash safety, not a filesystem power-loss guarantee",
                        "no resume or protected reread authority"]})


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--child":
        run_synthetic(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 3 and sys.argv[1] == "--artifacts":
        write_artifacts(sys.argv[2])
    else:
        unittest.main(verbosity=2)
