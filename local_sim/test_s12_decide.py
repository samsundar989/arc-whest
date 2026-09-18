"""Synthetic regression fixtures for the registered stage-3 verdict adapter."""
import json
import hashlib
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "local_sim"))
import s12_decide as s12  # noqa: E402
from s10_parity import _retained_energy, _wicked_query  # noqa: E402
import numpy as np  # noqa: E402


def fixture():
    arms = {}
    for name, path in s12.NAMES.items():
        mse, mult = (.011, .2) if name == "qoff" else (.020, .1)
        digest = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        arms[name] = {"path": path, "candidate_sha256": digest, "rows": [
            {"index": i, "seed": i, "mse": mse, "flops_used": mult * s12.B,
             "failed": False, "nonfinite": False, "gt_noise": .01}
            for i in range(24)]}
    query = {"network": {"width": 1024, "depth": 16},
             "overrides": {"age": 4, "rank": 256}, "arms": {name: {
        "path": s12.NAMES[name], "sha256": arms[name]["candidate_sha256"],
        "final_row_rel_vs_uncompressed": 2 if name == "qoff" else 1,
        "all_rows_rel_vs_uncompressed": 2 if name == "qoff" else 1,
        "c_over_b": .2,
        "per_layer": [
            {"layer": layer, **{k: 2 if name == "qoff" else 1 for k in
                (("mu_out", "cov_out") if layer == 0 else
                 ("D3", "mu_out") if layer == 15 else
                 ("E", "D3", "D21", "mu_out", "cov_out"))}}
            for layer in range(16)],
    } for name in s12.NAMES}}
    return {"arms": arms, "width": 1024, "depth": 16,
            "n_samples_per_replica": 2**24}, query


def run(score, query):
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        sp, qp, out = (root / x for x in ("score.json", "query.json", "out.json"))
        sp.write_text(json.dumps(score), encoding="utf-8")
        qp.write_text(json.dumps(query), encoding="utf-8")
        proc = subprocess.run([sys.executable, str(ROOT / "local_sim/s12_decide.py"),
                               "--score", str(sp), "--qe-a4", str(qp), "--out", str(out)],
                              cwd=ROOT, text=True, capture_output=True)
        return proc, json.loads(out.read_text(encoding="utf-8")) if out.exists() else None


class Stage3Verdict(unittest.TestCase):
    def test_official_precision_reverses_observed_false_pass(self):
        score, query = fixture()
        proc, out = run(score, query)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertGreater(out["comparisons"]["qE"]["observed_gain"], 1)
        self.assertLess(out["comparisons"]["qE"]["S1_gain"], 1)
        self.assertFalse(out["comparisons"]["qE"]["S1_adjusted_decide"])

    def test_mismatched_duplicate_and_missing_identities_refuse(self):
        for mutation in ("seed", "index", "duplicate", "missing"):
            with self.subTest(mutation=mutation):
                score, query = fixture()
                rows = score["arms"]["qE"]["rows"]
                if mutation == "seed":
                    for row in rows:
                        row["seed"] += 100
                elif mutation == "index":
                    rows[0]["index"] = 99
                elif mutation == "duplicate":
                    rows[0] = dict(rows[1])
                else:
                    rows.pop()
                proc, _ = run(score, query)
                self.assertNotEqual(proc.returncode, 0)

    def test_changed_failure_identity_fails_s2(self):
        score, query = fixture()
        score["arms"]["qoff"]["rows"][0]["failed"] = True
        score["arms"]["qE"]["rows"][1]["failed"] = True
        proc, out = run(score, query)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertFalse(out["comparisons"]["qE"]["S2_no_new_failures"])
        self.assertEqual(out["comparisons"]["qE"]["S2_new_failed_identities"], ["seed:1"])

    def test_missing_E_is_not_evidence(self):
        score, query = fixture()
        for arm in query["arms"].values():
            del arm["per_layer"][1]["E"]
        proc, out = run(score, query)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertFalse(out["comparisons"]["qE"]["S3_query_error_reduced"])
        self.assertIn("E", out["comparisons"]["qE"]["S3_detail"]["missing"])

    def test_truncated_query_layers_are_not_complete_evidence(self):
        score, query = fixture()
        for arm in query["arms"].values():
            arm["per_layer"].pop()
        proc, out = run(score, query)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertFalse(out["comparisons"]["qE"]["S3_query_error_reduced"])
        self.assertIn("per_layer", out["comparisons"]["qE"]["S3_detail"]["missing"])

    def test_stale_query_source_digest_refused(self):
        score, query = fixture()
        query["arms"]["qE"]["sha256"] = "0" * 64
        proc, _ = run(score, query)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("SHA differs", proc.stderr)

    def test_missing_shape_override_and_exact_path_refused(self):
        for mutation in ("shape", "override", "path"):
            with self.subTest(mutation=mutation):
                score, query = fixture()
                if mutation == "shape":
                    del score["width"]
                elif mutation == "override":
                    del query["overrides"]
                else:
                    query["arms"]["qE"]["path"] = "old/" + s12.NAMES["qE"]
                proc, _ = run(score, query)
                self.assertNotEqual(proc.returncode, 0)

    def test_same_state_energy_is_orthogonal_projection(self):
        E = np.array([[2., 1., 3.], [1., 4., 0.]])
        Q = np.array([[1.], [0.], [0.]])
        kept, total = _retained_energy(E, Q)
        self.assertAlmostEqual(kept, 5.)
        self.assertAlmostEqual(total, 31.)
        self.assertAlmostEqual(total - kept, np.linalg.norm(E - (E @ Q) @ Q.T)**2)

    def test_wick_query_uses_nonuniform_left_square_and_old_lineage(self):
        U = np.array([[1., 2.], [3., 4.]])
        J = np.array([[2., 0.], [1., 5.]])
        old = np.array([[3.], [2.]])
        old_basis = np.array([[1.], [4.]])
        gate = np.array([2., 3.])
        direct = np.diag(gate**2) @ (U @ J.T + old @ old_basis.T) @ np.diag(gate)
        np.testing.assert_allclose(_wicked_query(U, J, old, old_basis, gate), direct)

    def test_producer_emits_E_and_energy_from_tiny_chain(self):
        from s10_parity import query_error
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "query.json"
            result = query_error(16, 5, 90_000_016_001, str(path), age=2, rank=4)
            self.assertEqual(result["network"]["width"], 16)
            for arm in result["arms"].values():
                self.assertTrue(any(row.get("E") is not None for row in arm["per_layer"]))
                self.assertTrue(arm["query_energy_joins"])
                self.assertGreaterEqual(arm["retained_query_energy"], 0)
                self.assertLessEqual(arm["retained_query_energy"], 1.00001)

    def test_query_source_overrides_preserve_defaults_and_refuse_partial(self):
        from s10_parity import _query_sources, query_error
        defaults = _query_sources()
        with tempfile.TemporaryDirectory() as tmp:
            a = query_error(16, 5, 90_000_016_001,
                            str(pathlib.Path(tmp) / "default.json"), age=2, rank=4)
            b = query_error(16, 5, 90_000_016_001,
                            str(pathlib.Path(tmp) / "explicit.json"), age=2, rank=4,
                            sources=defaults)
        self.assertEqual(a, b)
        with self.assertRaisesRegex(ValueError, "together"):
            _query_sources({"qE": defaults["qE"]})
        with self.assertRaisesRegex(ValueError, "existing repository"):
            _query_sources({**defaults, "qEd": "../missing.py"})

    def test_pairq_source_override_has_one_reference_and_complete_fields(self):
        from s10_parity import query_error
        pairq = {
            "qoff": "estimator_k3_v41_pairq_a4r256_gram_p4d5.py",
            "qE": "estimator_k3_v42_pairq_a4r256_qEc_p4d5.py",
            "qEd": "estimator_k3_v42_pairq_a4r256_qEdc_p4d5.py",
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = query_error(16, 5, 90_000_016_001,
                                 str(pathlib.Path(tmp) / "pairq.json"), age=2, rank=4,
                                 sources=pairq)
        self.assertEqual([result["arms"][k]["path"] for k in pairq], list(pairq.values()))
        self.assertGreater(result["uncompressed_flops"], 0)
        for arm in result["arms"].values():
            self.assertGreater(arm["flops"], 0)
            self.assertIsNotNone(arm["retained_query_energy"])
            self.assertTrue(arm["query_energy_joins"])
            for layer in arm["per_layer"][1:-1]:
                self.assertTrue(all(k in layer for k in ("E", "D3", "D21", "mu_out", "cov_out")))


if __name__ == "__main__":
    unittest.main()
