"""Synthetic, label-free regressions for the fixed corrected-storage reader."""
import copy
import dataclasses
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "local_sim"))
import s16_corrected_storage_decide as m  # noqa: E402


def fixtures():
    score = {"artifact": "p2_design_score", "block": "P12", "tier": "DESIGN",
             "n": 24, "width": 1024, "depth": 16,
             "n_samples_per_replica": 2**24,
             "truth_sha256": m.P12_TRUTH_SHA256,
             "accuracy_only": True, "local_seconds_caps": "off",
             "arms": {}}
    for name, path in m.PATHS.items():
        if name == "qoff":
            continue
        mse, mult = (.011, .2) if name.startswith("old") else (.020, .1)
        score["arms"][name] = {"path": path, "candidate_sha256": m._sha(ROOT / path),
                               "rows": [{"index": i, "seed": seed, "mse": mse,
                                         "flops_used": mult * m.s12.B,
                                         "gt_noise": .01, "failed": False,
                                         "nonfinite": False, "error_code": ""}
                                        for i, seed in enumerate(m.ORDER)]}

    def query(corrected):
        names = {"qoff": "qoff", "qE": "qEc" if corrected else "old_qE",
                 "qEd": "qEdc" if corrected else "old_qEd"}
        return {"artifact": "s10_parity.query_error", "label_free": True,
                "network": {"width": 1024, "depth": 16, "seed": 5},
                "overrides": {"age": 4, "rank": 256},
                "uncompressed_flops": 123.0,
                "arms": {arm: {
                    "path": m.PATHS[label], "sha256": m._sha(ROOT / m.PATHS[label]),
                    "flops": 1e11, "c_over_b": .1,
                    "final_row_rel_vs_uncompressed": 2.0 if not corrected else 1.0,
                    "all_rows_rel_vs_uncompressed": 2.0 if not corrected else 1.0,
                    "retained_query_energy": .5,
                    "per_layer": [{"layer": layer, **{k: (2.0 if not corrected else 1.0)
                               for k in (("mu_out", "cov_out") if layer == 0 else
                                         ("D3", "mu_out") if layer == 15 else
                                         ("E", "D3", "D21", "mu_out", "cov_out"))}}
                                  for layer in range(16)]}
                         for arm, label in names.items()}}
    old, new = query(False), query(True)
    new["arms"]["qoff"] = copy.deepcopy(old["arms"]["qoff"])
    return score, old, new


def verified_sidecar(score):
    man = {key: score[key] for key in ("block", "n", "width", "depth",
                                       "truth_sha256", "n_samples_per_replica")}
    man.update(run_complete=True, ordered_network_seeds=m.ORDER,
               config={"accuracy_only": True, "local_seconds_caps": "off"},
               arms={})
    for name, arm in score["arms"].items():
        full = []
        for row in arm["rows"]:
            full.append(dataclasses.asdict(m.le.EvidenceRow(
                index=row["index"], input_seed=row["seed"], estimator_seed=1,
                candidate=arm["path"], candidate_sha256=arm["candidate_sha256"],
                truth_sha256=score["truth_sha256"], width=1024, depth=16,
                budget=m.s12.B, flops_used=row["flops_used"], n_ops=100,
                mse=row["mse"], zero_prediction_mse=1,
                gt_noise=row["gt_noise"], residual_wall_time_s=0,
                wall_time_s=0, budget_exhausted=bool(row["failed"]),
                error_code=row["error_code"],
                nonfinite=row["nonfinite"])))
        digest = m.le.rows_hash([m.le.EvidenceRow(**r) for r in full])
        arm["rows_hash"] = digest
        man["arms"][name] = {"path": arm["path"],
                             "candidate_sha256": arm["candidate_sha256"],
                             "rows_hash": digest, "evidence_rows": full}
    return {"manifest": man, "arms": {}}


def run(score, old, new, *, verified=None):
    if verified is None:
        verified = verified_sidecar(score)
    with tempfile.TemporaryDirectory() as tmp:
        files = [pathlib.Path(tmp) / name for name in ("score.json", "old.json", "new.json")]
        for path, data in zip(files, (score, old, new)):
            path.write_text(json.dumps(data), encoding="utf-8")
        with mock.patch.object(m.ds, "verify", return_value=verified):
            return m.decide(*(str(path) for path in files), str(pathlib.Path(tmp) / "sidecar"))


class CorrectedStorageDecision(unittest.TestCase):
    def test_primary_pass_uses_registered_decide_and_query(self):
        score, old, new = fixtures()
        for row in score["arms"]["qEc"]["rows"]:
            row["mse"], row["flops_used"] = .0105, .1 * m.s12.B
        out = run(score, old, new)
        self.assertTrue(out["comparisons"]["primary"]["score_decide"]["passed"])
        self.assertTrue(out["comparisons"]["primary"]["query_condition"])
        self.assertTrue(out["PRIMARY_SCREEN_PASSED"])

    def test_observed_false_pass_fails_official_primary(self):
        score, old, new = fixtures()
        out = run(score, old, new)
        primary = out["comparisons"]["primary"]
        self.assertGreater(primary["observed_gain"], 1)
        self.assertLess(primary["official_precision_gain"], 1)
        self.assertFalse(out["PRIMARY_SCREEN_PASSED"])

    def test_identity_and_source_mismatch_refuse(self):
        for mutation in ("seed", "duplicate", "missing", "source"):
            with self.subTest(mutation=mutation):
                score, old, new = fixtures()
                rows = score["arms"]["qEc"]["rows"]
                if mutation == "seed":
                    rows[0]["seed"] += 1
                elif mutation == "duplicate":
                    rows[0] = dict(rows[1])
                elif mutation == "missing":
                    rows.pop()
                else:
                    new["arms"]["qE"]["sha256"] = "0" * 64
                with self.assertRaises(ValueError):
                    run(score, old, new)

    def test_wrong_truth_or_caps_refuse(self):
        for mutation in ("truth", "caps", "noise"):
            with self.subTest(mutation=mutation):
                score, old, new = fixtures()
                if mutation == "truth":
                    score["truth_sha256"] = "0" * 64
                elif mutation == "caps":
                    score["local_seconds_caps"] = "official"
                else:
                    score["arms"]["qEc"]["rows"][0]["gt_noise"] *= 2
                with self.assertRaises(ValueError):
                    run(score, old, new)

    def test_production_requires_sidecar(self):
        score, old, new = fixtures()
        with tempfile.TemporaryDirectory() as tmp:
            files = [pathlib.Path(tmp) / name for name in ("score.json", "old.json", "new.json")]
            for path, data in zip(files, (score, old, new)):
                path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "verified prediction sidecar"):
                m.decide(*(str(path) for path in files), "")

    def test_score_summary_tamper_refused_against_verified_sidecar(self):
        score, old, new = fixtures()
        verified = verified_sidecar(score)
        score["arms"]["qEc"]["rows"][0]["mse"] += .001
        with self.assertRaisesRegex(ValueError, "score row differs"):
            run(score, old, new, verified=verified)

    def test_sidecar_verification_requires_sources_and_full_rows(self):
        score, _, _ = fixtures()
        verified = verified_sidecar(score)
        with mock.patch.object(m.ds, "verify", return_value=verified) as verifier:
            m._bind_sidecar(score, "synthetic_sidecar_dir")
        verifier.assert_called_once_with(
            "synthetic_sidecar_dir", check_sources=True,
            require_present_sources=True, require_full_row_binding=True,
            root=m.ROOT)

    def test_missing_E_refuses(self):
        score, old, new = fixtures()
        del new["arms"]["qE"]["per_layer"][1]["E"]
        with self.assertRaisesRegex(ValueError, "missing complete query layers"):
            run(score, old, new)

    def test_secondary_cannot_rescue_failed_primary(self):
        score, old, new = fixtures()
        # Make qEdc strongly better and qEc worse at official precision.
        for row in score["arms"]["qEdc"]["rows"]:
            row["mse"] = .0105
        out = run(score, old, new)
        self.assertGreater(out["comparisons"]["secondary_diagnostic"]["official_precision_gain"], 1)
        self.assertIsNone(out["comparisons"]["secondary_diagnostic"]["score_decide"])
        self.assertFalse(out["PRIMARY_SCREEN_PASSED"])

    def test_new_failed_identity_fails_primary(self):
        score, old, new = fixtures()
        for row in score["arms"]["qEc"]["rows"]:
            row["mse"], row["flops_used"] = .0105, .1 * m.s12.B
        score["arms"]["qEc"]["rows"][0]["failed"] = True
        out = run(score, old, new)
        self.assertEqual(out["comparisons"]["primary"]["new_failed_identities"],
                         [f"seed:{m.ORDER[0]}"])
        self.assertFalse(out["PRIMARY_SCREEN_PASSED"])


if __name__ == "__main__":
    unittest.main()
