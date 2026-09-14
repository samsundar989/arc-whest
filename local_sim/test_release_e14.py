"""Reusable-input identity, evidence-domain and candidate sensitivity mutations."""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or '.').resolve() != HERE] + [str(HERE)]
import numpy as np
import candidate_release_record as release
import certify_candidate as cert
import edge_cases as edge


class ReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recipe = json.loads((release.ROOT / 'experiments/2026-09-14_e14/f21_release_recipe.json').read_text())

    def test_stored_domains_remain_separate(self):
        result = release.build(self.recipe)
        self.assertEqual(result['domains']['LOCAL']['verdict'], 'REFUSED')
        self.assertEqual(result['domains']['SERVER_MEASURED']['verdict'], 'PASS')
        self.assertEqual(result['domains']['SERVER_MEASURED']['exact_byte_binding'], 'PROVISIONAL')
        self.assertFalse(result['normal_release_ready'])
        self.assertEqual(result['local_mini']['failed_rows'], 100)
        self.assertEqual(result['counted_ledger']['arms']['f64_readonly']['flops_used'], 928457637478)

    def test_wrong_candidate_hash_rejected(self):
        recipe = copy.deepcopy(self.recipe)
        recipe['candidate']['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            release.build(recipe)

    def test_different_valid_candidate_rejected(self):
        recipe = copy.deepcopy(self.recipe)
        recipe['candidate'] = recipe['baseline']
        with self.assertRaisesRegex(ValueError, 'certificate candidate mismatch'):
            release.build(recipe)

    def test_changed_evidence_rejected(self):
        recipe = copy.deepcopy(self.recipe)
        recipe['evidence']['certificate']['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            release.build(recipe)

    def test_unknown_server_is_not_pass(self):
        recipe = copy.deepcopy(self.recipe)
        del recipe['evidence']['server_report']
        self.assertEqual(release.build(recipe)['domains']['SERVER_MEASURED']['verdict'], 'NOT_RUN')

    def test_server_semantic_mutations_refused(self):
        original = json.loads((release.ROOT / self.recipe['evidence']['server_report']['path']).read_text())
        def duplicate(d):
            d['evaluation']['results']['per_mlp'][1] = d['evaluation']['results']['per_mlp'][0]
        def nan_later(d):
            d['evaluation']['results']['per_mlp'][1]['telemetry']['residual_wall_time_s'] = float('nan')
        def fail_flag(d):
            d['evaluation']['results']['per_mlp'][1]['telemetry']['time_exhausted'] = True
        def missing_flag(d):
            del d['evaluation']['results']['per_mlp'][1]['telemetry']['budget_exhausted']
        def incomplete(d):
            d['evaluation']['results']['per_mlp'].pop()
        for mutation in (duplicate, nan_later, fail_flag, missing_flag, incomplete):
            with self.subTest(mutation=mutation.__name__), tempfile.TemporaryDirectory(dir=release.ROOT / 'experiments') as tmp:
                fixture = copy.deepcopy(original); mutation(fixture)
                path = Path(tmp) / 'mutant.json'; path.write_text(json.dumps(fixture))
                recipe = copy.deepcopy(self.recipe)
                recipe['evidence']['server_report'] = {'path': str(path.relative_to(release.ROOT)), 'sha256': release.sha(path)}
                self.assertEqual(release.build(recipe)['domains']['SERVER_MEASURED']['verdict'], 'REFUSED')

    def test_wrong_server_submission_rejected(self):
        recipe = copy.deepcopy(self.recipe); recipe['server_submission_id'] = 1
        with self.assertRaisesRegex(ValueError, 'submission identity mismatch'):
            release.build(recipe)

    def test_numerical_supplement_is_candidate_and_regime_bound(self):
        recipe = json.loads((release.ROOT / 'experiments/2026-09-14_e14/f21_release_recipe_v3.json').read_text())
        self.assertEqual(release.build(recipe)['numerical_requirement']['status'], 'THREE_TARGET_PERMUTATION_CASES_PASS')
        original = json.loads((release.ROOT / recipe['evidence']['numerical']['path']).read_text())
        for mode in ('source', 'bar', 'case', 'dtype', 'shape', 'runner_hash', 'edge_hash', 'backend', 'integer_result', 'receipt_hash', 'receipt_case', 'meter'):
            with tempfile.TemporaryDirectory(dir=release.ROOT / 'experiments') as tmp:
                changed = copy.deepcopy(original)
                if mode == 'source': changed['candidate_sha256'] = '0' * 64
                elif mode == 'bar': changed['bar'] = 1.4e-8
                elif mode == 'case': changed['cases'].pop()
                elif mode == 'dtype': changed['cases'][0]['receipt']['world']['weights_mode'] = 'f32'
                elif mode == 'shape': changed['cases'][0]['receipt']['world']['width'] = 8
                elif mode == 'runner_hash': changed['instrument_sha256'] = '0' * 64
                elif mode == 'edge_hash': changed['cases'][0]['receipt']['edge_cases_sha256'] = '0' * 64
                elif mode == 'backend': changed['cases'][0]['receipt']['world']['real_flopscope'] = False
                elif mode == 'integer_result': changed['cases'][0]['receipt']['results'] = [1]
                elif mode == 'receipt_hash': changed['cases'][0]['receipt_sha256'] = '0' * 64
                elif mode == 'receipt_case': changed['cases'][0]['receipt']['case'] = 'permutation:1'
                elif mode == 'meter': del changed['cases'][0]['receipt']['metered_predictions']
                # Also bind a matching mutated receipt, exercising semantic checks
                # beyond the embedded-versus-file equality check.
                if mode in ('edge_hash', 'backend', 'integer_result', 'receipt_case', 'meter'):
                    receipt_path = Path(tmp) / 'receipt.json'
                    receipt_path.write_text(json.dumps(changed['cases'][0]['receipt']))
                    changed['cases'][0]['receipt_path'] = str(receipt_path)
                    changed['cases'][0]['receipt_sha256'] = release.sha(receipt_path)
                path = Path(tmp) / 'numerical.json'; path.write_text(json.dumps(changed))
                mutated = copy.deepcopy(recipe)
                mutated['evidence']['numerical'] = {'path': str(path.relative_to(release.ROOT)), 'sha256': release.sha(path)}
                self.assertEqual(release.build(mutated)['numerical_requirement']['status'], 'REQUIRES_CANDIDATE_SPECIFIC_EVIDENCE_AT_DECLARED_SCALE')

    def test_scale_is_explicit_tighten_only(self):
        for bad in (0, -1, float('nan'), float('inf'), 1e-3):
            recipe = copy.deepcopy(self.recipe); recipe['numerical_raw_mse_scale'] = bad
            with self.assertRaises(ValueError): release.build(recipe)

    def test_permutation_identity_and_excessive_drift(self):
        scale = 2e-8
        with patch.dict(os.environ, {'ARC_WHEST_RAW_MSE': repr(scale)}):
            self.assertEqual(edge.raw_mse_denominator()[0], scale)
            self.assertEqual(cert._effective_permutation_bar()[1:3], (scale, .01 * scale))
            for case in (0, 1, 3):
                details = cert._expected_permutation_details(case)
                self.assertEqual((details['depth'], details['width']), (16, 1024))
                self.assertEqual(details['denom'], scale)
            base = np.arange(16 * 8, dtype=np.float64).reshape(16, 8) / 128
            permutation = np.arange(7, -1, -1)
            for layer in (1, 8, 14):
                equivalent = base.copy(); equivalent[layer] = base[layer, permutation]
                self.assertTrue(edge.perm_gate(base, equivalent, permutation, layer, 16)['ok'])
                for row in (layer, 15):
                    mutant = equivalent.copy(); mutant[row] += np.sqrt(4e-10)
                    self.assertFalse(edge.perm_gate(base, mutant, permutation, layer, 16)['ok'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
