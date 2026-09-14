"""Tests for E14 precision diagnostics and their refusal behavior."""
import ast
import inspect
import unittest
import contextlib
import io
import json
import tempfile
import warnings
from pathlib import Path
from unittest import mock
import numpy as np
import p2_compact_fullrank_precision_e14 as f64
import flopscope as flops
import flopscope.numpy as fnp
import p2_compact_reference as reference
import p2_compact_f32_backend_e14 as backend
import p2_compact_fullrank_f32_e14 as f32


class BackendTransformation(ast.NodeTransformer):
    def visit_AugAssign(self, node):
        self.generic_visit(node)
        return ast.copy_location(ast.Assign(targets=[node.target], value=ast.BinOp(
            left=ast.Name(node.target.id, ast.Load()), op=node.op, right=node.value)), node)

    def visit_Call(self, node):
        self.generic_visit(node)
        if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'np' and node.func.attr == 'zeros'):
            node.keywords.append(ast.keyword(arg='dtype', value=ast.Attribute(
                value=ast.Name('np', ast.Load()), attr='float32', ctx=ast.Load())))
        return node


class TestPrecisionDiagnostic(unittest.TestCase):
    def test_nan_mutation_publishes_first_failure(self):
        original = f64.independent_wick
        def inject(*args, **kwargs):
            result = original(*args, **kwargs)
            result[(4, 1)] = result[(4, 1)].copy()
            result[(4, 1)][0] = float('nan')
            return result
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'failure.json'
            with mock.patch.object(f64, 'independent_wick', inject), \
                 mock.patch('sys.argv', ['e14', '--out', str(path)]), \
                 contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter('ignore')
                with self.assertRaises(SystemExit) as exit_info:
                    f64.main()
            self.assertEqual(exit_info.exception.code, 1)
            result = json.loads(path.read_text())
            self.assertEqual(result['failed_count'], 1)
            self.assertEqual(result['first_failure']['field'], 'wick.(4, 1)')
            self.assertEqual(result['first_failure']['got'][0], {'nonfinite': 'nan'})
            self.assertEqual(result['checkpoints'][0]['compact_pre']['w']['(4, 1)']['values'][0],
                             {'nonfinite': 'nan'})

    def test_f64_stops_at_first_mismatch_and_preserves_operands(self):
        checks = []
        with self.assertRaises(f64.FirstMismatch):
            f64.record(checks, 3, 'wick.(4, 1)', np.array([2.0]), np.array([1.0]))
        self.assertEqual(len(checks), 1)
        self.assertEqual(checks[0]['got'], [2.0])
        self.assertEqual(checks[0]['want'], [1.0])

    def test_original_threshold_remains_failure(self):
        checks = []
        f32.record(checks, 3, 'wick.(4, 1)', [1.001], [1.0])
        self.assertFalse(checks[0]['passed'])

    def test_declared_output_identity_and_drift_mutation(self):
        self.assertTrue(f32.output_check([1.0], [1.0])['passed'])
        self.assertFalse(f32.output_check([1.0 + np.sqrt(2 * f32.OUTPUT_ALLOWANCE)], [1.0])['passed'])

    def test_zero_scale_and_field_mutation(self):
        self.assertTrue(f32.field_envelope([0.0], [0.0], [0.0])['passed'])
        self.assertFalse(f32.field_envelope([1e-20], [0.0], [0.0])['passed'])
        self.assertFalse(f32.field_envelope([1.01], [1.0], [1.0])['passed'])

    def test_backend_only_declared_ast_transform(self):
        for name in ('materialize', 'reconstruct', 'compact_slices',
                     'weighted_gram_join', 'transport_shared'):
            original = ast.parse(inspect.getsource(getattr(reference, name)))
            adapted = BackendTransformation().visit(original)
            actual = ast.parse(inspect.getsource(getattr(backend, name)))
            self.assertEqual(ast.dump(adapted), ast.dump(actual), name)

    def test_dtype_inventory_refuses_hidden_f64(self):
        with flops.BudgetContext(flop_budget=2**20, quiet=True):
            value = f32.snapshot({'good': fnp.ones(2, dtype=fnp.float32),
                                  'bad': fnp.ones(2, dtype=fnp.float64)})
        rows = f32.dtype_inventory(value)
        self.assertEqual([r['passed'] for r in rows], [True, False])


if __name__ == '__main__':
    unittest.main()
