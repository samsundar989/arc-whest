"""Label-free tests for the registered own PAIR+Q fit controls."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s14_own_fit_controls as S


class OwnFitControls(unittest.TestCase):
    def test_bounds_and_asserted_source(self):
        src = S.BASE.read_text(encoding="utf-8")
        for vals in ((-0.01, 1, 0, 1), (1, 1.51, 0, 1),
                     (1, 1, -.51, 1), (1, 1, 0, 1.51)):
            with self.assertRaises(ValueError):
                S.build(src, *vals)
        built = S.build(src)
        self.assertIn("_own_lambda(_li, depth)", built)
        self.assertIn("_own_lambda(_li_h, depth)", built)
        self.assertIn("cA = cA * _OWN_S_FEED", built)
        self.assertIn("pK[out] = pK[out] + _own_extra(X) * off_mask", built)
        with self.assertRaises(ValueError):
            S.build(built)

    def test_lambda_formula_and_family_scale(self):
        import k4feed_oracle as K
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"own_fit.py"
            S.emit(path, theta_fam=.7, s_lam=1.2, t_lam=.4, s_feed=.8)
            mod = K._load(str(path))
            for i in (0, 4, 10, 14):
                expected = mod._K4_LAM[i]*1.2*(1+.4*(i/15-.5))
                self.assertAlmostEqual(mod._own_lambda(i, 16), expected)
            self.assertEqual(mod._own_lambda(16, 16), 0.)
            self.assertAlmostEqual(mod._own_extra(2.), 1.4)

    def test_default_complete_recurrence_matches_own_base(self):
        import flopscope as flops
        import harness
        import k4feed_oracle as K
        n, depth = 16, 5
        rng = np.random.default_rng(91614)
        weights = [rng.normal(size=(n,n))*np.sqrt(2/n) for _ in range(depth)]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"default.py"
            S.emit(path, theta_fam=1, s_lam=1, t_lam=0, s_feed=1)
            mlp = harness.MLP(weights=weights, width=n, depth=depth, seed=91614)
            def run(source):
                with flops.BudgetContext(flop_budget=2**55, quiet=True):
                    return np.asarray(K._load(str(source)).Estimator().predict(mlp, 2**55))
            base = run(S.BASE)
            got = run(path)
            np.testing.assert_array_equal(got, base)

    def test_changed_controls_execute_complete_recurrence(self):
        import flopscope as flops
        import harness
        import k4feed_oracle as K
        n, depth = 16, 5
        rng = np.random.default_rng(91615)
        weights = [rng.normal(size=(n,n))*np.sqrt(2/n) for _ in range(depth)]
        with tempfile.TemporaryDirectory() as temp:
            paths = (Path(temp)/"changed.py", Path(temp)/"zero_tilt.py")
            S.emit(paths[0], theta_fam=.7, s_lam=1.2, t_lam=.4, s_feed=.8)
            S.emit(paths[1], theta_fam=.7, s_lam=1.2, t_lam=0., s_feed=.8)
            mods = [K._load(str(p)) for p in paths]
            mlp = harness.MLP(weights=weights, width=n, depth=depth, seed=91615)
            def run(mod):
                with flops.BudgetContext(flop_budget=2**55, quiet=True):
                    return np.asarray(mod.Estimator().predict(mlp, 2**55))
            got, zero_tilt = [run(mod) for mod in mods]
            self.assertEqual(got.shape, (depth, n))
            self.assertTrue(np.isfinite(got).all())
            self.assertGreater(np.max(np.abs(got-zero_tilt)), 0.)


if __name__ == "__main__":
    unittest.main()
