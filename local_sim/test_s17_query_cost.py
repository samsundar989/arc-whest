"""Label-free source and tiny algebra checks for the old-D21 approximation."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

import numpy as np

import s17_query_cost as mod


class Tower:
    pass


def contract(source, tower, LA, LP, *, rank):
    tree = ast.parse(source)
    fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef)
              and x.name == "_shared_contract_lean")
    namespace = {"fnp": np, "_mm": np.matmul, "_SHARED_ON": False,
                 "_shared_rank": lambda n: 0, "_SHARED_AGE": 4,
                 "_OLD_QUERY_RANK": rank}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<query>", "exec"), namespace)
    return namespace[fn.name](tower, LA, LP, tower.A_L, tower.M)


class TestQueryCost(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = (mod.ROOT / "estimator_k3_v41_pairq_a4r256_gram_p4d5.py").read_text()
        cls.patched = mod.build(base)
        compile(cls.patched, "<generated>", "exec")

    def case(self, seed=10, n=7, r=3, ko=2):
        rng = np.random.default_rng(seed)
        tw = Tower()
        tw.ko, tw.k = ko, ko + 1
        tw.Q = rng.normal(size=(n, r))
        tw.FA = rng.normal(size=(r, ko*n))
        tw.FM = rng.normal(size=(r, ko*n))
        tw.A_L = rng.normal(size=(n, n))
        tw.M = rng.normal(size=(n, n))
        tw._qold = object()
        tw._wts = None
        return tw, rng.normal(size=(n, (ko+1)*n)), rng.normal(size=(n, (ko+1)*n))

    @staticmethod
    def exact(tw, LA, LP):
        c = tw.ko*tw.Q.shape[0]
        return (np.concatenate([LA[:, :c], LP[:, :c]], axis=1)
                @ np.concatenate([tw.FA, tw.FM], axis=1).T @ tw.Q.T
                + np.concatenate([LA[:, c:], LP[:, c:]], axis=1)
                @ np.concatenate([tw.A_L, tw.M], axis=1).T)

    def test_source_identity_and_stash_removal(self):
        self.assertEqual(self.patched.count("_OLD_QUERY_RANK = 32"), 1)
        self.assertNotIn("Xold =", self.patched)
        with self.assertRaises(ValueError):
            mod.build(self.patched)

    def test_q_at_least_r_is_exact(self):
        tw, LA, LP = self.case()
        got = contract(self.patched, tw, LA, LP, rank=3)
        np.testing.assert_allclose(got, self.exact(tw, LA, LP), rtol=1e-13, atol=1e-12)
        self.assertIsNone(tw._qold)

    def test_rank_two_coverage_and_permutation(self):
        tw, LA, LP = self.case()
        # The represented old query has rank two and its selected sketch covers it.
        tw.FA[2] = 0
        tw.FM[2] = 0
        expected = self.exact(tw, LA, LP)
        got = contract(self.patched, tw, LA, LP, rank=2)
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-11)
        perm = np.array([4, 2, 6, 0, 1, 5, 3])
        col_old = np.concatenate([perm + i*7 for i in range(2)])
        col_all = np.concatenate([perm + i*7 for i in range(3)])
        tp, Lp, Pp = self.case()
        tp.Q, tp.FA, tp.FM = tw.Q[perm], tw.FA[:, col_old], tw.FM[:, col_old]
        tp.A_L, tp.M = tw.A_L[np.ix_(perm, perm)], tw.M[np.ix_(perm, perm)]
        Lp, Pp = LA[np.ix_(perm, col_all)], LP[np.ix_(perm, col_all)]
        np.testing.assert_allclose(contract(self.patched, tp, Lp, Pp, rank=2),
                                   got[np.ix_(perm, perm)], rtol=1e-12, atol=1e-11)

    def test_approximate_path_permutation(self):
        tw, LA, LP = self.case()
        got = contract(self.patched, tw, LA, LP, rank=2)
        self.assertGreater(np.linalg.norm(got - self.exact(tw, LA, LP)), 1e-5)
        perm = np.array([4, 2, 6, 0, 1, 5, 3])
        old = np.concatenate([perm + i*7 for i in range(2)])
        all_cols = np.concatenate([perm + i*7 for i in range(3)])
        tp, _, _ = self.case()
        tp.Q, tp.FA, tp.FM = tw.Q[perm], tw.FA[:, old], tw.FM[:, old]
        tp.A_L, tp.M = tw.A_L[np.ix_(perm, perm)], tw.M[np.ix_(perm, perm)]
        np.testing.assert_allclose(
            contract(self.patched, tp, LA[np.ix_(perm, all_cols)],
                     LP[np.ix_(perm, all_cols)], rank=2),
            got[np.ix_(perm, perm)], rtol=1e-12, atol=1e-11)

    def test_zero_and_cancelled_sketch(self):
        tw, LA, LP = self.case()
        tw.Q[:] = 0
        np.testing.assert_allclose(contract(self.patched, tw, LA, LP, rank=2),
                                   self.exact(tw, LA, LP), atol=1e-12)

    def test_deficient_nonzero_sketch_falls_back(self):
        tw, LA, LP = self.case()
        tw.Q[:, 1] = tw.Q[:, 0]
        np.testing.assert_allclose(contract(self.patched, tw, LA, LP, rank=2),
                                   self.exact(tw, LA, LP), rtol=1e-12, atol=1e-11)
        tw, LA, LP = self.case()
        tw.FM = -tw.FA
        LP = LA.copy()  # signed old aggregate cancels, recent does too only if legs do
        np.testing.assert_allclose(contract(self.patched, tw, LA, LP, rank=2),
                                   self.exact(tw, LA, LP), atol=1e-12)

    def test_no_old_history(self):
        tw, LA, LP = self.case(ko=0)
        tw.Q = None
        tw.FA = tw.FM = None
        np.testing.assert_allclose(contract(self.patched, tw, LA, LP, rank=2),
                                   LA @ tw.A_L.T + LP @ tw.M.T, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
