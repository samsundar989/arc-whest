"""Tiny, label-free identities for the corrected shared-history objective."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s13_correct_query_objective as S


class CorrectQueryObjective(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1291)
        self.n, self.k, self.ko = 5, 3, 1
        n, k = self.n, self.k
        self.UA = rng.normal(size=(n, n))
        self.UP = rng.normal(size=(n, n))
        self.A = rng.normal(size=(n, k * n))
        self.M = rng.normal(size=(n, k * n))
        self.w2 = np.array([.12, .31, .42, .55, .78,
                            1.1, .22, .33, .88, .91,
                            .02, .74, .41, .66, .95])
        self.w1 = np.array([.19, .37, .55, .79, .96])
        q, _ = np.linalg.qr(rng.normal(size=(n, 2)))
        self.Q = q
        self.Xold = rng.normal(size=(n, 2))

    def test_post_wick_query_direct_and_projected(self):
        n, w = self.n, self.w1
        U = np.concatenate([self.UA, self.UP], axis=1)
        J = np.concatenate([self.A[:, n:2*n], self.M[:, n:2*n]], axis=1)
        # Direct post-Wick consumer: two left slots, one right slot.
        direct = w[:, None]**2 * (self.Xold @ self.Q.T + U @ J.T) * w[None, :]
        Qw = w[:, None] * self.Q
        Jw = w[:, None] * J
        represented = ((w[:, None]**2 * self.Xold) @ Qw.T
                       + (w[:, None]**2 * U) @ Jw.T)
        np.testing.assert_allclose(represented, direct, rtol=1e-14, atol=1e-14)
        old = self.Xold @ Qw.T + U @ Jw.T
        self.assertGreater(np.linalg.norm(old-direct), .1)

        # Rank below width: query action E^T(E Omega) is exactly the direct
        # consumer action, and projected loss equals the direct projection.
        Om = np.eye(n)[:, [0, 2]]
        z = (w[:, None]**2 * U) @ (Jw.T @ Om)
        z += (w[:, None]**2 * self.Xold) @ (Qw.T @ Om)
        y = Jw @ ((w[:, None]**2 * U).T @ z)
        y += Qw @ ((w[:, None]**2 * self.Xold).T @ z)
        np.testing.assert_allclose(y, direct.T @ (direct @ Om), rtol=1e-13, atol=1e-13)
        q, _ = np.linalg.qr(y)
        np.testing.assert_allclose(direct @ (np.eye(n)-q@q.T),
                                   represented @ (np.eye(n)-q@q.T), atol=1e-13)

        # Applying a further transport and next Wick gate keeps the identity.
        T = np.array([[.4, .1, 0, 0, .2], [0, .7, .2, 0, 0],
                      [.1, 0, .8, .1, 0], [0, .2, 0, .6, .1],
                      [.3, 0, 0, .1, .5]])
        next_w = np.array([.3, .8, .4, .9, .6])
        full_next = next_w[:, None]**2 * (T @ direct @ T.T) * next_w[None, :]
        rep_next = next_w[:, None]**2 * (T @ represented @ T.T) * next_w[None, :]
        np.testing.assert_allclose(rep_next, full_next, rtol=1e-13, atol=1e-13)

    def test_d3_summand_uses_square_and_joining_lineage(self):
        n, c = self.n, self.ko*self.n
        A, M = self.A[:, c:c+n], self.M[:, c:c+n]
        direct = A * M * (A * self.w2[c:c+n])
        corrected = A*A*M*self.w2[c:c+n]
        np.testing.assert_allclose(corrected, direct, rtol=1e-14, atol=1e-14)
        old = A*M*self.w2[:n]
        self.assertGreater(np.linalg.norm(corrected-old), .1)
        q, _ = np.linalg.qr(corrected[:, :2])
        self.assertEqual(q.shape, (n, 2))
        self.assertGreater(np.linalg.norm((np.eye(n)-q@q.T)@corrected), 0)

    def test_nonuniform_gate_changes_rank_one_choice(self):
        # Distinct singular directions make the missing left factor decisive.
        Epre = np.diag([50., 1., .1])
        w = np.array([.1, 1., .5])
        wrong = Epre * w[None, :]
        direct = w[:, None]**2 * Epre * w[None, :]
        old_v = np.linalg.svd(wrong)[2][0]
        new_v = np.linalg.svd(direct)[2][0]
        self.assertLess(abs(old_v @ new_v), 1e-12)
        old_loss = np.linalg.norm(direct @ (np.eye(3)-np.outer(old_v, old_v)))
        new_loss = np.linalg.norm(direct @ (np.eye(3)-np.outer(new_v, new_v)))
        self.assertGreater(old_loss, 5 * new_loss)

    def test_emitted_source_has_both_corrections(self):
        base = S.ROOT / "estimator_k3_v39_v37_a4r256_p4d5_qEd.py"
        src = S.build(base.read_text(encoding="utf-8"), d3=True)
        self.assertIn("out._qold = tw._qold * left", src)
        self.assertIn("tw.w2b[c:c + n]", src)
        self.assertNotIn(S.D3_OLD, src)
        with self.assertRaises(ValueError):
            S.build(src, d3=True)

    def test_actual_recurrent_chain_full_rank_and_thin(self):
        # Run the generated estimator's complete layer recurrence, including
        # birth, transport, Wick, joins, nonlinear update and next-layer mean.
        import k4feed_oracle as K
        import s10_parity as P
        n, depth = 16, 6
        rng = np.random.default_rng(66102)
        weights = [rng.normal(size=(n, n))*np.sqrt(2/n) for _ in range(depth)]
        base = S.ROOT / "estimator_k3_v41_pairq_a4r256_qE_p4d5.py"
        fixed = S.ROOT / "estimator_k3_v42_pairq_a4r256_qEc_p4d5.py"
        with tempfile.TemporaryDirectory() as temp:
            src = P._variant(str(base), temp, "uncompressed", _SHARED_ON=False)
            ref = P.layer_loop(K._load(src), weights, n, depth)
            for tag, path in (("old", base), ("corrected", fixed)):
                full = P._variant(str(path), temp, tag+"_full", _SHARED_RANK=n,
                                  _SHARED_FORCE=True, _SHARED_AGE=2)
                mod = K._load(full)
                joins = P.count_joins(mod)
                got = P.layer_loop(mod, weights, n, depth)
                self.assertGreater(joins["joins"], 0)
                np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-9)
            thin = P._variant(str(fixed), temp, "corrected_thin", _SHARED_RANK=8,
                              _SHARED_FORCE=True, _SHARED_AGE=2)
            mod = K._load(thin)
            joins = P.count_joins(mod)
            got = P.layer_loop(mod, weights, n, depth)
            self.assertGreater(joins["joins"], 0)
            self.assertTrue(np.isfinite(got).all())


if __name__ == "__main__":
    unittest.main()
