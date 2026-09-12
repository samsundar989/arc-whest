"""Independent dense enumeration for the first-source research cost kernel."""

import unittest

import numpy as np

from p2_k4_first_source_kernel import metered_program


def dense_fourth(v, p):
    """Enumerate the centered raw fourth moment and three Wick pairings."""
    n, s = v.shape
    cov = np.zeros((n, n))
    fourth = np.zeros((n, n, n, n))
    for h in range(s):
        for i in range(n):
            for j in range(n):
                cov[i, j] += p[h] * v[i, h] * v[j, h]
                for k in range(n):
                    for l in range(n):
                        fourth[i, j, k, l] += p[h] * v[i, h] * v[j, h] * v[k, h] * v[l, h]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    fourth[i, j, k, l] -= (cov[i, j] * cov[k, l]
                                              + cov[i, k] * cov[j, l]
                                              + cov[i, l] * cov[j, k])
    return fourth


def dense_feed(k, w1, w2):
    n = len(w1)
    feed = np.zeros((n, n, n))
    for i in range(n):
        for j in range(n):
            for h in range(n):
                feed[i, j, h] = 0.5 * (
                    w2[i] * w1[j] * w1[h] * k[i, i, j, h]
                    + w1[i] * w2[j] * w1[h] * k[i, j, j, h]
                    + w1[i] * w1[j] * w2[h] * k[i, j, h, h])
    return feed


def transported_fourth(k, b):
    return np.einsum("ia,jb,kc,ld,abcd->ijkl", b, b, b, b, k, optimize=True)


def transported_third(t, b):
    return np.einsum("ia,jb,kc,abc->ijk", b, b, b, t, optimize=True)


class FirstSourceKernelTest(unittest.TestCase):
    def test_full_tensors_survive_two_signed_future_maps(self):
        got = metered_program(n=4, depth=4, rank=5, capture=True)
        raw, p, source_map, maps, w1, w2 = got["_inputs"]
        self.assertTrue(all(np.min(b) < 0 < np.max(b) for b in maps))
        self.assertGreater(np.std(w1[0]), 0.01)
        self.assertGreater(np.std(w2[0]), 0.005)
        v = source_map @ (raw - (raw * p).sum(axis=1)[:, None])
        k = dense_fourth(v, p)
        v = maps[0] @ v
        k = transported_fourth(k, maps[0])
        np.testing.assert_allclose(k, dense_fourth(v, p), rtol=1e-5, atol=1e-6)
        direct = dense_feed(k, w1[0], w2[0])
        cov = (v * p[None, :]) @ v.T
        a = w2[0] * np.diag(cov)
        A = np.diag(w2[0])
        P = w1[0][:, None] * v
        birth_v = v.copy()

        def factor_tensor(A, P, a):
            # Enumerate all factor placements, including the covariance
            # subtraction.  This is separate from the kernel's slice code.
            q = (P * p[None, :]) @ birth_v.T
            h = (P * p[None, :]) @ P.T
            plus = A @ (birth_v * birth_v)
            full = np.zeros((4, 4, 4))
            for i in range(4):
                for j in range(4):
                    for t in range(4):
                        value = -0.5 * (a[i] * h[j, t] + a[j] * h[i, t]
                                        + a[t] * h[i, j])
                        for column in range(4):
                            value -= (A[i, column] * q[j, column] * q[t, column]
                                      + q[i, column] * A[j, column] * q[t, column]
                                      + q[i, column] * q[j, column] * A[t, column])
                        for particle in range(5):
                            value += 0.5 * p[particle] * (
                                plus[i, particle] * P[j, particle] * P[t, particle]
                                + P[i, particle] * plus[j, particle] * P[t, particle]
                                + P[i, particle] * P[j, particle] * plus[t, particle])
                        full[i, j, t] = value
            return full

        self.assertGreater(np.max(np.abs(direct)), 1e-4)
        np.testing.assert_allclose(factor_tensor(A, P, a), direct,
                                   rtol=1e-5, atol=1e-6)
        for step in (1, 2):
            b = maps[step]
            direct = transported_third(direct, b)
            A, P, a = b @ A, b @ P, b @ a
            np.testing.assert_allclose(factor_tensor(A, P, a), direct,
                                       rtol=1e-5, atol=1e-6)
            v = b @ v
            k = transported_fourth(k, b)
            np.testing.assert_allclose(k, dense_fourth(v, p),
                                       rtol=1e-5, atol=1e-6)
        d21 = np.einsum("iij->ij", direct)
        self.assertGreater(np.max(np.abs(d21 - d21.T)), 1e-5)

    def test_dense_independent_two_step_signed_transport(self):
        got = metered_program(n=4, depth=4, rank=5, capture=True)
        raw, p, source_map, maps, w1, w2 = got["_inputs"]
        v = raw - (raw * p).sum(axis=1)[:, None]
        v = source_map @ v
        k = dense_fourth(v, p)
        old_feeds = []
        checked = 0
        for row in got["_trace"]:
            layer = row["layer"]
            b = maps[layer - 1]
            k = transported_fourth(k, b)
            old_feeds = [transported_third(feed, b) for feed in old_feeds]
            d4 = np.einsum("iiii->i", k)
            np.testing.assert_allclose(row["D4"], d4, rtol=1e-5, atol=1e-6)
            checked += 1
            if layer < 3:
                d22 = np.einsum("iijj->ij", k)
                d31 = np.einsum("iiij->ij", k)
                np.testing.assert_allclose(row["D22"], d22, rtol=1e-5, atol=1e-6)
                np.testing.assert_allclose(row["D31"], d31, rtol=1e-5, atol=1e-6)
                checked += 2
            f = np.sum(old_feeds, axis=0) if old_feeds else np.zeros((4, 4, 4))
            np.testing.assert_allclose(row["D3"], np.einsum("iii->i", f),
                                       rtol=1e-5, atol=1e-6)
            checked += 1
            if layer < 3:
                np.testing.assert_allclose(row["D21"], np.einsum("iij->ij", f),
                                           rtol=1e-5, atol=1e-6)
                checked += 1
                old_feeds.append(dense_feed(k, w1[layer - 1], w2[layer - 1]))
        self.assertEqual(checked, 12)
        self.assertEqual(sum(got["families"].values()), got["flops"])
        self.assertEqual(got["schedule"], got["expected_schedule"])
        self.assertGreater(got["negative_D21_flops"], 0)


if __name__ == "__main__":
    unittest.main()
