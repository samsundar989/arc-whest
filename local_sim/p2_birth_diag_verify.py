"""Truth-free tiny oracle for the proposed P2 newborn diagonal correction.

This isolates the algebra in estimator_k3_v43_complete_c10.py. It does not
evaluate an MLP, read a seed block, or claim an accuracy improvement.
Run: uv run python local_sim/p2_birth_diag_verify.py
"""

import numpy as np


def compress(resid, rank):
    """NumPy copy of the source's deterministic two-pass P2 factor orientation."""
    n = len(resid)
    r = n if rank <= 0 else min(rank, n)
    om = resid if r >= n else resid[:, np.argsort((resid * resid).sum(0))[-r:]]
    q, _ = np.linalg.qr(resid @ om)
    q, _ = np.linalg.qr(resid @ (resid.T @ q))
    return (q.T @ resid).T * 3, q


def sym_d_tensor(ad, al, modes, wick):
    """Independent index oracle for D and L blocks, with averaged symmetry."""
    n, cols = modes.shape
    out = np.zeros((n, n, n))
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for j in range(cols):
                    out[a, b, c] += (
                        ad[a, j] * modes[b, j] * modes[c, j]
                        + ad[b, j] * modes[a, j] * modes[c, j]
                        + ad[c, j] * modes[a, j] * modes[b, j]
                    ) / 3
                    # Source's L block carries an explicit factor 3. The
                    # average over six permutations therefore weighs each
                    # permutation by 1/2.
                    out[a, b, c] += (
                        + al[a, j] * modes[b, j] * al[c, j] * wick[j]
                        + al[b, j] * modes[a, j] * al[c, j] * wick[j]
                        + al[c, j] * modes[a, j] * al[b, j] * wick[j]
                        + al[a, j] * modes[c, j] * al[b, j] * wick[j]
                        + al[b, j] * modes[c, j] * al[a, j] * wick[j]
                        + al[c, j] * modes[b, j] * al[a, j] * wick[j]
                    ) / 2
    return out


def slices(t):
    n = len(t)
    return np.array([t[i, i, i] for i in range(n)]), np.array(
        [[t[i, i, j] for j in range(n)] for i in range(n)])


def check_case(name, resid, rank, s3, e, al, wick, transport, expect_leak):
    n = len(resid)
    z, leg = compress(resid, rank)
    approx = z @ leg.T
    leak = np.sum(z * leg, axis=1)
    np.testing.assert_allclose(leak, np.diag(approx), atol=2e-14)
    assert (np.max(np.abs(leak)) > 1e-8) == expect_leak, name
    old_ad = np.diag(s3 + 3 * al.diagonal() * e) + approx
    new_ad = np.diag(s3 - leak + 3 * al.diagonal() * e) + approx
    dense_ad = np.diag(s3 + 3 * al.diagonal() * e) + 3 * resid.T
    eye = np.eye(n)
    old = sym_d_tensor(old_ad, al, eye, wick)
    new = sym_d_tensor(new_ad, al, eye, wick)
    dense = sym_d_tensor(dense_ad, al, eye, wick)
    old3, old21 = slices(old)
    new3, new21 = slices(new)
    dense3, dense21 = slices(dense)
    np.testing.assert_allclose(old3 - new3, leak, atol=2e-14)
    np.testing.assert_allclose(new3, dense3, atol=2e-14)
    # A diagonal source correction has no (i,i,j) effect for i != j at birth.
    mask = ~np.eye(n, dtype=bool)
    np.testing.assert_allclose((old21 - new21)[mask], 0, atol=2e-14)
    np.testing.assert_allclose((old21.T - new21.T)[mask], 0, atol=2e-14)
    if rank <= 0 or rank >= n:
        np.testing.assert_allclose(new21, dense21, atol=2e-13)
    # The correction is carried as M diag(sb); after a dense change of basis it
    # can affect off-diagonal repeated slices in both orientations.
    old_t = sym_d_tensor(transport @ old_ad, transport @ al,
                         transport @ eye, wick)
    new_t = sym_d_tensor(transport @ new_ad, transport @ al,
                         transport @ eye, wick)
    t3_old, t21_old = slices(old_t)
    t3_new, t21_new = slices(new_t)
    delta = old_t - new_t
    expected = np.einsum('aj,bj,cj,j->abc', transport, transport,
                         transport, leak)
    np.testing.assert_allclose(delta, expected, atol=3e-13)
    np.testing.assert_allclose(t3_old - t3_new, np.einsum(
        'ij,j->i', transport ** 3, leak), atol=3e-13)
    np.testing.assert_allclose(t21_old - t21_new,
        np.einsum('ij,kj,j->ik', transport ** 2, transport, leak), atol=3e-13)
    np.testing.assert_allclose((t21_old - t21_new).T,
        np.einsum('ij,kj,j->ik', transport ** 2, transport, leak).T,
        atol=3e-13)
    print(f'{name}: PASS; max leak {max(abs(leak)):.6g}; '
          f'transported D21 delta {np.max(np.abs(t21_old-t21_new)):.6g}')


def main():
    n = 3
    s3 = np.array([0.4, -0.7, 1.1])
    e = np.array([-0.3, 0.8, 0.2])
    wick = np.array([0.7, -0.4, 1.2])
    # Nonzero A_L off diagonal allows an independent two-leg contribution; its
    # zero diagonal proves that channel does not cancel the P2 leak at birth.
    al = np.array([[0, .2, -.4], [.5, 0, .3], [-.1, .6, 0]])
    transport = np.array([[.8, -.3, .2], [.4, .9, -.5], [-.2, .6, 1.1]])
    y_a = np.array([.2, -.8, .5])
    y_b = np.array([-.4, .6, .3])
    c_a = np.array([.7, .1, -.2])
    c_b = np.array([-.3, .9, .4])
    hub_birth_d3 = 2 * al.diagonal() * (y_a * c_a + y_b * c_b)
    np.testing.assert_array_equal(hub_birth_d3, np.zeros(n))
    ones_off = np.ones((n, n)) - np.eye(n)
    check_case('rank1 symmetric counterexample', ones_off, 1,
               s3, e, al, wick, transport, True)
    # The analytic best-rank-one approximation has diagonal 2 after the
    # source's factor 3. The actual two-pass selected-column range finder is
    # close but not identical, because its tied initial columns select one.
    best_rank1 = 2 * np.ones((n, n))
    np.testing.assert_allclose(np.diag(best_rank1), np.full(n, 2.))
    z, leg = compress(ones_off, 1)
    assert np.min(np.sum(z * leg, axis=1)) > 1.8
    check_case('full rank', ones_off, 0, s3, e, al, wick, transport, False)
    check_case('zero residual', np.zeros((n, n)), 1, s3, e, al, wick,
               transport, False)
    signed = np.array([[0, 2., -.4], [-.7, 0, 1.3], [.5, -1.1, 0]])
    check_case('signed nonuniform', signed, 1, s3, e, al, wick,
               transport, True)
    # A protected exact U V^T contribution is already compensated in sb.
    # Applying the combined factor leakage again would subtract its diagonal
    # twice. Test the branch decomposition algebra without enabling it in C10.
    u = np.array([[.3], [-.5], [.7]])
    v = np.array([[.6], [.2], [-.4]])
    z, leg = compress(signed, 1)
    protected_z = np.column_stack([z, 3 * v])
    protected_leg = np.column_stack([leg, u])
    protected_leak = 3 * np.sum(u * v, axis=1)
    sb = s3 - protected_leak - np.sum(z * leg, axis=1)
    np.testing.assert_allclose(sb + np.sum(protected_z * protected_leg, axis=1),
                               s3, atol=2e-14)
    assert np.max(abs(sb - (s3 - protected_leak
                            - np.sum(protected_z * protected_leg, axis=1)))) > .1
    print('protected-feed non-double-compensation: PASS')
    print('active hub birth D3 cancellation: PASS')
    print('P2 birth diagonal tiny verification: PASS')


if __name__ == '__main__':
    main()
