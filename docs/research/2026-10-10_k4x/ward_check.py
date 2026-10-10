# Check the radial (random-gain) cumulant shifts to first order in v = Var G, with z = G*y, G indep of y, E G = 1.
# Claim (Sym = average over permutations): dk3 = v[3 k3 + 6 Sym(mu x C)], dk4 = v[6 k4 + 12 Sym(C x C) + 12 Sym(mu x k3)]
# Readouts: Q31_ab += 3v(3 m_a D21_ab + m_b D3_a) + 12 v C_aa C_ab + 6v Q31_ab ; D22_ab += 6v(m_a D21_ba + m_b D21_ab) + 4v(C_aa C_bb + 2 C_ab^2) + 6v D22 ; D4_a += 12 v m_a D3_a + 12 v C_aa^2 + 6 v D4
# (D21_ab = k3(a,a,b); D21_ba = k3(b,b,a).) The residual of the k4 check scales like v (O(v^2) terms), i.e. the first-order formula is exact.
import numpy as np, itertools, math
rng = np.random.default_rng(0); d = 3; K = 9
Y = rng.normal(size=(K, d)) + np.array([0.7, -0.3, 1.1]); p = rng.random(K); p /= p.sum()
def cumulants(X, w):
    m = w @ X; Xc = X - m
    C = np.einsum('k,ki,kj->ij', w, Xc, Xc)
    k3 = np.einsum('k,ki,kj,kl->ijl', w, Xc, Xc, Xc)
    m4 = np.einsum('k,ki,kj,kl,kq->ijlq', w, Xc, Xc, Xc, Xc)
    k4 = m4 - (np.einsum('ij,lq->ijlq', C, C) + np.einsum('il,jq->ijlq', C, C) + np.einsum('iq,jl->ijlq', C, C))
    return m, C, k3, k4
def sym(T):
    r = T.ndim
    return sum(np.transpose(T, q) for q in itertools.permutations(range(r))) / math.factorial(r)
m, C, k3, k4 = cumulants(Y, p)
for v in (1e-3, 1e-4):
    g = np.array([1 - np.sqrt(v), 1 + np.sqrt(v)]); pg = np.array([0.5, 0.5])
    Z = (g[:, None, None] * Y[None]).reshape(-1, d); w = (pg[:, None] * p[None]).reshape(-1)
    mz, Cz, k3z, k4z = cumulants(Z, w)
    pred3 = 3*k3 + 6*sym(np.einsum('i,jl->ijl', m, C))
    pred4 = 6*k4 + 12*sym(np.einsum('ij,lq->ijlq', C, C)) + 12*sym(np.einsum('i,jlq->ijlq', m, k3))
    e3 = np.abs((k3z - k3)/v - pred3).max(); e4 = np.abs((k4z - k4)/v - pred4).max()
    a, b = 0, 2
    q31 = (k4z[a,a,a,b] - k4[a,a,a,b])/v
    q31p = 3*(3*m[a]*k3[a,a,b] + m[b]*k3[a,a,a]) + 12*C[a,a]*C[a,b] + 6*k4[a,a,a,b]
    d22 = (k4z[a,a,b,b] - k4[a,a,b,b])/v
    d22p = 6*(m[a]*k3[b,b,a] + m[b]*k3[a,a,b]) + 4*(C[a,a]*C[b,b] + 2*C[a,b]**2) + 6*k4[a,a,b,b]
    d4 = (k4z[a,a,a,a] - k4[a,a,a,a])/v
    d4p = 12*m[a]*k3[a,a,a] + 12*C[a,a]**2 + 6*k4[a,a,a,a]
    print(f"v={v:g}  max|dk3/v - pred| {e3:.2e}  max|dk4/v - pred| {e4:.2e}  Q31 {q31:.5f} vs {q31p:.5f}  D22 {d22:.5f} vs {d22p:.5f}  D4 {d4:.5f} vs {d4p:.5f}")
