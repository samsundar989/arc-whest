import numpy as np
from scipy.integrate import quad
from relu_hermite import *

rng = np.random.default_rng(0)
mu = np.array([0.0, 0.7, -0.5, 1.3]); sigma = np.array([1.0, 0.8, 1.2, 0.6])
relu = lambda z: np.maximum(z, 0.0)

def Eq(f, lo=-12, hi=12, pts=None):
    return quad(lambda x: f(x)*phi(x), lo, hi, points=pts, limit=400, epsabs=1e-13, epsrel=1e-13)[0]

print("== 1-D moments / Hermite coefficients vs quadrature ==")
m, V, k3, k4 = relu_moments(mu, sigma)
A = coefA(mu, sigma, 6); B = coefB(mu, sigma, 6); C = coefC(mu, sigma, 6)
err = 0
for i in range(4):
    a = mu[i]/sigma[i]
    h = lambda x: relu(mu[i]+sigma[i]*x)
    mq = Eq(h, pts=[-a]); Vq = Eq(lambda x: (h(x)-mq)**2, pts=[-a])
    k3q = Eq(lambda x: (h(x)-mq)**3, pts=[-a]); k4q = Eq(lambda x: (h(x)-mq)**4, pts=[-a]) - 3*Vq**2
    err = max(err, abs(mq-m[i]), abs(Vq-V[i]), abs(k3q-k3[i]), abs(k4q-k4[i]))
    for p in range(0, 7):
        Aq = Eq(lambda x: h(x)*He(p, x), pts=[-a]); Bq = Eq(lambda x: (h(x)-mq)**2*He(p, x), pts=[-a])
        Cq = Eq(lambda x: (h(x)-mq)**3*He(p, x), pts=[-a])
        err = max(err, abs(Aq-A[p, i]), abs(Bq-B[p, i]), abs(Cq-C[p, i]))
print("max abs err (m,V,k3,k4,A_p,B_p,C_p):", err)

print("== triple Hermite product formula vs MC (3 correlated normals) ==")
L = np.array([[1, 0, 0], [0.3, np.sqrt(1-0.09), 0], [-0.5, 0.2, np.sqrt(1-0.25-0.04)]])
R = L@L.T
N = 4_000_000; X = rng.standard_normal((N, 3))@L.T
from math import factorial
for (p, q, r) in [(1, 1, 2), (2, 2, 2), (3, 1, 2), (2, 1, 1), (3, 3, 2)]:
    mc = np.mean(He(p, X[:, 0])*He(q, X[:, 1])*He(r, X[:, 2]))
    th = 0.0
    for a in range(0, p+1):
        b = p-a; c = q-a
        if c < 0 or b+c != r or b < 0: continue
        th += factorial(p)*factorial(q)*factorial(r)/(factorial(a)*factorial(b)*factorial(c))*R[0, 1]**a*R[0, 2]**b*R[1, 2]**c
    print(f"  ({p},{q},{r}) MC {mc:+.4f}  formula {th:+.4f}")

print("== cov_relu vs 2-D quadrature ==")
for (i, j, rho) in [(0, 1, 0.3), (2, 3, -0.45), (1, 3, 0.8)]:
    # E[h_i h_j] = E_x[ h_i(x) E[h_j | x] ],  z_j|x ~ N(mu_j + sigma_j rho x, sigma_j^2 (1-rho^2))
    s = sigma[j]*np.sqrt(1-rho**2)
    def inner(x):
        mm = mu[j]+sigma[j]*rho*x
        return s*phi(mm/s) + mm*Phi(mm/s)
    Ehh = Eq(lambda x: relu(mu[i]+sigma[i]*x)*inner(x), pts=[-mu[i]/sigma[i]])
    rr = np.eye(4); rr[i, j] = rr[j, i] = rho
    Cv = cov_relu(mu, sigma, rr)
    print(f"  rho={rho:+.2f}: series {Cv[i,j]:.12f}  quad {Ehh-m[i]*m[j]:.12f}")

print("== Edgeworth readout c3, c4 (standardized Gamma family) ==")
# z = mu + sigma * (G-k)/sqrt(k), G~Gamma(k): skew 2/sqrt(k), excess kurt 6/k
from scipy.stats import gamma as Gam
for i in [0, 1, 2]:
    for k in [400.0, 100.0]:
        g3 = 2/np.sqrt(k); g4 = 6/k
        f = lambda t: relu(mu[i]+sigma[i]*(t-k)/np.sqrt(k))*Gam.pdf(t, k)
        Ex = quad(f, 0, k+40*np.sqrt(k), points=[k - np.sqrt(k)*mu[i]/sigma[i]], limit=400, epsabs=1e-14, epsrel=1e-13)[0]
        a = mu[i]/sigma[i]
        E0 = sigma[i]*phi(a)+mu[i]*Phi(a)
        E1 = E0 + readout_c3(a)*g3*sigma[i] + readout_c4(a)*g4*sigma[i]
        E2 = E1 + readout_c33(a)*g3**2*sigma[i]
        print(f"  alpha={a:+.3f} k={k:5.0f}: exact-E0 {Ex-E0:+.3e}  resid(1st) {Ex-E1:+.3e}  resid(+k3^2) {Ex-E2:+.3e}  (g3^3-scale {g3**3:.1e})")

print("== transport factors T3 (diag), T2/X2 (pair) via finite-difference families ==")
eps = 0.02
for i in [1, 2]:
    a = mu[i]/sigma[i]
    g, T2, X2, T3, Y2 = transport_factors(mu, sigma)
    # diagonal: z = mu + sigma (x + e (x^2-1)) ; kappa3(z) = sigma^3 (6e + 8e^3)
    def k3h(e):
        hz = lambda x: relu(mu[i]+sigma[i]*(x+e*(x*x-1)))
        roots = np.roots([e, 1, -e - a]); roots = [r.real for r in roots if abs(r.imag) < 1e-12]
        mm = Eq(hz, pts=roots); return Eq(lambda x: (hz(x)-mm)**3, pts=roots)
    d = (k3h(eps)-k3h(-eps))/(2*eps*sigma[i]**3*6)
    print(f"  diag i={i}: dk3(h)/dk3(z) FD {d:.5f}  T3 {T3[i]:.5f}  gate^3 {g[i]**3:.5f}")
    for kk in [0, 3]:
        ak = mu[kk]/sigma[kk]
        # pair T2: z_i = mu_i+sigma_i x ; z_k = mu_k + sigma_k (y + e (x^2-1)) -> kappa3(z_i,z_i,z_k) = 2 e sigma_i^2 sigma_k
        def k3iik(e):
            hi = lambda x: relu(mu[i]+sigma[i]*x)
            mk = lambda x: (sigma[kk]*phi((mu[kk]+sigma[kk]*e*(x*x-1))/sigma[kk]) + (mu[kk]+sigma[kk]*e*(x*x-1))*Phi((mu[kk]+sigma[kk]*e*(x*x-1))/sigma[kk]))
            mkbar = Eq(mk)
            return Eq(lambda x: (hi(x)-m[i])**2*(mk(x)-mkbar), pts=[-a])
        d2 = (k3iik(eps)-k3iik(-eps))/(2*eps*2*sigma[i]**2*sigma[kk])
        # pair X2: z_i = mu_i + sigma_i (x + e (y^2-1)), z_k = mu_k + sigma_k y -> kappa3(z_i,z_k,z_k) = 2 e sigma_i sigma_k^2 ; want kappa3(h)_iik
        def k3iik_b(e):
            hk = lambda y: relu(mu[kk]+sigma[kk]*y)
            def Edhi2(y):
                mm = mu[i]+sigma[i]*e*(y*y-1); s = sigma[i]
                M1 = s*phi(mm/s)+mm*Phi(mm/s); M2 = (mm*mm+s*s)*Phi(mm/s)+mm*s*phi(mm/s)
                return M2-2*m[i]*M1+m[i]**2
            return Eq(lambda y: Edhi2(y)*(hk(y)-m[kk]), pts=[-ak])
        d3 = (k3iik_b(eps)-k3iik_b(-eps))/(2*eps*2*sigma[i]*sigma[kk]**2)
        print(f"  pair i={i},k={kk}: dk3h_iik/dk3z_iik FD {d2:.5f}  T2_i g_k {T2[i]*g[kk]:.5f}  gate {g[i]**2*g[kk]:.5f} | dk3h_iik/dk3z_ikk FD {d3:.5f}  X2_i phi_k/sig_k {X2[i]*Y2[kk]:.5f}")

print("== shift coefficients v3 (dVar/dk3), c3 (dm/dk3) ==")
for i in [1, 2]:
    a = mu[i]/sigma[i]; c3, c4, v3, v4 = shift_coefs(mu, sigma)
    def mv(e):
        hz = lambda x: relu(mu[i]+sigma[i]*(x+e*(x*x-1)))
        roots = np.roots([e, 1, -e - a]); roots = [r.real for r in roots if abs(r.imag) < 1e-12]
        mm = Eq(hz, pts=roots); return mm, Eq(lambda x: (hz(x)-mm)**2, pts=roots)
    mp, vp = mv(eps); mn, vn = mv(-eps)
    dk3 = 2*eps*6*sigma[i]**3
    print(f"  i={i}: dm/dk3 FD {(mp-mn)/dk3:.6f} c3/sigma^2 {c3[i]/sigma[i]**2:.6f} | dVar/dk3 FD {(vp-vn)/dk3:.6f} v3 {v3[i]:.6f}")
