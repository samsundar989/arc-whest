"""
ReLU-of-Gaussian moment / Hermite machinery (numpy only).

Conventions
-----------
z ~ N(mu, sigma^2), alpha = mu/sigma, zhat = (z-mu)/sigma, h = ReLU(z).
Probabilists' Hermite He_p.  Hermite coefficients of h:
    A_p = E[h He_p(zhat)]            h = m + sum_{p>=1} A_p He_p(zhat)/p!
    A_1 = sigma Phi(alpha)  (= sigma*g),  A_p = sigma He_{p-2}(-alpha) phi(alpha)  (p>=2)
Coefficients of u = (h-m)^2:
    B_p = E[(h-m)^2 He_p(zhat)]
    B_0 = Var(h), B_1 = 2 sigma m (1-Phi), B_2 = 2 sigma^2 Phi - 2 m sigma phi,
    B_p = 2 sigma^2 He_{p-3}(-alpha) phi - 2 m sigma He_{p-2}(-alpha) phi  (p>=3)
Coefficients of (h-m)^3:
    C_p = E[(h-m)^3 He_p(zhat)]   (C_3 needed for the diagonal kappa3 transport)
Triple product of Hermite polynomials of jointly standard normals with correlations
    E[He_p(x)He_q(y)He_r(w)] = sum_{a+b=p, a+c=q, b+c=r} p!q!r!/(a!b!c!) rho_xy^a rho_xw^b rho_yw^c
"""
import numpy as np
from math import factorial
from scipy.special import ndtr as Phi
from scipy.special import eval_hermitenorm as He

SQ2PI = np.sqrt(2*np.pi)
def phi(x):
    return np.exp(-0.5*x*x)/SQ2PI

# ---------- 1-D moments of ReLU(N(mu,sigma^2)) ----------
def relu_moments(mu, sigma):
    """returns m, Var, kappa3, kappa4 (central cumulants) of ReLU(z)."""
    a = mu/sigma; P = Phi(a); p = phi(a)
    M1 = sigma*p + mu*P
    M2 = (mu**2 + sigma**2)*P + mu*sigma*p
    M3 = (mu**3 + 3*mu*sigma**2)*P + (mu**2 + 2*sigma**2)*sigma*p
    M4 = (mu**4 + 6*mu**2*sigma**2 + 3*sigma**4)*P + (mu**3 + 5*mu*sigma**2)*sigma*p
    V = M2 - M1**2
    k3 = M3 - 3*M1*M2 + 2*M1**3
    m4c = M4 - 4*M1*M3 + 6*M1**2*M2 - 3*M1**4
    k4 = m4c - 3*V**2
    return M1, V, k3, k4

def gate(mu, sigma):
    return Phi(mu/sigma)

# ---------- Hermite coefficients ----------
def coefA(mu, sigma, P):
    """A_p for p=0..P, shape (P+1, n). A_0 = m."""
    a = mu/sigma; p = phi(a)
    m = sigma*p + mu*Phi(a)
    A = np.empty((P+1,)+np.shape(mu))
    A[0] = m
    A[1] = sigma*Phi(a)
    for k in range(2, P+1):
        A[k] = sigma*He(k-2, -a)*p
    return A

def coefB(mu, sigma, P):
    """B_p for p=0..P of (h-m)^2."""
    a = mu/sigma; p = phi(a); Pa = Phi(a)
    m, V, _, _ = relu_moments(mu, sigma)
    B = np.empty((P+1,)+np.shape(mu))
    B[0] = V
    if P >= 1: B[1] = 2*sigma*m*(1-Pa)
    if P >= 2: B[2] = 2*sigma**2*Pa - 2*m*sigma*p
    for k in range(3, P+1):
        B[k] = 2*sigma**2*He(k-3, -a)*p - 2*m*sigma*He(k-2, -a)*p
    return B

def coefC3(mu, sigma):
    """C_3 = E[(h-m)^3 He_3(zhat)] = sigma^3 E[d^3/dz^3 (h-m)^3]
    (h-m)^3 ''' = 6*1_{z>0} - 6 m delta(z) + 3 m^2 delta'(z)
    E[delta(z)] = phi(a)/sigma, E[delta'(z)] = -p'(0) = -a phi(a)/sigma^2"""
    a = mu/sigma; p = phi(a); Pa = Phi(a)
    m = sigma*p + mu*Pa
    return sigma**3*(6*Pa - 6*m*p/sigma - 3*m**2*a*p/sigma**2)

# ---------- exact first-order responses of ReLU output cumulants ----------
def readout_c3(alpha):
    """dE/d(kappa3/sigma^2) = -alpha phi(alpha)/6"""
    return -alpha*phi(alpha)/6.0
def readout_c4(alpha):
    """dE/d(kappa4/sigma^3) = (alpha^2-1) phi(alpha)/24"""
    return (alpha**2-1)*phi(alpha)/24.0
def readout_c33(alpha):
    """second-order Edgeworth: dE/d(kappa3^2/sigma^5) = He_4(alpha) phi(alpha)/72"""
    return He(4, alpha)*phi(alpha)/72.0

def transport_factors(mu, sigma):
    """Exact first-order (zero-correlation) response factors of kappa3(h) to kappa3(z):
       distinct (i,j,k):  g_i g_j g_k                      (gate)
       pair (i,i,k):      T2[i]*g[k] * k3z_iik + X2[i]*phi_k/sigma_k * k3z_ikk
                          T2 = B_2/(2 sigma^2) = Phi - (m/sigma) phi,  X2 = B_1/(2 sigma) = m(1-Phi)
       diagonal (i,i,i):  T3 = (C_3 - 3 Var A_3)/(6 sigma^3)
    returns g, T2, X2, T3, and Y2 = phi/sigma (the A_2/sigma^2 factor)"""
    a = mu/sigma; p = phi(a); Pa = Phi(a)
    m, V, _, _ = relu_moments(mu, sigma)
    g = Pa
    T2 = Pa - (m/sigma)*p
    X2 = m*(1-Pa)
    Y2 = p/sigma
    A3 = -sigma*a*p
    T3 = (coefC3(mu, sigma) - 3*V*A3)/(6*sigma**3)
    return g, T2, X2, T3, Y2

def shift_coefs(mu, sigma):
    """first-order response of m and Var(h) to kappa3_iii(z), kappa4_iiii(z):
       dm   = c3 * k3/sigma^2 + c4 * k4/sigma^3
       dVar = v3 * k3 + v4 * k4
       v3 = phi (sigma + m alpha)/(3 sigma^2) ,  v4 = -phi (sigma alpha + m(alpha^2-1))/(12 sigma^3)"""
    a = mu/sigma; p = phi(a)
    m = sigma*p + mu*Phi(a)
    c3 = readout_c3(a); c4 = readout_c4(a)
    v3 = p*(sigma + m*a)/(3*sigma**2)
    v4 = -p*(sigma*a + m*(a*a-1))/(12*sigma**3)
    return c3, c4, v3, v4

# ---------- pair series ----------
def pair_series(Xi, Yj, rho, pmax, pmin=1):
    """sum_{p=pmin}^{pmax} Xi[p][:,None]*Yj[p][None,:]*rho**p/p!  (n x n)."""
    out = np.zeros_like(rho)
    rp = np.ones_like(rho)
    for p in range(1, pmax+1):
        rp = rp*rho
        if p >= pmin:
            out += np.outer(Xi[p], Yj[p])*rp/factorial(p)
    return out

def cov_relu(mu, sigma, rho, pmax=40):
    """exact Cov(ReLU(z_i), ReLU(z_j)) for Gaussian z: diag exact, offdiag Hermite series."""
    A = coefA(mu, sigma, pmax)
    Cov = pair_series(A, A, rho, pmax)
    _, V, _, _ = relu_moments(mu, sigma)
    np.fill_diagonal(Cov, V)
    return Cov

# ---------- triple Hermite / distinct-born kappa3 ----------
def triple_terms(maxorder):
    """list of (a,b,c, coefficient 1/(a!b!c!)) with a+b+c<=maxorder, each vertex degree>=1."""
    out = []
    for a in range(maxorder+1):
        for b in range(maxorder+1-a):
            for c in range(maxorder+1-a-b):
                if a+b >= 1 and a+c >= 1 and b+c >= 1:
                    out.append((a, b, c, 1.0/(factorial(a)*factorial(b)*factorial(c))))
    return out

def born_kappa3_full(mu, sigma, rho, maxorder=6, pmax=40, include_distinct=True,
                     include_pair=True, min_distinct_order=2, max_distinct_order=None):
    """Full n^3 third-cumulant tensor of h = ReLU(z), z ~ N(mu, C), rho = correlation matrix.
    diagonal: exact; pair (i,i,k) entries: Hermite pair series; distinct: triple series
    with a+b+c <= maxorder (optionally restricted to a band of orders)."""
    n = len(mu)
    A = coefA(mu, sigma, max(pmax, maxorder))
    K = np.zeros((n, n, n))
    if include_distinct:
        hi = maxorder if max_distinct_order is None else max_distinct_order
        for (a, b, c, coef) in triple_terms(hi):
            if a+b+c < min_distinct_order: continue
            K += coef*np.einsum('i,j,k,ij,ik,jk->ijk', A[a+b], A[a+c], A[b+c],
                                rho**a, rho**b, rho**c, optimize=True)
    # overwrite pair entries
    idx = np.arange(n)
    if include_pair:
        B = coefB(mu, sigma, pmax)
        Pm = pair_series(B, A, rho, pmax)       # Pm[i,k] = kappa3_{iik}
    else:
        Pm = np.zeros((n, n))
    K[idx, idx, :] = Pm
    K[idx, :, idx] = Pm
    K[:, idx, idx] = Pm.T
    # diagonal exact
    _, _, k3, _ = relu_moments(mu, sigma)
    K[idx, idx, idx] = k3
    return K

def born_kappa3_parts(mu, sigma, rho, pmax=40):
    """returns d (diag), Pm (pair matrix kappa3_iik) only."""
    A = coefA(mu, sigma, pmax); B = coefB(mu, sigma, pmax)
    Pm = pair_series(B, A, rho, pmax); np.fill_diagonal(Pm, 0.0)
    _, _, k3, _ = relu_moments(mu, sigma)
    return k3, Pm

# ---------- kappa4 patterns (diag, (iijj), (iiij)) ----------
def coefC(mu, sigma, P):
    """C_p = E[(h-m)^3 He_p], p=0..P via Stein: sigma^p E[d^p (h-m)^3].
    (h-m)^3: d1 = 3 dh^2 h', d2 = 6 dh 1_{z>0} + 3 m^2 delta,  d3 = 6*1 - 6 m delta + 3 m^2 delta',
    d^p (p>=4) = -6 m delta^{(p-3)} + 3 m^2 delta^{(p-2)} + 6 delta^{(p-4)}
    E[delta^{(k)}(z)] = sigma^{-k-1} He_k(-alpha) phi(alpha)"""
    a = mu/sigma; p = phi(a); Pa = Phi(a)
    m, V, k3, _ = relu_moments(mu, sigma)
    def Ed(k):  # E[delta^{(k)}(z)]
        return sigma**(-k-1)*He(k, -a)*p
    C = np.empty((P+1,)+np.shape(mu))
    C[0] = k3
    # d1 = 3 (h-m)^2 h' : E = 3 E[(h-m)^2 1_{z>0}] = 3( E[h^2 1] -2m E[h 1] + m^2 Phi ) = 3(M2 - 2 m M1 + m^2 Phi)
    M1 = m; M2 = (mu**2+sigma**2)*Pa + mu*sigma*p
    if P >= 1: C[1] = sigma*3*(M2 - 2*m*M1 + m*m*Pa)
    if P >= 2: C[2] = sigma**2*(6*(M1 - m*Pa) + 3*m*m*Ed(0))
    if P >= 3: C[3] = sigma**3*(6*Pa - 6*m*Ed(0) + 3*m*m*Ed(1))
    for k in range(4, P+1):
        C[k] = sigma**k*(-6*m*Ed(k-3) + 3*m*m*Ed(k-2) + 6*Ed(k-4))
    return C

def born_kappa4_patterns(mu, sigma, rho, pmax=40):
    """diag kappa4_iiii (exact), pair kappa4_iijj (series, zero diag), kappa4_iiij (series, zero diag).
    kappa4_iijj = E[dh_i^2 dh_j^2] - V_i V_j - 2 Cov_ij^2 = sum_{p>=1} B_p B_p rho^p/p! - 2 Cov^2
    kappa4_iiij = E[dh_i^3 dh_j] - 3 V_i Cov_ij = sum_{p>=1} C_p(i) A_p(j) rho^p/p! - 3 V_i Cov_ij"""
    A = coefA(mu, sigma, pmax); B = coefB(mu, sigma, pmax); C = coefC(mu, sigma, pmax)
    m, V, k3, k4 = relu_moments(mu, sigma)
    Cov = pair_series(A, A, rho, pmax); np.fill_diagonal(Cov, 0.0)
    K22 = pair_series(B, B, rho, pmax) - 2*Cov**2; np.fill_diagonal(K22, 0.0)
    K31 = pair_series(C, A, rho, pmax) - 3*V[:, None]*Cov; np.fill_diagonal(K31, 0.0)
    return k4, K22, K31

def e1_chi(n):
    """E[chi_n]/sqrt(n) = sqrt(2/n) Gamma((n+1)/2)/Gamma(n/2)"""
    from scipy.special import gammaln
    return np.exp(0.5*np.log(2.0/n) + gammaln((n+1)/2) - gammaln(n/2))
