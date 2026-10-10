# kappa4 (2,2)-block readout anatomy after ONE transport step P (toy; proxy block K_rc = C_rc^2).
# D4 and D31 have no cross terms (O(n^3)); D22 = coherent [(P∘P)K(P∘P)^T] + cross [2 Σ K_rc P_ar P_br P_ac P_bc] (n^4).
# coherent = const + additive (f_a+f_b) + non-additive remainder. (Strategist deep review 2026-10-10 §6.1.)
import numpy as np
def run(n, seed):
    rng = np.random.default_rng(seed)
    S = np.eye(n); mu = np.zeros(n)
    for _ in range(5):   # crude post-ReLU covariance proxy (zero-mean arc-cosine on 2nd moments)
        W = rng.normal(0, np.sqrt(2/n), (n, n))
        M2 = W @ (S + np.outer(mu, mu)) @ W.T
        d = np.sqrt(np.diag(M2)); rho = np.clip(M2/np.outer(d, d), -1, 1); th = np.arccos(rho)
        E = np.outer(d, d)*(np.sin(th) + (np.pi - th)*np.cos(th))/(2*np.pi)
        mu = d/np.sqrt(2*np.pi); S = E - np.outer(mu, mu)
    K = (S - np.diag(np.diag(S)))**2
    P = rng.normal(0, np.sqrt(2/n), (n, n)); P2 = P*P
    coh = P2 @ K @ P2.T
    cross = np.empty((n, n))
    for a in range(n):
        V = P[a][None, :]*P
        cross[a] = 2.0*np.einsum('br,br->b', V @ K, V)
    off = ~np.eye(n, dtype=bool)
    Xo = np.where(off, coh, 0.0); rs = Xo.sum(1); tot = Xo.sum()
    f = (rs - tot/n)/(n - 2); c = tot/(n*(n - 1)) - 2*f.mean()
    A = c + f[:, None] + f[None, :]
    NA = (coh - A)[off]; X = (cross - cross[off].mean())[off]; AD = (A - coh[off].mean())[off]
    rms = lambda v: np.sqrt((v**2).mean())
    print(f"n={n:3d} s={seed}  additive {rms(AD):.2e}  coh-nonadd {rms(NA):.2e}  cross {rms(X):.2e}  "
          f"cross/coh-nonadd {rms(X)/rms(NA):.2f}  corr {np.corrcoef(NA, X)[0,1]:+.2f}  "
          f"O(n^3) share of traceless D22 var {rms(NA)**2/((NA+X)**2).mean():.2f}")
for n in (64, 128, 256):
    for s in range(3): run(n, s)
