"""Lane P3 P3.2: the off-diagonal kappa4 core with memory, as dense torch formulas (twin only).

Scalar metric (ARC's `use_avg_metric`): kappa4_pre = m Sym(Core (x) I), Core = diag(dG) + Off.
The post-activation r = 1 core is 6/(n+4) * L with L_ab = sum_k kappa4_post(a, b, k, k); its
diagonal is the chain's existing g_out, this module builds the OFF-diagonal:

    L_ab = sum_{k != a,b} K211(k; a, b) + K31(a; b) + K31(b; a)

    K211 = pK211 - 2 pK1_k K111_kab - 2 K11_ka K11_kb          (cumulant of a product)
    K31  = pK31  - 3 pK1_a K21_ab   - 3 K11_ab (K2_a + pK1_a^2)

pK211 / pK31 are ARC's AUGMENT dense terms (`get_all_terms_iso(3, 4, augment=True)` restricted
by `factored_keeps_term`): 17 traced (2,1,1) terms (`l211`), 15 (3,1) terms (`p31`). The (1,1,1)
vector part of pK211 (K4.4) and the -2 pK1 K111 conversion (K4.3) need the factored kappa3 and
come in as contractions (`kappa3 contracted with a vector on one leg, distinct indices`).

Every formula here was checked against ARC's `factored_nonlin_kprop_k3` (augment, avg metric)
at n = 24 in float64: `experiments/2026-10-08_p3/k4loop_check.py`.
"""
from __future__ import annotations

import math

import torch

SQ2PI = 1.0 / math.sqrt(2.0 * math.pi)


def relu_wicks(mu, sigma, keys):
    """w(k, p) = E[d^k (relu(Z)^p)], Z ~ N(mu, sigma^2), for the requested (k, p)."""
    a = mu / sigma
    Phi = torch.special.ndtr(a)
    phi = torch.exp(-0.5 * a * a) * SQ2PI
    M = {0: Phi, 1: sigma * phi + mu * Phi, 2: (mu * mu + sigma * sigma) * Phi + mu * sigma * phi,
         3: (mu ** 3 + 3 * mu * sigma * sigma) * Phi + sigma * (mu * mu + 2 * sigma * sigma) * phi}
    he = {0: lambda x: torch.ones_like(x), 1: lambda x: x, 2: lambda x: x * x - 1,
          3: lambda x: x ** 3 - 3 * x, 4: lambda x: x ** 4 - 6 * x * x + 3}
    out = {}
    for k, p in keys:
        if k < p:
            out[(k, p)] = (math.factorial(p) / math.factorial(p - k)) * M[p - k]
        elif k == p:
            out[(k, p)] = math.factorial(p) * Phi
        else:
            j = k - p - 1
            out[(k, p)] = (math.factorial(p) * (-1) ** j) * he[j](a) * phi / sigma ** (k - p)
    return out


WKEYS = [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (1, 2), (2, 2), (3, 2),
         (1, 3), (2, 3), (3, 3), (4, 3), (5, 3)]


def zero_diag(X):
    return X - torch.diag(torch.diagonal(X))


def l211(w, m, Cm, D21, D22, Off, P22=None):
    """sum_{k != a,b} pK211(k; a, b) over ARC's kept AUGMENT terms, BEFORE the (a,b) symmetrisation.

    Cm: pre-activation covariance, zero diagonal. D21[a,c] = kappa3_aac (zero diag).
    D22[a,c] = kappa4_aacc (zero diag). Off: pre-activation core, zero diag (kappa4_kkab = m/6 Off_ab).
    Returns (L, P22) with P22 = Cm diag(w22) Cm (the one product E21b shares).
    """
    w11, w21, w31 = w[(1, 1)], w[(2, 1)], w[(3, 1)]
    w12, w22, w32 = w[(1, 2)], w[(2, 2)], w[(3, 2)]
    C = Cm
    t = C @ w12                          # t_b = sum_k w12_k C_kb
    if P22 is None:
        P22 = C @ (w22[:, None] * C)
    s = Off @ w12
    tb_ex = t[None, :] - w12[:, None] * C          # sum_{k != a} w12_k C_kb   (row a, col b)
    ta_ex = t[:, None] - w12[None, :] * C          # sum_{k != b} w12_k C_ka
    S22 = w22.sum()
    q = (C * C).T @ w22
    D21w12 = D21 @ w12                             # sum_k D21[b,k] w12_k
    D21tw22 = D21.T @ w22                          # sum_k D21[k,b] w22_k
    D22w22 = D22.T @ w22
    L = 2.0 * (w11[:, None] * w21[None, :]) * C * tb_ex                                   # 1
    L = L + (m / 6.0) * (w11[:, None] * w21[None, :]) * (s[:, None] - w12[None, :] * Off.T)  # 2
    L = L + (w11[:, None] * w11[None, :]) * P22                                          # 3
    L = L + (m / 12.0) * (w11[:, None] * w11[None, :]) * Off * (S22 - w22[:, None] - w22[None, :])  # 4
    L = L + (w11[:, None] * w31[None, :]) * C * (D21w12[None, :] - D21.T * w12[:, None])   # 5
    L = L + (w11[:, None] * w31[None, :]) * D21.T * tb_ex                                 # 6
    L = L + (w21[:, None] * w21[None, :]) * D21.T * ta_ex                                 # 7
    L = L + (w11[:, None] * w21[None, :]) * C * (D21tw22[None, :] - w22[:, None] * D21)   # 8
    L = L + (w21[:, None] * w11[None, :]) * (D21 @ (w22[:, None] * C))                    # 9
    L = L + (w11[:, None] * w11[None, :]) * (D21.T @ (w32[:, None] * C))                  # 10
    L = L + (w21[:, None] * w31[None, :]) * (C * C) * tb_ex                               # 11
    L = L + 0.5 * (w21[:, None] * w31[None, :]) * D22 * tb_ex                             # 12
    L = L + (w11[:, None] * w31[None, :]) * C * (q[None, :] - w22[:, None] * C * C)        # 13
    L = L + 0.5 * (w11[:, None] * w31[None, :]) * C * (D22w22[None, :] - w22[:, None] * D22)  # 14
    L = L + (w21[:, None] * w21[None, :]) * C * P22                                      # 15
    L = L + (w11[:, None] * w21[None, :]) * (C @ (w32[:, None] * (C * C)))                # 16
    L = L + 0.5 * (w21[:, None] * w11[None, :]) * (D22 @ (w32[:, None] * C))              # 17
    return L, P22


def p31(w, m, Cm, D21, D22, Off, D3, D4):
    """pK31[a, b] = cumulant-of-powers slice kappa(phi_a^3, phi_b), a != b, ARC's kept terms."""
    w11, w21, w31, w41, w51 = (w[(k, 1)] for k in range(1, 6))
    w13, w23, w33, w43, w53 = (w[(k, 3)] for k in range(1, 6))
    C = Cm
    o = lambda x, y: x[:, None] * y[None, :]
    P = o(w13, w11) * C                                        # 1
    P = P + 0.5 * o(w13, w21) * D21.T                          # 2  kappa3_abb = D21[b, a]
    P = P + 0.5 * o(w23, w11) * D21                            # 3  kappa3_aab = D21[a, b]
    P = P + (m / 12.0) * o(w13, w31) * Off                     # 4  kappa4_abbb = m Off/2, /3!
    P = P + 0.5 * o(w23, w21) * C * C                          # 5
    P = P + 0.25 * o(w23, w21) * D22                           # 6
    P = P + (m / 12.0) * o(w33, w11) * Off                     # 7
    P = P + (1.0 / 6.0) * o(w13, w41 * D3) * C                 # 8
    P = P + 0.5 * o(w23, w31) * C * D21.T                      # 9
    P = P + 0.5 * o(w33, w21) * C * D21                        # 10
    P = P + (1.0 / 6.0) * o(w43 * D3, w11) * C                 # 11
    P = P + (1.0 / 24.0) * o(w13, w51 * D4) * C                # 12
    P = P + (1.0 / 6.0) * o(w33, w31) * C * C * C              # 13
    P = P + 0.25 * o(w33, w31) * C * D22                       # 14
    P = P + (1.0 / 24.0) * o(w53 * D4, w11) * C                # 15
    return zero_diag(P)


def core_off(n, w, m, pre, post, k3_pre_e=None, k3_post_pk1=None, P22=None):
    """The post-activation OFF-diagonal r = 1 core, 6/(n+4) * L, zero diagonal.

    pre:  dict Cm, D21, D22, Off, D3, D4 (pre-activation, zero diagonals where off-diagonal)
    post: dict pK1, K2 (variance), K11 (covariance, zero diag), K21 (kappa3_aab, zero diag)
    k3_pre_e:   sum_{k != a,b} (w12 - 2 pK1 w11)_k kappa3pre_kab  for the INCOMING tower, already
                multiplied by w11_a w11_b (the K4.4 term plus the -2 pK1 conversion of its w1 image)
    k3_post_pk1: sum_{k != a,b} pK1_k kappa3post_kab for the post-activation BLOCKS BORN HERE
    """
    L, P22 = l211(w, m, pre["Cm"], pre["D21"], pre["D22"], pre["Off"], P22)
    K11 = post["K11"]
    L = L - 2.0 * (K11 @ K11)
    if k3_pre_e is not None:
        L = L + k3_pre_e
    if k3_post_pk1 is not None:
        L = L - 2.0 * k3_post_pk1
    pk1 = post["pK1"]
    K31 = (p31(w, m, pre["Cm"], pre["D21"], pre["D22"], pre["Off"], pre["D3"], pre["D4"])
           - 3.0 * pk1[:, None] * post["K21"] - 3.0 * K11 * (post["K2"] + pk1 * pk1)[:, None])
    L = 0.5 * (L + L.T) + K31 + K31.T
    return zero_diag(L) * (6.0 / (n + 4.0)), P22


def e21b_gain(w, mu, Cm, C, n, beta=0.7, P22=None):
    """Lane M2's gain-mode E21b core increment (post-activation basis), m6_terms.e21b_gain_closed."""
    w22 = w[(2, 2)]
    c2 = (mu @ (C @ mu)) / torch.clamp((mu @ mu) ** 2, min=1e-30)
    v = Cm @ (w22 * mu)
    q = (Cm * Cm).T @ w22
    P = P22 if P22 is not None else Cm @ (w22[:, None] * Cm)
    X = Cm * (v[None, :] - (w22 * mu)[:, None] * Cm)
    X = X + mu[:, None] * (q[None, :] - w22[:, None] * Cm * Cm)
    X = X + mu[None, :] * P.T
    X = (2 * beta * c2) * X
    B = w[(1, 1)][:, None] * X * w[(2, 1)][None, :]
    B = (24.0 / (2 * n + 8)) * 0.5 * (B + B.T)
    return zero_diag(B)
