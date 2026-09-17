"""Float64 reference for ONE nonlinear step, driven by a canonical term table.

Handoff 2026-09-17 step 1: "a small stable interface, not a new estimator
framework". The interface is the O(n^2) boundary a local consumer reads:

    mu_pre (n,)  cov_pre (n,n)  D3 (n,)  D21 (n,n) [kappa_iij, zero diagonal]
    dG (n,)  lam (scalar, the value this layer consumes)  metric (scalar)

and the carried fourth-cumulant MODEL both v32 and v17 use,

    kappa4 = m Sym(G (x) I),  G = diag(dG) + lam C_off
    ->  D4 = m dG,  D22 = (m/6)(dG_i + dG_j),  Q31 = kappa_iiij = (m/2) lam C_ij.

Outputs: every oriented pK term family, pK, K (moment -> connected), and the
regenerated per-neuron core g_out = (K4 + rowsum K22) 6/(n+4) + trace part.

It is numpy, float64 and slow on purpose: it is the audit instrument, never the
estimator. Its parity against the LIVE v32 `_layer` is `test_ref_step.py`.
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
from scipy.special import ndtr

ALPHA_CAP = 40.0


def hermite_prob(m, x):
    """Probabilists' Hermite He_m(x)."""
    h0, h1 = np.ones_like(x), x
    if m == 0:
        return h0
    for k in range(1, m):
        h0, h1 = h1, x * h1 - k * h0
    return h1


def _partial_std_moment(r, a, phi, Phi):
    """E[z^r 1{z > -a}], z ~ N(0,1): recursion E_r = (-a)^{r-1} phi(a) + (r-1) E_{r-2}."""
    E = [Phi, phi]
    for q in range(2, r + 1):
        E.append(((-a) ** (q - 1)) * phi + (q - 1) * E[q - 2])
    return E[r]


def wick(k, p, mu, sigma):
    """w(k,p) = E_{N(mu,sigma^2)}[ d^k/dy^k ReLU(y)^p ], distributional derivatives."""
    a = np.clip(mu / sigma, -ALPHA_CAP, ALPHA_CAP)
    phi = np.exp(-0.5 * a * a) / math.sqrt(2.0 * math.pi)
    Phi = ndtr(a)
    if k > p:
        r = k - p - 1
        return math.factorial(p) * ((-1.0) ** r) * sigma ** (-(r + 1)) * hermite_prob(r, a) * phi
    q = p - k
    # E[y^q 1{y>0}] with y = sigma (a + z)
    acc = np.zeros_like(mu)
    for r in range(q + 1):
        acc = acc + math.comb(q, r) * a ** (q - r) * _partial_std_moment(r, a, phi, Phi)
    return (math.factorial(p) / math.factorial(q)) * sigma ** q * acc


def blocks(bd):
    """Canonical block arrays from a boundary. Bivariate blocks are (n,n), zero diagonal."""
    n = bd["mu_pre"].shape[0]
    off = 1.0 - np.eye(n)
    m, lam = float(bd["metric"]), float(bd["lam"])
    C = bd["cov_pre"] * off
    dG = bd["dG"]
    D3, D21 = bd["D3"], bd["D21"] * off
    D4 = m * dG
    one = np.ones(n)
    return {
        "uni": {"D3": D3, "D4": D4},
        "biv": {"C": C, "D21": D21, "D12": D21.T, "D22": (m / 6.0) * (dG[:, None] + dG[None, :]) * off,
                "D3i": D3[:, None] * one[None, :], "D3j": one[:, None] * D3[None, :],
                "D4i": D4[:, None] * one[None, :], "D4j": one[:, None] * D4[None, :],
                "Q31": (0.5 * m * lam) * C, "Q13": (0.5 * m * lam) * C},
    }


def pk_terms(bd, table, want_families=False):
    """pK[output] = sum over the table's canonical records. Returns (pK, families)."""
    mu = bd["mu_pre"]
    var = np.maximum(np.diag(bd["cov_pre"]), 1e-12)
    sigma = np.sqrt(var)
    B = blocks(bd)
    n = mu.shape[0]
    off = 1.0 - np.eye(n)
    wcache = {}

    def w(k, p):
        if (k, p) not in wcache:
            wcache[(k, p)] = wick(k, p, mu, sigma)
        return wcache[(k, p)]

    pK, fam = {}, {}
    for t in table:
        out = tuple(t["output"])
        c = float(Fraction(t["coef"]))
        if len(out) == 1:
            val = np.full(n, c)
            for f in t["factors"]:
                val = val * B["uni"][f]
            val = val * w(t["derivs"][0], out[0])
        else:
            val = np.full((n, n), c)
            for f in t["factors"]:
                val = val * B["biv"][f]
            val = val * w(t["derivs"][0], out[0])[:, None] * w(t["derivs"][1], out[1])[None, :]
            val = val * off
        pK[out] = pK.get(out, 0.0) + val
        if want_families:
            fam[(out, tuple(t["factors"]), tuple(t["derivs"]))] = val
    return pK, fam


def pk_to_k(pK):
    p1, p2, p3, p4 = (pK[(q,)] for q in (1, 2, 3, 4))
    K = {(1,): p1,
         (2,): p2 - p1 * p1,
         (3,): p3 - 3 * p1 * p2 + 2 * p1 ** 3,
         (4,): p4 - 4 * p1 * p3 - 3 * p2 * p2 + 12 * p1 * p1 * p2 - 6 * p1 ** 4}
    p11, p21, p22 = pK[(1, 1)], pK[(2, 1)], pK[(2, 2)]
    p12 = pK[(1, 2)] if (1, 2) in pK else p21.T
    K[(1, 1)] = p11
    K[(2, 1)] = p21 - 2 * p1[:, None] * p11
    K[(2, 2)] = (p22 - 2 * p1[None, :] * p21 - 2 * p1[:, None] * p12 - 2 * p11 * p11
                 + 4 * p1[:, None] * p1[None, :] * p11)
    return K


def core(K):
    n = K[(1,)].shape[0]
    k22row = K[(2, 2)].sum(axis=1)
    k4 = K[(4,)]
    return (k4 + k22row) * (6.0 / (n + 4.0)) + (k4.sum() + k22row.sum()) * (-3.0 / ((n + 2.0) * (n + 4.0)))


def step(bd, table, want_families=False):
    pK, fam = pk_terms(bd, table, want_families)
    for out in [(1,), (2,), (3,), (4,), (1, 1), (2, 1), (2, 2)]:
        pK.setdefault(out, np.zeros_like(bd["mu_pre"]) if len(out) == 1 else np.zeros_like(bd["cov_pre"]))
    # symmetric outputs are already symmetric when the table lists both orientations
    K = pk_to_k(pK)
    return {"pK": pK, "K": K, "g_out": core(K), "mu_out": K[(1,)],
            "cov_out": np.diag(K[(2,)]) + K[(1, 1)], "families": fam}
