"""Stage 2 (P3) derivation: which kappa3 x covariance routes feed an all-distinct birth?

Label-free. Execution contract: `experiments/LOG.md`, 2026-09-16 amendment, P3.

THE EXACT LAW. x ~ N(mu, Sigma), Sigma = I + eps C (unit variances, 3 neurons),
z = x + delta q(x), q_i = xi^T B^i xi - tr(B^i Sigma), xi = x - mu, B^i symmetric.
To first order in delta: mean and covariance of z are unchanged, kappa4 = O(delta^2),
and  kappa3_abc(z) = 2 delta [ (S B^c S)_ab + (S B^b S)_ac + (S B^a S)_bc ].
The derivative of every post-ReLU moment in delta is a Gaussian expectation of
products of per-variable factors (H(z) z^p (z-mu)^q), computed by nested
Gauss-Legendre / Gauss-Hermite quadrature (smooth integrands only).

THE ESTIMATOR. The candidate's real `_layer` at W = I with the input kappa3
carried as a dense tensor (`dense_tower.py`), kappa4 = 0.

ROUTES into an all-distinct output entry at order kappa3 * C (first-order
cumulant expansion), summed over the 6 role assignments (d doubled, s single,
t third) of (a,b,c):
   (beta1)  1/2 kappa3_dds C_dt w3_d w1_s w1_t
   (beta2)  1/2 kappa3_dds C_st w2_d w2_s w1_t
   (gamma)  kappa3_abc C_pq w2_p w2_q w1_r  over the 3 pairs -- the DISTINCT
            input times a link; not representable cheaply, measured for size.
The zeroth-order route w1^3 kappa3_abc is in every candidate already.

Checks (thresholds fixed before running):
  A. control, eps = 0: estimator all-distinct delta-response == exact (1e-6 rel).
  B. discriminator: odd-eps delta*eps response, v32 vs exact.
  C. v32 + beta1 + beta2 (+ gamma) vs exact (5%): which routes close it.

Run: uv run python local_sim/k3feed_oracle.py [--json out]
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]

import numpy as np  # noqa: E402
from scipy.special import ndtr  # noqa: E402

import dense_tower as DT  # noqa: E402
import k4feed_oracle as K  # noqa: E402

_GL_X, _GL_W = np.polynomial.legendre.leggauss(120)
_HE_X, _HE_W = np.polynomial.hermite_e.hermegauss(90)
_HE_W = _HE_W / np.sqrt(2.0 * np.pi)
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<62} {detail}")


def _nodes(m, v, heav):
    s = np.sqrt(v)
    if heav:
        hi = max(m + 14.0 * s, 1e-12)
        t = 0.5 * hi * (_GL_X + 1.0)
        w = 0.5 * hi * _GL_W * K._phi((t - m) / s) / s
        return t, w
    return m + s * _HE_X, _HE_W.copy()


def _f(t, spec):
    p, q, h, c = spec
    out = np.ones_like(t)
    if p:
        out = out * t ** p
    if q:
        out = out * (t - c) ** q
    return out


def gmoment(mu, S, specs):
    """E[prod_i f_i(z_i)], z ~ N(mu, S); spec = (p, q, heaviside, center):
    f(z) = [H(z) if heaviside or p > 0] z^p (z - center)^q."""
    mu = np.asarray(mu, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    heav0 = specs[0][2] or specs[0][0] > 0
    t, w = _nodes(mu[0], S[0, 0], heav0)
    f0 = _f(t, specs[0])
    if len(specs) == 1:
        return float(np.sum(w * f0))
    c = S[1:, 0] / S[0, 0]
    Sc = S[1:, 1:] - np.outer(S[1:, 0], S[0, 1:]) / S[0, 0]
    vals = np.array([gmoment(mu[1:] + c * (tq - mu[0]), Sc, specs[1:]) for tq in t])
    return float(np.sum(w * f0 * vals))


def exact_response(mu, Sig, Bs):
    """d/d delta of kappa3(y0,y1,y2) and of its inputs, at delta = 0."""
    n = 3
    base = lambda idx: tuple((1, 0, False, mu[i]) if i in idx else (0, 0, False, mu[i]) for i in range(n))

    def E(idx):
        return gmoment(mu, Sig, base(idx))

    def dE(idx):
        # sum_i in idx  E[ H(z_i) prod_{j in idx, j != i} relu(z_j) q_i(xi) ]
        tot = 0.0
        for i in idx:
            Bi = Bs[i]
            const = float(np.trace(Bi @ Sig))
            for j in range(n):
                for k in range(n):
                    if Bi[j, k] == 0.0:
                        continue
                    specs = []
                    for v in range(n):
                        p = 1 if (v in idx and v != i) else 0
                        h = (v == i)
                        q = (1 if v == j else 0) + (1 if v == k else 0)
                        specs.append((p, q, h, mu[v]))
                    tot += Bi[j, k] * gmoment(mu, Sig, tuple(specs))
            specs = [((1 if (v in idx and v != i) else 0), 0, v == i, mu[v]) for v in range(n)]
            tot -= const * gmoment(mu, Sig, tuple(specs))
        return tot

    e = {k: E(k) for k in [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]}
    d = {k: dE(k) for k in e}
    m0, m1, m2 = e[(0,)], e[(1,)], e[(2,)]
    dm0, dm1, dm2 = d[(0,)], d[(1,)], d[(2,)]
    dk3 = (d[(0, 1, 2)]
           - (d[(0, 1)] * m2 + e[(0, 1)] * dm2)
           - (d[(0, 2)] * m1 + e[(0, 2)] * dm1)
           - (d[(1, 2)] * m0 + e[(1, 2)] * dm0)
           + 2.0 * (dm0 * m1 * m2 + m0 * dm1 * m2 + m0 * m1 * dm2))
    return dk3


def input_k3(Sig, Bs):
    n = Sig.shape[0]
    T = np.zeros((n, n, n))
    SBS = [Sig @ B @ Sig for B in Bs]
    for a, b, c in itertools.product(range(n), repeat=3):
        T[a, b, c] = 2.0 * (SBS[c][a, b] + SBS[b][a, c] + SBS[a][b, c])
    return T


def estimator_response(dmod, mu, Sig, Tin, h=1e-4):
    import flopscope as flops
    import flopscope.numpy as fnp
    n = 3

    def run(Tt):
        dt = fnp.float64
        table = dmod._tables()
        const = dmod.Estimator()._constants(n, dt)
        with flops.BudgetContext(flop_budget=2 ** 55, quiet=True):
            out = dmod._layer(fnp.asarray(np.eye(n), dtype=dt),
                              (fnp.asarray(mu, dtype=dt), fnp.asarray(Sig, dtype=dt),
                               DT.DenseTower(Tt, 1), 0.0, fnp.asarray(np.zeros(n), dtype=dt)),
                              table, dt, const, 2.0, last_layer=False, gens=None, layer_index=1)
        return out[2].T

    old = list(dmod._K4_LAM)
    dmod._K4_LAM = [0.0] * len(old)
    try:
        return (run(h * Tin) - run(-h * Tin)) / (2.0 * h)
    finally:
        dmod._K4_LAM = old


def routes(Tin, mu, Sig):
    """beta1, beta2, gamma full tensors (all-distinct entries meaningful)."""
    n = 3
    sig = np.sqrt(np.diag(Sig))
    al = mu / sig
    w1, w2, w3 = ndtr(al), K._phi(al) / sig, -al * K._phi(al) / sig ** 2
    C = Sig * (1.0 - np.eye(n))
    D21 = np.einsum("dds->ds", Tin)
    b1 = np.zeros((n, n, n))
    b2 = np.zeros((n, n, n))
    g = np.zeros((n, n, n))
    for a, b, c in itertools.product(range(n), repeat=3):
        if len({a, b, c}) < 3:
            continue
        for d, s, t in itertools.permutations((a, b, c)):
            b1[a, b, c] += 0.5 * D21[d, s] * C[d, t] * w3[d] * w1[s] * w1[t]
            b2[a, b, c] += 0.5 * D21[d, s] * C[s, t] * w2[d] * w2[s] * w1[t]
        for p, q, r in ((a, b, c), (a, c, b), (b, c, a)):
            g[a, b, c] += Tin[a, b, c] * C[p, q] * w2[p] * w2[q] * w1[r]
    return b1, b2, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default=os.path.join(_ROOT, "estimator_k3_v32_dense.py"))
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    import flopscope as flops
    flops.configure(symmetry_warnings=False)
    dmod, _ = DT.load_dense(a.candidate)
    rng = np.random.default_rng(20260917)
    mu = np.array([0.3, -0.25, 0.15])
    C = rng.normal(size=(3, 3))
    C = (C + C.T) * 0.5
    np.fill_diagonal(C, 0.0)
    Bs = []
    for _ in range(3):
        B = rng.normal(size=(3, 3)) * 0.3
        Bs.append((B + B.T) * 0.5)
    idx = (0, 1, 2)
    report = {"candidate": os.path.basename(a.candidate)}

    def at(eps):
        Sig = np.eye(3) + eps * C
        Tin = input_k3(Sig, Bs)
        ex = exact_response(mu, Sig, Bs)
        est = estimator_response(dmod, mu, Sig, Tin)[idx]
        b1, b2, g = routes(Tin, mu, Sig)
        return ex, est, b1[idx], b2[idx], g[idx]

    e0 = at(0.0)
    check("A. control eps=0: all-distinct response, estimator == exact (1e-6 rel)",
          abs(e0[0] - e0[1]) <= 1e-6 * abs(e0[0]), f"exact {e0[0]:+.6e}  est {e0[1]:+.6e}")
    eps = 0.01
    ep, em = at(eps), at(-eps)
    odd = [(x - y) / (2 * eps) for x, y in zip(ep, em)]
    ex, est, b1, b2, g = odd
    print(f"    odd-eps response per unit eps: exact {ex:+.5e}  v32 {est:+.5e}  "
          f"beta1 {b1:+.5e}  beta2 {b2:+.5e}  gamma {g:+.5e}")
    report.update({"eps0": {"exact": e0[0], "estimator": e0[1]},
                   "odd_eps": {"exact": ex, "estimator": est, "beta1": b1, "beta2": b2, "gamma": g}})
    check("B. discriminator: v32 misses > 20% of the odd-eps response",
          abs(ex - est) > 0.2 * abs(ex), f"missing {ex - est:+.4e}")

    def rel(x):
        return abs(x - ex) / abs(ex)
    check("C. v32 + beta1 + beta2 == exact (5%)", rel(est + b1 + b2) < 0.05, f"rel {rel(est + b1 + b2):.2e}")
    check("C. v32 + beta1 + beta2 + gamma == exact (5%)", rel(est + b1 + b2 + g) < 0.05,
          f"rel {rel(est + b1 + b2 + g):.2e}")
    report["checks"] = RESULTS
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
    print(f"\n{sum(r['ok'] for r in RESULTS)}/{len(RESULTS)} checks pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
