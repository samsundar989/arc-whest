"""Stage 1 (D2'): is v31's kappa4 -> kappa3 birth feed COMPLETE?  A label-free oracle.

Execution contract: `experiments/LOG.md`, 2026-09-16, *EXECUTION CONTRACT
AMENDMENT (pre-data)*, section "Changed D2/D3 discriminator".

THE QUESTION
------------
v31 routes the fourth cumulant into the new third-cumulant source only through
the term table's REPEATED slices `(3,)` and `(2,1)`. A source tensor can agree
on every repeated slice and still differ at all-distinct indices `(a,b,c)`, and
one dense transport mixes those entries back into the repeated slices the next
layer reads. So a repeated-slice test cannot settle completeness.

THE REFERENCE, AND WHY IT IS INDEPENDENT
----------------------------------------
Not 504aldo's code, and not v31's table. A Gaussian SCALE MIXTURE

    z = mu + s * L x,   x ~ N(0, I),   s^2 in {1 - d, 1 + d} equiprobable

has mean `mu`, covariance `Sigma = L L^T` exactly, kappa3 = 0, and

    kappa4_abcd = d^2 (S_ab S_cd + S_ac S_bd + S_ad S_bc).

With `Sigma = I + eps C` (unit variances) this is, to first order in `eps`,
exactly the fourth-cumulant MODEL the estimator carries,

    kappa4 = m Sym(G (x) I),   G = diag(g) + lam C_off,   m = 2,
    g = 1.5 d^2,   lam = 3 d^2,

where `Sym(G (x) I)_abcd = (1/6) sum over the 6 (pair|pair) splits`. Its
slices: `_iiii = g`, `_iijj = (g_i+g_j)/6`, `_iiij = G_ij/2`, `_iijk = G_jk/6`
-- i.e. `D4 = m dG` and `D22 = (m/6)(dG_i + dG_j)` as v31 consumes them.

The post-ReLU cumulants of the mixture are computed EXACTLY (to ~1e-13) by
nested Gauss-Legendre quadrature over the positive half-lines, where every
integrand is smooth. The part of the output third cumulant that is linear in
`d^2` and odd in `eps` is isolated by finite differences in both, so the
estimator's own Gaussian-closure truncation (even in eps) cancels.

Checks, printed with PASS/FAIL against thresholds fixed here before running:

  A. CONVENTION CONTROL -- mean `(1,)` and the `(2,1)` slice, response linear
     in d^2 at eps = 0 (kappa4_iiii and kappa4_iijj routes, which v31's table
     has): v31 must match exact. If this fails, the reference and the estimator
     disagree on conventions and nothing below is interpretable. (The odd-eps
     `(2,1)` response is printed for information only: it contains the `(3,1)`
     consumer v31 omits by contract.)
  B. THE DISCRIMINATOR -- all-distinct `(0,1,2)`: v31's d^2*eps response vs
     exact. A mismatch is missing distinct-index kappa4 content.
  C. DERIVED CORRECTION -- the connected first-order expansion has TWO routes
     into an all-distinct entry at order kappa4 * C:
       (i)  kappa4_ppxy touching all three:
            (1/2) sum_p kappa4_ppxy w2_p w1_x w1_y  = (m/12) lam sum_cyc w2 w1 C w1
       (ii) kappa4_aabb plus one covariance link to the third index:
            (1/4) kappa4_aabb [w2_a w3_b C_bc + w3_a w2_b C_ac] w1_c
     Reported separately; each is compensated (repeated content excluded).
  D. FULL-TENSOR STRUCTURE on a real tiny chain (f64, several lineages):
     reconstructing the tower's full tensor reproduces `_dslice3`/`_dslice21`
     after one dense transport; lam changes the repeated slices of the
     newborn source but leaves its all-distinct entries EXACTLY unchanged.
  E. TRANSPORT CONSEQUENCE: adding the compensated feed changes the consumed
     D3/D21 after one arbitrary dense transport (it is not a null edit).

Run:  uv run python local_sim/k4feed_oracle.py [--json out.json]
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]

import numpy as np  # noqa: E402
from scipy.special import ndtr  # noqa: E402

RESULTS: list[dict] = []


def check(name, ok, detail=""):
    RESULTS.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<66} {detail}")
    return ok


# --------------------------------------------------------------------------
# exact post-ReLU moments of a Gaussian, by quadrature on the positive axis
# --------------------------------------------------------------------------
_GL_X, _GL_W = np.polynomial.legendre.leggauss(160)


def _phi(x):
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def _e1(m, v):
    """E[relu(z)], z ~ N(m, v), closed form, vectorised."""
    s = np.sqrt(v)
    a = m / s
    return m * ndtr(a) + s * _phi(a)


def _half_line(m, v, upper_sd=14.0):
    """Gauss-Legendre nodes/weights for the density of N(m,v) on (0, m+14 sd)."""
    s = np.sqrt(v)
    hi = max(m + upper_sd * s, 1e-12)
    if hi <= 0.0:
        return np.zeros(1), np.zeros(1)
    t = 0.5 * hi * (_GL_X + 1.0)
    w = 0.5 * hi * _GL_W * _phi((t - m) / s) / s
    return t, w


def _moment(mu, S, powers):
    """E[prod_i relu(z_i)^p_i] for z ~ N(mu, S), up to 3 variables, p in {1,2,3}.

    Condition on the first variable (quadrature on z_0 > 0), then recurse on
    the conditional Gaussian of the rest. The one-variable leaf is closed-form
    for p = 1 and quadrature otherwise. Every integrand is smooth on (0, inf).
    """
    mu = np.asarray(mu, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    p0 = powers[0]
    if len(powers) == 1:
        if p0 == 1:
            return float(_e1(mu[0], S[0, 0]))
        t, w = _half_line(mu[0], S[0, 0])
        return float(np.sum(w * t ** p0))
    t, w = _half_line(mu[0], S[0, 0])
    c = S[1:, 0] / S[0, 0]
    Sc = S[1:, 1:] - np.outer(S[1:, 0], S[0, 1:]) / S[0, 0]
    vals = np.empty_like(t)
    for q, tq in enumerate(t):
        mc = mu[1:] + c * (tq - mu[0])
        vals[q] = _moment(mc, Sc, powers[1:])
    return float(np.sum(w * t ** p0 * vals))


def _mixture_cumulants(mu, Sigma, d):
    """Exact post-ReLU mean, kappa3(0,0,1) and kappa3(0,1,2) of the scale mixture."""
    scales = [1.0 - d, 1.0 + d]
    acc = {}
    for s2 in scales:
        S = Sigma * s2
        rows = {
            "0": _moment(mu[[0]], S[np.ix_([0], [0])], (1,)),
            "1": _moment(mu[[1]], S[np.ix_([1], [1])], (1,)),
            "2": _moment(mu[[2]], S[np.ix_([2], [2])], (1,)),
            "00": _moment(mu[[0]], S[np.ix_([0], [0])], (2,)),
            "01": _moment(mu[[0, 1]], S[np.ix_([0, 1], [0, 1])], (1, 1)),
            "02": _moment(mu[[0, 2]], S[np.ix_([0, 2], [0, 2])], (1, 1)),
            "12": _moment(mu[[1, 2]], S[np.ix_([1, 2], [1, 2])], (1, 1)),
            "001": _moment(mu[[0, 1]], S[np.ix_([0, 1], [0, 1])], (2, 1)),
            "012": _moment(mu, S, (1, 1, 1)),
        }
        for key, val in rows.items():
            acc[key] = acc.get(key, 0.0) + 0.5 * val
    m0, m1, m2 = acc["0"], acc["1"], acc["2"]
    k3_012 = (acc["012"] - acc["01"] * m2 - acc["02"] * m1 - acc["12"] * m0
              + 2.0 * m0 * m1 * m2)
    k3_001 = (acc["001"] - acc["00"] * m1 - 2.0 * acc["01"] * m0
              + 2.0 * m0 * m0 * m1)
    return {"mean0": m0, "k3_001": k3_001, "k3_012": k3_012}


# --------------------------------------------------------------------------
# the estimator, loaded unmodified, run in float64
# --------------------------------------------------------------------------
def _load(path):
    name = "cand_" + os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tower_tensor(tw, n):
    """Reconstruct the FULL (n,n,n) tensor a `_Tower` represents (numpy f64).

    Per column j of the wide layout, with P the carried mode (youngest = I):
      D block: (1/3) sum_{3 placements} A_D (x) P (x) P
      L block: (1/2) sum_{6 perms}      A_L (x) P (x) C_L,   C_L = A_L d(w2)
    and, when factored (P2), A_D = P d(sb) + 3 A_L d(eb) + Z L^T per lineage.
    These normalisations are the ones `_dslice3`/`_dslice21` imply; check D
    verifies that by transporting the tensor and comparing.
    """
    k = tw.k
    A_L = np.asarray(tw.A_L, dtype=np.float64)
    P = np.eye(n) if tw.M is None else np.concatenate(
        [np.asarray(tw.M, dtype=np.float64), np.eye(n)], axis=1)
    w2b = np.asarray(tw.w2b, dtype=np.float64)
    if tw.A_D is not None:
        A_D = np.asarray(tw.A_D, dtype=np.float64)
    else:
        sb = np.asarray(tw.sb, dtype=np.float64)
        eb = np.asarray(tw.eb, dtype=np.float64)
        Z = np.asarray(tw.Z, dtype=np.float64)
        L = np.asarray(tw.L, dtype=np.float64)
        r = L.shape[2]
        Zres = np.concatenate([Z[:, b * r:(b + 1) * r] @ L[b].T for b in range(k)], axis=1)
        A_D = P * sb[None, :] + 3.0 * A_L * eb[None, :] + Zres
    C_L = A_L * w2b[None, :]
    TD = np.einsum("aj,bj,cj->abc", A_D, P, P)
    T = (TD + TD.transpose(1, 0, 2) + TD.transpose(2, 1, 0)) / 3.0
    TL = np.einsum("aj,bj,cj->abc", A_L, P, C_L)
    T = T + 0.5 * sum(TL.transpose(pm) for pm in itertools.permutations(range(3)))
    if getattr(tw, "Y", None) is not None:
        # stage-1 hubs: sum_h sum_b Sym3(y_hb (x) (X P^T + P X^T)), X = A_L d(c_h)
        Y = np.asarray(tw.Y, dtype=np.float64).reshape(n, k, 2)
        for h, cvec in enumerate((tw.CA, tw.CB)):
            c = np.asarray(cvec, dtype=np.float64)
            for b in range(k):
                sl = slice(b * n, (b + 1) * n)
                X = A_L[:, sl] * c[None, sl]
                S = X @ P[:, sl].T + P[:, sl] @ X.T
                y = Y[:, b, h]
                T = T + (np.einsum("a,bc->abc", y, S) + np.einsum("b,ac->abc", y, S)
                         + np.einsum("c,ab->abc", y, S)) / 3.0
    return T


def slices(T):
    n = T.shape[0]
    idx = np.arange(n)
    D3 = T[idx, idx, idx]
    D21 = np.einsum("aai->ai", T).copy()
    D21[idx, idx] = 0.0
    return D3, D21


def distinct_mask(n):
    a, b, c = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    return (a != b) & (b != c) & (a != c)


def feed_tensor(lam, m, w1, w2, C_off):
    """The derived all-distinct feed, as a full symmetric tensor with its
    REPEATED content included: F = Sym3(S (x) y), S = lam d(w1) C_off d(w1),
    y = (m/4) w2, Sym3 = (1/3) sum over the 3 slots of y. Its all-distinct
    entries are (m/12) lam sum_cyc w2_c w1_a C_ab w1_b."""
    S = lam * (w1[:, None] * C_off * w1[None, :])
    y = 0.25 * m * w2
    F = (np.einsum("ab,c->abc", S, y) + np.einsum("ac,b->abc", S, y)
         + np.einsum("bc,a->abc", S, y)) / 3.0
    return F


def run_layer(mod, W, mu, cov, gk4, lam_prev, layer_index=1, tw=None):
    """One `_layer` of the loaded module in float64, with lam_prev patched in."""
    import flopscope as flops
    import flopscope.numpy as fnp

    dt = fnp.float64
    n = W.shape[0]
    table = mod._tables()
    est = mod.Estimator()
    const = est._constants(n, dt)
    old = list(mod._K4_LAM)
    lam_tab = list(old)
    lam_tab[layer_index - 1] = float(lam_prev)
    mod._K4_LAM = lam_tab
    try:
        with flops.BudgetContext(flop_budget=2 ** 50, quiet=True):
            state = (fnp.asarray(mu, dtype=dt), fnp.asarray(cov, dtype=dt), tw, 0.0,
                     fnp.asarray(gk4, dtype=dt))
            out = mod._layer(fnp.asarray(W, dtype=dt), state, table, dt, const, 2.0,
                             last_layer=False, gens=None, layer_index=layer_index)
    finally:
        mod._K4_LAM = old
    return out


# --------------------------------------------------------------------------
# A-C: the exact mixture reference
# --------------------------------------------------------------------------
def link_tensor(D22, w1, w2, w3, C_off):
    """Route (ii): kappa4_aabb linked to the third index by one covariance.

    First-order expansion, connected part at all-distinct (a,b,c):
        (1/4) kappa4_aabb [ w2_a w3_b C_bc w1_c + w3_a w2_b C_ac w1_c ]
    written as (1/4) sum over the 6 index permutations of
        Q_abc = D22_ab w2_a w3_b C_bc w1_c.
    Only its all-distinct entries are used (compensated)."""
    Q = np.einsum("ab,a,b,bc,c->abc", D22, w2, w3, C_off, w1)
    return 0.25 * sum(Q.transpose(pm) for pm in itertools.permutations(range(3)))


def t_mixture(mod, report):
    print("\nA-C. one birth from an exact non-Gaussian law (scale mixture), n = 3")
    rng = np.random.default_rng(20260916)
    n = 3
    mu = np.array([0.35, -0.2, 0.1])
    C = rng.normal(size=(n, n))
    C = (C + C.T) * 0.5
    np.fill_diagonal(C, 0.0)
    W = np.eye(n)
    m = 2.0
    keys = ("mean0", "k3_001", "k3_012")
    # unit variance: w1 = Phi(mu), w2 = phi(mu), w3 = -mu phi(mu)
    w1, w2, w3 = ndtr(mu), _phi(mu), -mu * _phi(mu)
    cache = {}

    def exact(d, eps):
        k = ("x", d, eps)
        if k not in cache:
            cache[k] = _mixture_cumulants(mu, np.eye(n) + eps * C, d)
        return cache[k]

    def estim(d, eps, routes=()):
        d2 = d * d
        cov = np.eye(n) + eps * C
        out = run_layer(mod, W, mu, cov, np.full(n, 1.5 * d2), 3.0 * d2)
        T = tower_tensor(out[2], n)
        if "i" in routes:
            T = T + _compensated(feed_tensor(3.0 * d2, m, w1, w2, eps * C))
        if "ii" in routes:
            D22 = (m / 6.0) * (np.full((n, n), 3.0 * d2)) * (1.0 - np.eye(n))
            T = T + _compensated(link_tensor(D22, w1, w2, w3, eps * C))
        return {"mean0": float(np.asarray(out[0], dtype=np.float64)[0]),
                "k3_001": float(T[0, 0, 1]), "k3_012": float(T[0, 1, 2])}

    def odd_resp(fn, d, eps, key):
        # linear in d^2, odd in eps: [f(d,e)-f(0,e)] - [f(d,-e)-f(0,-e)], / (2 d^2 e)
        a = fn(d, eps)[key] - fn(0.0, eps)[key]
        b = fn(d, -eps)[key] - fn(0.0, -eps)[key]
        return (a - b) / (2.0 * d * d * eps)

    def even0_resp(fn, d, key):
        # linear in d^2 at eps = 0
        return (fn(d, 0.0)[key] - fn(0.0, 0.0)[key]) / (d * d)

    arms = {"exact": exact,
            "v31": lambda a, b: estim(a, b),
            "v31+i": lambda a, b: estim(a, b, ("i",)),
            "v31+i+ii": lambda a, b: estim(a, b, ("i", "ii"))}
    rows = []
    for d, eps in ((0.04, 0.02), (0.02, 0.01)):
        row = {"d": d, "eps": eps, "even_eps0": {}, "odd_eps": {}}
        for name, fn in arms.items():
            row["even_eps0"][name] = {k: even0_resp(fn, d, k) for k in ("mean0", "k3_001")}
            row["odd_eps"][name] = {k: odd_resp(fn, d, eps, k) for k in keys}
        rows.append(row)
        print(f"    d={d} eps={eps}")
        print(f"      {'response':<22}" + "".join(f"{a:>14}" for a in arms))
        for k in ("mean0", "k3_001"):
            print(f"      {'d^2, eps=0: ' + k:<22}" + "".join(
                f"{row['even_eps0'][a][k]:>+14.5e}" for a in arms))
        for k in keys:
            print(f"      {'d^2*eps: ' + k:<22}" + "".join(
                f"{row['odd_eps'][a][k]:>+14.5e}" for a in arms))
    report["mixture"] = rows
    f = rows[-1]

    def rel(a, b):
        return abs(a - b) / max(abs(a), abs(b), 1e-300)

    pred_i = 0.5 * sum(w2[p] * w1[x] * w1[y] * C[x, y]
                       for p, x, y in ((0, 1, 2), (1, 0, 2), (2, 0, 1)))
    # Thresholds fixed before running: the finer finite-difference step must
    # agree to 5% (truncation O(d^2)+O(eps^2) is ~1e-3 relative here).
    for k in ("mean0", "k3_001"):
        check(f"A. convention control, eps=0 d^2 response of {k}: v31 == exact (5%)",
              rel(f["even_eps0"]["exact"][k], f["even_eps0"]["v31"][k]) < 0.05,
              f"exact {f['even_eps0']['exact'][k]:+.4e}  v31 {f['even_eps0']['v31'][k]:+.4e}")
    ex, v = f["odd_eps"]["exact"]["k3_012"], f["odd_eps"]["v31"]["k3_012"]
    check("B. DISCRIMINATOR: all-distinct d^2*eps response, v31 misses > 50% of it",
          abs(ex - v) > 0.5 * abs(ex), f"exact {ex:+.4e}  v31 {v:+.4e}")
    report["route_i_predicted"] = pred_i
    check("C. route (i) alone (m/12 lam feed) == exact all-distinct (5%)",
          rel(ex, f["odd_eps"]["v31+i"]["k3_012"]) < 0.05,
          f"v31+i {f['odd_eps']['v31+i']['k3_012']:+.4e}  (i) closed form {pred_i:+.4e}")
    check("C. routes (i)+(ii) == exact all-distinct (5%)",
          rel(ex, f["odd_eps"]["v31+i+ii"]["k3_012"]) < 0.05,
          f"v31+i+ii {f['odd_eps']['v31+i+ii']['k3_012']:+.4e}")
    for k in ("mean0", "k3_001"):
        check(f"C. adding the compensated routes leaves {k} unchanged (1e-12 abs)",
              abs(f["odd_eps"]["v31+i+ii"][k] - f["odd_eps"]["v31"][k]) < 1e-12
              and abs(f["even_eps0"]["v31+i+ii"][k] - f["even_eps0"]["v31"][k]) < 1e-12, "")
    print(f"      (info) odd-eps (2,1) response exact {f['odd_eps']['exact']['k3_001']:+.4e} "
          f"vs v31 {f['odd_eps']['v31']['k3_001']:+.4e}: includes the (3,1) consumer v31 omits by contract")


def _compensated(F):
    """Keep only F's all-distinct entries: the repeated content is subtracted
    from the residual at birth, so the table's repeated slices are unchanged."""
    return F * distinct_mask(F.shape[0])


# --------------------------------------------------------------------------
# D-E: full tensor on a real tiny chain
# --------------------------------------------------------------------------
def t_chain(mod, report, n=9, depth=5, seed=7):
    print(f"\nD-E. full tensor on a real float64 chain, n={n}, depth={depth}")
    import flopscope as flops
    import flopscope.numpy as fnp

    rng = np.random.default_rng(seed)
    Ws = [rng.normal(size=(n, n)) * np.sqrt(2.0 / n) for _ in range(depth + 1)]
    lam = 0.15
    state = (np.zeros(n), np.eye(n), None, 0.0, np.zeros(n))
    worst_recon = 0.0
    worst_lam_distinct = 0.0
    lam_repeated_move = 0.0
    feed_move = []
    for l in range(depth):
        mu, cov, tw, c4, gk4 = state
        out = run_layer(mod, Ws[l], np.asarray(mu, dtype=np.float64),
                        np.asarray(cov, dtype=np.float64), np.asarray(gk4, dtype=np.float64)
                        + (0.0 if l == 0 else 0.0), lam, layer_index=max(l, 1), tw=tw)
        out0 = run_layer(mod, Ws[l], np.asarray(mu, dtype=np.float64),
                         np.asarray(cov, dtype=np.float64), np.asarray(gk4, dtype=np.float64),
                         0.0, layer_index=max(l, 1), tw=tw)
        T = tower_tensor(out[2], n)
        T0 = tower_tensor(out0[2], n)
        dm = distinct_mask(n)
        scale = max(np.abs(T).max(), 1e-300)
        worst_lam_distinct = max(worst_lam_distinct, np.abs((T - T0) * dm).max() / scale)
        lam_repeated_move = max(lam_repeated_move, np.abs((T - T0) * ~dm).max() / scale)

        # one dense transport by the NEXT layer's weights: tensor vs the file's slices
        WT = Ws[l + 1].T
        Tt = np.einsum("ai,bj,ck,ijk->abc", WT, WT, WT, T)
        with flops.BudgetContext(flop_budget=2 ** 50, quiet=True):
            adv = mod._advance(fnp.asarray(WT, dtype=fnp.float64), out[2])
            off = fnp.ones((n, n), dtype=fnp.float64) - fnp.eye(n, dtype=fnp.float64)
            d3 = np.asarray(mod._dslice3(adv), dtype=np.float64)
            d21 = np.asarray(mod._dslice21(adv, off), dtype=np.float64)
        e3, e21 = slices(Tt)
        s = max(np.abs(e21).max(), np.abs(e3).max(), 1e-300)
        worst_recon = max(worst_recon, np.abs(d3 - e3).max() / s, np.abs(d21 - e21).max() / s)

        # E: the compensated feed at THIS birth, after one transport
        w_cov_pre = WT @ np.asarray(cov, dtype=np.float64) @ Ws[l].T.T if False else None
        cov_pre = Ws[l].T @ np.asarray(cov, dtype=np.float64) @ Ws[l]
        var = np.maximum(np.diag(cov_pre), 1e-12)
        sig = np.sqrt(var)
        alpha = (Ws[l].T @ np.asarray(mu, dtype=np.float64)) / sig
        w1 = ndtr(alpha)
        w2 = _phi(alpha) / sig
        C_off = cov_pre * (1.0 - np.eye(n))
        if l >= 1:
            F = _compensated(feed_tensor(lam, 2.0, w1, w2, C_off))
            Ft = np.einsum("ai,bj,ck,ijk->abc", WT, WT, WT, F)
            f3, f21 = slices(Ft)
            feed_move.append(float(max(np.abs(f3).max() / max(np.abs(e3).max(), 1e-300),
                                       np.abs(f21).max() / max(np.abs(e21).max(), 1e-300))))
        state = (np.asarray(out[0], dtype=np.float64), np.asarray(out[1], dtype=np.float64),
                 out[2], out[3], np.asarray(out[4], dtype=np.float64))
    report["chain"] = {"n": n, "depth": depth, "lam": lam,
                       "worst_reconstruction_rel": worst_recon,
                       "lam_distinct_move_rel": worst_lam_distinct,
                       "lam_repeated_move_rel": lam_repeated_move,
                       "feed_consumed_move_rel_per_layer": feed_move}
    check("D. reconstructed tensor, transported, == _dslice3/_dslice21 (1e-10)",
          worst_recon < 1e-10, f"worst {worst_recon:.2e}")
    check("D. lam moves the newborn source's REPEATED entries (> 1e-6)",
          lam_repeated_move > 1e-6, f"{lam_repeated_move:.2e}")
    check("D. lam leaves its ALL-DISTINCT entries unchanged (< 1e-12)",
          worst_lam_distinct < 1e-12, f"{worst_lam_distinct:.2e}")
    check("E. compensated feed moves consumed D3/D21 after one transport (> 1e-4)",
          len(feed_move) > 0 and max(feed_move) > 1e-4,
          "per layer " + ", ".join(f"{x:.2e}" for x in feed_move))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default=os.path.join(_ROOT, "estimator_k3_v31_dense.py"))
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    import flopscope as flops
    flops.configure(symmetry_warnings=False)
    mod = _load(args.candidate)
    report = {"candidate": os.path.basename(args.candidate)}
    t_mixture(mod, report)
    t_chain(mod, report)
    report["checks"] = RESULTS
    ok = all(r["ok"] for r in RESULTS)
    print(f"\n{sum(r['ok'] for r in RESULTS)}/{len(RESULTS)} checks pass")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
