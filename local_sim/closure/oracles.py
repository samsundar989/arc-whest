"""Oracles for the generated ReLU closure.  Tiny CPU work (about a minute).

Run from the repo root with the GPU venv (needs mpmath + scipy; torch optional,
used only for the upstream relu_wick_coef / term-set cross-checks):

    .venv-gpu\\Scripts\\python.exe local_sim/closure/oracles.py

Writes local_sim/closure/oracle_report.json and prints a PASS/FAIL table.

Numerics: exact tests use fractions.Fraction; the realizable-law references and the
closure evaluated on them use mpmath at 40 significant digits (float64 would put a
~1e-16 floor under errors that reach ~1e-18 at the smallest eps); the float32 study
compares float32 against a 60-digit mpmath reference.
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
from fractions import Fraction
from functools import lru_cache
from itertools import product
from pathlib import Path

import mpmath as mp
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))
import generate as G  # noqa: E402

UPSTREAM_SRC = r"C:\Users\samsu\Documents\mlp_cumulant_propagation\src"

RESULTS: list[dict] = []
DETAILS: dict = {}


def record(name, ok, detail, informational=False):
    status = "INFO" if informational else ("PASS" if ok else "FAIL")
    RESULTS.append({"check": name, "status": status, "detail": detail})
    print(f"[{status}] {name}: {detail}")


FORMS = {(f["k"], f["p"]): f for f in json.loads((HERE / "w_forms.json").read_text())["forms"]}
RULE_DOCS = {r.name: json.loads((HERE / f"closure_{r.name}.json").read_text()) for r in G.RULES}


def terms_of(rule_name, output):
    return [t for t in RULE_DOCS[rule_name]["terms"] if tuple(t["output"]) == tuple(output)]


def coef_value(s, kind):
    num, den = (int(x) for x in s.split("/"))
    if kind == "frac":
        return Fraction(num, den)
    if kind == "mp":
        return mp.mpf(num) / den
    return num / den


def eval_closure(terms, cum, wi, wj=None, kind="mp"):
    total = Fraction(0) if kind == "frac" else (mp.mpf(0) if kind == "mp" else 0.0)
    for t in terms:
        v = coef_value(t["coef"], kind)
        for f in t["factors"]:
            v = v * cum[f]
        out = t["output"]
        v = v * wi(t["derivs"][0], out[0])
        if len(out) == 2:
            v = v * wj(t["derivs"][1], out[1])
        total = total + v
    return total


# ============================================================================ w(k,p)
def poly_eval(coeffs, x):
    r = 0 * x
    for c in reversed(coeffs):
        r = r * x + c
    return r


def w_mp(k, p, mu, s):
    f = FORMS[(k, p)]
    sig = mp.sqrt(s)
    al = mu / sig
    phi = mp.npdf(al)
    Phi = mp.ncdf(al)
    return sig ** f["sigma_exp"] * (poly_eval([mp.mpf(c) for c in f["P_phi"]], al) * phi
                                    + poly_eval([mp.mpf(c) for c in f["P_Phi"]], al) * Phi)


def check_w_forms():
    mp.mp.dps = 30
    worst_quad, worst_diff = mp.mpf(0), mp.mpf(0)
    params = [(mp.mpf("0.3"), mp.mpf("1.7")), (mp.mpf("-1.1"), mp.mpf("0.6")), (mp.mpf("2.2"), mp.mpf("3.1"))]
    for (mu, s), p in product(params, range(1, 5)):
        sig = mp.sqrt(s)
        dens = lambda y, mu=mu, sig=sig: mp.npdf((y - mu) / sig) / sig
        for k in range(0, 9):
            got = w_mp(k, p, mu, s)
            if k <= p:
                ff = math.factorial(p) // math.factorial(p - k)
                ref = mp.quad(lambda y: ff * y ** (p - k) * dens(y), [0, mu + 12 * sig if mu > 0 else 12 * sig, mp.inf])
            else:
                m = k - p - 1
                ref = math.factorial(p) * (-1) ** m * mp.diff(dens, 0, m)
            worst_quad = max(worst_quad, abs(got - ref) / (1 + abs(ref)))
            if k < 8:  # distributional-derivative convention: w(k+1,p) = d/dmu w(k,p)
                d = mp.diff(lambda m_: w_mp(k, p, m_, s), mu)
                worst_diff = max(worst_diff, abs(w_mp(k + 1, p, mu, s) - d) / (1 + abs(d)))
    record("w(k,p) closed forms vs direct quadrature / density derivatives (k<=8,p<=4)",
           worst_quad < 1e-15, f"max rel err {mp.nstr(worst_quad, 3)}")
    record("w(k+1,p) = d/dmu w(k,p) (distributional derivative convention)",
           worst_diff < 1e-15, f"max rel err {mp.nstr(worst_diff, 3)}")
    # all coefficients integer
    ints = all(isinstance(c, int) for f in FORMS.values() for c in f["P_phi"] + f["P_Phi"])
    record("w forms have integer polynomial coefficients, sigma exponent p-k", ints and
           all(f["sigma_exp"] == f["p"] - f["k"] for f in FORMS.values()), f"{len(FORMS)} forms")
    # upstream
    try:
        sys.path.append(UPSTREAM_SRC)
        import torch
        from mlp_kprop.wick import relu_wick_coef
        worst = 0.0
        for (mu, s), p in product(params, range(1, 5)):
            for k in range(0, 9):
                up = float(relu_wick_coef(torch.tensor([float(mu)], dtype=torch.float64),
                                          torch.tensor([float(s)], dtype=torch.float64), k, p)[0])
                ours = float(w_mp(k, p, mu, s))
                worst = max(worst, abs(up - ours) / (1 + abs(ours)))
        record("w forms vs upstream mlp_kprop.wick.relu_wick_coef (float64)", worst < 1e-12, f"max rel err {worst:.2e}")
    except Exception as e:  # pragma: no cover
        record("w forms vs upstream relu_wick_coef", False, f"upstream unavailable: {e!r}")


# ============================================================================ conversion
def set_partitions(items):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for part in set_partitions(rest):
        for i in range(len(part)):
            yield part[:i] + [[first] + part[i]] + part[i + 1:]
        yield [[first]] + part


def joint_cumulant(law, idx):
    """law: list of (prob, (x, y)); idx: tuple of variable indices (0=X, 1=Y)."""
    def mom(block):
        return sum(pr * math.prod((pt[i] for i in block), start=Fraction(1)) for pr, pt in law)
    tot = Fraction(0)
    for part in set_partitions(list(idx)):
        n = len(part)
        tot += (-1) ** (n - 1) * math.factorial(n - 1) * math.prod((mom(b) for b in part), start=Fraction(1))
    return tot


def check_conversion():
    rng = random.Random(20260916)
    ok = True
    for trial in range(25):
        n = rng.randint(2, 6)
        ws = [rng.randint(1, 9) for _ in range(n)]
        law = [(Fraction(w, sum(ws)), (Fraction(rng.randint(0, 7), rng.randint(1, 3)),
                                       Fraction(rng.randint(0, 7), rng.randint(1, 3)))) for w in ws]
        E = lambda f: sum(pr * f(x, y) for pr, (x, y) in law)
        pK = {p: E(lambda x, y, p=p: x ** p) for p in range(1, 5)}
        pKj1 = E(lambda x, y: y)
        cov = lambda a, b: E(lambda x, y: x ** a * y ** b) - E(lambda x, y: x ** a) * E(lambda x, y: y ** b)
        p11, p21, p12, p22 = cov(1, 1), cov(2, 1), cov(1, 2), cov(2, 2)
        K = {
            "K1": pK[1], "K2": pK[2] - pK[1] ** 2, "K3": pK[3] - 3 * pK[1] * pK[2] + 2 * pK[1] ** 3,
            "K4": pK[4] - 4 * pK[1] * pK[3] - 3 * pK[2] ** 2 + 12 * pK[1] ** 2 * pK[2] - 6 * pK[1] ** 4,
            "K11": p11, "K21": p21 - 2 * pK[1] * p11, "K12": p12 - 2 * pKj1 * p11,
            "K22": p22 - 2 * pKj1 * p21 - 2 * pK[1] * p12 - 2 * p11 ** 2 + 4 * pK[1] * pKj1 * p11,
        }
        ref = {
            "K1": joint_cumulant(law, (0,)), "K2": joint_cumulant(law, (0, 0)), "K3": joint_cumulant(law, (0, 0, 0)),
            "K4": joint_cumulant(law, (0, 0, 0, 0)), "K11": joint_cumulant(law, (0, 1)),
            "K21": joint_cumulant(law, (0, 0, 1)), "K12": joint_cumulant(law, (0, 1, 1)),
            "K22": joint_cumulant(law, (0, 0, 1, 1)),
        }
        ok &= all(K[k] == ref[k] for k in K)
    record("pK -> K conversion equals joint cumulants (exact Fraction, 25 random discrete laws, 8 formulas)",
           ok, "exact equality" if ok else "MISMATCH")


# ============================================================================ bivariate series helpers (Fraction)
def ser_mul(a, b, cap):
    r = {}
    for (i1, j1), x in a.items():
        for (i2, j2), y in b.items():
            i, j = i1 + i2, j1 + j2
            if cap(i, j):
                r[(i, j)] = r.get((i, j), 0) + x * y
    return r


def moments_from_cumulants(kap, amax, bmax):
    """kap: {(r,s): value}. Returns E[y_i^r y_j^s] for r<=amax, s<=bmax via exp of the CGF series."""
    cap = lambda i, j: i <= amax and j <= bmax
    K = {rs: v / (math.factorial(rs[0]) * math.factorial(rs[1])) for rs, v in kap.items() if cap(*rs)}
    M = {(0, 0): Fraction(1)}
    term = {(0, 0): Fraction(1)}
    for n in range(1, amax + bmax + 1):
        term = ser_mul(term, K, cap)
        term = {k: v / n for k, v in term.items()}
        for k, v in term.items():
            M[k] = M.get(k, 0) + v
    return {k: v * math.factorial(k[0]) * math.factorial(k[1]) for k, v in M.items()}


def cumulants_from_moments(mom, deg):
    cap = lambda i, j: i + j <= deg
    X = {k: v / (math.factorial(k[0]) * math.factorial(k[1])) for k, v in mom.items() if k != (0, 0) and cap(*k)}
    L, P = {}, {(0, 0): Fraction(1)}
    for n in range(1, deg + 1):
        P = ser_mul(P, X, cap)
        for k, v in P.items():
            L[k] = L.get(k, 0) + Fraction((-1) ** (n + 1), n) * v
    return {k: v * math.factorial(k[0]) * math.factorial(k[1]) for k, v in L.items()}


# ============================================================================ (a) polynomial exactness
def gauss_moment(n, mu, s):
    return sum(math.comb(n, r) * mu ** (n - r) * s ** (r // 2) * math.prod(range(1, r, 2))
               for r in range(0, n + 1, 2))


def w_poly(mu, s):
    def w(k, p):
        if k > p:
            return Fraction(0)
        return Fraction(math.factorial(p), math.factorial(p - k)) * gauss_moment(p - k, mu, s)
    return w


def check_polynomial():
    F = Fraction
    mu_i, mu_j, s_i, s_j = F(3, 7), F(-5, 4), F(9, 5), F(2, 3)
    cum = {"C": F(-4, 9), "D21": F(3, 11), "D12": F(-7, 13), "D22": F(5, 17), "D3i": F(6, 5), "D3j": F(-2, 3),
           "D4i": F(8, 7), "D4j": F(3, 2)}
    kap = {(1, 0): mu_i, (0, 1): mu_j, (2, 0): s_i, (0, 2): s_j, (1, 1): cum["C"], (2, 1): cum["D21"],
           (1, 2): cum["D12"], (2, 2): cum["D22"], (3, 0): cum["D3i"], (0, 3): cum["D3j"], (4, 0): cum["D4i"],
           (0, 4): cum["D4j"]}
    wi, wj = w_poly(mu_i, s_i), w_poly(mu_j, s_j)
    mom = moments_from_cumulants(kap, 8, 8)
    rows, all_ok = [], True
    budgets = list(range(0, 9))
    for p in range(1, 9):
        exact = mom[(p, 0)]
        uni = {"D3": cum["D3i"], "D4": cum["D4i"]}
        terms = G.terms_for_output((p,), G.DerivCaps((p,)))
        val = eval_closure(terms, uni, wi, kind="frac")
        costs = [sum(G.block_cost(G.UNI_BLOCKS[f]) for f in t["factors"]) for t in terms]
        minB = next((B for B in budgets if eval_closure(G.terms_for_output((p,), G.Budget(B)), uni, wi, kind="frac") == exact), None)
        pair_ok = eval_closure(G.terms_for_output((p,), G.Pair()), uni, wi, kind="frac") == exact
        rows.append({"output": [p], "exact": val == exact, "n_terms_needed": len(terms), "max_cost_needed": max(costs),
                     "min_exact_budget": minB, "pair_exact": pair_ok})
        all_ok &= val == exact
    for a, b in product(range(1, 5), range(1, 5)):
        exact = mom[(a, b)] - mom[(a, 0)] * mom[(0, b)]
        terms = G.terms_for_output((a, b), G.DerivCaps((a, b)))
        val = eval_closure(terms, cum, wi, wj, kind="frac")
        costs = [sum(G.block_cost(G.BIV_BLOCKS[f]) for f in t["factors"]) for t in terms]
        worst = max(zip(costs, [t["factors"] for t in terms]))
        minB = next((B for B in budgets if eval_closure(G.terms_for_output((a, b), G.Budget(B)), cum, wi, wj, kind="frac") == exact), None)
        pair_ok = eval_closure(G.terms_for_output((a, b), G.Pair()), cum, wi, wj, kind="frac") == exact
        rows.append({"output": [a, b], "exact": val == exact, "n_terms_needed": len(terms), "max_cost_needed": max(costs),
                     "max_cost_example": worst[1], "min_exact_budget": minB, "pair_exact": pair_ok})
        all_ok &= val == exact
    DETAILS["polynomial_exactness"] = rows
    record("(a) polynomial exactness: untruncated connected operator == exact moments/covariances "
           "(Fraction; p<=8 univariate, (a,b)<=(4,4))", all_ok,
           f"{sum(r['exact'] for r in rows)}/{len(rows)} outputs exact")
    lines = "; ".join(f"{tuple(r['output'])}: minB={r['min_exact_budget']} pair={'Y' if r['pair_exact'] else 'N'}"
                      for r in rows if r["output"] in ([1], [2], [3], [4], [6], [8], [1, 1], [2, 1], [1, 2], [2, 2], [3, 3], [4, 4]))
    record("(a) truncation forced by polynomial degree (minimal exact BUDGET; PAIR exact?)", True, lines, informational=True)
    # the shipped outputs must be exact for polynomials of their own degree at budget 2
    shipped_ok = all(r["min_exact_budget"] is not None and r["min_exact_budget"] <= 2
                     for r in rows if r["output"] in ([1], [2], [3], [4], [1, 1], [2, 1], [1, 2], [2, 2]))
    record("(a) shipped outputs (p<=4, (a,b) in {11,21,12,22}) exact for polynomial F at BUDGET 2", shipped_ok, "")


# ============================================================================ upstream term sets
def check_upstream_terms():
    try:
        sys.path.append(UPSTREAM_SRC)
        import logging
        logging.disable(logging.WARNING)
        from mlp_kprop.kprop_harmonic import get_all_terms
        from mlp_kprop.partitions import vec_part_coef
    except Exception as e:  # pragma: no cover
        record("term sets vs upstream get_all_terms", False, f"upstream unavailable: {e!r}")
        return
    inv2 = {v: k for k, v in G.BIV_BLOCKS.items()}
    inv1 = {v: k for k, v in G.UNI_BLOCKS.items()}
    ok, msgs = True, []
    for B in (2, 3, 4):
        T = get_all_terms(B + 1)
        for out, inv in (((1,), inv1), ((1, 1), inv2)):
            ups = {}
            for ip, vp in T:
                if ip != out:
                    continue
                if all(v in inv for v in vp):
                    ups[tuple(sorted(inv[v] for v in vp))] = vec_part_coef(vp)
            ours = {tuple(t["factors"]): float(Fraction(t["coef"])) for t in terms_of(f"budget{B}", out)}
            same_set = set(ups) == set(ours)
            same_coef = same_set and all(abs(ups[k] - ours[k]) < 1e-15 for k in ours)
            ok &= same_coef
            msgs.append(f"B{B} {out}: ours={len(ours)} upstream(filtered)={len(ups)} {'match' if same_coef else 'DIFF'}")
    record("term sets + coefficients vs upstream get_all_terms(k_max=B+1)/vec_part_coef, filtered to available blocks",
           ok, "; ".join(msgs))


# ============================================================================ (c) invariances
def check_invariance():
    ok = True
    for rname in RULE_DOCS:
        for a, b in G.BIV_OUTPUTS:
            src = {(tuple(t["factors"]), tuple(t["derivs"]), t["coef"]) for t in terms_of(rname, (a, b))}
            mapped = {(tuple(sorted(G.SWAP[f] for f in fs)), (d[1], d[0]), c) for fs, d, c in src}
            dst = {(tuple(t["factors"]), tuple(t["derivs"]), t["coef"]) for t in terms_of(rname, (b, a))}
            ok &= mapped == dst
    record("(c) i<->j swap maps (a,b) term table onto (b,a) exactly, all rules; (1,1),(2,2) self-symmetric", ok, "")
    # numeric: evaluate with swapped inputs
    mp.mp.dps = 30
    rnd = random.Random(7)
    cum = {n: mp.mpf(rnd.uniform(-0.5, 0.5)) for n in G.BIV_BLOCKS}
    mu_i, mu_j, s_i, s_j = mp.mpf("0.4"), mp.mpf("-0.3"), mp.mpf("1.3"), mp.mpf("0.7")
    wi = lambda k, p: w_mp(k, p, mu_i, s_i)
    wj = lambda k, p: w_mp(k, p, mu_j, s_j)
    swc = {G.SWAP[n]: v for n, v in cum.items()}
    worst = mp.mpf(0)
    for rname in RULE_DOCS:
        for a, b in G.BIV_OUTPUTS:
            v1 = eval_closure(terms_of(rname, (a, b)), cum, wi, wj)
            v2 = eval_closure(terms_of(rname, (b, a)), swc, wj, wi)
            worst = max(worst, abs(v1 - v2))
    record("(c) numeric swap invariance pK(a,b)[i,j] == pK(b,a)[j,i]", worst < 1e-25, f"max abs diff {mp.nstr(worst, 3)}")


# ============================================================================ (b) realizable law
def fpoly_mul(a, b):
    r = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            r[i + j] += x * y
    return r


def fpoly_pow(a, n):
    r = [Fraction(1)]
    for _ in range(n):
        r = fpoly_mul(r, a)
    return r


def q_coeffs(mu, sig, a, b):
    """mu + sig*(g + a(g^2-1) + b(g^3-3g)), ascending."""
    return [mu - sig * a, sig * (1 - 3 * b), sig * a, sig * b]


def std_moment(n):
    return 0 if n % 2 else math.prod(range(1, n, 2))


class Law:
    def __init__(self, mu_i, sig_i, a_i, b_i, mu_j, sig_j, a_j, b_j, c):
        self.qi = q_coeffs(mu_i, sig_i, a_i, b_i)
        self.qj = q_coeffs(mu_j, sig_j, a_j, b_j)
        self.c = c
        self._cum = None

    @lru_cache(maxsize=None)
    def gm(self, m, n):  # E[g1^m u^n], u = c g1 + sqrt(1-c^2) g2
        c = self.c
        return sum(math.comb(n, k) * c ** k * (1 - c * c) ** ((n - k) // 2) * std_moment(m + k) * std_moment(n - k)
                   for k in range(n + 1) if (n - k) % 2 == 0)

    def cumulants(self, deg=6):
        if self._cum is None:
            mom = {}
            pis = [fpoly_pow(self.qi, r) for r in range(deg + 1)]
            pjs = [fpoly_pow(self.qj, s) for s in range(deg + 1)]
            for r in range(deg + 1):
                for s in range(deg + 1 - r):
                    mom[(r, s)] = sum(x * y * self.gm(m, n) for m, x in enumerate(pis[r]) for n, y in enumerate(pjs[s]) if x and y)
            self._cum = cumulants_from_moments(mom, deg)
        return self._cum

    # -------- exact ReLU expectations (mpmath)
    @staticmethod
    def pos_intervals(coeffs):
        cs = [mp.mpf(x.numerator) / x.denominator if isinstance(x, Fraction) else mp.mpf(x) for x in coeffs]
        while len(cs) > 1 and cs[-1] == 0:
            cs.pop()
        val = lambda x: poly_eval(cs, x)
        if len(cs) == 1:
            return [(-mp.inf, mp.inf)] if cs[0] > 0 else []
        roots = mp.polyroots(list(reversed(cs)), maxsteps=400, extraprec=400, error=False)
        if not isinstance(roots, (list, tuple)):
            roots = [roots]
        real = sorted(mp.re(r) for r in roots if abs(mp.im(r)) < mp.mpf(10) ** (-mp.mp.dps + 8) * (1 + abs(r)))
        pts = [-mp.inf] + real + [mp.inf]
        out = []
        for lo, hi in zip(pts[:-1], pts[1:]):
            if lo == -mp.inf and hi == mp.inf:
                mid = 0
            elif lo == -mp.inf:
                mid = hi - 1
            elif hi == mp.inf:
                mid = lo + 1
            else:
                mid = (lo + hi) / 2
            if val(mid) > 0:
                out.append((lo, hi))
        return out

    @staticmethod
    def gauss_poly_int(P, lo, hi):
        """int_lo^hi P(z) phi(z) dz, P ascending mp coefficients."""
        def edge(x, r):
            if x in (mp.inf, -mp.inf):
                return mp.mpf(0)
            return (x ** (r - 1) if r >= 1 else 0) * mp.npdf(x)
        Phi = lambda x: mp.mpf(1) if x == mp.inf else (mp.mpf(0) if x == -mp.inf else mp.ncdf(x))
        J = [Phi(hi) - Phi(lo)]
        if len(P) > 1:
            J.append((0 if lo == -mp.inf else mp.npdf(lo)) - (0 if hi == mp.inf else mp.npdf(hi)))
        for r in range(2, len(P)):
            J.append((r - 1) * J[r - 2] + edge(lo, r) - edge(hi, r))
        return sum(c * J[r] for r, c in enumerate(P))

    @staticmethod
    def compose_shift(coeffs, m, sd):
        """coefficients of q(m + sd z) in z (mp)."""
        cs = [mp.mpf(x.numerator) / x.denominator for x in coeffs]
        out = [mp.mpf(0)] * len(cs)
        for n, c in enumerate(cs):
            for r in range(n + 1):
                out[r] += c * math.comb(n, r) * m ** (n - r) * sd ** r
        return out

    @staticmethod
    def mp_pow(P, n):
        r = [mp.mpf(1)]
        for _ in range(n):
            s = [mp.mpf(0)] * (len(r) + len(P) - 1)
            for i, x in enumerate(r):
                for j, y in enumerate(P):
                    s[i + j] += x * y
            r = s
        return r

    def E_uni(self, which, p, sign=1):
        q = [sign * x for x in (self.qi if which == "i" else self.qj)]
        P = self.mp_pow([mp.mpf(x.numerator) / x.denominator for x in q], p)
        return sum(self.gauss_poly_int(P, lo, hi) for lo, hi in self.pos_intervals(q))

    def E_biv(self, a, b, si=1, sj=1, method="tanh-sinh"):
        qi = [si * x for x in self.qi]
        qj = [sj * x for x in self.qj]
        c = mp.mpf(self.c.numerator) / self.c.denominator
        sd = mp.sqrt(1 - c * c)
        u_int = self.pos_intervals(qj)
        qi_mp = [mp.mpf(x.numerator) / x.denominator for x in qi]

        def h(g1):
            m = c * g1
            P = self.mp_pow(self.compose_shift(qj, m, sd), b)
            return sum(self.gauss_poly_int(P, (lo - m) / sd if lo != -mp.inf else -mp.inf,
                                           (hi - m) / sd if hi != mp.inf else mp.inf) for lo, hi in u_int)

        f = lambda g: poly_eval(qi_mp, g) ** a * h(g) * mp.npdf(g)
        tot = mp.mpf(0)
        for lo, hi in self.pos_intervals(qi):
            # split infinite ranges at a finite point so the quadrature sees the Gaussian scale
            pts = [lo, hi]
            if lo == -mp.inf and hi == mp.inf:
                pts = [-mp.inf, 0, mp.inf]
            tot += mp.quad(f, pts, method=method)
        return tot


def closure_on_law(law, rname, outputs_uni=(1,), outputs_biv=((1, 1),)):
    k = law.cumulants()
    toM = lambda x: mp.mpf(x.numerator) / x.denominator
    mu_i, mu_j, s_i, s_j = toM(k[(1, 0)]), toM(k[(0, 1)]), toM(k[(2, 0)]), toM(k[(0, 2)])
    cum = {"C": k[(1, 1)], "D21": k[(2, 1)], "D12": k[(1, 2)], "D22": k[(2, 2)], "D3i": k[(3, 0)], "D3j": k[(0, 3)],
           "D4i": k[(4, 0)], "D4j": k[(0, 4)]}
    cum = {n: toM(v) for n, v in cum.items()}
    cum["D3"], cum["D4"] = cum["D3i"], cum["D4i"]
    wi = lambda kk, p: w_mp(kk, p, mu_i, s_i)
    wj = lambda kk, p: w_mp(kk, p, mu_j, s_j)
    res = {}
    for p in outputs_uni:
        res[(p,)] = eval_closure(terms_of(rname, (p,)), cum, wi)
    for ab in outputs_biv:
        res[ab] = eval_closure(terms_of(rname, ab), cum, wi, wj)
    return res


def exact_on_law(law, outputs_uni=(1,), outputs_biv=((1, 1),)):
    res = {}
    for p in outputs_uni:
        res[(p,)] = law.E_uni("i", p)
    for a, b in outputs_biv:
        res[(a, b)] = law.E_biv(a, b) - law.E_uni("i", a) * law.E_uni("j", b)
    return res


def full_block_closure(law, B, out):
    """Oracle-only: the closure with EVERY non-base cumulant block of cost <= B (as upstream
    get_vec_cond does), including blocks the generated tables treat as unavailable
    ((3,1), (1,3), order 5-6).  Used to attribute the error plateaus of the generated rules."""
    k = law.cumulants()
    toM = lambda x: mp.mpf(x.numerator) / x.denominator
    mu_i, mu_j, s_i, s_j = toM(k[(1, 0)]), toM(k[(0, 1)]), toM(k[(2, 0)]), toM(k[(0, 2)])
    if len(out) == 1:
        blocks = {f"k{r}": (r,) for r in range(3, 7) if G.block_cost((r,)) <= B}
        val = {f"k{r}": toM(k[(r, 0)]) for r in range(3, 7)}
    else:
        blocks = {f"k{r}_{t}": (r, t) for r in range(0, 7) for t in range(0, 7)
                  if 2 <= r + t <= 6 and (r, t) not in ((2, 0), (0, 2)) and G.block_cost((r, t)) <= B}
        val = {f"k{r}_{t}": toM(k[(r, t)]) for (r, t) in blocks.values()}
    monos = G.enumerate_monomials(blocks, G.Budget(B))
    if len(out) == 2:
        monos = [m for m in monos if any(min(blocks[n]) > 0 for n in m)]
    tot = mp.mpf(0)
    for m in monos:
        d = G.derivs(m, blocks, len(out))
        c = G.coef_of(m, blocks)
        v = mp.mpf(c.numerator) / c.denominator
        for n in m:
            v *= val[n]
        v *= w_mp(d[0], out[0], mu_i, s_i)
        if len(out) == 2:
            v *= w_mp(d[1], out[1], mu_j, s_j)
        tot += v
    return tot


def slope(eps, errs, last=4):
    x = np.log(np.array([float(e) for e in eps[-last:]]))
    y = np.log(np.array([float(e) for e in errs[-last:]]))
    return float(np.polyfit(x, y, 1)[0])


def fam_A(eps):  # skew-dominated, fixed O(1) correlation
    F = Fraction
    return Law(F(3, 10), F(6, 5), eps, F(0), F(-2, 5), F(4, 5), -eps / 2, F(0), F(3, 10))


def fam_B(eps):  # power-counting-like: C ~ eps, higher cumulants ~ eps^2 or smaller
    F = Fraction
    return Law(F(3, 10), F(6, 5), eps * eps, eps * eps / 2, F(-2, 5), F(4, 5), -F(3, 4) * eps * eps, eps * eps / 3,
               F(4, 5) * eps)


def fam_C():  # NOT small non-Gaussianity
    F = Fraction
    return Law(F(3, 10), F(6, 5), F(7, 20), F(1, 10), F(-2, 5), F(4, 5), F(-3, 10), F(1, 20), F(3, 5))


def check_realizable():
    mp.mp.dps = 40
    rules = list(RULE_DOCS)
    # --- quadrature verification on the large fixture
    law = fam_C()
    e_ts = law.E_biv(1, 1)
    e_gl = law.E_biv(1, 1, method="gauss-legendre")
    # identity E[y_i y_j] = sum_{si,sj} si*sj*E[ReLU(si y_i) ReLU(sj y_j)]
    k = law.cumulants()
    Eyy = mp.mpf((k[(1, 1)] + k[(1, 0)] * k[(0, 1)]).numerator) / (k[(1, 1)] + k[(1, 0)] * k[(0, 1)]).denominator
    ident = sum(si * sj * law.E_biv(1, 1, si, sj) for si in (1, -1) for sj in (1, -1))
    Ey = mp.mpf(k[(1, 0)].numerator) / k[(1, 0)].denominator
    ident_u = law.E_uni("i", 1) - law.E_uni("i", 1, sign=-1)
    # naive float64 Gauss-Hermite 2D product rule on the raw integrand
    gh = {}
    for n in (20, 80, 320):
        x, w = np.polynomial.hermite_e.hermegauss(n)
        w = w / w.sum()
        G1, G2 = np.meshgrid(x, x, indexing="ij")
        W = np.outer(w, w)
        c = float(law.c)
        U = c * G1 + math.sqrt(1 - c * c) * G2
        yi = np.polynomial.polynomial.polyval(G1, [float(v) for v in law.qi])
        yj = np.polynomial.polynomial.polyval(U, [float(v) for v in law.qj])
        Ri, Rj = np.maximum(yi, 0), np.maximum(yj, 0)
        gh[n] = float((W * Ri * Rj).sum() - (W * Ri).sum() * (W * Rj).sum())
    ref = float(e_ts - law.E_uni("i", 1) * law.E_uni("j", 1))
    ok_q = abs(e_ts - e_gl) < 1e-25 and abs(ident - Eyy) < 1e-25 and abs(ident_u - Ey) < 1e-30
    record("(b) exact-reference quadrature: tanh-sinh vs Gauss-Legendre, E[y_i y_j] and E[y] sign identities",
           ok_q, f"|TS-GL|={mp.nstr(abs(e_ts - e_gl), 3)} |ident2-E[yiyj]|={mp.nstr(abs(ident - Eyy), 3)} "
                 f"|ident1-E[y]|={mp.nstr(abs(ident_u - Ey), 3)}")
    record("(b) naive float64 2-D Gauss-Hermite converges to the semi-analytic reference (kink => slow)",
           abs(gh[320] - ref) < abs(gh[20] - ref) and abs(gh[320] - ref) < 1e-4,
           "; ".join(f"n={n}: err {abs(v - ref):.2e}" for n, v in gh.items()))

    fam_out = {}
    eps_list = [Fraction(1, 10 * 2 ** n) for n in range(6)]
    for fname, fam in (("A_skew_fixed_corr", fam_A), ("B_powercount_corr_eps", fam_B)):
        errs = {r: {"1": [], "1,1": []} for r in rules}
        ignored = {"k31": [], "k50": [], "k(3,0)": [], "k(1,1)": []}
        for eps in eps_list:
            law = fam(eps)
            ex = exact_on_law(law)
            k = law.cumulants()
            ignored["k31"].append(float(k[(3, 1)]))
            ignored["k50"].append(float(k[(5, 0)]))
            ignored["k(3,0)"].append(float(k[(3, 0)]))
            ignored["k(1,1)"].append(float(k[(1, 1)]))
            for r in rules:
                cl = closure_on_law(law, r)
                errs[r]["1"].append(abs(cl[(1,)] - ex[(1,)]))
                errs[r]["1,1"].append(abs(cl[(1, 1)] - ex[(1, 1)]))
            if fname.startswith("B"):
                for B in (2, 3, 4):
                    fe = errs.setdefault(f"FULLBLOCKS_budget{B}", {"1": [], "1,1": []})
                    fe["1"].append(abs(full_block_closure(law, B, (1,)) - ex[(1,)]))
                    fe["1,1"].append(abs(full_block_closure(law, B, (1, 1)) - ex[(1, 1)]))
        slopes = {r: {o: slope(eps_list, errs[r][o]) for o in ("1", "1,1")} for r in errs}
        fam_out[fname] = {
            "eps": [str(e) for e in eps_list],
            "abs_err": {r: {o: [mp.nstr(v, 6) for v in errs[r][o]] for o in errs[r]} for r in errs},
            "slope_last4": slopes,
            "cumulant_scaling": ignored,
            "cumulant_scaling_slopes": {kk: slope(eps_list, [abs(v) for v in vv]) for kk, vv in ignored.items()},
        }
        record(f"(b) family {fname}: log-log slopes |closure-exact| (mean pK(1) / pK(1,1))", True,
               "; ".join(f"{r}: {slopes[r]['1']:.2f} / {slopes[r]['1,1']:.2f}" for r in slopes)
               + " | cumulant slopes " + ", ".join(f"{kk}={v:.2f}" for kk, v in fam_out[fname]["cumulant_scaling_slopes"].items()),
               informational=True)
    sA = fam_out["A_skew_fixed_corr"]["slope_last4"]
    record("(b) BUDGET 4 improves the asymptotic order of the mean over BUDGET 2 (family A, gain >= 0.8)",
           sA["budget4"]["1"] - sA["budget2"]["1"] >= 0.8,
           f"budget2 {sA['budget2']['1']:.2f} -> budget4 {sA['budget4']['1']:.2f}")
    # family A at eps = 0 (exactly Gaussian, fixed correlation): the C-power-series truncation floor
    law0 = fam_A(Fraction(0))
    ex0 = exact_on_law(law0)
    floor = {r: float(abs(closure_on_law(law0, r)[(1, 1)] - ex0[(1, 1)])) for r in rules}
    fam_out["A_skew_fixed_corr"]["pK11_abs_err_at_eps0"] = floor
    record("(b) family A pK(1,1) at eps=0 (Gaussian, corr fixed): C-series truncation floor, so its eps-slope is not an order",
           True, ", ".join(f"{r}: {v:.2e}" for r, v in floor.items()), informational=True)
    sB = fam_out["B_powercount_corr_eps"]["slope_last4"]
    csB = fam_out["B_powercount_corr_eps"]["cumulant_scaling_slopes"]
    # First-run prediction (written before any data): generated-rule pK(1,1) order rises with BUDGET.
    record("(b) [first-run prediction; refuted] family B generated-table pK(1,1) order increases BUDGET 2 < 3 < 4",
           True, f"{sB['budget2']['1,1']:.2f} < {sB['budget3']['1,1']:.2f} < {sB['budget4']['1,1']:.2f} : "
                 + ("holds" if (sB["budget2"]["1,1"] + 0.5 < sB["budget3"]["1,1"] and sB["budget3"]["1,1"] + 0.5 < sB["budget4"]["1,1"]) else "REFUTED"),
           informational=True)
    # Attribution (hypothesis formed after the first run, tested by the full-block closure, not run before):
    # the plateau is the first UNAVAILABLE cumulant: kappa_(3,1) (cost 3) for pK(1,1), kappa_5 (cost 4) for the mean.
    fb = [sB[f"FULLBLOCKS_budget{B}"]["1,1"] for B in (2, 3, 4)]
    ok_full = fb[0] + 0.7 < fb[1] and fb[1] + 0.7 < fb[2] and sB["FULLBLOCKS_budget4"]["1"] >= sB["budget4"]["1"] + 0.8
    ok_plateau = (abs(sB["budget3"]["1,1"] - csB["k31"]) < 0.3 and abs(sB["budget4"]["1,1"] - csB["k31"]) < 0.3
                  and abs(sB["budget4"]["1"] - csB["k50"]) < 0.3)
    record("(b) family B attribution: with ALL blocks of cost<=B the order rises (pK(1,1) and B4 mean); generated tables "
           "plateau at the scaling of kappa_(3,1) (pK(1,1)) and kappa_5 (mean)", ok_full and ok_plateau,
           f"full-block pK(1,1) slopes {fb[0]:.2f}/{fb[1]:.2f}/{fb[2]:.2f}, full-block mean B2/B4 "
           f"{sB['FULLBLOCKS_budget2']['1']:.2f}/{sB['FULLBLOCKS_budget4']['1']:.2f}; "
           f"generated B3/B4 pK(1,1) {sB['budget3']['1,1']:.2f}/{sB['budget4']['1,1']:.2f} vs k31 slope {csB['k31']:.2f}; "
           f"generated B4 mean {sB['budget4']['1']:.2f} vs k5 slope {csB['k50']:.2f} (criterion written after first run)")

    # --- large non-Gaussianity fixture: all outputs
    law = fam_C()
    ub, bb = (1, 2, 3, 4), ((1, 1), (2, 1), (1, 2), (2, 2))
    ex = exact_on_law(law, ub, bb)
    k = law.cumulants()
    big = {"exact": {",".join(map(str, o)): mp.nstr(v, 10) for o, v in ex.items()},
           "standardised_cumulants_i": {f"k{r}": float(k[(r, 0)]) / float(k[(2, 0)]) ** (r / 2) for r in (3, 4, 5, 6)},
           "ignored_k31": float(k[(3, 1)]), "ignored_k13": float(k[(1, 3)]), "corr": float(k[(1, 1)]) / math.sqrt(float(k[(2, 0)] * k[(0, 2)])),
           "rel_err": {}}
    for r in rules:
        cl = closure_on_law(law, r, ub, bb)
        big["rel_err"][r] = {",".join(map(str, o)): float(abs(cl[o] - ex[o]) / abs(ex[o])) for o in ex}
    fam_out["C_large_nongaussian"] = big
    record("(b) family C (large non-Gaussianity, skew_i=%.2f, exkurt_i=%.2f, corr=%.2f): relative errors"
           % (big["standardised_cumulants_i"]["k3"], big["standardised_cumulants_i"]["k4"], big["corr"]), True,
           " | ".join(f"{r}: " + ", ".join(f"{o}:{v:.1e}" for o, v in big["rel_err"][r].items()) for r in rules),
           informational=True)
    DETAILS["realizable"] = fam_out


# ============================================================================ (d) float32 stability
def check_float32():
    from scipy.special import erfc
    mp.mp.dps = 60
    sigmas = [10.0 ** e for e in np.arange(-8, 4.5, 0.5)]
    alphas = np.linspace(-40, 40, 81)
    f32 = np.float32
    eps32 = float(np.finfo(np.float32).eps)

    def W32(k, p, al):
        f = FORMS[(k, p)]
        al = al.astype(f32)
        phi = (np.exp(-al * al / f32(2)) / f32(math.sqrt(2 * math.pi))).astype(f32)
        Phi = (f32(0.5) * erfc((-al / f32(math.sqrt(2))).astype(f32))).astype(f32)
        P1 = poly_eval([f32(c) for c in f["P_phi"]], al) if f["P_phi"] else np.zeros_like(al)
        P2 = poly_eval([f32(c) for c in f["P_Phi"]], al) if f["P_Phi"] else np.zeros_like(al)
        bound = poly_eval([f32(abs(c)) for c in f["P_phi"]], np.abs(al)) * phi if f["P_phi"] else np.zeros_like(al)
        bound = bound + (poly_eval([f32(abs(c)) for c in f["P_Phi"]], np.abs(al)) * Phi if f["P_Phi"] else 0)
        return (P1 * phi + P2 * Phi).astype(f32), bound

    @lru_cache(maxsize=None)
    def Wref(k, p, al):
        f = FORMS[(k, p)]
        a = mp.mpf(al)
        return poly_eval([mp.mpf(c) for c in f["P_phi"]], a) * mp.npdf(a) + poly_eval([mp.mpf(c) for c in f["P_Phi"]], a) * mp.ncdf(a)

    # 1) normalised W in float32: finite everywhere, error within a rounding bound
    worst_ratio, nonfinite, worst_rel_mid, worst_ratio_sig, worst_scaled = 0.0, 0, 0.0, 0.0, 0.0
    cancel_region = {}
    for p in range(1, 5):
        for k in range(0, 9):
            w32, bnd = W32(k, p, alphas)
            ref = np.array([float(Wref(k, p, float(a))) for a in alphas])
            nonfinite += int((~np.isfinite(w32)).sum())
            err = np.abs(w32.astype(np.float64) - ref)
            ratio = err / (64 * eps32 * np.maximum(bnd.astype(np.float64), 1e-300) + 1e-38)
            worst_ratio = max(worst_ratio, float(ratio.max()))
            sig_mask = np.abs(ref) > 1e-25
            worst_ratio_sig = max(worst_ratio_sig, float(ratio[sig_mask].max()) if sig_mask.any() else 0.0)
            scale = np.maximum(1.0, np.abs(ref))
            worst_scaled = max(worst_scaled, float((err / scale).max()))
            rel = err / np.maximum(np.abs(ref), 1e-300)
            bad = alphas[(rel > 1e-3) & (np.abs(ref) > 0)]
            if bad.size:
                cancel_region[f"({k},{p})"] = [float(bad.min()), float(bad.max())]
            mid = np.abs(alphas) <= 4
            worst_rel_mid = max(worst_rel_mid, float(rel[mid & (np.abs(ref) > 1e-30)].max()))
    DETAILS.setdefault("float32", {})["W_rel_err_gt_1e-3_alpha_ranges"] = cancel_region
    record("(d) normalised W(k,p)=sigma^(k-p) w in float32 on |alpha|<=40: finite and mixed abs/rel err "
           "|W32-Wref|/max(1,|Wref|) <= 1e-5 (third criterion; earlier two were miscalibrated, see detail row)",
           nonfinite == 0 and worst_scaled <= 1e-5,
           f"nonfinite={nonfinite}, worst |err|/max(1,|W|)={worst_scaled:.1e}")
    record("(d) float32 W relative accuracy (first-run criterion |err|<=64*eps32*(|P_phi|phi+|P_Phi|Phi) failed; see detail)",
           True,
           f"worst err/(64 eps32 bound)={worst_ratio:.2f} overall, {worst_ratio_sig:.2f} where |W|>1e-25; "
           f"worst rel err for |alpha|<=4: {worst_rel_mid:.1e} (subtractive cancellation, k<p at alpha=-4 where |W|~1e-6); "
           f"(k,p) pairs with rel err>1e-3 somewhere on the grid: {len(cancel_region)}", informational=True)

    # 2) raw vs normalised term products over sigma, using standardised cumulants rho ~ O(1)
    rng = np.random.default_rng(3)
    raw_bad, norm_bad = {}, {}
    rho = {n: float(rng.uniform(0.2, 0.9)) * (1 if rng.random() < 0.5 else -1) for n in G.BIV_BLOCKS}
    rho["D3"], rho["D4"] = rho["D3i"], rho["D4i"]
    blocks_all = dict(G.BIV_BLOCKS)
    blocks_all.update(G.UNI_BLOCKS)
    for rname in ("budget4", "pair"):
        for t in RULE_DOCS[rname]["terms"]:
            out = t["output"]
            cf = float(Fraction(t["coef"]))
            for sig in sigmas:
                for al in (-3.0, 0.5, 3.0):
                    # reference in mp (sigma_i = sigma_j = sig, alpha_i = alpha_j = al)
                    ref = mp.mpf(cf) * mp.mpf(sig) ** sum(out)
                    for f in t["factors"]:
                        ref *= rho[f]
                    for kk, pp in zip(t["derivs"], out):
                        ref *= Wref(kk, pp, al)
                    # raw: kappa = rho*sig^order, w = sig^(p-k) W, multiplied in natural units
                    s32 = f32(sig)
                    raw = f32(cf)
                    with np.errstate(all="ignore"):
                        for f in t["factors"]:
                            raw = f32(raw * f32(rho[f]) * s32 ** f32(sum(blocks_all[f])))
                        for kk, pp in zip(t["derivs"], out):
                            w = W32(kk, pp, np.array([al]))[0][0]
                            raw = f32(raw * s32 ** f32(pp - kk) * w)
                        norm = f32(cf)
                        for f in t["factors"]:
                            norm = f32(norm * f32(rho[f]))
                        for kk, pp in zip(t["derivs"], out):
                            norm = f32(norm * W32(kk, pp, np.array([al]))[0][0])
                        norm = f32(norm * s32 ** f32(sum(out)))
                    for label, val, store in (("raw", raw, raw_bad), ("norm", norm, norm_bad)):
                        refv = float(ref)
                        badv = (not np.isfinite(val)) or (abs(refv) > 1e-30 and abs(float(val) - refv) > 1e-3 * abs(refv))
                        if badv:
                            key = f"{rname}:{'.'.join(t['factors']) or '1'}@{out}"
                            store.setdefault(key, []).append(sig)
    def sig_range(d):
        allsig = sorted({s for v in d.values() for s in v})
        return (min(allsig), max(allsig)) if allsig else None
    raw_in = {k: v for k, v in raw_bad.items() if any(1e-3 <= s <= 10 for s in v)}
    norm_in = {k: v for k, v in norm_bad.items() if any(1e-3 <= s <= 10 for s in v)}
    raw_small = sorted({s for v in raw_bad.values() for s in v if s < 1e-3})
    DETAILS["float32"]["raw_term_failures"] = {k: sorted(set(v)) for k, v in raw_bad.items()}
    DETAILS["float32"]["normalised_term_failures"] = {k: sorted(set(v)) for k, v in norm_bad.items()}
    record("(d) raw natural-unit term products in float32 (kappa=rho*sigma^order times sigma^(p-k) W): failures in sigma in [1e-3,10]",
           True, f"{len(raw_in)} terms fail inside [1e-3,10]; outside: largest failing sigma below 1e-3 = "
                 f"{max(raw_small) if raw_small else None}, failing sigma range overall {sig_range(raw_bad)}", informational=True)
    record("(d) normalised spelling (sigma^sum(output) * coef * prod(kappa/sigma^order) * prod W) finite and within 1e-3 rel, sigma in [1e-3,10]",
           len(norm_in) == 0, f"{len(norm_in)} failing terms inside range; overall failing sigma range {sig_range(norm_bad)}")


def main():
    t0 = time.time()
    check_w_forms()
    check_conversion()
    check_polynomial()
    check_upstream_terms()
    check_invariance()
    check_realizable()
    check_float32()
    counts = {r: RULE_DOCS[r]["counts"] for r in RULE_DOCS}
    report = {"results": RESULTS, "details": DETAILS, "term_counts": counts, "runtime_s": round(time.time() - t0, 1)}
    (HERE / "oracle_report.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print("\n==== summary ====")
    for r in RESULTS:
        print(f"{r['status']:5s} {r['check']}")
    nfail = sum(r["status"] == "FAIL" for r in RESULTS)
    print(f"{nfail} FAIL, runtime {report['runtime_s']} s")
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
