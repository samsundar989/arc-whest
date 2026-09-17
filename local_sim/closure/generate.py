"""Symbolic generator for the ReLU nonlinear moment closure (exact rationals).

Pure stdlib (fractions, itertools, json). Run from the repo root:

    uv run python local_sim/closure/generate.py

Writes, next to this file:
    closure_budget2.json, closure_budget3.json, closure_budget4.json, closure_pair.json
    w_forms.json        closed forms of w(k,p) = E_{N(mu,s)}[d^k ReLU(y)^p], k<=8, p<=4

Derivation (summarised; README.md has the statement of assumptions)
-------------------------------------------------------------------
Let the pre-activation law have joint cumulants kappa_v.  Its characteristic
function is exp(sum_v kappa_v (i t)^v / v!).  Split off the "base" blocks
(means v=e_i, variances v=2e_i): these define independent Gaussian marginals
N(mu_i, s_i).  Every remaining block multiplies the base characteristic function
by exp(kappa_v (i t)^v / v!); since (i t)^v acting on a Fourier transform is the
derivative d^v with respect to the mean (translation generator), formally

    E[F(y)] = exp( L ) E_0[F(y)],   L = sum_{v non-base} kappa_v / v! d^v,

where E_0 is under independent N(mu_i, s_i) and d^v = prod_i d^{v_i}/dmu_i^{v_i}.
Since d/dmu E_0[f(y_i)] = E_0[f'(y_i)], each monomial of exp(L) becomes
    coef * prod(kappa) * prod_i w(k_i, p_i)_i ,   k_i = sum of i-entries of the blocks,
with coef = prod_{distinct blocks} 1 / (mult! * (v!)^mult)   (multinomial exp expansion).

Connected projection for bivariate outputs.  For F = ReLU(y_i)^a, G = ReLU(y_j)^b
split L = L_i + L_j + L_mix (L_i: blocks supported on i only: D3i, D4i; L_j likewise;
L_mix: blocks with support on both: C, D21, D12, D22).  All pieces commute, and
E_0 factorises: E_0[FG] = E_0[F] E_0[G], with d_i acting only on the F factor.  Hence
    E[F]   = exp(L_i) E_0[F],    E[G] = exp(L_j) E_0[G],
    E[F]E[G] = exp(L_i + L_j) E_0[FG],
    E[FG]    = exp(L_i + L_j) exp(L_mix) E_0[FG],
    Cov(F,G) = exp(L_i + L_j) (exp(L_mix) - 1) E_0[FG].
So pK(a,b) is exactly the set of monomials containing at least one mixed block;
purely-univariate products (e.g. D3i*D3j) cancel identically.  Truncation is then
applied monomial-by-monomial to this connected set.  (Truncating E[FG], E[F], E[G]
separately and subtracting is NOT the same: E[F]_trunc*E[G]_trunc contains products
such as D3i*D3j of combined cost 4 that are not cancelled at budget 2.)
"""
from __future__ import annotations

import json
import math
from fractions import Fraction
from itertools import product
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEMA = "arc-whest/relu-closure/v1"

# ----------------------------------------------------------------------------- blocks
BIV_BLOCKS: dict[str, tuple[int, int]] = {
    "C": (1, 1),
    "D21": (2, 1),
    "D12": (1, 2),
    "D22": (2, 2),
    "D3i": (3, 0),
    "D3j": (0, 3),
    "D4i": (4, 0),
    "D4j": (0, 4),
}
MIXED = {"C", "D21", "D12", "D22"}
UNI_BLOCKS: dict[str, tuple[int, ...]] = {"D3": (3,), "D4": (4,)}
SWAP = {"C": "C", "D21": "D12", "D12": "D21", "D22": "D22",
        "D3i": "D3j", "D3j": "D3i", "D4i": "D4j", "D4j": "D4i"}

UNAVAILABLE = [
    "kappa_iiij (multi-index (3,1)) and kappa_ijjj ((1,3)) are unavailable and treated as zero",
    "all cumulants of total order >= 5 (univariate and mixed) are unavailable and treated as zero",
    "means and variances (multi-indices e_i, 2e_i) are absorbed into the independent Gaussian base N(mu_i, s_i); "
    "the covariance C is NOT in the base, it is an operator block (1,1)",
]

UNI_OUTPUTS = [(1,), (2,), (3,), (4,)]
BIV_OUTPUTS = [(1, 1), (2, 1), (1, 2), (2, 2)]


def block_cost(v) -> int:
    """Pinned upstream power counting (mlp_kprop.kprop_harmonic.vec_block_cost)."""
    return sum(v) - 1 - (1 if all(x % 2 == 0 for x in v) else 0)


def frac_str(x: Fraction) -> str:
    return f"{x.numerator}/{x.denominator}"


# ----------------------------------------------------------------------------- rules
class Rule:
    """keep(multiset) must be monotone (if a multiset fails, every superset fails)."""

    name: str

    def keep(self, names: list[str], blocks: dict) -> bool:  # pragma: no cover
        raise NotImplementedError

    def describe(self) -> dict:  # pragma: no cover
        raise NotImplementedError


class Budget(Rule):
    def __init__(self, budget: int):
        self.budget = budget
        self.name = f"budget{budget}"

    def keep(self, names, blocks):
        return sum(block_cost(blocks[n]) for n in names) <= self.budget

    def describe(self):
        return {
            "kind": "BUDGET",
            "budget": self.budget,
            "statement": "keep a monomial iff sum over its factors of cost(v) <= budget, "
                         "cost(v) = sum(v) - 1 - [all entries of v even] (entries over (i,j))",
            "upstream_equivalent": f"mlp_kprop get_vec_cond(k_max={self.budget + 1}, augment=False) "
                                   "restricted to the available blocks",
        }


class Pair(Rule):
    name = "pair"

    def keep(self, names, blocks):
        return len(names) <= 2

    def describe(self):
        return {"kind": "PAIR", "statement": "keep every monomial with at most two cumulant factors, any cost"}


class DerivCaps(Rule):
    """Untruncated operator restricted to derivative orders <= caps (exact for polynomials of those degrees)."""

    def __init__(self, caps):
        self.caps = tuple(caps)
        self.name = f"caps{self.caps}"

    def keep(self, names, blocks):
        k = derivs(names, blocks, len(self.caps))
        return all(ki <= ci for ki, ci in zip(k, self.caps))

    def describe(self):
        return {"kind": "DERIV_CAPS", "caps": list(self.caps)}


# ----------------------------------------------------------------------------- enumeration
def derivs(names, blocks, dim):
    k = [0] * dim
    for n in names:
        for a in range(dim):
            k[a] += blocks[n][a]
    return k


def coef_of(names, blocks) -> Fraction:
    c = Fraction(1)
    for n in set(names):
        m = names.count(n)
        vfac = math.prod(math.factorial(x) for x in blocks[n])
        c /= math.factorial(m) * vfac ** m
    return c


def enumerate_monomials(blocks: dict, rule: Rule, max_factors: int = 64):
    """All multisets of block names (sorted lists) accepted by the monotone rule."""
    order = sorted(blocks)
    out = []

    def rec(idx, cur):
        out.append(list(cur))
        if len(cur) >= max_factors:
            return
        for t in range(idx, len(order)):
            cand = cur + [order[t]]
            if rule.keep(cand, blocks):
                rec(t, cand)

    if rule.keep([], blocks):
        rec(0, [])
    return out


def terms_for_output(output: tuple[int, ...], rule: Rule):
    if len(output) == 1:
        blocks = {k: v for k, v in UNI_BLOCKS.items()}
        monos = enumerate_monomials(blocks, rule)
    else:
        blocks = BIV_BLOCKS
        monos = [m for m in enumerate_monomials(blocks, rule) if any(n in MIXED for n in m)]
    terms = []
    for m in monos:
        names = sorted(m)
        terms.append({
            "output": list(output),
            "factors": names,
            "derivs": derivs(names, blocks, len(output)),
            "coef": frac_str(coef_of(names, blocks)),
        })
    terms.sort(key=lambda t: (len(t["factors"]), t["derivs"], t["factors"]))
    return terms


# ----------------------------------------------------------------------------- w(k,p) closed forms
# Polynomials are integer coefficient lists, ascending powers of alpha.
def padd(a, b):
    n = max(len(a), len(b))
    return [(a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0) for i in range(n)]


def pscale(a, c):
    return [c * x for x in a]


def pmul(a, b):
    if not a or not b:
        return []
    r = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            r[i + j] += x * y
    return r


def ptrim(a):
    a = list(a)
    while a and a[-1] == 0:
        a.pop()
    return a


def hermite_prob(m: int):
    """Probabilists' Hermite He_m as ascending integer coefficients."""
    h0, h1 = [1], [0, 1]
    if m == 0:
        return h0
    for n in range(1, m):
        h0, h1 = h1, padd(pmul([0, 1], h1), pscale(h0, -n))
    return h1


def partial_std_moment(r: int):
    """M_r(alpha) = E[z^r 1{z > -alpha}], z~N(0,1), as (P_phi, P_Phi).
    M_0 = Phi, M_1 = phi, M_r = (r-1) M_{r-2} + (-alpha)^{r-1} phi."""
    if r == 0:
        return [], [1]
    if r == 1:
        return [1], []
    p1, p2 = partial_std_moment(r - 2)
    mono = [0] * (r - 1) + [(-1) ** (r - 1)]
    return padd(pscale(p1, r - 1), mono), pscale(p2, r - 1)


def w_form(k: int, p: int) -> dict:
    """w(k,p) = E_{N(mu,sigma^2)}[d^k ReLU(y)^p] = sigma^(p-k) (P_phi(alpha) phi(alpha) + P_Phi(alpha) Phi(alpha)).

    k <= p:  d^k ReLU^p = p!/(p-k)! y^(p-k) 1{y>0}   (for k=p this is p! 1{y>0});
             y = sigma (alpha + z), expand binomially, use partial standard moments M_r.
    k >  p:  d^k ReLU^p = p! delta^(m)(y), m = k-p-1;
             E[delta^(m)(y)] = (-1)^m g^(m)(0), g the N(mu,sigma^2) density,
             g^(m)(y) = sigma^(-m-1) (-1)^m He_m((y-mu)/sigma) phi(.), evaluated at z=-alpha:
             => p! (-1)^m He_m(alpha) phi(alpha) sigma^(-(m+1)).
    """
    assert p >= 1 and k >= 0
    if k <= p:
        n = p - k
        ff = math.factorial(p) // math.factorial(n)
        P1, P2 = [], []
        for r in range(n + 1):
            # C(n,r) alpha^(n-r) M_r
            m1, m2 = partial_std_moment(r)
            shift = [0] * (n - r) + [math.comb(n, r)]
            P1 = padd(P1, pmul(shift, m1))
            P2 = padd(P2, pmul(shift, m2))
        P1, P2 = pscale(P1, ff), pscale(P2, ff)
        branch = "partial_gaussian_moment"
    else:
        m = k - p - 1
        P1 = pscale(hermite_prob(m), math.factorial(p) * (-1) ** m)
        P2 = []
        branch = "delta_derivative_hermite"
    return {"k": k, "p": p, "sigma_exp": p - k, "P_phi": ptrim(P1), "P_Phi": ptrim(P2), "branch": branch}


# ----------------------------------------------------------------------------- conversion
CONVERSION = {
    "description": "moment-type pK -> joint cumulants K of post-activation outputs phi_i = ReLU(y_i). "
                   "Each entry: list of [integer coefficient, [pK factors]]; 'pK1_i' is pK(1) of neuron i, "
                   "'pK(2,1)' = Cov(phi_i^2, phi_j), 'pK(1,2)' = Cov(phi_i, phi_j^2).",
    "K(1)": [[1, ["pK1"]]],
    "K(2)": [[1, ["pK2"]], [-1, ["pK1", "pK1"]]],
    "K(3)": [[1, ["pK3"]], [-3, ["pK1", "pK2"]], [2, ["pK1", "pK1", "pK1"]]],
    "K(4)": [[1, ["pK4"]], [-4, ["pK1", "pK3"]], [-3, ["pK2", "pK2"]], [12, ["pK1", "pK1", "pK2"]],
             [-6, ["pK1", "pK1", "pK1", "pK1"]]],
    "K(1,1)": [[1, ["pK(1,1)"]]],
    "K(2,1)": [[1, ["pK(2,1)"]], [-2, ["pK1_i", "pK(1,1)"]]],
    "K(1,2)": [[1, ["pK(1,2)"]], [-2, ["pK1_j", "pK(1,1)"]]],
    "K(2,2)": [[1, ["pK(2,2)"]], [-2, ["pK1_j", "pK(2,1)"]], [-2, ["pK1_i", "pK(1,2)"]],
               [-2, ["pK(1,1)", "pK(1,1)"]], [4, ["pK1_i", "pK1_j", "pK(1,1)"]]],
    "meaning": "K(2,1) = kappa(phi_i, phi_i, phi_j); K(2,2) = kappa(phi_i, phi_i, phi_j, phi_j). "
               "Verified exactly (Fraction arithmetic, random discrete laws) by oracles.py.",
}

RULES = [Budget(2), Budget(3), Budget(4), Pair()]


def build(rule: Rule) -> dict:
    blocks_meta = {}
    for n, v in BIV_BLOCKS.items():
        blocks_meta[n] = {"multi_index": list(v), "cost": block_cost(v), "mixed": n in MIXED}
    for n, v in UNI_BLOCKS.items():
        blocks_meta[n] = {"multi_index": list(v), "cost": block_cost(v), "mixed": False,
                          "note": "univariate-output spelling of D3i/D4i"}
    terms, counts = [], {}
    wneed = set()
    for out in UNI_OUTPUTS + BIV_OUTPUTS:
        ts = terms_for_output(out, rule)
        counts[",".join(map(str, out))] = len(ts)
        terms.extend(ts)
        for t in ts:
            for k, p in zip(t["derivs"], out):
                wneed.add((k, p))
    return {
        "schema": SCHEMA,
        "rule": rule.describe(),
        "assumptions": UNAVAILABLE,
        "conventions": {
            "operator": "E[F(y)] = exp(sum_v kappa_v/v! d^v) E_0[F(y)], E_0 under independent N(mu_i, s_i)",
            "w": "w(k,p)_i = E_{N(mu_i,s_i)}[d^k/dy^k ReLU(y)^p] (distributional derivatives); see w_forms.json",
            "term_value": "coef * prod(factor cumulant values) * prod_axis w(derivs[axis], output[axis])_axis",
            "pK_univariate": "pK(p) = E[ReLU(y_i)^p], all monomials in D3, D4 (identity term included)",
            "pK_bivariate": "pK(a,b) = Cov(ReLU(y_i)^a, ReLU(y_j)^b): only monomials with >= 1 mixed block "
                            "(C, D21, D12, D22); both orientations listed explicitly, index 0 = i, index 1 = j",
            "coef": "prod over distinct factors of 1/(multiplicity! * (prod v!)^multiplicity)",
            "base_variance_note": "s_i must be the FULL variance kappa_ii (not a residual)",
        },
        "blocks": blocks_meta,
        "counts": counts,
        "w_needed": sorted([list(x) for x in wneed]),
        "max_deriv": max(k for k, _ in wneed),
        "conversion": CONVERSION,
        "terms": terms,
    }


def dump(obj, path: Path):
    # bytes, LF on every OS: the v35 arms embed this file's sha256
    path.write_bytes((json.dumps(obj, sort_keys=True, indent=1) + "\n").encode("utf-8"))


def main():
    for rule in RULES:
        doc = build(rule)
        dump(doc, HERE / f"closure_{rule.name}.json")
        print(rule.name, doc["counts"], "max_deriv", doc["max_deriv"])
    forms = [w_form(k, p) for p in range(1, 5) for k in range(0, 9)]
    dump({"schema": SCHEMA + "/w_forms",
          "statement": "w(k,p) = sigma^sigma_exp * (P_phi(alpha) phi(alpha) + P_Phi(alpha) Phi(alpha)), "
                       "alpha = mu/sigma, polynomials as ascending integer coefficient lists",
          "normalised_spelling": "W(k,p) := sigma^(k-p) w(k,p) = P_phi phi + P_Phi Phi is dimensionless; a term "
                                 "coef*prod(kappa_v)*w(k_i,a)_i*w(k_j,b)_j equals sigma_i^a sigma_j^b * coef * "
                                 "prod(kappa_v / (sigma_i^v_i sigma_j^v_j)) * W(k_i,a)_i * W(k_j,b)_j",
          "forms": forms}, HERE / "w_forms.json")
    print("wrote", len(forms), "w forms")


if __name__ == "__main__":
    main()
