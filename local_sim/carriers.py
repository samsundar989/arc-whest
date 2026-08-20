"""
CARRIERS — one interface for every angular point set, so comparisons are about
the point set and nothing else.

Stage 1 / item 5 of the implementation handoff. Contract and predictions S1-S5:
`experiments/LOG.md`, 2026-08-19 (night).

WHY ONE INTERFACE
-----------------
`docs/CURRENT_STATE.md` open question #3 leaves exactly one input-space lane
alive: a true degree-4/5 angular design, external ceiling ~1.46x. Every previous
carrier comparison in this repo built its point set inline inside its own probe,
so a difference between two arms could always be a difference in the surrounding
engineering. Here a carrier is a function `points(n, dim, seed) -> (n, dim)` and
nothing else changes.

THE RADIAL IDENTITY, and why it is exact rather than an approximation
---------------------------------------------------------------------
A bias-free ReLU network is positively homogeneous: `f(c x) = c f(x)` for `c > 0`.
So for `x ~ N(0, I_d)`, writing `x = R u` with `R = ||x||` and `u = x/R` uniform
on the sphere, and using that `R` and `u` are INDEPENDENT:

    E[f(x)] = E[R] * E[f(u)]           with  E[R] = sqrt(2) * Gamma((d+1)/2) / Gamma(d/2)

Sampling `R` therefore contributes pure variance and nothing else. `exact_radius`
replaces it by its mean. This is an identity for our problem, not a modelling
choice — and it is the first thing that breaks if Phase 2 introduces biases.

DESIGN STRENGTH
---------------
A spherical t-design integrates every polynomial of degree <= t exactly. What the
estimator actually needs is the low-degree part of the network response, so "how
strong a design is this point set" is the quantity that decides whether the lane
is worth building. `local_sim/test_carriers.py` measures it; nothing here claims
it.

Run:  uv run python local_sim/test_carriers.py
"""

from __future__ import annotations

import math
import os
import sys
from typing import Callable, Dict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

_SOBOL_BITS = 30
_DIRNUM = None


# --------------------------------------------------------------------------- #
# radial part
# --------------------------------------------------------------------------- #


def mean_radius(dim: int) -> float:
    """E||x|| for x ~ N(0, I_dim). Exact, via lgamma to survive dim = 256."""
    return math.exp(0.5 * math.log(2.0)
                    + math.lgamma((dim + 1) / 2.0) - math.lgamma(dim / 2.0))


def to_sphere(x: np.ndarray) -> np.ndarray:
    """Project onto the unit sphere. Rows of norm 0 are left alone."""
    r = np.linalg.norm(x.astype(np.float64), axis=1, keepdims=True)
    r = np.where(r > 0, r, 1.0)
    return (x.astype(np.float64) / r)


def exact_radius(u: np.ndarray, dim: int) -> np.ndarray:
    """Scale unit vectors to the exact mean Gaussian radius. See module docstring."""
    return u * mean_radius(dim)


# --------------------------------------------------------------------------- #
# the shipped carrier — Sobol, digitally shifted, inverse-CDF
# --------------------------------------------------------------------------- #


def _dirnum() -> np.ndarray:
    global _DIRNUM
    if _DIRNUM is None:
        blob = np.load(os.path.join(_HERE, "sobol_dirnum.npz"))
        _DIRNUM = blob[blob.files[0]]
    return _DIRNUM


def _sobol_int(n: int, dim: int) -> np.ndarray:
    """Unshifted Sobol integers, gray-code order. Mirrors estimator._base_points."""
    dn = _dirnum()
    idx = np.arange(n, dtype=np.uint32)
    idx = idx ^ (idx >> np.uint32(1))
    idx = idx.reshape((n, 1))
    pts = np.zeros((n, dim), dtype=np.uint32)
    for k in range(_SOBOL_BITS):
        bit = (idx >> np.uint32(k)) & np.uint32(1)
        pts ^= bit * dn[k][:dim].astype(np.uint32)
    return pts


def _norm_ppf(u: np.ndarray) -> np.ndarray:
    """Inverse standard normal CDF, vectorised. Acklam, |err| < 1.2e-9."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    out = np.empty_like(u, dtype=np.float64)

    lo = u < plow
    if lo.any():
        q = np.sqrt(-2 * np.log(u[lo]))
        out[lo] = ((((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])
                   / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1))
    hi = u > phigh
    if hi.any():
        q = np.sqrt(-2 * np.log(1 - u[hi]))
        out[hi] = -((((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])
                    / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1))
    mid = ~(lo | hi)
    if mid.any():
        q = u[mid] - 0.5
        r = q * q
        out[mid] = ((((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q
                    / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1))
    return out


def sobol(n: int, dim: int, seed: int) -> np.ndarray:
    """The SHIPPED carrier: digitally shifted Sobol through the inverse CDF."""
    rng = np.random.default_rng(seed)
    shift = rng.integers(0, 1 << _SOBOL_BITS, size=dim, dtype=np.uint32)
    p = np.bitwise_xor(_sobol_int(n, dim), shift)
    u = (p.astype(np.float64) + 0.5) * (1.0 / float(1 << _SOBOL_BITS))
    return _norm_ppf(u)


def antithetic(base: Callable[[int, int, int], np.ndarray]):
    """Half the points, each with its negation. Forces every ODD moment to zero."""
    def f(n: int, dim: int, seed: int) -> np.ndarray:
        if n & 1:
            raise ValueError("antithetic needs an even n")
        h = base(n // 2, dim, seed)
        return np.concatenate([h, -h], axis=0)
    return f


def spherical(base: Callable[[int, int, int], np.ndarray]):
    """Project a carrier onto the sphere and rescale to the exact mean radius."""
    def f(n: int, dim: int, seed: int) -> np.ndarray:
        return exact_radius(to_sphere(base(n, dim, seed)), dim)
    return f


# --------------------------------------------------------------------------- #
# frame banks — K orthonormal bases, with antipodes
# --------------------------------------------------------------------------- #


def haar_frames(n: int, dim: int, seed: int) -> np.ndarray:
    """K Haar-random orthonormal bases, each with its antipodes, on the sphere.

    Exact at degree 2 by construction: for ANY orthonormal basis {q_j},
    `mean_j (a . q_j)^2 = ||a||^2 / d` identically, so averaging over bases cannot
    break it. Antipodes kill every odd moment. Nothing makes it exact at degree 4.
    That combination is what makes it the right control for a real design.
    """
    if n % (2 * dim):
        raise ValueError(f"haar_frames needs n divisible by 2*dim = {2 * dim}")
    k = n // (2 * dim)
    rng = np.random.default_rng(seed)
    out = np.empty((n, dim), dtype=np.float64)
    for i in range(k):
        q, r = np.linalg.qr(rng.standard_normal((dim, dim)))
        # QR sign convention is not Haar on its own; fix it with the R diagonal.
        q = q * np.sign(np.diag(r))
        out[2 * i * dim:(2 * i + 1) * dim] = q.T
        out[(2 * i + 1) * dim:(2 * i + 2) * dim] = -q.T
    return exact_radius(out, dim)


def iid(n: int, dim: int, seed: int) -> np.ndarray:
    """Plain Monte Carlo. The floor everything else is measured against."""
    return np.random.default_rng(seed).standard_normal((n, dim))


def iid_sphere(n: int, dim: int, seed: int) -> np.ndarray:
    return exact_radius(to_sphere(iid(n, dim, seed)), dim)


# --------------------------------------------------------------------------- #
# dual-BCH orthogonal array — a candidate degree-4/5 carrier
# --------------------------------------------------------------------------- #


def _gf256_tables(poly: int = 0x11D):
    """Log/antilog tables for F_256 with a primitive polynomial. Default 0x11D."""
    exp = np.zeros(512, dtype=np.int32)
    log = np.zeros(256, dtype=np.int32)
    x = 1
    for i in range(255):
        exp[i] = x
        log[x] = i
        x <<= 1
        if x & 0x100:
            x ^= poly
    for i in range(255, 512):
        exp[i] = exp[i - 255]
    return exp, log


def _gf_mul(a: np.ndarray, b: np.ndarray, exp, log) -> np.ndarray:
    out = np.zeros(np.broadcast(a, b).shape, dtype=np.int32)
    nz = (a != 0) & (b != 0)
    out[nz] = exp[(log[a * np.ones_like(out)][nz] + log[b * np.ones_like(out)][nz]) % 255]
    return out


def bch_oa_signs(m: int = 8, poly: int = 0x11D) -> np.ndarray:
    """The +-1 rows of the extended dual-BCH orthogonal array on d = 2^m factors.

    WHY THIS SET. Let B be the 2-error-correcting BCH code of length 2^m - 1; its
    designed distance is 5. The codewords of a linear code C form an orthogonal
    array of strength `d(C_perp) - 1`, so taking C = B_perp gives strength 4.
    B_perp is the classical `{Tr(a x + b x^3)}` code of dimension 2m, and adding
    the all-ones coset (the extension) closes it under negation, which makes every
    ODD moment vanish for free.

    Rows: 2^(2m+1) = 131,072 at m = 8, i.e. 65,536 antipodal pairs, on d = 256
    coordinates. Scaled to the unit sphere.

    An orthogonal array is a design for the CUBE, not for the sphere: every
    coordinate of every point is +-1/sqrt(d), so `E[(a.u)^4]` cannot match the
    spherical value along coordinate-aligned `a`. Whether it is useful here is an
    empirical question about generic directions — which is what
    `test_carriers.py` measures, and why this is built behind the verifier.
    """
    d = 1 << m
    exp, log = _gf256_tables(poly)
    if m != 8:
        raise NotImplementedError("the F_256 tables above are m = 8 only")

    x = np.arange(d, dtype=np.int32)                       # all of F_256, 0 included
    x3 = _gf_mul(_gf_mul(x, x, exp, log), x, exp, log)     # x^3 in F_256

    def trace(v: np.ndarray) -> np.ndarray:
        """Absolute trace F_256 -> F_2, as v + v^2 + v^4 + ... + v^128."""
        t = np.zeros_like(v)
        acc = v.copy()
        for _ in range(m):
            t ^= acc
            acc = _gf_mul(acc, acc, exp, log)
        return t & 1

    ab = np.arange(d, dtype=np.int32)
    rows = np.empty((d * d, d), dtype=np.int8)
    k = 0
    for a in ab:
        ax = _gf_mul(np.full(d, a, dtype=np.int32), x, exp, log)
        for b in ab:
            bx3 = _gf_mul(np.full(d, b, dtype=np.int32), x3, exp, log)
            rows[k] = trace(ax ^ bx3).astype(np.int8)
            k += 1
    signs = 1 - 2 * rows.astype(np.int8)                   # 0 -> +1, 1 -> -1
    return np.concatenate([signs, -signs], axis=0)         # the extension coset


def bch_oa_subsample(n: int, dim: int, seed: int) -> np.ndarray:
    """A random subset of the FULL array — the affordable option, strength measured.

    WHY NOT SIMPLY HALVE IT. Taking one row per antipodal pair returns the linear
    code itself, and that code has a **constant column**: coordinate `x = 0`
    carries `Tr(a*0 + b*0^3) = 0` for every `(a, b)`. Its centroid then has norm
    **1.0** — the point set sits off-centre by a full unit vector and explores a
    255-dimensional slice. Measured end to end it was **41x worse than Sobol**.

    Puncturing to `F_256*` and appending the overall parity does not fix it
    either: `sum_{x in F*} x = 0` and `sum_{x in F*} x^3 = 0` (the cubes are the
    index-3 subgroup, whose elements are the roots of `X^85 - 1` and therefore sum
    to zero), so the parity column is constant too. Both were built, both were
    caught by the centroid check, and neither is offered here.

    What IS offered is an honest approximation: a random subset of the full,
    verified, antipodally-closed array. Exact combinatorial strength does not
    survive subsampling — `test_carriers.py` measures what actually does.
    """
    if dim != 256:
        raise NotImplementedError("built for dim = 256")
    full = bch_oa_signs()
    if n > full.shape[0]:
        raise ValueError(f"only {full.shape[0]} rows exist")
    rng = np.random.default_rng(seed)
    idx = rng.choice(full.shape[0], size=n, replace=False)
    q, r = np.linalg.qr(rng.standard_normal((dim, dim)))
    q = q * np.sign(np.diag(r))
    return exact_radius(to_sphere(full[idx].astype(np.float64) @ q.T), dim)


def bch_oa(n: int, dim: int, seed: int) -> np.ndarray:
    """Randomly rotated dual-BCH OA, scaled to the exact mean Gaussian radius.

    The rotation is what makes this an unbiased rule for a general integrand: an
    unrotated OA is aligned with the coordinate axes, and the estimator's own
    rank-32 rotation already occupies that role. `n` selects a random subset of
    rows when it is smaller than the full array — WHICH DESTROYS THE STRENGTH, and
    is offered only so the verifier can show that it does.
    """
    if dim != 256:
        raise NotImplementedError("dual-BCH OA is built for dim = 256")
    full = bch_oa_signs()
    rng = np.random.default_rng(seed)
    if n < full.shape[0]:
        idx = rng.choice(full.shape[0], size=n, replace=False)
        pts = full[idx].astype(np.float64)
    else:
        pts = full.astype(np.float64)
    q, r = np.linalg.qr(rng.standard_normal((dim, dim)))
    q = q * np.sign(np.diag(r))
    return exact_radius(to_sphere(pts @ q.T), dim)


def bch_oa_unrotated(n: int, dim: int, seed: int) -> np.ndarray:
    """The same array with NO rotation, so the axis alignment is visible."""
    if dim != 256:
        raise NotImplementedError("dual-BCH OA is built for dim = 256")
    full = bch_oa_signs()
    if n < full.shape[0]:
        rng = np.random.default_rng(seed)
        idx = rng.choice(full.shape[0], size=n, replace=False)
        full = full[idx]
    return exact_radius(to_sphere(full.astype(np.float64)), dim)


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #

CARRIERS: Dict[str, Callable[[int, int, int], np.ndarray]] = {
    "iid": iid,
    "iid+sphere": iid_sphere,
    "sobol (SHIPPED)": sobol,
    "sobol+anti": antithetic(sobol),
    "sobol+anti+sphere": spherical(antithetic(sobol)),
    "haar frames": haar_frames,
    "bch OA (rotated)": bch_oa,
    "bch OA half (random)": bch_oa_subsample,
    "bch OA (unrotated)": bch_oa_unrotated,
}


def build(name: str, n: int, dim: int, seed: int) -> np.ndarray:
    if name not in CARRIERS:
        raise KeyError(f"unknown carrier {name!r}; known: {sorted(CARRIERS)}")
    return CARRIERS[name](n, dim, seed)
