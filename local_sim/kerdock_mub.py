"""Real mutually-unbiased bases in R^(2^(m+1)) from GR(4, m). Phase 1A generator.

Specification: `docs/KERDOCK_MUB_SPEC.md`, written before this file existed.
Prediction K1-K11: `experiments/LOG.md`, `PHASE 1A ... PREDICTION`.

WHAT THIS EMITS
---------------
For odd `m`, `N = 2^(m+1)`, a deterministic subset of the `2^m + 1` real MUBs
that exist in `R^N` — entries `±1/√N`, exact within-basis orthogonality, and
cross-basis coherence exactly `1/√N`. At `m = 9`: `N = 1024`, coherence `1/32`,
and 16 bases give `16 · 1024 · 2 = 32,768` antipodal rows.

WHAT IT DOES NOT BUILD
----------------------
The Z4-Kerdock code `K(9)` has `4^10 = 1,048,576` words. **It is never
enumerated.** Only the `λ` of the requested bases are touched — 16 length-512
`Z4` words, not a million — which is the whole point of emitting directly from
the ring rather than filtering a code.

THE STRUCTURE THAT MAKES IT CHEAP
---------------------------------
`B_λ[x, y] = N^(−1/2) (−1)^(g_λ[x] ⊕ ⟨x,y⟩) = N^(−1/2) · s_λ[x] · W[x, y]`

with `W` the `±1` Walsh matrix and `s_λ = (−1)^(g_λ)`. So a basis is the Walsh
matrix with its **rows** sign-flipped by one length-`N` vector. One `W` is shared
by every basis; a basis costs one vector, not a matrix.

That factorisation is also why the invariants are exact integers:

* within a basis the flips cancel — `⟨s⊙W[:,y], s⊙W[:,y']⟩ = ⟨W[:,y], W[:,y']⟩`,
  which is `N·δ`. **This is why within-basis orthogonality is a weak check: it
  holds for ANY `s`, including a wrong one.** See the spec, §5.
* across bases the correlation is the Walsh coefficient of `g_λ ⊕ g_μ` at
  `y ⊕ y'`. That is `±2^((m+1)/2)` for every argument **iff** `g_λ ⊕ g_μ` is
  bent, which is exactly the Kerdock property. So the whole cross-basis identity
  reduces to one flat-spectrum test per pair — `C(16,2) = 120` transforms, not
  `1024²` inner products per pair.

Run:  uv run python local_sim/kerdock_mub.py
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# APPEND, never insert -- local_sim/ holds numpy shims named flopscope/whestbench.
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np  # noqa: E402

#: Primitive binary polynomials, by degree, as coefficient lists (index = power).
#: Only ODD m is meaningful here: CCKS give the real line-sets for odd m, and a
#: bent function on F2^(m+1) needs m+1 even. `_assert_ring` checks both.
PRIMITIVE = {
    3: [1, 1, 0, 1],                             # x^3 + x + 1
    5: [1, 0, 1, 0, 0, 1],                       # x^5 + x^2 + 1
    7: [1, 1, 0, 0, 0, 0, 0, 1],                 # x^7 + x + 1
    9: [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],           # x^9 + x^4 + 1
}


# --------------------------------------------------------------------------- #
# the ring GR(4, m)
# --------------------------------------------------------------------------- #

def hensel_lift(f: list, m: int) -> list:
    """The basic primitive polynomial over Z4 lifting `f`, by the Graeff method.

    Writing `f(x) = e(x²) + x·o(x²)` over F2 with coefficients lifted to {0,1},
    the lift satisfies `h(y) = ±( e(y)² − y·o(y)² ) mod 4`, normalised monic.

    The result is not trusted: `_assert_ring` checks `h ≡ f (mod 2)` and that `ξ`
    has order `2^m − 1`, which is what "basic primitive" means and what a wrong
    lift would fail.
    """
    e = [f[2 * i] for i in range((m + 2) // 2)]
    o = [f[2 * i + 1] for i in range((m + 1) // 2)]

    def mul(a, b):
        out = [0] * (len(a) + len(b) - 1)
        for i, ai in enumerate(a):
            if ai:
                for j, bj in enumerate(b):
                    out[i + j] = (out[i + j] + ai * bj) % 4
        return out

    e2, o2 = mul(e, e), mul(o, o)
    g = [0] * (m + 1)
    for i, v in enumerate(e2):
        if i <= m:
            g[i] = (g[i] + v) % 4
    for i, v in enumerate(o2):          # subtract y·o(y)²
        if i + 1 <= m:
            g[i + 1] = (g[i + 1] - v) % 4
    if g[m] % 4 == 3:                    # normalise monic
        g = [(-c) % 4 for c in g]
    return g


class GaloisRing:
    """`GR(4, m) = Z4[x]/(h)`, with the Teichmüller set and the trace.

    Elements are length-`m` tuples of `Z4` coefficients. Only what the carrier
    needs is implemented; this is not a general ring library.
    """

    def __init__(self, m: int):
        if m not in PRIMITIVE:
            raise ValueError(f"no primitive polynomial recorded for degree {m}")
        self.m = m
        self.n = m + 1
        self.N = 1 << self.n
        self.f = PRIMITIVE[m]
        self.h = hensel_lift(self.f, m)
        self.order = (1 << m) - 1
        self._build_powers()
        self._assert_ring()

    # -- arithmetic ------------------------------------------------------- #
    def _mul(self, a: tuple, b: tuple) -> tuple:
        m, h = self.m, self.h
        acc = [0] * (2 * m - 1)
        for i, ai in enumerate(a):
            if ai:
                for j, bj in enumerate(b):
                    acc[i + j] = (acc[i + j] + ai * bj) & 3
        for d in range(2 * m - 2, m - 1, -1):   # reduce mod h (monic)
            c = acc[d]
            if c:
                acc[d] = 0
                for k in range(m):
                    acc[d - m + k] = (acc[d - m + k] - c * h[k]) & 3
        return tuple(acc[:m])

    def _add(self, a: tuple, b: tuple) -> tuple:
        return tuple((x + y) & 3 for x, y in zip(a, b))

    def _build_powers(self) -> None:
        """`self.pw[i] = ξ^i` for `0 ≤ i < 2^m − 1`. `ξ` is the class of x."""
        m = self.m
        one = tuple([1] + [0] * (m - 1))
        xi = tuple([0, 1] + [0] * (m - 2)) if m > 1 else (1,)
        self.pw = [one]
        for _ in range(self.order - 1):
            self.pw.append(self._mul(self.pw[-1], xi))
        self.one, self.xi = one, xi
        self.zero = tuple([0] * m)

    def _assert_ring(self) -> None:
        m = self.m
        if m % 2 == 0:
            raise ValueError(f"m must be ODD for the REAL line-sets; got {m}")
        if self.n % 2 != 0:
            raise ValueError(f"n = m+1 must be EVEN for a bent function; got {self.n}")
        # h lifts f
        if [c & 1 for c in self.h] != self.f:
            raise AssertionError("Hensel lift does not reduce to f mod 2")
        # xi has order exactly 2^m - 1: closing the cycle, and no earlier return
        if self._mul(self.pw[-1], self.xi) != self.one:
            raise AssertionError("xi^(2^m-1) != 1: h is not basic primitive")
        if len(set(self.pw)) != self.order:
            raise AssertionError("xi has order < 2^m - 1: h is not primitive")

    # -- Teichmüller set and trace ---------------------------------------- #
    def teichmuller(self) -> list:
        """`T = {0} ∪ {ξ^i}`, in the fixed enumeration the spec pins (§4)."""
        return [self.zero] + list(self.pw)

    def trace_of_power(self, k: int) -> int:
        """`T_R(ξ^k) ∈ Z4`. Asserts the result really is a Z4 constant."""
        acc = self.zero
        for i in range(self.m):
            acc = self._add(acc, self.pw[(k << i) % self.order])
        if any(c for c in acc[1:]):
            raise AssertionError(f"trace of xi^{k} is not in Z4: {acc}")
        return acc[0]


# --------------------------------------------------------------------------- #
# Gray map, Kerdock words, and the sign vectors
# --------------------------------------------------------------------------- #

def gray(z: int) -> tuple:
    """`φ(0)=(0,0) φ(1)=(0,1) φ(2)=(1,1) φ(3)=(1,0)` — the standard convention."""
    hi = z >> 1
    return hi, hi ^ (z & 1)


def sign_vectors(m: int, k: int) -> np.ndarray:
    """`s_λ = (−1)^(g_λ)` for the first `k` bases. Shape `(k, N)`, values `±1`.

    `g_λ` is the Gray image of the `Z4` word `c_λ(x) = T_R(λ x)`, `x ∈ T`, laid
    out at `index = 2·rank(x) + t` exactly as the spec fixes in §4.

    Only `k` words are ever formed. The `4^(m+1)`-word code is not enumerated.
    """
    R = GaloisRing(m)
    T = R.teichmuller()
    if not 1 <= k <= len(T):
        raise ValueError(f"k must be in 1..{len(T)}; got {k}")
    N = R.N
    out = np.empty((k, N), dtype=np.int8)

    # THE COORDINATE IDENTIFICATION, and the first version of it was wrong.
    #
    # Position must be indexed by the ADDITIVE image of x, not by its rank in the
    # multiplicative enumeration `0, ξ^0, ξ^1, …`. Reduction mod 2 is a bijection
    # `T → R/2R ≅ GF(2^m)`, and *that* is F2-linear; `rank` is not linear in
    # anything. Indexing by rank produced functions of ANF degree 9 instead of 2,
    # singular matrix differences, and a cross-basis spectrum of
    # `[0, 4, 8, …, 104]` instead of a flat 32 — while within-basis orthogonality
    # passed perfectly, exactly as the spec warned a weak check would.
    # `docs/KERDOCK_MUB_SPEC.md` §4 is corrected to match.
    pos = {}
    for x in T:
        bits = 0
        for i, c in enumerate(x):
            if c & 1:
                bits |= 1 << i
        pos[x] = bits
    if len(set(pos.values())) != len(T):
        raise AssertionError("T -> GF(2^m) is not injective; the ring is wrong")

    for bi in range(k):
        lam = T[bi]
        lam_k = None if lam == R.zero else R.pw.index(lam)
        g = np.empty(N, dtype=np.int8)
        for x in T:
            if lam == R.zero or x == R.zero:
                z = 0                       # T_R(0) = 0
            else:
                x_k = R.pw.index(x)
                z = R.trace_of_power((lam_k + x_k) % R.order)
            hi, lo = gray(z)
            base = 2 * pos[x]
            g[base] = hi
            g[base + 1] = lo
        out[bi] = 1 - 2 * g                  # (-1)^g
    return out


def walsh_matrix(n: int) -> np.ndarray:
    """`W[x, y] = (−1)^⟨x,y⟩` on `F2^n`, as `int8`. Shared by every basis."""
    N = 1 << n
    x = np.arange(N, dtype=np.uint32)[:, None]
    y = np.arange(N, dtype=np.uint32)[None, :]
    par = np.bitwise_count(x & y).astype(np.uint8) & 1 \
        if hasattr(np, "bitwise_count") else _popcount_parity(x & y)
    return (1 - 2 * par).astype(np.int8)


def _popcount_parity(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.uint32).copy()
    v ^= v >> 16
    v ^= v >> 8
    v ^= v >> 4
    v ^= v >> 2
    v ^= v >> 1
    return (v & 1).astype(np.uint8)


# --------------------------------------------------------------------------- #
# the carrier
# --------------------------------------------------------------------------- #

def carrier_int8(m: int = 9, k: int = 16, antipodes: bool = True) -> np.ndarray:
    """The unscaled `±1` rows. Shape `(k·N·(2 if antipodes else 1), N)`.

    Row order is deterministic and stated: basis-major, then column `y`
    ascending, then `+v` before `−v`. No seed is involved anywhere.
    """
    R = GaloisRing(m)
    N, n = R.N, R.n
    s = sign_vectors(m, k)
    W = walsh_matrix(n)
    # basis λ, column y  ->  row  s_λ ⊙ W[:, y].  W is symmetric, so W[:, y] is
    # W[y, :]; taking rows keeps the result C-contiguous.
    rows = (s[:, None, :] * W[None, :, :]).reshape(k * N, N)
    if not antipodes:
        return np.ascontiguousarray(rows)
    both = np.empty((2 * k * N, N), dtype=np.int8)
    both[0::2] = rows
    both[1::2] = -rows
    return both


def carrier_float32(m: int = 9, k: int = 16, antipodes: bool = True) -> np.ndarray:
    """The carrier scaled to unit rows: entries `±1/√N`, `float32`.

    At `m = 9, k = 16` this is `32,768 × 1024` float32 = **128 MiB**, one
    allocation, far below the 4 GiB single-array cap.
    """
    N = 1 << (m + 1)
    return (carrier_int8(m, k, antipodes) / np.float32(np.sqrt(N))).astype(np.float32)


def main() -> int:
    for m in (3, 5, 7, 9):
        R = GaloisRing(m)
        k = min(16, 1 << m)
        s = sign_vectors(m, k)
        print(f"m={m:>2}  N={R.N:>5}  |T|={1 << m:>4}  MUBs available={(1 << m) + 1:>4}"
              f"  h={R.h}  bases emitted={k}  coherence=1/{int(np.sqrt(R.N))}")
    rows = carrier_int8(9, 16)
    print(f"\nm=9, 16 bases, antipodal: rows={rows.shape[0]:,} dim={rows.shape[1]} "
          f"({rows.nbytes / 2**20:.0f} MiB int8, "
          f"{rows.shape[0] * rows.shape[1] * 4 / 2**20:.0f} MiB float32)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
