"""WP1 — the Kerdock/real-MUB angular carrier, built in stdlib + `flopscope.numpy`.

`GAME_PLAN.md` § WP1. Specification: `docs/KERDOCK_MUB_SPEC.md`. Parity oracle:
`local_sim/kerdock_mub.py`, which is FROZEN and is not modified to make parity
easier.

WHY THIS FILE EXISTS AND `kerdock_mub.py` IS NOT ENOUGH
-------------------------------------------------------
`kerdock_mub.py` builds the carrier with **raw NumPy**. `docs/CURRENT_STATE.md`
records the consequence plainly: *"The current probe wraps a NumPy-built carrier
in `fnp`; it is not yet a grader-legal construction path."* The grader sandbox
has no NumPy, and hard rule 1 permits only stdlib, `flopscope`,
`flopscope.numpy as fnp` and `whestbench`. Wrapping a NumPy result is not legal
evidence and would not run at all on the grader.

So everything here is either **pure Python integer arithmetic** (the Galois
ring, the Teichmüller enumeration, the trace, the Gray map — no arrays at all)
or **`fnp`** (the Walsh matrix, the sign flip, the antipodal pairing, the
radial scale, the propagation). NumPy is imported nowhere.

THE STRUCTURE THAT MAKES IT CHEAP
---------------------------------
    B_lam[x, y] = N^(-1/2) * (-1)^(g_lam[x] XOR <x,y>)
                = N^(-1/2) * s_lam[x] * W[x, y]

with `W` the Walsh (Sylvester-Hadamard) matrix and `s_lam = (-1)^(g_lam)`. One
`W` is shared by every basis; a basis costs one length-`N` sign vector, not a
matrix. The Z4-Kerdock code `K(9)` has `4^10 = 1,048,576` words and **is never
enumerated** — only the `k` chosen `lambda` are touched.

THE THREE THINGS THAT MUST NOT DRIFT, AND ARE TESTED
-----------------------------------------------------
1. **The coordinate identification** (spec §4): position is `2*iota(x) + t` with
   `iota(x)` the ADDITIVE image of `x` in `R/2R ~ GF(2^m)`, read little-endian.
   Not `rank(x)`. The rank version passed within-basis orthogonality perfectly
   and produced ANF degree 9 instead of 2; it is preserved as an executable
   negative control in `kerdock_mub.bad_mapping_sign_vectors`.
2. **Row order**: basis-major, then column `y` ascending, then `+v` before `-v`.
3. **The radial factor is separate** (spec §8). A bias-free ReLU MLP is
   positively homogeneous of degree 1, so for `x ~ N(0, I_N)`
   `E[f(x)] = E[||x||] * E_{u ~ S^{N-1}}[f(u)]` **exactly**. The carrier
   estimates the angular factor on UNIT rows; the radial constant is a Python
   float computed by `lgamma` and validated on its own against the layer-1
   closed form. Conflating the two is how a correct carrier returns a wrong
   answer.

WHAT THIS MODULE DOES NOT CLAIM
-------------------------------
No accuracy, no score, no basis-count selection. Eight bases is a frozen panel
point, not an answer inferred from the public board. Selection is WP2's job and
is governed by `GAME_PLAN.md`; nothing here reads truth.

Run:  uv run python local_sim/p2_carrier_fnp.py
"""

from __future__ import annotations

import math

# NOTHING ELSE. No numpy, no scipy. `fnp` is passed in by the caller so this
# module can be exercised against the real `flopscope.numpy` and, in tests,
# against the shim -- and so a future single-file estimator can inline it
# without an import this repository controls.

#: Primitive binary polynomials by degree, coefficient lists (index = power).
#: Only ODD `m` is meaningful: CCKS give the REAL line-sets for odd `m`, and a
#: bent function on `F2^(m+1)` needs `m+1` even. `_assert_ring` checks both.
#: Byte-identical to `kerdock_mub.PRIMITIVE`; a divergence here would silently
#: build a different object that still passed every within-basis check.
PRIMITIVE = {
    3: [1, 1, 0, 1],                             # x^3 + x + 1
    5: [1, 0, 1, 0, 0, 1],                       # x^5 + x^2 + 1
    7: [1, 1, 0, 0, 0, 0, 0, 1],                 # x^7 + x + 1
    9: [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],           # x^9 + x^4 + 1
}


# --------------------------------------------------------------------------- #
# 1. the ring GR(4, m) — pure Python integers, no arrays
# --------------------------------------------------------------------------- #

def hensel_lift(f, m):
    """The basic primitive polynomial over Z4 lifting `f`, by the Graeff method.

    Writing `f(x) = e(x^2) + x*o(x^2)` over F2 with coefficients lifted to
    {0,1}, the lift satisfies `h(y) = +/-( e(y)^2 - y*o(y)^2 ) mod 4`,
    normalised monic. The result is not trusted: `_assert_ring` checks
    `h == f (mod 2)` and that `xi` has order `2^m - 1`, which is what "basic
    primitive" means and what a wrong lift would fail.
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
    for i, v in enumerate(o2):          # subtract y * o(y)^2
        if i + 1 <= m:
            g[i + 1] = (g[i + 1] - v) % 4
    if g[m] % 4 == 3:                    # normalise monic
        g = [(-c) % 4 for c in g]
    return g


class GaloisRing:
    """`GR(4, m) = Z4[x]/(h)` with the Teichmuller set and the trace.

    Elements are length-`m` tuples of Z4 coefficients. Only what the carrier
    needs is implemented.

    **One deliberate difference from `kerdock_mub.GaloisRing`, and it is a
    speed fix with no effect on the bytes:** that class calls `self.pw.index(x)`
    inside the per-basis loop, an `O(2^m)` scan run `2^m` times per basis --
    at `m = 9`, 262,144 list comparisons per basis and 4.2 million over sixteen.
    That is seconds of pure Python inside a **5-second setup ceiling** that also
    has to cover worker spawn and `import whestbench` (1.3 s quiet, 4.7-9.1 s
    under load). `self.idx` is the same map as a dict, built once in `O(2^m)`.
    `test_p2_carrier_fnp.py` asserts the two agree element for element.
    """

    def __init__(self, m):
        if m not in PRIMITIVE:
            raise ValueError("no primitive polynomial recorded for degree %d" % m)
        self.m = m
        self.n = m + 1
        self.N = 1 << self.n
        self.f = PRIMITIVE[m]
        self.h = hensel_lift(self.f, m)
        self.order = (1 << m) - 1
        self._build_powers()
        self._assert_ring()

    def _mul(self, a, b):
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

    def _add(self, a, b):
        return tuple((x + y) & 3 for x, y in zip(a, b))

    def _build_powers(self):
        m = self.m
        one = tuple([1] + [0] * (m - 1))
        xi = tuple([0, 1] + [0] * (m - 2)) if m > 1 else (1,)
        self.pw = [one]
        for _ in range(self.order - 1):
            self.pw.append(self._mul(self.pw[-1], xi))
        self.one, self.xi = one, xi
        self.zero = tuple([0] * m)
        #: element -> its exponent. The O(1) replacement for `pw.index`.
        self.idx = {e: i for i, e in enumerate(self.pw)}

    def _assert_ring(self):
        m = self.m
        if m % 2 == 0:
            raise ValueError("m must be ODD for the REAL line-sets; got %d" % m)
        if self.n % 2 != 0:
            raise ValueError("n = m+1 must be EVEN for a bent function; got %d"
                             % self.n)
        if [c & 1 for c in self.h] != self.f:
            raise AssertionError("Hensel lift does not reduce to f mod 2")
        if self._mul(self.pw[-1], self.xi) != self.one:
            raise AssertionError("xi^(2^m-1) != 1: h is not basic primitive")
        if len(set(self.pw)) != self.order:
            raise AssertionError("xi has order < 2^m - 1: h is not primitive")
        if len(self.idx) != self.order:
            raise AssertionError("the power index is not injective")

    def teichmuller(self):
        """`T = {0} U {xi^i}`, in the fixed enumeration the spec pins (§4)."""
        return [self.zero] + list(self.pw)

    def trace_of_power(self, k):
        """`T_R(xi^k)` in Z4. Asserts the result really is a Z4 constant."""
        acc = self.zero
        for i in range(self.m):
            acc = self._add(acc, self.pw[(k << i) % self.order])
        if any(c for c in acc[1:]):
            raise AssertionError("trace of xi^%d is not in Z4: %r" % (k, acc))
        return acc[0]


def gray(z):
    """`phi(0)=(0,0) phi(1)=(0,1) phi(2)=(1,1) phi(3)=(1,0)` — the standard map."""
    hi = z >> 1
    return hi, hi ^ (z & 1)


def sign_vectors_py(m, k=16, basis_indices=None):
    """`s_lam = (-1)^(g_lam)` for `k` bases, as a list of lists of Python ints.

    Shape `(k, N)`, values `+1 / -1`. No arrays: this is the part that must run
    where NumPy does not exist.

    THE COORDINATE IDENTIFICATION IS THE ONE PLACE A PLAUSIBLE WRONG CHOICE
    EXISTS. Position is `2 * iota(x) + t`, `iota(x)` the ADDITIVE image of `x`
    in `R/2R ~ GF(2^m)` read little-endian -- reduction mod 2 is a bijection
    `T -> GF(2^m)` and it is the map that respects the F2-linear structure the
    Walsh phase `<x, y>` needs. Indexing by the position of `x` in the
    MULTIPLICATIVE enumeration is linear in nothing; it produced Boolean
    functions of ANF degree 9 instead of 2 and a cross-basis spectrum wildly
    unlike a flat 32, **while within-basis orthogonality passed perfectly**.
    """
    R = GaloisRing(m)
    T = R.teichmuller()
    if basis_indices is None:
        if not 1 <= k <= len(T):
            raise ValueError("k must be in 1..%d; got %d" % (len(T), k))
        basis_indices = list(range(k))
    else:
        basis_indices = list(basis_indices)
        if len(set(basis_indices)) != len(basis_indices):
            raise ValueError("basis_indices must be distinct")
        if not all(0 <= b < len(T) for b in basis_indices):
            raise ValueError("basis_indices must lie in 0..%d" % (len(T) - 1))
        k = len(basis_indices)
    N = R.N

    pos = {}
    for x in T:
        bits = 0
        for i, c in enumerate(x):
            if c & 1:
                bits |= 1 << i
        pos[x] = bits
    if len(set(pos.values())) != len(T):
        raise AssertionError("T -> GF(2^m) is not injective; the ring is wrong")

    out = []
    for bi in basis_indices:
        lam = T[bi]
        lam_k = None if lam == R.zero else R.idx[lam]
        g = [0] * N
        for x in T:
            if lam == R.zero or x == R.zero:
                z = 0                       # T_R(0) = 0
            else:
                z = R.trace_of_power((lam_k + R.idx[x]) % R.order)
            hi, lo = gray(z)
            base = 2 * pos[x]
            g[base] = hi
            g[base + 1] = lo
        out.append([1 - 2 * v for v in g])
    return out


# --------------------------------------------------------------------------- #
# 2. the radial factor — separate, and separately validated (spec §8)
# --------------------------------------------------------------------------- #

def radial_factor(width):
    """`E||x||` for `x ~ N(0, I_width)`, as an EXACT Python float.

    `E||x|| = sqrt(2) * Gamma((N+1)/2) / Gamma(N/2)`, computed through `lgamma`
    because the two Gammas overflow far below `N = 1024` (`Gamma(512)` is about
    `1e1166`). Their RATIO is `~31.99`, perfectly representable; the naive
    quotient is `inf/inf`.

    This is a Python float, not a flopscope value: it is computed in `setup()`,
    costs no counted FLOP, and materialises no array scalar during `predict`.
    """
    return math.sqrt(2.0) * math.exp(
        math.lgamma((width + 1) / 2.0) - math.lgamma(width / 2.0))


# --------------------------------------------------------------------------- #
# 3. the Walsh matrix and the carrier, in fnp
# --------------------------------------------------------------------------- #

def walsh_fnp(fnp, n, dtype=None):
    """`W[x, y] = (-1)^<x,y>` on `F2^n`, by the Sylvester recursion, in `fnp`.

    `H_1 = [[1]]`,  `H_{2s} = [[H, H], [H, -H]]`. Ten doublings reach 1024.

    The recursion is used rather than a popcount-parity table because it is
    array-shaped: `n` concatenations and `n` negations, roughly `2 * N^2`
    counted element-ops in total (`~2.1e6` at `N = 1024`), against the
    `2^39`-`2^40` of the propagation it feeds. A per-element parity computed in
    Python would be `N^2 = 1,048,576` interpreter iterations inside the setup
    ceiling.
    """
    h = fnp.ones((1, 1), dtype=dtype) if dtype is not None else fnp.ones((1, 1))
    for _ in range(n):
        top = fnp.concatenate([h, h], axis=1)
        bot = fnp.concatenate([h, -h], axis=1)
        h = fnp.concatenate([top, bot], axis=0)
    return h


def carrier_fnp(fnp, m=9, k=16, antipodes=True, basis_indices=None,
                dtype=None, unit_norm=True):
    """The carrier rows, in `fnp`. Shape `(k*N*(2 if antipodes else 1), N)`.

    Row order is deterministic and stated: **basis-major, then column `y`
    ascending, then `+v` before `-v`** -- the same order as
    `kerdock_mub.carrier_int8`, which is the parity oracle.

    `unit_norm=True` scales to `+/- 1/sqrt(N)`, so each row is a unit vector and
    the radial factor of §8 composes exactly. `unit_norm=False` returns the raw
    `+/- 1` and is used only by the byte-parity test against the integer oracle.
    """
    n = m + 1
    N = 1 << n
    s_py = sign_vectors_py(m, k, basis_indices)
    kk = len(s_py)

    W = walsh_fnp(fnp, n, dtype=dtype)                       # (N, N)
    S = fnp.array(s_py, dtype=dtype) if dtype is not None else fnp.array(s_py)

    # basis lam, column y -> row  s_lam * W[:, y].  W is symmetric, so W[:, y]
    # is W[y, :]; taking ROWS keeps the result C-contiguous and keeps the row
    # order basis-major then y-ascending.
    rows = fnp.reshape(S[:, None, :] * W[None, :, :], (kk * N, N))

    if antipodes:
        # `+v` immediately followed by `-v`. Interleaving via a (R, 2, N) stack
        # and one reshape avoids the item assignment hard rule 1 forbids --
        # grader arrays are immutable and `both[0::2] = rows` is not available.
        rows = fnp.reshape(fnp.stack([rows, -rows], axis=1), (2 * kk * N, N))

    if unit_norm:
        rows = rows * _inv_sqrt_n(fnp, N, dtype)
    return rows


def _inv_sqrt_n(fnp, N, dtype=None):
    """`1/sqrt(N)` as a scalar of the carrier's own dtype.

    `N` is a power of two, so `1/sqrt(N)` is a power of two whenever `n` is even
    -- at `N = 1024` it is exactly `1/32`, representable with no rounding at all
    in any binary float. That is why the spec can call the entries `+/- 1/32`
    exactly, and why scaling cannot perturb byte parity at the live width.
    """
    v = 1.0 / math.sqrt(N)
    return fnp.array(v, dtype=dtype) if dtype is not None else fnp.array(v)


# --------------------------------------------------------------------------- #
# 4. streamed propagation
# --------------------------------------------------------------------------- #

def propagate_means(fnp, carrier, weights, chunk_rows=4096, radial=1.0,
                    dtype=None):
    """Push the carrier through every layer and accumulate per-layer means.

    Returns a `(depth, width)` array of estimated post-ReLU means, already
    multiplied by `radial`.

    The forward map is the network's own: `h <- max(0, h @ W_l)`, weights stored
    `(input, output)` -- `docs/KERDOCK_MUB_SPEC.md` and `phase2_manifest.json`
    both pin `h_l = max(0, h_{l-1} @ W_l)` with the row vector on the LEFT.

    **Chunking is a memory decision, not a convenience.** The full carrier at
    `k = 16` is `32,768 x 1024` float32 = 128 MiB, and every layer needs a
    second buffer of the same size, so an unchunked forward pass holds 256 MiB
    live on top of 64 MiB of weights. Chunking bounds it at
    `2 * chunk_rows * width * 4` bytes. It must also be EXACT: sums over a
    partition of the rows equal the sum over all of them, so the chunked result
    is the dense one up to float association only -- `test_p2_carrier_fnp.py`
    pins that, including a tail chunk that does not divide the row count.

    No scalar is materialised. The loop bounds are Python integers derived from
    `carrier.shape`, never from an array VALUE, so nothing here can force the
    synchronisation the 0.4 s residual cliff punishes.
    """
    n_rows = int(carrier.shape[0])
    depth = len(weights)
    width = int(carrier.shape[1])
    if chunk_rows is None or chunk_rows <= 0 or chunk_rows > n_rows:
        chunk_rows = n_rows

    zero = fnp.array(0.0, dtype=dtype) if dtype is not None else fnp.array(0.0)
    totals = None
    for start in range(0, n_rows, chunk_rows):
        stop = start + chunk_rows
        if stop > n_rows:
            stop = n_rows
        h = carrier[start:stop]
        layer_sums = []
        for li in range(depth):
            h = fnp.maximum(h @ weights[li], zero)
            layer_sums.append(fnp.sum(h, axis=0))
        block = fnp.stack(layer_sums)                       # (depth, width)
        totals = block if totals is None else totals + block

    out = totals * (radial / float(n_rows))
    if int(out.shape[0]) != depth or int(out.shape[1]) != width:
        raise AssertionError("propagate_means produced %r, expected %r"
                             % (out.shape, (depth, width)))
    return out


# --------------------------------------------------------------------------- #
# 5. the deployable bundle
# --------------------------------------------------------------------------- #

class CarrierPlan:
    """Everything `setup()` can precompute, and nothing it cannot.

    The carrier depends only on the WIDTH, so it is built once per submission
    rather than once per MLP -- which is the whole reason the grader's lifecycle
    calls `setup()` a single time, off-budget.
    """

    def __init__(self, fnp, width, k=8, antipodes=True, chunk_rows=4096,
                 dtype=None):
        n = width.bit_length() - 1
        if 1 << n != width:
            raise ValueError("carrier width must be a power of two; got %d" % width)
        m = n - 1
        if m not in PRIMITIVE:
            raise ValueError(
                "no Kerdock construction recorded for width %d (m = %d). "
                "Supported widths: %s" % (width, m,
                                          sorted((1 << (mm + 1)) for mm in PRIMITIVE)))
        self.width = width
        self.m = m
        self.k = k
        self.antipodes = antipodes
        self.chunk_rows = chunk_rows
        self.radial = radial_factor(width)
        self.carrier = carrier_fnp(fnp, m=m, k=k, antipodes=antipodes,
                                   dtype=dtype, unit_norm=True)
        self.n_rows = int(self.carrier.shape[0])
        self.dtype = dtype

    @property
    def distinct_lines(self):
        return self.k * self.width

    def forward_flops(self, depth):
        """Counted FLOPs of the propagation alone: `2 * R * n^2 * L`."""
        return 2 * self.n_rows * self.width * self.width * depth

    def predict(self, fnp, weights):
        return propagate_means(fnp, self.carrier, weights,
                               chunk_rows=self.chunk_rows,
                               radial=self.radial, dtype=self.dtype)

    def __repr__(self):
        return ("CarrierPlan(width=%d, m=%d, k=%d, antipodes=%s, rows=%d, "
                "lines=%d, radial=%.6f)"
                % (self.width, self.m, self.k, self.antipodes, self.n_rows,
                   self.distinct_lines, self.radial))


def main():
    import importlib
    import os
    import sys

    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    # APPEND, never insert: local_sim ships shims NAMED flopscope/whestbench.
    sys.path[:] = [p for p in sys.path
                   if os.path.abspath(p or ".") != here] + [root, here]
    fnp = importlib.import_module("flopscope.numpy")

    print("carrier construction, stdlib + fnp only")
    for m in (3, 5, 7, 9):
        N = 1 << (m + 1)
        k = min(4, 1 << m)
        s = sign_vectors_py(m, k)
        print("  m=%2d  N=%5d  |T|=%4d  MUBs=%4d  bases=%d  "
              "E||x||=%.6f" % (m, N, 1 << m, (1 << m) + 1, len(s),
                               radial_factor(N)))
    plan = CarrierPlan(fnp, 1024, k=8)
    print()
    print("  %r" % (plan,))
    print("  carrier %s %s  = %.1f MiB"
          % (plan.carrier.shape, plan.carrier.dtype,
             plan.n_rows * plan.width * 4 / 2 ** 20))
    print("  forward FLOPs at depth 16: %s (%.4f of 2**41)"
          % (format(plan.forward_flops(16), ","),
             plan.forward_flops(16) / 2 ** 41))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
