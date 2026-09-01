"""GENERATED research candidate — do NOT edit, and do NOT promote from here.

Built by `local_sim/p2_build_candidate.py` from:
  local_sim/p2_carrier_fnp.py
  local_sim/p2_candidate_architecture.py

Configuration frozen at build time: carrier
Build inputs (sha256 of each source, in order): 57af6d7d00dd f3e8805a17f4

This is a RESEARCH artifact for pricing and certification. Promotion into
`estimator.py` and the creation of a frozen `estimator_v*.py` archive are
separate STRATEGIC steps (`docs/GAME_PLAN.md` section 6) requiring a contract,
Samuel's prediction, a full preflight, a CONFIRM and an independent Skeptic.

Imports are `math`, `flopscope`, `flopscope.numpy` and
`whestbench.BaseEstimator` only -- the subset of AGENTS.md hard rule 1 this
architecture needs. Nothing from `local_sim` is imported: that is the property
this file exists to demonstrate.
"""

from __future__ import annotations

import math

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator


# -------------------------------------------------------------------------
# from p2_carrier_fnp.py
# -------------------------------------------------------------------------
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

def exact_layer1_means(fnp, w0, dtype=None):
    """`E[ReLU(w.x)] = ||w||_2 / sqrt(2 pi)` for `x ~ N(0, I)`. EXACT.

    `z_j = <W[:, j], x>` is exactly `N(0, ||W[:, j]||^2)` and
    `E[ReLU(N(0, s^2))] = s / sqrt(2 pi)`. No sampling, no closure, no
    approximation.

    **Why the carrier must not estimate this row.** Quadrature over a finite
    point set gets it to about 1% — measured at `9.27e-3` max relative at the
    live width with 8 bases, and `1.92e-2` at width 256 — because `ReLU` is not
    a polynomial and the 8-basis subset is provably not a 4-design
    (`docs/CURRENT_STATE.md`: the carrier-aligned fourth moment is `21.688x` the
    spherical target). Spending quadrature on a row that has a closed form
    throws away exactness for nothing.

    In **float64 regardless of the carrier's dtype**. This is the one row whose
    answer is exactly known, so its precision is worth buying: in float32 the
    sum of `n` squares lands about `4.4e-7` relative from the closed form, which
    is float rounding rather than method error but still three orders above what
    `edge_cases.py` section 8 asks of a row that has a closed form. It costs
    `2 * n^2` FLOPs and promotes the returned array to float64 — where the
    harness's in-window `materialise_predictions` bills `2` FLOPs per element,
    `32,768` at the live shape. Both terms together are `6e-8` of the branch.
    """
    w = fnp.astype(w0, "float64")
    v = fnp.sum(w * w, axis=0)
    return fnp.sqrt(v) * (1.0 / math.sqrt(2.0 * math.pi))


def propagate_means(fnp, carrier, weights, chunk_rows=4096, radial=1.0,
                    dtype=None, exact_layer1=True):
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

    if exact_layer1:
        # Row 0 has a CLOSED FORM. Replacing the quadrature estimate with it is
        # strictly more accurate, and it is what lets a quadrature branch pass
        # `edge_cases.py` section 8 -- an invariant that demands `1e-12` and is
        # right to, because that row is exactly computable. This is not a
        # test-passing device: the exact value is better by three orders.
        #
        # Built by CONCATENATION, never by item assignment: grader arrays are
        # immutable (hard rule 1) and `out[0] = ...` is not available.
        row0 = exact_layer1_means(fnp, weights[0], dtype=dtype)
        out = fnp.concatenate([fnp.reshape(row0, (1, width)), out[1:]], axis=0)

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

    #: float32 by default. The carrier is a set of DIRECTIONS, not a state: at
    #: `k = 8` it is 16,384 x 1024, which is 64 MiB in float32 and 128 MiB in
    #: float64, and every layer needs a second buffer of the same size. The
    #: propagation is also the whole cost of this branch, and `docs/GOTCHAS.md`
    #: records that flopscope 0.12's dtype pricing is NOT a universal 2x -- so
    #: this is a memory choice, and the FLOP consequence is measured, never
    #: assumed.
    def __init__(self, fnp, width, k=8, antipodes=True, chunk_rows=4096,
                 dtype="float32", exact_layer1=True):
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
        self.exact_layer1 = exact_layer1
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
                               radial=self.radial, dtype=self.dtype,
                               exact_layer1=self.exact_layer1)

    def __repr__(self):
        return ("CarrierPlan(width=%d, m=%d, k=%d, antipodes=%s, rows=%d, "
                "lines=%d, radial=%.6f)"
                % (self.width, self.m, self.k, self.antipodes, self.n_rows,
                   self.distinct_lines, self.radial))


# -------------------------------------------------------------------------
# from p2_candidate_architecture.py
# -------------------------------------------------------------------------
#: Configurations, in the order a report should list them.
CONFIGS = ("k2_diag", "carrier", "carrier_k2_zero", "carrier_k2_frozen",
           "augment", "augment_k3")

IMPLEMENTED = ("k2_diag", "carrier", "carrier_k2_zero")

_SQRT_2PI = math.sqrt(2.0 * math.pi)


class CandidateRefusal(RuntimeError):
    """A configuration that exists but must not silently produce a number."""


# --------------------------------------------------------------------------- #
# branch 1 — the analytic control
# --------------------------------------------------------------------------- #

def _k2_diag(fnp, flops, weights, width, depth, dtype):
    """Diagonal second-moment (K2) propagation. The CONTROL, not the baseline.

    `h_0 = x ~ N(0, I)`, `z_l = h_{l-1} @ W_l`, `h_l = max(0, z_l)`, weights
    stored `(input, output)`.

    Layer 1 is EXACT and needs no Gaussian map: `z_j ~ N(0, ||W[:,j]||^2)`, so
    `E[ReLU(z_j)] = ||W[:,j]|| / sqrt(2 pi)`. Later layers use the exact ReLU
    Gaussian moments under a per-neuron normal assumption:

        a          = mu / sd
        E[ReLU]    = mu * Phi(a) + sd * phi(a)
        E[ReLU^2]  = (mu^2 + var) * Phi(a) + mu * sd * phi(a)

    and propagate `var_z = var_h @ (W * W)`, which DROPS the off-diagonal
    covariance. That omission is the whole reason this is a control: 94.61% of
    K2's error is accumulated mean/variance-state drift
    (`docs/CURRENT_STATE.md`, P2-CP3R).

    `flops.stats.norm.cdf/pdf` are used for `Phi`/`phi`. Flopscope 0.12
    **promotes their float32 input to float64** and says so with a
    `FlopscopeWarning`; `docs/CURRENT_STATE.md` records that this is exactly how
    an "f32" arm stayed float64 in name only. The dtype of the returned state is
    therefore reported by the telemetry rather than assumed.
    """
    zero = fnp.array(0.0, dtype=dtype)
    tiny = fnp.array(1e-30, dtype=dtype)

    w0 = weights[0]
    var_z = fnp.sum(w0 * w0, axis=0)                     # (width,)
    sd = fnp.sqrt(fnp.maximum(var_z, tiny))
    m = sd / _SQRT_2PI                                   # exact, layer 1
    e2 = 0.5 * var_z                                     # E[ReLU(z)^2] = var/2
    v = fnp.maximum(e2 - m * m, zero)
    # The REPORTED row 0 comes from the shared float64 closed form, not from
    # this branch's working-precision copy of it. In float32 the same identity
    # lands ~2.3e-7 from exact -- rounding, not method error, but still three
    # orders above what a closed-form row should cost. Reusing
    # `exact_layer1_means` also keeps ONE implementation of the identity: the
    # state below still needs `sd` in the working dtype, and that is the only
    # reason the two lines above exist at all.
    rows = [exact_layer1_means(fnp, w0, dtype=dtype)]

    for li in range(1, depth):
        wl = weights[li]
        mu = m @ wl
        var = v @ (wl * wl)
        sd = fnp.sqrt(fnp.maximum(var, tiny))
        a = mu / sd
        Phi = flops.stats.norm.cdf(a)
        phi = flops.stats.norm.pdf(a)
        m = mu * Phi + sd * phi
        e2 = (mu * mu + var) * Phi + mu * sd * phi
        v = fnp.maximum(e2 - m * m, zero)
        rows.append(m)
    return fnp.stack(rows)


# --------------------------------------------------------------------------- #
# the candidate
# --------------------------------------------------------------------------- #

class ResearchCandidate:
    """One configurable, shape-generic Phase-2 candidate.

    `setup()` builds whatever depends only on the WIDTH — for the carrier
    branch, the whole point set — because the grader calls it once, off-budget.
    `predict()` is metered and must materialise no scalar.
    """

    def __init__(self, config="carrier", k=8, antipodes=True, chunk_rows=4096,
                 dtype="float32", blend=None, strict=False,
                 randomize=False, exact_layer1=True):
        if config not in CONFIGS:
            raise ValueError("unknown config %r; known: %s"
                             % (config, ", ".join(CONFIGS)))
        self.config = config
        self.k = k
        self.antipodes = antipodes
        self.chunk_rows = chunk_rows
        self.dtype = dtype
        self.strict = strict
        self.randomize = randomize
        #: Row 0 has a closed form; the carrier must not estimate it. See
        #: `p2_carrier_fnp.exact_layer1_means`. Off only for research
        #: comparisons of the pure quadrature.
        self.exact_layer1 = exact_layer1
        #: The carrier/analytic mixing weight. `None` means UNSET, and an unset
        #: coefficient must never be guessed.
        self.blend = blend
        self.plan = None
        self.fallback_reason = ""
        self.telemetry = {}

    # -- lifecycle --------------------------------------------------------- #
    def setup(self, context):
        fnp = _fnp()
        width = int(context.width)
        self.width = width
        self.depth = int(getattr(context, "depth", 0) or 0)

        if self.config in ("augment", "augment_k3"):
            raise CandidateRefusal(
                "config %r is declared but NOT IMPLEMENTED (GAME_PLAN WP3/WP4). "
                "It refuses rather than silently returning another branch's "
                "answer." % self.config)
        if self.config == "carrier_k2_frozen" and self.blend is None:
            raise CandidateRefusal(
                "config 'carrier_k2_frozen' has NO coefficient. GAME_PLAN "
                "section 3D requires a frozen hybrid's interaction to be "
                "preregistered, and no DESIGN contract exists. Picking a "
                "number here would fit a constant with no provenance.")

        if self._wants_carrier():
            try:
                self.plan = CarrierPlan(
                    fnp, width, k=self.k, antipodes=self.antipodes,
                    chunk_rows=self.chunk_rows, dtype=self.dtype,
                    exact_layer1=self.exact_layer1)
            except ValueError as exc:
                # A width with no Kerdock construction. Say so, loudly.
                self.fallback_reason = (
                    "no Kerdock construction at width %d (%s); fell back to "
                    "k2_diag" % (width, exc))
                if self.strict:
                    raise CandidateRefusal(
                        "strict=True and " + self.fallback_reason) from exc
                self.plan = None
        return self

    def predict(self, mlp, flop_budget):
        fnp, flops = _fnp(), _flops()
        weights = mlp.weights
        depth = len(weights)
        width = int(mlp.width)

        analytic = None
        carrier = None
        if self._use_carrier(width):
            car = self.plan.carrier
            if self.randomize:
                car = self._randomize(fnp, car, mlp)
            carrier = propagate_means(
                fnp, car, weights, chunk_rows=self.chunk_rows,
                radial=self.plan.radial, dtype=self.dtype,
                exact_layer1=self.exact_layer1)
        if self._use_analytic(width):
            analytic = _k2_diag(fnp, flops, weights, width, depth, self.dtype)

        if carrier is None and analytic is None:
            raise CandidateRefusal(
                "config %r produced no branch at width %d" % (self.config, width))
        if carrier is None:
            return analytic
        if analytic is None:
            return carrier
        w = 0.0 if self.blend is None else float(self.blend)
        return carrier * (1.0 - w) + analytic * w

    # -- policy ------------------------------------------------------------ #
    def _wants_carrier(self):
        return self.config in ("carrier", "carrier_k2_zero", "carrier_k2_frozen")

    def _use_carrier(self, width):
        return self.plan is not None and int(self.plan.width) == width

    def _use_analytic(self, width):
        if self.config == "k2_diag":
            return True
        if self.config in ("carrier_k2_zero", "carrier_k2_frozen"):
            return True
        return not self._use_carrier(width)      # the fallback

    def _randomize(self, fnp, car, mlp):
        """A measure-preserving sign flip derived ONLY from `mlp.seed`.

        The carrier is closed under negation, so flipping the sign of any subset
        of COLUMNS maps the point set to another valid one with the same Gram
        structure -- it is an isometry of the sphere that permutes nothing about
        the design's strength. `GAME_PLAN.md` § 3B permits exactly this.

        `fnp.random.default_rng(mlp.seed)` is the only randomness, per hard rule
        2, and the flip is applied as an array multiply, so no scalar is
        materialised. Default OFF: whether randomising helps is a DESIGN
        question with no contract yet.
        """
        rng = fnp.random.default_rng(mlp.seed)
        bits = rng.integers(0, 2, size=int(car.shape[1]))
        signs = fnp.astype(1 - 2 * bits, car.dtype)
        return car * signs

    # -- metadata ---------------------------------------------------------- #
    def cost_model(self, depth):
        """Counted FLOPs this configuration expects, as an explicit formula.

        Reported so a measurement can DISAGREE with it. The carrier term is
        exact: `2 * R * n^2 * L` for the propagation. The analytic term is an
        order estimate only -- `flops.stats.norm` pricing is not modelled here
        and must be measured.
        """
        out = {"config": self.config, "depth": depth, "carrier_flops": 0,
               "analytic_flops_order": 0, "exact": False}
        if self.plan is not None:
            out["carrier_flops"] = self.plan.forward_flops(depth)
            out["carrier_rows"] = self.plan.n_rows
            out["exact"] = True
        if self._use_analytic(getattr(self, "width", 0)):
            n = getattr(self, "width", 0)
            out["analytic_flops_order"] = 4 * n * n * max(depth - 1, 0)
            out["exact"] = False
        out["multiplier_floor_headroom"] = (
            None if not out["carrier_flops"]
            else 0.1 * (2 ** 41) - out["carrier_flops"])
        return out

    def describe(self):
        return {
            "config": self.config,
            "implemented": self.config in IMPLEMENTED,
            "k": self.k, "antipodes": self.antipodes,
            "chunk_rows": self.chunk_rows, "dtype": self.dtype,
            "blend": self.blend, "strict": self.strict,
            "randomize": self.randomize, "exact_layer1": self.exact_layer1,
            "carrier_rows": None if self.plan is None else self.plan.n_rows,
            "carrier_lines": None if self.plan is None else self.plan.distinct_lines,
            "radial": None if self.plan is None else self.plan.radial,
            "fallback_reason": self.fallback_reason,
        }

    def __repr__(self):
        return "ResearchCandidate(%s)" % ", ".join(
            "%s=%r" % kv for kv in sorted(self.describe().items()))


def _fnp():
    return fnp        # module-level


def _flops():
    return flops     # module-level

# --------------------------------------------------------------------------- #
# the grader's entry point
# --------------------------------------------------------------------------- #

_CONFIG = 'carrier'
_KWARGS = {'k': 8, 'chunk_rows': 4096, 'dtype': 'float32'}


class Estimator(BaseEstimator):
    """The grader lifecycle: one instance, `setup()` once off-budget.

    Subclassing `whestbench.BaseEstimator` is a CONTRACT REQUIREMENT, not a
    convention: `whest validate` refuses a file with
    `SCORING_VALIDATION_ERROR: No BaseEstimator subclasses found`. The first
    build of this footer defined a bare class and was correctly rejected --
    while every other section of the certification passed, which is exactly why
    section 1 runs the official validator instead of trusting the shape of the
    code.
    """

    def __init__(self):
        self._inner = ResearchCandidate(config=_CONFIG, **_KWARGS)

    def setup(self, context):
        self._inner.setup(context)

    def predict(self, mlp, flop_budget):
        return self._inner.predict(mlp, flop_budget)
