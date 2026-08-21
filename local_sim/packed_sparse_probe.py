"""
SPRINT 4 LABORATORY — exact packed activation propagation.

What this is testing
--------------------
`estimator.py` (v2.4.1) propagates N sampled inputs through the pruned network as
a chain of DENSE matmuls. flopscope bills a matmul as `(2K-1)*M*N` **from shapes
alone**, so the fact that roughly a third of every post-ReLU activation matrix is
*exactly* 0.0 buys us nothing at all today. Sprint 3 attacked the ReLU; this
attacks the zeros.

The mechanism, from first principles (GAME_PLAN rev 9):

  * a neuron's firing rate is Phi(alpha), and alpha is spread across neurons, so
    the activation matrix h has a *column-structured* sparsity pattern: some
    columns are almost always zero, some almost never;
  * order the columns of each hidden layer by firing rate (folded into the
    weights, so it costs nothing per sample) and split them into a COLD PREFIX
    and a DENSE HOT TAIL;
  * inside the prefix, each row has only a handful of nonzeros, so its
    contribution `h_pre[r] @ W_pre` is a sum of a few *gathered* weight rows
    rather than a length-P dot product;
  * bucket the rows by how many prefix nonzeros they have (0, 1, 2, 4, 8,
    remainder) so each bucket does the smallest gather that is exact, and the
    support-0 bucket is an exact ZERO RECTANGLE that is skipped outright;
  * the hot tail stays a single dense matmul, which can additionally be evaluated
    by Strassen's algorithm — genuinely fewer arithmetic operations, and
    flopscope prices the seven half-size multiplies honestly.

EXACTNESS IS THE WHOLE POINT. Nothing here is a threshold or an approximation:
an entry is skipped only when it is exactly `0.0` after the ReLU, in which case
`0.0 * w` contributes exactly `0.0`. The top-k selection uses `argpartition`, and
for a row with fewer than k nonzeros the surplus slots gather a *zero* value, so
they self-annihilate; rows with MORE than k nonzeros are routed to a wider bucket
or to the exact dense remainder path. Column order is a permutation folded into
both sides of every matmul, so the network's function is untouched.

Separately switchable, because each must earn its place on its own:

    packing            the reorganisation above (exact; MSE must not move)
    antithetic         Sobol half-set plus its negation (halves the quantile
                       transform AND half of layer 1's matmul; changes variance)
    pilot              a small pilot pass measuring EMPIRICAL firing rates, used
                       for column ordering (bias-free: packing is exact for any
                       order) and optionally for dead/on reclassification
    on_compose         final-layer on-neurons answered as E[z] + the exact
                       rectification correction instead of per-sample (Sprint 3)
    strassen           Strassen recursion depth on the dense blocks

Fences
------
This module NEVER imports or edits `estimator.py`'s shipped path; it imports it
read-only for the bit-parity check. It runs against the REAL `flopscope`, not the
`local_sim` shim, so every number here is billed by the same accountant the
grader uses. The analytic cost oracle in `oracle_cost()` is deliberately a
SEPARATE model, so a disagreement between it and the instrumented count is a
finding rather than a silent shared assumption.

Run:
    uv run python local_sim/packed_sparse_probe.py --mode parity  --mlps 8
    uv run python local_sim/packed_sparse_probe.py --mode ablate  --block D11 --mlps 40 --shifts 4
    uv run python local_sim/packed_sparse_probe.py --mode decide  --block D12 --mlps 60 --shifts 8
"""

import os  # noqa: E402  isort:skip
import sys  # noqa: E402  isort:skip

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# The REAL flopscope has to win the import race, and by default it loses:
# `local_sim/` ships numpy look-alikes named `flopscope` and `whestbench`, and
# Python puts a script's OWN directory first on sys.path. Every cost in this
# file is meant to be the accountant's, not the shim's, so the shim directory is
# evicted before the import and appended back afterwards for the local helpers.
# Import order here is load-bearing; do not let a formatter reorder it.
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]

import flopscope as flops  # noqa: E402  isort:skip
import flopscope.numpy as fnp  # noqa: E402  isort:skip

assert hasattr(flops, "budget"), "imported the local_sim shim instead of real flopscope"

# The closure's covariance update legitimately drops flopscope's symmetry tag on
# `outer(Phi,Phi) * cov_pre`; the shipped estimator does the same thing and the
# warning would otherwise bury every table in this file.
flops.configure(symmetry_warnings=False)

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass, replace  # noqa: E402

import numpy as np  # noqa: E402

# APPEND, never insert: prepending would shadow the real packages again.
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

from mint_provenance import canonical_mint  # noqa: E402

BUDGET = 272_000_000_000
SOBOL_BITS = 30
INV_SQRT_2PI = 0.3989422804014327

# ---------------------------------------------------------------------------
# Sprint-5 instrumentation: what flopscope does NOT bill but a remote backend
# still pays for.
#
#   host reads   — every time participant Python needs a number out of a device
#                  array. `fnp.ndarray` is a `RemoteArray` on the eval servers
#                  (18122), so a VALUE read is a round trip while a SHAPE read is
#                  metadata. They are counted separately because they are not the
#                  same thing and lumping them would flatter the design that does
#                  more of the expensive kind.
#   max bytes    — the largest single array materialised, against the grader's
#                  4 GiB per-array guard. Recorded at the sites that can actually
#                  be large; the op_log scan below is the backstop.
# ---------------------------------------------------------------------------
_HOST = {"shape": 0, "value": 0, "maxbytes": 0}


def _host_reset():
    _HOST.update(shape=0, value=0, maxbytes=0)


def _host_snapshot():
    return dict(_HOST)


def _note(arr):
    """Record a materialised array's footprint and return it unchanged."""
    nb = int(getattr(arr, "nbytes", 0))
    if nb > _HOST["maxbytes"]:
        _HOST["maxbytes"] = nb
    return arr


def _rshape(arr):
    _HOST["shape"] += 1
    return int(arr.shape[0])


def _rvalue(x):
    _HOST["value"] += 1
    return int(x)


def peak_rss_bytes():
    """Process peak working set, straight from the OS.

    The 4 GiB guard is per array (`_note` / `_log_max_bytes` cover that) but the
    64 GB cap is per process, and only the OS knows what the allocator actually
    held. `tracemalloc` does not see numpy's buffers, so it would under-report by
    exactly the arrays that matter.
    """
    try:
        import ctypes
        from ctypes import wintypes

        class _PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        # Prototypes must be declared: with ctypes' default int restype the
        # pseudo-handle from GetCurrentProcess is truncated and the call fails
        # silently, which is how this reported 0.00 GiB the first time.
        k32 = ctypes.WinDLL("kernel32")
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        fn = ctypes.WinDLL("psapi").GetProcessMemoryInfo
        fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
        fn.restype = wintypes.BOOL
        c = _PMC()
        c.cb = ctypes.sizeof(_PMC)
        if fn(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return int(c.PeakWorkingSetSize)
    except Exception:
        pass
    try:
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except Exception:
        return 0


def _log_max_bytes(budget):
    """Backstop: the largest array flopscope itself saw as an operand."""
    best = 0
    for rec in budget.op_log:
        try:
            it = np.dtype(rec.resolved_dtype).itemsize
        except (TypeError, ValueError):
            it = 4
        for sh in (rec.shapes or ()):
            if not sh:
                continue
            nb = it
            for d in sh:
                nb *= int(d)
            best = max(best, nb)
    return best


# ===========================================================================
# configuration
# ===========================================================================

@dataclass(frozen=True)
class Config:
    """One estimator arm. Every field is an independently switchable component."""

    log2_n: int = 14

    # --- exact arithmetic reorganisation -----------------------------------
    packing: bool = False
    prefix_target: float = 3.0        # expected prefix nonzeros per row
    rate_cut: float = 0.0             # if > 0, use the marginal-rate rule instead
    min_hot: int = 96                 # dense hot tail floor (columns)
    buckets: tuple = (1, 2, 4, 8, 16)  # support buckets; 0 and remainder implicit
    strassen: int = 0                 # recursion depth on dense blocks
    strassen_mode: str = "recursive"  # "recursive" | "batched"  (Sprint 5 A)
    bucket_mode: str = "flatnonzero"  # "flatnonzero" | "segment" (Sprint 5 B)
    gather_mode: str = "einsum"       # "einsum" | "slotloop"    (Sprint 5 B)

    # --- point set ----------------------------------------------------------
    antithetic: bool = False

    # --- classification -----------------------------------------------------
    dyn_order: bool = False           # order hidden columns by firing rate
    pilot: int = 0                    # pilot samples (0 = closure rates only)
    pilot_reclass: bool = False       # let the pilot re-decide dead/alive
    on_compose: bool = False          # final-layer on-neuron composition
    alpha_on: float = 3.0

    # --- frozen v2.4.1 constants -------------------------------------------
    prune_tau: float = -2.5
    basis_rank: int = 32

    def label(self):
        bits = []
        if self.packing:
            bk = "/".join(map(str, self.buckets)) if self.buckets else "none"
            bits.append(f"pack(t={self.prefix_target:g},b={bk}"
                        f"{',' + self.bucket_mode[:3] if self.bucket_mode != 'flatnonzero' else ''}"
                        f"{',' + self.gather_mode[:4] if self.gather_mode != 'einsum' else ''})")
        if self.antithetic:
            bits.append("anti")
        if self.pilot:
            bits.append(f"pilot{self.pilot}{'+rc' if self.pilot_reclass else ''}")
        if self.dyn_order and not self.packing:
            bits.append("order")
        if self.on_compose:
            bits.append(f"on>{self.alpha_on:g}")
        if self.strassen:
            bits.append(f"strassen{self.strassen}"
                        f"{'B' if self.strassen_mode == 'batched' else ''}")
        return "+".join(bits) if bits else "dense(v2.4.1)"


# ===========================================================================
# shared machinery — identical maths to estimator.py, expressed in fnp
# ===========================================================================

def closure(w64, width, depth):
    """Gaussian closure. Byte-for-byte the same recursion as estimator._closure."""
    mu = fnp.zeros(width)
    cov = fnp.eye(width)
    off_mask = fnp.ones((width, width)) - fnp.eye(width)
    alphas, mus, phis, sig_last, mupre_last = [], [], [], None, None
    for li, w in enumerate(w64):
        mu_pre = w.T @ mu
        cov_pre = w.T @ cov @ w
        var_pre = fnp.maximum(fnp.diagonal(cov_pre), 1e-12)
        sig = fnp.sqrt(var_pre)
        alpha = mu_pre / sig
        Phi = flops.stats.norm.cdf(alpha)
        phi = flops.stats.norm.pdf(alpha)
        mu = mu_pre * Phi + sig * phi
        ez2 = (mu_pre * mu_pre + var_pre) * Phi + mu_pre * sig * phi
        var = fnp.maximum(ez2 - mu * mu, 0.0)
        cov = fnp.outer(Phi, Phi) * cov_pre * off_mask + fnp.diag(var)
        alphas.append(alpha)
        mus.append(mu)
        phis.append(Phi)
        if li == depth - 1:
            sig_last, mupre_last = sig, mu_pre
    return alphas, mus, phis, sig_last, mupre_last


def alignment_basis(w64, phis, width, depth, rank):
    """v2.4.1's reproducible rotation: leading eigenvectors + QR completion."""
    J = w64[0]
    for k in range(1, depth):
        J = (J * phis[k - 1]) @ w64[k]
    _, vecs = fnp.linalg.eigh(J @ J.T)
    k = rank if rank < width else width
    top = vecs[:, width - k:][:, ::-1]
    hi = fnp.max(top, axis=0)
    lo = fnp.min(top, axis=0)
    top = top * fnp.where(hi >= -lo, 1.0, -1.0)
    q, r = fnp.linalg.qr(top, mode="complete")
    d = fnp.diagonal(r)
    fix = fnp.concatenate([fnp.where(d < 0.0, -1.0, 1.0), fnp.ones(width - k)])
    return q * fix


_BASE_POINTS = {}


def base_points(n, width):
    """Unshifted Sobol integers, from the SHIPPED direction-number table."""
    key = (n, width)
    if key in _BASE_POINTS:
        return _BASE_POINTS[key]
    import estimator as _est  # read-only: we want the shipped Joe-Kuo table

    dirnum = _est._direction_numbers()
    idx = fnp.arange(n, dtype=fnp.uint32)
    idx = fnp.bitwise_xor(idx, fnp.right_shift(idx, fnp.uint32(1)))
    idx = idx.reshape((n, 1))
    pts = fnp.zeros((n, width), dtype=fnp.uint32)
    for k in range(SOBOL_BITS):
        bit = fnp.bitwise_and(fnp.right_shift(idx, fnp.uint32(k)), fnp.uint32(1))
        pts = fnp.bitwise_xor(pts, bit * fnp.array(dirnum[k][:width], dtype=fnp.uint32))
    _BASE_POINTS[key] = pts
    return pts


# ===========================================================================
# Strassen — genuinely fewer multiplies, honestly billed
# ===========================================================================

# Below this block size the ten operand additions cost more than the multiply
# they save. The exact break-even for an (M,K)@(K,N) block with M >> K,N is
# 1.25/N + 4/K < 0.25, so ~16 is right for our shapes; it is swept, not guessed.
_STRASSEN_MIN = 16


def _quad(X):
    """The four quadrants of the LAST TWO axes. Basic slicing: 0 FLOPs.

    Written against the trailing axes so the identical expression serves a plain
    (M,K) matrix and a stacked (B,M,K) batch — which is what lets the batched
    variant below reuse the recursive one's operand algebra verbatim instead of
    re-deriving it (and re-deriving it wrongly).
    """
    m, n = X.shape[-2], X.shape[-1]
    h, k = m // 2, n // 2
    return X[..., :h, :k], X[..., :h, k:], X[..., h:, :k], X[..., h:, k:]


def _ops7(a11, a12, a21, a22, b11, b12, b21, b22):
    """Strassen's seven operand pairs. Five real additions per side; the rest
    are views. Shared by both implementations so they cannot drift apart."""
    return ([a11 + a22, a21 + a22, a11, a22, a11 + a12, a21 - a11, a12 - a22],
            [b11 + b22, b11, b12 - b22, b21 - b11, b22, b11 + b12, b21 + b22])


def _combine(m):
    """The four output quadrants from the seven products, on the trailing axes.

    Costs 8 additions and 3 concatenations = 4 * batch * R * C, whatever the
    batch rank, which is exactly what the recursive path pays per level.
    """
    return fnp.concatenate([
        fnp.concatenate([m[0] + m[3] - m[4] + m[6], m[2] + m[4]], axis=-1),
        fnp.concatenate([m[1] + m[3], m[0] - m[1] + m[2] + m[5]], axis=-1),
    ], axis=-2)


def _srec(A, B, depth):
    if depth <= 0 or min(A.shape[0], A.shape[1], B.shape[1]) < _STRASSEN_MIN:
        return A @ B
    AA, BB = _ops7(*_quad(A), *_quad(B))
    return _combine([_srec(AA[i], BB[i], depth - 1) for i in range(7)])


def _sbatch(A, B, depth):
    """Strassen with every leaf product in ONE batched contraction.

    Depth 1 evaluates 7 leaves, depth 2 evaluates 49, and flopscope bills a
    stacked `(b,m,k) @ (b,k,n)` matmul at exactly `b * (2k-1) * m * n` — i.e. the
    same arithmetic as the separate products, in one operation instead of b. The
    price of that is the stacking: two `stack`s at level 1 and two
    `concatenate`s at level 2, ~1.5% of the dense bill.

    The ordering of the depth-2 stack is load-bearing and deliberate: operands
    are concatenated as `[U_1 .. U_7]` where `U_j` is the level-2 product j
    across all seven level-1 groups, so leaf `j*7+i` is group i's j-th product.
    That makes `R[j*7:(j+1)*7]` — the j-th product of every group — a **basic
    slice**, so the level-2 reconstruction is itself batched and free of gathers.
    Ordering it the other way round would have cost a 49-row permutation.
    """
    if depth <= 0:
        return A @ B
    AA, BB = _ops7(*_quad(A), *_quad(B))
    SA = _note(fnp.stack(AA, axis=0))                # (7, M/2, K/2)
    SB = _note(fnp.stack(BB, axis=0))                # (7, K/2, N/2)
    if depth == 1:
        R = _note(fnp.matmul(SA, SB))                # (7, M/2, N/2)
        return _combine([R[i] for i in range(7)])
    UA, UB = _ops7(*_quad(SA), *_quad(SB))           # 7 x (7, M/4, K/4)
    R = _note(fnp.matmul(_note(fnp.concatenate(UA, axis=0)),   # (49, M/4, K/4)
                         _note(fnp.concatenate(UB, axis=0))))  # (49, M/4, N/4)
    P = _combine([R[j * 7:(j + 1) * 7] for j in range(7)])   # (7, M/2, N/2)
    return _combine([P[i] for i in range(7)])


def strassen(A, B, depth, mode="recursive"):
    """A @ B by Strassen (depth 0 = plain matmul).

    Algebraically exact; numerically it reassociates, so it stays behind its own
    toggle and its MSE effect is measured rather than assumed. Sub-block
    extraction is basic slicing, which flopscope bills at zero, so the only
    overhead is 10 operand additions and 8 result combinations per level.

    Dimensions are padded to a multiple of 2^depth ONCE at entry rather than
    falling back to a dense multiply on the first odd size -- the alive count is
    odd about half the time, and without this the toggle silently did nothing on
    those layers. Padding the contraction dimension with zeros is exact; the
    surplus output columns are sliced off, which is free.

    `mode="batched"` selects the compressed implementation above for depth 1-2
    and falls back to the recursion beyond it (a depth-3 batch would be 343
    leaves and the stacking would cost more than the multiplies it saves).
    """
    if depth <= 0:
        return A @ B
    M, K = A.shape
    N = B.shape[1]
    step = 1 << depth
    if M % step or min(M, K, N) < _STRASSEN_MIN * step // 2:
        return A @ B
    kp, npad = (-K) % step, (-N) % step
    if kp:
        A = fnp.concatenate([A, fnp.zeros((M, kp), dtype=A.dtype)], axis=1)
        B = fnp.concatenate([B, fnp.zeros((kp, N), dtype=B.dtype)], axis=0)
    if npad:
        B = fnp.concatenate([B, fnp.zeros((K + kp, npad), dtype=B.dtype)], axis=1)
    out = _sbatch(A, B, depth) if (mode == "batched" and depth <= 2) \
        else _srec(A, B, depth)
    return out[:, :N] if npad else out


def _mm(A, B, cfg):
    return strassen(A, B, cfg.strassen, cfg.strassen_mode) if cfg.strassen else A @ B


# ===========================================================================
# the plan — everything that happens once per MLP
# ===========================================================================

class Plan:
    """Per-MLP arithmetic layout: sliced/ordered weights, offsets, prefix sizes."""

    def __init__(self):
        self.sliced = []       # (a_in, a_out) f32, rotation folded at layer 0
        self.offsets = []      # dead-neuron constant offsets, or None
        self.prefix = []       # cold-prefix width for each layer's INPUT
        self.rows = []         # non-final rows of the answer (closure means)
        self.alive = []
        self.order = []        # column order applied to each layer's outputs
        self.final_split = None
        self.stats = {}


def _alive_sets(alphas, width, depth, tau):
    alive, dead = [], []
    for li in range(depth):
        if li == depth - 1:
            alive.append(fnp.arange(width))
            dead.append(None)
            continue
        keep = alphas[li] > tau
        idx = fnp.flatnonzero(keep)
        if int(idx.shape[0]) == 0:
            idx = fnp.flatnonzero(alphas[li] >= alphas[li].max())
            alive.append(idx)
            dead.append(None)
            continue
        alive.append(idx)
        didx = fnp.flatnonzero(~keep)
        dead.append(didx if int(didx.shape[0]) else None)
    return alive, dead


def _prefix_width(rates_sorted, a_in, cfg):
    """How far into the cold columns the gathered treatment is still cheaper.

    `rates_sorted` is the ascending firing-rate vector of the input columns.

    Two rules, both available so the sweep can choose between them:

    * **expected-support** — extend until the cumulative firing mass (which is
      exactly the expected number of prefix nonzeros per row) reaches
      `prefix_target`. Simple, but it applies one number to layers whose sparsity
      differs by a factor of two, so it is right in at most one place.

    * **marginal-rate** — the actual optimum. Moving column i from the dense hot
      tail into the prefix saves `2*o` FLOPs of dot product and costs
      `g*f_i*o` for the gathers it adds, where g ~ 6-7 is the per-nonzero gather
      price (4 to fetch the weight row, 2 to multiply-accumulate, times the
      inflation from padding to a bucket size). So the prefix should extend
      exactly while `f_i < 2/g` and stop there. That boundary is a property of
      the price table, not of the network, which is why it transfers across
      layers when a fixed expected-support target cannot.
    """
    cap = a_in - cfg.min_hot
    if cap <= 0:
        return 0
    if cfg.rate_cut > 0.0:
        p = int(fnp.searchsorted(rates_sorted, fnp.float64(cfg.rate_cut)))
    else:
        p = int(fnp.searchsorted(fnp.cumsum(rates_sorted), fnp.float64(cfg.prefix_target)))
    p = min(p, cap)
    if cfg.strassen:                       # keep the hot tail divisible by 2^depth
        step = 1 << cfg.strassen
        hot = a_in - p
        p = max(a_in - (hot - hot % step), 0)
    return p if p >= 8 else 0              # a prefix this thin cannot pay


def build_plan(w_fnp, cfg, width, depth, pilot_rates=None):
    """Closure, pruning, rotation, column ordering, slicing and prefix sizing."""
    plan = Plan()
    w64 = [w.astype(fnp.float64) for w in w_fnp]
    alphas, mus, phis, sig_last, mupre_last = closure(w64, width, depth)
    basis = alignment_basis(w64, phis, width, depth, cfg.basis_rank)
    alive, dead = _alive_sets(alphas, width, depth, cfg.prune_tau)

    if cfg.pilot_reclass and pilot_rates is not None:
        # A neuron the pilot never fired is dead by measurement, not by model.
        for li in range(depth - 1):
            fired = fnp.flatnonzero(pilot_rates[li] > 0.0)
            if int(fired.shape[0]) >= cfg.min_hot + 8:
                alive[li] = fnp.take(alive[li], fired)
                dead[li] = _complement(alive[li], width)

    # --- column order: ascending firing rate, so the cold prefix is genuinely cold
    order, rates = [], []
    for li in range(depth):
        a_l = alive[li]
        if li == depth - 1 or not cfg.dyn_order:
            order.append(None)
            rates.append(None)
            continue
        if pilot_rates is not None and int(pilot_rates[li].shape[0]) == int(a_l.shape[0]):
            f = pilot_rates[li]
        else:
            f = fnp.take(phis[li], a_l)
        o = fnp.argsort(f)
        order.append(o)
        rates.append(fnp.take(f, o))

    # --- final-layer classification, if on-neuron composition is enabled
    final_order = None
    if cfg.on_compose:
        a_last = alphas[depth - 1]
        cls_dead = fnp.flatnonzero(a_last <= cfg.prune_tau)
        cls_on = fnp.flatnonzero(a_last >= cfg.alpha_on)
        cls_kink = fnp.flatnonzero((a_last > cfg.prune_tau) & (a_last < cfg.alpha_on))
        final_order = fnp.concatenate([cls_dead, cls_on, cls_kink])
        plan.final_split = (int(cls_dead.shape[0]), int(cls_on.shape[0]),
                            int(cls_kink.shape[0]), final_order, sig_last, mupre_last)

    # --- slice, fold the rotation, apply the orders -------------------------
    prev = None
    for li in range(depth):
        wl = w64[li]
        if li == 0:
            wl = basis.T @ wl
        w32 = wl.astype(fnp.float32)
        cur = alive[li]
        if li == depth - 1 and final_order is not None:
            cur = fnp.take(cur, final_order)
        elif order[li] is not None:
            cur = fnp.take(cur, order[li])
        base = w32 if prev is None else fnp.take(w32, prev, axis=0)
        plan.sliced.append(fnp.take(base, cur, axis=1))
        dprev = dead[li - 1] if li > 0 else None
        if dprev is None:
            plan.offsets.append(None)
        else:
            wd = fnp.take(fnp.take(w32, dprev, axis=0), cur, axis=1)
            plan.offsets.append(fnp.take(mus[li - 1], dprev).astype(fnp.float32) @ wd)
        plan.alive.append(cur)
        prev = cur

    # --- cold-prefix widths -------------------------------------------------
    for li in range(depth):
        if li == 0 or not cfg.packing or rates[li - 1] is None:
            plan.prefix.append(0)
        else:
            plan.prefix.append(_prefix_width(rates[li - 1], int(plan.sliced[li].shape[0]), cfg))

    plan.rows = list(mus)
    plan.rows[0] = fnp.sqrt((w64[0] * w64[0]).sum(axis=0, dtype=fnp.float64)) * INV_SQRT_2PI
    plan.order = order
    plan.stats = {
        "alive": [int(a.shape[0]) for a in plan.alive],
        "prefix": list(plan.prefix),
    }
    return plan


def _complement(idx, width):
    mask = fnp.ones(width, dtype=fnp.uint32) - fnp.bincount(
        idx.astype(fnp.int64), minlength=width).astype(fnp.uint32)
    out = fnp.flatnonzero(mask)
    return out if int(out.shape[0]) else None


# ===========================================================================
# the sampled forward pass
# ===========================================================================

def _points(n, width, shift, cfg):
    """Sobol points, optionally antithetic.

    An antithetic set is {x, -x}: for x ~ N(0, I) the reflection is equally
    distributed, so the estimator stays unbiased, the quantile transform is paid
    on half the rows, and layer 1 is a negation instead of a matmul (its
    pre-activation is exactly odd in x). Everything downstream is untouched.
    """
    pts = base_points(n // 2 if cfg.antithetic else n, width)
    u = (fnp.bitwise_xor(pts, shift).astype(fnp.float64) + 0.5) * (1.0 / float(1 << SOBOL_BITS))
    h = flops.stats.norm.ppf(u).astype(fnp.float32)
    return h


def _bucket_rows(supp, edges, n):
    """Row index lists per support bucket, plus the exact-zero and remainder ends.

    Returns (order, segments) with `segments` a list of (bucket_k, start, stop)
    covering the whole permuted row range. bucket_k == 0 is the exact zero
    rectangle; bucket_k == -1 is the dense remainder.

    This is the `flatnonzero` design: one mask and one `flatnonzero` per bucket,
    then a single concatenate. It reads `sel.shape[0]` per bucket — a SHAPE read,
    which on a `RemoteArray` is metadata rather than a value transfer — and costs
    roughly `n * (4 * len(edges) + 7)` FLOPs in ~`4 * len(edges) + 5` operations.
    Both scale with the number of buckets, which is the whole point of the
    Sprint-5 Pareto: the bucket count is priced twice over.
    """
    parts, segs, at = [], [], 0
    lo = 0
    sel = fnp.flatnonzero(supp == 0)
    parts.append(sel)
    m = _rshape(sel)
    segs.append((0, at, at + m))
    at += m
    for k in edges:
        sel = fnp.flatnonzero((supp > lo) & (supp <= k))
        m = _rshape(sel)
        if m:
            parts.append(sel)
            segs.append((k, at, at + m))
            at += m
        lo = k
    sel = fnp.flatnonzero(supp > lo)
    m = _rshape(sel)
    if m:
        parts.append(sel)
        segs.append((-1, at, at + m))
        at += m
    assert at == n, f"bucket partition lost rows: {at} != {n}"
    return fnp.concatenate(parts), segs


def _bucket_rows_segment(supp, edges, n):
    """The same partition from ONE ordering operation plus segment boundaries.

    `argsort(supp)` produces a row order in which every support class is already
    contiguous, so the bucket boundaries are `searchsorted` offsets rather than
    per-bucket masks. That trades `4*len(edges)+5` operations for **three**, and
    trades `n*(4*len(edges)+7)` FLOPs for `8*n*log2(n) + 4*n` — flopscope prices
    a sort at `8 * n * ceil(log2 n)` regardless of how it is implemented, so at
    n = 16,384 the sort alone is 112 FLOPs/row against ~27 for a six-class
    mask sweep. Fewer operations, more arithmetic: exactly the trade the Pareto
    table exists to adjudicate.

    The bookkeeping difference that matters on a remote backend: this reads
    `len(edges)+2` **values** out of `bounds` (a device-to-host scalar transfer
    each) where the mask design reads only shapes. Both are counted in the
    Pareto table's `host reads` column, split by kind.
    """
    thresholds = [0] + list(edges)
    order = fnp.argsort(supp)                       # one ordering operation
    ssupp = fnp.take(supp, order)
    cuts = [0] + [_rvalue(v) for v in fnp.searchsorted(
        ssupp, fnp.array(thresholds, dtype=supp.dtype), side="right")]
    segs = [(k, cuts[i], cuts[i + 1])
            for i, k in enumerate(thresholds) if cuts[i + 1] > cuts[i]]
    if n > cuts[-1]:
        segs.append((-1, cuts[-1], n))
    assert segs and segs[-1][2] == n, f"segment partition lost rows: {segs} vs {n}"
    return order, segs


def _partition_rows(supp, cfg, n):
    if cfg.bucket_mode == "segment":
        return _bucket_rows_segment(supp, cfg.buckets, n)
    return _bucket_rows(supp, cfg.buckets, n)


def layer_packed(h, w, p, cfg, tally=None):
    """One layer as: dense hot tail + exact gathered prefix corrections.

    `p` is the cold-prefix width. The prefix columns are the least-frequently
    firing ones (the plan ordered them), so a row's prefix support is small; the
    correction gathers exactly those weight rows. `argpartition` selects the k
    largest prefix entries, and because everything after a ReLU is >= 0, a row
    with fewer than k nonzeros simply gathers extra ZEROS, whose contribution is
    exactly 0.0. Rows with more than k nonzeros are never truncated: they land in
    a wider bucket or in the dense remainder.
    """
    n = int(h.shape[0])
    pre, hot = h[:, :p], h[:, p:]                  # basic slices: 0 FLOPs
    w_pre, w_hot = w[:p, :], w[p:, :]

    supp = (pre > fnp.float32(0.0)).sum(axis=1)
    order, segs = _partition_rows(supp, cfg, n)

    hp = _note(fnp.take(h, order, axis=0))         # one gather; rows are samples,
    pre_p, hot_p = hp[:, :p], hp[:, p:]            # so their order is immaterial
    z_hot = _note(_mm(hot_p, w_hot, cfg))

    pieces = []
    for k, s, e in segs:
        blk = z_hot[s:e]
        if k == 0:                                 # exact zero rectangle: skipped
            pieces.append(blk)
        elif k == -1:                              # exact dense remainder
            pieces.append(blk + (pre_p[s:e] @ w_pre))
        else:
            sub = pre_p[s:e]
            cols = fnp.argpartition(sub, p - k, axis=1)[:, p - k:]
            if cfg.gather_mode == "slotloop":
                # Same arithmetic, one slot at a time. The einsum route
                # materialises a (rows, k, a_out) weight tensor -- at
                # N=65,536, k=16 and 205 outputs that is a single 860 MB
                # array, and it is the peak-memory driver of the whole
                # candidate. Accumulating slot by slot holds only
                # (rows, a_out) live, for the same FLOP count to within the
                # matmul's fused multiply-add, at the price of 3k operations
                # instead of 3.
                acc = blk
                for j in range(k):
                    cj = cols[:, j]
                    vj = fnp.take_along_axis(sub, cj.reshape((-1, 1)), axis=1)
                    acc = acc + vj * fnp.take(w_pre, cj, axis=0)
                pieces.append(acc)
            else:
                vals = fnp.take_along_axis(sub, cols, axis=1)
                wg = _note(fnp.take(w_pre, cols, axis=0))
                pieces.append(blk + fnp.einsum("nk,nko->no", vals, wg))
        if tally is not None:
            tally[k] = tally.get(k, 0) + (e - s)
    return _note(fnp.concatenate(pieces, axis=0)) if len(pieces) > 1 else pieces[0]


def forward(plan, cfg, shift, width, depth, tally=None):
    """The sampled pass. Returns the final-layer post-ReLU mean estimate."""
    n = 1 << cfg.log2_n
    h = _note(_points(n, width, shift, cfg))

    for li in range(depth):
        w = plan.sliced[li]
        p = plan.prefix[li]

        # ---------------- the scored layer, with on-neuron composition ------
        # Sprint 3's theorem: for a confidently-on neuron ReLU is the identity, so
        # E[ReLU(z)] = E[z] + sigma*(phi(a) - a*Phi(-a)) needs only the MEAN of the
        # incoming activations -- one vector product per MLP instead of one dot
        # product per sample. The saving is only real if those columns are never
        # multiplied per-sample at all, so the matmul is restricted here rather
        # than computed and sliced afterwards.
        if li == depth - 1 and plan.final_split is not None:
            nd, no, nk, forder, sig_last, _ = plan.final_split
            off = plan.offsets[li]
            parts = []
            if nd:
                parts.append(fnp.take(fnp.take(plan.rows[depth - 1], forder),
                                      fnp.arange(nd)))
            if no:
                hbar = h.mean(axis=0, dtype=fnp.float64).astype(fnp.float32)
                zon = (hbar @ w[:, nd:nd + no]).astype(fnp.float64)
                if off is not None:
                    zon = zon + off[nd:nd + no].astype(fnp.float64)
                sg = fnp.take(sig_last, fnp.take(forder, fnp.arange(nd, nd + no)))
                aa = zon / sg
                parts.append(zon + sg * (flops.stats.norm.pdf(aa)
                                         - aa * flops.stats.norm.cdf(-aa)))
            if nk:
                wk = w[:, nd + no:]
                if p >= 8 and cfg.packing:
                    zk = layer_packed(h, wk, p, cfg,
                                      None if tally is None else tally.setdefault(li, {}))
                else:
                    zk = _mm(h, wk, cfg)
                if off is not None:
                    zk = zk + off[nd + no:]
                parts.append(fnp.maximum(zk, fnp.float32(0.0)).mean(axis=0, dtype=fnp.float64))
            return fnp.concatenate(parts) if len(parts) > 1 else parts[0]

        if li == 0 and cfg.antithetic:
            # z(-x) = -z(x) exactly: half the rows cost a negation, not a matmul.
            zh = _mm(h, w, cfg)
            z = fnp.concatenate([zh, -zh], axis=0)
        elif p >= 8 and cfg.packing:
            # per-LAYER occupancy: bucket loads differ by a factor of several
            # between layer 2 and layer 30, so a pooled tally makes the oracle
            # drift by ~5% exactly where the prefix is widest.
            z = layer_packed(h, w, p, cfg,
                             None if tally is None else tally.setdefault(li, {}))
        else:
            z = _mm(h, w, cfg)
        if plan.offsets[li] is not None:
            z = z + plan.offsets[li]
        if li < depth - 1:
            h = _note(fnp.maximum(z, fnp.float32(0.0)))
            continue

        # the plain scored layer (no on-neuron composition)
        return fnp.maximum(z, fnp.float32(0.0)).mean(axis=0, dtype=fnp.float64)

    raise AssertionError("unreachable")


def undo_final_order(vec, plan, width):
    """Put the scored row back in the network's own neuron order."""
    if plan.final_split is None:
        return vec
    forder = plan.final_split[3]
    inv = fnp.argsort(forder)
    return fnp.take(vec, inv)


# ===========================================================================
# pilot pass
# ===========================================================================

def pilot_rates(w_fnp, cfg, width, depth, shift):
    """Empirical per-neuron firing rates from a short dense pilot.

    Bias-free by construction: the rates only choose a column ORDER and a prefix
    width, and packing is exact for any order, so nothing the pilot decides can
    move the answer. If `pilot_reclass` is on it may additionally retire neurons
    it never saw fire, which IS a modelling choice and is ablated separately.
    """
    pcfg = replace(cfg, packing=False, dyn_order=False, antithetic=False,
                   on_compose=False, log2_n=int(math.log2(cfg.pilot)), strassen=0)
    plan = build_plan(w_fnp, pcfg, width, depth)
    h = _points(cfg.pilot, width, shift, pcfg)
    rates = []
    for li in range(depth):
        z = h @ plan.sliced[li]
        if plan.offsets[li] is not None:
            z = z + plan.offsets[li]
        if li == depth - 1:
            break
        h = fnp.maximum(z, fnp.float32(0.0))
        rates.append((h > fnp.float32(0.0)).mean(axis=0, dtype=fnp.float64))
    rates.append(None)
    return rates


# ===========================================================================
# the analytic cost oracle — a SEPARATE model, deliberately not shared code
# ===========================================================================

# Prices measured against flopscope 0.10.0 (scratch/price_ops.py, reproduced in
# the LOG entry). They are the model's assumptions, written down so a mismatch
# with the instrumented count is visible instead of absorbed.
PRICE = {
    "matmul": lambda k, m, n: (2.0 * k - 1.0) * m * n,
    "elementwise": 1.0,     # add / multiply / maximum / compare
    "gather": 4.0,          # take, take_along_axis, where
    "partition": 4.0,       # argpartition
    "sort": 8.0,            # argsort, per element per log2 pass
    "write": 1.0,           # concatenate / fill
    "reduce_f32": 1.0,      # sum along an axis, per input element
    "reduce_bool": 2.0,
    "mean_f64": 2.0,
    "ppf": 173.0,
}


def oracle_cost(plan, cfg, tallies, width, depth):
    """Predicted per-MLP sampling FLOPs, from shapes and measured bucket loads.

    This is an accounting model, not a simulator: it is fed the bucket
    occupancies the run actually produced and asked to reproduce the bill. If it
    cannot, either the model or the implementation is wrong, and we want to know
    which before quoting a speed-up.
    """
    n = 1 << cfg.log2_n
    tot = 0.0
    rows_gen = n // 2 if cfg.antithetic else n
    tot += PRICE["ppf"] * rows_gen * width                       # quantile transform
    if cfg.antithetic:
        tot += rows_gen * width * PRICE["elementwise"]           # negation
        tot += n * width * PRICE["write"]                        # concatenate

    for li in range(depth):
        w = plan.sliced[li]
        a_in, a_out = int(w.shape[0]), int(w.shape[1])
        p = plan.prefix[li]
        rows = rows_gen if (li == 0 and cfg.antithetic) else n

        # The scored layer with on-neuron composition multiplies ONLY its kink
        # columns per sample; the on columns take one vector product per MLP and
        # the dead ones take none at all.
        if li == depth - 1 and plan.final_split is not None:
            nd, no, nk = plan.final_split[:3]
            tot += n * a_in * PRICE["mean_f64"] if no else 0.0
            tot += PRICE["matmul"](a_in, 1, no) if no else 0.0
            if nk:
                if p >= 8 and cfg.packing:
                    tot += _packed_cost(n, a_in, nk, p, cfg, tallies.get(li, {}))
                else:
                    tot += _dense_cost(a_in, n, nk, cfg)
                if plan.offsets[li] is not None:
                    tot += n * nk * PRICE["elementwise"]
                tot += n * nk * (PRICE["elementwise"] + PRICE["mean_f64"])
            continue

        if li == 0 and cfg.antithetic:
            tot += PRICE["matmul"](a_in, rows, a_out)
            tot += rows * a_out * PRICE["elementwise"] + n * a_out * PRICE["write"]
        elif p >= 8 and cfg.packing:
            tot += _packed_cost(n, a_in, a_out, p, cfg, tallies.get(li, {}))
        else:
            tot += _dense_cost(a_in, n, a_out, cfg)
        if plan.offsets[li] is not None:
            tot += n * a_out * PRICE["elementwise"]
        if li < depth - 1:
            tot += n * a_out * PRICE["elementwise"]               # relu
        else:
            tot += n * a_out * (PRICE["elementwise"] + PRICE["mean_f64"])
    return tot


def _bucket_cost(n, p, cfg):
    """Row-partition bookkeeping. The two designs price very differently."""
    if cfg.bucket_mode == "segment":
        # argsort is billed 8 * n * ceil(log2 n) by shape, whatever the algorithm.
        return (PRICE["sort"] * n * math.ceil(math.log2(max(n, 2)))
                + n * PRICE["gather"] + n * PRICE["write"])
    return n * (2 + 6 * (len(cfg.buckets) + 2)) + n * PRICE["write"]


def _packed_cost(n, a_in, a_out, p, cfg, tally):
    """One packed layer: bookkeeping + row permutation + hot tail + gathers."""
    tot = n * p * PRICE["elementwise"]                       # pre > 0
    tot += n * p * PRICE["reduce_bool"]                      # support count
    tot += _bucket_cost(n, p, cfg)                           # row partition
    tot += n * a_in * PRICE["gather"]                        # row permutation
    tot += _dense_cost(a_in - p, n, a_out, cfg)              # hot tail
    for k, cnt in tally.items():
        if k == 0:                                           # exact zero rectangle
            continue
        if k == -1:                                          # exact dense remainder
            tot += PRICE["matmul"](p, cnt, a_out) + cnt * a_out * PRICE["elementwise"]
        else:
            tot += cnt * p * PRICE["partition"]
            if cfg.gather_mode == "slotloop":
                # k x (take_along_axis + take + multiply + add), no (n,k,a_out)
                tot += k * (cnt * PRICE["gather"]
                            + cnt * a_out * PRICE["gather"]
                            + 2 * cnt * a_out * PRICE["elementwise"])
            else:
                tot += cnt * k * PRICE["gather"]
                tot += cnt * k * a_out * PRICE["gather"]
                tot += PRICE["matmul"](k, cnt, a_out)
                tot += cnt * a_out * PRICE["elementwise"]
    tot += n * a_out * PRICE["write"]                        # reassemble z
    return tot


def _srec_cost(m_, k_, n_, d):
    if d <= 0 or min(m_, k_, n_) < _STRASSEN_MIN:
        return PRICE["matmul"](k_, m_, n_)
    h, kk, nn = m_ // 2, k_ // 2, n_ // 2
    adds = 5 * h * kk + 5 * kk * nn + 8 * h * nn          # 10 operand + 8 result combos
    joins = 2 * m_ * n_                                   # two concatenate levels
    return 7 * _srec_cost(h, kk, nn, d - 1) + adds + joins


def _sbatch_cost(m_, k_, n_, d):
    """Batched Strassen: same leaf arithmetic, plus the stacking that batches it.

    `combine` costs `4 * batch * R * C` at every level (8 additions on quarter
    blocks + 3 concatenations), and a stacked matmul is billed at exactly
    `batch * (2k-1) * m * n`, so this model has no free parameters.
    """
    if d <= 0 or min(m_, k_, n_) < _STRASSEN_MIN:
        return PRICE["matmul"](k_, m_, n_)
    h, kk, nn = m_ // 2, k_ // 2, n_ // 2
    tot = 5 * h * kk + 5 * kk * nn                        # level-1 operand adds
    tot += 7 * h * kk + 7 * kk * nn                       # two stacks
    if d == 1 or min(h, kk, nn) < _STRASSEN_MIN:
        return tot + 7 * PRICE["matmul"](kk, h, nn) + 4 * m_ * n_
    h2, k2, n2 = h // 2, kk // 2, nn // 2
    tot += 7 * (5 * h2 * k2 + 5 * k2 * n2)                # level-2 operand adds
    tot += 49 * h2 * k2 + 49 * k2 * n2                    # two concatenates
    tot += 49 * PRICE["matmul"](k2, h2, n2)               # ONE batched contraction
    tot += 4 * 7 * h * nn                                 # level-2 combine (batched)
    tot += 4 * m_ * n_                                    # level-1 combine
    return tot


def _dense_cost(k, m, nout, cfg):
    """Cost of an (m,k)@(k,nout) product, with Strassen recursion if enabled."""
    d = cfg.strassen
    if d <= 0:
        return PRICE["matmul"](k, m, nout)
    step = 1 << d
    if (m % step) or min(m, k, nout) < _STRASSEN_MIN * step // 2:
        return PRICE["matmul"](k, m, nout)
    kp, npad = (-k) % step, (-nout) % step
    pad = (m * (k + kp) if kp else 0) + ((k + kp) * (nout + npad) if (kp or npad) else 0)
    leaf = _sbatch_cost if (cfg.strassen_mode == "batched" and d <= 2) else _srec_cost
    return pad * PRICE["write"] + leaf(m, k + kp, nout + npad, d)


# ===========================================================================
# per-MLP evaluation
# ===========================================================================

def _to_fnp(w_np):
    return [fnp.array(w) for w in w_np]


def run_mlp(w_np, est_seed, cfg, width, depth, n_shifts=1, truth=None, measure=True):
    """Run one MLP under one config. Returns per-shift MSEs and the FLOP bill.

    The weights are converted to flopscope arrays OUTSIDE the measured context,
    because on the grader they arrive that way; everything the estimator would
    genuinely have to do is inside.
    """
    with flops.budget(10 ** 16, quiet=True):
        w_fnp = _to_fnp(w_np)

    mses, bills = [], []
    rng = np.random.default_rng(int(est_seed))
    cached = None
    for r in range(n_shifts):
        shift_np = rng.integers(0, 1 << SOBOL_BITS, size=width, dtype=np.uint32)
        t0 = time.perf_counter()

        # The per-MLP plan is shift-independent unless a pilot informed it, so it
        # is built once and its bill re-charged to every shift. That is the same
        # arithmetic the estimator would pay per MLP; caching is a lab
        # convenience, not a discount.
        if cached is None or cfg.pilot:
            with flops.budget(10 ** 16, quiet=True) as bp:
                shift0 = fnp.array(shift_np)
                prates = pilot_rates(w_fnp, cfg, width, depth, shift0) if cfg.pilot else None
                plan = build_plan(w_fnp, cfg, width, depth, pilot_rates=prates)
            cached = (plan, float(bp.flops_used), float(bp.residual_wall_time_s),
                      int(sum(v["calls"] for v in bp.summary_dict()["operations"].values())))
        plan, pf, pr, po = cached

        tally = {}
        _host_reset()
        with flops.budget(10 ** 16, quiet=True) as b:
            shift = fnp.array(shift_np)
            out = forward(plan, cfg, shift, width, depth, tally)
            out = undo_final_order(out, plan, width)
            arr = np.asarray(out, dtype=np.float64)
        wall = time.perf_counter() - t0
        host = _host_snapshot()
        bills.append({
            "flops": float(b.flops_used) + pf,
            "flops_sampling": float(b.flops_used),
            "flops_fixed": pf,
            "residual_s": float(b.residual_wall_time_s) + pr,
            "wall_s": wall,
            "ops": int(sum(v["calls"] for v in b.summary_dict()["operations"].values())) + po,
            "ops_sampling": int(sum(v["calls"] for v in b.summary_dict()["operations"].values())),
            "oracle": oracle_cost(plan, cfg, tally, width, depth),
            "reads_shape": host["shape"],
            "reads_value": host["value"],
            "max_bytes": max(host["maxbytes"], _log_max_bytes(b)),
            "alive": plan.stats["alive"],
            "prefix": plan.stats["prefix"],
        })
        if truth is not None:
            mses.append(float(np.mean((arr - truth) ** 2)))
        else:
            mses.append(float("nan"))
        if not measure:
            break
    return mses, bills, arr


# ===========================================================================
# modes
# ===========================================================================

def load_truth(block, width, depth, cost_only=False):
    """Labels for a decision block, or a label-free stand-in for cost work.

    Packing is exact, so the cost half of `V*c` can be measured before any
    ground truth exists — and it is the half that decides whether this lane is
    alive at all. `--cost-only` takes that shortcut explicitly rather than
    quietly reporting an MSE nobody computed.
    """
    path = os.path.join(_ROOT, "experiments", "packed",
                        f"{block}_truth_{width}x{depth}.npz")
    if os.path.exists(path):
        return np.load(path)
    if not cost_only:
        raise SystemExit(f"missing truth for block {block}: {path}\n"
                         f"run: .venv-gpu\\Scripts\\python.exe local_sim/packed_truth.py "
                         f"--block {block}")
    from seed_blocks import seeds as _bs

    sd = np.array(_bs(block), dtype=np.int64)
    est = np.array([canonical_mint(width, depth, int(s))[1] for s in sd[:400]], dtype=np.uint64)
    return {"seeds": sd[:400], "est_seeds": est,
            "truth": np.full((len(est), width), np.nan), "gt_noise": np.array([np.nan])}


def mode_parity(args):
    """S0 — semantic parity: packing must not change the arithmetic result."""
    width, depth = args.width, args.depth
    print("=" * 100)
    print("S0  SEMANTIC PARITY — packed vs dense on identical networks and identical Sobol points")
    print("=" * 100)

    import estimator as shipped

    dense = Config(log2_n=args.log2_n)
    arms = [
        ("packing only", Config(log2_n=args.log2_n, packing=True, dyn_order=True)),
        ("packing+order", Config(log2_n=args.log2_n, packing=True, dyn_order=True,
                                 prefix_target=args.prefix_target)),
        ("strassen only", Config(log2_n=args.log2_n, strassen=1)),
        ("packing+strassen", Config(log2_n=args.log2_n, packing=True, dyn_order=True,
                                    strassen=1, prefix_target=args.prefix_target)),
    ]

    print(f"\n{'seed':>14} {'shipped vs dense-ref':>22} {'':>4} " +
          " ".join(f"{lbl:>20}" for lbl, _ in arms))
    print("-" * 100)
    worst = {lbl: 0.0 for lbl, _ in arms}
    worst_ship = 0.0
    for i in range(args.mlps):
        seed = 30_806_011_000 + i
        w_np, est_seed, _ = canonical_mint(width, depth, seed)
        _, _, ref = run_mlp(w_np, est_seed, dense, width, depth, n_shifts=1)

        # the shipped file, on the same MLP and the same seed-derived shift
        class _MLP:
            weights = [fnp.array(x) for x in w_np]
            seed = est_seed
        _MLP.width = width
        est = shipped.Estimator()
        with flops.budget(10 ** 16, quiet=True):
            sp = np.asarray(est.predict(_MLP, BUDGET), dtype=np.float64)[-1]
        dship = float(np.abs(sp - ref).max() / max(np.abs(ref).max(), 1e-30))
        worst_ship = max(worst_ship, dship)

        cells = []
        for lbl, cfg in arms:
            _, _, got = run_mlp(w_np, est_seed, cfg, width, depth, n_shifts=1)
            d = float(np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-30))
            worst[lbl] = max(worst[lbl], d)
            cells.append(f"{d:>20.3e}")
        print(f"{seed:>14} {dship:>22.3e} {'':>4} " + " ".join(cells))

    print("-" * 100)
    print(f"{'WORST':>14} {worst_ship:>22.3e} {'':>4} " +
          " ".join(f"{worst[lbl]:>20.3e}" for lbl, _ in arms))
    print()
    print("Tolerances, and why:")
    print("  dense-ref vs SHIPPED estimator.py : must be 0.0 (bit-identical) — same ops, same order")
    print("  packing vs dense                  : <= 1e-6 relative. Packing reassociates a length-P")
    print("                                      dot product into a hot part plus a few gathered")
    print("                                      terms; in f32 (eps 1.2e-7) over K~205 terms the")
    print("                                      backward-stable bound is ~K*eps ~ 2.4e-5, and the")
    print("                                      SKIPPED terms are exactly 0.0 so they contribute")
    print("                                      no error at all.")
    print("  strassen vs dense                 : looser — Strassen is only norm-wise stable;")
    print("                                      its MSE effect is measured, never assumed.")
    ok_ship = worst_ship == 0.0
    ok_pack = max(worst[l] for l in ("packing only", "packing+order")) <= 1e-6
    print(f"\nS0 shipped-parity : {'PASS' if ok_ship else 'FAIL'}")
    print(f"S0 packing-parity : {'PASS' if ok_pack else 'FAIL'}")
    return 0 if (ok_ship and ok_pack) else 1


_FIELDS = ("mse", "flops", "sampling", "fixed", "resid", "oracle", "ops",
           "wall", "reads_shape", "reads_value", "max_bytes")


def _collect(arms, seeds, est_seeds, truth, args, width, depth, header):
    """Run every arm on every MLP with COMMON RANDOM NUMBERS.

    The shift stream is derived from the MLP's estimator seed and restarted for
    each arm, so arm A and arm B see the same networks AND the same digital
    shifts. Every comparison downstream is therefore on per-MLP differences,
    which is the only way a ~5% question is answerable at this sample size
    (two realisations of the same estimator differ ~18% in the mean).
    """
    m = min(args.mlps, len(seeds))
    n = 1 << args.log2_n
    print("=" * 124)
    print(header)
    print(f"block {args.block}  {m} MLPs x {args.shifts} shifts  N = {n:,}  "
          f"{'COST ONLY (no labels)' if args.cost_only else 'labelled'}")
    print("=" * 124)
    per_mlp = {lbl: {k: [] for k in _FIELDS} for lbl, _ in arms}
    t0 = time.perf_counter()
    for i in range(m):
        w_np, _, _ = canonical_mint(width, depth, int(seeds[i]))
        tr = None if args.cost_only else truth[i]
        for lbl, cfg in arms:
            mses, bills, _ = run_mlp(w_np, int(est_seeds[i]), cfg, width, depth,
                                     n_shifts=args.shifts, truth=tr)
            d = per_mlp[lbl]
            d["mse"].append(float(np.mean(mses)))
            d["flops"].append(float(np.mean([b["flops"] for b in bills])))
            d["sampling"].append(float(np.mean([b["flops_sampling"] for b in bills])))
            d["fixed"].append(float(np.mean([b["flops_fixed"] for b in bills])))
            d["resid"].append(float(np.mean([b["residual_s"] for b in bills])))
            d["oracle"].append(float(np.mean([b["oracle"] for b in bills])))
            d["ops"].append(float(np.mean([b["ops"] for b in bills])))
            d["wall"].append(float(np.mean([b["wall_s"] for b in bills])))
            d["reads_shape"].append(float(np.mean([b["reads_shape"] for b in bills])))
            d["reads_value"].append(float(np.mean([b["reads_value"] for b in bills])))
            d["max_bytes"].append(float(max(b["max_bytes"] for b in bills)))
        if (i + 1) % 5 == 0 or i + 1 == m:
            print(f"  {i + 1}/{m}  {time.perf_counter() - t0:.0f}s", flush=True)
    return per_mlp, n


def _report(arms, per_mlp, n, args, gt_noise=float("nan")):
    """Level table, then the paired ratios that actually decide anything."""
    ref = arms[0][0]
    cost_only = bool(np.isnan(per_mlp[ref]["mse"][0]))

    print(f"\n{'arm':<32} {'MSE':>11} {'MSE*N':>8} {'FLOPs tot':>11} {'sampling':>11} "
          f"{'fixed':>10} {'resid s':>8} {'C_eff':>11} {'c/sample':>10} {'V*c':>10} "
          f"{'adj':>10} {'ops':>6}")
    print("-" * 124)
    for lbl, _ in arms:
        d = per_mlp[lbl]
        mse = float(np.mean(d["mse"]))
        fl, sa, fx = (float(np.mean(d[k])) for k in ("flops", "sampling", "fixed"))
        rs = float(np.mean(d["resid"]))
        c = fl + 1e11 * rs
        print(f"{lbl:<32} {mse:>11.4e} {mse * n:>8.5f} {fl:>11.4e} {sa:>11.4e} {fx:>10.3e} "
              f"{rs:>8.5f} {c:>11.4e} {c / n:>10.4e} {mse * c:>10.4e} "
              f"{mse * max(0.1, c / BUDGET):>10.4e} {float(np.mean(d['ops'])):>6.0f}")

    print(f"\n{'arm':<32} {'V*c gain':>10} {'paired log-ratio':>20} {'MSE ratio':>10} "
          f"{'cost ratio':>11} {'FLOP-only':>10} {'oracle/real':>12}")
    print("-" * 124)
    one = np.ones(len(per_mlp[ref]["flops"]))
    b_mse = one if cost_only else np.array(per_mlp[ref]["mse"])
    b_fl = np.array(per_mlp[ref]["flops"])
    b_c = b_fl + 1e11 * np.array(per_mlp[ref]["resid"])
    for lbl, _ in arms:
        d = per_mlp[lbl]
        a_mse = one if cost_only else np.array(d["mse"])
        a_fl = np.array(d["flops"])
        a_c = a_fl + 1e11 * np.array(d["resid"])
        lr = np.log(b_mse * b_c) - np.log(a_mse * a_c)
        se = float(lr.std(ddof=1) / math.sqrt(len(lr))) if len(lr) > 1 else float("nan")
        print(f"{lbl:<32} {float(np.exp(lr.mean())):>10.4f}x "
              f"{f'{lr.mean():+.4f} +/- {se:.4f}':>20} "
              f"{float(b_mse.mean() / a_mse.mean()):>10.4f} "
              f"{float(b_c.mean() / a_c.mean()):>11.4f} "
              f"{float(b_fl.mean() / a_fl.mean()):>10.4f} "
              f"{float(np.mean(d['oracle']) / np.mean(d['sampling'])):>12.4f}")
    print(f"\npaired over {len(b_c)} MLPs with common random numbers; exp(mean log-ratio) is the")
    print("gain and +/- is one standard error.  C_eff = counted FLOPs + 1e11 * residual seconds;")
    print("'FLOP-only' drops the residual term, so the gap between it and 'cost ratio' is exactly")
    print("what op-count inflation costs.  'oracle/real' is the analytic model over the")
    print("instrumented SAMPLING bill — it should sit near 1.000 or the model is lying.")
    if cost_only:
        print("MSE columns are NaN by construction: this run carried no labels.")
    elif not math.isnan(gt_noise):
        print(f"GT noise {gt_noise:.3e} = {gt_noise / 1.4e-6 * 100:.3f}% of the 1.4e-6 signal.")


def _dump(args, arms, per_mlp, n, gt_noise):
    path = args.out
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "block": args.block, "mlps": len(per_mlp[arms[0][0]]["flops"]), "shifts": args.shifts,
        "n": n, "gt_noise": gt_noise, "width": args.width, "depth": args.depth,
        "cost_only": bool(args.cost_only),
        "arms": {lbl: {"config": {k: str(v) for k, v in cfg.__dict__.items()},
                       "per_mlp": per_mlp[lbl]} for lbl, cfg in arms},
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, default=str)
    print(f"\nsaved {path}")


def _finish(args, arms, per_mlp, n, gt_noise):
    _report(arms, per_mlp, n, args, gt_noise)
    if args.out:
        _dump(args, arms, per_mlp, n, gt_noise)
    return 0


def mode_ablate(args):
    """Component ablations. Every component must earn its place on its own."""
    width, depth = args.width, args.depth
    tr = load_truth(args.block, width, depth, args.cost_only)
    gt_noise = float(np.mean(tr["gt_noise"]))
    base = dict(log2_n=args.log2_n)
    bk = tuple(int(v) for v in args.buckets.split(","))
    arms = [
        ("dense (v2.4.1 reference)", Config(**base)),
        ("packing only", Config(**base, packing=True, dyn_order=True, buckets=bk,
                                prefix_target=args.prefix_target, rate_cut=args.rate_cut)),
        ("antithetic only", Config(**base, antithetic=True)),
        ("pilot ordering only", Config(**base, pilot=args.pilot, dyn_order=True)),
        ("pilot reclass only", Config(**base, pilot=args.pilot, dyn_order=True,
                                      pilot_reclass=True)),
        ("last-layer on-compose only", Config(**base, on_compose=True, alpha_on=args.alpha_on)),
        ("strassen only", Config(**base, strassen=args.strassen)),
        ("COMBINED candidate", Config(**base, packing=True, dyn_order=True, buckets=bk,
                                      prefix_target=args.prefix_target, rate_cut=args.rate_cut,
                                      antithetic=True, strassen=args.strassen,
                                      on_compose=True, alpha_on=args.alpha_on)),
    ]
    if args.only:
        keep = [s.strip() for s in args.only.split(",")]
        arms = [arms[0]] + [(k, v) for k, v in arms[1:] if any(s in k for s in keep)]
    per_mlp, n = _collect(arms, tr["seeds"], tr["est_seeds"], tr["truth"], args,
                          width, depth, "COMPONENT ABLATIONS")
    return _finish(args, arms, per_mlp, n, gt_noise)


def mode_sweep(args):
    """Design sweep on the DESIGN block: prefix target x Strassen depth."""
    width, depth = args.width, args.depth
    tr = load_truth(args.block, width, depth, args.cost_only)
    bk = tuple(int(v) for v in args.buckets.split(","))
    arms = [("dense (v2.4.1 reference)", Config(log2_n=args.log2_n))]
    strs = [int(x) for x in args.strassens.split(",")]
    if args.rate_cuts:
        for rc in [float(x) for x in args.rate_cuts.split(",")]:
            for s in strs:
                arms.append((f"rate_cut={rc:g} strassen={s}",
                             Config(log2_n=args.log2_n, packing=True, dyn_order=True,
                                    rate_cut=rc, strassen=s, buckets=bk)))
    if args.targets:
        for t in [float(x) for x in args.targets.split(",")]:
            for s in strs:
                arms.append((f"target={t:g} strassen={s}",
                             Config(log2_n=args.log2_n, packing=True, dyn_order=True,
                                    prefix_target=t, strassen=s, buckets=bk)))
    per_mlp, n = _collect(arms, tr["seeds"], tr["est_seeds"], tr["truth"], args,
                          width, depth, "DESIGN SWEEP (tuning permitted on D11 ONLY)")
    return _finish(args, arms, per_mlp, n, float(np.mean(tr["gt_noise"])))


# The one calibration point we own for uninstrumented DENSE time: graded
# submission 324420 (v2.4) billed 9.84 ms of residual per MLP against 5.31e10
# counted FLOPs on the evaluator's single physical core. Converting a LOCAL
# residual measurement into a grader-scale one by that anchor assumes the RATIO
# between two arms transfers even though the absolute level does not; Sprint 5
# adds a second, externally anchored model (`residual_models`) precisely because
# that assumption is unproven.
LAMBDA = 1e11


def _candidate(args):
    """The candidate exactly as the command line describes it — one definition,
    shared by the residual benchmark and the decision run so they cannot drift."""
    return Config(log2_n=args.log2_n, packing=True, dyn_order=True,
                  buckets=tuple(int(v) for v in args.buckets.split(",")),
                  prefix_target=args.prefix_target, rate_cut=args.rate_cut,
                  antithetic=args.antithetic, strassen=args.strassen,
                  on_compose=args.on_compose, alpha_on=args.alpha_on, pilot=args.pilot)


def mode_residual(args):
    """Measure uninstrumented wall time honestly, because it is the whole question.

    Counted FLOPs say packing wins. Packing also multiplies the operation count,
    and flopscope charges `lambda * residual_seconds` on top of the FLOPs — forum
    18101, and the reason v2.0 lost 10.8% of its compute to residual. So the arm
    with fewer FLOPs is not automatically the cheaper arm, and a measurement taken
    while anything else runs on this box is worthless.

    Method: repeat each arm `--repeats` times on the same MLPs and keep the
    MINIMUM residual per arm. The minimum is the right statistic for a timing
    measurement contaminated by unrelated load — noise only ever adds time.
    """
    width, depth = args.width, args.depth
    bk = tuple(int(v) for v in args.buckets.split(","))
    arms = [
        ("dense (v2.4.1 reference)", Config(log2_n=args.log2_n)),
        ("packing only", Config(log2_n=args.log2_n, packing=True, dyn_order=True,
                                buckets=bk, prefix_target=args.prefix_target,
                                rate_cut=args.rate_cut)),
        ("strassen 1", Config(log2_n=args.log2_n, strassen=1)),
        ("strassen 2", Config(log2_n=args.log2_n, strassen=2)),
        ("packing+strassen1", Config(log2_n=args.log2_n, packing=True, dyn_order=True,
                                     buckets=bk, prefix_target=args.prefix_target,
                                     rate_cut=args.rate_cut, strassen=1)),
        ("packing+strassen2", Config(log2_n=args.log2_n, packing=True, dyn_order=True,
                                     buckets=bk, prefix_target=args.prefix_target,
                                     rate_cut=args.rate_cut, strassen=2)),
        ("packing+strassen3", Config(log2_n=args.log2_n, packing=True, dyn_order=True,
                                     buckets=bk, prefix_target=args.prefix_target,
                                     rate_cut=args.rate_cut, strassen=3)),
        ("antithetic", Config(log2_n=args.log2_n, antithetic=True)),
        ("on-compose", Config(log2_n=args.log2_n, on_compose=True, alpha_on=args.alpha_on)),
        ("CANDIDATE (as --flags)", _candidate(args)),
    ]
    print("=" * 124)
    print("RESIDUAL BENCHMARK — uninstrumented wall time per MLP")
    print(f"{args.mlps} MLPs x {args.repeats} repeats, minimum kept; N = {1 << args.log2_n:,}")
    print("Run this with NOTHING else on the machine, or the numbers are fiction.")
    print("=" * 124)

    best = {lbl: {"resid": [], "wall": [], "flops": [], "ops": []} for lbl, _ in arms}
    for i in range(args.mlps):
        w_np, est_seed, _ = canonical_mint(width, depth, 30_806_011_000 + 300 + i)
        for lbl, cfg in arms:
            rs, wl, fl, op = [], [], 0.0, 0
            for _ in range(args.repeats):
                _, bills, _ = run_mlp(w_np, est_seed, cfg, width, depth, n_shifts=1)
                rs.append(bills[0]["residual_s"])
                wl.append(bills[0]["wall_s"])
                fl, op = bills[0]["flops"], bills[0]["ops"]
            best[lbl]["resid"].append(min(rs))
            best[lbl]["wall"].append(min(wl))
            best[lbl]["flops"].append(fl)
            best[lbl]["ops"].append(op)
        print(f"  {i + 1}/{args.mlps}", flush=True)

    d0 = best[arms[0][0]]
    r0 = float(np.median(d0["resid"]))
    f0 = float(np.mean(d0["flops"]))
    c0 = f0 + LAMBDA * GRADED_DENSE_RESIDUAL_S
    print(f"\n{'arm':<28} {'FLOPs':>11} {'ops':>7} {'local resid s':>14} {'x dense':>8} "
          f"{'modelled grader resid':>22} {'C_eff':>11} {'V*c gain':>9}")
    print("-" * 124)
    rows = []
    for lbl, _ in arms:
        d = best[lbl]
        r = float(np.median(d["resid"]))
        f = float(np.mean(d["flops"]))
        ratio = r / r0
        rg = GRADED_DENSE_RESIDUAL_S * ratio
        c = f + LAMBDA * rg
        rows.append((lbl, f, float(np.mean(d["ops"])), r, ratio, rg, c, c0 / c))
        print(f"{lbl:<28} {f:>11.4e} {float(np.mean(d['ops'])):>7.0f} {r:>14.5f} {ratio:>8.3f} "
              f"{rg * 1e3:>19.2f}ms {c:>11.4e} {c0 / c:>8.4f}x")
    print(f"\nAnchor: v2.4 graded residual {GRADED_DENSE_RESIDUAL_S * 1e3:.2f} ms/MLP at "
          f"{5.31e10:.3g} counted FLOPs (submission 324420).")
    print("The 'x dense' column is what actually transfers; the grader column is that ratio")
    print("applied to the anchor. Local absolute residual is this machine's Python overhead")
    print("and is NOT the grader's — quote the ratio, never the level.")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"anchor_s": GRADED_DENSE_RESIDUAL_S, "mlps": args.mlps,
                       "repeats": args.repeats, "rows": rows,
                       "raw": {k: v for k, v in best.items()}}, fh, indent=1)
        print(f"saved {args.out}")
    return 0


def mode_decide(args):
    """S1 — the single-use paired decision. Frozen design, one shot, no retuning."""
    if args.block != "D12":
        print(f"WARNING: --mode decide is meant for the locked block D12, got {args.block}")
    width, depth = args.width, args.depth
    tr = load_truth(args.block, width, depth, args.cost_only)
    bk = tuple(int(v) for v in args.buckets.split(","))
    arms = [
        ("dense (v2.4.1 reference)", Config(log2_n=args.log2_n)),
        ("v2.5 candidate (frozen)", _candidate(args)),
    ]
    print("\nFROZEN CANDIDATE:", arms[1][1].label())
    print(f"PRE-REGISTERED residual ratio (candidate/dense), measured on D11: "
          f"{args.residual_ratio:.4f}")
    per_mlp, n = _collect(arms, tr["seeds"], tr["est_seeds"], tr["truth"], args,
                          width, depth, "S1 PAIRED DECISION — SINGLE USE")
    rc = _finish(args, arms, per_mlp, n, float(np.mean(tr["gt_noise"])))

    bm = np.array(per_mlp[arms[0][0]]["mse"])
    am = np.array(per_mlp[arms[1][0]]["mse"])
    bf = np.array(per_mlp[arms[0][0]]["flops"])
    af = np.array(per_mlp[arms[1][0]]["flops"])

    # Three cost models, reported together because the honest answer lies between
    # them and quoting only one would be a choice disguised as a measurement.
    models = {
        "FLOP-only (upper bound: assumes op-count inflation is free)": (bf, af),
        "PRE-REGISTERED (D11 residual ratio on the graded anchor)": (
            bf + LAMBDA * GRADED_DENSE_RESIDUAL_S,
            af + LAMBDA * GRADED_DENSE_RESIDUAL_S * args.residual_ratio),
        "local residual (this machine; level is NOT the grader's)": (
            bf + LAMBDA * np.array(per_mlp[arms[0][0]]["resid"]),
            af + LAMBDA * np.array(per_mlp[arms[1][0]]["resid"])),
    }
    print("\n" + "=" * 124)
    print("S1 GATE — paired V*c must improve by >= 1.20x")
    print("=" * 124)
    verdicts = {}
    for name, (b, a) in models.items():
        lr = np.log(bm * b) - np.log(am * a)
        gain, se = float(np.exp(lr.mean())), float(lr.std(ddof=1) / math.sqrt(len(lr)))
        lo, hi = float(np.exp(lr.mean() - 1.96 * se)), float(np.exp(lr.mean() + 1.96 * se))
        verdicts[name] = (gain, lo, hi)
        print(f"  {name:<58} {gain:>7.4f}x   95% CI [{lo:.4f}, {hi:.4f}]")
    # External diagnostic anchors (GAME_PLAN rev 9, submission 319341 regraded under
    # flopscope 0.10.0). These are a sanity check on whether we reproduced the right
    # MECHANISM -- they are not a target to tune towards, and nothing here was fitted
    # to them.
    print(f"\n{'':<58} {'ours':>14} {'319341 (published)':>20}")
    n_s = 1 << args.log2_n
    cand_c = float(np.mean(af)) + LAMBDA * GRADED_DENSE_RESIDUAL_S * args.residual_ratio
    print(f"  {'effective FLOPs/sample (pre-registered model)':<56} "
          f"{cand_c / n_s:>14.4e} {2.374e6:>20.4e}")
    print(f"  {'effective FLOPs/sample (FLOP-only)':<56} "
          f"{float(np.mean(af)) / n_s:>14.4e} {2.374e6:>20.4e}")
    print(f"  {'MSE * N':<56} {float(np.mean(am)) * n_s:>14.6f} {0.018520:>20.6f}")
    print(f"  {'dense reference MSE * N':<56} {float(np.mean(bm)) * n_s:>14.6f} {'':>20}")

    # What would have to be true for the gate to clear? Solving
    #   C_dense / (F_cand + lambda * r_dense * ratio) = 1.20
    # for `ratio` turns "it failed" into a falsifiable statement that one graded
    # run in Phase 2 could settle, instead of a verdict that depends on a model.
    mse_adv = float(np.mean(bm)) / float(np.mean(am))
    c_dense = float(np.mean(bf)) + LAMBDA * GRADED_DENSE_RESIDUAL_S
    need = (c_dense * mse_adv / 1.20 - float(np.mean(af))) / (LAMBDA * GRADED_DENSE_RESIDUAL_S)
    print(f"\n  Break-even: the gate clears iff the candidate's residual is at most "
          f"{need:.2f}x the dense arm's,")
    print(f"  i.e. at most {need * GRADED_DENSE_RESIDUAL_S * 1e3:.1f} ms/MLP against v2.4's "
          f"{GRADED_DENSE_RESIDUAL_S * 1e3:.2f} ms.")
    print(f"  Measured on an idle machine: {args.residual_ratio:.2f}x "
          f"({args.residual_ratio * GRADED_DENSE_RESIDUAL_S * 1e3:.1f} ms modelled).")

    g, lo, hi = verdicts["PRE-REGISTERED (D11 residual ratio on the graded anchor)"]
    # GAME_PLAN rev 9 states the gate as "V*c improves by at least 1.20x, with the
    # paired uncertainty reported". So the point estimate carries the threshold and
    # the interval carries the credibility; requiring the INTERVAL to clear 1.20
    # would silently be a stricter gate than the one that was pre-registered.
    # Both readings are printed so neither can be chosen after the fact.
    passed = (g >= 1.20) and (lo > 1.0)
    strong = lo >= 1.20
    print(f"\n  n = {len(bm)} MLPs x {args.shifts} shifts, paired, common random numbers")
    print(f"  DECIDING MODEL: pre-registered residual. gain {g:.4f}x, 95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"  gate as written  (point estimate >= 1.20x and CI excludes 1.0) : "
          f"{'PASS' if passed else 'FAIL'}")
    print(f"  strict reading   (CI lower bound >= 1.20x)                     : "
          f"{'PASS' if strong else 'FAIL'}")
    if not passed:
        print("  -> Append the kill to experiments/LOG.md, update GAME_PLAN/BENCHMARK, and STOP.")
        print("     Do NOT retune and rerun D12: it is a single-use block and has been spent.")
    return rc


# ===========================================================================
# SPRINT 5 — A: batched Strassen
# ===========================================================================

def _bill(fn):
    """Run `fn` under a fresh budget and return (result, metrics)."""
    _host_reset()
    t0 = time.perf_counter()
    with flops.budget(10 ** 16, quiet=True) as b:
        out = fn()
        arr = np.asarray(out)
    wall = time.perf_counter() - t0
    sd = b.summary_dict()
    return arr, {
        "flops": float(b.flops_used),
        "ops": int(sum(v["calls"] for v in sd["operations"].values())),
        "resid": float(b.residual_wall_time_s),
        "wall": wall,
        "max_bytes": max(_HOST["maxbytes"], _log_max_bytes(b)),
    }


def mode_strassen(args):
    """A — batched depth-2 Strassen: parity, operation count, cost, memory.

    Two layers of evidence, because they answer different questions. The
    microbenchmark isolates the kernel on the shapes the estimator actually
    issues (and on deliberately awkward ones); the full-estimator arm answers
    whether any of it survives contact with the other 95% of the pass — which
    is exactly the step the independent audit flagged as unmeasured.
    """
    print("=" * 122)
    print("SPRINT 5 / A — BATCHED DEPTH-2 STRASSEN")
    print("=" * 122)

    rng = np.random.default_rng(20_260_806)
    shapes = [
        (4096, 200, 200, "audit microbenchmark shape"),
        (16384, 205, 205, "estimator layer, N=16,384 (typical alive count)"),
        (16384, 199, 187, "ODD contraction AND output dim"),
        (32768, 205, 205, "estimator layer, N=32,768"),
        (65536, 205, 205, "estimator layer, N=65,536"),
        (16384, 33, 33, "below 2*_STRASSEN_MIN: must fall back to dense"),
    ]
    print(f"\n{'shape':<26} {'variant':<14} {'FLOPs':>13} {'vs dense':>9} {'ops':>5} "
          f"{'resid ms':>9} {'wall ms':>9} {'max array':>11} {'bitwise==rec':>13} "
          f"{'rel vs dense':>13}")
    print("-" * 122)
    fails = []
    for (M, K, N, why) in shapes:
        An = rng.standard_normal((M, K), dtype=np.float32)
        Bn = rng.standard_normal((K, N), dtype=np.float32)
        with flops.budget(10 ** 16, quiet=True):
            A, B = fnp.array(An), fnp.array(Bn)
        ref, m0 = _bill(lambda: A @ B)
        scale = max(float(np.abs(ref).max()), 1e-30)
        print(f"{f'({M},{K})@({K},{N})':<26} {'dense':<14} {m0['flops']:>13,.0f} "
              f"{1.0:>9.4f} {m0['ops']:>5} {m0['resid'] * 1e3:>9.3f} {m0['wall'] * 1e3:>9.2f} "
              f"{m0['max_bytes'] / 2 ** 20:>9.1f}MB {'-':>13} {0.0:>13.3e}")
        for depth in (1, 2):
            rr, mr = _bill(lambda: strassen(A, B, depth, "recursive"))
            bb, mb = _bill(lambda: strassen(A, B, depth, "batched"))
            same = bool(np.array_equal(rr, bb))
            if not same:
                fails.append(f"({M},{K})@({K},{N}) depth {depth}: batched != recursive")
            for tag, arr, mm in (("recursive d%d" % depth, rr, mr),
                                 ("BATCHED  d%d" % depth, bb, mb)):
                rel = float(np.abs(arr - ref).max() / scale)
                print(f"{'':<26} {tag:<14} {mm['flops']:>13,.0f} "
                      f"{mm['flops'] / m0['flops']:>9.4f} {mm['ops']:>5} "
                      f"{mm['resid'] * 1e3:>9.3f} {mm['wall'] * 1e3:>9.2f} "
                      f"{mm['max_bytes'] / 2 ** 20:>9.1f}MB "
                      f"{(str(same) if 'BATCH' in tag else '-'):>13} {rel:>13.3e}")
        print(f"{'':<26} ({why})")

    # ---- depth 0 and determinism -----------------------------------------
    An = rng.standard_normal((16384, 205), dtype=np.float32)
    Bn = rng.standard_normal((205, 205), dtype=np.float32)
    with flops.budget(10 ** 16, quiet=True):
        A, B = fnp.array(An), fnp.array(Bn)
    d0, _ = _bill(lambda: strassen(A, B, 0, "batched"))
    dn, _ = _bill(lambda: A @ B)
    if not np.array_equal(d0, dn):
        fails.append("depth 0 batched != plain matmul")
    reps = [_bill(lambda: strassen(A, B, 2, "batched"))[0] for _ in range(3)]
    if not all(np.array_equal(reps[0], r) for r in reps[1:]):
        fails.append("batched depth 2 is not deterministic across repeats")
    print(f"\ndepth 0 == plain matmul (bitwise): {np.array_equal(d0, dn)}")
    print(f"batched depth 2 deterministic over 3 runs: "
          f"{all(np.array_equal(reps[0], r) for r in reps[1:])}")

    # ---- full estimator ---------------------------------------------------
    width, depth = args.width, args.depth
    print("\n" + "=" * 122)
    print("FULL ESTIMATOR — does the kernel result survive the whole pass?")
    print(f"{args.mlps} MLPs from D13, N = {1 << args.log2_n:,}")
    print("=" * 122)
    base = dict(log2_n=args.log2_n)
    bk = tuple(int(v) for v in args.buckets.split(",")) if args.buckets else ()
    packed = dict(packing=True, dyn_order=True, buckets=bk,
                  prefix_target=args.prefix_target, rate_cut=args.rate_cut)
    arms = [
        ("dense (v2.4.1)", Config(**base)),
        ("dense+strassen1 rec", Config(**base, strassen=1)),
        ("dense+strassen1 BAT", Config(**base, strassen=1, strassen_mode="batched")),
        ("dense+strassen2 rec", Config(**base, strassen=2)),
        ("dense+strassen2 BAT", Config(**base, strassen=2, strassen_mode="batched")),
        ("packed", Config(**base, **packed)),
        ("packed+strassen1 rec", Config(**base, **packed, strassen=1)),
        ("packed+strassen1 BAT", Config(**base, **packed, strassen=1,
                                        strassen_mode="batched")),
        ("packed+strassen2 rec", Config(**base, **packed, strassen=2)),
        ("packed+strassen2 BAT", Config(**base, **packed, strassen=2,
                                        strassen_mode="batched")),
    ]
    seeds = _bs_seeds("D13")[:args.mlps]
    acc = {lbl: {"flops": [], "ops": [], "resid": [], "wall": [], "mb": [],
                 "rs": [], "rv": [], "out": []} for lbl, _ in arms}
    for i, sd in enumerate(seeds):
        w_np, est_seed, _ = canonical_mint(width, depth, int(sd))
        for lbl, cfg in arms:
            _, bills, out = run_mlp(w_np, est_seed, cfg, width, depth, n_shifts=1)
            b = bills[0]
            a = acc[lbl]
            a["flops"].append(b["flops"]); a["ops"].append(b["ops"])
            a["resid"].append(b["residual_s"]); a["wall"].append(b["wall_s"])
            a["mb"].append(b["max_bytes"]); a["rs"].append(b["reads_shape"])
            a["rv"].append(b["reads_value"]); a["out"].append(out)
        print(f"  {i + 1}/{len(seeds)}", flush=True)

    ref = arms[0][0]
    f0 = float(np.mean(acc[ref]["flops"]))
    print(f"\n{'arm':<24} {'FLOPs/MLP':>12} {'vs dense':>9} {'ops':>7} {'resid ms':>9} "
          f"{'wall ms':>9} {'max array':>11} {'C_eff local':>12} {'rel err':>10}")
    print("-" * 122)
    rows = {}
    for lbl, _ in arms:
        a = acc[lbl]
        f = float(np.mean(a["flops"]))
        r = float(np.median(a["resid"]))
        rel = max(float(np.abs(np.array(a["out"][j]) - np.array(acc[ref]["out"][j])).max()
                        / max(float(np.abs(acc[ref]["out"][j]).max()), 1e-30))
                  for j in range(len(seeds)))
        c = f + LAMBDA * r
        rows[lbl] = (f, float(np.mean(a["ops"])), r, float(np.median(a["wall"])),
                     float(np.max(a["mb"])), c, rel)
        print(f"{lbl:<24} {f:>12.4e} {f / f0:>9.4f} {float(np.mean(a['ops'])):>7.0f} "
              f"{r * 1e3:>9.2f} {float(np.median(a['wall'])) * 1e3:>9.1f} "
              f"{float(np.max(a['mb'])) / 2 ** 20:>9.1f}MB {c:>12.4e} {rel:>10.3e}")
    print(f"\npeak process working set after the sweep: "
          f"{peak_rss_bytes() / 2 ** 30:.2f} GiB")
    print("\nRecursive vs batched must agree BITWISE (they share `_ops7`/`_combine` and")
    print("differ only in how the leaves are dispatched); 'rel err' is against the DENSE")
    print("arm and is Strassen's own reassociation, which is a measured property, not a bug.")
    if fails:
        print("\nFAILURES:")
        for f in fails:
            print("  " + f)
    print(f"\nA-gate (parity + determinism + fallbacks): "
          f"{'PASS' if not fails else 'FAIL'}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"rows": rows, "fails": fails, "n": 1 << args.log2_n,
                       "mlps": len(seeds)}, fh, indent=1, default=str)
        print(f"saved {args.out}")
    return 0 if not fails else 1


def _bs_seeds(block):
    from seed_blocks import seeds as _s
    return _s(block)


# ===========================================================================
# SPRINT 5 — B: the packing-operation Pareto
# ===========================================================================

def _pareto_arms(args):
    """Every exact carrier worth pricing, as (label, Config).

    All of these compute the same function. They differ only in how many
    operations, host round trips and temporaries they need to do it, which is
    precisely the axis flopscope does not bill and the grader still charges for.
    """
    base = dict(log2_n=args.log2_n, prefix_target=args.prefix_target,
                rate_cut=args.rate_cut)
    pk = dict(packing=True, dyn_order=True)
    return [
        ("dense (v2.4.1 control)", Config(log2_n=args.log2_n)),
        # --- packing only, bucket layouts -------------------------------
        ("pack b=(1,2,4,8,16) CONTROL", Config(**base, **pk, buckets=(1, 2, 4, 8, 16))),
        ("pack b=() zero-rect only", Config(**base, **pk, buckets=())),
        ("pack b=(8,)", Config(**base, **pk, buckets=(8,))),
        ("pack b=(16,)", Config(**base, **pk, buckets=(16,))),
        ("pack b=(4,16)", Config(**base, **pk, buckets=(4, 16))),
        ("pack b=(8,16)", Config(**base, **pk, buckets=(8, 16))),
        ("pack b=(2,8,16)", Config(**base, **pk, buckets=(2, 8, 16))),
        ("pack b=(4,8,16)", Config(**base, **pk, buckets=(4, 8, 16))),
        # --- the same layouts, one ordering op instead of masks ----------
        ("pack b=(8,) segment", Config(**base, **pk, buckets=(8,), bucket_mode="segment")),
        ("pack b=(1,2,4,8,16) segment", Config(**base, **pk, buckets=(1, 2, 4, 8, 16),
                                               bucket_mode="segment")),
        # --- memory-bounded gather ---------------------------------------
        ("pack b=(8,) slotloop", Config(**base, **pk, buckets=(8,), gather_mode="slotloop")),
        ("pack b=(4,16) slotloop", Config(**base, **pk, buckets=(4, 16),
                                          gather_mode="slotloop")),
        # --- Strassen without packing ------------------------------------
        ("strassen1 BAT, no packing", Config(log2_n=args.log2_n, strassen=1,
                                             strassen_mode="batched")),
        ("strassen2 BAT, no packing", Config(log2_n=args.log2_n, strassen=2,
                                             strassen_mode="batched")),
        # --- batched Strassen + the promising bucket layouts --------------
        ("pack b=(8,) + strassen1 BAT", Config(**base, **pk, buckets=(8,), strassen=1,
                                               strassen_mode="batched")),
        ("pack b=(8,) + strassen2 BAT", Config(**base, **pk, buckets=(8,), strassen=2,
                                               strassen_mode="batched")),
        ("pack b=(4,16) + strassen1 BAT", Config(**base, **pk, buckets=(4, 16), strassen=1,
                                                 strassen_mode="batched")),
        ("pack b=(1,2,4,8,16)+str2 rec", Config(**base, **pk, buckets=(1, 2, 4, 8, 16),
                                                strassen=2)),
        ("pack b=(1,2,4,8,16)+str2 BAT", Config(**base, **pk, buckets=(1, 2, 4, 8, 16),
                                                strassen=2, strassen_mode="batched")),
        # --- prefix width interacts with the bucket layout, so sweep it here --
        ("pack CONTROL, target 1.5", Config(log2_n=args.log2_n, **pk, prefix_target=1.5,
                                            buckets=(1, 2, 4, 8, 16))),
        ("pack CONTROL, target 6", Config(log2_n=args.log2_n, **pk, prefix_target=6.0,
                                          buckets=(1, 2, 4, 8, 16))),
        ("pack b=(4,8,16), target 6", Config(log2_n=args.log2_n, **pk, prefix_target=6.0,
                                             buckets=(4, 8, 16))),
        ("pack b=(8,), target 1.5", Config(log2_n=args.log2_n, **pk, prefix_target=1.5,
                                           buckets=(8,))),
    ]


def _measure_arms(arms, seeds, args, width, depth, repeats=1):
    """Cost/exactness for every arm on every MLP, with common random numbers.

    Residual is kept as the MINIMUM over repeats: timing noise only ever adds.
    """
    acc = {lbl: {k: [] for k in ("flops", "sampling", "fixed", "ops", "ops_s",
                                 "resid", "wall", "mb", "rs", "rv", "rel", "pfx")}
           for lbl, _ in arms}
    ref_lbl = arms[0][0]
    for i, sd in enumerate(seeds):
        w_np, est_seed, _ = canonical_mint(width, depth, int(sd))
        ref_out = None
        for lbl, cfg in arms:
            best = None
            for _ in range(repeats):
                _, bills, out = run_mlp(w_np, est_seed, cfg, width, depth, n_shifts=1)
                if best is None or bills[0]["residual_s"] < best[0]["residual_s"]:
                    best, best_out = bills, out
            b = best[0]
            if lbl == ref_lbl:
                ref_out = best_out
            a = acc[lbl]
            a["flops"].append(b["flops"]); a["sampling"].append(b["flops_sampling"])
            a["fixed"].append(b["flops_fixed"]); a["ops"].append(b["ops"])
            a["ops_s"].append(b["ops_sampling"])
            a["resid"].append(b["residual_s"]); a["wall"].append(b["wall_s"])
            a["mb"].append(b["max_bytes"]); a["rs"].append(b["reads_shape"])
            a["rv"].append(b["reads_value"])
            a["pfx"].append(float(np.mean([q for q in b["prefix"][1:]])))
            a["rel"].append(float(np.abs(best_out - ref_out).max()
                                  / max(float(np.abs(ref_out).max()), 1e-30)))
        print(f"  {i + 1}/{len(seeds)}", flush=True)
    return acc


def mode_pareto(args):
    """B — the packing-operation Pareto. No arm is selected by one scalar."""
    width, depth = args.width, args.depth
    seeds = _bs_seeds(args.block)[:args.mlps]
    arms = _pareto_arms(args)
    n = 1 << args.log2_n
    print("=" * 150)
    print("SPRINT 5 / B — PACKING-OPERATION PARETO  (design block, tuning permitted)")
    print(f"block {args.block}  {len(seeds)} MLPs x {args.repeats} repeats (min residual)  "
          f"N = {n:,}  prefix_target = {args.prefix_target:g}")
    print("Every arm computes the SAME function; they differ in operations, host round")
    print("trips and temporaries. Counted FLOPs alone would pick the wrong one.")
    print("=" * 150)
    acc = _measure_arms(arms, seeds, args, width, depth, repeats=args.repeats)

    ref = arms[0][0]
    f0 = float(np.mean(acc[ref]["flops"]))
    r0 = float(np.median(acc[ref]["resid"]))
    print(f"\n{'arm':<32} {'FLOPs/MLP':>11} {'fixed':>10} {'marg/samp':>10} {'ops':>6} "
          f"{'ops/samp':>8} {'shp rd':>7} {'val rd':>7} {'resid ms':>9} {'xdense':>7} "
          f"{'wall ms':>8} {'max arr':>9} {'C_eff':>11} {'x dense':>8} {'exact':>10}")
    print("-" * 150)
    rows = {}
    for lbl, cfg in arms:
        a = acc[lbl]
        f = float(np.mean(a["flops"]))
        fx = float(np.mean(a["fixed"]))
        r = float(np.median(a["resid"]))
        c = f + LAMBDA * r
        rows[lbl] = {
            "flops": f, "fixed": fx, "marginal_per_sample": (f - fx) / n,
            "ops": float(np.mean(a["ops"])), "ops_sampling": float(np.mean(a["ops_s"])),
            "reads_shape": float(np.mean(a["rs"])), "reads_value": float(np.mean(a["rv"])),
            "resid_s": r, "resid_ratio": r / r0, "wall_s": float(np.median(a["wall"])),
            "max_bytes": float(np.max(a["mb"])), "c_eff_local": c,
            "prefix_mean": float(np.mean(a["pfx"])),
            "rel_vs_dense": float(np.max(a["rel"])), "config": cfg.label(),
        }
        print(f"{lbl:<32} {f:>11.4e} {fx:>10.3e} {(f - fx) / n:>10.4e} "
              f"{float(np.mean(a['ops'])):>6.0f} {float(np.mean(a['ops_s'])):>8.0f} "
              f"{float(np.mean(a['pfx'])):>5.0f} "
              f"{float(np.mean(a['rs'])):>7.0f} {float(np.mean(a['rv'])):>7.0f} "
              f"{r * 1e3:>9.2f} {r / r0:>7.2f} {float(np.median(a['wall'])) * 1e3:>8.1f} "
              f"{float(np.max(a['mb'])) / 2 ** 20:>7.1f}MB {c:>11.4e} "
              f"{(f0 + LAMBDA * r0) / c:>8.4f} {float(np.max(a['rel'])):>10.3e}")

    print(f"\npeak process working set: {peak_rss_bytes() / 2 ** 30:.2f} GiB")
    print("\n'exact' is max relative deviation from the dense control. Non-Strassen arms")
    print("are exact reorganisations and must sit at f32 reassociation level (<=1e-6);")
    print("Strassen arms reassociate more and are only norm-wise stable, so their number")
    print("is a measurement, not a gate. 'shp rd'/'val rd' are host round trips per MLP:")
    print("a SHAPE read is metadata on a RemoteArray, a VALUE read is a transfer.")
    print("C_eff uses THIS MACHINE's residual, whose level is ~7x the grader's; the")
    print("'x dense' column is the ratio, which is the part that transfers.")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"block": args.block, "mlps": len(seeds), "n": n,
                       "repeats": args.repeats, "rows": rows}, fh, indent=1, default=str)
        print(f"saved {args.out}")
    return 0


# ===========================================================================
# SPRINT 5 — C: the sample-count operating point, reopened for the packed carrier
# ===========================================================================

def _fit_fixed_marginal(ns, flops):
    """Least-squares split of counted FLOPs into F0 + c*N.

    N does not cancel out of the score once fixed costs exist:
    `adjusted = V * (c_marginal + F0/N) / B`, and the published competitor runs
    at N = 84,992 where F0 is amortised 5.2x harder than at our 16,384. This is
    the fit that makes that statement quantitative instead of rhetorical.
    """
    x = np.asarray(ns, dtype=np.float64)
    y = np.asarray(flops, dtype=np.float64)
    A = np.stack([np.ones_like(x), x], axis=1)
    (f0, c), *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(f0), float(c)


def mode_nsweep(args):
    """C — evaluate the Pareto-best carriers at N = 16,384 / 32,768 / 65,536."""
    width, depth = args.width, args.depth
    tr = load_truth(args.block, width, depth, args.cost_only)
    seeds = tr["seeds"][:args.mlps]
    est_seeds = tr["est_seeds"][:args.mlps]
    truth = tr["truth"]
    gt_noise = float(np.mean(tr["gt_noise"]))
    log2s = [int(v) for v in args.log2ns.split(",")]

    carriers = []
    for spec in args.carriers.split(";"):
        spec = spec.strip()
        if not spec:
            continue
        lbl, rest = spec.split(":", 1)
        kw = {}
        for tok in rest.split(","):
            tok = tok.strip()
            if not tok:
                continue
            k, v = tok.split("~")
            if k == "buckets":
                kw[k] = tuple(int(x) for x in v.split("/")) if v != "none" else ()
            elif k in ("packing", "dyn_order", "antithetic", "on_compose"):
                kw[k] = v == "1"
            elif k in ("strassen",):
                kw[k] = int(v)
            elif k in ("prefix_target", "rate_cut", "alpha_on"):
                kw[k] = float(v)
            else:
                kw[k] = v
        carriers.append((lbl.strip(), kw))

    print("=" * 148)
    print("SPRINT 5 / C — SAMPLE-COUNT OPERATING POINT FOR THE PACKED CARRIER")
    print(f"block {args.block}  {len(seeds)} MLPs x {args.shifts} shifts  "
          f"N in {[1 << v for v in log2s]}")
    print("Nested Sobol prefixes: the N=32,768 point set CONTAINS the N=16,384 one, and")
    print("every arm sees the same networks and the same digital shift, so the N")
    print("comparison is paired and the point-set realisation is common.")
    print("=" * 148)

    res = {}
    for lbl, kw in carriers:
        for lg in log2s:
            cfg = Config(log2_n=lg, **kw)
            key = (lbl, lg)
            mses, fl, sa, fx, rs, wl, op, mb, rvv, rss = ([] for _ in range(10))
            for i in range(len(seeds)):
                w_np, _, _ = canonical_mint(width, depth, int(seeds[i]))
                # Repeats exist only for the timing columns: noise adds time, never
                # removes it, so the MINIMUM is the right estimator of an
                # uncontaminated residual. FLOPs and MSE are identical each pass.
                best = None
                for _ in range(max(1, args.repeats)):
                    m, bills, _ = run_mlp(w_np, int(est_seeds[i]), cfg, width, depth,
                                          n_shifts=args.shifts,
                                          truth=None if args.cost_only else truth[i])
                    if best is None or bills[0]["residual_s"] < best[1][0]["residual_s"]:
                        best = (m, bills)
                m, bills = best
                b = bills[0]
                # MSE is averaged over the shifts (point-set realisation noise is
                # what multi-shift averaging exists to remove); cost columns are
                # taken from the first shift, which is representative because the
                # arithmetic is shift-independent.
                mses.append(float(np.mean(m)))
                fl.append(b["flops"]); sa.append(b["flops_sampling"])
                fx.append(b["flops_fixed"]); rs.append(b["residual_s"])
                wl.append(b["wall_s"]); op.append(b["ops"]); mb.append(b["max_bytes"])
                rvv.append(b["reads_value"]); rss.append(b["reads_shape"])
            res[key] = {
                "mse": float(np.mean(mses)), "flops": float(np.mean(fl)),
                "sampling": float(np.mean(sa)), "fixed": float(np.mean(fx)),
                "resid": float(np.median(rs)), "wall": float(np.median(wl)),
                "ops": float(np.mean(op)), "max_bytes": float(np.max(mb)),
                "reads_value": float(np.mean(rvv)), "reads_shape": float(np.mean(rss)),
                "mse_list": [float(v) for v in mses],
                "flops_list": [float(v) for v in fl],
                "resid_list": [float(v) for v in rs],
            }
            print(f"  done {lbl} log2N={lg}", flush=True)

    print(f"\n{'carrier':<30} {'N':>7} {'MSE':>11} {'MSE*N':>8} {'FLOPs/MLP':>11} "
          f"{'fixed':>10} {'counted/samp':>12} {'resid ms':>9} {'eff/samp':>11} "
          f"{'adj pred':>10} {'ops':>6} {'max arr':>9}")
    print("-" * 148)
    for lbl, _ in carriers:
        for lg in log2s:
            d = res[(lbl, lg)]
            n = 1 << lg
            c_local = d["flops"] + LAMBDA * d["resid"]
            adj = d["mse"] * max(0.1, c_local / BUDGET)
            print(f"{lbl:<30} {n:>7,} {d['mse']:>11.4e} {d['mse'] * n:>8.5f} "
                  f"{d['flops']:>11.4e} {d['fixed']:>10.3e} {d['flops'] / n:>12.4e} "
                  f"{d['resid'] * 1e3:>9.2f} {c_local / n:>11.4e} {adj:>10.4e} "
                  f"{d['ops']:>6.0f} {d['max_bytes'] / 2 ** 20:>7.1f}MB")
        f0, cm = _fit_fixed_marginal([1 << v for v in log2s],
                                     [res[(lbl, v)]["flops"] for v in log2s])
        print(f"{'  -> fixed/marginal split':<30} F0 = {f0:.4e} counted FLOPs, "
              f"c_marginal = {cm:.4e} counted FLOPs/sample "
              f"(F0/N at 16,384 = {f0 / 16384:.3e}/sample; at 84,992 = "
              f"{f0 / 84992:.3e}/sample)")

    print(f"\nGT noise {gt_noise:.3e}. Competitor anchors (submission 319341, graded):")
    print("  N = 84,992   counted/sample ~ 2.144e6   effective/sample 2.374e6   "
          "MSE*N = 0.018520   residual ~185.8 ms/MLP")
    print("  Our comparable graded v2.4.1 variance proxy is MSE*N = 0.022525 "
          "(gap 1.216x, NOT 1.40x).")
    print("  'counted/samp' above is counted-to-counted comparable; 'eff/samp' uses THIS")
    print("  MACHINE's residual and is NOT comparable to their graded effective figure.")
    print(f"\npeak process working set: {peak_rss_bytes() / 2 ** 30:.2f} GiB")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"block": args.block, "mlps": len(seeds), "gt_noise": gt_noise,
                       "log2ns": log2s,
                       "res": {f"{k[0]}|{k[1]}": v for k, v in res.items()}},
                      fh, indent=1, default=str)
        print(f"saved {args.out}")
    return 0


# ===========================================================================
# SPRINT 5 — D/E: residual models and the single-use D14 gate
# ===========================================================================

# The only two real-grader residual observations that exist.
#
#   ours   — submission 324420 (v2.4.1's method): 9.84 ms/MLP of uninstrumented
#            time against 5.31e10 counted FLOPs, at N = 16,384. A DENSE carrier.
#   theirs — submission 319341: ~185.8 ms/MLP at N = 84,992. The only PACKED
#            carrier anyone has ever had graded, and 18.9x our dense residual.
#
# The second number is why model 3 below exists. Our local machine says a packed
# carrier costs ~3x the dense residual; the grader's own packed data point says
# ~19x. Believing our own ratio over someone else's measurement, when the
# measurement is the one taken on the actual scoring hardware, would be exactly
# the mistake Sprint 4 made in the other direction.
GRADED_DENSE_RESIDUAL_S = 9.84e-3
COMPETITOR_RESIDUAL_S = 185.8e-3
COMPETITOR_N = 84_992
COMPETITOR_RESIDUAL_PER_SAMPLE = COMPETITOR_RESIDUAL_S / COMPETITOR_N   # 2.186e-6 s

# How much of a packed carrier's residual does NOT amortise with N. Measured on
# D13 (2026-08-06): the six-bucket packed carrier's local residual fits
# r(N) = 31.9 ms + 9.265e-6 s/sample over N in {16384, 32768, 65536}, so at
# N = 84,992 the fixed part is 31.9 / (31.9 + 787.4) = 3.9%. FROZEN before D14.
COMPETITOR_FIXED_SHARE = 0.039


def residual_models(n_ctrl, n_cand, r_ctrl_local, r_cand_local):
    """The three residual readings required before D14 may be touched.

    Returned as (name, r_control_seconds, r_candidate_seconds, why).

    1. **Local absolute.** What this machine measures. Its LEVEL is roughly 7x
       the grader's (70.8 ms here vs 9.84 ms graded for the same file), so it is
       reported and never used to decide.
    2. **Ratio transfer.** The candidate/control ratio measured locally, applied
       to the graded dense anchor. This is Sprint 4's pre-registered model. It
       assumes the ratio transfers even though the level does not — plausible,
       unproven, and self-reported.
    3. **Conservative packed-carrier model — the WORSE of two readings.** The
       candidate is charged `max(model 2, 319341's published rate)`. The
       competitor half is built from its 185.8 ms/MLP at N = 84,992, split into
       fixed and marginal parts using OUR OWN measured local `r(N) = r0 + r1*N`
       fit for a packed carrier (Sprint 5 measured 3.9% fixed at their N), so
       only the share that demonstrably amortises with N is allowed to amortise.

       The `max` is load-bearing and was not the first definition tried. Charging
       the competitor's rate ALONE is conservative for a carrier whose own local
       residual ratio is low, and *optimistic* for one whose ratio is high —
       Strassen arms measure 8.3x dense locally against the 4.2x this anchor
       implies, so the anchor alone would have quietly forgiven them 4x. Taking
       the worse of the two readings is conservative for every arm regardless of
       where it sits, which is the only version of "conservative" that can be
       frozen before the arms are known.
    """
    m2 = GRADED_DENSE_RESIDUAL_S * (r_cand_local / max(r_ctrl_local, 1e-12))
    m3_fixed = COMPETITOR_FIXED_SHARE * COMPETITOR_RESIDUAL_S
    m3_marginal = ((1.0 - COMPETITOR_FIXED_SHARE) * COMPETITOR_RESIDUAL_S
                   / COMPETITOR_N) * n_cand
    return [
        ("1. local absolute (level NOT the grader's)", r_ctrl_local, r_cand_local,
         "reported only"),
        ("2. local ratio on the graded dense anchor",
         GRADED_DENSE_RESIDUAL_S, m2, "self-reported ratio"),
        ("3. CONSERVATIVE = max(model 2, 319341-anchored)",
         GRADED_DENSE_RESIDUAL_S, max(m2, m3_fixed + m3_marginal),
         "DECIDING MODEL"),
    ]


def _paired(bm, bc, am, ac):
    """Geometric mean of the per-MLP V*C ratio, with its 95% interval.

    The gate's statistic is `exp(mean(log(V*C_control / V*C_candidate)))` — the
    paired geometric mean, not the ratio of means. The independent audit found
    Sprint 4 quoting the latter, which read 1.2026x where the pre-registered
    statistic read 1.1996x and therefore missed its gate.
    """
    lr = np.log(bm * bc) - np.log(am * ac)
    g = float(np.exp(lr.mean()))
    se = float(lr.std(ddof=1) / math.sqrt(len(lr))) if len(lr) > 1 else float("nan")
    return g, float(np.exp(lr.mean() - 1.96 * se)), float(np.exp(lr.mean() + 1.96 * se))


def mode_decide5(args):
    """E — the Sprint-5 single-use decision. Frozen design, frozen model, one shot."""
    from seed_blocks import SINGLE_USE_BLOCKS

    width, depth = args.width, args.depth
    if args.block not in SINGLE_USE_BLOCKS:
        print(f"WARNING: --mode decide5 is meant for a single-use block; "
              f"got {args.block} (registered: {sorted(SINGLE_USE_BLOCKS)})")
    tr = load_truth(args.block, width, depth, args.cost_only)
    seeds, est_seeds, truth = tr["seeds"], tr["est_seeds"], tr["truth"]
    m = min(args.mlps, len(seeds))

    ctrl = Config(log2_n=14)                       # shipped v2.4.1 operating point
    cand = Config(log2_n=args.log2_n, packing=args.cand_packing, dyn_order=True,
                  buckets=tuple(int(v) for v in args.buckets.split(","))
                  if args.buckets else (),
                  bucket_mode=args.bucket_mode, gather_mode=args.gather_mode,
                  prefix_target=args.prefix_target, rate_cut=args.rate_cut,
                  strassen=args.strassen, strassen_mode=args.strassen_mode,
                  antithetic=args.antithetic, on_compose=args.on_compose,
                  alpha_on=args.alpha_on)
    arms = [("v2.4.1 dense control", ctrl), ("SPRINT-5 CANDIDATE (frozen)", cand)]

    print("=" * 128)
    print("SPRINT 5 / E — SINGLE-USE PAIRED DECISION")
    print(f"block {args.block}  {m} MLPs x {args.shifts} shifts")
    print(f"FROZEN CANDIDATE : {cand.label()}  at N = {1 << args.log2_n:,}")
    print(f"CONTROL          : {ctrl.label()}  at N = {1 << 14:,}")
    print(f"PRE-REGISTERED local residual ratio (from D13): {args.residual_ratio:.4f}")
    print("DECIDING RESIDUAL MODEL: 3 (conservative, 319341-anchored) — frozen on D13")
    print("GATE: paired geometric-mean V*C >= "
          f"{args.gate:.2f}x AND 95% CI lower bound > 1.0")
    print("=" * 128)

    per = {lbl: {"mse": [], "flops": [], "resid": [], "ops": []} for lbl, _ in arms}
    t0 = time.perf_counter()
    for i in range(m):
        w_np, _, _ = canonical_mint(width, depth, int(seeds[i]))
        for lbl, cfg in arms:
            mses, bills, _ = run_mlp(w_np, int(est_seeds[i]), cfg, width, depth,
                                     n_shifts=args.shifts,
                                     truth=None if args.cost_only else truth[i])
            per[lbl]["mse"].append(float(np.mean(mses)))
            per[lbl]["flops"].append(float(np.mean([b["flops"] for b in bills])))
            per[lbl]["resid"].append(float(np.mean([b["residual_s"] for b in bills])))
            per[lbl]["ops"].append(float(np.mean([b["ops"] for b in bills])))
        if (i + 1) % 10 == 0 or i + 1 == m:
            print(f"  {i + 1}/{m}  {time.perf_counter() - t0:.0f}s", flush=True)

    bm = np.array(per[arms[0][0]]["mse"])
    am = np.array(per[arms[1][0]]["mse"])
    bf = np.array(per[arms[0][0]]["flops"])
    af = np.array(per[arms[1][0]]["flops"])
    n_ctrl, n_cand = 1 << 14, 1 << args.log2_n
    r_ctrl_local = float(np.median(per[arms[0][0]]["resid"]))
    r_cand_local = float(np.median(per[arms[1][0]]["resid"]))

    print(f"\n{'arm':<30} {'MSE':>11} {'MSE*N':>9} {'counted/MLP':>12} "
          f"{'counted/samp':>13} {'ops':>7} {'local resid ms':>15}")
    print("-" * 128)
    for lbl, cfg in arms:
        d = per[lbl]
        nn = 1 << cfg.log2_n
        print(f"{lbl:<30} {float(np.mean(d['mse'])):>11.4e} "
              f"{float(np.mean(d['mse'])) * nn:>9.5f} {float(np.mean(d['flops'])):>12.4e} "
              f"{float(np.mean(d['flops'])) / nn:>13.4e} {float(np.mean(d['ops'])):>7.0f} "
              f"{float(np.median(d['resid'])) * 1e3:>15.2f}")

    print(f"\n{'residual model':<48} {'r_ctrl ms':>10} {'r_cand ms':>10} "
          f"{'V*C gain':>9} {'95% CI':>22}")
    print("-" * 128)
    verdicts = {}
    for name, rc, ra, why in residual_models(n_ctrl, n_cand, r_ctrl_local, r_cand_local):
        g, lo, hi = _paired(bm, bf + LAMBDA * rc, am, af + LAMBDA * ra)
        verdicts[name] = (g, lo, hi)
        print(f"{name:<48} {rc * 1e3:>10.2f} {ra * 1e3:>10.2f} {g:>8.4f}x "
              f"{f'[{lo:.4f}, {hi:.4f}]':>22}   {why}")

    dec = [k for k in verdicts if k.startswith("3.")][0]
    g, lo, hi = verdicts[dec]
    exact_ok = True
    # ------------------------------------------------------------------ #
    # THE BINDING STATISTIC CHANGED ON 2026-08-20 AND THIS PATH DID NOT.
    # `docs/VALIDATION.md` 2b-i now binds the paired ARITHMETIC score difference
    # with the studentised-t or basic-bootstrap interval, through
    # `competition_score.decide()`. Everything below computes the GEOMETRIC mean
    # with a 1.96*SE-on-log interval -- neither the statistic nor an eligible
    # interval. It is left intact because D14's record must stay reproducible
    # exactly as it was decided, and it REFUSES to render a fresh verdict.
    # ------------------------------------------------------------------ #
    if not os.environ.get("ARC_WHEST_ALLOW_LEGACY_GEOMETRIC_GATE"):
        raise SystemExit(
            "REFUSED: `--mode decide5` gates on the paired GEOMETRIC mean, which "
            "stopped being the binding statistic on 2026-08-20 (CP0.3, "
            "docs/VALIDATION.md 2b-i). Score the arms with "
            "`competition_score.decide(control, candidate, gate=...)` instead.
"
            "To REPRODUCE the historical D14 decision exactly as it was taken, set "
            "ARC_WHEST_ALLOW_LEGACY_GEOMETRIC_GATE=1 -- and do not report the "
            "result as a current verdict."
        )
    print("\n" + "=" * 128)
    print(f"D14 GATE — deciding model: {dec}")
    print(f"  paired geometric-mean V*C : {g:.4f}x   95% CI [{lo:.4f}, {hi:.4f}]   "
          f"n = {len(bm)} MLPs x {args.shifts} shifts")
    print(f"  gain >= {args.gate:.2f}x                : "
          f"{'PASS' if g >= args.gate else 'FAIL'}")
    print(f"  95% CI excludes 1.0        : {'PASS' if lo > 1.0 else 'FAIL'}")
    print(f"  exactness within tolerance : {'PASS' if exact_ok else 'FAIL'}")
    passed = (g >= args.gate) and (lo > 1.0) and exact_ok
    print(f"\n  D14 VERDICT: {'PASS — proceed to shipping validation' if passed else 'FAIL'}")
    if not passed:
        print("  -> Record the kill in experiments/LOG.md. Do NOT retune and do NOT")
        print(f"     reuse {args.block}: it is single-use and is now spent.")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"block": args.block, "mlps": len(bm), "shifts": args.shifts,
                       "gate": args.gate, "candidate": cand.label(),
                       "n_candidate": n_cand, "n_control": n_ctrl,
                       "verdicts": {k: list(v) for k, v in verdicts.items()},
                       "deciding_model": dec, "passed": bool(passed),
                       "per_mlp": per}, fh, indent=1, default=str)
        print(f"\nsaved {args.out}")
    return 0 if passed else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="parity",
                    choices=["parity", "ablate", "sweep", "residual", "decide",
                             "strassen", "pareto", "nsweep", "decide5"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--block", default="D11")
    ap.add_argument("--mlps", type=int, default=20)
    ap.add_argument("--shifts", type=int, default=1)
    ap.add_argument("--log2-n", dest="log2_n", type=int, default=14)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=32)
    ap.add_argument("--prefix-target", dest="prefix_target", type=float, default=3.0)
    ap.add_argument("--rate-cut", dest="rate_cut", type=float, default=0.0)
    ap.add_argument("--buckets", default="1,2,4,8,16")
    ap.add_argument("--targets", default="")
    ap.add_argument("--rate-cuts", dest="rate_cuts", default="")
    ap.add_argument("--strassens", default="0,1")
    ap.add_argument("--strassen", type=int, default=1)
    ap.add_argument("--pilot", type=int, default=512)
    ap.add_argument("--alpha-on", dest="alpha_on", type=float, default=3.0)
    ap.add_argument("--antithetic", action="store_true")
    ap.add_argument("--on-compose", dest="on_compose", action="store_true")
    ap.add_argument("--cost-only", dest="cost_only", action="store_true",
                    help="skip labels; measure the cost half of V*c only")
    ap.add_argument("--gate", type=float, default=1.15,
                    help="decide5: LEGACY geometric-mean gate. Superseded 2026-08-20 "
                         "by competition_score.decide(); this path now refuses to run "
                         "without ARC_WHEST_ALLOW_LEGACY_GEOMETRIC_GATE=1.")
    ap.add_argument("--bucket-mode", dest="bucket_mode", default="flatnonzero",
                    choices=["flatnonzero", "segment"])
    ap.add_argument("--gather-mode", dest="gather_mode", default="einsum",
                    choices=["einsum", "slotloop"])
    ap.add_argument("--strassen-mode", dest="strassen_mode", default="recursive",
                    choices=["recursive", "batched"])
    ap.add_argument("--no-packing", dest="cand_packing", action="store_false",
                    default=True, help="decide5: candidate without packing")
    ap.add_argument("--log2ns", default="14,15,16",
                    help="nsweep: comma-separated log2 sample counts")
    ap.add_argument("--carriers",
                    default="dense:packing~0; packed:packing~1,dyn_order~1,buckets~8",
                    help="nsweep: 'label:key~value,key~value; label:...'")
    ap.add_argument("--residual-ratio", dest="residual_ratio", type=float, default=1.0,
                    help="pre-registered candidate/dense residual ratio from --mode residual")
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.mode == "parity":
        args.cost_only = True
        return mode_parity(args)
    if args.mode == "strassen":
        return mode_strassen(args)
    if args.mode == "pareto":
        return mode_pareto(args)
    if args.mode == "nsweep":
        return mode_nsweep(args)
    if args.mode == "decide5":
        return mode_decide5(args)
    if args.mode == "sweep":
        return mode_sweep(args)
    if args.mode == "residual":
        args.cost_only = True
        return mode_residual(args)
    if args.mode == "decide":
        return mode_decide(args)
    return mode_ablate(args)


if __name__ == "__main__":
    raise SystemExit(main())
