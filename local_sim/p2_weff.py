"""P2-WEFF: the exact sparse/equality-aware factorized-K3 spelling, priced by flopscope 0.12.

**Research only. Never imported by `estimator.py`.**

Contract: `experiments/LOG.md`, "2026-08-28 -- CONTRACT P2-WEFF", committed at
`08bb4d5` before this file produced a single number.

## What this module answers

One question, and it is a *price*, not an accuracy: can ARC's exact factorized-K3
mathematics (`k_max=3, factor=True, kind=SIMPLE`) be spelled so the whole
16-layer, width-1024 tower costs `<= 0.35 B` counted FLOPs under flopscope 0.12,
at exact parity with the unmodified reference and with zero scalar
materializations in the priced path?

## The two lines that matter, and why only they are re-spelled

`experiments/2026-08-27_pb_rank/cost_ladder.json` attributes **95.35%** of
factored-K3 cost to two operations:

    FactoredTensor3.get_dslice   1.5560e12   47.83%
    FactoredTensor3.contract_W   1.5462e12   47.52%

Both are `Theta(n^2 r)` and both are billed by upstream at `factor = 1.0`.
W-EFF is a *spelling* change confined to them plus the two block assemblies that
feed them. The recurrence itself is untouched, which is exactly why parity can be
proven operation-for-operation against upstream's own implementations rather than
against a re-derivation of the whole algorithm.

## The structure being exploited, stated exactly

Read from `factor_k3.py` at rev `93d091a4`. Every `nonlin` step in SIMPLE mode
appends `2n` columns to `K[3]`, in two blocks:

- the **two-leg** block, `add_factors_((fac1, torch.eye(n)*3, fac3.T))`
  (`factor_k3.py:400-407`) -- mode 2 is `3I`;
- the **from_dstensor** block, whose factors are `(dense, eye, eye)`
  (`factor_k3.py:262-277`) -- modes 2 *and* 3 are `I`.

Naive "sparsity" is worthless here: `contract_W` destroys identity at the first
contraction. What survives is **mode equality**, and it survives forever, because
`contract_W` and `contract_wick_` apply the *same* operator to every mode:

    from_dstensor  (A_D, I,  I )  --W-->  (W A_D, W, W)        --> (A_D, M, M)
    two-leg        (A_L, 3I, C_L) --W-->  (W A_L, 3W, W C_L)   --> (A_L, 3M, C_L)

with **one shared M per birth layer**. So a "lineage" -- the `2n` columns born at
one layer -- carries four distinct `(n, n)` matrices `(A_D, A_L, C_L, M)` where a
dense spelling carries six, and its `(2,1)` diagonal slice needs five matmuls
where a dense spelling needs six (the D block's `{0,1}|{2}` and `{0,2}|{1}`
permutations are *identically the same product*).

That is the whole of W-EFF, and it is a bounded constant factor. It cannot change
`Theta(n^3 L^2)`, because at layer `l` the one-hot-supported columns are `2n` of
`2nl`.

## What is measured versus what is modelled

Measured, executed, billed by the real flopscope 0.12:

- every `contract_W`, `get_dslice`, block assembly and wick contraction, at the
  true width-1024 shapes, for every layer `l = 0..15`, in both spellings and both
  dtypes;
- exact layerwise parity of both `fnp` spellings against **upstream's own torch
  functions** at small shapes and depths;
- scalar materializations, by instrumenting `FlopscopeArray.__bool__`,
  `__float__`, `__int__` and `__index__` -- counted, not inspected.

Modelled: nothing that can be executed. The rank trajectory `r_l = 2n*l` is
*asserted* against upstream, not assumed.

**Scope, stated plainly:** the integrated figure this module reports is a
**lower bound** on the whole path -- it is the sum of the executed dominant lines
(which are 95.35% of upstream's ledger) plus the executed block assemblies. The
small remainder (`pK_to_K`, `DS_harmonic_proj`, `nonlin_sum`, `eval_part`, and
friends) is not re-spelled and not executed here. A lower bound is sufficient to
*fail* an upper-bound screen and insufficient to *pass* one, and the result entry
says so.

Run:

    uv run python local_sim/p2_weff.py --parity
    uv run python local_sim/p2_weff.py --price
    uv run python local_sim/p2_weff.py --all --out experiments/2026-08-28_weff
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# APPEND, never insert: `local_sim/flopscope` and `local_sim/whestbench` are numpy
# shims, and an insert would silently swap them in for the real packages.
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np  # noqa: E402

WIDTH, DEPTH = 1024, 16
BUDGET = 2 ** 41
SCREEN = 0.35                     # the preregistered affordability screen, in budgets
PROBE_SEEDS = (1000301, 1000302, 1000303)   # unregistered; see the contract
UPSTREAM_REV = "93d091a4c26c042bfffa28f2e76a81bc0aba94bb"


# ==========================================================================
# 0. the stack must be the real one
# ==========================================================================

def assert_real_stack() -> dict:
    """The shim must not be answering. Checked, never assumed."""
    import flopscope
    import flopscope.numpy as fnp

    fpath = os.path.abspath(flopscope.__file__)
    shim = os.path.join(_HERE, "flopscope")
    if fpath.startswith(os.path.abspath(shim)):
        raise SystemExit(
            f"REFUSED: flopscope resolved to the local shim at {fpath}. "
            "sys.path was inserted rather than appended; every counted FLOP "
            "below would be fiction."
        )
    ver = getattr(flopscope, "__version__", "?")
    if not str(ver).startswith("0.12"):
        raise SystemExit(f"REFUSED: flopscope {ver}, expected 0.12.x")
    return {"flopscope_version": str(ver), "flopscope_file": fpath,
            "numpy_version": np.__version__, "fnp_file": os.path.abspath(fnp.__file__)}


def verify_upstream() -> dict:
    """The reference checkout is rev 93d091a4 plus exactly the recorded patch."""
    import p2_kprop_bridge as bridge
    src = bridge._src()
    return bridge.verify_upstream(src)


# ==========================================================================
# 1. scalar-materialization instrumentation
# ==========================================================================

class MaterializationCounter:
    """Counts `__bool__` / `__float__` / `__int__` / `__index__` on flopscope arrays.

    A forum-reported deployment invariant: a scalar materialization inside
    `predict` can force a synchronization and eat the 0.4 s residual allowance
    even when the counted mathematics is unchanged. The requirement is zero, and
    zero is *counted*, not asserted by reading the source.
    """

    DUNDERS = ("__bool__", "__float__", "__int__", "__index__")

    def __init__(self) -> None:
        self.counts = {d: 0 for d in self.DUNDERS}
        self._orig: dict[str, object] = {}
        self._cls = None

    def __enter__(self) -> "MaterializationCounter":
        from flopscope._ndarray import FlopscopeArray

        self._cls = FlopscopeArray
        for d in self.DUNDERS:
            orig = getattr(FlopscopeArray, d)
            self._orig[d] = orig

            def make(dname, o):
                def wrapper(self_arr, *a, **kw):
                    counter.counts[dname] += 1
                    return o(self_arr, *a, **kw)
                return wrapper

            counter = self
            setattr(FlopscopeArray, d, make(d, orig))
        return self

    def __exit__(self, *exc) -> None:
        for d, orig in self._orig.items():
            setattr(self._cls, d, orig)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


# ==========================================================================
# 2. the billed primitives -- dense spelling (faithful transcription)
# ==========================================================================
#
# Every function here is a line-for-line transcription of upstream at rev
# 93d091a4, expressed in `fnp` so flopscope prices it. Nothing is optimised.

def contract_W_dense(fnp, W, factors):
    """`FactoredTensor.contract_W` (`factor_k3.py:181-192`): `W @ factor`, three times."""
    return tuple(W @ f for f in factors)


def contract_wick_dense(fnp, w, factors):
    """`FactoredTensor.contract_wick_` (`factor_k3.py:194-198`): `factor * wick[:, None]`."""
    return tuple(f * w[:, None] for f in factors)


def _unfactor_dense(fnp, mats):
    """`_unfactor` (`factor_k3.py:25-29`) for the two block shapes we ever need."""
    if len(mats) == 1:
        return fnp.sum(mats[0], axis=1)
    if len(mats) == 2:
        return mats[0] @ mats[1].T
    raise NotImplementedError(len(mats))


def dslice_dense(fnp, factors, part):
    """`_factored_get_dslice` (`factor_k3.py:31-54`), d = 3.

    `(3,)`  -> one block  {0,1,2}                         -> sum_r A*B*C
    `(2,1)` -> three blocks {0,1}|{2}, {0,2}|{1}, {1,2}|{0}
    """
    A, B, C = factors
    if part == (3,):
        ret = _unfactor_dense(fnp, (A * B * C,))
        coef = math.factorial(3) / math.factorial(3)          # = 1.0
        return coef * ret                                     # (3,) has no repeats to zero
    if part == (2, 1):
        ret = _unfactor_dense(fnp, (A * B, C))
        ret = ret + _unfactor_dense(fnp, (A * C, B))
        ret = ret + _unfactor_dense(fnp, (B * C, A))
        coef = (math.factorial(2) * math.factorial(1)) / math.factorial(3)   # = 1/3
        return coef * _zero_repeated_2d(fnp, ret)
    raise NotImplementedError(part)


def _zero_repeated_2d(fnp, M):
    """`zero_repeated` (`diagslice.py:31-35`) on a `(2,1)` slice: zero the diagonal.

    Immutable spelling -- flopscope arrays do not accept item assignment.
    """
    n = M.shape[0]
    return M * (1.0 - fnp.eye(n, dtype=M.dtype))


def twoleg_factors(fnp, w1, WK11):
    """The SIMPLE-mode two-leg block (`factor_k3.py:389-407`).

    `legs = [(1, 1, WK_11, 1)]`, `budget = 2`, so `rights == legs` and

        fac1 = w(1)[:, None] * WK_11
        fac2 = eye(n) * 3
        fac3 = (w(1)[:, None] * w(1)[None, :] * WK_11).T
    """
    n = WK11.shape[0]
    fac1 = w1[:, None] * WK11
    fac2 = fnp.eye(n, dtype=WK11.dtype) * 3.0
    fac3 = (w1[:, None] * w1[None, :] * WK11).T
    return (fac1, fac2, fac3)


def from_dstensor_factors(fnp, s3, s21):
    """`FactoredTensor.from_dstensor` (`factor_k3.py:261-277`): `(dense, eye, eye)`."""
    n = s3.shape[0]
    eye = fnp.eye(n, dtype=s3.dtype)
    A = s3[:, None] * eye + s21.T * 3.0
    return (A, eye, eye)


def contract_W_basic(fnp, W, core):
    """The K2-level line (`tensor_utils.contract_W_basic`): `W @ core @ W.T`."""
    return (W @ core) @ W.T


# ==========================================================================
# 3. the billed primitives -- W-EFF spelling (exact same mathematics)
# ==========================================================================

class Lineage:
    """The `2n` columns born at one layer, in mode-equality form.

    Represents exactly

        Sym( sum_j A_D[:,j] (x) M[:,j] (x) M[:,j] )
      + Sym( sum_j A_L[:,j] (x) 3 M[:,j] (x) C_L[:,j] )

    `m_is_identity` means `M == I` symbolically: the array is never built, and
    `W @ I` is `W` with no arithmetic. `has_D` / `has_L` say which of the two
    blocks this lineage carries: a freshly born two-leg block exists on its own
    until its from_dstensor partner is added at the end of the same layer.
    """

    __slots__ = ("A_D", "A_L", "C_L", "M", "m_is_identity", "has_D", "has_L")

    def __init__(self, A_D=None, A_L=None, C_L=None, M=None,
                 m_is_identity=False, has_D=True, has_L=True):
        self.A_D, self.A_L, self.C_L, self.M = A_D, A_L, C_L, M
        self.m_is_identity = m_is_identity
        self.has_D = has_D          # the from_dstensor block (A_D, M, M)
        self.has_L = has_L          # the two-leg block      (A_L, 3M, C_L)

    def rank(self, n: int) -> int:
        return n * (int(self.has_D) + int(self.has_L))


def contract_W_weff(fnp, W, lin: Lineage) -> Lineage:
    """`contract_W` on one lineage.

    Dense would be six `(n,n) @ (n,n)` matmuls for the same `2n` columns.
    This is four -- three when `M` is still symbolically `I`, because `W @ I`
    is `W` and `W @ 3I` is `3W`, and both reuse the *same* advanced `M`.
    """
    if lin.m_is_identity:
        M = W                       # W @ I == W, exactly, at zero FLOPs
        m_id = False
    else:
        M = W @ lin.M
        m_id = False
    return Lineage(
        A_D=(W @ lin.A_D) if lin.A_D is not None else None,
        A_L=(W @ lin.A_L) if lin.A_L is not None else None,
        C_L=(W @ lin.C_L) if lin.C_L is not None else None,
        M=M, m_is_identity=m_id, has_D=lin.has_D, has_L=lin.has_L,
    )


def contract_wick_weff(fnp, w, lin: Lineage) -> Lineage:
    """`contract_wick_` on one lineage: one scale of `M`, shared by both blocks.

    The `3` on the two-leg mode stays a scalar and is never materialised into an
    `(n,n)` array, so `3M` costs nothing beyond `M`.
    """
    col = w[:, None]
    if lin.m_is_identity:
        # `M == I` scaled by the wick vector is `diag(w)`, NOT the identity.
        # Unreachable in upstream's order -- `contract_W` always precedes
        # `contract_wick_`, so `M` is materialised by then -- but the first
        # version of this function silently returned `M = None` with
        # `m_is_identity = False`, which is a corrupt lineage. The commutation
        # test added after an independent review found it. Implemented exactly
        # rather than guarded, so the test can exercise the fresh cases too.
        base = lin.A_D if lin.A_D is not None else lin.A_L
        n = base.shape[0]
        M = fnp.eye(n, dtype=base.dtype) * col
    else:
        M = (lin.M * col) if lin.M is not None else None
    return Lineage(
        A_D=(lin.A_D * col) if lin.A_D is not None else None,
        A_L=(lin.A_L * col) if lin.A_L is not None else None,
        C_L=(lin.C_L * col) if lin.C_L is not None else None,
        M=M,
        m_is_identity=False, has_D=lin.has_D, has_L=lin.has_L,
    )


def dslice_weff(fnp, lineages, n, part, dtype):
    """`get_dslice` over a list of lineages -- exact, and five matmuls per lineage.

    The `(2,1)` slice is linear in the columns, so it splits over lineages
    exactly. Per lineage:

      D block `(A_D, M, M)`
        {0,1}|{2} -> (A_D * M) @ M.T
        {0,2}|{1} -> (A_D * M) @ M.T      <-- *identically the same product*
        {1,2}|{0} -> (M * M) @ A_D.T
      two matmuls, not three.

      L block `(A_L, 3M, C_L)` -- three distinct products; the 3 is a scalar.

      A *freshly born* two-leg block has `M == I`, and then every product is
      diagonal or a row-scale, so all three permutations are `O(n^2)`.
    """
    acc = None

    def add(x):
        nonlocal acc
        acc = x if acc is None else acc + x

    for lin in lineages:
        if part == (3,):
            if lin.has_L:
                if lin.m_is_identity:
                    # M == I: sum_r A_L[i,r] 3 I[i,r] C_L[i,r] = 3 A_L[i,i] C_L[i,i]
                    add(3.0 * (fnp.diagonal(lin.A_L) * fnp.diagonal(lin.C_L)))
                else:
                    add(3.0 * fnp.sum(lin.A_L * lin.M * lin.C_L, axis=1))
            if lin.has_D:
                if lin.m_is_identity:
                    add(fnp.diagonal(lin.A_D))
                else:
                    add(fnp.sum(lin.A_D * lin.M * lin.M, axis=1))
            continue

        if part != (2, 1):
            raise NotImplementedError(part)

        if lin.has_D:                                  # the D block
            if lin.m_is_identity:
                # M == I: {0,1}|{2} and {0,2}|{1} are both diag(diagonal(A_D));
                #         {1,2}|{0} is (I*I) @ A_D.T == A_D.T
                dA = fnp.diagonal(lin.A_D)
                add(2.0 * (dA[:, None] * fnp.eye(n, dtype=dtype)))
                add(lin.A_D.T)
            else:
                AM = lin.A_D * lin.M
                add(2.0 * (AM @ lin.M.T))
                add((lin.M * lin.M) @ lin.A_D.T)
        if lin.has_L:                                  # the two-leg block
            if lin.m_is_identity:
                # M == I: (A*3I) is diag(A)*3 on the diagonal, so
                #   {0,1}|{2} -> 3 * diag(A_L)[:,None] * C_L.T
                #   {0,2}|{1} -> 3 * (A_L * C_L)
                #   {1,2}|{0} -> 3 * diag(C_L)[:,None] * A_L.T
                dA = fnp.diagonal(lin.A_L)
                dC = fnp.diagonal(lin.C_L)
                add(3.0 * (dA[:, None] * lin.C_L.T))
                add(3.0 * (lin.A_L * lin.C_L))
                add(3.0 * (dC[:, None] * lin.A_L.T))
            else:
                AM = lin.A_L * lin.M
                add(3.0 * (AM @ lin.C_L.T))
                add(3.0 * ((lin.A_L * lin.C_L) @ lin.M.T))
                add(3.0 * ((lin.M * lin.C_L) @ lin.A_L.T))

    if part == (3,):
        return acc
    coef = (math.factorial(2) * math.factorial(1)) / math.factorial(3)
    return coef * _zero_repeated_2d(fnp, acc)


# ==========================================================================
# 4. building the lineage state that a layer actually sees
# ==========================================================================

def _rand(fnp, rng, shape, dtype):
    return fnp.array(np.asarray(rng.standard_normal(shape), dtype=dtype))


def _rank_at(n: int, layer: int) -> int:
    """`r_l = 2 n l`. Asserted against upstream by `--verify-rank`, never assumed."""
    return 2 * n * layer


# ==========================================================================
# 5. pricing one layer, both spellings
# ==========================================================================

def price_layer(layer: int, n: int, dtype_name: str, rng) -> dict:
    """Bill every dominant operation of one layer, in both spellings.

    Operand *synthesis* is billed in a throwaway context and excluded: the inputs
    of layer `l` are outputs of layer `l-1`, already billed there. Only the
    layer's own operations are charged.
    """
    import flopscope as flops
    import flopscope.numpy as fnp

    dtype = np.float32 if dtype_name == "f32" else np.float64
    r = _rank_at(n, layer)
    out: dict = {"layer": layer, "n": n, "dtype": dtype_name, "rank_in": r, "lines": {}}

    leaks: list = []

    def _assert_instrumented(tag, obj, depth=0):
        """Hard rule 3, enforced rather than asserted in prose.

        The contract claimed "the module asserts every intermediate in the
        priced path is a flopscope array". It did not -- only `assert_real_stack`
        was implemented. An independent review caught that, and this is the
        repair. Every value a priced op returns is walked and required to be a
        `FlopscopeArray`; a bare `np.ndarray` or a numpy scalar means arithmetic
        escaped the meter, which is a hard-rule violation, not a saving.

        It also closes the materialization counter's known blind spot in the one
        direction that matters here: integer indexing (`a[0]`) drops the
        `FlopscopeArray` wrapper and yields a bare `np.float64`, which
        `MaterializationCounter` cannot see -- but which this check would.
        """
        from flopscope._ndarray import FlopscopeArray

        if obj is None or depth > 3:
            return
        if isinstance(obj, (tuple, list)):
            for x in obj:
                _assert_instrumented(tag, x, depth + 1)
            return
        if isinstance(obj, Lineage):
            for x in (obj.A_D, obj.A_L, obj.C_L, obj.M):
                _assert_instrumented(tag, x, depth + 1)
            return
        if isinstance(obj, FlopscopeArray):
            return
        if isinstance(obj, (np.ndarray, np.generic)):
            leaks.append({"line": tag, "type": type(obj).__name__,
                          "shape": getattr(obj, "shape", None)})

    def bill(tag, fn):
        gc.collect()
        with flops.BudgetContext(flop_budget=BUDGET * 64, quiet=True) as c:
            res = fn()
        out["lines"][tag] = int(c.flops_used)
        _assert_instrumented(tag, res)
        return res

    def setup(fn):
        with flops.BudgetContext(flop_budget=BUDGET * 64, quiet=True):
            return fn()

    # ---- shared per-layer operands -------------------------------------
    W = setup(lambda: _rand(fnp, rng, (n, n), dtype))
    w1 = setup(lambda: fnp.array(np.abs(rng.standard_normal(n)).astype(dtype) + 0.5))
    WK11 = setup(lambda: _rand(fnp, rng, (n, n), dtype))
    core = setup(lambda: _rand(fnp, rng, (n, n), dtype))
    s3 = setup(lambda: _rand(fnp, rng, (n,), dtype))
    s21 = setup(lambda: _rand(fnp, rng, (n, n), dtype))

    # ---- the K2-level line, identical in both spellings -----------------
    bill("contract_W_basic", lambda: contract_W_basic(fnp, W, core))

    # =================== DENSE spelling ==================================
    if r > 0:
        facs = setup(lambda: tuple(_rand(fnp, rng, (n, r), dtype) for _ in range(3)))
        wf = bill("dense/contract_W", lambda: contract_W_dense(fnp, W, facs))
        del facs
        wf = bill("dense/contract_wick", lambda: contract_wick_dense(fnp, w1, wf))
    else:
        wf = None
        out["lines"]["dense/contract_W"] = 0
        out["lines"]["dense/contract_wick"] = 0

    tl = bill("dense/twoleg_assembly", lambda: twoleg_factors(fnp, w1, WK11))

    # The slice is billed AGED / FRESH separately, because that is exactly how
    # UPSTREAM bins it and the entry's first version compared the wrong bins.
    #
    # `eval_part` in section 3.1 calls `diagslice(WK[3], ...)`, which populates
    # `WK[3].repeated.slices` -- billed to `FactoredTensor3.get_dslice` at rank
    # `2nl`. `pK_111 = WK[3].clone()` then copies that POPULATED cache, so
    # `add_factors_` (`factor_k3.py:145-146`) updates it incrementally with the
    # fresh two-leg block's own `(2,1)` slice -- and because that call sits
    # inside `with flop_name('nonlin_sum 111 factored')` (`factor_k3.py:371`),
    # upstream bills the fresh block to `nonlin_sum`, not to `get_dslice`.
    # `get_repeated()` later finds both parts cached and pays nothing.
    #
    # A single call on the rank-`(2nl + n)` concatenation computes the same
    # thing; splitting it here makes the two meters comparable bin for bin.
    # Cost of splitting: one extra `zero_repeated` per call (`3n^2`), which is
    # counted and negligible, and is why the split total is a few n^2 above the
    # unsplit one rather than equal to it.
    if wf is not None:
        cat = bill("dense/add_factors_concat",
                   lambda: tuple(fnp.concatenate([wf[i], tl[i]], axis=1) for i in range(3)))
        bill("dense/get_dslice_21_aged", lambda: dslice_dense(fnp, wf, (2, 1)))
        bill("dense/get_dslice_3_aged", lambda: dslice_dense(fnp, wf, (3,)))
    else:
        cat = tl
        out["lines"]["dense/add_factors_concat"] = 0
        out["lines"]["dense/get_dslice_21_aged"] = 0
        out["lines"]["dense/get_dslice_3_aged"] = 0
    bill("dense/get_dslice_21_fresh", lambda: dslice_dense(fnp, tl, (2, 1)))
    bill("dense/get_dslice_3_fresh", lambda: dslice_dense(fnp, tl, (3,)))
    del wf, tl, cat
    gc.collect()
    bill("dense/from_dstensor", lambda: from_dstensor_factors(fnp, s3, s21))

    # =================== W-EFF spelling ==================================
    # `layer` aged lineages (born at layers 0..layer-1) each carry four dense
    # (n, n) matrices; the youngest still has M == I at the moment of contraction.
    lins = []
    for j in range(layer):
        m_id = (j == layer - 1)
        lins.append(Lineage(
            A_D=setup(lambda: _rand(fnp, rng, (n, n), dtype)),
            A_L=setup(lambda: _rand(fnp, rng, (n, n), dtype)),
            C_L=setup(lambda: _rand(fnp, rng, (n, n), dtype)),
            M=None if m_id else setup(lambda: _rand(fnp, rng, (n, n), dtype)),
            m_is_identity=m_id,
        ))
    if lins:
        lins = bill("weff/contract_W", lambda: [contract_W_weff(fnp, W, L) for L in lins])
        lins = bill("weff/contract_wick", lambda: [contract_wick_weff(fnp, w1, L) for L in lins])
    else:
        out["lines"]["weff/contract_W"] = 0
        out["lines"]["weff/contract_wick"] = 0

    tl2 = bill("weff/twoleg_assembly", lambda: _twoleg_lineage(fnp, w1, WK11))
    out["lines"]["weff/add_factors_concat"] = 0        # a list append; no arithmetic
    gc.collect()
    # same AGED / FRESH split, so the saving is attributable line by line
    bill("weff/get_dslice_21_aged", lambda: dslice_weff(fnp, lins, n, (2, 1), dtype)
         if lins else None)
    bill("weff/get_dslice_3_aged", lambda: dslice_weff(fnp, lins, n, (3,), dtype)
         if lins else None)
    bill("weff/get_dslice_21_fresh", lambda: dslice_weff(fnp, [tl2], n, (2, 1), dtype))
    bill("weff/get_dslice_3_fresh", lambda: dslice_weff(fnp, [tl2], n, (3,), dtype))
    del lins, tl2
    gc.collect()
    bill("weff/from_dstensor", lambda: _from_dstensor_lineage(fnp, s3, s21))

    out["dense_total"] = sum(v for k, v in out["lines"].items() if not k.startswith("weff/"))
    out["weff_total"] = sum(v for k, v in out["lines"].items() if not k.startswith("dense/"))
    out["uninstrumented_leaks"] = leaks
    return out


def _twoleg_lineage(fnp, w1, WK11):
    """The two-leg block as a lineage: mode 2 is symbolically `3I`, never built."""
    fac1 = w1[:, None] * WK11
    fac3 = (w1[:, None] * w1[None, :] * WK11).T
    return Lineage(A_L=fac1, C_L=fac3, M=None, m_is_identity=True,
                   has_D=False, has_L=True)


def _from_dstensor_lineage(fnp, s3, s21):
    """`from_dstensor` as a lineage: modes 2 and 3 are symbolically `I`, never built."""
    n = s3.shape[0]
    A = s3[:, None] * fnp.eye(n, dtype=s3.dtype) + s21.T * 3.0
    return Lineage(A_D=A, M=None, m_is_identity=True, has_D=True, has_L=False)


# ==========================================================================
# 5b. memory accounting -- reported, never claimed enforced
# ==========================================================================

def peak_memory() -> dict:
    """Peak working set and peak commit for this process.

    Windows has no `RLIMIT_AS`, so the 8,192 MB solution-memory limit is an
    accounting constraint here and never a test passed. `RLIMIT_AS` caps
    virtual address space, which exceeds both counters below by an amount no
    Windows counter reports, so any headroom computed from these is an
    UPPER BOUND on the true headroom, not a safety margin.
    """
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD),
                        ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        c = PMC()
        c.cb = ctypes.sizeof(PMC)
        fn = None
        for lib, name in ((ctypes.windll.kernel32, "K32GetProcessMemoryInfo"),
                          (ctypes.windll.psapi, "GetProcessMemoryInfo")):
            try:
                fn = getattr(lib, name)
                break
            except AttributeError:
                continue
        if fn is None:
            return {"available": False, "error": "no GetProcessMemoryInfo"}
        fn.restype = wintypes.BOOL
        fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        ok = fn(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        if not ok:
            return {"available": False,
                    "error": f"GetProcessMemoryInfo failed, GetLastError="
                             f"{ctypes.get_last_error()}"}
        return {"available": True,
                "peak_working_set_mb": c.PeakWorkingSetSize / 2 ** 20,
                "peak_commit_mb": c.PeakPagefileUsage / 2 ** 20,
                "limit_mb": 8192,
                "upper_bound_headroom_x": 8192.0 / max(
                    c.PeakPagefileUsage / 2 ** 20, 1e-9),
                "enforced": False,
                "note": "Windows has no RLIMIT_AS; this is accounting, not a test."}
    except Exception as e:                                        # noqa: BLE001
        return {"available": False, "error": repr(e)}


def factor_storage_bytes(n: int, depth: int, itemsize: int) -> dict:
    """Exact factor storage from the asserted rank trajectory, both spellings.

    Dense carries three `(n, r_l)` factors; W-EFF carries four `(n, n)`
    matrices per lineage, which is the same `6 n^2` per lineage -> hmm, no:
    four, not six. Both are computed from shapes, not measured, and the
    shapes are the ones the priced path actually allocated.
    """
    peak_l = depth - 1
    r = 2 * n * peak_l
    dense = 3 * n * (r + n) * itemsize
    weff = 4 * n * n * itemsize * peak_l + 2 * n * n * itemsize
    return {"peak_layer": peak_l, "rank_at_peak": r,
            "dense_factor_bytes": dense, "weff_factor_bytes": weff,
            "dense_factor_mb": dense / 2 ** 20, "weff_factor_mb": weff / 2 ** 20}


# ==========================================================================
# 6. parity -- against upstream's own functions
# ==========================================================================

def parity_emit(out_dir, shapes=((8, 3), (12, 3), (16, 4))) -> dict:
    """Run both `fnp` spellings on a deterministic small-shape chain and dump
    every factor and every slice, so the torch venv can score them.

    flopscope and torch do not live in the same environment here, so parity
    against upstream is mediated by an artifact rather than by an import. That
    is strictly stronger than an in-process comparison: upstream consumes the
    *identical bytes* our spelling produced, with no re-randomisation between.

    Also computes, in-process, the W-EFF-versus-dense comparison, which needs
    no torch at all.
    """
    import flopscope as flops
    import flopscope.numpy as fnp

    os.makedirs(out_dir, exist_ok=True)
    blobs: dict = {}
    rows = []
    worst_sp = 0.0

    for n, depth in shapes:
        rng = np.random.default_rng(20260828 + n)
        with flops.BudgetContext(flop_budget=10 ** 15, quiet=True):
            def R(shape):
                return np.asarray(rng.standard_normal(shape), dtype=np.float64)

            lins = [Lineage(A_D=fnp.array(R((n, n))), A_L=fnp.array(R((n, n))),
                            C_L=fnp.array(R((n, n))), M=fnp.array(R((n, n))),
                            m_is_identity=False) for _ in range(2)]

            for layer in range(depth):
                Wn = R((n, n)) / math.sqrt(n)
                w1n = np.abs(R(n)) + 0.5
                WK11n = R((n, n))
                W, w1, WK11 = fnp.array(Wn), fnp.array(w1n), fnp.array(WK11n)

                # ---------- one W-EFF layer ----------
                lins = [contract_W_weff(fnp, W, L) for L in lins]
                lins = [contract_wick_weff(fnp, w1, L) for L in lins]
                lins = lins + [_twoleg_lineage(fnp, w1, WK11)]

                sp3 = np.asarray(dslice_weff(fnp, lins, n, (3,), np.float64))
                sp21 = np.asarray(dslice_weff(fnp, lins, n, (2, 1), np.float64))

                # ---------- the SAME state, densely ----------
                facs = _lineages_to_dense(fnp, lins, n, np.float64)
                dn3 = np.asarray(dslice_dense(fnp, facs, (3,)))
                dn21 = np.asarray(dslice_dense(fnp, facs, (2, 1)))
                dnc = tuple(np.asarray(x) for x in contract_W_dense(fnp, W, facs))

                tag = "n%d_l%d" % (n, layer)
                blobs[tag + "_A"] = np.asarray(facs[0])
                blobs[tag + "_B"] = np.asarray(facs[1])
                blobs[tag + "_C"] = np.asarray(facs[2])
                blobs[tag + "_W"] = Wn
                blobs[tag + "_dn3"] = dn3
                blobs[tag + "_dn21"] = dn21
                blobs[tag + "_dnc0"] = dnc[0]
                blobs[tag + "_dnc1"] = dnc[1]
                blobs[tag + "_dnc2"] = dnc[2]

                e_sp3 = _relerr(sp3, dn3)
                e_sp21 = _relerr(sp21, dn21)
                worst_sp = max(worst_sp, e_sp3, e_sp21)
                rows.append({"n": n, "layer": layer,
                             "rank_dense": int(facs[0].shape[1]),
                             "rank_lineage": int(sum(L.rank(n) for L in lins)),
                             "weff_vs_dense_dslice3": e_sp3,
                             "weff_vs_dense_dslice21": e_sp21})

                # fold in the from_dstensor partner, exactly as
                # `K_out[3] = pK_111 + from_dstensor(K_ds[3])` does
                fd = _from_dstensor_lineage(fnp, fnp.array(R(n)), fnp.array(R((n, n))))
                tl = lins[-1]
                assert (not tl.has_D) and (not fd.has_L)
                assert tl.m_is_identity and fd.m_is_identity
                lins[-1] = Lineage(A_D=fd.A_D, A_L=tl.A_L, C_L=tl.C_L,
                                   M=None, m_is_identity=True)

    path = os.path.join(out_dir, "parity_fnp.npz")
    np.savez_compressed(path, **blobs)
    return {"npz": path, "rows": rows, "shapes": [list(x) for x in shapes],
            "worst_weff_vs_dense": worst_sp,
            "tol_weff": 1e-12,
            "weff_vs_dense_pass": bool(worst_sp <= 1e-12),
            "rank_matches": all(r["rank_dense"] == r["rank_lineage"] for r in rows)}


def parity_upstream(out_dir) -> dict:
    """Score the emitted factors with UPSTREAM's own torch functions.

    Runs in the torch venv. Consumes the identical arrays the `fnp` spelling
    produced; computes nothing of its own beyond calling upstream.
    """
    import torch
    import p2_kprop_bridge as bridge
    bridge._src()
    from mlp_kprop.factor_k3 import FactoredTensor, _factored_get_dslice

    z = np.load(os.path.join(out_dir, "parity_fnp.npz"))
    tags = sorted({k.rsplit("_", 1)[0] for k in z.files})
    worst3 = worst21 = worstc = 0.0
    rows = []
    for tag in tags:
        A = torch.as_tensor(z[tag + "_A"], dtype=torch.float64)
        B = torch.as_tensor(z[tag + "_B"], dtype=torch.float64)
        C = torch.as_tensor(z[tag + "_C"], dtype=torch.float64)
        W = torch.as_tensor(z[tag + "_W"], dtype=torch.float64)
        n = A.shape[0]

        up3 = _factored_get_dslice((A, B, C), (3,)).numpy()
        up21 = _factored_get_dslice((A, B, C), (2, 1)).numpy()
        FT = FactoredTensor(n=n, d=3, factors=(A, B, C))
        upc = tuple(t.numpy() for t in FT.contract_W(W)._factors)

        e3 = _relerr(z[tag + "_dn3"], up3)
        e21 = _relerr(z[tag + "_dn21"], up21)
        ec = max(_relerr(z[tag + "_dnc%d" % i], upc[i]) for i in range(3))
        worst3, worst21, worstc = max(worst3, e3), max(worst21, e21), max(worstc, ec)
        rows.append({"tag": tag, "rank": int(A.shape[1]),
                     "dslice3": e3, "dslice21": e21, "contract_W": ec})

    worst = max(worst3, worst21, worstc)
    res = {"rows": rows, "worst_dslice3": worst3, "worst_dslice21": worst21,
           "worst_contract_W": worstc, "worst_fnp_vs_upstream": worst,
           "tol_upstream": 1e-10, "pass": bool(worst <= 1e-10),
           "upstream": bridge.verify_upstream(bridge._src())}
    with open(os.path.join(out_dir, "parity_upstream.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=float)
    return res


def _lineages_to_dense(fnp, lins, n, dtype):
    """Materialise the lineage list as the concatenated `(A, B, C)` a dense
    spelling would carry. This is the *definition* the parity test compares to."""
    A_cols, B_cols, C_cols = [], [], []
    eye = fnp.eye(n, dtype=dtype)
    for lin in lins:
        M = eye if lin.m_is_identity else lin.M
        if lin.has_D:
            A_cols.append(lin.A_D); B_cols.append(M); C_cols.append(M)
        if lin.has_L:
            A_cols.append(lin.A_L); B_cols.append(3.0 * M); C_cols.append(lin.C_L)
    return (fnp.concatenate(A_cols, axis=1),
            fnp.concatenate(B_cols, axis=1),
            fnp.concatenate(C_cols, axis=1))


def _relerr(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    den = max(float(np.max(np.abs(b))), 1e-300)
    return float(np.max(np.abs(a - b)) / den)



def commutation_parity(shapes=((8,), (12,), (16,)), tol=1e-12) -> dict:
    """Close the hole: is `contract_W_weff` itself right?

    `parity_emit` builds its dense reference with `_lineages_to_dense` from the
    ALREADY-W-EFF-PROPAGATED state, so `contract_W_weff` and
    `contract_wick_weff` were never compared against `contract_W_dense` /
    `contract_wick_dense`. A bug in either would have propagated identically
    into both sides of that comparison and been invisible -- the `contract_W`
    row of the parity table tests the DENSE transcription against upstream, not
    the W-EFF one. An independent review found this; this is the repair.

    The right test is commutation. Materialising a lineage list to dense factors
    is a map `phi`. The claim is that W-EFF's operator is the same operator:

        phi( weff_op(S) )  ==  dense_op( phi(S) )

    Tested on every structural case the tower actually produces -- an aged
    lineage, a freshly born two-leg block (`M == I`), a fresh from_dstensor
    block (`M == I`), and a mixture -- because the `M == I` branches are exactly
    the ones that skip a matmul and are therefore where a bug would hide.
    """
    import flopscope as flops
    import flopscope.numpy as fnp

    rows = []
    worst_cw = worst_ck = 0.0
    for (n,) in shapes:
        rng = np.random.default_rng(90210 + n)
        with flops.BudgetContext(flop_budget=10 ** 15, quiet=True):
            def R(sh):
                return fnp.array(np.asarray(rng.standard_normal(sh), dtype=np.float64))

            aged = Lineage(A_D=R((n, n)), A_L=R((n, n)), C_L=R((n, n)), M=R((n, n)))
            fresh_L = _twoleg_lineage(fnp, R(n), R((n, n)))
            fresh_D = _from_dstensor_lineage(fnp, R(n), R((n, n)))
            cases = {"aged": [aged], "fresh_twoleg": [fresh_L],
                     "fresh_from_dstensor": [fresh_D],
                     "mixed": [aged, fresh_L, fresh_D]}

            for name, S in cases.items():
                W = R((n, n))
                w = R(n)

                # contract_W
                lhs = _lineages_to_dense(
                    fnp, [contract_W_weff(fnp, W, L) for L in S], n, np.float64)
                rhs = contract_W_dense(fnp, W, _lineages_to_dense(fnp, S, n, np.float64))
                e_cw = max(_relerr(np.asarray(a), np.asarray(b))
                           for a, b in zip(lhs, rhs))

                # contract_wick
                lhs2 = _lineages_to_dense(
                    fnp, [contract_wick_weff(fnp, w, L) for L in S], n, np.float64)
                rhs2 = contract_wick_dense(fnp, w,
                                           _lineages_to_dense(fnp, S, n, np.float64))
                e_ck = max(_relerr(np.asarray(a), np.asarray(b))
                           for a, b in zip(lhs2, rhs2))

                worst_cw = max(worst_cw, e_cw)
                worst_ck = max(worst_ck, e_ck)
                rows.append({"n": n, "case": name,
                             "contract_W_commutes": e_cw,
                             "contract_wick_commutes": e_ck})

    return {"rows": rows, "worst_contract_W": worst_cw,
            "worst_contract_wick": worst_ck, "tol": tol,
            "pass": bool(worst_cw <= tol and worst_ck <= tol)}


def symmetry_discount_probe() -> dict:
    """The contract's own open sub-question, which the first pass never measured.

    The contract said flopscope's `operand_symmetries` / `identity_pattern`
    discounts were "an open sub-question this slice measures rather than
    assumes", and then did not measure them. An independent review called that
    an unfulfilled contract item. This is it.

    `contract_W_basic` computes `W core W^T` with `core` symmetric and the
    result symmetric. Upstream evidently takes a discount there -- it bills
    `3 n^3` where this spelling bills `4 n^3`. The question is whether
    flopscope grants a LEGITIMATE one, i.e. one that reflects real algebraic
    structure rather than a price-table quirk.
    """
    import flopscope as flops
    import flopscope.numpy as fnp

    n = 1024
    res = {}
    with flops.BudgetContext(flop_budget=10 ** 15, quiet=True):
        rng = np.random.default_rng(5150)
        Wn = np.asarray(rng.standard_normal((n, n)), dtype=np.float32)
        Cn = np.asarray(rng.standard_normal((n, n)), dtype=np.float32)
        Cn = (Cn + Cn.T) / 2.0
        W, C = fnp.array(Wn), fnp.array(Cn)

    with flops.BudgetContext(flop_budget=10 ** 15, quiet=True) as c1:
        plain = (W @ C) @ W.T
    res["plain"] = int(c1.flops_used)

    try:
        with flops.BudgetContext(flop_budget=10 ** 15, quiet=True) as c2:
            Cs = flops.as_symmetric(C, symmetry=(0, 1))
            sym = (W @ Cs) @ W.T
        res["as_symmetric_core"] = int(c2.flops_used)
        res["ratio"] = res["as_symmetric_core"] / res["plain"]
        res["values_agree"] = float(_relerr(np.asarray(sym), np.asarray(plain)))
        res["available"] = True
    except Exception as e:                                        # noqa: BLE001
        res["available"] = False
        res["error"] = repr(e)

    res["upstream_contract_W_basic"] = 51_589_939_200
    res["note"] = ("Measured, not assumed. A discount here is worth ~0.008 B on "
                   "the tower and bears on the m<=3 window margin; it cannot "
                   "rescue a 3.06x screen miss.")
    return res


# ==========================================================================
# 7. mutation tests -- the shortcut must break when the block structure breaks
# ==========================================================================

def mutations(n=12) -> dict:
    """Each mutation must make W-EFF disagree with the dense reference.

    A mutation that still passes is not a reassurance; it is evidence that the
    parity check cannot see that class of defect, and it is reported as such.
    """
    import flopscope as flops
    import flopscope.numpy as fnp

    rng = np.random.default_rng(4242)
    with flops.BudgetContext(flop_budget=10 ** 15, quiet=True):
        def R(s):
            return fnp.array(np.asarray(rng.standard_normal(s), dtype=np.float64))

        lins = [Lineage(A_D=R((n, n)), A_L=R((n, n)), C_L=R((n, n)), M=R((n, n)))
                for _ in range(3)]
        lins.append(_twoleg_lineage(fnp, R(n) * 0 + 1.3, R((n, n))))
        facs = _lineages_to_dense(fnp, lins, n, np.float64)
        ref = np.asarray(dslice_dense(fnp, facs, (2, 1)))

        base = _relerr(np.asarray(dslice_weff(fnp, lins, n, (2, 1), np.float64)), ref)

        results = {"baseline_relerr": base, "baseline_pass": base <= 1e-12}

        # M1 -- drop a lineage block
        m = _relerr(np.asarray(dslice_weff(fnp, lins[:-1], n, (2, 1), np.float64)), ref)
        results["M1_drop_block"] = {"relerr": m, "detected": m > 1e-8}

        # M2 -- duplicate a lineage block
        m = _relerr(np.asarray(dslice_weff(fnp, lins + [lins[0]], n, (2, 1), np.float64)), ref)
        results["M2_duplicate_block"] = {"relerr": m, "detected": m > 1e-8}

        # M3 -- permute the COLUMNS of one mode, breaking the rank-1 term pairing.
        #
        # NOTE, and it matters: swapping whole MODES (A_L <-> C_L) is *not* a
        # usable control. `_factored_get_dslice` sums over every block
        # permutation, so the slice of a symmetric tensor is invariant under
        # mode order -- measured at 2.4e-16, i.e. exactly invariant. A control
        # that cannot fail is not a control, so it is replaced here by a
        # column permutation, which genuinely changes the tensor.
        perm = list(lins)
        p0 = perm[0]
        idx = fnp.array(np.roll(np.arange(n), 1))
        perm[0] = Lineage(A_D=p0.A_D, A_L=p0.A_L, C_L=p0.C_L[:, np.roll(np.arange(n), 1)],
                          M=p0.M)
        m = _relerr(np.asarray(dslice_weff(fnp, perm, n, (2, 1), np.float64)), ref)
        results["M3_permute_columns"] = {"relerr": m, "detected": m > 1e-8}

        # M3b -- the recorded NEGATIVE control: mode permutation is a true
        # invariance and must NOT be counted as a detection.
        p0 = lins[0]
        modeswap = list(lins)
        modeswap[0] = Lineage(A_D=p0.A_D, A_L=p0.C_L, C_L=p0.A_L, M=p0.M)
        m = _relerr(np.asarray(dslice_weff(fnp, modeswap, n, (2, 1), np.float64)), ref)
        results["M3b_mode_swap_is_an_invariance"] = {
            "relerr": m, "invariant_as_expected": m <= 1e-12,
            "note": "symmetric tensor: mode order cannot change the slice"}

        # M7 -- transpose the shared mode matrix M of one lineage
        p0 = lins[0]
        tr = list(lins)
        tr[0] = Lineage(A_D=p0.A_D, A_L=p0.A_L, C_L=p0.C_L, M=p0.M.T)
        m = _relerr(np.asarray(dslice_weff(fnp, tr, n, (2, 1), np.float64)), ref)
        results["M7_transpose_shared_M"] = {"relerr": m, "detected": m > 1e-8}

        # M4 -- forget the factor 3 on the two-leg middle mode
        m = _relerr(np.asarray(_dslice_weff_no3(fnp, lins, n, np.float64)), ref)
        results["M4_drop_scalar_3"] = {"relerr": m, "detected": m > 1e-8}

        # M5 -- count the D block's coincident permutation once instead of twice
        m = _relerr(np.asarray(_dslice_weff_single_d(fnp, lins, n, np.float64)), ref)
        results["M5_single_D_perm"] = {"relerr": m, "detected": m > 1e-8}

        # M6 -- swap the two identity-mode blocks' roles (D <-> L structure)
        swap = [Lineage(A_D=L.A_L if L.A_L is not None else L.A_D,
                        A_L=L.A_D if L.A_D is not None else L.A_L,
                        C_L=L.C_L, M=L.M, m_is_identity=L.m_is_identity,
                        has_D=L.has_D, has_L=L.has_L) for L in lins]
        m = _relerr(np.asarray(dslice_weff(fnp, swap, n, (2, 1), np.float64)), ref)
        results["M6_swap_block_roles"] = {"relerr": m, "detected": m > 1e-8}

        # M8 -- break contract_W_weff's identity shortcut: pretend `M == I`
        # advanced to `W @ M` when it should be `W`. The operator must change.
        p0 = _twoleg_lineage(fnp, R(n) * 0 + 1.1, R((n, n)))
        Wm = R((n, n))
        good = _lineages_to_dense(fnp, [contract_W_weff(fnp, Wm, p0)], n, np.float64)
        bad_lin = Lineage(A_L=Wm @ p0.A_L, C_L=Wm @ p0.C_L,
                          M=Wm @ Wm, m_is_identity=False,
                          has_D=False, has_L=True)
        bad = _lineages_to_dense(fnp, [bad_lin], n, np.float64)
        m = max(_relerr(np.asarray(x), np.asarray(y)) for x, y in zip(bad, good))
        results["M8_break_identity_shortcut"] = {"relerr": m, "detected": m > 1e-8}

    results["all_detected"] = all(
        v["detected"] for k, v in results.items()
        if k.startswith("M") and isinstance(v, dict) and "detected" in v)
    return results


def _dslice_weff_no3(fnp, lineages, n, dtype):
    acc = None
    for lin in lineages:
        if lin.has_D:
            if lin.m_is_identity:
                continue
            AM = lin.A_D * lin.M
            x = 2.0 * (AM @ lin.M.T) + (lin.M * lin.M) @ lin.A_D.T
            acc = x if acc is None else acc + x
        if lin.has_L:
            if lin.m_is_identity:
                dA = fnp.diagonal(lin.A_L); dC = fnp.diagonal(lin.C_L)
                x = (dA[:, None] * lin.C_L.T) + (lin.A_L * lin.C_L) + (dC[:, None] * lin.A_L.T)
            else:
                AM = lin.A_L * lin.M
                x = (AM @ lin.C_L.T) + ((lin.A_L * lin.C_L) @ lin.M.T) + ((lin.M * lin.C_L) @ lin.A_L.T)
            acc = x if acc is None else acc + x
    return (1.0 / 3.0) * _zero_repeated_2d(fnp, acc)


def _dslice_weff_single_d(fnp, lineages, n, dtype):
    acc = None
    for lin in lineages:
        if lin.has_D and not lin.m_is_identity:
            AM = lin.A_D * lin.M
            x = 1.0 * (AM @ lin.M.T) + (lin.M * lin.M) @ lin.A_D.T
            acc = x if acc is None else acc + x
        if lin.has_L:
            if lin.m_is_identity:
                dA = fnp.diagonal(lin.A_L); dC = fnp.diagonal(lin.C_L)
                x = 3.0 * (dA[:, None] * lin.C_L.T) + 3.0 * (lin.A_L * lin.C_L) \
                    + 3.0 * (dC[:, None] * lin.A_L.T)
            else:
                AM = lin.A_L * lin.M
                x = 3.0 * (AM @ lin.C_L.T) + 3.0 * ((lin.A_L * lin.C_L) @ lin.M.T) \
                    + 3.0 * ((lin.M * lin.C_L) @ lin.A_L.T)
            acc = x if acc is None else acc + x
    return (1.0 / 3.0) * _zero_repeated_2d(fnp, acc)


# ==========================================================================
# 8. rank-trajectory assertion
# ==========================================================================

def verify_rank_trajectory(n=64, depth=6) -> dict:
    """`r_l = 2 n l`, asserted by running upstream, not assumed."""
    import torch
    import p2_kprop_bridge as bridge
    bridge._src()
    from mlp_kprop.kprop_harmonic import SIMPLE, mlp_kprop
    from mlp_kprop.factor_k3 import FactoredTensor

    rng = np.random.default_rng(777)
    Ws = [np.asarray(rng.standard_normal((n, n)) * math.sqrt(2.0 / n), dtype=np.float64)
          for _ in range(depth)]
    mlp = bridge.build_mlp(Ws, device="cpu", dtype=torch.float64)
    K_in = {1: torch.zeros(n, dtype=torch.float64),
            2: torch.eye(n, dtype=torch.float64)}
    out = mlp_kprop(mlp, K_in, k_max=3, output_all=True, kind=SIMPLE,
                    factor=True, use_avg_metric=True,
                    up_to_layer=f"act{depth - 1}")
    ranks = {}
    for key, tower in out.items():
        if not key.startswith("act"):
            continue
        t3 = tower.get(3)
        if isinstance(t3, FactoredTensor):
            ranks[key] = int(t3._factors[0].shape[1])
    expected = {f"act{l}": 2 * n * (l + 1) for l in range(depth)}
    ok = all(ranks.get(k) == v for k, v in expected.items() if k in ranks)
    return {"n": n, "observed": ranks, "expected": expected, "pass": bool(ok and ranks)}



# ==========================================================================
# 8b. bin reconciliation -- turning the flop_name-scope claim into a TEST
# ==========================================================================

def bin_reconciliation(artifact: str, n: int = 16, depths=(1, 2, 3, 4, 5)) -> dict:
    """Prove, by measurement, WHERE upstream bills the fresh two-leg slice.

    Two assertions, both of which a wrong binning story would fail:

    (i) **`contract_wick_` lives inside upstream's `nonlin_sum 111 factored`
        bin.** Its cost is `3n * r = 3n * 2nl`, which grows with `l`; the fresh
        two-leg slice is `3 * 2n^3`, which does not. So

            delta_l( nonlin_sum ) - 3n * 2n * l

        must be **constant in `l`**. If it is not, the bin contains something
        else and every "like for like" ratio built on it is void.

    (ii) At width 1024 the exhaustive bin mapping must close the observed
        fnp-versus-upstream total gap to **zero**, with a named residual equal
        to the upstream lines this module genuinely never executes.

    This exists because two successive prose accounts of the same binning were
    wrong -- the first mis-attributed `~1.0e11` FLOPs, the second mis-stated the
    remainder by `4.31x` and closed only to `16,384`. A claim this load-bearing
    should not be an argument.
    """
    import torch
    import p2_kprop_bridge as bridge
    bridge._src()
    from mlp_kprop.kprop_harmonic import SIMPLE, mlp_kprop
    from mlp_kprop.flop_utils import NamedFlopCounter

    rows = []
    prev = None
    residuals = []
    for depth in depths:
        rng = np.random.default_rng(31337)
        Ws = [np.asarray(rng.standard_normal((n, n)) * math.sqrt(2.0 / n),
                         dtype=np.float64) for _ in range(depth)]
        mlp = bridge.build_mlp(Ws, device="cpu", dtype=torch.float64)
        K_in = {1: torch.zeros(n, dtype=torch.float64),
                2: torch.eye(n, dtype=torch.float64)}
        with torch.no_grad(), NamedFlopCounter() as ctr:
            mlp_kprop(mlp, K_in, k_max=3, output_all=True, kind=SIMPLE,
                      use_avg_metric=True, factor=True,
                      up_to_layer=f"act{depth - 1}", output_d_max=1)
        fl = ctr.flop_dict()
        cur = int(fl.get("nonlin_sum 111 factored", 0))
        row = {"depth": depth, "nonlin_sum_111_factored": cur}
        if prev is not None:
            l = depth - 1
            delta = cur - prev
            wick = 3 * n * (2 * n * l)
            row.update({"increment": delta, "wick_term_3n_2nl": wick,
                        "increment_minus_wick": delta - wick})
            residuals.append(delta - wick)
        rows.append(row)
        prev = cur

    const = len(set(residuals)) == 1 if residuals else False

    # ---- (ii) the width-1024 closure ------------------------------------
    with open(artifact, "r", encoding="utf-8") as fh:
        art = json.load(fh)
    seed = sorted(art["price"]["per_seed"])[0]
    b = art["price"]["per_seed"][seed]["f32"]
    D = b["lines_dense"]
    L0 = b["layers"][0]["lines"]
    fresh0 = L0["dense/get_dslice_21_fresh"] + L0["dense/get_dslice_3_fresh"]
    aged = D["dense/get_dslice_21_aged"] + D["dense/get_dslice_3_aged"]
    fresh = D["dense/get_dslice_21_fresh"] + D["dense/get_dslice_3_fresh"]

    up_path = os.path.join(_ROOT, "experiments", "2026-08-27_pb_rank",
                           "cost_ladder.json")
    with open(up_path, "r", encoding="utf-8") as fh:
        cell = [c for c in json.load(fh)["cells"] if c["cell"] == "none"][0]
    U, UT = cell["flops"], cell["total_flops"]

    groups = {
        "contract_W_basic": (D["contract_W_basic"], U["contract_W_basic"]),
        "contract_W": (D["dense/contract_W"], U["FactoredTensor3.contract_W"]),
        "get_dslice(aged+L0fresh)": (aged + fresh0,
                                     U["FactoredTensor3.get_dslice"]),
        "nonlin_sum bin": ((fresh - fresh0) + D["dense/contract_wick"]
                           + D["dense/twoleg_assembly"]
                           + D["dense/add_factors_concat"],
                           U["nonlin_sum 111 factored"]),
        "from_dstensor": (D["dense/from_dstensor"],
                          U["FactoredTensor3.from_dstensor"]),
    }
    matched_up = {"contract_W_basic", "FactoredTensor3.contract_W",
                  "FactoredTensor3.get_dslice", "nonlin_sum 111 factored",
                  "FactoredTensor3.from_dstensor"}
    unexecuted = int(sum(v for k, v in U.items() if k not in matched_up))
    total_delta = sum(f - u for f, u in groups.values()) - unexecuted
    gap = int(b["dense_total"]) - int(UT)

    return {
        "increment_rows": rows,
        "increment_minus_wick_constant": bool(const),
        "increment_minus_wick_values": residuals,
        "closure": {k: {"fnp": int(f), "upstream": int(u), "delta": int(f - u),
                        "ratio": f / u} for k, (f, u) in groups.items()},
        "unexecuted_flops": unexecuted,
        "unexecuted_budgets": unexecuted / BUDGET,
        "observed_gap": gap, "reconstructed_gap": int(total_delta),
        "closes_exactly": bool(int(total_delta) == gap),
        "weff_whole_path_flops": int(b["weff_total"]) + unexecuted,
        "weff_whole_path_budgets": (int(b["weff_total"]) + unexecuted) / BUDGET,
        "screen_budgets": SCREEN,
        "x_over_screen": ((int(b["weff_total"]) + unexecuted) / BUDGET) / SCREEN,
        "pass": bool(const and int(total_delta) == gap),
    }


# ==========================================================================
# 9. driver
# ==========================================================================

def run_price(n=WIDTH, depth=DEPTH, dtypes=("f32", "f64"), seeds=PROBE_SEEDS) -> dict:
    res: dict = {"width": n, "depth": depth, "budget": BUDGET, "screen_budgets": SCREEN,
                 "per_seed": {}}
    for si, seed in enumerate(seeds):
        per_dtype = {}
        for dt in dtypes:
            rng = np.random.default_rng(seed)
            layers = []
            t0 = time.perf_counter()
            with MaterializationCounter() as mc:
                for l in range(depth):
                    layers.append(price_layer(l, n, dt, rng))
                    print(f"    seed {seed} {dt} layer {l:2d} "
                          f"dense={layers[-1]['dense_total']:>16,d} "
                          f"weff={layers[-1]['weff_total']:>16,d}", flush=True)
            dense = sum(x["dense_total"] for x in layers)
            weff = sum(x["weff_total"] for x in layers)
            lines_dense: dict = {}
            lines_weff: dict = {}
            for x in layers:
                for k, v in x["lines"].items():
                    (lines_weff if k.startswith("weff/") else lines_dense)[k] = \
                        (lines_weff if k.startswith("weff/") else lines_dense).get(k, 0) + v
            per_dtype[dt] = {
                "layers": layers,
                "dense_total": dense, "weff_total": weff,
                "dense_budgets": dense / BUDGET, "weff_budgets": weff / BUDGET,
                "saving_fraction": 1.0 - weff / dense if dense else 0.0,
                "lines_dense": lines_dense, "lines_weff": lines_weff,
                "materializations": dict(mc.counts), "materializations_total": mc.total,
                "seconds": time.perf_counter() - t0,
            }
        res["per_seed"][str(seed)] = per_dtype
        if si == 0:
            continue
    # network-independence (prediction 6)
    keys = list(res["per_seed"])
    ident = {}
    for dt in dtypes:
        vals = {res["per_seed"][k][dt]["weff_total"] for k in keys}
        vals_d = {res["per_seed"][k][dt]["dense_total"] for k in keys}
        ident[dt] = {"weff_identical": len(vals) == 1, "dense_identical": len(vals_d) == 1,
                     "weff_values": sorted(vals), "dense_values": sorted(vals_d)}
    res["network_independence"] = ident
    res["memory"] = peak_memory()
    res["factor_storage"] = {
        dt: factor_storage_bytes(n, depth, 4 if dt == "f32" else 8)
        for dt in dtypes}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--parity-upstream", action="store_true")
    ap.add_argument("--symmetry", action="store_true")
    ap.add_argument("--bincheck", action="store_true")
    ap.add_argument("--mutations", action="store_true")
    ap.add_argument("--verify-rank", action="store_true")
    ap.add_argument("--price", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--depth", type=int, default=DEPTH)
    ap.add_argument("--dtypes", default="f32,f64")
    ap.add_argument("--seeds", default=",".join(str(s) for s in PROBE_SEEDS))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    if not any([a.parity, a.parity_upstream, a.mutations, a.verify_rank,
                a.price, a.symmetry, a.bincheck, a.all]):
        ap.error("choose at least one of --parity --parity-upstream "
                 "--mutations --verify-rank --price --all")

    _torch_only = (a.parity_upstream or a.verify_rank or a.bincheck) and not (
        a.all or a.parity or a.price or a.mutations or a.symmetry)
    if _torch_only:
        # The torch venv has no real flopscope on its path -- `import
        # flopscope` there resolves to `local_sim/flopscope`, the numpy
        # SHIM. These two passes bill nothing and must not touch it.
        stack = {"note": "upstream-scoring pass; flopscope deliberately not "
                          "imported (the torch venv would resolve the shim)"}
    else:
        stack = assert_real_stack()
    print("stack:", json.dumps(stack, indent=1))
    report: dict = {"stack": stack, "host": platform.platform(),
                    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "screen_budgets": SCREEN, "budget": BUDGET,
                    "probe_seeds": list(PROBE_SEEDS)}

    if a.bincheck:
        print("\n== bin reconciliation ==")
        art = os.path.join(_ROOT, "experiments", "2026-08-28_weff",
                           "weff_price_f32_1000301-1000302-1000303.json")
        report["bincheck"] = bin_reconciliation(art)
        bc = report["bincheck"]
        for r in bc["increment_rows"]:
            if "increment" in r:
                print(f"  depth {r['depth']}  d(nonlin_sum) {r['increment']:>10,d}"
                      f"  - 3n*2nl {r['wick_term_3n_2nl']:>10,d}"
                      f"  = {r['increment_minus_wick']:>10,d}")
        print(f"  constant in l                : {bc['increment_minus_wick_constant']}")
        for k, v in bc["closure"].items():
            print(f"  {k:26s} fnp {v['fnp']:>18,d}  up {v['upstream']:>18,d}"
                  f"  ratio {v['ratio']:.5f}")
        print(f"  genuinely unexecuted         : {bc['unexecuted_flops']:,d}"
              f" = {bc['unexecuted_budgets']:.6f} B")
        print(f"  gap closes exactly           : {bc['closes_exactly']}"
              f"  ({bc['reconstructed_gap']:,d} vs {bc['observed_gap']:,d})")
        print(f"  W-EFF whole path             : {bc['weff_whole_path_budgets']:.4f} B"
              f"  = {bc['x_over_screen']:.4f}x the {SCREEN} B screen")
        print(f"  PASS = {bc['pass']}")

    if a.verify_rank:
        print("\n== rank trajectory ==")
        try:
            report["upstream"] = verify_upstream()
            print("  upstream pin:", report["upstream"]["pass"])
        except Exception as e:                                   # noqa: BLE001
            report["upstream"] = {"pass": False, "error": repr(e)}
        try:
            report["rank_trajectory"] = verify_rank_trajectory()
        except Exception as e:                                   # noqa: BLE001
            report["rank_trajectory"] = {"pass": False, "error": repr(e)}
        print(json.dumps(report["rank_trajectory"], indent=1)[:1500])

    if a.all or a.parity:
        print("\n== parity (emit) ==")
        d = a.out or os.path.join(_ROOT, "experiments", "2026-08-28_weff")
        report["parity_emit"] = parity_emit(d)
        pe = report["parity_emit"]
        print(f"  ranks agree           : {pe['rank_matches']}")
        print(f"  worst weff-vs-dense   : {pe['worst_weff_vs_dense']:.3e}")
        print(f"  PASS = {pe['weff_vs_dense_pass']}")
        print(f"  emitted {pe['npz']}")
        print("  -> now run --parity-upstream in the torch venv")

    if a.parity_upstream:
        print("\n== parity (upstream) ==")
        d = a.out or os.path.join(_ROOT, "experiments", "2026-08-28_weff")
        report["parity_upstream"] = parity_upstream(d)
        pu = report["parity_upstream"]
        print(f"  upstream pin          : {pu['upstream']['pass']}")
        print(f"  worst dslice(3,)      : {pu['worst_dslice3']:.3e}")
        print(f"  worst dslice(2,1)     : {pu['worst_dslice21']:.3e}")
        print(f"  worst contract_W      : {pu['worst_contract_W']:.3e}")
        print(f"  PASS = {pu['pass']}")

    if a.all or a.parity:
        print("\n== commutation parity (contract_W_weff / contract_wick_weff) ==")
        report["commutation"] = commutation_parity()
        cm = report["commutation"]
        print(f"  worst contract_W commutes   : {cm['worst_contract_W']:.3e}")
        print(f"  worst contract_wick commutes: {cm['worst_contract_wick']:.3e}")
        print(f"  PASS = {cm['pass']}")

    if a.all or a.symmetry:
        print("\n== symmetry-discount probe (the contract's open sub-question) ==")
        report["symmetry"] = symmetry_discount_probe()
        print(json.dumps(report["symmetry"], indent=1, default=float))

    if a.all or a.mutations:
        print("\n== mutations ==")
        report["mutations"] = mutations()
        print(f"  baseline relerr={report['mutations']['baseline_relerr']:.3e} "
              f"pass={report['mutations']['baseline_pass']}")
        for k, v in report["mutations"].items():
            if k.startswith("M") and isinstance(v, dict) and "detected" in v:
                print(f"  {k:34s} relerr={v['relerr']:.3e} detected={v['detected']}")
            elif k.startswith("M") and isinstance(v, dict):
                print(f"  {k:34s} relerr={v['relerr']:.3e} "
                      f"invariant_as_expected={v['invariant_as_expected']}")
        print(f"  all_detected = {report['mutations']['all_detected']}")

    if a.all or a.price:
        print("\n== price ==")
        dtypes = tuple(a.dtypes.split(","))
        seeds = tuple(int(s) for s in a.seeds.split(","))
        report["price"] = run_price(a.width, a.depth, dtypes, seeds)
        first = report["price"]["per_seed"][str(seeds[0])]
        for dt, blob in first.items():
            print(f"\n  --- {dt} ---")
            print(f"  dense (measured dominant lines) : {blob['dense_total']:>18,d}"
                  f"  = {blob['dense_budgets']:.4f} B")
            print(f"  W-EFF (measured dominant lines) : {blob['weff_total']:>18,d}"
                  f"  = {blob['weff_budgets']:.4f} B")
            print(f"  saving                          : {blob['saving_fraction']*100:.2f}%")
            print(f"  screen {SCREEN} B                    : "
                  f"{'PASS' if blob['weff_budgets'] <= SCREEN else 'FAIL'}"
                  f"  ({blob['weff_budgets']/SCREEN:.2f}x the screen)")
            print(f"  scalar materializations         : {blob['materializations_total']}")
        mem = report["price"]["memory"]
        if mem.get("available"):
            print(f"\n  peak working set                : {mem['peak_working_set_mb']:.1f} MB")
            print(f"  peak commit                     : {mem['peak_commit_mb']:.1f} MB  (upper-bound headroom {mem['upper_bound_headroom_x']:.1f}x of 8192 MB, NOT enforced on Windows)")

    if a.out:
        os.makedirs(a.out, exist_ok=True)
        # Run-specific names. The 2026-08-21 trap -- an evidence file that a
        # documented invocation can silently shrink or overwrite -- is why no
        # two modes share an artifact name.
        if a.parity_upstream:
            name = "weff_upstream.json"
        elif a.bincheck:
            name = "weff_bincheck.json"
        elif a.verify_rank:
            name = "weff_rank.json"
        elif a.price and not (a.all or a.parity):
            name = "weff_price_%s_%s.json" % (a.dtypes.replace(",", "-"),
                                              a.seeds.replace(",", "-"))
        else:
            name = "weff.json"
        path = os.path.join(a.out, name)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1, default=float)
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
