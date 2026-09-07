"""Does flopscope bill a BATCHED 3-D matmul as ONE operation, and at what rate?

Slice 2's whole cost plan rests on this. The ranked lever list says batched
Strassen-Winograd is worth `(7/8)^L` in counted FLOPs, and warns that the repo's
NAIVE recursion issues ~1.1M array operations per MLP -- which at the measured
`12.1 us/op` graded residual would be `13 s` against a `0.4 s` cliff, i.e. 33x
over. The only escape is a spelling where the seven sub-products of every
recursion level are issued as ONE batched call, so that operations grow LINEARLY
in the recursion depth instead of as `7^L`.

That escape exists only if:

  1. `fnp.matmul` on 3-D operands is ONE flopscope operation, not `batch` of them;
  2. it is billed at `batch * m * n * (2k-1)`, i.e. no batching surcharge;
  3. the glue -- stack / slice / concatenate / reshape -- is cheap in operations
     AND in FLOPs, because Strassen's 15 Winograd additions are pure glue.

None of that is documented anywhere we control, so it is measured here.

ROUTINE: a lab script. Touches no seed block, no shipped file, no vault.
Run:  uv run python local_sim/p2_batched_matmul_probe.py
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# APPEND, never insert -- `local_sim/` holds numpy shims named `flopscope` and
# `whestbench`, and this probe must reach the REAL packages.
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

import flopscope as flops  # noqa: E402
import flopscope.numpy as fnp  # noqa: E402

flops.configure(symmetry_warnings=False)

BIG = int(1e15)


def run(fn):
    """(counted FLOPs, operation count) for one callable."""
    with flops.BudgetContext(flop_budget=BIG) as ctx:
        fn()
        return int(ctx.flops_used), len(ctx.op_log)


_rng = None


def ones(*shape):
    """A NON-SYMMETRIC operand of the given shape.

    `fnp.zeros(shape) + 1.0` looks like the obvious spelling and is a TRAP:
    flopscope tags a square all-equal array as symmetric and then bills
    operations on it at the unique-entry rate. Measured here at (7,128,128):
    a batched add cost 57,792 FLOPs, which is `7 * 128 * 129 / 2` -- the
    SYMMETRIC count -- not `7 * 128 * 128 = 114,688`. Every FLOP figure taken
    with such an operand is wrong by ~2x, so the operands are random.
    """
    global _rng
    import numpy as np
    if _rng is None:
        _rng = np.random.default_rng(20260907)
    return fnp.asarray(_rng.standard_normal(shape).astype(np.float32))


def _hdr(title):
    print()
    print(title)
    print("-" * len(title))


# --------------------------------------------------------------------------
# 1. Is a batched matmul one operation, and is it billed without surcharge?
# --------------------------------------------------------------------------

def q1():
    _hdr("1. batched (b,m,k) @ (b,k,n) vs b separate (m,k) @ (k,n)")
    n = 128
    print(f"{'b':>5}{'separate FLOPs':>18}{'ops':>6}"
          f"{'batched FLOPs':>18}{'ops':>6}{'FLOP ratio':>12}{'op ratio':>10}")
    for b in (2, 7, 32, 128):
        A = [ones(n, n) for _ in range(b)]
        B = [ones(n, n) for _ in range(b)]
        fs, os_ = run(lambda: [a @ c for a, c in zip(A, B)])

        SA = ones(b, n, n)
        SB = ones(b, n, n)
        fb, ob = run(lambda: SA @ SB)
        print(f"{b:>5}{fs:>18,}{os_:>6}{fb:>18,}{ob:>6}"
              f"{fb / fs:>12.6f}{ob / os_:>10.4f}")

    # The billing law, stated and checked rather than assumed.
    b, m, k, n2 = 7, 64, 96, 48
    SA, SB = ones(b, m, k), ones(b, k, n2)
    f, o = run(lambda: SA @ SB)
    expect = b * m * n2 * (2 * k - 1)
    print(f"\n  law check: (b={b}, m={m}, k={k}, n={n2})")
    print(f"    billed            {f:,}   in {o} operation(s)")
    print(f"    b*m*n*(2k-1)      {expect:,}   match: {f == expect}")


# --------------------------------------------------------------------------
# 2. What does the Strassen glue cost?
# --------------------------------------------------------------------------

def q2():
    _hdr("2. the glue: slicing, stacking, adding, concatenating")
    n = 256
    h = n // 2
    A = ones(n, n)

    f, o = run(lambda: A[:h, :h])
    print(f"  a 2-D slice (a view)                    {f:>14,} FLOPs  {o} op(s)")

    def four_slices():
        return A[:h, :h], A[:h, h:], A[h:, :h], A[h:, h:]
    f, o = run(four_slices)
    print(f"  four quadrant slices                    {f:>14,} FLOPs  {o} op(s)")

    a11, a12, a21, a22 = four_slices()
    f, o = run(lambda: a11 + a22)
    print(f"  one (h,h) add                           {f:>14,} FLOPs  {o} op(s)"
          f"   [h*h = {h * h:,}]")

    f, o = run(lambda: fnp.stack([a11 + a22, a21 + a22, a11, a22,
                                  a11 + a12, a21 - a11, a12 - a22], axis=0))
    print(f"  the 7-way A-side stack (5 adds + stack) {f:>14,} FLOPs  {o} op(s)")

    S = fnp.stack([a11, a12, a21, a22, a11, a12, a21], axis=0)
    f, o = run(lambda: S[0] + S[1])
    print(f"  indexing one slab out of a stack        {f:>14,} FLOPs  {o} op(s)")

    f, o = run(lambda: fnp.concatenate(
        [fnp.concatenate([a11, a12], axis=1),
         fnp.concatenate([a21, a22], axis=1)], axis=0))
    print(f"  reassembling a 2x2 block grid           {f:>14,} FLOPs  {o} op(s)")

    # Can the 15 Winograd combinations be done as BATCHED adds on one stack?
    # If a (7,h,h) +/- (7,h,h) is one op, the glue is O(1) ops per level.
    P = ones(7, h, h)
    Q = ones(7, h, h)
    f, o = run(lambda: P + Q)
    print(f"  a batched (7,h,h) add                   {f:>14,} FLOPs  {o} op(s)"
          f"   [7*h*h = {7 * h * h:,}]")


# --------------------------------------------------------------------------
# 3. One batched Strassen level, end to end, on a batch of matmuls
# --------------------------------------------------------------------------

def strassen_batched(A, B, depth):
    """One (or more) Strassen-Winograd levels on a BATCH of matmuls.

    `A` is `(b, m, k)` and `B` is `(b, k, n)`. Every recursion level multiplies
    the batch axis by 7 and halves each of m, k, n, so the whole level is issued
    as ONE batched matmul no matter how wide the batch has grown. That is the
    entire point: operations grow LINEARLY in `depth`, not as `7^depth`.
    """
    b, m, k = A.shape
    n = B.shape[2]
    if depth == 0 or m % 2 or k % 2 or n % 2:
        return A @ B
    mh, kh, nh = m // 2, k // 2, n // 2

    a11, a12 = A[:, :mh, :kh], A[:, :mh, kh:]
    a21, a22 = A[:, mh:, :kh], A[:, mh:, kh:]
    b11, b12 = B[:, :kh, :nh], B[:, :kh, nh:]
    b21, b22 = B[:, kh:, :nh], B[:, kh:, nh:]

    SA = fnp.concatenate([a11 + a22, a21 + a22, a11, a22,
                          a11 + a12, a21 - a11, a12 - a22], axis=0)
    SB = fnp.concatenate([b11 + b22, b11, b12 - b22, b21 - b11,
                          b22, b11 + b12, b21 + b22], axis=0)

    M = strassen_batched(SA, SB, depth - 1)          # (7b, mh, nh)
    m1, m2, m3, m4, m5, m6, m7 = (M[i * b:(i + 1) * b] for i in range(7))

    c11 = m1 + m4 - m5 + m7
    c12 = m3 + m5
    c21 = m2 + m4
    c22 = m1 - m2 + m3 + m6
    return fnp.concatenate([fnp.concatenate([c11, c12], axis=2),
                            fnp.concatenate([c21, c22], axis=2)], axis=1)


def q3():
    _hdr("3. batched Strassen-Winograd: FLOPs and OPERATIONS vs depth")
    # The live tower's dominant shape class: (n, k n) @ (k n, n) with n = 1024.
    # Probed smaller so the sweep is cheap; the LAW is what transfers, and the
    # FLOP ratio is shape-free while the op count is depth-only.
    n, b = 512, 8
    A, B = ones(b, n, n), ones(b, n, n)

    f0, o0 = run(lambda: A @ B)
    print(f"  baseline: batch of {b} x ({n},{n})@({n},{n})")
    print(f"    {f0:,} FLOPs in {o0} operation(s)\n")
    print(f"{'L':>3}{'counted FLOPs':>20}{'ops':>7}{'FLOP x':>10}"
          f"{'(7/8)^L':>10}{'ops added':>11}{'value':>14}")
    for L in (0, 1, 2, 3, 4):
        f, o = run(lambda: strassen_batched(A, B, L))
        saved = f0 - f
        print(f"{L:>3}{f:>20,}{o:>7}{f / f0:>10.4f}"
              f"{(7 / 8) ** L:>10.4f}{o - o0:>11}"
              f"{saved / max(o - o0, 1):>14,.0f}")
    print("\n  'value' = counted FLOPs saved per EXTRA operation spent.")
    print("  Operations are the scarce resource: the gate is 24,000 ops/MLP")
    print("  (0.290 s at the measured 12.1 us/op, against a 0.4 s cliff).")


# --------------------------------------------------------------------------
# 4. Does correctness survive? Strassen is exact in real arithmetic only.
# --------------------------------------------------------------------------

def q4():
    _hdr("4. numerical parity of the batched spelling, float32")
    import numpy as np
    rng = np.random.default_rng(0)
    n, b = 128, 3
    a = rng.standard_normal((b, n, n)).astype(np.float32)
    c = rng.standard_normal((b, n, n)).astype(np.float32)

    with flops.BudgetContext(flop_budget=BIG):
        A = fnp.asarray(a)
        C = fnp.asarray(c)
        ref = np.asarray(A @ C)
        for L in (1, 2, 3):
            got = np.asarray(strassen_batched(A, C, L))
            num = float(np.abs(got - ref).max())
            den = float(np.abs(ref).max())
            print(f"  L={L}   max abs {num:.3e}   relative {num / den:.3e}")
    print("\n  Strassen is exact in REAL arithmetic; the error above is float32")
    print("  reassociation only. The handoff's bar is 1e-6 relative.")


if __name__ == "__main__":
    print(f"flopscope {flops.__version__}")
    q1()
    q2()
    q3()
    q4()
