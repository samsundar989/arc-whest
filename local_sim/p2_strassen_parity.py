"""Is batched Strassen EXACT ENOUGH in float32 at the live shapes?

This is the gate on lever 2a, and it is the one thing the cost frontier in
`p2_strassen_plan.py` cannot answer. Strassen-Winograd is exact in REAL
arithmetic; in float32 it reassociates, and its error grows with recursion depth
and with the matrix dimension. The handoff's bar is **1e-6 relative**, and the
first probe -- at `n = 128`, on random N(0,1) -- already read `1.3e-06` at L=3.
`n = 128` is 8x smaller than the live shape in every dimension, so that reading
is optimistic and the bar may already be broken at L=2.

WHAT IS MEASURED
----------------
Relative error of the batched Strassen spelling against a plain `@`, at the
chain's ACTUAL matmul shapes, in float32, against a float64 reference so that the
baseline's own error is visible too. A comparison against the float32 `@` alone
would flatter Strassen, because the float32 baseline is not exact either.

Reported per (shape, L):

    strassen_vs_f32     what a parity test would see
    strassen_vs_f64     the honest error
    baseline_vs_f64     what the CURRENT chain already spends
    excess              strassen_vs_f64 / baseline_vs_f64

**`excess` is the decision column.** An arm that multiplies the existing float32
error by 1.2x is not a new error source; one that multiplies it by 20x is.

The operand distribution matters and is swept: the tower's blocks are not
N(0,1). Covariance-like operands (PSD, strongly correlated columns, decaying
spectrum) reassociate differently from white noise, and the real `A_D`/`A_L`
blocks are closer to the former.

# seed-ledger: no block named -- operands are synthetic, drawn from a fixed
local seed, and no MLP is minted. ROUTINE: measures a candidate that does not
exist yet and moves no gate.

Run:  uv run python local_sim/p2_strassen_parity.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]

import numpy as np  # noqa: E402

#: (m, k, n, calls) -- a spread of the chain's real classes: the widest, a
#: mid-depth one, the transposed family, and the narrow early-layer shape.
SHAPES = [
    (1024, 14336, 1024, 4),
    (1024, 1024, 14336, 3),
    (1024, 7168, 1024, 4),
    (1024, 1024, 1024, 2),
]


def _white(rng, *shape):
    return rng.standard_normal(shape).astype(np.float32)


def _covlike(rng, *shape):
    """A correlated, decaying-spectrum operand -- closer to the tower's blocks.

    Built as a low-rank-plus-noise product so columns are strongly correlated and
    the singular values decay, which is what a propagated covariance/cumulant
    block looks like. White noise is the easy case for reassociation error.
    """
    b, r, c = shape
    rank = max(4, min(r, c) // 8)
    u = rng.standard_normal((b, r, rank))
    v = rng.standard_normal((b, rank, c))
    scale = (1.0 / np.arange(1, rank + 1) ** 0.75).astype(np.float64)
    u = u * scale[None, None, :]
    x = np.matmul(u, v)
    x /= np.abs(x).max()
    return x.astype(np.float32)


DISTS = {"white": _white, "covlike": _covlike}


def strassen_np(A, B, depth, split_top=0):
    """The same spelling as `p2_strassen_memory.strassen`, in plain numpy.

    Plain numpy, not fnp: this measures ARITHMETIC, not billing, and running it
    unmetered keeps the sweep cheap. The operand order and the association tree
    are identical, which is the only thing that affects the answer.
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
    pairs = [
        (a11 + a22, b11 + b22), (a21 + a22, b11), (a11, b12 - b22),
        (a22, b21 - b11), (a11 + a12, b22), (a21 - a11, b11 + b12),
        (a12 - a22, b21 + b22),
    ]
    if split_top > 0:
        m1, m2, m3, m4, m5, m6, m7 = [
            strassen_np(sa, sb, depth - 1, split_top - 1) for (sa, sb) in pairs]
    else:
        SA = np.concatenate([p[0] for p in pairs], axis=0)
        SB = np.concatenate([p[1] for p in pairs], axis=0)
        M = strassen_np(SA, SB, depth - 1, 0)
        m1, m2, m3, m4, m5, m6, m7 = (M[i * b:(i + 1) * b] for i in range(7))
    c11 = m1 + m4 - m5 + m7
    c12 = m3 + m5
    c21 = m2 + m4
    c22 = m1 - m2 + m3 + m6
    return np.concatenate([np.concatenate([c11, c12], axis=2),
                           np.concatenate([c21, c22], axis=2)], axis=1)


def rel(x, ref):
    d = float(np.abs(x.astype(np.float64) - ref).max())
    s = float(np.abs(ref).max())
    return d / s if s else float("nan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--levels", default="1,2,3,4")
    ap.add_argument("--bar", type=float, default=1e-6)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    levels = [int(x) for x in a.levels.split(",")]

    print("=" * 100)
    print("BATCHED STRASSEN, FLOAT32 PARITY AT THE LIVE SHAPES")
    print(f"bar = {a.bar:.0e} relative      reference = float64 matmul")
    print("=" * 100)

    rows = []
    for dist_name, gen in DISTS.items():
        print(f"\n  operand distribution: {dist_name}")
        print(f"  {'m':>6}{'k':>7}{'n':>7}{'L':>3}"
              f"{'strassen vs f32':>18}{'strassen vs f64':>18}"
              f"{'baseline vs f64':>18}{'excess':>9}{'bar':>6}")
        for (m, k, n, b) in SHAPES:
            rng = np.random.default_rng(20260907)
            A = gen(rng, b, m, k)
            B = gen(rng, b, k, n)
            ref = A.astype(np.float64) @ B.astype(np.float64)
            base = A @ B
            base_err = rel(base, ref)
            for L in levels:
                got = strassen_np(A, B, L, 1)
                e32 = rel(got, base.astype(np.float64))
                e64 = rel(got, ref)
                exc = e64 / base_err if base_err else float("inf")
                ok = "ok" if e32 <= a.bar else "OVER"
                print(f"  {m:>6}{k:>7}{n:>7}{L:>3}"
                      f"{e32:>18.3e}{e64:>18.3e}{base_err:>18.3e}"
                      f"{exc:>9.2f}{ok:>6}")
                rows.append({"dist": dist_name, "m": m, "k": k, "n": n, "L": L,
                             "strassen_vs_f32": e32, "strassen_vs_f64": e64,
                             "baseline_vs_f64": base_err, "excess": exc,
                             "under_bar": bool(e32 <= a.bar)})
            del A, B, ref, base

    print("\n  HOW TO READ THIS")
    print("  * 'strassen vs f32' is what a naive parity test compares, and it is")
    print("    the column the 1e-6 bar was written against.")
    print("  * 'excess' is the honest one: how much MORE error than the chain")
    print("    already carries in float32. A chain whose baseline is 1e-6 and")
    print("    whose Strassen arm is 2e-6 has not gained a new error source; it")
    print("    has doubled an existing one, and only the 24-net paired raw MSE")
    print("    can say whether that matters.")
    print("  * this measures ONE matmul in isolation. The chain applies 16 layers")
    print("    of them and errors compound, so a per-matmul pass is NECESSARY")
    print("    and NOT SUFFICIENT. The binding test remains the paired raw MSE on")
    print("    24 nets against the GT noise floor.")

    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump({"bar": a.bar, "rows": rows}, fh, indent=2)
        print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
