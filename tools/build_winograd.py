"""Step 4(a): swap the Strassen block recursion for the Strassen-WINOGRAD variant.

    python tools/build_winograd.py --base estimator_k3_v67_s3.py --out estimator_k3_v68_wino.py

PLAN rev 13 §3 Step 4(a) ("15-add Winograd"). `_sb` today is classic Strassen: 10
pre-additions and 8 post-additions per level (18 block adds). Winograd's variant uses
the same seven products with 15 adds (8 pre, 7 post):

    S1 = A21 + A22   S2 = S1 - A11   S3 = A11 - A21   S4 = A12 - S2
    T1 = B12 - B11   T2 = B22 - T1   T3 = B22 - B12   T4 = T2 - B21
    P1 = A11 B11  P2 = A12 B21  P3 = S4 B22  P4 = A22 T4  P5 = S1 T1  P6 = S2 T2  P7 = S3 T3
    U2 = P1 + P6   U3 = U2 + P7   U4 = U2 + P5
    C11 = P1 + P2   C12 = U4 + P3   C21 = U3 - P4   C22 = U3 + P5

Exact in real arithmetic (verified below on random blocks); in float32 the rounding
differs, which the twin DEV read prices. The seven products are still issued as ONE
batched matmul per level, so the op count falls by 3 per level and FLOPs by the three
block additions. Only `_sb`'s body between the eight block slices and the final
concatenate changes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

OLD = '''    SA = fnp.concatenate([a11 + a22, a21 + a22, a11, a22,
                          a11 + a12, a21 - a11, a12 - a22], axis=0)
    SB = fnp.concatenate([b11 + b22, b11, b12 - b22, b21 - b11,
                          b22, b11 + b12, b21 + b22], axis=0)

    P = _sb(SA, SB, depth - 1)
    p1 = P[0:b]
    p2 = P[b:2 * b]
    p3 = P[2 * b:3 * b]
    p4 = P[3 * b:4 * b]
    p5 = P[4 * b:5 * b]
    p6 = P[5 * b:6 * b]
    p7 = P[6 * b:7 * b]

    c11 = p1 + p4 - p5 + p7
    c12 = p3 + p5
    c21 = p2 + p4
    c22 = p1 - p2 + p3 + p6
'''

NEW = '''    # Strassen-WINOGRAD (tools/build_winograd.py): 15 block adds per level, not 18.
    s1 = a21 + a22
    s2 = s1 - a11
    s3 = a11 - a21
    s4 = a12 - s2
    t1 = b12 - b11
    t2 = b22 - t1
    t3 = b22 - b12
    t4 = t2 - b21
    SA = fnp.concatenate([a11, a12, s4, a22, s1, s2, s3], axis=0)
    SB = fnp.concatenate([b11, b21, b22, t4, t1, t2, t3], axis=0)

    P = _sb(SA, SB, depth - 1)
    p1 = P[0:b]
    p2 = P[b:2 * b]
    p3 = P[2 * b:3 * b]
    p4 = P[3 * b:4 * b]
    p5 = P[4 * b:5 * b]
    p6 = P[5 * b:6 * b]
    p7 = P[6 * b:7 * b]

    u2 = p1 + p6
    u3 = u2 + p7
    c11 = p1 + p2
    c12 = (u2 + p5) + p3
    c21 = u3 - p4
    c22 = u3 + p5
'''


def selftest() -> None:
    """The identity, in float64, on random blocks."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((1, 8, 8))
    B = rng.standard_normal((1, 8, 8))
    h = 4
    a11, a12, a21, a22 = A[:, :h, :h], A[:, :h, h:], A[:, h:, :h], A[:, h:, h:]
    b11, b12, b21, b22 = B[:, :h, :h], B[:, :h, h:], B[:, h:, :h], B[:, h:, h:]
    s1 = a21 + a22; s2 = s1 - a11; s3 = a11 - a21; s4 = a12 - s2
    t1 = b12 - b11; t2 = b22 - t1; t3 = b22 - b12; t4 = t2 - b21
    p1, p2, p3, p4 = a11 @ b11, a12 @ b21, s4 @ b22, a22 @ t4
    p5, p6, p7 = s1 @ t1, s2 @ t2, s3 @ t3
    u2 = p1 + p6; u3 = u2 + p7
    C = np.concatenate([np.concatenate([p1 + p2, (u2 + p5) + p3], axis=2),
                        np.concatenate([u3 - p4, u3 + p5], axis=2)], axis=1)
    err = np.abs(C - A @ B).max()
    assert err < 1e-12, err
    print(f"selftest: Winograd identity exact to {err:.1e}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="estimator_k3_v67_s3.py")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    selftest()
    src = (ROOT / a.base).read_text(encoding="utf-8")
    if src.count(OLD) != 1:
        raise SystemExit(f"classic Strassen block occurs {src.count(OLD)}x, expected 1")
    Path(a.out).write_text(src.replace(OLD, NEW, 1), encoding="utf-8", newline="\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
