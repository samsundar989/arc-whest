"""Atom `sb2` -- Lane S, HANDOFF 2026-10-12 Lane S step S1: zero-copy Winograd glue at today's depths.

Variant B of `docs/research/2026-10-12_strassen/swproto.py` (`sw_out(fused=True)`) behind `_mm`, with
v84's `_depth_for` unchanged. BIT-IDENTICAL to `_sb` (same products, same additions in the same order;
only where results land changes). The legacy `_sb` stays in the file as the mixed-dtype fallback (never
taken today: all 236 `_mm` calls on DEV net 0 are float32 x float32).
Anchors: v84's `_mm` function, verbatim; touches nothing else, assumes no other atom.
Price on v84 (real meter, DEV net): see `experiments/2026-10-13_s/S1_checks.md`.
Twin arm: none -- the twin keeps the legacy `_sb` (bit-identical), i.e. the twin build omits this atom.
"""
from tools.atoms import rep

_MM_OLD = '''def _mm(A, B):
    """The chain's ONE matmul spelling. `_STRASSEN_DEPTH = 0` is a plain `@`."""
    if _STRASSEN_DEPTH <= 0:
        return A @ B
    d = _depth_for(A.shape[0], A.shape[1], B.shape[1])
    if d <= 0:
        return A @ B
    return _sb(A[None], B[None], d)[0]
'''

_MM_SB2 = '''def _sbuf(S, r, c, like, dt):
    """An `(S, r, c)` buffer laid out per matrix like `like` (row- or column-major).

    `_sb`'s `concatenate` keeps its inputs' layout, so a transposed operand gives
    column-major stacks at every level and its leaf products take BLAS's
    transposed path, which rounds differently. Matching the layout here is what
    keeps `_swo` bit-identical to `_sb`. `.transpose` is a free view.
    """
    if like.strides[1] < like.strides[2]:
        return fnp.empty((S, c, r), dtype=dt).transpose(0, 2, 1)
    return fnp.empty((S, r, c), dtype=dt)


def _swo(A, B, C, depth):
    """`_sb` with zero-copy glue: `(S,m,k) @ (S,k,n)` written into `C` (any `(S,m,n)` view).

    Lane S S1 (2026-10-13). The same seven products and the same additions in the
    same order as `_sb`, so the result is bit-identical; only where they land
    changes. Every operand combination is written with `out=` straight into its
    slot of the stacked buffer, every post-addition straight into its output
    quadrant, and the deepest level is issued as seven batched products on
    quadrant views, so no `concatenate`/`block` copy is billed. An `out=` write
    bills numel(out) exactly like the expression it replaces (flopscope 0.12.1,
    `docs/research/2026-10-12_strassen/probe_out.log`); every `out` here has
    exactly the broadcast shape of its operands (flopscope #275). Per level: 21
    quarter areas (15 on the deepest), against 33 for `_sb`.
    """
    S = A.shape[0]
    m = A.shape[1]
    k = A.shape[2]
    n = B.shape[2]
    if depth <= 0 or (m % 2) or (k % 2) or (n % 2):
        fnp.matmul(A, B, out=C)
        return
    mh = m // 2
    kh = k // 2
    nh = n // 2
    a11 = A[:, :mh, :kh]
    a12 = A[:, :mh, kh:]
    a21 = A[:, mh:, :kh]
    a22 = A[:, mh:, kh:]
    b11 = B[:, :kh, :nh]
    b12 = B[:, :kh, nh:]
    b21 = B[:, kh:, :nh]
    b22 = B[:, kh:, nh:]
    c11 = C[:, :mh, :nh]
    c12 = C[:, :mh, nh:]
    c21 = C[:, mh:, :nh]
    c22 = C[:, mh:, nh:]
    dt = A.dtype
    if depth == 1 or (mh % 2) or (kh % 2) or (nh % 2):
        # Deepest level: the next one would be the leaf, so issue its seven
        # products directly on the quadrant views (the plain operands need no copy).
        L = _sbuf(S, mh, kh, A, dt)
        R = _sbuf(S, kh, nh, B, dt)
        T1 = fnp.empty((S, mh, nh), dtype=dt)
        T2 = fnp.empty((S, mh, nh), dtype=dt)
        fnp.matmul(a11, b11, out=T1)                       # p1
        fnp.matmul(a12, b21, out=c11)                      # p2
        fnp.add(c11, T1, out=c11)                          # c11 = p1 + p2
        fnp.add(a21, a22, out=L)                           # s1
        fnp.subtract(b12, b11, out=R)                      # t1
        fnp.matmul(L, R, out=c22)                          # p5
        fnp.subtract(L, a11, out=L)                        # s2
        fnp.subtract(b22, R, out=R)                        # t2
        fnp.matmul(L, R, out=c12)                          # p6
        fnp.add(c12, T1, out=c12)                          # u2 = p1 + p6
        fnp.subtract(a12, L, out=L)                        # s4
        fnp.subtract(R, b21, out=R)                        # t4
        fnp.matmul(L, b22, out=T1)                         # p3
        fnp.matmul(a22, R, out=c21)                        # p4
        fnp.subtract(a11, a21, out=L)                      # s3
        fnp.subtract(b22, b12, out=R)                      # t3
        fnp.matmul(L, R, out=T2)                           # p7
        fnp.add(c12, T2, out=T2)                           # u3 = u2 + p7
        fnp.subtract(T2, c21, out=c21)                     # c21 = u3 - p4
        fnp.add(c12, c22, out=c12)                         # u2 + p5
        fnp.add(c12, T1, out=c12)                          # c12 = (u2 + p5) + p3
        fnp.add(T2, c22, out=c22)                          # c22 = u3 + p5
        return
    # Stacked level. Slots: 0 p1 (a11,b11)  1 p2 (a12,b21)  2 p4 (a22,t4)
    # 3 p3 (s4,b22)  4 p5 (s1,t1)  5 p6 (s2,t2)  6 p7 (s3,t3).
    SA = _sbuf(7 * S, mh, kh, A, dt)
    fnp.concatenate([a11, a12, a22], axis=0, out=SA[:3 * S])
    s1 = fnp.add(a21, a22, out=SA[4 * S:5 * S])
    s2 = fnp.subtract(s1, a11, out=SA[5 * S:6 * S])
    fnp.subtract(a11, a21, out=SA[6 * S:])
    fnp.subtract(a12, s2, out=SA[3 * S:4 * S])
    SB = _sbuf(7 * S, kh, nh, B, dt)
    fnp.concatenate([b11, b21], axis=0, out=SB[:2 * S])
    fnp.copyto(SB[3 * S:4 * S], b22)
    t1 = fnp.subtract(b12, b11, out=SB[4 * S:5 * S])
    t2 = fnp.subtract(b22, t1, out=SB[5 * S:6 * S])
    fnp.subtract(b22, b12, out=SB[6 * S:])
    fnp.subtract(t2, b21, out=SB[2 * S:3 * S])
    del s1, s2, t1, t2
    P = fnp.empty((7 * S, mh, nh), dtype=dt)
    _swo(SA, SB, P, depth - 1)
    del SA, SB
    p1 = P[0:S]
    p2 = P[S:2 * S]
    p4 = P[2 * S:3 * S]
    p3 = P[3 * S:4 * S]
    p5 = P[4 * S:5 * S]
    p6 = P[5 * S:6 * S]
    p7 = P[6 * S:]
    fnp.add(p1, p2, out=c11)                               # c11
    fnp.add(p1, p6, out=c12)                               # u2
    fnp.add(c12, p7, out=c21)                              # u3
    fnp.add(c21, p5, out=c22)                              # c22 = u3 + p5
    fnp.subtract(c21, p4, out=c21)                         # c21 = u3 - p4
    fnp.add(c12, p5, out=c12)                              # u2 + p5
    fnp.add(c12, p3, out=c12)                              # c12 = (u2 + p5) + p3


def _mm(A, B):
    """The chain's ONE matmul spelling. `_STRASSEN_DEPTH = 0` is a plain `@`."""
    if _STRASSEN_DEPTH <= 0:
        return A @ B
    d = _depth_for(A.shape[0], A.shape[1], B.shape[1])
    if d <= 0:
        return A @ B
    if A.dtype != B.dtype:
        # Mixed dtypes never occur in the chain today (all float32); the legacy
        # spelling keeps the promotion semantics exact if one ever does.
        return _sb(A[None], B[None], d)[0]
    C = fnp.empty((1, A.shape[0], B.shape[1]), dtype=A.dtype)
    _swo(A[None], B[None], C, d)
    return C[0]
'''


def apply(src: str, z) -> str:
    return rep(src, _MM_OLD, _MM_SB2)
