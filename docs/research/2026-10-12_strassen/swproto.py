"""Strassen-Winograd prototypes with out=-view glue, for flopscope 0.12.1.

  sb_ref(A3, B3, depth)        verbatim copy of estimator_k3_v84_age6.py::_sb   (33 q / level)
  sw_out(A, B, depth, fused)   design A: stacked operand buffers written with out= (21 q / level)
                               fused=True -> design B: deepest level unstacked (15 q on that level)
  sw_ks(A, B, depth)           design C: alternative-basis (Karstadt-Schwartz) Winograd, Morton
                               7-slot layout, everything in place (12 q / level + O(L) transforms)
  cost_ref / cost_out / cost_ks   exact (flops, ops) models of the three (float32).
q = one quadrant pass at that level (b*mh*kh, b*kh*nh or b*mh*nh).
"""
import numpy as np
import flopscope as flops, flopscope.numpy as fnp

F32 = np.float32


# ----------------------------------------------------------------------------- reference
def sb_ref(A, B, depth):
    b = A.shape[0]; m = A.shape[1]; k = A.shape[2]; n = B.shape[2]
    if depth <= 0 or (m % 2) or (k % 2) or (n % 2):
        return A @ B
    mh = m // 2; kh = k // 2; nh = n // 2
    a11 = A[:, :mh, :kh]; a12 = A[:, :mh, kh:]; a21 = A[:, mh:, :kh]; a22 = A[:, mh:, kh:]
    b11 = B[:, :kh, :nh]; b12 = B[:, :kh, nh:]; b21 = B[:, kh:, :nh]; b22 = B[:, kh:, nh:]
    s1 = a21 + a22; s2 = s1 - a11; s3 = a11 - a21; s4 = a12 - s2
    t1 = b12 - b11; t2 = b22 - t1; t3 = b22 - b12; t4 = t2 - b21
    SA = fnp.concatenate([a11, a12, s4, a22, s1, s2, s3], axis=0)
    SB = fnp.concatenate([b11, b21, b22, t4, t1, t2, t3], axis=0)
    del s1, s2, s3, s4, t1, t2, t3, t4
    P = sb_ref(SA, SB, depth - 1)
    p1 = P[0:b]; p2 = P[b:2 * b]; p3 = P[2 * b:3 * b]; p4 = P[3 * b:4 * b]
    p5 = P[4 * b:5 * b]; p6 = P[5 * b:6 * b]; p7 = P[6 * b:7 * b]
    u2 = p1 + p6; u3 = u2 + p7
    c11 = p1 + p2; c12 = (u2 + p5) + p3; c21 = u3 - p4; c22 = u3 + p5
    return fnp.block([[c11, c12], [c21, c22]])


def mm_ref(A, B, depth):
    return sb_ref(A[None], B[None], depth)[0] if depth > 0 else A @ B


def cost_ref(m, k, n, depth, b=1):
    if depth <= 0 or (m % 2) or (k % 2) or (n % 2):
        return b * m * n * (2 * k - 1), 1
    mh, kh, nh = m // 2, k // 2, n // 2
    glue = 11 * b * mh * kh + 11 * b * kh * nh + 7 * b * mh * nh + b * m * n
    f, o = cost_ref(mh, kh, nh, depth - 1, 7 * b)
    return glue + f, 18 + o


# ----------------------------------------------------------------------------- design A / B
def _swo(A, B, C, depth, fused):
    """A (S,m,k) @ B (S,k,n) written into C (S,m,n) (any view)."""
    S, m, k = A.shape; n = B.shape[2]
    if depth <= 0 or (m % 2) or (k % 2) or (n % 2):
        fnp.matmul(A, B, out=C)
        return
    mh = m // 2; kh = k // 2; nh = n // 2
    a11 = A[:, :mh, :kh]; a12 = A[:, :mh, kh:]; a21 = A[:, mh:, :kh]; a22 = A[:, mh:, kh:]
    b11 = B[:, :kh, :nh]; b12 = B[:, :kh, nh:]; b21 = B[:, kh:, :nh]; b22 = B[:, kh:, nh:]
    c11 = C[:, :mh, :nh]; c12 = C[:, :mh, nh:]; c21 = C[:, mh:, :nh]; c22 = C[:, mh:, nh:]
    last = depth == 1 or (mh % 2) or (kh % 2) or (nh % 2)
    if fused and last:
        # deepest level unstacked: 7 batched matmuls, plain quadrants used as views (no copies)
        L = fnp.empty((S, mh, kh), dtype=A.dtype); R = fnp.empty((S, kh, nh), dtype=A.dtype)
        T1 = fnp.empty((S, mh, nh), dtype=A.dtype); T2 = fnp.empty((S, mh, nh), dtype=A.dtype)
        fnp.matmul(a11, b11, out=T1)                                   # p1
        fnp.matmul(a12, b21, out=c11); fnp.add(c11, T1, out=c11)       # c11 = p2 + p1
        fnp.add(a21, a22, out=L); fnp.subtract(b12, b11, out=R)        # s1, t1
        fnp.matmul(L, R, out=c22)                                      # p5 -> c22
        fnp.subtract(L, a11, out=L); fnp.subtract(b22, R, out=R)       # s2, t2
        fnp.matmul(L, R, out=c12); fnp.add(c12, T1, out=c12)           # u2 = p6 + p1 -> c12
        fnp.subtract(a12, L, out=L); fnp.subtract(R, b21, out=R)       # s4, t4
        fnp.matmul(L, b22, out=T1)                                     # p3
        fnp.matmul(a22, R, out=c21)                                    # p4 -> c21
        fnp.subtract(a11, a21, out=L); fnp.subtract(b22, b12, out=R)   # s3, t3
        fnp.matmul(L, R, out=T2)                                       # p7
        fnp.add(c12, T2, out=T2)                                       # u3 = u2 + p7
        fnp.subtract(T2, c21, out=c21)                                 # c21 = u3 - p4
        fnp.add(c12, c22, out=c12); fnp.add(c12, T1, out=c12)          # c12 = u2 + p5 + p3
        fnp.add(T2, c22, out=c22)                                      # c22 = u3 + p5
        return
    # slots: 0 (a11,b11)=p1  1 (a12,b21)=p2  2 (a22,t4)=p4  3 (s4,b22)=p3  4 (s1,t1)=p5  5 (s2,t2)=p6  6 (s3,t3)=p7
    SA = fnp.empty((7 * S, mh, kh), dtype=A.dtype)
    fnp.concatenate([a11, a12, a22], axis=0, out=SA[:3 * S])
    s1 = fnp.add(a21, a22, out=SA[4 * S:5 * S])
    s2 = fnp.subtract(s1, a11, out=SA[5 * S:6 * S])
    fnp.subtract(a11, a21, out=SA[6 * S:])
    fnp.subtract(a12, s2, out=SA[3 * S:4 * S])
    SB = fnp.empty((7 * S, kh, nh), dtype=A.dtype)
    fnp.concatenate([b11, b21], axis=0, out=SB[:2 * S])
    fnp.copyto(SB[3 * S:4 * S], b22)
    t1 = fnp.subtract(b12, b11, out=SB[4 * S:5 * S])
    t2 = fnp.subtract(b22, t1, out=SB[5 * S:6 * S])
    fnp.subtract(b22, b12, out=SB[6 * S:])
    fnp.subtract(t2, b21, out=SB[2 * S:3 * S])
    P = fnp.empty((7 * S, mh, nh), dtype=A.dtype)
    _swo(SA, SB, P, depth - 1, fused)
    del SA, SB
    p1 = P[0:S]; p2 = P[S:2 * S]; p4 = P[2 * S:3 * S]; p3 = P[3 * S:4 * S]
    p5 = P[4 * S:5 * S]; p6 = P[5 * S:6 * S]; p7 = P[6 * S:]
    fnp.add(p1, p2, out=c11)
    fnp.add(p1, p6, out=c12)            # u2
    fnp.add(c12, p7, out=c21)           # u3
    fnp.add(c21, p5, out=c22)           # c22 = u3 + p5
    fnp.subtract(c21, p4, out=c21)      # c21 = u3 - p4
    fnp.add(c12, p5, out=c12)
    fnp.add(c12, p3, out=c12)           # c12 = u2 + p5 + p3


def sw_out(A, B, depth, fused=False):
    if depth <= 0:
        return A @ B
    C = fnp.empty((1, A.shape[0], B.shape[1]), dtype=A.dtype)
    _swo(A[None], B[None], C, depth, fused)
    return C[0]


def cost_out(m, k, n, depth, fused=False, b=1, top=True):
    f0, o0 = (0, 1) if top and depth > 0 else (0, 0)       # the output `empty`
    if depth <= 0 or (m % 2) or (k % 2) or (n % 2):
        return b * m * n * (2 * k - 1), 1 + o0
    mh, kh, nh = m // 2, k // 2, n // 2
    last = depth == 1 or (mh % 2) or (kh % 2) or (nh % 2)
    if fused and last:
        return (4 * b * mh * kh + 4 * b * kh * nh + 7 * b * mh * nh
                + 7 * b * mh * nh * (2 * kh - 1)), 4 + 8 + 7 + 7 + o0
    glue = 7 * b * mh * kh + 7 * b * kh * nh + 7 * b * mh * nh
    f, o = cost_out(mh, kh, nh, depth - 1, fused, 7 * b, False)
    return glue + f, 3 + 5 + 6 + 7 + o + o0


# ----------------------------------------------------------------------------- design C
def _sub(arr, ax, slot, L):
    """arr[<ax axes full>, slot, <remaining level axes restricted to quadrant slots 0..3>]."""
    return arr[(slice(None),) * ax + (slot,) + (slice(0, 4),) * (L - 1 - ax)]


def _morton_in(X, L, colmajor, dtype):
    """(p, q) -> 7-slot Morton buffer (7,)*L + (p/2^L, q/2^L), forward basis change applied.
    slot = 2r+c (A side) or 2c+r (B side) on every level."""
    p, q = X.shape
    pl, ql = p >> L, q >> L
    T = fnp.reshape(X, (2,) * L + (pl,) + (2,) * L + (ql,))             # billed numel
    perm = []
    for l in range(L):
        perm += [L + 1 + l, l] if colmajor else [l, L + 1 + l]
    T = T.transpose(perm + [L, 2 * L + 1])                               # free view
    M = fnp.reshape(T, (4,) * L + (pl, ql))                              # billed numel (physical copy)
    P = fnp.empty((7,) * L + (pl, ql), dtype=dtype)
    # forward basis change on level axes 1..L-1, in place inside M (3 passes of numel/4 each)
    for ax in range(1, L):
        v = [M[(slice(None),) * ax + (s,)] for s in range(4)]
        if not colmajor:      # [a11,a12,a21,a22] -> [s1,a12,s3,s2]
            fnp.subtract(v[0], v[2], out=v[2]); fnp.subtract(v[3], v[2], out=v[3]); fnp.add(v[3], v[0], out=v[0])
        else:                 # [b11,b21,b12,b22] -> [t1,b21,t3,t2]
            fnp.subtract(v[2], v[0], out=v[0]); fnp.subtract(v[3], v[2], out=v[2]); fnp.subtract(v[3], v[0], out=v[3])
    # level axis 0: out of place, landing in the quadrant sub-block of the 7-slot buffer
    Q = P[(slice(0, 4),) * L]
    if not colmajor:
        fnp.subtract(M[0], M[2], out=Q[2]); fnp.subtract(M[3], Q[2], out=Q[3]); fnp.add(Q[3], M[0], out=Q[0])
        fnp.copyto(Q[1], M[1])
    else:
        fnp.subtract(M[2], M[0], out=Q[0]); fnp.subtract(M[3], M[2], out=Q[2]); fnp.subtract(M[3], Q[0], out=Q[3])
        fnp.copyto(Q[1], M[1])
    return P


def sw_ks(A, B, depth):
    """Alternative-basis Strassen-Winograd. Needs m, k, n divisible by 2**depth."""
    L = depth
    if L <= 0:
        return A @ B
    m, k = A.shape; n = B.shape[1]
    PA = _morton_in(A, L, False, A.dtype)     # quadrant slots per level: [s1, a12, s3, s2]
    PB = _morton_in(B, L, True, A.dtype)      #                           [t1, b21, t3, t2]
    for ax in range(L):                        # 3 + 3 passes per level; slots 4,5,6 = p1, p3, p4 operands
        a = [_sub(PA, ax, s, L) for s in range(7)]
        fnp.subtract(a[0], a[3], out=a[4]); fnp.subtract(a[1], a[3], out=a[5]); fnp.add(a[3], a[2], out=a[6])
        bb = [_sub(PB, ax, s, L) for s in range(7)]
        fnp.subtract(bb[3], bb[2], out=bb[4]); fnp.add(bb[0], bb[3], out=bb[5]); fnp.subtract(bb[3], bb[1], out=bb[6])
    PC = fnp.matmul(PA, PB)                    # ONE batched leaf product, batch dims (7,)*L
    del PA, PB
    for ax in range(L - 1, -1, -1):            # 6 passes per level, in place; product slots [p5,p2,p7,p6,p1,p3,p4]
        p = [_sub(PC, ax, s, L) for s in range(7)]
        fnp.add(p[4], p[1], out=p[1])          # c11
        fnp.add(p[4], p[3], out=p[4])          # p1 + p6
        fnp.subtract(p[5], p[2], out=p[3])     # c12 - c22
        fnp.add(p[4], p[2], out=p[4])          # p1 + p6 + p7
        fnp.add(p[4], p[0], out=p[2])          # c22
        fnp.add(p[6], p[0], out=p[0])          # c22 - c21
    Q = PC[(slice(0, 4),) * L]                 # slots [c22-c21, c11, c22, c12-c22] on every level
    for ax in range(L):                        # inverse basis change, 2 passes per level, in place
        v = [Q[(slice(None),) * ax + (s,)] for s in range(4)]
        fnp.subtract(v[2], v[0], out=v[0]); fnp.add(v[3], v[2], out=v[3])
    # slots now [c21, c11, c22, c12]: slot = 2x+y with block (row, col) = (1-y, x)
    ml, nl = m >> L, n >> L
    T = fnp.reshape(Q, (2, 2) * L + (ml, nl))                            # billed numel (physical copy)
    T = T[tuple(x for _ in range(L) for x in (slice(None), slice(None, None, -1)))]   # flip every y axis: free
    perm = [2 * l + 1 for l in range(L)] + [2 * L] + [2 * l for l in range(L)] + [2 * L + 1]
    return fnp.reshape(T.transpose(perm), (m, n))                        # billed numel


def cost_ks(m, k, n, depth):
    L = depth
    if L <= 0:
        return m * n * (2 * k - 1), 1
    d = 1 << L
    assert m % d == 0 and k % d == 0 and n % d == 0
    ml, kl, nl = m // d, k // d, n // d
    f = 7 ** L * ml * nl * (2 * kl - 1)
    qa, qb, qc = m * k // 4, k * n // 4, m * n // 4
    for l in range(L):
        f += (7 ** l) * (3 * qa + 3 * qb + 6 * qc) // (4 ** l)
    f += (2 * m * k + 3 * (L - 1) * qa + 4 * qa) + (2 * k * n + 3 * (L - 1) * qb + 4 * qb)
    f += 2 * L * qc + 2 * m * n
    ops = 2 * (2 + 1 + 3 * (L - 1) + 4) + 6 * L + 1 + 6 * L + 2 * L + 2
    return f, ops


def depth_for_v84(m, k, n, DEPTH=5, MINB=20, BIGV=1073741824, BIGB=16):
    lo = min(m, k, n); d = 0
    while d < DEPTH and (lo >> (d + 1)) >= MINB:
        d += 1
    if d < DEPTH and m * k * n >= BIGV and (lo >> (d + 1)) >= BIGB:
        d += 1
    return d
