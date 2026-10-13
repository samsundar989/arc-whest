"""Atom `sbtrim` -- Lane S, HANDOFF 2026-10-12 Lane S S2 "ops trim": fewer op calls in `_swo`, same arithmetic.

After v85 (#335697) measured 11.75-12.64 us/op of server residual, op calls are the binding budget (gate: ops <= 26,100).
Two bit-identical trims to sb2's `_swo`:
  1. stacked levels: slots ordered [p4, p1, p2, p3, p5, p6, p7], so each side's three plain blocks are adjacent and
     written by ONE `concatenate(out=)` (the B side's `copyto` goes): 21 -> 20 ops per level;
  2. deepest level: no `empty` buffers -- the first write into L, R, T1, T2 allocates its own result (an elementwise
     result follows its inputs' memory order, exactly as legacy `_sb`'s temporaries did; matmul outputs are only
     ever read elementwise, so their layout cannot change a bit): 26 -> 22 ops.
FLOPs unchanged. Bit-identical to sb2 (and so to v84). Requires `sb2` earlier in --atoms; put it BEFORE any depth
atom (the depth atoms read `_SW_TRIM` to price the trimmed op counts).
Twin arm: none (twin keeps legacy `_sb`).
"""
from tools.atoms import rep

_FUSED_OLD = '''        L = _sbuf(S, mh, kh, A, dt)
        R = _sbuf(S, kh, nh, B, dt)
        T1 = fnp.empty((S, mh, nh), dtype=dt)
        T2 = fnp.empty((S, mh, nh), dtype=dt)
        fnp.matmul(a11, b11, out=T1)                       # p1
        fnp.matmul(a12, b21, out=c11)                      # p2
        fnp.add(c11, T1, out=c11)                          # c11 = p1 + p2
        fnp.add(a21, a22, out=L)                           # s1
        fnp.subtract(b12, b11, out=R)                      # t1
'''
_FUSED_NEW = '''        # sbtrim: no `empty` here; each buffer is allocated by its first write.
        T1 = fnp.matmul(a11, b11)                          # p1
        fnp.matmul(a12, b21, out=c11)                      # p2
        fnp.add(c11, T1, out=c11)                          # c11 = p1 + p2
        L = fnp.add(a21, a22)                              # s1
        R = fnp.subtract(b12, b11)                         # t1
'''
_P7_OLD = '''        fnp.matmul(L, R, out=T2)                           # p7
'''
_P7_NEW = '''        T2 = fnp.matmul(L, R)                              # p7
'''
_STACK_OLD = '''    # Stacked level. Slots: 0 p1 (a11,b11)  1 p2 (a12,b21)  2 p4 (a22,t4)
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
'''
_STACK_NEW = '''    # Stacked level (sbtrim order). Slots: 0 p4 (a22,t4)  1 p1 (a11,b11)
    # 2 p2 (a12,b21)  3 p3 (s4,b22)  4 p5 (s1,t1)  5 p6 (s2,t2)  6 p7 (s3,t3):
    # each side's three plain blocks are adjacent, so ONE concatenate writes them.
    SA = _sbuf(7 * S, mh, kh, A, dt)
    fnp.concatenate([a22, a11, a12], axis=0, out=SA[:3 * S])
    s1 = fnp.add(a21, a22, out=SA[4 * S:5 * S])
    s2 = fnp.subtract(s1, a11, out=SA[5 * S:6 * S])
    fnp.subtract(a11, a21, out=SA[6 * S:])
    fnp.subtract(a12, s2, out=SA[3 * S:4 * S])
    SB = _sbuf(7 * S, kh, nh, B, dt)
    fnp.concatenate([b11, b21, b22], axis=0, out=SB[S:4 * S])
    t1 = fnp.subtract(b12, b11, out=SB[4 * S:5 * S])
    t2 = fnp.subtract(b22, t1, out=SB[5 * S:6 * S])
    fnp.subtract(b22, b12, out=SB[6 * S:])
    fnp.subtract(t2, b21, out=SB[:S])
    del s1, s2, t1, t2
    P = fnp.empty((7 * S, mh, nh), dtype=dt)
    _swo(SA, SB, P, depth - 1)
    del SA, SB
    p4 = P[0:S]
    p1 = P[S:2 * S]
    p2 = P[2 * S:3 * S]
    p3 = P[3 * S:4 * S]
'''
_FLAG_OLD = '''def _sbuf(S, r, c, like, dt):'''
_FLAG_NEW = '''#: Lane S `sbtrim`: `_swo` issues 20 ops per stacked level and 22 on the deepest.
_SW_TRIM = True


def _sbuf(S, r, c, like, dt):'''


def apply(src: str, z) -> str:
    if "def _swo(" not in src:
        raise SystemExit("sbtrim patches sb2's `_swo`: put sb2 before it in --atoms")
    if "_SW_OP_FLOPS" in src:
        raise SystemExit("put sbtrim BEFORE the depth atoms (they price the trimmed op counts)")
    for a, b in ((_FUSED_OLD, _FUSED_NEW), (_P7_OLD, _P7_NEW), (_STACK_OLD, _STACK_NEW), (_FLAG_OLD, _FLAG_NEW)):
        src = rep(src, a, b)
    return src
