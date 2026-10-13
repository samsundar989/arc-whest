"""Shared body of Lane S's depth atoms (`depth16`, `depth8`): HANDOFF 2026-10-12 Lane S step S2.

Replaces v84's `_depth_for` (verbatim anchor) by an exact-price argmin: for every depth d whose split keeps all
three dimensions even and the leaf's smallest dimension >= `leaf`, price `_mm` through `_swo` exactly
(`_swo_cost`, Python ints, equals the real meter to the FLOP and the op) and take the d minimising
FLOPs + lam * op_calls, d <= cap. `lam` is what one op call is worth in FLOPs: it trades the residual / wall
budget against F (lam = 0 is the pure-F optimum; the handoff's break-even rule is its local form).
REQUIRES `sb2` earlier in --atoms (the price is `_swo`'s); refuses to build otherwise. With `sbtrim` before it,
the op counts priced are the trimmed ones (20 / 22 instead of 21 / 26).
"""
from tools.atoms import rep

_OLD = '''def _depth_for(m, k, n):
    """How many Strassen levels this shape can actually afford.

    Deterministic, shape-derived, and never larger than `_STRASSEN_DEPTH`.
    """
    lo = m if m < k else k
    lo = lo if lo < n else n
    d = 0
    while d < _STRASSEN_DEPTH and (lo >> (d + 1)) >= _STRASSEN_MIN_BLOCK:
        d += 1
    # P2.4.5: one more level for BIG products whose leaf would stay >= the big block.
    # Python ints only (no flopscope op); FLOPs per added op are best on these.
    if (d < _STRASSEN_DEPTH and m * k * n >= _STRASSEN_BIG_VOL
            and (lo >> (d + 1)) >= _STRASSEN_BIG_BLOCK):
        d += 1
    return d
'''

_NEW = '''#: Lane S S2: depth by exact price. Leaf floor, depth cap, and the FLOPs one op
#: call is worth when choosing a depth (trades the residual/wall budget against F).
_SW_LEAF = {leaf}
_SW_CAP = {cap}
_SW_OP_FLOPS = {lam}


def _swo_cost(m, k, n, d):
    """(FLOPs, op calls) of `_mm` on (m,k)@(k,n) at depth `d` through `_swo`: exact."""
    if d <= 0:
        return m * n * (2 * k - 1), 1
    f = 0
    o = 1
    b = 1
    while True:
        if d <= 0 or (m % 2) or (k % 2) or (n % 2):
            return f + b * m * n * (2 * k - 1), o + 1
        mh = m // 2
        kh = k // 2
        nh = n // 2
        if d == 1 or (mh % 2) or (kh % 2) or (nh % 2):
            return (f + 4 * b * mh * kh + 4 * b * kh * nh + 7 * b * mh * nh
                    + 7 * b * mh * nh * (2 * kh - 1)), o + {fu}
        f += 7 * b * (mh * kh + kh * nh + mh * nh)
        o += {st}
        m, k, n, d, b = mh, kh, nh, d - 1, 7 * b


def _depth_for(m, k, n):
    """How many Strassen levels this shape should get: argmin of FLOPs + `_SW_OP_FLOPS` x ops.

    Deterministic, shape-derived, Python ints only (no flopscope op). A depth is
    eligible while every dimension stays even and the leaf's smallest dimension
    stays >= `_SW_LEAF`, up to `_SW_CAP`; ties go to the shallower depth.
    """
    lo = m if m < k else k
    lo = lo if lo < n else n
    best = 0
    best_c = None
    d = 0
    while d <= _SW_CAP:
        if d > 0:
            s = 1 << d
            if (m % s) or (k % s) or (n % s) or lo // s < _SW_LEAF:
                break
        f, o = _swo_cost(m, k, n, d)
        c = f + _SW_OP_FLOPS * o
        if best_c is None or c < best_c:
            best = d
            best_c = c
        d += 1
    return best
'''


def patch(src: str, leaf: int, cap: int, lam: int) -> str:
    if "def _swo(" not in src:
        raise SystemExit("depth atoms price `_swo`: put sb2 before them in --atoms")
    trim = "_SW_TRIM = True" in src     # sbtrim: 20 ops per stacked level, 22 on the deepest
    return rep(src, _OLD, _NEW.format(leaf=int(leaf), cap=int(cap), lam=int(lam),
                                      st=20 if trim else 21, fu=22 if trim else 26))


def arm(leaf: int, arg: str):
    """'lam', 'lam/cap' or 'lam/cap/leaf' (the last overrides the atom's leaf floor), e.g. '1.2e6/5', '1.2e6/6/12'."""
    parts = arg.split("/")
    lam = int(float(parts[0]))
    cap = int(parts[1]) if len(parts) > 1 and parts[1] else 6
    leaf = int(parts[2]) if len(parts) > 2 and parts[2] else leaf
    return lambda src, z: patch(src, leaf, cap, lam)
