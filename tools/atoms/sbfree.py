"""Atom `sbfree` -- Lane S: free each level's stacked buffers one level down (peak memory), same arithmetic.

In `_swo` every stacked level's inputs stayed referenced -- by the caller's `SA`/`SB` locals and by this level's
quadrant views -- until the whole recursion below returned, so at depth L all levels' operand stacks coexisted:
(7/4 + (7/4)^2 + ... + (7/4)^(L-1)) x the operand size. Depth 6 adds the largest term, which is why d12c6 peaked
~1.4 GB over depth16 (P7, 2026-10-13). A level only reads its inputs while it fills its own stacks, so the caller now
hands them over in a one-shot list the callee empties, and the callee drops its inputs and their views as soon as its
stacks are written: each stack dies one level down. Same ops, same FLOPs, bit-identical.
Requires `sb2` and `sbtrim` earlier in --atoms (anchors on the trimmed `_swo`). Do not combine with `sbks` (parked),
whose anchor is the old `_mm` call.
Twin arm: none.
"""
from tools.atoms import rep

_SIG_OLD = '''def _swo(A, B, C, depth):
'''
_SIG_NEW = '''def _swo(AB, C, depth):
'''
_HEAD_OLD = '''    exactly the broadcast shape of its operands (flopscope #275). Per level: 21
    quarter areas (15 on the deepest), against 33 for `_sb`.
    """
    S = A.shape[0]
'''
_HEAD_NEW = '''    exactly the broadcast shape of its operands (flopscope #275). Per level: 21
    quarter areas (15 on the deepest), against 33 for `_sb`.

    sbfree: the operands arrive as a one-shot list `AB = [A, B]` that is emptied
    here, so this frame holds the only references and can release them early.
    """
    A, B = AB
    AB.clear()
    S = A.shape[0]
'''
_REC_OLD = '''    del s1, s2, t1, t2
    P = fnp.empty((7 * S, mh, nh), dtype=dt)
    _swo(SA, SB, P, depth - 1)
    del SA, SB
'''
_REC_NEW = '''    del s1, s2, t1, t2
    # sbfree: this level's inputs are fully read; drop them and their views, and
    # hand SA/SB to the child, so each stack is freed one level down.
    del a11, a12, a21, a22, b11, b12, b21, b22, A, B
    P = fnp.empty((7 * S, mh, nh), dtype=dt)
    AB = [SA, SB]
    del SA, SB
    _swo(AB, P, depth - 1)
'''
_TOP_OLD = '''    _swo(A[None], B[None], C, d)
'''
_TOP_NEW = '''    _swo([A[None], B[None]], C, d)
'''


def apply(src: str, z) -> str:
    if "_SW_TRIM = True" not in src:
        raise SystemExit("sbfree anchors on the trimmed `_swo`: put sb2+sbtrim before it in --atoms")
    if "def _sks(" in src:
        raise SystemExit("sbfree is not combined with the parked sbks")
    for a, b in ((_SIG_OLD, _SIG_NEW), (_HEAD_OLD, _HEAD_NEW), (_REC_OLD, _REC_NEW), (_TOP_OLD, _TOP_NEW)):
        src = rep(src, a, b)
    return src
