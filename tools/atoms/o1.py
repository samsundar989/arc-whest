"""Lane O, HANDOFF 2026-10-12 Lane O step O1: the exact-SVD old-tier join, optionally at a smaller rank.

    o1        exact top-r SVD join basis at the base rank (256)  = price.py `join_svd` (M6.2 "jsvd")
    o1:<r>    the same with `_SHARED_RANK = r` (e.g. o1:224, o1:192, o1:128)

Not bit-identical (it changes the join basis). The anchors are price.py's own `join_svd` / `rank` anchors on the
v84 base (`fold+age6`), and assume no earlier atom touched `_shared_join_lean`'s basis lines or `_SHARED_RANK`.
Exact interpreter F/B and ops on the v84 base, and the twin arm each corresponds to (the built file, `--raw`):
`experiments/2026-10-13_o/O1/RESULTS.md`.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2] / "docs/research/2026-10-11_p6price"))
import price  # noqa: E402


def apply(src, z):
    return price.join_svd(src)


def make(arg):
    r = int(arg)

    def f(src, z):
        return price.rank(r)(price.join_svd(src))
    return f
