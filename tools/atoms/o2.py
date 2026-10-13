"""Lane O, HANDOFF 2026-10-12 Lane O step O2: separate old-tier ranks for the two legs.

    o2:<rA>/<rM>    e.g. o2:256/128, o2:256/192, o2:192/256, o2:128/256

The shared old-tier basis serves both legs (old A_L = Q FA, old M = Q FM). Here each leg gets its own basis,
built by the same one-pass warm-started range finder on its OWN Gram (A: J_A d_A J_A^T + Q_A Sg_A Q_A^T; M: the same
with the M leg), with the join's weights unchanged. The two bases travel as ONE array `Q = [Q_A | Q_M]`, so every
row-wise operation on Q (transport W^T Q, the Wick column scale, KAS row gathers) is untouched; FA has rA rows and
FM has rM rows, which is how a reader finds the split. Sg becomes the pair (Sg_A, Sg_M).

Not bit-identical (o2:256/256 is NOT v84: two 256-dim bases instead of one shared one). Anchors are on the v84 base
(`fold+age6`, one-pass join, `_QUERY_BASIS = False`); they do not overlap `o1`'s, but o1's SVD branch would bypass
this join, so do not stack o1 with o2. Prices, ops and the twin arm (the built file, `--raw`):
`experiments/2026-10-13_o/O2/RESULTS.md`.
"""
from tools.atoms import rep

_JOIN = '''

def _join_leg(J, d, r, Qo, Fo, So):
    """O2: one leg's join -- the v84 one-pass range finder on this leg's Gram alone."""
    if Qo is not None:
        Om = Qo
    else:
        X = J * fnp.sqrt(d)
        Om = X[:, fnp.argsort((X * X).sum(axis=0))[-r:]]
    for _p in range(int(_SHARED_PASSES)):
        Y = _mm(J, (_mm(J.T, Om)) * d[:, None])
        if Qo is not None:
            Y = Y + Qo @ (So @ (Qo.T @ Om))
        Om, _ = fnp.linalg.qr(Y)
    FJ = _mm(Om.T, J)
    Sj = _mm(FJ * d, FJ.T)
    if Qo is None:
        return Om, FJ, Sj
    T = Om.T @ Qo
    return Om, fnp.concatenate([_mm(T, Fo), FJ], axis=1), (T @ So) @ T.T + Sj


def _join_split(tw, n, dJ):
    """O2: the lean join with one basis per leg, carried as `Q = [Q_A | Q_M]`."""
    ra, rm = min(int(_RANK_A), n), min(int(_RANK_M), n)
    if tw.ko:
        ka = tw.FA.shape[0]
        QA, QM, SA, SM = tw.Q[:, :ka], tw.Q[:, ka:], tw.Sg[0], tw.Sg[1]
    else:
        QA = QM = SA = SM = None
    QnA, FA, SgA = _join_leg(tw.A_L[:, :n], dJ[:n], ra, QA, tw.FA, SA)
    QnM, FM, SgM = _join_leg(tw.M[:, :n], dJ[n:], rm, QM, tw.FM, SM)
    out = _Tower(tw.A_D, tw.A_L[:, n:], tw.M[:, n:], tw.w2b, tw.k,
                 tw.Z, tw.L, tw.sb, tw.eb)
    out.Y, out.CA, out.CB = tw.Y, tw.CA, tw.CB
    if tw.F is not None:
        out.F, out.R3, out.R4, out.w3b = tw.F, tw.R3, tw.R4, tw.w3b
    out.Q, out.FA, out.FM, out.ko, out.Sg = (fnp.concatenate([QnA, QnM], axis=1), FA, FM,
                                             tw.ko + 1, (SgA, SgM))
    return out
'''


def make(arg):
    ra, rm = (int(x) for x in arg.split("/"))

    def f(src, z):
        assert "_QUERY_BASIS = False\n" in src and '_SHARED_WEIGHTS = "cons"\n' in src
        src = rep(src, "_SHARED_RANK = 256\n",
                  f"_SHARED_RANK = {max(ra, rm)}\n"
                  "#: O2: separate old-tier ranks for the A and M legs (one basis per leg).\n"
                  f"_RANK_A = {ra}\n_RANK_M = {rm}\n")
        src = rep(src,
                  "        c = tw.FA.shape[1]\n"
                  "        F = _mm(tw.Q, fnp.concatenate([tw.FA, tw.FM], axis=1))\n"
                  "        tw._legs = (fnp.concatenate([F[:, :c], tw.A_L], axis=1),\n"
                  "                    fnp.concatenate([F[:, c:], tw.M], axis=1))\n",
                  "        ka = tw.FA.shape[0]\n"
                  "        tw._legs = (fnp.concatenate([_mm(tw.Q[:, :ka], tw.FA), tw.A_L], axis=1),\n"
                  "                    fnp.concatenate([_mm(tw.Q[:, ka:], tw.FM), tw.M], axis=1))\n")
        src = rep(src, "    FF = fnp.concatenate([tw.FA, tw.FM], axis=1)\n", "")
        src = rep(src,
                  "    Xold = _mm(fnp.concatenate([LA[:, :c], LP[:, :c]], axis=1), FF.T)\n",
                  "    Xold = fnp.concatenate([_mm(LA[:, :c], tw.FA.T), _mm(LP[:, :c], tw.FM.T)], axis=1)\n")
        src = rep(src,
                  "    if tw.ko:\n"
                  "        Om = tw.Q\n"
                  "    else:\n"
                  "        X = J * fnp.sqrt(dJ)\n",
                  "    if True:  # O2\n"
                  "        return _join_split(tw, n, dJ)\n"
                  "    if tw.ko:\n"
                  "        Om = tw.Q\n"
                  "    else:\n"
                  "        X = J * fnp.sqrt(dJ)\n")
        return rep(src, "\n\ndef _proj_coef_d4_r2(n):\n", _JOIN + "\n\ndef _proj_coef_d4_r2(n):\n")
    return f


def apply(src, z):
    raise SystemExit("o2 needs an arm: o2:<rA>/<rM>")
