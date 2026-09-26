"""Step 1 (HANDOFF 2026-09-27): the HARMONIC OLD TIER, as arms built from v68_wino.

    python tools/build_harm.py --base estimator_k3_v68_wino.py --out-dir experiments/<dir>/arms \
        --arms "H5c,H5d,H4c,H3c"

Arm name H<age><metric>: `<age>` = the number of exact lineages kept (5 = v68's exact
set), `<metric>` = `c` (full covariance) or `d` (diag(var)).

WHAT CHANGES. v68 keeps a lineage exact while it is <= 5 layers old, then folds its
legs into a shared rank-256 basis `Q` that every later layer transports and
contracts (~10 units/layer at depth). Here the aged lineage is instead reduced to
its RADIAL (harmonic r = 1) part with respect to the metric C:

    T_s  ~  Sym(u_s (x) C),      u_s = 3/(n+2) (T_s : C^-1)

(the trace over two indices; `Sym(u (x) C) : C^-1 = u (n+2)/3`, so the remainder
T_s - Sym(u_s (x) C) is traceless in the metric C). The radial part is linear in u,
so every folded lineage collapses into ONE carried n-vector `U`, and the lineage
itself -- A_L, M, w2b, sb, eb, Z, L, Y, CA, CB -- is dropped. The shared basis is
switched off (`_SHARED_ON = False`).

WHERE. The fold sits where v68's join sat: the end of layer l, after the Wick carry
(`_wick_tower`), i.e. in post-activation coordinates with metric `cov_out(l)`.
That is EXACTLY the pre-activation fold at layer l+1 with metric
C_pre(l+1) = W^T cov_out W (the harmonic projection commutes with an invertible
transport), and it does not pay the folded lineage's `_advance`.

CARRY. `U <- g_u[l] W^T U` into layer l (with the per-layer gain `_HARM_GAIN`,
init 1), `U <- w11 . U` at the Wick carry. The metric U refers to is then read as
the CURRENT C_pre -- exact for the newest fold, and off by (cov_out vs
w11 w11^T . C_pre) for older ones; that mismatch is what g_u absorbs.

CONSUMERS (`D21[a, i] = T_aai`, the file's convention -- `_dslice21`'s
`{0,1}|{2}[a,i]`):
    D3_i     += U_i C_ii
    D21[a,i] += (2 U_a C_ai + U_i C_aa) / 3              (a != i)
Hubs / P2 / kappa4 get nothing from the old tier in this first build.

THE TRACE, factored (one lineage: legs A = A_L, P = M, C_L = A d(w2),
A_D = P d(sb) + 3 A d(eb) + Z L^T, hubs Sym(y_h (x) S_h), S_h = sum_j c_hj (A_j P_j^T +
P_j A_j^T)); with K = C^-1, YP = K P, YA = K A and the column dots
p = diag(P^T K P), q = diag(A^T K P), r = diag(A^T K A):

    D block  (1/3) [A_D p + 2 P (A_D . YP)^T 1]
    L block  A (2 w2 q) + P (w2 r)
    hub h    (1/3) [y_h 2 (c_h . q)^T 1 + 2 (A (c_h . P^T K y_h) + P (c_h . A^T K y_h))]

Verified against a dense (n,n,n) expansion by `tools/harm_check.py`.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_CONSTS = '''
#: STEP 1 (HANDOFF 09-27, tools/build_harm.py): the HARMONIC OLD TIER. A lineage is
#: kept exact while fewer than `_HARM_FOLD_AGE` are carried; the oldest is then
#: reduced to its radial part Sym(u (x) C) and its u joins ONE carried vector U.
_HARM_ON = True
_HARM_FOLD_AGE = {age}
#: "C" = metric cov_out (one inverse + one (n, 2n) product per fold);
#: "diag" = metric diag(var_out) (elementwise).
_HARM_METRIC = "{metric}"
#: per-layer gain on U at transport into layer l (init 1; fitted in the twin)
_HARM_GAIN = {gain}
#: per-layer gain on the old tier's D21 term (init 1; fitted in the twin)
_HARM_D21_GAIN = {d21gain}
'''

_FUNCS = '''

def _harm_trace(tw, K, Kdiag, n):
    """`T : K` for the OLDEST lineage of `tw` (see tools/build_harm.py).

    `K` is the metric inverse as a matrix, or None with `Kdiag` its diagonal."""
    A = tw.A_L[:, :n]
    P = tw.M[:, :n]
    w2 = tw.w2b[:n]
    sb = tw.sb[:n]
    eb = tw.eb[:n]
    r = tw.L.shape[2]
    Zs = tw.Z[:, :r]
    Ls = tw.L[0]
    if K is None:
        YP = P * Kdiag[:, None]
        YA = A * Kdiag[:, None]
        KZ = Zs * Kdiag[:, None]
    else:
        Y2 = _mm(K, fnp.concatenate([P, A, Zs], axis=1))
        YP = Y2[:, :n]
        YA = Y2[:, n:2 * n]
        KZ = Y2[:, 2 * n:]
    p = (P * YP).sum(axis=0)
    q = (A * YP).sum(axis=0)
    rr = (A * YA).sum(axis=0)
    # colsum(A_D . YP) with A_D = P d(sb) + 3 A d(eb) + Zs Ls^T
    zl = (Ls * (P.T @ KZ)).sum(axis=1)
    adp = sb * p + (eb * q) * 3.0 + zl
    t = (P @ (sb * p) + (A @ (eb * p)) * 3.0 + Zs @ (Ls.T @ p) + (P @ adp) * 2.0) * (1.0 / 3.0)
    t = t + A @ (w2 * q * 2.0) + P @ (w2 * rr)
    if tw.Y is not None:
        for h, cv in ((0, tw.CA), (1, tw.CB)):
            y = tw.Y[:, h]
            c = cv[:n]
            Ky = y * Kdiag if K is None else K @ y
            t = t + (y * ((c * q).sum() * 2.0)
                     + (A @ (c * (P.T @ Ky)) + P @ (c * (A.T @ Ky))) * 2.0) * (1.0 / 3.0)
    return t


def _harm_drop(tw, n):
    """`tw` without its OLDEST lineage (all per-column state)."""
    r = tw.L.shape[2]
    out = _Tower(None, tw.A_L[:, n:], tw.M[:, n:], tw.w2b[n:], tw.k - 1,
                 tw.Z[:, r:], tw.L[1:], tw.sb[n:], tw.eb[n:])
    if tw.Y is not None:
        out.Y, out.CA, out.CB = tw.Y[:, 2:], tw.CA[n:], tw.CB[n:]
    return out


def _harm_fold(tw, U, cov_out, n, dtype):
    """Fold every lineage beyond `_HARM_FOLD_AGE - 1` into U (end of a layer).

    u is chosen so that T - Sym(u (x) C) is traceless under the weight K:
        u = 3 (tau I + 2 C K)^-1 (T : K),   tau = tr(C K),
    applied to first order in 2CK/tau (<= ~3% here). K = C^-1 ("C") is the harmonic
    projection proper, u = 3/(n+2) T:C^-1, taken exactly. K = C ("euc") is the
    Euclidean least-squares fit of Sym(u (x) C) to T. "diag" = diag(1/var),
    "ridge" = (C + mean(var) I)^-1.
    """
    if tw is None or tw.k < _HARM_FOLD_AGE:
        return tw, U
    if _HARM_METRIC == "zero":          # control: drop the aged lineage, carry nothing
        while tw.k >= _HARM_FOLD_AGE:
            tw = _harm_drop(tw, n)
        return tw, U
    K = Kdiag = None
    if _HARM_METRIC == "C":
        K = fnp.linalg.inv(cov_out)
    elif _HARM_METRIC == "ridge":
        _eps = fnp.diagonal(cov_out).sum() * (1.0 / n)
        K = fnp.linalg.inv(cov_out + fnp.eye(n, dtype=dtype) * _eps)
    elif _HARM_METRIC == "euc":
        K = cov_out
    else:
        Kdiag = 1.0 / fnp.maximum(fnp.diagonal(cov_out), dtype(1e-12))
    if _HARM_METRIC != "C":
        tau = ((cov_out * K).sum() if Kdiag is None
               else (fnp.diagonal(cov_out) * Kdiag).sum())
    while tw.k >= _HARM_FOLD_AGE:
        b = _harm_trace(tw, K, Kdiag, n)
        if _HARM_METRIC == "C":
            u = b * (3.0 / (n + 2.0))
        else:
            Kb = K @ b if K is not None else b * Kdiag
            u = (b - (cov_out @ Kb) * (2.0 / tau)) * (3.0 / tau)
        U = u if U is None else U + u
        tw = _harm_drop(tw, n)
    return tw, U


def _harm_gain(tab, l):
    return tab[l] if 0 <= l < len(tab) else 1.0
'''

# (anchor, replacement) pairs, each anchor must occur exactly once.
_PATCHES = [
    ("_SHARED_ON = True\n", "_SHARED_ON = False\n"),
    ('                 "_qU", "_qold", "F", "R3", "R4", "w3b", "_p3")\n',
     '                 "_qU", "_qold", "F", "R3", "R4", "w3b", "_p3",\n'
     '                 # STEP 1 (HANDOFF 09-27): the harmonic old tier\'s carried vector\n'
     '                 "U")\n'),
    ("        self._p3 = None\n\n\n",
     "        self._p3 = None\n        self.U = None\n\n\n"),
    ("    adv = None if tw is None else _advance(WT, tw)\n    tw = None\n",
     "    _U = None if tw is None else tw.U\n"
     "    adv = None if tw is None else _advance(WT, tw)\n    tw = None\n"
     "    if _U is not None:\n"
     "        _U = (WT @ _U) * _harm_gain(_HARM_GAIN, layer_index)\n"),
    ("    D3 = _dslice3(adv) if adv is not None else fnp.zeros(n, dtype=dtype)\n",
     "    D3 = _dslice3(adv) if adv is not None else fnp.zeros(n, dtype=dtype)\n"
     "    if _U is not None:\n"
     "        D3 = D3 + _U * var\n"),
    ("        D21 = (_dslice21(adv, off_mask) if adv is not None\n"
     "               else fnp.zeros((n, n), dtype=dtype))\n",
     "        D21 = (_dslice21(adv, off_mask) if adv is not None\n"
     "               else fnp.zeros((n, n), dtype=dtype))\n"
     "        if _U is not None:\n"
     "            # radial old tier: D21[a,i] = T_aai = (2 U_a C_ai + U_i C_aa) / 3\n"
     "            _hg = _harm_gain(_HARM_D21_GAIN, layer_index) / 3.0\n"
     "            D21 = D21 + ((_U[:, None] * C11) * (2.0 * _hg)\n"
     "                         + (var[:, None] * _U[None, :]) * _hg) * off_mask\n"),
    ("        if _SHARED_ON and gens is None and not last_layer:\n"
     "            adv = _shared_join_lean(adv)\n",
     "        if _SHARED_ON and gens is None and not last_layer:\n"
     "            adv = _shared_join_lean(adv)\n"
     "        if _U is not None:\n"
     "            _U = _U * w[(1, 1)]\n"
     "        if _HARM_ON and gens is None and not last_layer:\n"
     "            adv, _U = _harm_fold(adv, _U, cov_out, n, dtype)\n"),
    ("    out_tw = _truncate(out_tw, gens, n)\n",
     "    out_tw = _truncate(out_tw, gens, n)\n"
     "    if out_tw is not None:\n"
     "        out_tw.U = _U\n"),
]


def build(src: str, age: int, metric: str, gain=None, d21gain=None) -> str:
    for a, b in _PATCHES:
        if src.count(a) != 1:
            raise SystemExit(f"anchor occurs {src.count(a)}x:\n{a}")
        src = src.replace(a, b, 1)
    gain = gain or [1.0] * 16
    d21gain = d21gain or [1.0] * 16
    anchor = "\ndef _shared_rank(n):\n"
    if src.count(anchor) != 1:
        raise SystemExit("no _shared_rank anchor")
    src = src.replace(anchor, _CONSTS.format(age=age, metric=metric, gain=gain,
                                             d21gain=d21gain) + _FUNCS + anchor, 1)
    return src


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="estimator_k3_v68_wino.py")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--arms", default="H5c,H5d,H4c,H3c")
    a = ap.parse_args(argv)
    src = (ROOT / a.base).read_text(encoding="utf-8")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(a.base).stem
    for arm in [x for x in a.arms.split(",") if x]:
        m = re.fullmatch(r"H(\d+)([cdrez])", arm)
        if not m:
            raise SystemExit(f"bad arm {arm!r}")
        p = out / f"{stem}__harm_{arm}.py"
        metric = {"c": "C", "d": "diag", "r": "ridge", "e": "euc", "z": "zero"}[m.group(2)]
        p.write_text(build(src, int(m.group(1)), metric),
                     encoding="utf-8", newline="\n")
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
