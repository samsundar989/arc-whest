"""Atom k3 (Lane K, HANDOFF 2026-10-12 Lane K, K3): two-column-set H.

Not bit-identical. Requires `k1` FIRST (any arm, `k1:off` included; its anchors are on k1's text):
`--atoms k0+k1:2.5_3.0+k2:3.0+k3`. Twin arm: `mask+dress+bank+fold+kgain --kg h=<ktab_h25>`.

The lineage born at layer l keeps two column sets:
  * M-side (the KAS H set, top K_H by alpha = every non-dead column): M (= W^T columns), sb, the
    residual rows L, and every per-column vector (w2b, eb, CA, CB, w3b), STORED AT THE M WIDTH;
  * A-side: the K_A birth columns with the smallest alpha^2 inside the M set (K_A = FIT-100 q50
    of |alpha| <= 2.5, nearest 32). A_L is stored ONLY on these columns. w2b, eb, CA, CB are zero on
    the M-only columns (the P6.1 'literal' masks: A_L / w2b / hubs on kink columns, e_b's separable
    row folded into the residual), so every elementwise A-side term vanishes there exactly as in
    the twin arm.
Column order inside a lineage: the A columns first, then the M-only ones. So
  * the transport multiplies the narrow A_L (the saving), and the old-tier join reads the joiner's
    A part as `A_L[:, :na]` and its weights as `dA[:na]`;
  * `_full_legs_lean` hands the elementwise readers A scattered back to the M width (one gather from
    the zero-padded block), so `_dslice3`, `_p2_recon`, hubs, star-D4 and fold are unchanged;
  * the D21 readout gathers LA back onto the A columns where it meets the narrow A_L in a product
    (recent part and old part separately).
`_Tower.aw` carries the per-lineage A widths (None = same as M); `_k3_ix` caches the index arrays.

Exact price (x3_tables S2b on v84): -3.13 % F (a two-width model of the same shapes).
"""
from tools.atoms import rep

#: Baked from experiments/2026-10-13_k/ktab_h25.json (FIT-100 q50 of |alpha| <= 2.5, nearest 32).
K3_A = [1024, 1024, 1024, 992, 928, 896, 832, 800, 768, 736, 704, 640, 608, 576, 576, 544]


def _in_func(src, fname, a, b, cnt=1):
    """rep() restricted to the body of top-level function `fname`."""
    i = src.index(f"\ndef {fname}(") + 1
    j = src.find("\ndef ", i + 1)
    j = len(src) if j < 0 else j
    body = src[i:j]
    if body.count(a) != cnt:
        raise SystemExit(f"atom k3: anchor occurs {body.count(a)}x in {fname}: {a[:80]!r}")
    return src[:i] + body.replace(a, b) + src[j:]


HELPERS = '''

def _lin_awidths(tw):
    """Lane K k3: every lineage's A-side width (= its M width unless two-column)."""
    return list(tw.aw) if tw.aw is not None else _lin_widths(tw)


def _k3_ix(tw):
    """Lane K k3: (ma_full, am_old, am_rec) index arrays for a two-column tower, else None.

    ma_full maps each M column (all lineages) to its A column, or to the zero pad column;
    am_old / am_rec pick the A columns out of the old / recent M-width blocks."""
    if tw.aw is None:
        return None
    if tw._aix is None:
        wa, wm = list(tw.aw), _lin_widths(tw)
        ta, oa, om = sum(wa), 0, 0
        cm = sum(wm[:tw.ko])
        ma, ao, ar = [], [], []
        for i, (a, m) in enumerate(zip(wa, wm)):
            ma += list(range(oa, oa + a)) + [ta] * (m - a)
            if i < tw.ko:
                ao += list(range(om, om + a))
            else:
                ar += list(range(om - cm, om - cm + a))
            oa += a
            om += m
        tw._aix = (fnp.array(ma), fnp.array(ao) if ao else None, fnp.array(ar))
    return tw._aix


def _k3_toM(X, ma):
    """Lane K k3: A-width columns -> M width (zeros on the M-only columns)."""
    return fnp.take(fnp.pad(X, ((0, 0), (0, 1))), ma, axis=1)
'''


def _patch(src, ka):
    src = rep(src,
              "_KAS_K = {\"R\": ",
              "#: Lane K k3: A-side (kink) birth columns per layer, inside the KAS H set.\n"
              f"_K3_A = {ka!r}\n"
              "_KAS_K = {\"R\": ")
    # ---- tower: A widths and the index cache ----
    src = rep(src, "                 \"hcols\")\n",
              "                 \"hcols\",\n"
              "                 # Lane K k3: per-lineage A-side widths (None = M widths); index cache\n"
              "                 \"aw\", \"_aix\")\n")
    src = rep(src, "        self.rows = self.kd = self.hcols = None\n",
              "        self.rows = self.kd = self.hcols = None\n        self.aw = self._aix = None\n")
    src = rep(src,
              "def _carry_shared(dst, src, Q):\n",
              "def _carry_shared(dst, src, Q):\n    dst.aw = src.aw\n")
    src = rep(src, "\n\ndef _kas_rows(tw, idx):\n", HELPERS + "\n\ndef _kas_rows(tw, idx):\n")
    # ---- _append: record the new lineage's A width ----
    src = rep(src,
              "        if Y is not None:\n"
              "            out.Y, out.CA, out.CB = Y, CA, CB\n"
              "        return out\n",
              "        if Y is not None:\n"
              "            out.Y, out.CA, out.CB = Y, CA, CB\n"
              "        if Lb is not None and A_L.shape[1] != Lb.shape[0]:\n"
              "            out.aw = [int(A_L.shape[1])]\n"
              "        return out\n")
    src = rep(src,
              "    return _carry_shared(out, tw, tw.Q)\n\n\n# ===",
              "    out = _carry_shared(out, tw, tw.Q)\n"
              "    if Lb is not None and (tw.aw is not None or A_L.shape[1] != Lb.shape[0]):\n"
              "        out.aw = _lin_awidths(tw) + [int(A_L.shape[1])]\n"
              "    return out\n\n\n# ===")
    # ---- full legs: A at the M width ----
    src = _in_func(src, "_full_legs_lean",
                   "    if not tw.ko:\n        return tw.A_L, tw.M\n",
                   "    _ix = _k3_ix(tw)\n"
                   "    if not tw.ko:\n"
                   "        if _ix is None:\n"
                   "            return tw.A_L, tw.M\n"
                   "        if tw._legs is None:\n"
                   "            tw._legs = (_k3_toM(tw.A_L, _ix[0]), tw.M)\n"
                   "        return tw._legs\n")
    src = _in_func(src, "_full_legs_lean",
                   "        tw._legs = (fnp.concatenate([F[:, :c], tw.A_L], axis=1),\n"
                   "                    fnp.concatenate([F[:, c:], tw.M], axis=1))\n",
                   "        _A = fnp.concatenate([F[:, :c], tw.A_L], axis=1)\n"
                   "        tw._legs = (_A if _ix is None else _k3_toM(_A, _ix[0]),\n"
                   "                    fnp.concatenate([F[:, c:], tw.M], axis=1))\n")
    # ---- D21 readout: narrow right A operand, LA gathered onto the A columns ----
    f = "_shared_contract_lean"
    src = _in_func(src, f,
                   "    _R = fnp.concatenate([tw.A_L if tw.ko else A_L, tw.M if tw.ko else M], axis=1)\n",
                   "    _R = fnp.concatenate([tw.A_L, tw.M], axis=1)\n")
    src = _in_func(src, f,
                   "        _Ar = tw.A_L if tw.ko else A_L\n        _Mr = tw.M if tw.ko else M\n",
                   "        _Ar = A_L[:, c:]\n        _Mr = M[:, c:]\n")
    src = _in_func(src, f,
                   "    rec = _mm(fnp.concatenate([LAr, LPr], axis=1), _R.T)\n",
                   "    _ix = _k3_ix(tw)\n"
                   "    if _ix is not None:\n"
                   "        LAr = fnp.take(LAr, _ix[2], axis=1)\n"
                   "    rec = _mm(fnp.concatenate([LAr, LPr], axis=1), _R.T)\n")
    src = _in_func(src, f,
                   "    Xold = _mm(fnp.concatenate([LA[:, :c], LP[:, :c]], axis=1), FF.T)\n",
                   "    Xold = _mm(fnp.concatenate([LA[:, :c] if _ix is None else fnp.take(LA, _ix[1], axis=1),\n"
                   "                                LP[:, :c]], axis=1), FF.T)\n")
    # ---- old-tier join: the joiner's A part is its first `na` columns ----
    f = "_shared_join_lean"
    src = _in_func(src, f,
                   "    n = _lin_widths(tw)[tw.ko]                   # KAS: the joiner's width\n",
                   "    n = _lin_widths(tw)[tw.ko]                   # KAS: the joiner's width\n"
                   "    na = _lin_awidths(tw)[tw.ko]                 # Lane K k3: its A-side width\n")
    src = _in_func(src, f,
                   "    J = fnp.concatenate([tw.A_L[:, :n], tw.M[:, :n]], axis=1)\n    dA, dM = tw._wts\n",
                   "    J = fnp.concatenate([tw.A_L[:, :na], tw.M[:, :n]], axis=1)\n    dA, dM = tw._wts\n"
                   "    if na != n:\n        dA = dA[:na]\n")
    src = _in_func(src, f, "        FA = fnp.concatenate([FF[:, :c], FJ[:, :n]], axis=1)\n"
                           "        FM = fnp.concatenate([FF[:, c:], FJ[:, n:]], axis=1)\n",
                   "        FA = fnp.concatenate([FF[:, :c], FJ[:, :na]], axis=1)\n"
                   "        FM = fnp.concatenate([FF[:, c:], FJ[:, na:]], axis=1)\n")
    src = _in_func(src, f, "        FA, FM, Sg = FJ[:, :n], FJ[:, n:], Sj\n",
                   "        FA, FM, Sg = FJ[:, :na], FJ[:, na:], Sj\n")
    src = _in_func(src, f, "    out = _Tower(tw.A_D, tw.A_L[:, n:], tw.M[:, n:], tw.w2b, tw.k,\n",
                   "    out = _Tower(tw.A_D, tw.A_L[:, na:], tw.M[:, n:], tw.w2b, tw.k,\n")
    src = _in_func(src, f, "    out.Q, out.FA, out.FM, out.ko, out.Sg = Qn, FA, FM, tw.ko + 1, Sg\n",
                   "    out.Q, out.FA, out.FM, out.ko, out.Sg = Qn, FA, FM, tw.ko + 1, Sg\n    out.aw = tw.aw\n")
    # ---- birth: the A set inside the H set, A columns first ----
    src = rep(src, "    _kr = _kd = _kz = _mh = _kh = _k1r = None\n",
              "    _kr = _kd = _kz = _mh = _kh = _k1r = _ma = None\n    _KA = n\n")
    src = rep(src,
              "            _kh = _perm[n - _KH:]\n",
              "            _kh = _perm[n - _KH:]\n"
              "            _KA = int(_K3_A[layer_index]) if layer_index < len(_K3_A) else _KH\n"
              "            if _KA < _KH:\n"
              "                # Lane K k3: the K_A smallest alpha^2 inside the H set carry A_L / w2b / hubs\n"
              "                _ra = fnp.argsort(fnp.argsort(fnp.where(_mh > 0.0, alpha * alpha, dtype(1e30))))\n"
              "                _ma = fnp.where(_ra < _KA, dtype(1.0), dtype(0.0))\n"
              "                _kh = fnp.argsort((_mh + _ma) * dtype(-1.0))[:_KH]   # A first, then H-only\n")
    src = rep(src,
              "        if _mh is not None:\n"
              "            fac1 = fac1 * _mh[None, :]\n"
              "            w2m = w2m * _mh\n",
              "        _mha = _mh if _ma is None else _ma   # Lane K k3: the A-side mask\n"
              "        if _mh is not None:\n"
              "            fac1 = fac1 * _mha[None, :]\n"
              "            w2m = w2m * _mha\n")
    src = rep(src, "            S_sep = S_sep * _mh[:, None]\n", "            S_sep = S_sep * _mha[:, None]\n")
    src = rep(src, "                cA = cA * _mh\n                cB = cB * _mh\n",
              "                cA = cA * _mha\n                cB = cB * _mha\n")
    src = rep(src, "                e_b = e_b * _mh\n", "                e_b = e_b * _mha\n")
    src = rep(src, "                fac1 = fnp.take(fac1, _kh, axis=1)\n",
              "                fac1 = fnp.take(fac1, _kh if _ma is None else _kh[:_KA], axis=1)\n")
    return src


def apply(src, z):
    return _patch(src, K3_A)
