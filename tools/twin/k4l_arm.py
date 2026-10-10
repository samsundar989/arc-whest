"""Twin arm `k4l` (HANDOFF 2026-10-08 Lane P3, P3.2): the kappa4-core loop on the v62_gb chain.

Source patches for `tools/twin/twin.py` ARMS["k4l"], applied AFTER `mask` and `dress` (use
`--arm mask+dress+k4l`). Inert while `_K4L` is None (exact identity). `_K4L` keys:

  core  1      carry the OFF-diagonal r = 1 core Off as memory (k4loop.core_off births,
               transport Core <- W^T Core W); dG = diag(W^T Core W); D31 = m Off/2 replaces the
               memoryless (m/2) lam C in Q31.
  k211  exact | trunc | none   the kappa3 -> core terms (K4.3 / K4.4): contraction of the
               incoming tower (exact: every lineage; trunc: the age-1 lineage only) plus the
               blocks born here.
  rad   hub | exact | fold | none   211 -> 111: `hub` keeps the chain's hub route i (the same
               term with the memoryless core lam C); `exact` replaces it with
               (m/4) Sym((w1 w1^T . Off_pre) (x) w2), transported exactly; `fold` keeps hub i
               with lam_eff = <Off_pre, C>/<C, C> per layer; `none` drops it.
  e21b  0 | 1  Lane M2's gain-mode E21b core increment.
  old   keep | drop   drop = aged-out lineages are discarded instead of joined into Q.
  (old tier rank: set the module's _SHARED_RANK directly.)

State: the chain's unused `c4` slot carries {'c4', 'off', 'rad'} between layers, so torch's
per-layer checkpoint recomputation sees it as an input (a module global would not survive it).
"""

HELPERS = r'''
_K4L = None
import k4loop as _k4lib


def _k4l_get(key, default=None):
    return default if _K4L is None else _K4L.get(key, default)


def _k4l_legs(tw):
    """(A, M, AD, n, k): full legs of every lineage, AD the P2 from_dstensor leg."""
    A, M = _full_legs_lean(tw)
    n, k = M.shape[0], tw.k
    if tw.A_D is None:
        r = tw.L.shape[2]
        Z3 = fnp.reshape(tw.Z, (n, k, r)).transpose(1, 0, 2)
        Zres = fnp.reshape((Z3 @ tw.L.transpose(0, 2, 1)).transpose(1, 0, 2), (n, k * n))
        AD = M * tw.sb + (A * tw.eb) * 3.0 + Zres
    else:
        AD = tw.A_D
    return A, M, AD, n, k


def _k4l_tower_cd(tw, v, lmask=None):
    """(C, D21): C = sum_k v_k kappa3_kab (all entries) and D21 = kappa3_aab of the tower,
    restricted to the lineages with lmask = 1 (None: all). Dense (n, n) each."""
    A, M, AD, n, k = _k4l_legs(tw)
    w2 = tw.w2b
    if lmask is not None:
        cm = fnp.reshape(lmask[:, None] * fnp.ones((k, n), dtype=A.dtype), (k * n,))
        w2 = w2 * cm
        AD = AD * cm
    Av = v @ A
    Mv = v @ M
    C = (A * (w2 * Mv)) @ A.T
    X = (M * (w2 * Av)) @ A.T
    C = C + X + X.T
    ADv = v @ AD
    X = (AD * Mv) @ M.T
    C = C + ((M * ADv) @ M.T + X + X.T) * (1.0 / 3.0)
    AM = A * M
    D = (AM * w2) @ A.T * 2.0 + ((A * A) * w2) @ M.T
    D = D + ((AD * M) @ M.T * 2.0 + (M * M) @ AD.T) * (1.0 / 3.0)
    if tw.Y is not None:
        Y3 = fnp.reshape(tw.Y, (n, k, 2))
        cA = fnp.reshape(tw.CA, (k, n))
        cB = fnp.reshape(tw.CB, (k, n))
        if lmask is not None:
            cA = cA * lmask[:, None]
            cB = cB * lmask[:, None]
        yv = v @ tw.Y
        gcol = fnp.reshape(yv[0::2][:, None] * cA + yv[1::2][:, None] * cB, (k * n,))
        G = (A * gcol) @ M.T
        U3 = fnp.reshape(A * Mv + M * Av, (n, k, n))
        R = fnp.reshape(fnp.stack([(U3 * cA[None]).sum(axis=2), (U3 * cB[None]).sum(axis=2)],
                                  axis=2), (n, 2 * k))
        YR = tw.Y @ R.T
        C = C + (G + G.T + YR + YR.T) * (1.0 / 3.0)
        Gy = fnp.reshape(Y3[:, :, 0][:, :, None] * cA[None] + Y3[:, :, 1][:, :, None] * cB[None],
                         (n, k * n))
        RY = fnp.reshape(fnp.stack([(fnp.reshape(AM, (n, k, n)) * cA[None]).sum(axis=2),
                                    (fnp.reshape(AM, (n, k, n)) * cB[None]).sum(axis=2)], axis=2),
                         (n, 2 * k))
        D = D + ((A * Gy) @ M.T + (M * Gy) @ A.T + RY @ tw.Y.T) * (2.0 / 3.0)
    return C, D


def _k4l_hub_cd(y, S, v):
    """(C, D21, D3) of Sym3(y (x) S) = (1/3)(y_p S_qr + y_q S_pr + y_r S_pq), S symmetric."""
    Sv = S @ v
    C = (S * (v @ y) + y[:, None] * Sv[None, :] + Sv[:, None] * y[None, :]) * (1.0 / 3.0)
    dS = fnp.diagonal(S)
    D = (y[:, None] * S * 2.0 + dS[:, None] * y[None, :]) * (1.0 / 3.0)
    return C, D, y * dS


def _k4l_dist(C, D, v, off_mask):
    """sum over k != a, b: the full contraction minus its two repeated-index entries."""
    return (C - v[:, None] * D - v[None, :] * D.T) * off_mask


def _k4l_drop(tw):
    """`old = drop`: discard the oldest lineage where the join would have moved it into Q."""
    n = tw.A_L.shape[0]
    if (tw.k - tw.ko) < _SHARED_AGE or tw._wts is None:
        return tw
    r = tw.L.shape[2]
    out = _Tower(None, tw.A_L[:, n:], tw.M[:, n:], tw.w2b[n:], tw.k - 1,
                 tw.Z[:, r:], tw.L[1:], tw.sb[n:], tw.eb[n:])
    if tw.Y is not None:
        out.Y, out.CA, out.CB = tw.Y[:, 2:], tw.CA[n:], tw.CB[n:]
    return out


'''

ANCHOR_LAYER = "def _layer(W, state, table, dtype, const, init_scale, last_layer, gens,\n"

PATCHES = [
    (ANCHOR_LAYER, HELPERS + ANCHOR_LAYER),
    # old tier: drop instead of join
    ('def _shared_join_lean(tw):\n    """LEAN join: the range finder\'s four leg products fused into two."""\n',
     'def _shared_join_lean(tw):\n    """LEAN join: the range finder\'s four leg products fused into two."""\n'
     "    if _K4L is not None and _K4L.get('old') == 'drop':\n"
     "        return _k4l_drop(tw)\n"),
    # unpack the carried state
    ("    mu, cov, tw, c4, gk4 = state\n",
     "    mu, cov, tw, c4, gk4 = state\n"
     "    _k4st = c4 if isinstance(c4, dict) else {}\n"
     "    _k4on = _K4L is not None and bool(_K4L.get('core'))\n"
     "    _k4rad = _k4st.get('rad') or []\n"),
    # radial transport (exact): y <- W^T y, S <- W^T S W
    ("    adv = None if tw is None else _advance(WT, tw)\n",
     "    adv = None if tw is None else _advance(WT, tw)\n"
     "    if _k4rad:\n"
     "        _k4rad = [(WT @ _y, _p4_sym_sandwich(WT, _S, W)) for _y, _S in _k4rad]\n"),
    # radial readouts: D3
    ("    D3 = _dslice3(adv) if adv is not None else fnp.zeros(n, dtype=dtype)\n",
     "    D3 = _dslice3(adv) if adv is not None else fnp.zeros(n, dtype=dtype)\n"
     "    for _y, _S in _k4rad:\n"
     "        D3 = D3 + _y * fnp.diagonal(_S)\n"),
    # core transport and dG
    ("    dG = fnp.where(fnp.isfinite(dG), dG, dtype(0.0))\n",
     "    _k4l_offpre = None\n"
     "    if _k4on:\n"
     "        if _k4st.get('off') is not None:\n"
     "            _k4X = _p4_sym_sandwich(WT, _k4st['off'], W)\n"
     "            dG = _WT2 @ gk4 + fnp.diagonal(_k4X)\n"
     "            _k4l_offpre = _k4X * off_mask\n"
     "        else:\n"
     "            dG = _WT2 @ gk4\n"
     "            _k4l_offpre = fnp.zeros((n, n), dtype=dtype)\n"
     "    dG = fnp.where(fnp.isfinite(dG), dG, dtype(0.0))\n"),
    # radial readouts: D21 (before the mask arm copies it)
    ("        D21 = (_dslice21(adv, off_mask) if adv is not None\n"
     "               else fnp.zeros((n, n), dtype=dtype))\n",
     "        D21 = (_dslice21(adv, off_mask) if adv is not None\n"
     "               else fnp.zeros((n, n), dtype=dtype))\n"
     "        for _y, _S in _k4rad:\n"
     "            D21 = D21 + _k4l_hub_cd(_y, _S, _y)[1] * off_mask\n"
     "        _k4l_D21 = D21\n"),
    # the closure reads the carried core
    ('"lam": _lam_prev, "metric": metric})',
     '"lam": _lam_prev, "metric": metric, "off": _k4l_offpre})'),
    ('        nb["Q31"] = Cn * inv2[:, None] * (0.5 * blocks["metric"] * blocks["lam"])\n',
     '        nb["Q31"] = Cn * inv2[:, None] * (0.5 * blocks["metric"] * blocks["lam"])\n'
     '        if blocks.get("off") is not None:\n'
     '            nb["Q31"] = (blocks["off"] * (inv2 * inv)[:, None]) * (inv[None, :]\n'
     '                         * (0.5 * blocks["metric"]))\n'),
    # hub route i: the memoryless 211 -> 111 term
    ("                cA = w1 * (_hm * _lam_h)\n",
     "                _k4r = _k4l_get('rad', 'hub') if _k4on else 'hub'\n"
     "                if _k4r == 'hub':\n"
     "                    cA = w1 * (_hm * _lam_h)\n"
     "                elif _k4r == 'fold':\n"
     "                    _lef = ((_k4l_offpre * C11).sum()\n"
     "                            / fnp.maximum((C11 * C11).sum(), dtype(1e-30)))\n"
     "                    cA = w1 * (_hm * _lef)\n"
     "                else:\n"
     "                    cA = w1 * 0.0\n"),
    # K4.4 + the -2 pK1 conversion of the incoming tower's w1 image: BEFORE the Wick scale
    ("    if adv is not None:\n        adv = _wick_tower(w1t[:, None], adv)\n",
     "    _k4l_old = None\n"
     "    if _k4on and not last_layer and _K4L.get('k211', 'exact') != 'none':\n"
     "        _k4e = w[(1, 2)] - 2.0 * pK[(1,)] * w1\n"
     "        _k4w = w1[:, None] * w1[None, :]\n"
     "        _acc = None\n"
     "        if adv is not None:\n"
     "            _lm = None\n"
     "            if _K4L.get('k211') == 'trunc':\n"
     "                _lm = fnp.zeros(adv.k, dtype=dtype)\n"
     "                _lm[-1] = 1.0\n"
     "            _c, _d = _k4l_tower_cd(adv, _k4e, _lm)\n"
     "            _acc = _k4l_dist(_c, _d, _k4e, off_mask)\n"
     "        _rl = _k4rad if _K4L.get('k211') != 'trunc' else _k4rad[-1:]\n"
     "        for _y, _S in _rl:\n"
     "            _c, _d, _ = _k4l_hub_cd(_y, _S, _k4e)\n"
     "            _t = _k4l_dist(_c, _d, _k4e, off_mask)\n"
     "            _acc = _t if _acc is None else _acc + _t\n"
     "        if _acc is not None:\n"
     "            _k4l_old = _acc * _k4w\n"
     "    if adv is not None:\n        adv = _wick_tower(w1t[:, None], adv)\n"),
    # exact radial: compensate its repeated (2,1) content in the residual before compression
    ("            if _PROTECT_FEED:\n",
     "            _k4l_newrad = None\n"
     "            if _k4on and _k4l_get('rad') == 'exact':\n"
     "                _sfr = _OWN_S_FEED * _prof_at('feed', layer_index, depth)\n"
     "                if _TWG is not None:\n"
     "                    _sfr = _sfr * _TWG['feed'][layer_index]\n"
     "                _Sn = ((w1[:, None] * w1[None, :]) * _k4l_offpre) * (0.25 * metric * _sfr)\n"
     "                _k4l_newrad = (w2, _Sn)\n"
     "                resid = resid - (w2[:, None] * _Sn) * (2.0 / 3.0)\n"
     "            if _PROTECT_FEED:\n"),
    # the post-activation off-diagonal core, built from the blocks as they are appended
    ("            out_tw = _append(adv, None, fac1, w2b_vec, n, Zb, Lb, _sb, e_b,\n",
     "            _k4l_offpost = None\n"
     "            if _k4on:\n"
     "                _wk = _k4lib.relu_wicks(alpha * sigma, sigma, _k4lib.WKEYS)\n"
     "                _pk1 = pK[(1,)]\n"
     "                _new = None\n"
     "                if _K4L.get('k211', 'exact') != 'none':\n"
     "                    _w2n = w2b_vec\n"
     "                    _Av = _pk1 @ fac1\n"
     "                    _X = (fac1 * (_w2n * _pk1)) @ fac1.T\n"
     "                    _Y2 = fac1.T * (_w2n * _Av)[:, None]\n"
     "                    _c = _X + _Y2 + _Y2.T\n"
     "                    _d = (fac1 * fac1) * _w2n[None, :]\n"
     "                    for _yh, _ch in ((yA, cA), (yB, cB)):\n"
     "                        _Sh = fac1 * _ch[None, :] + _ch[:, None] * fac1.T\n"
     "                        _c2, _d2, _ = _k4l_hub_cd(_yh, _Sh, _pk1)\n"
     "                        _c = _c + _c2\n"
     "                        _d = _d + _d2\n"
     "                    if _k4l_newrad is not None:\n"
     "                        _c2, _d2, _ = _k4l_hub_cd(_k4l_newrad[0], _k4l_newrad[1], _pk1)\n"
     "                        _c = _c + _c2\n"
     "                        _d = _d + _d2\n"
     "                    _new = _k4l_dist(_c, _d * off_mask, _pk1, off_mask)\n"
     "                _pre = dict(Cm=C11, D21=_k4l_D21, D22=D22, Off=_k4l_offpre, D3=D3, D4=D4)\n"
     "                _post = dict(pK1=_pk1, K2=K[(2,)], K11=K[(1, 1)] * off_mask,\n"
     "                             K21=K[(2, 1)] * off_mask)\n"
     "                _k4l_offpost, _P22 = _k4lib.core_off(n, _wk, metric, _pre, _post,\n"
     "                                                     k3_pre_e=_k4l_old, k3_post_pk1=_new)\n"
     "                if _K4L.get('e21b'):\n"
     "                    _k4l_offpost = _k4l_offpost + _k4lib.e21b_gain(\n"
     "                        _wk, alpha * sigma, C11, cov_pre, n, P22=_P22)\n"
     "                _k4l_offpost = fnp.where(fnp.isfinite(_k4l_offpost), _k4l_offpost,\n"
     "                                         dtype(0.0))\n"
     "            out_tw = _append(adv, None, fac1, w2b_vec, n, Zb, Lb, _sb, e_b,\n"),
    # pack the carried state
    ("    return (mu_out, cov_out, out_tw, c4_out, g_out)\n",
     "    if _k4on:\n"
     "        _rad = []\n"
     "        if _k4l_get('rad') == 'exact':\n"
     "            _rad = [(_y * w1t, (w1t[:, None] * _S) * w1t[None, :]) for _y, _S in _k4rad]\n"
     "            if _k4l_newrad is not None:\n"
     "                _rad.append(_k4l_newrad)\n"
     "        c4_out = {'c4': c4_out, 'off': _k4l_offpost, 'rad': _rad}\n"
     "    return (mu_out, cov_out, out_tw, c4_out, g_out)\n"),
]


def parse(spec):
    """'core=1,k211=exact,rad=exact,e21b=0,old=keep' -> dict; '' -> None (inert)."""
    if not spec:
        return None
    out = {"core": 1, "k211": "exact", "rad": "hub", "e21b": 0, "old": "keep"}
    for part in filter(None, spec.split(",")):
        k, v = part.split("=")
        if k in ("core", "e21b"):
            out[k] = int(v)
        elif k == "k211":
            assert v in ("exact", "trunc", "none"), v
            out[k] = v
        elif k == "rad":
            assert v in ("hub", "exact", "fold", "none"), v
            out[k] = v
        elif k == "old":
            assert v in ("keep", "drop"), v
            out[k] = v
        else:
            raise SystemExit(f"--k4l: unknown key {k}")
    return out
