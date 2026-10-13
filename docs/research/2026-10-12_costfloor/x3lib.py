"""x3 cost-modelling: patched variants of estimator_k3_v84_age6.py for the exact shape-only interpreter.

build(cfg) -> source text. Every patch is an exact text replacement whose anchor must occur once.
Shapes are exact; numerics are irrelevant (the interpreter only tracks shapes), so index arrays that a
real implementation would derive from argsort(alpha) are spelled here as free `fnp.zeros(K, int64)`
placeholders where the real argsort is already billed elsewhere in the layer.

cfg keys (all optional):
  R, Dr, Dc, H : 16-entry per-layer keep counts (R = leg rows kept after the wick scale = next transport's
                 inner dim; Dr/Dc = D21 rows / columns; H = birth columns of the lineage born at that layer)
  HC           : 16-entry per-layer correction (columns removed from the youngest lineage's A_L in the
                 transport) -- used for S2b's two-column-set lineage (H = (wA+wM)/2, HC = (wM-wA)/2)
  O            : 16-entry transport OUTPUT row counts (None = all 1024 rows)
  olast        : apply O at the last layer too (default True when O given)
  habs, drabs  : rank H / D21 rows by alpha^2 (kink-active) -- adds the real argsort ops
  age          : _SHARED_AGE
  old          : True (rank-256 old tier) | False (no old tier: only the `age` youngest lineages carried)
  rank         : _SHARED_RANK
  glue         : 'std' (8.25) | 'out' (5.25: 3 raw copies per side, no block) | 'min' (3.75: no copies at all)
  depth        : 'std' | ('opt', min_leaf)
  lastk        : K youngest lineages transported into the last layer (None = all)
"""
import os, re
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "estimator_k3_v84_age6.py")

BASE = {"R": [1024, 1024, 992, 960, 928, 896, 896, 864, 832, 832, 800, 768, 768, 768, 736, 736],
        "D": [1024, 1024, 1024, 992, 992, 960, 928, 928, 896, 864, 864, 832, 832, 800, 800, 800],
        "H": [1024, 1024, 1024, 992, 992, 960, 928, 928, 896, 864, 864, 832, 832, 800, 800, 800]}


def rep(s, a, b, cnt=1):
    c = s.count(a)
    if c != cnt:
        raise SystemExit("anchor occurs %dx (want %d): %r" % (c, cnt, a[:100]))
    return s.replace(a, b)


def build(cfg, src=None):
    s = open(src or SRC, encoding="utf-8").read()
    n = 1024
    R = list(cfg.get("R", BASE["R"])); Dr = list(cfg.get("Dr", BASE["D"])); Dc = list(cfg.get("Dc", BASE["D"]))
    H = list(cfg.get("H", BASE["H"])); HC = list(cfg.get("HC", [0] * 16))
    O = cfg.get("O"); O = [n] * 16 if O is None else list(O)
    if not cfg.get("olast", True):
        O[15] = n
    habs = bool(cfg.get("habs")); drabs = bool(cfg.get("drabs"))
    # ---- tables ------------------------------------------------------------------------------
    s = rep(s,
            '_KAS_K = {"R": [1024, 1024, 992, 960, 928, 896, 896, 864, 832, 832, 800, 768, 768, 768, 736, 736],\n'
            '          "D": [1024, 1024, 1024, 992, 992, 960, 928, 928, 896, 864, 864, 832, 832, 800, 800, 800],\n'
            '          "H": [1024, 1024, 1024, 992, 992, 960, 928, 928, 896, 864, 864, 832, 832, 800, 800, 800]}\n',
            '_KAS_K = {"R": %r,\n          "D": %r,\n          "Dc": %r,\n          "H": %r,\n          "HC": %r,\n          "O": %r}\n'
            '_X3_HABS = %r\n_X3_DRABS = %r\n_X3_LASTK = %r\n_X3_GENS = %r\n'
            % (R, Dr, Dc, H, HC, O, habs, drabs, cfg.get("lastk"), (None if cfg.get("old", True) else int(cfg.get("age", 6)))))
    # ---- transport: output rows O, youngest-A_L column correction (S2b) -----------------------
    s = rep(s,
            "    WTr = WT if tw.rows is None else fnp.take(WT, tw.rows, axis=1)\n",
            "    WTo = WT if orow is None else fnp.take(WT, orow, axis=0)\n"
            "    WTr = WTo if tw.rows is None else fnp.take(WTo, tw.rows, axis=1)\n")
    s = rep(s, "def _advance(WT, tw):\n", "def _advance(WT, tw, orow=None, hc=0):\n")
    s = rep(s,
            "    _parts = [x for x in (tw.A_L, tw.M, tw.Z, tw.Y, tw.F, tw.Q) if x is not None]\n"
            "    if _FUSE_ADV and len(_parts) > 1:\n"
            "        _tot = sum(int(x.shape[1]) for x in _parts)\n"
            "        _pad = (-_tot) % 32\n"
            "        if _pad:\n"
            "            _parts.append(fnp.zeros((WTr.shape[1], _pad), dtype=tw.A_L.dtype))\n"
            "        _Of = _mm(WTr, fnp.concatenate(_parts, axis=1))\n",
            "    _parts = [x for x in (tw.A_L, tw.M, tw.Z, tw.Y, tw.F, tw.Q) if x is not None]\n"
            "    if _FUSE_ADV and len(_parts) > 1:\n"
            "        _tot0 = sum(int(x.shape[1]) for x in _parts)\n"
            "        if hc:\n"
            "            _parts[0] = _parts[0][:, :_parts[0].shape[1] - hc]\n"
            "        _tot = sum(int(x.shape[1]) for x in _parts)\n"
            "        _pad = (-_tot) % 32\n"
            "        if _pad:\n"
            "            _parts.append(fnp.zeros((WTr.shape[1], _pad), dtype=tw.A_L.dtype))\n"
            "        _Of = _mm(WTr, fnp.concatenate(_parts, axis=1))\n"
            "        if hc:\n"
            "            _Of = fnp.zeros((_Of.shape[0], _tot0), dtype=tw.A_L.dtype)   # shape-only stand-in\n")
    s = rep(s, "    WTh = WT if tw.hcols is None else fnp.take(WT, tw.hcols, axis=1)\n",
            "    WTh = WTo if tw.hcols is None else fnp.take(WTo, tw.hcols, axis=1)\n")
    # ---- layer: the O set exists before the transport (alpha_l needs only mu, cov) ------------
    s = rep(s,
            "    adv = None if tw is None else _advance(WT, tw)\n    tw = None\n",
            "    _orow = _oinv = None\n"
            "    if (_KAS_ON and (n, depth) == _KAS_SHAPE and layer_index < len(_KAS_K['O'])\n"
            "            and int(_KAS_K['O'][layer_index]) < n and tw is not None):\n"
            "        # shape-only placeholders: the real indices are slices of argsort(alpha), billed below\n"
            "        _orow = fnp.zeros((int(_KAS_K['O'][layer_index]),), dtype=fnp.int64)\n"
            "        _oinv = fnp.zeros((n,), dtype=fnp.int64)\n"
            "    if last_layer and _X3_LASTK is not None and tw is not None:\n"
            "        tw = _truncate(tw, int(_X3_LASTK), n)\n"
            "        if isinstance(c4, list):\n"
            "            c4 = c4[-tw.k:]\n"
            "    _hc = 0\n"
            "    if (n, depth) == _KAS_SHAPE and 1 <= layer_index <= len(_KAS_K['HC']):\n"
            "        _hc = int(_KAS_K['HC'][layer_index - 1])\n"
            "    adv = None if tw is None else _advance(WT, tw, _orow, _hc)\n    tw = None\n")
    # KAS selection
    s = rep(s,
            "        _perm = fnp.argsort(alpha)\n"
            "        _KR = int(_KAS_K[\"R\"][layer_index])\n"
            "        _KD = int(_KAS_K[\"D\"][layer_index])\n"
            "        _KH = int(_KAS_K[\"H\"][layer_index])\n"
            "        if _KH < n:\n"
            "            _rkh = fnp.argsort(_perm)\n"
            "            _mh = fnp.where(_rkh >= n - _KH, dtype(1.0), dtype(0.0))\n"
            "            _kh = _perm[n - _KH:]\n"
            "        if _KR < n:\n"
            "            _kr = _perm[n - _KR:]\n"
            "        if _KD < n and adv is not None:\n"
            "            _rank = fnp.argsort(_perm)\n"
            "            adv.kd = (_perm[n - _KD:],\n"
            "                      fnp.where(_rank >= n - _KD, _rank - (n - _KD), _KD))\n"
            "            _kz = fnp.where(_rank >= n - _KD, dtype(1.0), dtype(0.0))\n",
            "        _perm = fnp.argsort(alpha)\n"
            "        _KR = int(_KAS_K[\"R\"][layer_index])\n"
            "        _KD = int(_KAS_K[\"D\"][layer_index])\n"
            "        _KC = int(_KAS_K[\"Dc\"][layer_index])\n"
            "        _KH = int(_KAS_K[\"H\"][layer_index])\n"
            "        _pa = _rka = None\n"
            "        if (_X3_HABS and _KH < n) or (_X3_DRABS and _KD < n and adv is not None):\n"
            "            _pa = fnp.argsort(alpha * alpha)\n"
            "            _rka = fnp.argsort(_pa)\n"
            "        if _KH < n:\n"
            "            if _X3_HABS:\n"
            "                _mh = fnp.where(_rka < _KH, dtype(1.0), dtype(0.0))\n"
            "                _kh = _pa[:_KH]\n"
            "            else:\n"
            "                _rkh = fnp.argsort(_perm)\n"
            "                _mh = fnp.where(_rkh >= n - _KH, dtype(1.0), dtype(0.0))\n"
            "                _kh = _perm[n - _KH:]\n"
            "        if _KR < n:\n"
            "            _kr = _perm[n - _KR:]\n"
            "        if (_KD < n or _KC < n or _orow is not None) and adv is not None:\n"
            "            _rank = fnp.argsort(_perm)\n"
            "            _kci = _perm[n - _KC:]\n"
            "            _kcv = fnp.where(_rank >= n - _KC, _rank - (n - _KC), _KC)\n"
            "            if _X3_DRABS:\n"
            "                if _pa is None:\n"
            "                    _pa = fnp.argsort(alpha * alpha)\n"
            "                    _rka = fnp.argsort(_pa)\n"
            "                _kri = _pa[:_KD]\n"
            "                _krv = fnp.where(_rka < _KD, _rka, _KD)\n"
            "                _kz = fnp.where(_rka < _KD, dtype(1.0), dtype(0.0))\n"
            "            elif _KD == _KC:\n"
            "                _kri, _krv = _kci, _kcv\n"
            "                _kz = fnp.where(_rank >= n - _KD, dtype(1.0), dtype(0.0))\n"
            "            else:\n"
            "                _kri = _perm[n - _KD:]\n"
            "                _krv = fnp.where(_rank >= n - _KD, _rank - (n - _KD), _KD)\n"
            "                _kz = fnp.where(_rank >= n - _KD, dtype(1.0), dtype(0.0))\n"
            "            adv.kd = (_kri, _krv, _kci, _kcv)\n")
    # D3 / D4-star scatter back to n when the tower lives on the O rows
    s = rep(s,
            "    D3 = _dslice3(adv) if adv is not None else fnp.zeros(n, dtype=dtype)\n",
            "    D3 = _dslice3(adv) if adv is not None else fnp.zeros(n, dtype=dtype)\n"
            "    if _orow is not None:\n"
            "        D3 = fnp.take(fnp.pad(D3, ((0, 1),)), _oinv)\n")
    s = rep(s,
            "        D4 = D4 + (((_AL * _AL) * (_AL * _ML)) @ _w3c) * (4.0 * _STAR_GAIN[layer_index])\n",
            "        _d4s = (((_AL * _AL) * (_AL * _ML)) @ _w3c) * (4.0 * _STAR_GAIN[layer_index])\n"
            "        if _orow is not None:\n"
            "            _d4s = fnp.take(fnp.pad(_d4s, ((0, 1),)), _oinv)\n"
            "        D4 = D4 + _d4s\n")
    s = rep(s,
            "        adv = _wick_tower(w1t[:, None], adv)\n",
            "        adv = _wick_tower((w1t if _orow is None else fnp.take(w1t, _orow))[:, None], adv)\n")
    # ---- D21: separate row / column sets -------------------------------------------------------
    s = rep(s,
            "        kd = tw.kd\n"
            "        kr = None if kd is None else kd[0]\n"
            "        acc = _shared_contract_lean(tw, LA, LP, A_L, M, kr)\n",
            "        kd = tw.kd\n"
            "        kr = None if kd is None else kd[0]\n"
            "        kc = None if kd is None else kd[2]\n"
            "        acc = _shared_contract_lean(tw, LA, LP, A_L, M, kr, kc)\n")
    s = rep(s,
            "            acc = acc + _mm(RYk * (2.0 / 3.0),\n"
            "                            (tw.Y if kr is None else fnp.take(tw.Y, kr, axis=0)).T)\n",
            "            acc = acc + _mm(RYk * (2.0 / 3.0),\n"
            "                            (tw.Y if kr is None else fnp.take(tw.Y, kc, axis=0)).T)\n")
    s = rep(s,
            "        acc = acc + _mm(PPL, (tw.Z if kr is None else fnp.take(tw.Z, kr, axis=0)).T) * (1.0 / 3.0)\n",
            "        acc = acc + _mm(PPL, (tw.Z if kr is None else fnp.take(tw.Z, kc, axis=0)).T) * (1.0 / 3.0)\n")
    s = rep(s,
            "            acc = fnp.take(fnp.take(fnp.pad(acc, ((0, 1), (0, 1))), kd[1], axis=0),\n"
            "                           kd[1], axis=1)\n",
            "            acc = fnp.take(fnp.take(fnp.pad(acc, ((0, 1), (0, 1))), kd[1], axis=0),\n"
            "                           kd[3], axis=1)\n")
    s = rep(s, "def _shared_contract_lean(tw, LA, LP, A_L, M, kr=None):\n",
            "def _shared_contract_lean(tw, LA, LP, A_L, M, kr=None, kc=None):\n")
    s = rep(s,
            "        _R = fnp.take(_R, kr, axis=0)\n",
            "        _R = fnp.take(_R, kc, axis=0)\n")
    s = rep(s,
            "    old = _mm(Xold, (tw.Q if kr is None else fnp.take(tw.Q, kr, axis=0)).T)\n",
            "    old = _mm(Xold, (tw.Q if kr is None else fnp.take(tw.Q, kc, axis=0)).T)\n")
    # ---- ages / old tier -----------------------------------------------------------------------
    if "age" in cfg:
        s = rep(s, "_SHARED_AGE = 6\n", "_SHARED_AGE = %d\n" % int(cfg["age"]))
    if "rank" in cfg:
        s = rep(s, "_SHARED_RANK = 256\n", "_SHARED_RANK = %d\n" % int(cfg["rank"]))
    if not cfg.get("old", True):
        s = rep(s, "_SHARED_ON = True\n", "_SHARED_ON = False\n")
        s = rep(s, "        gens = _fit_generations(width, depth, budget)\n", "        gens = _X3_GENS\n")
    # width-aware truncate (the shipped one assumes one lineage width)
    s = rep(s, "def _truncate(tw, gens, n):\n",
            "def _truncate(tw, gens, n):\n"
            "    if tw is None or gens is None or tw.k <= gens:\n"
            "        return tw\n"
            "    g = int(gens)\n"
            "    ws = _lin_widths(tw)\n"
            "    keep = sum(ws[-g:])\n"
            "    keepM = sum(ws[-g:-1]) if g > 1 else 0\n"
            "    M = None if (tw.M is None or g <= 1) else tw.M[:, tw.M.shape[1] - keepM:]\n"
            "    r = tw.L[0].shape[2]\n"
            "    Ls = []\n"
            "    drop = tw.k - g\n"
            "    for Lg in tw.L:\n"
            "        c = Lg.shape[0]\n"
            "        if drop >= c:\n"
            "            drop -= c\n"
            "            continue\n"
            "        Ls.append(Lg[drop:])\n"
            "        drop = 0\n"
            "    out = _Tower(None, tw.A_L[:, tw.A_L.shape[1] - keep:], M, tw.w2b[tw.w2b.shape[0] - keep:], g,\n"
            "                 tw.Z[:, tw.Z.shape[1] - g * r:], Ls, tw.sb[tw.sb.shape[0] - keep:],\n"
            "                 tw.eb[tw.eb.shape[0] - keep:])\n"
            "    if tw.Y is not None:\n"
            "        out.Y = tw.Y[:, tw.Y.shape[1] - 2 * g:]\n"
            "        out.CA = tw.CA[tw.CA.shape[0] - keep:]\n"
            "        out.CB = tw.CB[tw.CB.shape[0] - keep:]\n"
            "    out.rows, out.hcols = tw.rows, tw.hcols\n"
            "    return out\n\n\n"
            "def _truncate_shipped(tw, gens, n):\n")
    # ---- Strassen glue -------------------------------------------------------------------------
    glue = cfg.get("glue", "std")
    if glue != "std":
        ncopy = {"out": 3, "min": 0}[glue]
        s = rep(s, "    SA = fnp.concatenate([a11, a12, s4, a22, s1, s2, s3], axis=0)\n",
                "    SA = fnp.x3cat([a11, a12, s4, a22, s1, s2, s3], %d)\n" % ncopy)
        s = rep(s, "    SB = fnp.concatenate([b11, b21, b22, t4, t1, t2, t3], axis=0)\n",
                "    SB = fnp.x3cat([b11, b21, b22, t4, t1, t2, t3], %d)\n" % ncopy)
        s = rep(s, "    return fnp.block([[c11, c12], [c21, c22]])\n",
                "    return fnp.x3block([[c11, c12], [c21, c22]])\n")
    dp = cfg.get("depth", "std")
    if dp != "std":
        min_leaf = int(dp[1]); maxd = 8
        gl = {"std": "4*b*mh*kh + 4*b*kh*nh + 7*b*mh*kh + 7*b*kh*nh + 7*b*mh*nh + b*m*n",
              "out": "7*b*mh*kh + 7*b*kh*nh + 7*b*mh*nh",
              "min": "4*b*mh*kh + 4*b*kh*nh + 7*b*mh*nh"}[glue]
        s = rep(s, "def _depth_for(m, k, n):\n", "def _depth_for_old(m, k, n):\n")
        s = rep(s, "def _mm(A, B):\n",
                "def _sbc(b, m, k, n, d):\n"
                "    if d <= 0 or (m % 2) or (k % 2) or (n % 2):\n"
                "        return b * m * n * (2 * k - 1)\n"
                "    mh, kh, nh = m // 2, k // 2, n // 2\n"
                "    g = " + gl + "\n"
                "    return g + _sbc(7 * b, mh, kh, nh, d - 1)\n\n\n"
                "_DF_CACHE = {}\n\n\n"
                "def _depth_for(m, k, n):\n"
                "    key = (m, k, n)\n"
                "    if key in _DF_CACHE:\n"
                "        return _DF_CACHE[key]\n"
                "    best = None\n"
                "    for d in range(0, %d + 1):\n"
                "        if d and (min(m, k, n) >> d) < %d:\n"
                "            break\n"
                "        c = _sbc(1, m, k, n, d)\n"
                "        if best is None or c < best[0]:\n"
                "            best = (c, d)\n"
                "    _DF_CACHE[key] = best[1]\n"
                "    return best[1]\n\n\n"
                "def _mm(A, B):\n" % (maxd, min_leaf))
    return s
