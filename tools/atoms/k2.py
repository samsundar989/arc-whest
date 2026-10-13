"""Atom k2 (Lane K, HANDOFF 2026-10-12 Lane K, K2): transport only leg output rows alpha > -tau_o.

Not bit-identical. Requires `k0` and `k1` FIRST (`--atoms k0+k1:off+k2:3.0` or `k0+k1:3.0+k2:3.0`).
Twin arm: `mask+dress+bank+fold+kgain --kg o=<ktab_o..>` (+ rows=... for K1).

alpha needs only mu and C, so the transport moves after the covariance sandwich; `_advance` then
multiplies only the K_O output rows (top K_O by alpha; FIT-100 q50, nearest 32; last layer
included). Index sets in the tower's row space ("O space"):
  * KAS R / D rows and k1's kink rows are subsets of O (all top-K by alpha; kink scores outside O
    are pushed to the end), mapped into O space by one gather of `_oinv`;
  * D3 and the star part of D4 come out on K_O rows and scatter back by a gather from the
    zero-padded vector; the dropped rows are filled with the gain mode,
        D3 = 6 v mu sigma^2,   D4 = 12 v4 sigma^4,
    v from k0 (now also at the last layer, over kink rows inside O), v4 = mean(D4 / sigma^4) / 12
    over the same rows, times `_K2_VG4[l]` (twin leaf gain/kv4);
  * the wick scale and the fold's row vector are gathered onto O rows.
Exact prices (x3_tables S3, out > -tau_o, in > -2.0, last layer included): -3.27 % (3.5),
-4.64 % (3.0), -6.19 % (2.5) F.

`make('<tau>')` reads `gain__kv4` from the params when present, else needs the ':unrefit' suffix
(e.g. 'k2:3.0:unrefit' -> gains 1.0).
"""
from tools.atoms import rep

ARMS = {"3.5": "o35", "3.0": "o30", "2.5": "o25"}
#: Baked from experiments/2026-10-13_k/ktab_o*.json (FIT-100 q50 of non-dead at tau_o, nearest 32).
OTAB = {"o35": [1024, 1024, 1024, 1024, 1024, 1024, 992, 992, 960, 960, 928, 928, 896, 896, 896, 864],
        "o30": [1024, 1024, 1024, 1024, 992, 992, 960, 960, 928, 928, 896, 896, 864, 864, 832, 832],
        "o25": [1024, 1024, 1024, 992, 992, 960, 928, 928, 896, 864, 864, 832, 832, 800, 800, 800]}


def _patch(src, otab, vg4):
    src = rep(src,
              "_KAS_K = {\"R\": ",
              "#: Lane K k2: transported leg output rows per layer (top K by alpha), and the v4 gain.\n"
              f"_K2_O = {otab!r}\n_K2_VG4 = {vg4!r}\n"
              "_KAS_K = {\"R\": ")
    # ---- transport after alpha, on the O rows only ----
    src = rep(src,
              "    adv = None if tw is None else _advance(WT, tw)\n    tw = None\n",
              "    adv = None\n")
    src = rep(src,
              "    # ---- KAS selection (variant A): fixed per-layer counts, device-side indices ----\n",
              "    # ---- Lane K k2: the output-row set exists before the transport ----\n"
              "    _orow = _oinv = _ok = _operm = None\n"
              "    _KO = (int(_K2_O[layer_index]) if (_KAS_ON and (n, depth) == _KAS_SHAPE and tw is not None\n"
              "                                      and layer_index < len(_K2_O)) else n)\n"
              "    if _KO < n:\n"
              "        _operm = fnp.argsort(alpha)\n"
              "        _orow = _operm[n - _KO:]\n"
              "        _orank = fnp.argsort(_operm)\n"
              "        _oinv = fnp.where(_orank >= n - _KO, _orank - (n - _KO), _KO)\n"
              "        _ok = fnp.where(_orank >= n - _KO, dtype(1.0), dtype(0.0))\n"
              "    adv = None if tw is None else _advance(WT, tw, _orow)\n"
              "    tw = None\n"
              "    # ---- KAS selection (variant A): fixed per-layer counts, device-side indices ----\n")
    src = rep(src,
              "        _perm = fnp.argsort(alpha)\n        _KR = int(_KAS_K[\"R\"][layer_index])\n",
              "        _perm = fnp.argsort(alpha) if _operm is None else _operm\n"
              "        _KR = int(_KAS_K[\"R\"][layer_index])\n")
    src = rep(src,
              "                _pa = fnp.argsort(_sc)\n",
              "                _pa = fnp.argsort(_sc if _ok is None else fnp.where(_ok > 0.0, _sc, dtype(1e30)))\n")
    src = rep(src,
              "            adv.kd = (_kri, _krv, _kci, _kcv)\n",
              "            if _orow is not None:\n"
              "                # Lane K k2: the tower's rows are the O set -> O-space row indices\n"
              "                _kri = fnp.take(_oinv, _kri)\n"
              "                _kci = fnp.take(_oinv, _kci)\n"
              "            adv.kd = (_kri, _krv, _kci, _kcv)\n")
    src = rep(src, "def _advance(WT, tw):\n", "def _advance(WT, tw, orow=None):\n")
    src = rep(src,
              "    WTr = WT if tw.rows is None else fnp.take(WT, tw.rows, axis=1)\n",
              "    WTo = WT if orow is None else fnp.take(WT, orow, axis=0)   # Lane K k2: output rows\n"
              "    WTr = WTo if tw.rows is None else fnp.take(WTo, tw.rows, axis=1)\n")
    src = rep(src,
              "    WTh = WT if tw.hcols is None else fnp.take(WT, tw.hcols, axis=1)\n",
              "    WTh = WTo if tw.hcols is None else fnp.take(WTo, tw.hcols, axis=1)\n")
    # ---- D3: scatter back, v over kink rows inside O (also at the last layer), fill ----
    src = rep(src,
              "    D3 = _dslice3(adv) if adv is not None else fnp.zeros(n, dtype=dtype)\n    _k0v = None\n"
              "    if _kz is not None and layer_index < len(_K0_VG):\n",
              "    D3 = _dslice3(adv) if adv is not None else fnp.zeros(n, dtype=dtype)\n"
              "    if _orow is not None:\n"
              "        D3 = fnp.take(fnp.pad(D3, ((0, 1),)), _oinv)\n"
              "    _k0v = None\n"
              "    if (_kz is not None or _orow is not None) and layer_index < len(_K0_VG):\n")
    src = rep(src,
              "        _k0a = fnp.where(alpha * alpha <= dtype(6.25), alpha, dtype(0.0))\n",
              "        _k0a = fnp.where(alpha * alpha <= dtype(6.25), alpha, dtype(0.0))\n"
              "        if _ok is not None:\n"
              "            _k0a = _k0a * _ok\n")
    src = rep(src,
              "                / fnp.maximum((_k0a * alpha).sum() * 6.0, dtype(1e-30))) * _K0_VG[layer_index]\n",
              "                / fnp.maximum((_k0a * alpha).sum() * 6.0, dtype(1e-30))) * _K0_VG[layer_index]\n"
              "    if _orow is not None:\n"
              "        # Lane K k2: the rows not transported get the gain mode's D3\n"
              "        D3 = D3 + ((mu_pre * var) * (_k0v * 6.0)) * (1.0 - _ok)\n")
    # ---- D4: star part scattered back; dropped rows get 12 v4 sigma^4 ----
    src = rep(src,
              "        D4 = D4 + (((_AL * _AL) * (_AL * _ML)) @ _w3c) * (4.0 * _STAR_GAIN[layer_index])\n",
              "        _d4s = (((_AL * _AL) * (_AL * _ML)) @ _w3c) * (4.0 * _STAR_GAIN[layer_index])\n"
              "        if _orow is not None:\n"
              "            _d4s = fnp.take(fnp.pad(_d4s, ((0, 1),)), _oinv)\n"
              "        D4 = D4 + _d4s\n"
              "    if _orow is not None:\n"
              "        _k2m = fnp.where(alpha * alpha <= dtype(6.25), _ok, dtype(0.0))\n"
              "        _k2v4 = (((D4 / (var * var)) * _k2m).sum()\n"
              "                 / fnp.maximum(_k2m.sum() * 12.0, dtype(1e-30))) * _K2_VG4[layer_index]\n"
              "        D4 = D4 * _ok + ((var * var) * (_k2v4 * 12.0)) * (1.0 - _ok)\n")
    # ---- wick scale, R compaction and the fold's row vector on O rows ----
    src = rep(src,
              "        adv = _wick_tower(w1t[:, None], adv)\n        if _kr is not None:\n"
              "            adv = _kas_rows(adv, _kr)\n",
              "        adv = _wick_tower((w1t if _orow is None else fnp.take(w1t, _orow))[:, None], adv)\n"
              "        if _kr is not None:\n"
              "            adv = _kas_rows(adv, _kr if _orow is None else fnp.take(_oinv, _kr))\n")
    src = rep(src,
              "            _FOLD_V = (_rv, c4[adv.ko:])\n",
              "            if _orow is not None:\n"
              "                _rv = fnp.take(_rv, _orow)\n"
              "            _FOLD_V = (_rv, c4[adv.ko:])\n")
    return src


def make(arg):
    tau, _, mode = arg.partition(":")
    if tau not in ARMS or mode not in ("", "unrefit"):
        raise SystemExit(f"atom k2: unknown arm {arg!r} (3.5 | 3.0 | 2.5, optionally ':unrefit')")
    otab = OTAB[ARMS[tau]]

    def f(src, z):
        if mode == "unrefit":
            vg4 = [1.0] * 16
        elif "gain__kv4" in z.files:
            vg4 = [float(x) for x in z["gain__kv4"]]
        else:
            raise SystemExit("atom k2: params have no gain__kv4 (refit with stage ...k), or use ':unrefit'")
        return _patch(src, otab, vg4)
    return f


def apply(src, z):
    raise SystemExit("atom k2 needs an arm: k2:3.5 | k2:3.0 | k2:2.5 (optionally ':unrefit')")
