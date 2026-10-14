"""Atom k5 (Lane K, HANDOFF 2026-10-12 Lane K, K5): per-network counts instead of the fixed q50 tables.

Not bit-identical. Requires `k0`, `k1`, `k2` and `k3` FIRST (it replaces their table lookups):
`--atoms sb2+sbtrim+<depth>+k0+k1:2.5_3.0+k2:3.0+k3+k5`. Twin arm: `--kg ...,adapt=1` (tools/twin/kgain_arm.py).

Per layer (1024 x 16 only, as KAS), this network's threshold counts, each rounded to the nearest multiple of 32
(Python round, clamped to [32, n]) -- the same definitions the tables were built from (FIT-100 q50):
    R    alpha >= -2.0             (KAS R: rows carried to the next transport)
    D/H  alpha >= -2.5             (KAS D: D21 columns / corrector; KAS H: birth columns)
    W    -2.5 <= alpha <= 3.0      (k1: D21 lineage rows)
    O    alpha >= -3.0             (k2: transported output rows; also at the last layer)
    A    alpha^2 <= 6.25           (k3: A-side birth columns, ranked inside the H set)
The five counts are packed into ONE integer, sum c_i * 1025^i (< 2^53), and read back with a single `int()`:
one scalar materialization per layer (16 per MLP; PLAN rev 18 §5 allows <= 48 when measured). Mean F over nets
is unchanged (the tables are the medians of these counts); per-net F spreads (CV ~3 %, deep review §5.3).
Twin screen (stack B, R1 B params, VAL-100): -1.42 % raw (per-net -1.37 +- 0.27 %, 68/100) vs the fixed tables.
"""
from tools.atoms import rep


def apply(src, z):
    src = rep(src,
              "    # ---- Lane K k2: the output-row set exists before the transport ----\n"
              "    _orow = _oinv = _ok = _operm = None\n",
              "    # ---- Lane K k5: this network's counts (one packed read-back per layer) ----\n"
              "    _k5 = None\n"
              "    if _KAS_ON and (n, depth) == _KAS_SHAPE and layer_index < len(_KAS_K[\"R\"]):\n"
              "        _k5d = alpha >= dtype(-2.5)\n"
              "        _k5p = ((alpha >= dtype(-2.0)).sum()\n"
              "                + _k5d.sum() * 1025\n"
              "                + (_k5d & (alpha <= dtype(3.0))).sum() * 1050625\n"
              "                + (alpha >= dtype(-3.0)).sum() * 1076890625\n"
              "                + (alpha * alpha <= dtype(6.25)).sum() * 1103812890625)\n"
              "        _k5v = int(_k5p)\n"
              "        _k5 = {}\n"
              "        for _k5n in (\"R\", \"D\", \"W\", \"O\", \"A\"):\n"
              "            _k5c = _k5v % 1025\n"
              "            _k5v //= 1025\n"
              "            _k5[_k5n] = max(32, min(n, int(round(_k5c / 32.0)) * 32))\n"
              "    # ---- Lane K k2: the output-row set exists before the transport ----\n"
              "    _orow = _oinv = _ok = _operm = None\n")
    src = rep(src,
              "    _KO = (int(_K2_O[layer_index]) if (_KAS_ON and (n, depth) == _KAS_SHAPE and tw is not None\n"
              "                                      and layer_index < len(_K2_O)) else n)\n",
              "    _KO = (int(_K2_O[layer_index]) if (_KAS_ON and (n, depth) == _KAS_SHAPE and tw is not None\n"
              "                                      and layer_index < len(_K2_O)) else n)\n"
              "    if _k5 is not None and tw is not None:\n"
              "        _KO = _k5[\"O\"]\n")
    src = rep(src,
              "        _KR = int(_KAS_K[\"R\"][layer_index])\n"
              "        _KD = int(_KAS_K[\"D\"][layer_index])\n"
              "        _KH = int(_KAS_K[\"H\"][layer_index])\n",
              "        _KR = int(_KAS_K[\"R\"][layer_index]) if _k5 is None else _k5[\"R\"]\n"
              "        _KD = int(_KAS_K[\"D\"][layer_index]) if _k5 is None else _k5[\"D\"]\n"
              "        _KH = int(_KAS_K[\"H\"][layer_index]) if _k5 is None else _k5[\"D\"]\n")
    src = rep(src,
              "            _KA = int(_K3_A[layer_index]) if layer_index < len(_K3_A) else _KH\n",
              "            _KA = int(_K3_A[layer_index]) if layer_index < len(_K3_A) else _KH\n"
              "            if _k5 is not None:\n"
              "                _KA = _k5[\"A\"]\n")
    src = rep(src,
              "        _K1R = int(_K1_ROWS[layer_index]) if layer_index < len(_K1_ROWS) else n\n",
              "        _K1R = int(_K1_ROWS[layer_index]) if layer_index < len(_K1_ROWS) else n\n"
              "        if _k5 is not None:\n"
              "            _K1R = _k5[\"W\"]\n")
    return src
