"""Build the KAS (keep-alive-subset) compact chain from a v73-family estimator.

    uv run python tools/build_kas.py --ktab experiments/2026-10-06_mask/ktab/rd_m32.json \
        --out estimator_k3_v75_kas.py                      # base defaults to v73_gb

PLAN rev 18 Lane P, P2 (HANDOFF 2026-10-06). Measured in P1 (RESULTS.md § "PLAN rev 18 —
Lane P") on the twin mask arm: decided neurons carry almost nothing through the kappa3
machinery, so their arithmetic can be skipped. This builder ships two of the three roles:

  R  ROWS. After layer l's Wick scale every carried leg (A_L, M, Z, Y, old-tier Q) keeps
     only the K_R(l) rows with the largest alpha_l, and so do the legs born at layer l.
     The next transport multiplies W^T[:, kept] (n x K) by the compact legs, which is the
     inner dimension of the most expensive matmuls in the chain. The youngest block's
     implicit M = I is never compacted (it enters the next transport as W^T, free).
  D  D21 ROWS ONLY, and no corrector delta on those rows ("Dz": their d21e feature would
     read 0, which the corrector never saw). The (2,1) readout forms only the K_D(l) kept
     rows; the dead rows
     are zero, scattered back as a GATHER from the block padded with one zero row (grader
     arrays are immutable). Columns stay, so the mean corrector's d21e feature is exact
     for every kept neuron (the two-sided mask cost +7 % raw through that feature).

H (dead birth columns) is not built here: it changes each lineage's column width, and the
wide (n, k n) layout assumes one width.

Selection is VARIANT A: fixed per-layer counts (`_KAS_K`), indices from `fnp.argsort`, zero
scalar materializations, constant FLOPs. Any shape other than the tables' (1024, 16) runs
the unpruned chain (whestbench issue #149: the smoke test uses non-suite shapes).

`_STRASSEN_MIN_BLOCK` drops 32 -> 20. Measured (experiments/2026-10-06_mask/strassen_probe):
at 32 every compact K < 1024 loses the fifth Strassen level, and K = 992 then costs 4.4 %
MORE than K = 1024. At 20 the compact products track K/n; v73's own products are unchanged
except a few that gain a level (whole chain -0.02 % FLOPs, +220 ops).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PATCHES = [
    # ---- constants -------------------------------------------------------------
    ("_STRASSEN_MIN_BLOCK = 32\n",
     "#: KAS: 20, not 32. A compact inner dimension K < 1024 (the pruned rows) only gets the\n"
     "#: fifth Strassen level with a smaller leaf; at 32 K = 992 cost 4.4 % MORE than 1024\n"
     "#: (experiments/2026-10-06_mask/strassen_probe.json). v73's own products: -0.02 %.\n"
     "_STRASSEN_MIN_BLOCK = 20\n"),
    # ---- the tower carries its row selection and the D21 row selection ----------
    ('"_qU", "_qold", "F", "R3", "R4", "w3b", "_p3")',
     '"_qU", "_qold", "F", "R3", "R4", "w3b", "_p3",\n'
     "                 # KAS: `rows` = kept row indices of a compacted tower (None = all);\n"
     "                 # `kd` = (kept D21 rows, inverse index) for this layer's readout\n"
     '                 "rows", "kd")'),
    ("        self._p3 = None\n",
     "        self._p3 = None\n        self.rows = self.kd = None\n"),
    # ---- transport: W^T restricted to the kept rows -------------------------------
    ("    A_D = None if tw.A_D is None else _mm(WT, tw.A_D)\n    A_L = _mm(WT, tw.A_L)\n"
     "    M = WT if tw.M is None else fnp.concatenate([_mm(WT, tw.M), WT], axis=1)\n",
     "    # KAS (R): a row-compacted tower multiplies only the kept columns of W^T. The\n"
     "    # youngest block's implicit M = I is never compacted, so it still enters as W^T.\n"
     "    WTr = WT if tw.rows is None else fnp.take(WT, tw.rows, axis=1)\n"
     "    A_D = None if tw.A_D is None else _mm(WTr, tw.A_D)\n    A_L = _mm(WTr, tw.A_L)\n"
     "    M = WT if tw.M is None else fnp.concatenate([_mm(WTr, tw.M), WT], axis=1)\n"),
    ("_mm(WT, tw.Z)", "_mm(WTr, tw.Z)"),
    ("_mm(WT, tw.Y)", "_mm(WTr, tw.Y)"),
    ("_mm(WT, tw.F)", "_mm(WTr, tw.F)"),
    ("_mm(WT, tw.Q))", "_mm(WTr, tw.Q))"),
    # ---- D21 readout on the kept rows only ---------------------------------------
    ("def _shared_contract_lean(tw, LA, LP, A_L, M):\n"
     '    """LEAN: `[LA | LP] @ [A | M]^T` as ONE product per part (old via Q, recent exact)."""\n'
     "    n = LA.shape[0]\n",
     "def _shared_contract_lean(tw, LA, LP, A_L, M, kr=None):\n"
     '    """LEAN: `[LA | LP] @ [A | M]^T` as ONE product per part (old via Q, recent exact).\n\n'
     "    KAS: `kr` restricts the LEFT rows (the D21 rows formed) after the joiner's weights\n"
     "    are read from the full rows. `n` is the LINEAGE WIDTH, not a row count.\n"
     '    """\n'
     "    n = tw.w2b.shape[0] // tw.k\n"),
    ("        tw._qU = (LA[:, c:c + n], LP[:, c:c + n])\n",
     "        tw._qU = (LA[:, c:c + n], LP[:, c:c + n])\n"
     "    if kr is not None:\n"
     "        LA = fnp.take(LA, kr, axis=0)\n"
     "        LP = fnp.take(LP, kr, axis=0)\n"),
    ("        acc = _shared_contract_lean(tw, LA, LP, A_L, M)\n"
     "        if tw.F is not None:\n"
     "            acc = acc + _mm(TT, tw.F.T)\n"
     "        if tw.Y is not None:\n"
     "            acc = acc + _mm(RY * (2.0 / 3.0), tw.Y.T)\n",
     "        kd = tw.kd\n"
     "        kr = None if kd is None else kd[0]\n"
     "        acc = _shared_contract_lean(tw, LA, LP, A_L, M, kr)\n"
     "        if tw.F is not None:\n"
     "            acc = acc + _mm(TT if kr is None else fnp.take(TT, kr, axis=0), tw.F.T)\n"
     "        if tw.Y is not None:\n"
     "            RYk = RY if kr is None else fnp.take(RY, kr, axis=0)\n"
     "            acc = acc + _mm(RYk * (2.0 / 3.0), tw.Y.T)\n"
     "        if kr is not None:\n"
     "            PPL = fnp.take(PPL, kr, axis=0)\n"),
    ("        return (acc + _mm(PPL, tw.Z.T) * (1.0 / 3.0)) * off_mask\n",
     "        acc = acc + _mm(PPL, tw.Z.T) * (1.0 / 3.0)\n"
     "        if kd is not None:\n"
     "            # KAS (D, rows only): the dead rows were never formed. Scatter the kept\n"
     "            # rows back as a GATHER from the block padded with one zero row.\n"
     "            acc = fnp.take(fnp.pad(acc, ((0, 1), (0, 0))), kd[1], axis=0)\n"
     "        return acc * off_mask\n"),
    # ---- the joiner's lineage width is not its row count --------------------------
    ("    n = tw.A_L.shape[0]\n    r = _shared_rank(n)\n"
     "    if not r or (tw.k - tw.ko) < _SHARED_AGE or tw._wts is None:\n"
     "        return tw\n    if _QUERY_BASIS",
     "    n = tw.w2b.shape[0] // tw.k                  # KAS: lineage width, not rows\n"
     "    r = _shared_rank(n)\n"
     "    if not r or (tw.k - tw.ko) < _SHARED_AGE or tw._wts is None:\n"
     "        return tw\n    if _QUERY_BASIS"),
    # ---- _layer: select, compact, carry ------------------------------------------
    ('    w = _wick(mu_pre, sigma, alpha, table, dtype, const["wick"])\n',
     '    w = _wick(mu_pre, sigma, alpha, table, dtype, const["wick"])\n'
     "    # ---- KAS selection (variant A): fixed per-layer counts, device-side indices ----\n"
     "    _kr = _kd = _kz = None\n"
     "    if (_KAS_ON and not last_layer and (n, depth) == _KAS_SHAPE\n"
     '            and layer_index < len(_KAS_K["R"])):\n'
     "        _perm = fnp.argsort(alpha)\n"
     '        _KR = int(_KAS_K["R"][layer_index])\n'
     '        _KD = int(_KAS_K["D"][layer_index])\n'
     "        if _KR < n:\n"
     "            _kr = _perm[n - _KR:]\n"
     "        if _KD < n and adv is not None:\n"
     "            _rank = fnp.argsort(_perm)\n"
     "            adv.kd = (_perm[n - _KD:],\n"
     "                      fnp.where(_rank >= n - _KD, _rank - (n - _KD), _KD))\n"
     "            _kz = fnp.where(_rank >= n - _KD, dtype(1.0), dtype(0.0))\n"),
    # Dz: no mean correction on the rows whose D21 was never formed
    ("        mu_out = mu_out + _corr_delta(\n"
     '            layer_index, const["corr"], mu_pre, sigma, D3, D4, W,\n'
     "            None if last_layer else D21, mu, mu_out, n, dtype)\n",
     "        _cd = _corr_delta(\n"
     '            layer_index, const["corr"], mu_pre, sigma, D3, D4, W,\n'
     "            None if last_layer else D21, mu, mu_out, n, dtype)\n"
     "        if _kz is not None:\n"
     "            # KAS (D): their d21e feature would read 0, which the corrector was\n"
     "            # never fitted on (P1: +2.5 % raw at tau 2.5 with it, +0.6 % without)\n"
     "            _cd = _cd * _kz\n"
     "        mu_out = mu_out + _cd\n"),
    ("        adv = _wick_tower(w1t[:, None], adv)\n"
     "        if _SHARED_ON and gens is None and not last_layer:\n"
     "            adv = _shared_join_lean(adv)\n",
     "        adv = _wick_tower(w1t[:, None], adv)\n"
     "        if _kr is not None:\n"
     "            adv = _kas_rows(adv, _kr)\n"
     "        if _SHARED_ON and gens is None and not last_layer:\n"
     "            adv = _shared_join_lean(adv)\n"
     "            adv.rows = _kr\n"),
    ("            out_tw = _append(adv, None, fac1, w2b_vec, n, Zb, Lb, _sb, e_b,\n"
     "                             fnp.stack([yA, yB], axis=1), cA, cB)\n",
     "            _Yb = fnp.stack([yA, yB], axis=1)\n"
     "            if _kr is not None:\n"
     "                # KAS (R): the newborn's transported legs are born compact too\n"
     "                fac1 = fnp.take(fac1, _kr, axis=0)\n"
     "                Zb = fnp.take(Zb, _kr, axis=0)\n"
     "                _Yb = fnp.take(_Yb, _kr, axis=0)\n"
     "            out_tw = _append(adv, None, fac1, w2b_vec, n, Zb, Lb, _sb, e_b,\n"
     "                             _Yb, cA, cB)\n"
     "            out_tw.rows = _kr\n"),
]

#: --hd: H = DEAD BIRTH COLUMNS (P1: dropping a dead column whole -- A_L, M, two-leg,
#: hubs, slice -- costs +0.15 % raw at tau 2.5; always-on columns must stay, their kappa4
#: hubs are not small). Each lineage then has its own width K_H(b), so the P2 pages `L`
#: travel as a LIST of groups of consecutive equal-width lineages, each (g, w, r), and the
#: two readouts that reshape the wide layout per lineage loop over those groups. The masks at
#: birth are the twin's `Hd` mode exactly, then only the live columns are stored.
HD_PATCHES = [
    ("        _KD = int(_KAS_K[\"D\"][layer_index])\n",
     "        _KD = int(_KAS_K[\"D\"][layer_index])\n"
     "        _KH = int(_KAS_K[\"H\"][layer_index])\n"
     "        if _KH < n:\n"
     "            _rkh = fnp.argsort(_perm)\n"
     "            _mh = fnp.where(_rkh >= n - _KH, dtype(1.0), dtype(0.0))\n"
     "            _kh = _perm[n - _KH:]\n"),
    ("    _kr = _kd = _kz = None\n", "    _kr = _kd = _kz = _mh = _kh = None\n"),
    ('                 "rows", "kd")',
     '                 "rows", "kd",\n'
     "                 # KAS (H): live birth columns of the YOUNGEST lineage (its M = I)\n"
     '                 "hcols")'),
    ("        self.rows = self.kd = None\n", "        self.rows = self.kd = self.hcols = None\n"),
    # the youngest block's implicit M = I keeps only its live columns
    ("    M = WT if tw.M is None else fnp.concatenate([_mm(WTr, tw.M), WT], axis=1)\n",
     "    WTh = WT if tw.hcols is None else fnp.take(WT, tw.hcols, axis=1)\n"
     "    M = WTh if tw.M is None else fnp.concatenate([_mm(WTr, tw.M), WTh], axis=1)\n"),
    # P2 pages as groups
    ("                     None if Lb is None else Lb[None], sb, eb)\n",
     "                     None if Lb is None else [Lb[None]], sb, eb)\n"),
    ("                 None if Lb is None else fnp.concatenate([tw.L, Lb[None]], axis=0),\n",
     "                 None if Lb is None else _l_append(tw.L, Lb),\n"),
    ("        r = tw.L.shape[2]\n", "        r = tw.L[0].shape[2]\n"),
    ("                     tw.Z[:, -(g * r):], tw.L[-g:],\n",
     "                     tw.Z[:, -(g * r):], [fnp.concatenate(tw.L, axis=0)[-g:]],\n"),
    ("    n, k = M.shape[0], tw.k\n    r = tw.L.shape[2]\n"
     "    # (n, k r) -> (k, n, r): SMALL, `k n r` not `k n^2`.\n"
     "    Z3 = fnp.reshape(tw.Z, (n, k, r)).transpose(1, 0, 2)\n"
     "    Zres = fnp.reshape((Z3 @ tw.L.transpose(0, 2, 1)).transpose(1, 0, 2),\n"
     "                       (n, k * n))\n"
     "    PP = M * M\n"
     "    MP = PP * tw.sb + (A_L * M) * (3.0 * tw.eb) + Zres * M\n"
     "    PPL = fnp.reshape(PP, (n, k, n)).transpose(1, 0, 2) @ tw.L\n"
     "    tw._p2 = (MP, PP, fnp.reshape(PPL.transpose(1, 0, 2), (n, k * r)))\n",
     "    n = M.shape[0]\n"
     "    PP = M * M\n"
     "    # KAS (H): one batched product per GROUP of equal-width lineages\n"
     "    Zres, PPL = [], []\n"
     "    off = lin = 0\n"
     "    for Lg in tw.L:\n"
     "        g, w, r = Lg.shape[0], Lg.shape[1], Lg.shape[2]\n"
     "        Z3 = fnp.reshape(tw.Z[:, lin * r:(lin + g) * r], (n, g, r)).transpose(1, 0, 2)\n"
     "        Zres.append(fnp.reshape((Z3 @ Lg.transpose(0, 2, 1)).transpose(1, 0, 2),\n"
     "                                (n, g * w)))\n"
     "        P3 = fnp.reshape(PP[:, off:off + g * w], (n, g, w)).transpose(1, 0, 2) @ Lg\n"
     "        PPL.append(fnp.reshape(P3.transpose(1, 0, 2), (n, g * r)))\n"
     "        off += g * w\n"
     "        lin += g\n"
     "    Zres = _cat1(Zres)\n"
     "    MP = PP * tw.sb + (A_L * M) * (3.0 * tw.eb) + Zres * M\n"
     "    tw._p2 = (MP, PP, _cat1(PPL))\n"),
    ("    n, k = M.shape[0], tw.k\n"
     "    Y3 = fnp.reshape(tw.Y, (n, k, 2))\n"
     "    cA = fnp.reshape(tw.CA, (k, n))\n"
     "    cB = fnp.reshape(tw.CB, (k, n))\n"
     "    yA = Y3[:, :, 0]\n"
     "    yB = Y3[:, :, 1]\n"
     "    G = fnp.reshape(yA[:, :, None] * cA[None, :, :] + yB[:, :, None] * cB[None, :, :],\n"
     "                    (n, k * n))\n"
     "    AM3 = fnp.reshape(A_L * M, (n, k, n))\n"
     "    RA = (AM3 * cA[None, :, :]).sum(axis=2)\n"
     "    RB = (AM3 * cB[None, :, :]).sum(axis=2)\n",
     "    n, k = M.shape[0], tw.k\n"
     "    Y3 = fnp.reshape(tw.Y, (n, k, 2))\n"
     "    yA = Y3[:, :, 0]\n"
     "    yB = Y3[:, :, 1]\n"
     "    AM = A_L * M\n"
     "    # KAS (H): per GROUP of equal-width lineages\n"
     "    G, RA, RB = [], [], []\n"
     "    off = lin = 0\n"
     "    for Lg in tw.L:\n"
     "        g, w = Lg.shape[0], Lg.shape[1]\n"
     "        cA = fnp.reshape(tw.CA[off:off + g * w], (g, w))\n"
     "        cB = fnp.reshape(tw.CB[off:off + g * w], (g, w))\n"
     "        G.append(fnp.reshape(yA[:, lin:lin + g, None] * cA[None, :, :]\n"
     "                             + yB[:, lin:lin + g, None] * cB[None, :, :], (n, g * w)))\n"
     "        AM3 = fnp.reshape(AM[:, off:off + g * w], (n, g, w))\n"
     "        RA.append((AM3 * cA[None, :, :]).sum(axis=2))\n"
     "        RB.append((AM3 * cB[None, :, :]).sum(axis=2))\n"
     "        off += g * w\n"
     "        lin += g\n"
     "    G, RA, RB = _cat1(G), _cat1(RA), _cat1(RB)\n"),
    # lineage widths, not w2b / k
    ("    n = tw.w2b.shape[0] // tw.k\n    c = tw.ko * n\n",
     "    _w = _lin_widths(tw)\n    c = sum(_w[:tw.ko])\n    n = _w[tw.ko]\n"),
    ("    n = tw.w2b.shape[0] // tw.k                  # KAS: lineage width, not rows\n",
     "    n = _lin_widths(tw)[tw.ko]                   # KAS: the joiner's width\n"),
    # birth: the twin's Hd masks, in the same places
    ("        fac1 = w1e[:, None] * C11\n",
     "        fac1 = w1e[:, None] * C11\n"
     "        if _mh is not None:\n"
     "            fac1 = fac1 * _mh[None, :]\n"
     "            w2m = w2m * _mh\n"),
    ("        S_sep = (e_b[:, None] * C11) * w1e[None, :]\n",
     "        S_sep = (e_b[:, None] * C11) * w1e[None, :]\n"
     "        if _mh is not None:\n"
     "            S_sep = S_sep * _mh[:, None]\n"),
    ("            resid = S21 - S_sep\n",
     "            if _mh is not None:\n"
     "                cA = cA * _mh\n"
     "                cB = cB * _mh\n"
     "            resid = S21 - S_sep\n"),
    ("            if _PROTECT_FEED:\n",
     "            if _mh is not None:\n"
     "                e_b = e_b * _mh\n"
     "                resid = resid * _mh[:, None]\n"
     "                S3 = S3 * _mh\n"
     "            if _PROTECT_FEED:\n"),
    ("            out_tw = _append(adv, None, fac1, w2b_vec, n, Zb, Lb, _sb, e_b,\n"
     "                             _Yb, cA, cB)\n"
     "            out_tw.rows = _kr\n",
     "            if _kh is not None:\n"
     "                # KAS (H): a dead birth column carries nothing -- store the live ones\n"
     "                fac1 = fnp.take(fac1, _kh, axis=1)\n"
     "                w2b_vec = fnp.take(w2b_vec, _kh)\n"
     "                _sb = fnp.take(_sb, _kh)\n"
     "                e_b = fnp.take(e_b, _kh)\n"
     "                cA = fnp.take(cA, _kh)\n"
     "                cB = fnp.take(cB, _kh)\n"
     "                Lb = fnp.take(Lb, _kh, axis=0)\n"
     "            out_tw = _append(adv, None, fac1, w2b_vec, n, Zb, Lb, _sb, e_b,\n"
     "                             _Yb, cA, cB)\n"
     "            out_tw.rows = _kr\n"
     "            out_tw.hcols = _kh\n"),
]

HD_HELPERS = (
    "def _cat1(xs):\n"
    "    return xs[0] if len(xs) == 1 else fnp.concatenate(xs, axis=1)\n\n\n"
    "def _l_append(L, Lb):\n"
    '    """Append a P2 page to its width group (a new group when the width changes)."""\n'
    "    if L[-1].shape[1] == Lb.shape[0]:\n"
    "        return L[:-1] + [fnp.concatenate([L[-1], Lb[None]], axis=0)]\n"
    "    return L + [Lb[None]]\n\n\n"
    "def _lin_widths(tw):\n"
    '    """Every lineage\'s column width, oldest first, from the page groups (Python ints)."""\n'
    "    out = []\n"
    "    for Lg in tw.L:\n"
    "        out += [Lg.shape[1]] * Lg.shape[0]\n"
    "    return out\n\n\n")

HELPERS_ANCHOR = "def _truncate(tw, gens, n):\n"


def helpers(ktab):
    K = ktab["K"]
    return (
        "# ==========================================================================\n"
        "# KAS -- decided-neuron pruning (PLAN rev 18 Lane P, P2; tools/build_kas.py)\n"
        "# ==========================================================================\n"
        "#: Per-layer keep counts, (width, depth) they were measured at, and the switch.\n"
        f"#: Rule: {ktab['rule']} of the alpha kept fractions on {ktab['stats']}\n"
        f"#: at config {ktab['config']}. Any other shape runs the unpruned chain.\n"
        "_KAS_ON = True\n"
        "_KAS_SHAPE = (1024, 16)\n"
        + (f"_KAS_K = {{\"R\": {K['R']},\n          \"D\": {K['D']},\n"
           f"          \"H\": {K['H']}}}\n\n\n" if "H" in K else
           f"_KAS_K = {{\"R\": {K['R']},\n          \"D\": {K['D']}}}\n\n\n")
        + ("" if "H" not in K else HD_HELPERS) +
        "def _kas_rows(tw, idx):\n"
        '    """Keep only rows `idx` of every carried leg (R). Lineage width unchanged."""\n'
        "    out = _Tower(None, fnp.take(tw.A_L, idx, axis=0),\n"
        "                 None if tw.M is None else fnp.take(tw.M, idx, axis=0),\n"
        "                 tw.w2b, tw.k, None if tw.Z is None else fnp.take(tw.Z, idx, axis=0),\n"
        "                 tw.L, tw.sb, tw.eb)\n"
        "    if tw.Y is not None:\n"
        "        out.Y, out.CA, out.CB = fnp.take(tw.Y, idx, axis=0), tw.CA, tw.CB\n"
        "    if tw.F is not None:\n"
        "        out.F, out.R3, out.R4, out.w3b = (fnp.take(tw.F, idx, axis=0), tw.R3, tw.R4,\n"
        "                                          tw.w3b)\n"
        "    _carry_shared(out, tw, None if tw.Q is None else fnp.take(tw.Q, idx, axis=0))\n"
        "    out._wts = tw._wts\n"
        "    out.rows = idx\n"
        "    return out\n\n\n")


def build(base: Path, ktab: dict) -> str:
    src = base.read_text(encoding="utf-8")
    for a, b in PATCHES + (HD_PATCHES if "H" in ktab["K"] else []):
        if src.count(a) != 1:
            raise SystemExit(f"anchor occurs {src.count(a)}x:\n{a}")
        src = src.replace(a, b, 1)
    if src.count(HELPERS_ANCHOR) != 1:
        raise SystemExit("helpers anchor")
    return src.replace(HELPERS_ANCHOR, helpers(ktab) + HELPERS_ANCHOR, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(ROOT / "estimator_k3_v73_gb.py"))
    ap.add_argument("--ktab", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src = build(Path(a.base), json.loads(Path(a.ktab).read_text()))
    Path(a.out).write_text(src, encoding="utf-8", newline="\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
