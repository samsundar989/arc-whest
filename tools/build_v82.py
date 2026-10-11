"""Build v82 (HANDOFF 2026-10-10 Lane P5, P5.1 bundle) from v82a by exact text patches.

    uv run python tools/build_v82.py --items 1,2,3,4,5 --out estimator_k3_v82_bundle.py

Items (each one a separate measured step; prices from the exact cost model,
`docs/research/2026-10-10_flopmodel/`, alone on v81d):
  1  skip the old-tier joins at the last two non-final layers (L14, L15)     -0.90 % F
  2  last-layer output rows K16 = 800 (D3 / star-D4 of the rest = 0)        -0.79 %
  3  old-tier rebuild Q.[FA|FM] on the D-kept rows only                     -1.30 %
  4  fused W^T_r products in `_advance` (one `_mm`, padded to 32)           -0.37 %, -2.1k ops
  5  two-level symmetric sandwich split                                     -0.23 %, +585 ops
  6  covariance input pruning on the previous D set  (CLOSED: twin +24 %)  -1.24 %
  7  V8b: D21 columns on the D set + rho-scaled d21e (P5.2)                -3.55 %
Every per-shape table falls back to v82a's path off the suite shape.
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "estimator_k3_v82a_block.py"


def _rep(src: str, a: str, b: str) -> str:
    if src.count(a) != 1:
        raise SystemExit(f"anchor occurs {src.count(a)}x:\n{a}")
    return src.replace(a, b, 1)


def item1(src):
    src = _rep(src, "_SHARED_ON = True\n",
               "_SHARED_ON = True\n"
               "#: P5.1 item 1: no old-tier join at the last `_JOIN_SKIP_TAIL` non-final layers.\n"
               "#: The lineage stays exact (recent tier) to the end instead of being compressed.\n"
               "_JOIN_SKIP_TAIL = 2\n")
    return _rep(src,
                "        if _SHARED_ON and gens is None and not last_layer:\n"
                "            adv = _shared_join_lean(adv)\n"
                "            adv.rows = _kr\n",
                "        if _SHARED_ON and gens is None and not last_layer:\n"
                "            if layer_index < depth - 1 - _JOIN_SKIP_TAIL:\n"
                "                adv = _shared_join_lean(adv)\n"
                "            adv.rows = _kr\n")


def item2(src):
    src = _rep(src, "_KAS_SHAPE = (1024, 16)\n",
               "_KAS_SHAPE = (1024, 16)\n"
               "#: P5.1 item 2: the LAST layer transports only its K16 largest-alpha output rows;\n"
               "#: D3 and star-D4 of the others are 0 (their means are ~0). Suite shape only.\n"
               "_KAS_K16 = 800\n")
    src = _rep(src,
               "    adv = None if tw is None else _advance(WT, tw)\n    tw = None\n",
               "    _k16 = (int(_KAS_K16) if (_KAS_ON and last_layer and (n, depth) == _KAS_SHAPE)\n"
               "            else n)\n"
               "    _tw_last = tw if _k16 < n else None\n"
               "    adv = None if (tw is None or _tw_last is not None) else _advance(WT, tw)\n"
               "    tw = None\n")
    src = _rep(src,
               '    w = _wick(mu_pre, sigma, alpha, table, dtype, const["wick"])\n',
               '    w = _wick(mu_pre, sigma, alpha, table, dtype, const["wick"])\n'
               "    _inv16 = None\n"
               "    if _tw_last is not None:\n"
               "        _p16 = fnp.argsort(alpha)\n"
               "        _r16 = fnp.argsort(_p16)\n"
               "        _inv16 = fnp.where(_r16 >= n - _k16, _r16 - (n - _k16), _k16)\n"
               "        adv = _advance(fnp.take(WT, _p16[n - _k16:], axis=0), _tw_last)\n"
               "        _tw_last = None\n")
    src = _rep(src,
               "    D3 = _dslice3(adv) if adv is not None else fnp.zeros(n, dtype=dtype)\n",
               "    D3 = _dslice3(adv) if adv is not None else fnp.zeros(n, dtype=dtype)\n"
               "    if _inv16 is not None:\n"
               "        D3 = fnp.take(fnp.pad(D3, ((0, 1),)), _inv16)\n")
    return _rep(src,
                "        D4 = D4 + (((_AL * _AL) * (_AL * _ML)) @ _w3c) * (4.0 * _STAR_GAIN[layer_index])\n",
                "        _d4s = (((_AL * _AL) * (_AL * _ML)) @ _w3c) * (4.0 * _STAR_GAIN[layer_index])\n"
                "        if _inv16 is not None:\n"
                "            _d4s = fnp.take(fnp.pad(_d4s, ((0, 1),)), _inv16)\n"
                "        D4 = D4 + _d4s\n")


def item3(src):
    return _rep(src,
                "        F = _mm(tw.Q, fnp.concatenate([tw.FA, tw.FM], axis=1))\n",
                "        if tw.kd is not None:\n"
                "            # P5.1 item 3: rebuild on the D-kept rows only; the rest are zero\n"
                "            F = _mm(fnp.take(tw.Q, tw.kd[0], axis=0),\n"
                "                    fnp.concatenate([tw.FA, tw.FM], axis=1))\n"
                "            F = fnp.take(fnp.pad(F, ((0, 1), (0, 0))), tw.kd[1], axis=0)\n"
                "        else:\n"
                "            F = _mm(tw.Q, fnp.concatenate([tw.FA, tw.FM], axis=1))\n")


def item4(src):
    old = (
        "    WTr = WT if tw.rows is None else fnp.take(WT, tw.rows, axis=1)\n"
        "    A_D = None if tw.A_D is None else _mm(WTr, tw.A_D)\n"
        "    A_L = _mm(WTr, tw.A_L)\n"
        "    WTh = WT if tw.hcols is None else fnp.take(WT, tw.hcols, axis=1)\n"
        "    M = WTh if tw.M is None else fnp.concatenate([_mm(WTr, tw.M), WTh], axis=1)\n")
    new = (
        "    WTr = WT if tw.rows is None else fnp.take(WT, tw.rows, axis=1)\n"
        "    A_D = None if tw.A_D is None else _mm(WTr, tw.A_D)\n"
        "    # P5.1 item 4: every right operand in ONE product (fewer ops, one Strassen\n"
        "    # glue), padded to a multiple of 32 so the leaves stay aligned; the outputs\n"
        "    # are free column slices. Off the fused path (`_FUSE_ADV` False) it is v82a.\n"
        "    _parts = [x for x in (tw.A_L, tw.M, tw.Z, tw.Y, tw.F, tw.Q) if x is not None]\n"
        "    if _FUSE_ADV and len(_parts) > 1:\n"
        "        _tot = sum(int(x.shape[1]) for x in _parts)\n"
        "        _pad = (-_tot) % 32\n"
        "        if _pad:\n"
        "            _parts.append(fnp.zeros((WTr.shape[1], _pad), dtype=tw.A_L.dtype))\n"
        "        _Of = _mm(WTr, fnp.concatenate(_parts, axis=1))\n"
        "        _o = [0]\n"
        "\n"
        "        def _tr(x):\n"
        "            if x is None:\n"
        "                return None\n"
        "            r = _Of[:, _o[0]:_o[0] + x.shape[1]]\n"
        "            _o[0] += x.shape[1]\n"
        "            return r\n"
        "    else:\n"
        "        def _tr(x):\n"
        "            return None if x is None else _mm(WTr, x)\n"
        "    A_L = _tr(tw.A_L)\n"
        "    WTh = WT if tw.hcols is None else fnp.take(WT, tw.hcols, axis=1)\n"
        "    M = WTh if tw.M is None else fnp.concatenate([_tr(tw.M), WTh], axis=1)\n")
    src = _rep(src, old, new)
    src = _rep(src, "    Z = None if tw.Z is None else _mm(WTr, tw.Z)\n",
               "    Z = _tr(tw.Z)\n")
    src = _rep(src, "        out.Y, out.CA, out.CB = _mm(WTr, tw.Y), tw.CA, tw.CB\n",
               "        out.Y, out.CA, out.CB = _tr(tw.Y), tw.CA, tw.CB\n")
    src = _rep(src, "        out.F, out.R3, out.R4, out.w3b = _mm(WTr, tw.F), tw.R3, tw.R4, tw.w3b\n"
                    "    return _carry_shared(out, tw, None if tw.Q is None else _mm(WTr, tw.Q))\n",
               "        out.F, out.R3, out.R4, out.w3b = _tr(tw.F), tw.R3, tw.R4, tw.w3b\n"
               "    return _carry_shared(out, tw, _tr(tw.Q))\n")
    return _rep(src, "_STRASSEN_BIG_BLOCK = 16\n",
                "_STRASSEN_BIG_BLOCK = 16\n"
                "#: P5.1 item 4: `_advance` issues its leg products as one fused `_mm`.\n"
                "_FUSE_ADV = True\n")


def item5(src):
    src = _rep(src,
               "def _p4_sym_sandwich(WT, cov, W):\n",
               "def _p4_sym_rest2(X, W):\n"
               "    \"\"\"P5.1 item 5: the second product of `W^T C W` with a TWO-level symmetric\n"
               "    split -- rows `[:h]` full, rows `[h:h+q]` on columns `h:`, rows `[h+q:]` on\n"
               "    columns `h+q:`; the rest are transposes (exact: the result is symmetric).\"\"\"\n"
               "    n = X.shape[0]\n"
               "    h = n // 2\n"
               "    q = h // 2\n"
               "    top = _mm(X[:h], W)\n"
               "    t2 = _mm(X[h:h + q], W[:, h:])\n"
               "    b22 = _mm(X[h + q:], W[:, h + q:])\n"
               "    tl = top[:, h:].T\n"
               "    # R[h+q+i, h+j] = R[h+j, h+q+i] = t2[j, q+i]  ->  t2[:, q:].T\n"
               "    return fnp.block([[top], [tl[:q], t2], [tl[q:], t2[:, q:].T, b22]])\n"
               "\n"
               "\n"
               "def _p4_sym_sandwich(WT, cov, W):\n")
    old_s = ("    X = _mm(WT, cov)\n"
             "    n = X.shape[0]\n"
             "    if n % 2:\n"
             "        return _mm(X, W)\n"
             "    h = n // 2\n")
    new_s = ("    X = _mm(WT, cov)\n"
             "    n = X.shape[0]\n"
             "    if n % 2:\n"
             "        return _mm(X, W)\n"
             "    if n % 4 == 0:\n"
             "        return _p4_sym_rest2(X, W)\n"
             "    h = n // 2\n")
    src = _rep(src, old_s, new_s)
    old_r = ("    n = X.shape[0]\n"
             "    if n % 2:\n"
             "        return _mm(X, W)\n"
             "    h = n // 2\n"
             "    top = _mm(X[:h], W)\n"
             "    b22 = _mm(X[h:], W[:, h:])\n"
             "    return fnp.block([[top], [top[:, h:].T, b22]])\n\n\ndef _p3_compress")
    new_r = ("    n = X.shape[0]\n"
             "    if n % 2:\n"
             "        return _mm(X, W)\n"
             "    if n % 4 == 0:\n"
             "        return _p4_sym_rest2(X, W)\n"
             "    h = n // 2\n"
             "    top = _mm(X[:h], W)\n"
             "    b22 = _mm(X[h:], W[:, h:])\n"
             "    return fnp.block([[top], [top[:, h:].T, b22]])\n\n\ndef _p3_compress")
    return _rep(src, old_r, new_r)


def item6(src):
    """Covariance input pruning: the sandwich reads C_post on the PREVIOUS layer's
    D-kept rows and columns only (dead neurons' variances and covariances dropped)."""
    src = _rep(src, "_KAS_SHAPE = (1024, 16)\n",
               "_KAS_SHAPE = (1024, 16)\n"
               "#: P5.1 item 6: the previous layer's D-kept indices, published by `_layer`.\n"
               "_COV_PRUNE = True\n"
               "_COV_KD = None\n")
    src = _rep(src, "    global _CUR_LAYER, _CUR_DEPTH\n    _CUR_LAYER, _CUR_DEPTH = layer_index, depth\n",
               "    global _CUR_LAYER, _CUR_DEPTH, _COV_KD\n    _CUR_LAYER, _CUR_DEPTH = layer_index, depth\n"
               "    _ckd = _COV_KD if layer_index > 0 else None\n"
               "    _COV_KD = None\n")
    src = _rep(src,
               "    if _P4_FINAL_VAR and last_layer:\n"
               "        # P4: the last layer reads means only -- the diagonal of W^T C W is\n"
               "        # rowsum((W^T C) . W^T), one product instead of two.\n"
               "        _X4 = _mm(WT, cov)\n"
               "        cov_pre = None\n"
               "        var = fnp.maximum((_X4 * WT).sum(axis=1), dtype(1e-12))\n",
               "    _WTc, _covc = WT, cov\n"
               "    if _ckd is not None:\n"
               "        _WTc = fnp.take(WT, _ckd, axis=1)\n"
               "        _covc = fnp.take(fnp.take(cov, _ckd, axis=0), _ckd, axis=1)\n"
               "    if _P4_FINAL_VAR and last_layer:\n"
               "        # P4: the last layer reads means only -- the diagonal of W^T C W is\n"
               "        # rowsum((W^T C) . W^T), one product instead of two.\n"
               "        _X4 = _mm(_WTc, _covc)\n"
               "        cov_pre = None\n"
               "        var = fnp.maximum((_X4 * _WTc).sum(axis=1), dtype(1e-12))\n")
    src = _rep(src,
               "            cov_pre = (_p4_sym_sandwich(WT, cov, W) if _P4_BLOCKSYM\n"
               "                       else _mm(_mm(WT, cov), W))\n",
               "            cov_pre = (_p4_sym_sandwich(_WTc, _covc, _WTc.T) if _P4_BLOCKSYM\n"
               "                       else _mm(_mm(_WTc, _covc), _WTc.T))\n")
    return _rep(src,
                "        if _KR < n:\n            _kr = _perm[n - _KR:]\n",
                "        if _KR < n:\n            _kr = _perm[n - _KR:]\n"
                "        if _COV_PRUNE and _KD < n:\n            _COV_KD = _perm[n - _KD:]\n")


def item7(src, rho=None):
    """V8b (P5.2): D21 COLUMNS restricted to the D set (right operand rows gathered), the
    K_D x K_D block scattered back on both axes; the corrector's d21e then reads only kept
    columns and is scaled by the per-layer energy ratio rho (twin: -0.16 % unrefit)."""
    import json
    rho = rho or json.loads((ROOT / "experiments/2026-10-10_p5/d2/rho.json").read_text())["rho"]
    src = _rep(src,
               "    rec = _mm(fnp.concatenate([LA[:, c:], LP[:, c:]], axis=1),\n"
               "              fnp.concatenate([tw.A_L if tw.ko else A_L, tw.M if tw.ko else M], axis=1).T)\n",
               "    _R = fnp.concatenate([tw.A_L if tw.ko else A_L, tw.M if tw.ko else M], axis=1)\n"
               "    if kr is not None:\n"
               "        # P5.2 V8b: D21 COLUMNS on the D set too (right operand rows gathered)\n"
               "        _R = fnp.take(_R, kr, axis=0)\n"
               "    rec = _mm(fnp.concatenate([LA[:, c:], LP[:, c:]], axis=1), _R.T)\n")
    src = _rep(src, "    old = _mm(Xold, tw.Q.T)\n",
               "    old = _mm(Xold, (tw.Q if kr is None else fnp.take(tw.Q, kr, axis=0)).T)\n")
    src = _rep(src,
               "            acc = acc + _mm(RYk * (2.0 / 3.0), tw.Y.T)\n",
               "            acc = acc + _mm(RYk * (2.0 / 3.0),\n"
               "                            (tw.Y if kr is None else fnp.take(tw.Y, kr, axis=0)).T)\n")
    src = _rep(src,
               "        acc = acc + _mm(PPL, tw.Z.T) * (1.0 / 3.0)\n",
               "        acc = acc + _mm(PPL, (tw.Z if kr is None else fnp.take(tw.Z, kr, axis=0)).T) * (1.0 / 3.0)\n")
    src = _rep(src,
               "            acc = fnp.take(fnp.pad(acc, ((0, 1), (0, 0))), kd[1], axis=0)\n",
               "            # P5.2 V8b: rows AND columns, so the gather runs on both axes\n"
               "            acc = fnp.take(fnp.take(fnp.pad(acc, ((0, 1), (0, 1))), kd[1], axis=0),\n"
               "                           kd[1], axis=1)\n")
    if "acc = acc + _mm(TT if kr is None else fnp.take(TT, kr, axis=0), tw.F.T)\n" in src:
        src = _rep(src, "acc = acc + _mm(TT if kr is None else fnp.take(TT, kr, axis=0), tw.F.T)\n",
                   "acc = acc + _mm(TT if kr is None else fnp.take(TT, kr, axis=0),\n"
                   "                            (tw.F if kr is None else fnp.take(tw.F, kr, axis=0)).T)\n")
    src = _rep(src, "_KAS_SHAPE = (1024, 16)\n",
               "_KAS_SHAPE = (1024, 16)\n"
               "#: P5.2 V8b: d21e energy ratio all / kept columns per layer (FIT-100, twin control chain).\n"
               f"_D21E_RHO = {[round(float(r), 6) for r in rho]}\n")
    return _rep(src,
                "        d21e = (D21 * D21).sum(axis=1) * (inv2 * inv2)\n",
                "        _rh = (_D21E_RHO[_CUR_LAYER] if (_KAS_ON and (sigma.shape[0], _CUR_DEPTH) == _KAS_SHAPE\n"
                "                                       and _CUR_LAYER < len(_D21E_RHO)) else 1.0)\n"
                "        if _rh != 1.0:\n"
                "            # P5.2 V8b: d21e over KEPT columns only, rescaled to the all-column energy\n"
                "            d21e = (D21 * D21).sum(axis=1) * ((inv2 * inv2) * _rh)\n"
                "        else:\n"
                "            d21e = (D21 * D21).sum(axis=1) * (inv2 * inv2)\n")


def item0(src):
    """v82a (P5.0c): `fnp.block` reassembly in `_sb` and the sandwich + `nan_to_num`
    closure guard -- bit-identical. For building from a v81d-family file (build_star)."""
    src = _rep(src,
               "    return fnp.concatenate([fnp.concatenate([c11, c12], axis=2),\n"
               "                            fnp.concatenate([c21, c22], axis=2)], axis=1)\n",
               "    # P5.0c: one `block` bills numel(out) once; the nested concatenates billed it twice.\n"
               "    return fnp.block([[c11, c12], [c21, c22]])\n")
    old = "    return fnp.concatenate([top, fnp.concatenate([top[:, h:].T, b22], axis=1)], axis=0)\n"
    if src.count(old) != 2:
        raise SystemExit(f"sandwich anchor occurs {src.count(old)}x")
    src = src.replace(old, "    return fnp.block([[top], [top[:, h:].T, b22]])\n")
    return _rep(src,
                '        X = fnp.einsum("tij,ti,tj->ij", Mt, Li, Rj)\n'
                "        X = fnp.where(fnp.isfinite(X), X, dtype(0.0))\n",
                '        X = fnp.einsum("tij,ti,tj->ij", Mt, Li, Rj)\n'
                "        X = fnp.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)\n")


ITEMS = {"0": item0, "1": item1, "2": item2, "3": item3, "4": item4, "5": item5, "6": item6, "7": item7}


def build(items, base=BASE):
    src = base.read_text(encoding="utf-8")
    for k in items:
        src = ITEMS[k](src)
    return src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="1,2,3,4,5")
    ap.add_argument("--base", default=str(BASE))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    items = [x for x in a.items.split(",") if x]
    Path(a.out).write_text(build(items, Path(a.base)), encoding="utf-8", newline="\n")
    print(f"{a.out}: items {items}")


if __name__ == "__main__":
    main()
