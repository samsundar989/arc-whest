"""M5.1 hand-over: apply the masked star-Q31 fold (recent lineages, `rec` mode) to a v81d-layout file.

    python experiments/2026-10-10_m5/build_fold.py <src.py> <params.npz|-> <out.py> [--tau 2.0]

`params.npz` is a twin refit (`gain__fg` = g★ per layer; also bakes `gain__sg` into _STAR_GAIN and
the corrector / gain leaves are NOT touched -- Lane P's builder bakes those). `-` uses g★ = _STAR_GAIN.

The fold (twin arm `fold`, mode `rec`; see tools/twin/k4x_arm.py): D21's left operands over the
RECENT lineages' columns get
    LA += (M.A.A) diag(c4) * rv,   LP += (A.A.A) diag(c4) * rv / 3,
    rv_a = g★[l] * (-alpha_a / sigma_a) * 1{alpha_a^2 <= tau^2},
added after the joiner's weights / query views are read (so the old-tier join is not steered) and
inside the one recent-tier product (zero extra products). Elementwise only.
"""
import sys
from pathlib import Path

import numpy as np


def rep(src, a, b):
    if src.count(a) != 1:
        raise SystemExit(f"anchor occurs {src.count(a)}x:\n{a}")
    return src.replace(a, b, 1)


def build(src, gains, tau):
    src = rep(src, "_STAR_GAIN = [",
              f"#: M5.1 masked star-Q31 fold (recent lineages): per-layer gain g* and mask threshold.\n"
              f"_FOLD_GAIN = {[float(x) for x in gains]!r}\n_FOLD_TAU = {float(tau)!r}\n"
              f"#: set by `_layer` just before `_dslice21`, cleared right after (rv, w3 of recent).\n"
              f"_FOLD_V = None\n_STAR_GAIN = [")
    # _layer: publish rv before D21, clear after
    src = rep(src,
              "        D21 = (_dslice21(adv, off_mask) if adv is not None\n"
              "               else fnp.zeros((n, n), dtype=dtype))\n",
              "        global _FOLD_V\n"
              "        _FOLD_V = None\n"
              "        if (isinstance(c4, list) and adv is not None and adv.k > adv.ko\n"
              "                and layer_index < len(_FOLD_GAIN) and _FOLD_GAIN[layer_index] != 0.0):\n"
              "            _rv = fnp.where(alpha * alpha <= dtype(_FOLD_TAU * _FOLD_TAU),\n"
              "                            (alpha / sigma) * dtype(-_FOLD_GAIN[layer_index]), dtype(0.0))\n"
              "            _FOLD_V = (_rv, c4[adv.ko:])\n"
              "        D21 = (_dslice21(adv, off_mask) if adv is not None\n"
              "               else fnp.zeros((n, n), dtype=dtype))\n"
              "        _FOLD_V = None\n")
    # _shared_contract_lean: fold into the recent product's left operand
    src = rep(src,
              "    rec = _mm(fnp.concatenate([LA[:, c:], LP[:, c:]], axis=1),\n",
              "    LAr, LPr = LA[:, c:], LP[:, c:]\n"
              "    if _FOLD_V is not None:\n"
              "        _rv, _w3r = _FOLD_V\n"
              "        _w3c = _w3r[0] if len(_w3r) == 1 else fnp.concatenate(_w3r)\n"
              "        _Ar = tw.A_L if tw.ko else A_L\n"
              "        _Mr = tw.M if tw.ko else M\n"
              "        if kr is not None:\n"
              "            _rv = fnp.take(_rv, kr)\n"
              "        _A2 = (_Ar * _Ar) * _w3c\n"
              "        if kr is not None:\n"
              "            _A2k = fnp.take(_A2, kr, axis=0)\n"
              "            _Ak = fnp.take(_Ar, kr, axis=0)\n"
              "            _Mk = fnp.take(_Mr, kr, axis=0)\n"
              "        else:\n"
              "            _A2k, _Ak, _Mk = _A2, _Ar, _Mr\n"
              "        LAr = LAr + (_A2k * _Mk) * _rv[:, None]\n"
              "        LPr = LPr + (_A2k * _Ak) * (_rv * (1.0 / 3.0))[:, None]\n"
              "    rec = _mm(fnp.concatenate([LAr, LPr], axis=1),\n")
    return src


def main():
    a = sys.argv[1:]
    src_p, par, out_p = Path(a[0]), a[1], Path(a[2])
    tau = float(a[a.index("--tau") + 1]) if "--tau" in a else 2.0
    src = src_p.read_text(encoding="utf-8")
    if par == "-":
        import re
        m = re.search(r"_STAR_GAIN = (\[[^\]]*\])", src)
        gains = eval(m.group(1))
    else:
        z = np.load(par)
        gains = z["gain__fg"].tolist()
    out_p.write_text(build(src, gains, tau), encoding="utf-8", newline="\n")
    print(f"wrote {out_p}: g* = {[round(g, 3) for g in gains]}")


if __name__ == "__main__":
    main()
