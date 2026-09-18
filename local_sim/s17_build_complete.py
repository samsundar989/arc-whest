"""Compose the registered P3, shared-gram, own-control, P4 package.

The frozen v41 gram/P4 source is the arithmetic base. P3 is added at the
existing tower sites, with its consumers using reconstructed old legs. This
builder never reads labels or seed blocks and writes only new versioned files.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import s2_build_p3 as p3
import s14_own_fit_controls as own
import s4_build_p4 as p4

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "estimator_k3_v41_pairq_a4r256_gram_p4d5.py"
BASE_SHA = "1649056ab80732e7c54ada3a0e90bbc09fc0437b8684e51037c20b2a0348ed6a"


def once(src: str, old: str, new: str, label: str) -> str:
    count = src.count(old)
    if count != 1:
        raise ValueError(f"{label}: expected one anchor, found {count}")
    return src.replace(old, new, 1)


def add_p3(src: str, on: bool) -> str:
    src = once(src, "_K4_PERNEURON = True",
               f"# P3 distinct-index feed, rank 32, represented-birth compensation.\n"
               f"_P3_ON = {on!r}\n_P3_RANK = 32\n_P3_COMPENSATE = True\n\n"
               "_K4_PERNEURON = True", "flags")
    helpers = p3._P3_FUNCS.replace("    AM = tw.A_L * tw.M", "    A_L, M = _full_legs_lean(tw)\n    AM = A_L * M")
    helpers = helpers.replace("n, k = tw.M.shape[0], tw.k", "n, k = tw.M.shape[0], tw.k")
    src = once(src, "def _dslice3(tw):", helpers.lstrip("\n") + "def _dslice3(tw):", "helpers")
    src = once(src, '"_qU", "_qold")',
               '"_qU", "_qold", "F", "R3", "R4", "w3b", "_p3")', "slots")
    src = once(src, "        self._qU = self._qold = None",
               "        self._qU = self._qold = None\n"
               "        self.F = self.R3 = self.R4 = self.w3b = None\n"
               "        self._p3 = None", "init")
    src = once(src, "    return _carry_shared(out, tw, None if tw.Q is None else _mm(WT, tw.Q))",
               "    if tw.F is not None:\n"
               "        out.F, out.R3, out.R4, out.w3b = _mm(WT, tw.F), tw.R3, tw.R4, tw.w3b\n"
               "    return _carry_shared(out, tw, None if tw.Q is None else _mm(WT, tw.Q))", "advance")
    src = once(src, "    out._wts = tw._wts\n    return out",
               "    out._wts = tw._wts\n"
               "    if tw.F is not None:\n"
               "        out.F, out.R3, out.R4, out.w3b = tw.F * col, tw.R3, tw.R4, tw.w3b\n"
               "    return out", "wick")
    src = once(src, "    return _carry_shared(out, tw, tw.Q)\n\n\ndef _truncate",
               "    if tw is not None and tw.F is not None:\n"
               "        out.F, out.R3, out.R4, out.w3b = tw.F, tw.R3, tw.R4, tw.w3b\n"
               "    return _carry_shared(out, tw, tw.Q)\n\n\ndef _truncate", "append")
    src = once(src, "        return d3\n    t = (tw.A_D * M * M)",
               "        if tw.F is not None:\n"
               "            d3 = d3 + _p3_parts(tw)[2]\n"
               "        return d3\n    t = (tw.A_D * M * M)", "d3")
    src = once(src, "        acc = _shared_contract_lean(tw, LA, LP, A_L, M)",
               "        if tw.F is not None:\n"
               "            Gp, TT, _ = _p3_parts(tw)\n"
               "            LA = LA + Gp * M\n"
               "            LP = LP + Gp * A_L\n"
               "        acc = _shared_contract_lean(tw, LA, LP, A_L, M)\n"
               "        if tw.F is not None:\n"
               "            acc = acc + _mm(TT, tw.F.T)", "d21")
    src = once(src, "        return out\n    return _Tower(tw.A_D[:, -keep:]",
               "        if tw.F is not None:\n"
               "            r3 = tw.R3.shape[2]\n"
               "            out.F, out.R3, out.R4 = tw.F[:, -(2 * g * r3):], tw.R3[-g:], tw.R4[-g:]\n"
               "            out.w3b = tw.w3b[-keep:]\n"
               "        return out\n    return _Tower(tw.A_D[:, -keep:]", "truncate")
    old_join = "    out.Y, out.CA, out.CB = tw.Y, tw.CA, tw.CB\n    out.Q, out.FA"
    if src.count(old_join) != 2:
        raise ValueError("gram joins: expected two")
    src = src.replace(old_join,
               "    out.Y, out.CA, out.CB = tw.Y, tw.CA, tw.CB\n"
               "    if tw.F is not None:\n"
               "        out.F, out.R3, out.R4, out.w3b = tw.F, tw.R3, tw.R4, tw.w3b\n"
               "    out.Q, out.FA")
    src = once(src, "        e_b = w[(1, 2)] - 2.0 * pK[(1,)] * w1",
               "        if _P3_ON:\n"
               "            w3_p3 = w[(3, 1)]\n"
               "            if adv is None:\n"
               "                _r3 = n if _P3_RANK <= 0 else min(int(_P3_RANK), n)\n"
               "                U3 = fnp.zeros((n, _r3), dtype=dtype)\n"
               "                V3 = U3\n"
               "            else:\n"
               "                U3, V3 = _p3_compress(D21)\n"
               "            F3b = fnp.concatenate([w1[:, None] * V3, w2[:, None] * U3], axis=1)\n"
               "            if _P3_COMPENSATE and adv is not None:\n"
               "                D21c = U3 @ V3.T\n"
               "                S21 = (S21 - ((w1_sq[:, None] * D21c.T) * C11) * w3_p3[None, :]\n"
               "                       - (((w1 * w2)[:, None] * D21c) * C11) * w2[None, :])\n"
               "        e_b = w[(1, 2)] - 2.0 * pK[(1,)] * w1", "birth")
    src = once(src, "    out_tw = _truncate(out_tw, gens, n)",
               "    if _P3_ON and _P2_SEPARABLE:\n"
               "        if adv is None or adv.F is None:\n"
               "            out_tw.F, out_tw.R3, out_tw.R4, out_tw.w3b = F3b, U3[None], V3[None], w3_p3\n"
               "        else:\n"
               "            out_tw.F = fnp.concatenate([adv.F, F3b], axis=1)\n"
               "            out_tw.R3 = fnp.concatenate([adv.R3, U3[None]], axis=0)\n"
               "            out_tw.R4 = fnp.concatenate([adv.R4, V3[None]], axis=0)\n"
               "            out_tw.w3b = fnp.concatenate([adv.w3b, w3_p3])\n"
               "    out_tw = _truncate(out_tw, gens, n)", "birth attach")
    src = once(src, "    return base + thin\n\n\ndef _fit_generations",
               "        if _P3_ON:\n"
               "            r3 = n if _P3_RANK <= 0 else min(int(_P3_RANK), n)\n"
               "            if k:\n"
               "                thin += n * (2 * k * r3) * (2 * n - 1)\n"
               "                thin += 2 * k * n * n * (2 * r3 - 1)\n"
               "                thin += 2 * k * n * r3 * (2 * n - 1)\n"
               "                thin += n * n * (4 * k * r3 - 1) + 20 * k * n * n\n"
               "            if l < depth - 1:\n"
               "                thin += 4 * n * r3 * (2 * n - 1) + 4 * n * r3 * r3\n"
               "                thin += n * n * (2 * r3 - 1) + 14 * n * n\n"
               "    return base + thin\n\n\ndef _fit_generations", "P3 guard")
    return src


def build(on: bool, fitted: bool) -> str:
    raw = BASE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != BASE_SHA:
        raise ValueError("frozen gram/P4 base changed")
    src = add_p3(raw.decode("utf-8"), on)
    if fitted:
        src = own.build(src, theta_fam=.625, s_lam=1.225, t_lam=-.1, s_feed=1.)
    return src


def emit() -> None:
    for label, on, fitted in (("C00", False, False), ("C10", False, True),
                              ("C01", True, False), ("C11", True, True)):
        path = ROOT / f"estimator_k3_v43_complete_{label.lower()}.py"
        if path.exists():
            raise FileExistsError(path)
        src = build(on, fitted)
        path.write_text(src, encoding="utf-8", newline="\n")
        print(label, path.name, hashlib.sha256(path.read_bytes()).hexdigest())


def emit_sentinel() -> None:
    """Matched fitted dense PAIR+Q with P4; P3 and shared storage absent."""
    raw = own.BASE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != own.BASE_SHA256:
        raise ValueError("frozen PAIR+Q base changed")
    src = own.build(raw.decode("utf-8"), theta_fam=.625, s_lam=1.225,
                    t_lam=-.1, s_feed=1.)
    src = p4.build(src, final_var=True, blocksym=True, depth=5,
                   min_block=32, tag="S17-fitted-sentinel")
    path = ROOT / "estimator_k3_v43_fitted_uncompressed_p4d5.py"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(src, encoding="utf-8", newline="\n")
    print("sentinel", path.name, hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    emit_sentinel() if sys.argv[1:] == ["--sentinel"] else emit()
