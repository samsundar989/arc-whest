"""Atom k1 (Lane K, HANDOFF 2026-10-12 Lane K, K1): D21 rows = kink-active, gain-filled elsewhere.

Not bit-identical. Requires `k0` FIRST (`--atoms k0+k1:3.0`): it reuses k0's in-chain v and fill.
Twin arm: `mask+dress+bank+fold+kgain --kg rows=<ktab_r..>,rank=...,rhok=<rho_k1_..>`.

Rows of the D21 readout: the K_r SMALLEST scores (abs: alpha^2; asym a:b: max(-alpha/a, alpha/b)),
K_r from ROWS below (FIT-100 q50, nearest 32). Columns stay the KAS D set (top K_D by
alpha). The readout forms only the K_r x K_D block (left operands gathered on rows, right operands
on columns) and scatters both axes by a gather from the zero-padded block. Every row outside the
kink set gets k0's gain formula on the D columns; the corrector delta stays on for every row.
`_D21E_RHO` is replaced by the kink-row-set calibration (RHO below: twin --kg stats over FIT-100,
the compute_rho.py recipe; experiments/2026-10-13_k/rho_k1_*.json).

Arms (`make`): '3.0' (|alpha| <= 3.0), '2.5_3.0' (-2.5 <= alpha <= 3.0), '2.5' (|alpha| <= 2.5).
Exact prices (x3_tables S1 on v84, D rows only): -2.47 % / -4.21 % / -5.88 % F.
"""
from tools.atoms import rep

ARMS = {"3.0": ("r30", ("abs",)), "2.5_3.0": ("r2530", ("asym", 2.5, 3.0)), "2.5": ("r25", ("abs",))}
#: Baked from experiments/2026-10-13_k/ktab_r*.json (FIT-100 q50, nearest 32) and rho_k1_r*.json (FIT-100,
#: kink-row d21e calibration), so the atom is self-contained.
ROWS = {"r30": [1024, 1024, 1024, 1024, 992, 960, 928, 896, 864, 832, 768, 736, 704, 672, 672, 640],
        "r2530": [1024, 1024, 1024, 992, 960, 928, 896, 832, 800, 768, 736, 704, 672, 640, 608, 576],
        "r25": [1024, 1024, 1024, 992, 928, 896, 832, 800, 768, 736, 704, 640, 608, 576, 576, 544]}
RHO = {"r30": [1.0, 1.0, 1.0, 1.119693, 1.128062, 1.235029, 1.330911, 1.340714, 1.431008, 1.517653, 1.516231,
               1.600863, 1.603809, 1.663087, 1.678437, 1.0],
       "r2530": [1.0, 1.0, 1.0, 1.119845, 1.128175, 1.235213, 1.331151, 1.341012, 1.431486, 1.518197, 1.516411,
                 1.601132, 1.604033, 1.663356, 1.679038, 1.0],
       "r25": [1.0, 1.0, 1.0, 1.11982, 1.12823, 1.235294, 1.331374, 1.341067, 1.431506, 1.518142, 1.516327,
               1.601306, 1.604, 1.663418, 1.678725, 1.0]}


def _patch(src, rows, rank, rho):
    src = rep(src,
              "_KAS_K = {\"R\": ",
              "#: Lane K k1: D21 lineage rows = the K smallest kink scores (rank %r); rows outside get k0's fill.\n"
              "_K1_ROWS = %r\n_K1_RANK = %r\n" % (rank, rows, rank)
              + "_KAS_K = {\"R\": ")
    if rho is not None:
        i = src.index("_D21E_RHO = [")
        j = src.index("\n", i)
        src = src[:i] + "_D21E_RHO = %r  # Lane K k1: kink-row-set calibration" % (rho,) + src[j:]
    # ---- selection: separate row (kink score) and column (alpha) index sets ----
    src = rep(src,
              "        if _KD < n and adv is not None:\n"
              "            _rank = fnp.argsort(_perm)\n"
              "            adv.kd = (_perm[n - _KD:],\n"
              "                      fnp.where(_rank >= n - _KD, _rank - (n - _KD), _KD))\n"
              "            _kz = fnp.where(_rank >= n - _KD, dtype(1.0), dtype(0.0))\n",
              "        _K1R = int(_K1_ROWS[layer_index]) if layer_index < len(_K1_ROWS) else n\n"
              "        _k1r = None\n"
              "        if (_KD < n or _K1R < n) and adv is not None:\n"
              "            _rank = fnp.argsort(_perm)\n"
              "            _kci = _perm[n - _KD:]\n"
              "            _kcv = fnp.where(_rank >= n - _KD, _rank - (n - _KD), _KD)\n"
              "            _kz = fnp.where(_rank >= n - _KD, dtype(1.0), dtype(0.0))\n"
              "            if _K1R < n:\n"
              "                # Lane K k1: the formed rows are the K1R smallest kink scores\n"
              "                _sc = (alpha * alpha if _K1_RANK[0] == 'abs' else\n"
              "                       fnp.maximum(alpha * dtype(-1.0 / _K1_RANK[1]),\n"
              "                                   alpha * dtype(1.0 / _K1_RANK[2])))\n"
              "                _pa = fnp.argsort(_sc)\n"
              "                _rka = fnp.argsort(_pa)\n"
              "                _kri = _pa[:_K1R]\n"
              "                _krv = fnp.where(_rka < _K1R, _rka, _K1R)\n"
              "                _k1r = fnp.where(_rka < _K1R, dtype(1.0), dtype(0.0))\n"
              "            else:\n"
              "                _kri, _krv, _k1r = _kci, _kcv, _kz\n"
              "            adv.kd = (_kri, _krv, _kci, _kcv)\n")
    # ---- readout: left operands on kd[0] rows, right operands on kd[2] rows ----
    src = rep(src,
              "        kd = tw.kd\n"
              "        kr = None if kd is None else kd[0]\n"
              "        acc = _shared_contract_lean(tw, LA, LP, A_L, M, kr)\n",
              "        kd = tw.kd\n"
              "        kr = None if kd is None else kd[0]\n"
              "        kc = None if kd is None else kd[2]\n"
              "        acc = _shared_contract_lean(tw, LA, LP, A_L, M, kr, kc)\n")
    src = rep(src,
              "                            (tw.F if kr is None else fnp.take(tw.F, kr, axis=0)).T)\n",
              "                            (tw.F if kr is None else fnp.take(tw.F, kc, axis=0)).T)\n")
    src = rep(src,
              "                            (tw.Y if kr is None else fnp.take(tw.Y, kr, axis=0)).T)\n",
              "                            (tw.Y if kr is None else fnp.take(tw.Y, kc, axis=0)).T)\n")
    src = rep(src,
              "        acc = acc + _mm(PPL, (tw.Z if kr is None else fnp.take(tw.Z, kr, axis=0)).T) * (1.0 / 3.0)\n",
              "        acc = acc + _mm(PPL, (tw.Z if kr is None else fnp.take(tw.Z, kc, axis=0)).T) * (1.0 / 3.0)\n")
    src = rep(src,
              "            acc = fnp.take(fnp.take(fnp.pad(acc, ((0, 1), (0, 1))), kd[1], axis=0),\n"
              "                           kd[1], axis=1)\n",
              "            acc = fnp.take(fnp.take(fnp.pad(acc, ((0, 1), (0, 1))), kd[1], axis=0),\n"
              "                           kd[3], axis=1)\n")
    src = rep(src, "def _shared_contract_lean(tw, LA, LP, A_L, M, kr=None):\n",
              "def _shared_contract_lean(tw, LA, LP, A_L, M, kr=None, kc=None):\n")
    src = rep(src, "        _R = fnp.take(_R, kr, axis=0)\n", "        _R = fnp.take(_R, kc, axis=0)\n")
    src = rep(src,
              "    old = _mm(Xold, (tw.Q if kr is None else fnp.take(tw.Q, kr, axis=0)).T)\n",
              "    old = _mm(Xold, (tw.Q if kr is None else fnp.take(tw.Q, kc, axis=0)).T)\n")
    # ---- k0's fill now covers every non-kink row ----
    src = rep(src,
              "            D21 = D21 + ((_k0G * off_mask) * (1.0 - _kz)[:, None]) * _kz[None, :]\n",
              "            D21 = D21 + ((_k0G * off_mask) * (1.0 - _k1r)[:, None]) * _kz[None, :]\n")
    src = rep(src, "    _kr = _kd = _kz = _mh = _kh = None\n", "    _kr = _kd = _kz = _mh = _kh = _k1r = None\n")
    return src


def make(arg):
    if arg == "off":
        # the k1 plumbing with every row formed on the D set: numerically k0 (base for k2 without K1)
        return lambda src, z: _patch(src, [1024] * 16, ("abs",), None)
    if arg not in ARMS:
        raise SystemExit(f"atom k1: unknown arm {arg!r} ({' | '.join(ARMS)})")
    tab, rank = ARMS[arg]
    return lambda src, z: _patch(src, ROWS[tab], rank, RHO[tab])


def apply(src, z):
    raise SystemExit("atom k1 needs an arm: k1:3.0 | k1:2.5_3.0 | k1:2.5")
