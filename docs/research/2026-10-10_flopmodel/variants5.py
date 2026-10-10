import sys, os, json
sys.path.insert(0, os.path.expanduser("~/scratch"))
from variants4 import *

def two_sided_D(mod):
    """V8: D21 rows AND columns restricted to the K_D kept neurons; the transport then produces
    only those K_D output rows (W^T[kd, rows]); D3/star of dead rows = 0; R rows = free slice."""
    exec_patch_func(mod, "_layer", "class Estimator(BaseEstimator):", [
        ("    adv = None if tw is None else _advance(WT, tw)\n    tw = None\n",
         "    _tw_in = tw\n    adv = None\n    tw = None\n"),
        ("        if _KD < n and adv is not None:\n"
         "            _rank = fnp.argsort(_perm)\n"
         "            adv.kd = (_perm[n - _KD:],\n"
         "                      fnp.where(_rank >= n - _KD, _rank - (n - _KD), _KD))\n"
         "            _kz = fnp.where(_rank >= n - _KD, dtype(1.0), dtype(0.0))\n",
         "        if _KD < n and _tw_in is not None:\n"
         "            _rank = fnp.argsort(_perm)\n"
         "            _kdi = _perm[n - _KD:]\n"
         "            _sc = fnp.where(_rank >= n - _KD, _rank - (n - _KD), _KD)\n"
         "            _kz = fnp.where(_rank >= n - _KD, dtype(1.0), dtype(0.0))\n"
         "            adv = _advance(fnp.take(WT, _kdi, axis=0), _tw_in)\n"
         "            adv.kd = (None, _sc)\n"
         "            _two = (_kdi, _sc, _KD, _KR)\n"
         "    if adv is None and _tw_in is not None:\n"
         "        adv = _advance(WT, _tw_in)\n"
         "    _tw_in = None\n"),
        ("    _kr = _kd = _kz = _mh = _kh = None\n",
         "    _kr = _kd = _kz = _mh = _kh = None\n    _two = None\n"),
        ("    D3 = _dslice3(adv) if adv is not None else fnp.zeros(n, dtype=dtype)\n",
         "    D3 = _dslice3(adv) if adv is not None else fnp.zeros(n, dtype=dtype)\n"
         "    if _two is not None:\n"
         "        D3 = fnp.take(fnp.pad(D3, ((0, 1),)), _two[1])\n"),
        ("        D4 = D4 + (((_AL * _AL) * (_AL * _ML)) @ _w3c) * (4.0 * _STAR_GAIN[layer_index])\n",
         "        _d4s = (((_AL * _AL) * (_AL * _ML)) @ _w3c) * (4.0 * _STAR_GAIN[layer_index])\n"
         "        if _two is not None:\n"
         "            _d4s = fnp.take(fnp.pad(_d4s, ((0, 1),)), _two[1])\n"
         "        D4 = D4 + _d4s\n"),
        ("        adv = _wick_tower(w1t[:, None], adv)\n"
         "        if _kr is not None:\n"
         "            adv = _kas_rows(adv, _kr)\n",
         "        if _two is not None:\n"
         "            adv = _wick_tower(fnp.take(w1t, _two[0])[:, None], adv)\n"
         "            adv = _kas_rows_slice(adv, _two[2] - _two[3], _kr)\n"
         "        else:\n"
         "            adv = _wick_tower(w1t[:, None], adv)\n"
         "            if _kr is not None:\n"
         "                adv = _kas_rows(adv, _kr)\n"),
    ])
    exec_patch_func(mod, "_dslice21", "def _append(", [
        ("            acc = fnp.take(fnp.pad(acc, ((0, 1), (0, 0))), kd[1], axis=0)\n",
         "            if kd[0] is None:\n"
         "                acc = fnp.take(fnp.take(fnp.pad(acc, ((0, 1), (0, 1))), kd[1], axis=0), kd[1], axis=1)\n"
         "            else:\n"
         "                acc = fnp.take(fnp.pad(acc, ((0, 1), (0, 0))), kd[1], axis=0)\n")])
    def _kas_rows_slice(tw, start, idx):
        sl = slice(start, None)
        out = mod._Tower(None, tw.A_L[sl], None if tw.M is None else tw.M[sl], tw.w2b, tw.k,
                         None if tw.Z is None else tw.Z[sl], tw.L, tw.sb, tw.eb)
        if tw.Y is not None:
            out.Y, out.CA, out.CB = tw.Y[sl], tw.CA, tw.CB
        mod._carry_shared(out, tw, None if tw.Q is None else tw.Q[sl])
        out._wts = tw._wts
        out.rows = idx
        return out
    mod._kas_rows_slice = _kas_rows_slice

if __name__ == "__main__":
    run([two_sided_D], "V8 two-sided D (cols too) + transport on kd rows")
    pass
    run([two_sided_D, set_kas("D", T["3.0"]["keep"])], "V8 at D=keep3.0 (safer)")
