import sys, os
sys.path.insert(0, os.path.expanduser("~/scratch"))
from variants2 import *

def exec_patch_func(mod, fname, nextname, repl):
    src = open(mod.__file__).read()
    start = src.index("def %s(" % fname)
    end = src.index(nextname, start + 10)
    cache = mod.__dict__.setdefault("_PATCHED_SRC", {})
    body = cache.get(fname, src[start:end])
    for a, b in repl:
        assert a in body, (fname, a[:60])
        body = body.replace(a, b)
    cache[fname] = body
    exec(compile("\n" * src[:start].count("\n") + body, mod.__file__, "exec"), mod.__dict__)

def last_layer_rows(K16):
    """V6: at the LAST layer, transport/rebuild only the K16 largest-alpha output rows (D3/D4star of the rest = 0)."""
    def p(mod):
        mod._K16 = K16
        exec_patch_func(mod, "_layer", "class Estimator(BaseEstimator):", [
            ("    adv = None if tw is None else _advance(WT, tw)\n    tw = None\n",
             "    _tw_last = tw if last_layer else None\n    adv = None if (tw is None or last_layer) else _advance(WT, tw)\n    tw = None\n"),
            ("    w = _wick(mu_pre, sigma, alpha, table, dtype, const[\"wick\"])\n",
             "    w = _wick(mu_pre, sigma, alpha, table, dtype, const[\"wick\"])\n"
             "    _inv16 = None\n"
             "    if last_layer and _tw_last is not None:\n"
             "        _p16 = fnp.argsort(alpha)\n"
             "        _sel16 = _p16[n - _K16:]\n"
             "        _r16 = fnp.argsort(_p16)\n"
             "        _inv16 = fnp.where(_r16 >= n - _K16, _r16 - (n - _K16), _K16)\n"
             "        adv = _advance(fnp.take(WT, _sel16, axis=0), _tw_last)\n"
             "        _tw_last = None\n"),
            ("    D3 = _dslice3(adv) if adv is not None else fnp.zeros(n, dtype=dtype)\n",
             "    D3 = _dslice3(adv) if adv is not None else fnp.zeros(n, dtype=dtype)\n"
             "    if _inv16 is not None:\n"
             "        D3 = fnp.take(fnp.pad(D3, ((0, 1),)), _inv16)\n"),
            ("        D4 = D4 + (((_AL * _AL) * (_AL * _ML)) @ _w3c) * (4.0 * _STAR_GAIN[layer_index])\n",
             "        _d4s = (((_AL * _AL) * (_AL * _ML)) @ _w3c) * (4.0 * _STAR_GAIN[layer_index])\n"
             "        if _inv16 is not None:\n"
             "            _d4s = fnp.take(fnp.pad(_d4s, ((0, 1),)), _inv16)\n"
             "        D4 = D4 + _d4s\n"),
        ])
    return p

def old_rebuild_kept_rows(mod):
    """V7: old-tier legs rebuilt on the D21-kept rows only (others zero), scattered back by pad+take."""
    exec_patch_func(mod, "_full_legs_lean", "def _shared_contract_lean", [
        ("        F = _mm(tw.Q, fnp.concatenate([tw.FA, tw.FM], axis=1))\n",
         "        if tw.kd is not None:\n"
         "            F = _mm(fnp.take(tw.Q, tw.kd[0], axis=0), fnp.concatenate([tw.FA, tw.FM], axis=1))\n"
         "            F = fnp.take(fnp.pad(F, ((0, 1), (0, 0))), tw.kd[1], axis=0)\n"
         "        else:\n"
         "            F = _mm(tw.Q, fnp.concatenate([tw.FA, tw.FM], axis=1))\n")])

if __name__ == "__main__":
    run([last_layer_rows(800)], "V6a last-layer rows K16=800 (tau2.5)")
    run([last_layer_rows(736)], "V6b last-layer rows K16=736 (tau2.0)")
    run([old_rebuild_kept_rows], "V7 old rebuild on kd rows")
