import sys, os
sys.path.insert(0, os.path.expanduser("~/scratch"))
from variants6 import *
def v8_plus_v6(K16):
    def p(mod):
        mod._K16 = K16
        two_sided_D(mod)
        exec_patch_func(mod, "_layer", "class Estimator(BaseEstimator):", [
            ("    if adv is None and _tw_in is not None:\n        adv = _advance(WT, _tw_in)\n",
             "    _inv16 = None\n"
             "    if adv is None and _tw_in is not None:\n"
             "        if last_layer:\n"
             "            _p16 = fnp.argsort(alpha)\n"
             "            _sel16 = _p16[n - _K16:]\n"
             "            _r16 = fnp.argsort(_p16)\n"
             "            _inv16 = fnp.where(_r16 >= n - _K16, _r16 - (n - _K16), _K16)\n"
             "            adv = _advance(fnp.take(WT, _sel16, axis=0), _tw_in)\n"
             "        else:\n"
             "            adv = _advance(WT, _tw_in)\n"),
            ("    if _two is not None:\n        D3 = fnp.take(fnp.pad(D3, ((0, 1),)), _two[1])\n",
             "    if _two is not None:\n        D3 = fnp.take(fnp.pad(D3, ((0, 1),)), _two[1])\n"
             "    if _inv16 is not None:\n        D3 = fnp.take(fnp.pad(D3, ((0, 1),)), _inv16)\n"),
            ("        if _two is not None:\n            _d4s = fnp.take(fnp.pad(_d4s, ((0, 1),)), _two[1])\n",
             "        if _two is not None:\n            _d4s = fnp.take(fnp.pad(_d4s, ((0, 1),)), _two[1])\n"
             "        if _inv16 is not None:\n            _d4s = fnp.take(fnp.pad(_d4s, ((0, 1),)), _inv16)\n"),
        ])
    return p
if __name__ == "__main__":
    C1s = [p_block_sandwich, lambda m: fused_advance(m), join_mm, lambda m: skip_join_at(m, {13, 14}), sandwich_split2]
    run([v8_plus_v6(800)], "V8 + V6 (no engineering)")
    run(C1s + [v8_plus_v6(800)], "C1s + V8 + V6")
    run(C1s + [set_depth(6, 20, 12), v8_plus_v6(800)], "C2 + V8 + V6")
    run(C1s + [two_sided_D, set_kas("D", T["3.0"]["keep"]), last_layer_rows(800)] if False else C1s + [v8_plus_v6(800), set_kas("D", T["3.0"]["keep"])], "C1s + V8(D=keep3.0) + V6")
