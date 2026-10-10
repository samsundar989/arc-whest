import sys, os
sys.path.insert(0, os.path.expanduser("~/scratch"))
from variants8 import *
def cols_only_D(mod):
    """V8b: D21 COLUMNS restricted to the kd set too (right operand rows gathered), transport unchanged
    (so D3 etc. stay full); output K_D x K_D scattered on both axes."""
    exec_patch_func(mod, "_shared_contract_lean", "def _shared_join_lean", [
        ("    rec = _mm(fnp.concatenate([LA[:, c:], LP[:, c:]], axis=1),\n"
         "              fnp.concatenate([tw.A_L if tw.ko else A_L, tw.M if tw.ko else M], axis=1).T)\n",
         "    _R = fnp.concatenate([tw.A_L if tw.ko else A_L, tw.M if tw.ko else M], axis=1)\n"
         "    if kr is not None:\n"
         "        _R = fnp.take(_R, kr, axis=0)\n"
         "    rec = _mm(fnp.concatenate([LA[:, c:], LP[:, c:]], axis=1), _R.T)\n"),
        ("    old = _mm(Xold, tw.Q.T)\n",
         "    old = _mm(Xold, (tw.Q if kr is None else fnp.take(tw.Q, kr, axis=0)).T)\n")])
    exec_patch_func(mod, "_dslice21", "def _append(", [
        ("            acc = acc + _mm(TT if kr is None else fnp.take(TT, kr, axis=0), tw.F.T)\n",
         "            acc = acc + _mm(TT if kr is None else fnp.take(TT, kr, axis=0), tw.F.T)\n"),
        ("            acc = acc + _mm(RYk * (2.0 / 3.0), tw.Y.T)\n",
         "            acc = acc + _mm(RYk * (2.0 / 3.0), (tw.Y if kr is None else fnp.take(tw.Y, kr, axis=0)).T)\n"),
        ("        acc = acc + _mm(PPL, tw.Z.T) * (1.0 / 3.0)\n",
         "        acc = acc + _mm(PPL, (tw.Z if kr is None else fnp.take(tw.Z, kr, axis=0)).T) * (1.0 / 3.0)\n"),
        ("            acc = fnp.take(fnp.pad(acc, ((0, 1), (0, 0))), kd[1], axis=0)\n",
         "            acc = fnp.take(fnp.take(fnp.pad(acc, ((0, 1), (0, 1))), kd[1], axis=0), kd[1], axis=1)\n")])
if __name__ == "__main__":
    run([cols_only_D], "V8b D21 cols restricted (transport full, D3 full)")
    C1s = [p_block_sandwich, lambda m: fused_advance(m), join_mm, lambda m: skip_join_at(m, {13, 14}), sandwich_split2]
    run(C1s + [cols_only_D], "C1s + V8b")
