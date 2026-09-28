"""Identity check: _dslice21_young with m = k (all lineages, old via reconstructed legs)
must equal production _dslice21 on real v72 state, at every layer (TRAIN full net 0)."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "tools" / "twin"))
import twin, torch
tw = twin.Twin(HERE / "arm_d21age4.py", raw=True)
mod = tw.mod
w, truth = twin.Data().net("full", 0)
orig = mod._layer
def spy(W, state, table, dtype, const, init_scale, last_layer, gens, layer_index=0, depth=1):
    t = state[2]
    if t is not None and not last_layer:
        adv = mod._advance(W.T, t)
        mod._D21_AGE = 0
        ref = mod._dslice21(adv, const["off_mask"])
        A_L, M = mod._full_legs_lean(adv)
        MP, PP, PPL = mod._p2_recon(adv)
        mod._D21_AGE = adv.k
        got = mod._dslice21_young(adv, const["off_mask"], A_L, M, MP, PP, PPL)
        mod._D21_AGE = 4
        rel = float((got - ref).norm() / ref.norm())
        print(f"layer {layer_index} k={adv.k} ko={adv.ko} rel={rel:.2e}", flush=True)
    return orig(W, state, table, dtype, const, init_scale, last_layer, gens, layer_index, depth)
mod._layer = spy
with torch.no_grad():
    tw.forward(w)
