"""WP4 re-test with the propagator INTACT.

p2_adjoint_k3.run_arm's `scale_at:l:eps` sets factor=0.0 at every layer != l,
which deletes the kappa_3 TRANSPORT paths that J_(L<-l) in the contract's own
object `delta m_L ~= sum_l J_(L<-l) S_l` is made of.  Here the only change is
   factor = eps if l == at_layer else 1.0
i.e. leave every other layer's kappa_3 alone.  Nothing else differs; this file
imports upstream through the same bridge and never writes to the repository.
"""
import os, sys
import numpy as np
ROOT = r"C:\Users\samsu\Documents\arc-whest"
sys.path.append(ROOT); sys.path.append(os.path.join(ROOT,"local_sim"))
import p2_adjoint_k3 as P
import p2_kprop_bridge as BRIDGE
P._ensure_upstream()
from mint_provenance import canonical_mint

def run_bg(mlp, K_in, at_layer=None, eps=1.0, k_max=3):
    from mlp_kprop.kprop_harmonic import (SIMPLE, clone_tower, coerce_input,
                                          linear_kprop, nonlin_kprop)
    from mlp_kprop.wick import WICK_COEF_D
    nonlin_wick=[WICK_COEF_D[nm] for nm in mlp.nonlin_names]
    L=len(mlp.nonlins)
    K=coerce_input(K_in,k_max=k_max,kind=SIMPLE); out={}
    for l,Wm in enumerate(mlp.Ws):
        if l>=L: break
        K=linear_kprop(K,Wm.weight,k_max=k_max,set_metric=mlp.init_scale[l],bias=Wm.bias)
        K=nonlin_kprop(K,nonlin_wick_coef=nonlin_wick[l],k_max=k_max,kind=SIMPLE,
                       use_pK=True,factor=False)
        if 3 in K:
            f = eps if (at_layer is None or l==at_layer) else 1.0   # <-- ONLY CHANGE
            if f != 1.0:
                K=dict(K); K[3]=P._scale_k3(K[3],f)
        out["act%d"%l]=clone_tower(K,d_max=1)
    return BRIDGE._degree1(out["act%d"%(L-1)][1])

LAD=P.EPS_LADDER
for n,L in ((32,6),(32,10)):
  for seed in (90000000801, 90000000802):
    w,_,_=canonical_mint(n,L,seed); mlp,K_in=P.build(w)
    m_none = run_bg(mlp,K_in,None,0.0)        # all kappa_3 memory off
    m_full = run_bg(mlp,K_in,None,1.0)        # untouched flow
    total  = m_full-m_none
    # (1) LINEARITY of one layer's source ON A FULL BACKGROUND
    worst_all=[]
    for l in range(L):
        ms={e: run_bg(mlp,K_in,l,e) for e in LAD}
        m0=ms[0.0]; slope=(ms[LAD[1]]-m0)/LAD[1]
        w_=max(np.linalg.norm(ms[e]-(m0+slope*e))/(np.linalg.norm(ms[e]-m0) or 1.0)
               for e in LAD if e>0)
        worst_all.append(w_)
    # (2) ADDITIVITY with the propagator intact (leave-one-out responses)
    S=[m_full-run_bg(mlp,K_in,l,0.0) for l in range(L)]
    summed=np.sum(S,axis=0)
    exp=1.0-float((total-summed)@(total-summed))/float(total@total)
    cos=float(summed@total/(np.linalg.norm(summed)*np.linalg.norm(total)))
    print(f"n={n} L={L} seed={seed}")
    print(f"   Q3 (single source, FULL background): worst dev over layers = {max(worst_all):.4f}"
          f"  [per-layer {' '.join('%.3f'%x for x in worst_all)}]")
    print(f"   Q4 (leave-one-out, transport intact): explained = {exp:.4f}   cosine = {cos:.4f}")
    sys.stdout.flush()
