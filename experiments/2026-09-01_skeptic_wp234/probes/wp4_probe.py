import os, sys, json
import numpy as np
ROOT = r"C:\Users\samsu\Documents\arc-whest"
sys.path.append(ROOT); sys.path.append(os.path.join(ROOT,"local_sim"))
import p2_adjoint_k3 as P
P._ensure_upstream()
from mint_provenance import canonical_mint

LAD = P.EPS_LADDER

def worst_dev(ms, ladder):
    m0 = ms[ladder[0]]
    slope = (ms[ladder[1]] - m0)/ladder[1]
    worst = 0.0; rows=[]
    for e in ladder:
        if e == 0: continue
        pred = m0 + slope*e
        act = ms[e]
        d = np.linalg.norm(act-pred)/ (np.linalg.norm(act-m0) or 1.0)
        rows.append((e, float(np.linalg.norm(act-m0)), float(d)))
        worst = max(worst, d)
    return worst, rows

for (n,L) in ((32,3),(32,6),(32,10)):
    seed = 90000000801
    w,_,_ = canonical_mint(n,L,seed)
    mlp,K_in = P.build(w)
    # A) the entry's own arm: eps applied at EVERY layer (cumulative)
    ms_all = {e: P.means(mlp,K_in,arm="scale:%g"%e)[-1] for e in LAD}
    wa,_ = worst_dev(ms_all, LAD)
    # B) single-source arm: eps applied at ONE layer only, others zeroed
    print(f"=== n={n} L={L} seed={seed}")
    print(f"  A) scale:eps  (entry's arm, eps at EVERY layer)  worst_rel_dev = {wa:.4f}")
    for l in (L-1, L-2, max(0,L//2), 0):
        ms = {}
        for e in LAD:
            ms[e] = P.means(mlp,K_in,arm="scale_at:%d:%g"%(l,e))[-1]
        wb, rows = worst_dev(ms, LAD)
        dn = np.linalg.norm(ms[1.0]-ms[0.0])
        print(f"  B) scale_at:{l}:eps  SINGLE SOURCE            worst_rel_dev = {wb:.4f}   drift@1 = {dn:.3e}")
    sys.stdout.flush()
