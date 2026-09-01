import os, sys, time
ROOT = r"C:\Users\samsu\Documents\arc-whest"
sys.path.append(ROOT); sys.path.append(os.path.join(ROOT,"local_sim"))
import p2_augment as A
A._ensure_upstream()
from mint_provenance import canonical_mint
import torch
torch.set_num_threads(4)
SEED=90000000701
for n,L in ((1024,16),):
    w,_,_ = canonical_mint(n,L,SEED)
    t={}
    for kind in ("BASE","AUGMENT","BASE","AUGMENT","BASE","AUGMENT"):
        t0=time.perf_counter()
        with torch.no_grad():
            A.run_augment(w, k_max=2, kind=kind, factor=False, dtype=torch.float64)
        t.setdefault(kind,[]).append(time.perf_counter()-t0)
    b=sorted(t["BASE"])[1]; a=sorted(t["AUGMENT"])[1]
    print(f"n={n} L={L}  BASE median {b:.3f}s  AUGMENT median {a:.3f}s  WALL ratio {a/b:.4f}")
    print("  all BASE", [round(x,3) for x in t["BASE"]], " all AUG", [round(x,3) for x in t["AUGMENT"]])
