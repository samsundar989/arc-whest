import os, sys
ROOT = r"C:\Users\samsu\Documents\arc-whest"
sys.path.append(ROOT); sys.path.append(os.path.join(ROOT,"local_sim"))
import p2_augment as A
A._ensure_upstream()
from mint_provenance import canonical_mint
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from collections import Counter

class OpCount(TorchDispatchMode):
    def __init__(self): self.c = Counter()
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.c[str(func)] += 1
        return func(*args, **(kwargs or {}))

SEED=90000000701
for n,L in ((1024,16),):
    w,_,_ = canonical_mint(n,L,SEED)
    tot={}
    for kind in ("BASE","AUGMENT"):
        m=OpCount()
        with torch.no_grad(), m:
            A.run_augment(w, k_max=2, kind=kind, factor=False, dtype=torch.float64)
        tot[kind]=m.c
        print(f"{kind}: {sum(m.c.values())} aten op invocations, {len(m.c)} distinct")
    print(f"  OP-COUNT ratio AUGMENT/BASE = {sum(tot['AUGMENT'].values())/sum(tot['BASE'].values()):.4f}")
    print("  ops present in AUGMENT but NOT in BASE (top 15 by count):")
    for k,v in sorted(((k,v) for k,v in tot['AUGMENT'].items() if k not in tot['BASE']), key=lambda x:-x[1])[:15]:
        print(f"     {v:>7}  {k}")
    print("  biggest count increases:")
    d = sorted(((k, tot['AUGMENT'][k]-tot['BASE'].get(k,0)) for k in tot['AUGMENT']), key=lambda x:-x[1])[:12]
    for k,v in d: print(f"     +{v:<7} {k}   (base {tot['BASE'].get(k,0)} -> aug {tot['AUGMENT'][k]})")
