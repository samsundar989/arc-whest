import os, sys, logging
import numpy as np
ROOT = r"C:\Users\samsu\Documents\arc-whest"
sys.path.append(ROOT); sys.path.append(os.path.join(ROOT,"local_sim"))
import p2_augment as A
A._ensure_upstream()
from mint_provenance import canonical_mint
import mlp_kprop.wick as wick
SEED = 90000000701
count = {"n":0}
class Counter(logging.Handler):
    def emit(self, rec):
        if "Snapping" in rec.getMessage(): count["n"] += 1
h = Counter(); wick.logger.addHandler(h); wick.logger.setLevel(logging.WARNING)
wick.logger.propagate = False
for n,L in ((32,16),(64,16),(128,16),(1024,16),(1024,4)):
    w,_,_ = canonical_mint(n,L,SEED)
    for kind in ("BASE","AUGMENT"):
        count["n"]=0
        A.run_augment(w, kind=kind, factor=False)
        print(f"n={n:<5} L={L:<3} {kind:<8} snap warnings = {count['n']}")
