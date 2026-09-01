import os, sys, json
import numpy as np
ROOT = r"C:\Users\samsu\Documents\arc-whest"
sys.path.append(ROOT); sys.path.append(os.path.join(ROOT,"local_sim"))
import p2_augment as A
A._ensure_upstream()
from mint_provenance import canonical_mint
SEED = 90000000701
print(f"{'n':>5} {'L':>3} {'test-metric(all)':>17} {'test-metric(row)':>17} {'relRMS(row)':>12} {'rms(base row)':>14} {'absRMS diff':>12}")
for L in (16,):
    for n in (32,64,128,256,512,1024):
        w,_,_ = canonical_mint(n,L,SEED)
        aug = A.degree1_means(A.run_augment(w, kind="AUGMENT", factor=False), L)
        bas = A.degree1_means(A.run_augment(w, kind="BASE", factor=False), L)
        rel  = np.abs(aug-bas).max()/np.abs(bas).max()
        relf = np.abs(aug[-1]-bas[-1]).max()/np.abs(bas[-1]).max()
        d = aug[-1]-bas[-1]
        rrms = np.sqrt((d**2).mean())
        brms = np.sqrt((bas[-1]**2).mean())
        print(f"{n:>5} {L:>3} {rel:>17.4e} {relf:>17.4e} {rrms/brms:>12.4e} {brms:>14.6f} {rrms:>12.4e}")
        sys.stdout.flush()
