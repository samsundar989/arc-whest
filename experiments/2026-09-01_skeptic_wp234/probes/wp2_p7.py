import os, sys, math, time
import numpy as np
ROOT = r"C:\Users\samsu\Documents\arc-whest"
sys.path.append(ROOT); sys.path.append(os.path.join(ROOT,"local_sim"))
import p2_carrier_objective as O
import kerdock_mub as REF

L = O.DEFAULT_DEPTH
print("depth", L)
for m in (5, 7, 9):
    n = 2 ** (m + 1)
    k = 4
    rows = 2 * k * n
    cl = O.c_L(n, L)
    car = O.mmd2_carrier_closed(k, n, depth=L, cl=cl)
    # sanity: general path on the actual carrier rows
    U = REF.carrier_float32(m, k, antipodes=True).astype(np.float64)
    car_gen = O.mmd2(U, depth=L, cl=cl)
    vals = []
    t0=time.time()
    NDRAW = 16
    for s in range(7000, 7000+NDRAW):          # FRESH seed family, not the suite's
        C = O.control_random_orthogonal(rows, n, s)
        vals.append(O.mmd2(C, depth=L, cl=cl))
    v = np.array(vals)
    sd = v.std(ddof=1)
    gap = v.mean() - car
    print(f"m={m} n={n} k={k} rows={rows}  carrier(closed)={car:.6e} carrier(gen)={car_gen:.6e}")
    print(f"    random_orthogonal over {NDRAW} FRESH draws: mean={v.mean():.6e} sd={sd:.3e} min={v.min():.6e} max={v.max():.6e}")
    print(f"    ratio mean/carrier = {v.mean()/car:.5f}   gap={gap:.4e}  gap/sd = {gap/sd:.1f}  [{time.time()-t0:.1f}s]")
    sys.stdout.flush()
