"""What is the ACHIEVABLE per-sample cost, measured on real flopscope 0.10.0?
Pruning leaves ~205-wide matrices; Strassen needs even splits.  Measure the
interaction rather than model it."""
import flopscope as flops, flopscope.numpy as fnp
import numpy as np

N, W, DEPTH, B = 16384, 256, 32, 2.72e11
ALIVE = 205                                   # measured mean alive count at tau=-2.5

def meas(fn, *a):
    with flops.BudgetContext(flop_budget=int(1e14)) as ctx:
        fn(*a); return int(ctx.flops_used)

def strassen(A, Bm, d):
    if d == 0 or A.shape[0] % 2 or A.shape[1] % 2 or Bm.shape[1] % 2:
        return A @ Bm
    m, k = A.shape; n = Bm.shape[1]; h, kh, nh = m//2, k//2, n//2
    a11,a12,a21,a22 = A[:h,:kh],A[:h,kh:],A[h:,:kh],A[h:,kh:]
    b11,b12,b21,b22 = Bm[:kh,:nh],Bm[:kh,nh:],Bm[kh:,:nh],Bm[kh:,nh:]
    r = lambda x,y: strassen(x,y,d-1)
    m1=r(a11+a22,b11+b22); m2=r(a21+a22,b11); m3=r(a11,b12-b22); m4=r(a22,b21-b11)
    m5=r(a11+a12,b22); m6=r(a21-a11,b11+b12); m7=r(a12-a22,b21+b22)
    return fnp.concatenate([fnp.concatenate([m1+m4-m5+m7, m3+m5],axis=1),
                            fnp.concatenate([m2+m4, m1-m2+m3+m6],axis=1)],axis=0)

def ones(shape): return fnp.zeros(shape, dtype=fnp.float32) + fnp.float32(1.0)

print(f"{'config':<44}{'FLOPs/layer':>16}{'/sample':>11}{'32L C/B':>10}{'vs ship':>10}")
rows = []
for label, m, k, n, d in [
    ("dense 256 (v1.1 carrier)",          N, 256, 256, 0),
    ("pruned 205 (SHIPPED v2.4.1)",       N, ALIVE, ALIVE, 0),
    ("dense 256 + strassen d2",           N, 256, 256, 2),
    ("dense 256 + strassen d3",           N, 256, 256, 3),
    ("pad-to-256 pruned + strassen d2",   N, 256, 256, 2),
    ("pruned 192 (tau tightened) + d2",   N, 192, 192, 2),
    ("pruned 192 + strassen d1",          N, 192, 192, 1),
    ("pruned 128 (hypothetical) + d2",    N, 128, 128, 2),
]:
    A, Wt = ones((m, k)), ones((k, n))
    try:
        c = meas(lambda: strassen(A, Wt, d))
        rows.append((label, c))
    except Exception as e:
        print(f"{label:<44}  FAILED {type(e).__name__}"); continue
ship = [c for l, c in rows if "SHIPPED" in l][0]
for label, c in rows:
    print(f"{label:<44}{c:>16,}{c/N:>11,.0f}{c*DEPTH/B:>10.4f}{ship/c:>9.3f}x")

print(f"\nshipped per-sample counted (32 layers): {ship*DEPTH/N:,.0f} FLOPs")
print(f"graded per-sample counted (incl. F0):   3,253,000 FLOPs")
