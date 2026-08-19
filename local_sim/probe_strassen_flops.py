"""Does Strassen-Winograd reduce flopscope-COUNTED FLOPs on our DENSE chain?
Measured against the real installed flopscope, not a model."""
import flopscope as flops, flopscope.numpy as fnp

N, W, DEPTH, B = 16384, 256, 32, 2.72e11

def measure(fn, *a, **k):
    with flops.BudgetContext(flop_budget=int(1e14)) as ctx:
        fn(*a, **k)
        return int(ctx.flops_used)

def dense(A, Bm):
    return A @ Bm

def strassen(A, Bm, depth):
    if depth == 0 or A.shape[1] % 2 or Bm.shape[1] % 2:
        return A @ Bm
    m, k = A.shape; n = Bm.shape[1]
    h, kh, nh = m // 2, k // 2, n // 2
    a11, a12 = A[:h, :kh], A[:h, kh:]
    a21, a22 = A[h:, :kh], A[h:, kh:]
    b11, b12 = Bm[:kh, :nh], Bm[:kh, nh:]
    b21, b22 = Bm[kh:, :nh], Bm[kh:, nh:]
    r = lambda x, y: strassen(x, y, depth - 1)
    m1 = r(a11 + a22, b11 + b22); m2 = r(a21 + a22, b11)
    m3 = r(a11, b12 - b22);       m4 = r(a22, b21 - b11)
    m5 = r(a11 + a12, b22);       m6 = r(a21 - a11, b11 + b12)
    m7 = r(a12 - a22, b21 + b22)
    c11 = m1 + m4 - m5 + m7; c12 = m3 + m5
    c21 = m2 + m4;           c22 = m1 - m2 + m3 + m6
    return fnp.concatenate([fnp.concatenate([c11, c12], axis=1),
                            fnp.concatenate([c21, c22], axis=1)], axis=0)

A = fnp.zeros((N, W), dtype=fnp.float32) + fnp.float32(1.0)
Wt = fnp.zeros((W, W), dtype=fnp.float32) + fnp.float32(1.0)

d0 = measure(dense, A, Wt)
print(f"one layer, dense (16384,256)@(256,256): {d0:,} FLOPs   (model m*n*(2k-1) = {N*W*(2*W-1):,})")
print(f"{'strassen depth':>15}{'FLOPs/layer':>18}{'vs dense':>11}{'32-layer C/B':>15}{'score gain':>12}")
print(f"{0:>15}{d0:>18,}{1.0:>10.3f}x{d0*DEPTH/B:>15.4f}{1.0:>11.3f}x")
for dep in (1, 2, 3, 4):
    try:
        c = measure(strassen, A, Wt, dep)
        print(f"{dep:>15}{c:>18,}{d0/c:>10.3f}x{c*DEPTH/B:>15.4f}{d0/c:>11.3f}x")
    except Exception as e:
        print(f"{dep:>15}  FAILED: {type(e).__name__}: {e}")
