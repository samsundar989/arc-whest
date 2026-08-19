"""The residual risk: Strassen trades counted FLOPs for OPERATION COUNT, and
flopscope bills operations as residual wall time at lambda = 1e11.
Our v2.4.1 runs ~1416 ops/MLP at 9.845 ms graded residual => ~7 us/op equivalent.
Count ops exactly, and price the trade."""
import flopscope as flops, flopscope.numpy as fnp

N, W, DEPTH, B, LAM = 16384, 192, 32, 2.72e11, 1e11
SHIPPED_OPS, SHIPPED_RESID_MS = 1416.0, 9.845     # graded, submission 327743

def run(fn):
    with flops.BudgetContext(flop_budget=int(1e14)) as ctx:
        fn(); return int(ctx.flops_used), len(ctx.op_log)

def strassen(A, Bm, d):
    if d == 0 or A.shape[0] % 2 or A.shape[1] % 2 or Bm.shape[1] % 2:
        return A @ Bm
    m,k = A.shape; n = Bm.shape[1]; h,kh,nh = m//2,k//2,n//2
    a11,a12,a21,a22 = A[:h,:kh],A[:h,kh:],A[h:,:kh],A[h:,kh:]
    b11,b12,b21,b22 = Bm[:kh,:nh],Bm[:kh,nh:],Bm[kh:,:nh],Bm[kh:,nh:]
    r = lambda x,y: strassen(x,y,d-1)
    m1=r(a11+a22,b11+b22); m2=r(a21+a22,b11); m3=r(a11,b12-b22); m4=r(a22,b21-b11)
    m5=r(a11+a12,b22); m6=r(a21-a11,b11+b12); m7=r(a12-a22,b21+b22)
    return fnp.concatenate([fnp.concatenate([m1+m4-m5+m7,m3+m5],axis=1),
                            fnp.concatenate([m2+m4,m1-m2+m3+m6],axis=1)],axis=0)

def chain(d):
    A = fnp.zeros((N, W), dtype=fnp.float32) + fnp.float32(1.0)
    Ws = [fnp.zeros((W, W), dtype=fnp.float32) + fnp.float32(1.0) for _ in range(DEPTH)]
    def go():
        h = A
        for w in Ws:
            h = fnp.maximum(strassen(h, w, d), fnp.float32(0.0))
        return h.mean(axis=0, dtype=fnp.float64)
    return run(go)

f0, o0 = chain(0)
print(f"32-layer chain, width {W}, N={N}\n")
print(f"{'depth':>6}{'counted FLOPs':>18}{'ops':>8}{'FLOP saving':>15}{'extra ops':>11}"
      f"{'us/op break-even':>19}{'net @7us/op':>13}")
for d in (0, 1, 2, 3):
    f, o = chain(d)
    save = f0 - f
    dops = o - o0
    be = (save / LAM) * 1e6 / dops if dops else float('inf')     # us per op to break even
    net = f0 / (f + dops * 7e-6 * LAM)                            # at 7 us/op
    print(f"{d:>6}{f:>18,}{o:>8}{save:>15,}{dops:>11,}{be:>18.1f}{net:>12.3f}x")

print(f"\nreference: shipped v2.4.1 = {SHIPPED_OPS:.0f} ops at {SHIPPED_RESID_MS} ms graded")
print(f"           => {SHIPPED_RESID_MS*1000/SHIPPED_OPS:.1f} us/op, i.e. "
      f"{SHIPPED_RESID_MS*1e-3*LAM/SHIPPED_OPS:,.0f} FLOP-equivalents per op")
