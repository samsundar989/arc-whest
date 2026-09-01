import os, sys, math
import numpy as np
ROOT = r"C:\Users\samsu\Documents\arc-whest"
sys.path.append(ROOT); sys.path.append(os.path.join(ROOT,"local_sim"))
import p2_carrier_objective as O

n, L = 1024, 16
cl = O.c_L(n, L)
c  = 1.0/math.sqrt(n)
S  = float(O.kappa_L(c,L)) + float(O.kappa_L(-c,L))
Ap = ((1.0 + float(O.kappa_L(-1.0,L))) + 2*(n-1)*float(O.kappa_L(0.0,L)))/(2*n)
C1 = Ap - S/2.0
C0 = S/2.0 - cl
print("ANALYTIC affine-in-1/k decomposition of the CLOSED FORM")
print("  c_L        = %.17g" % cl)
print("  S/2        = %.17g" % (S/2))
print("  C1 (1/k)   = %.17g" % C1)
print("  C0 (const) = %.17g   (C0/C1 = %.3e)" % (C0, C0/C1))
print()
print("  k      closed-form MMD^2        C1/k + C0            rel resid       k*MMD^2")
for k in (1,2,4,6,8,10,12,16,64,256,1024,20000):
    m = O.mmd2_carrier_closed(k, n, depth=L, cl=cl)
    pred = C1/k + C0
    print("  %-6d %-22.10e %-20.10e %-14.2e %-.6e" % (k, m, pred, abs(m-pred)/abs(m), k*m))
