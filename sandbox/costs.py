"""Q3(iv): matmul / GF counts for n=1024, L=16 schemes."""
import numpy as np
n, L = 1024, 16
UNIT = 2*n**3/1e9          # GF per n^3-class matmul (2n^3 flops)
STR7 = (7/8)**7

def counts():
    rows = []
    nb = L-1                       # births h_1..h_{L-1}
    k2 = 1 + 2*(L-1)               # C_1 = W1 W1^T, then W Cov W^T per layer
    k2_syrk = 0.5 + (L-1)*(1/6 + 0.5 + 0.5)   # chol + trmm + syrk
    legs_all = sum(L-1-l for l in range(1, L))          # forward diag legs, all ages = 105
    rows.append(('exact K2 chain (gemm)', k2))
    rows.append(('exact K2 chain (chol+trmm+syrk)', k2_syrk))
    rows.append(('forward lineages, diag legs only, all ages (direct+indirect mean)', legs_all))
    rows.append(('forward lineages + pair & hub readout, all ages', legs_all + 3*nb))
    rows.append(('forward lineages + pair matrices every layer (indirect cov, diag-born only)', legs_all + 3*nb + legs_all))
    bwd = (L-2) + 3*nb             # V recursion + 3 contractions per birth
    rows.append(('backward adjoint, direct: diag+pair+hub', bwd))
    rows.append(('backward adjoint, direct: diag+pair (no hub)', (L-2) + 1*nb))
    rows.append(('backward adjoint, direct: + order-3 stars (2 more matmuls/birth)', bwd + 2*nb))
    for a in range(1, 8):
        young = sum(min(a, L-1-l) for l in range(1, L))   # legs through <= a layers per birth
        rows.append((f'hybrid(a={a}): adjoint direct + young legs (indirect mean, age<={a})', bwd + young))
        rows.append((f'hybrid(a={a}): + pair matrices for indirect cov (diag-born) age<={a}', bwd + 2*young))
    return rows

if __name__ == '__main__':
    print(f"unit = {UNIT:.3f} GF (d=0), {UNIT*STR7:.3f} GF (Strassen d=7); floor 220 GF = {220/UNIT:.0f} units (d=0) / {220/UNIT/STR7:.0f} units (d=7)")
    print(f"{'scheme':80s} {'units':>7s} {'GF d=0':>8s} {'GF d=7':>8s}")
    for name, u in counts():
        print(f"{name:80s} {u:7.1f} {u*UNIT:8.1f} {u*UNIT*STR7:8.1f}")
