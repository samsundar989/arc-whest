"""Search alternative bases (Karstadt-Schwartz style) for our Strassen-Winograd <U,V,W>.
a' = U[R] a, b' = V[R] b  (the 4 quadrant slots hold the operands of the 4 products in R),
c' = W'^T p with W' = W inv(W[R2]).  Count elementwise passes (each = 1 quadrant numel)."""
import itertools, numpy as np
from fractions import Fraction
U = np.array([[1,0,0,0],[0,1,0,0],[1,1,-1,-1],[0,0,0,1],[0,0,1,1],[-1,0,1,1],[1,0,-1,0]])
V = np.array([[1,0,0,0],[0,0,1,0],[0,0,0,1],[1,-1,-1,1],[-1,1,0,0],[1,-1,0,1],[0,-1,0,1]])
W = np.array([[1,1,1,1],[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,1,0,1],[0,1,1,1],[0,0,1,1]])
# Brent check: sum_s U[s,(i,j)] V[s,(j',k)] W[s,(i',k')] = [j=j'][i=i'][k=k']
T = np.einsum("sa,sb,sc->abc", U, V, W)
ok = True
for i in range(2):
  for j in range(2):
    for j2 in range(2):
      for k in range(2):
        for i2 in range(2):
          for k2 in range(2):
            ok &= T[2*i+j, 2*j2+k, 2*i2+k2] == (j==j2 and i==i2 and k==k2)
print("Winograd Brent equations hold:", ok)
def inv_int(M):
    d = round(np.linalg.det(M))
    if d == 0: return None
    Mi = np.linalg.inv(M)
    Mr = np.round(Mi)
    if np.abs(Mi - Mr).max() > 1e-9: return None
    return Mr.astype(int)
res = []
for R in itertools.combinations(range(7), 4):
    R = list(R); rest = [s for s in range(7) if s not in R]
    iu, iv = inv_int(U[R]), inv_int(V[R])
    if iu is None or iv is None: continue
    Up, Vp = U @ iu, V @ iv
    cin_u = sum(np.count_nonzero(Up[s]) - 1 for s in rest); cin_v = sum(np.count_nonzero(Vp[s]) - 1 for s in rest)
    bu = sum(np.count_nonzero(U[R][j]) - 1 for j in range(4)); bv = sum(np.count_nonzero(V[R][j]) - 1 for j in range(4))
    for R2 in itertools.combinations(range(7), 4):
        R2 = list(R2)
        iw = inv_int(W[R2])
        if iw is None: continue
        Wp = W @ iw
        cout = sum(np.count_nonzero(Wp[:, j]) - 1 for j in range(4))
        bw = sum(np.count_nonzero(W[R2][:, j]) - 1 for j in range(4))
        res.append((cin_u + cin_v + cout, bu + bv + bw, cin_u, cin_v, cout, bu, bv, bw, R, R2))
res.sort(key=lambda r: (r[0], r[1]))
print("n candidates", len(res))
for r in res[:12]: print(r)
best = res[0]
R, R2 = best[8], best[9]
print("R", R, "R2", R2)
print("phi = U[R]\n", U[R], "\nU' =\n", U @ inv_int(U[R]))
print("psi = V[R]\n", V[R], "\nV' =\n", V @ inv_int(V[R]))
print("W[R2] (c = W[R2]^T c')\n", W[R2], "\nW' =\n", W @ inv_int(W[R2]))
# same-R options for in-place output
print("--- R2 == R options")
for r in res:
    if r[8] == r[9]: print(r)
