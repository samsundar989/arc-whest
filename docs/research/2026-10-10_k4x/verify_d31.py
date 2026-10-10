# Verifies the O(n^3) D31 = kappa4(a,a,a,b) readouts against explicit 4-tensors (errors ~1e-13).
# CONVENTION: Sym = AVERAGE over the 24 index permutations. The estimator's star-D4 convention
# (D4[a] += 4*sum_j w3_j A_aj^3 M_aj) is 4x this average for the star (sum over the 4 placements of
# the centre), i.e. star D31 in code convention = sum_j w3_j (A_aj^3 M_bj + 3 M_aj A_aj^2 A_bj).
# Match conventions before building.
import numpy as np, itertools
rng = np.random.default_rng(3); n = 7
A = rng.normal(size=(n,n)); B = rng.normal(size=(n,n)); w3 = rng.normal(size=n)
G = rng.normal(size=(n,n)); G = G+G.T
def sym(T):
    return sum(np.transpose(T, p) for p in itertools.permutations(range(4)))/24.0
# star: T = sum_j w3_j B_j (x) A_j (x) A_j (x) A_j   (B = the M leg)
T = np.einsum('j,aj,bj,cj,dj->abcd', w3, B, A, A, A); S = sym(T)
D31 = np.array([[S[a,a,a,b] for b in range(n)] for a in range(n)])
f = 0.25*(((A*A*A)*w3) @ B.T + 3*((B*A*A)*w3) @ A.T)
print("star D31 max err", np.abs(D31 - f).max())
# path: T = sum_jk G_jk A_j (x) B_j (x) B_k (x) A_k
T = np.einsum('jk,aj,bj,ck,dk->abcd', G, A, B, B, A); S = sym(T)
D31 = np.array([[S[a,a,a,b] for b in range(n)] for a in range(n)])
V = (A*B) @ G
f = 0.5*((B*V) @ A.T + (A*V) @ B.T)
print("path D31 max err", np.abs(D31 - f).max())
# (2,2) block moved by P: kappa_{ijkl} = K_rc on the multiset {r,r,c,c}
K = rng.normal(size=(n,n)); K = K+K.T; np.fill_diagonal(K, 0); P = rng.normal(size=(n,n))
kap = np.zeros((n,)*4)
for r in range(n):
    for c in range(n):
        if r!=c:
            for t in set(itertools.permutations((r,r,c,c))): kap[t] = K[r,c]
kp = np.einsum('ai,bj,ck,dl,ijkl->abcd', P,P,P,P,kap)
D31 = np.array([[kp[a,a,a,b] for b in range(n)] for a in range(n)])
f = 3*((((P*P)@K)*P) @ P.T)
print("(2,2) D31 max err", np.abs(D31 - f).max())
