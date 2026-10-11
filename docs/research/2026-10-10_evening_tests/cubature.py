# How good is a spherical-design (orthonormal basis + antipodes) estimator of E[h_16] vs plain MC, on real nets with exact labels?
# Uses homogeneity: E[h(x)] = E||x|| * E_theta[h(theta)], theta uniform on the unit sphere.
import numpy as np, sys, time, math
ROOT = __import__('os').environ.get('WHEST_ROOT') or str(__import__('pathlib').Path(__file__).resolve().parents[3])  # repo root
nets = [int(s) for s in sys.argv[1].split(',')] if len(sys.argv) > 1 else [0]
K = int(sys.argv[2]) if len(sys.argv) > 2 else 8
W = np.memmap(f'{ROOT}/cache/twin/full_w.f32', dtype=np.float32, mode='r', shape=(1000, 16, 1024, 1024))
truth = np.load(f'{ROOT}/cache/twin/full_truth.npy', mmap_mode='r')   # (1000, 16, 1024) all-layer means
n = 1024
Echi = math.sqrt(2) * math.exp(math.lgamma((n + 1) / 2) - math.lgamma(n / 2))  # E||x|| for x ~ N(0, I_n)
def hadamard(n):
    H = np.array([[1.0]], dtype=np.float32)
    while H.shape[0] < n: H = np.block([[H, H], [H, -H]])
    return H
Hn = hadamard(n) / math.sqrt(n)
def forward(X, Ws):            # X: (m, n) inputs (rows); returns last-layer post-activations (m, n)
    y = X
    for w in Ws:
        y = np.maximum(y @ w, 0.0)    # convention z = W^T h  <=>  row-vector y @ W
    return y
rng = np.random.default_rng(1)
for g in nets:
    Ws = [np.asarray(W[g, l], dtype=np.float32) for l in range(16)]
    tr = np.asarray(truth[g, 15], dtype=np.float64)
    t0 = time.time()
    # antipodal trick at layer 1: relu(-z) = relu(z) - z
    def basis_mean(B):          # B: (n, n) orthonormal rows; returns E-estimate from 2n points +-B
        z = B @ Ws[0]
        a = np.maximum(z, 0.0)
        y = np.concatenate((a, a - z), axis=0)
        for w in Ws[1:]:
            y = np.maximum(y @ w, 0.0)
        return y.mean(axis=0).astype(np.float64) * Echi
    est_rand = []; est_had = []
    for k in range(K):
        Q, _ = np.linalg.qr(rng.standard_normal((n, n)).astype(np.float64)); est_rand.append(basis_mean(Q.astype(np.float32)))
        D = rng.choice([-1.0, 1.0], size=n).astype(np.float32)
        P = rng.permutation(n)
        est_had.append(basis_mean((Hn * D[None, :])[P]))     # randomized Hadamard basis (rows orthonormal)
    # plain Gaussian MC with the same number of forward passes (2n per basis)
    est_mc = []
    for k in range(K):
        X = rng.standard_normal((2 * n, n)).astype(np.float32)
        est_mc.append(forward(X, Ws).mean(axis=0).astype(np.float64))
    print(f'net {g}: time {time.time()-t0:.1f}s; layer-16 mean var(h) proxy: mean truth^2 {np.mean(tr**2):.3e}')
    for name, E in (('random-orthonormal', est_rand), ('rand-Hadamard', est_had), ('plain MC', est_mc)):
        E = np.array(E)
        for k in (1, 2, 4, K):
            if k > len(E): continue
            m = E[:k].mean(axis=0)
            print(f'   {name:20s} k={k:2d} bases ({2*n*k:6d} pts, F/B {2*n*k*16*2*n*n/2**41:.3f}): MSE {np.mean((m-tr)**2):.3e}')
