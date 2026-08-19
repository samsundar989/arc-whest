"""DESIGN-TIER PROBE 2. Is a STRUCTURED DESIGN carrier better than our Sobol?

The honest frontier (4 independent teams) uses a deterministic spherical
5-design (66,048 Kerdock directions) or a strength-4 orthogonal array.
Cheap unbiased proxy available today: a bank of Haar-rotated orthonormal
frames, +/- Q e_i, scaled to the sphere of radius E||x||.  With K rotations
that is N = 2*256*K points, exact for every polynomial of degree <= 3 on the
sphere, and unbiased by the Haar randomisation.

Also tested: antithetic pairing on our Sobol (kills all odd degrees exactly).
Same 24 MLPs, same GT, paired by realisation index.
"""
import sys, time, json
import numpy as np, torch
sys.path.insert(0, "local_sim")
from scipy.stats import norm
from mint_provenance import canonical_mint

torch.backends.cuda.matmul.allow_tf32 = False
dev = "cuda"
BITS, DIM, DEPTH, N = 30, 256, 32, 16384
N_MLPS, N_REPS, N_GT = 24, 24, 1 << 23
SEED0 = 30_806_016_200

dn = np.load("local_sim/sobol_dirnum.npz"); dirnum = dn[dn.files[0]]
def base_points(n, w):
    idx = np.arange(n, dtype=np.uint32); idx = idx ^ (idx >> np.uint32(1))
    idx = idx.reshape((n, 1)); pts = np.zeros((n, w), dtype=np.uint32)
    for k in range(BITS):
        pts ^= (((idx >> np.uint32(k)) & np.uint32(1)) * dirnum[k][:w].astype(np.uint32))
    return pts
PTS = base_points(N, DIM); PTS_H = base_points(N // 2, DIM)

lg = lambda z: torch.lgamma(torch.tensor(z)).item()
E_R = float(np.exp(0.5*np.log(2.0) + lg((DIM+1)/2.) - lg(DIM/2.)))

def sobol(seed, n=N, pts=None):
    rng = np.random.default_rng(seed)
    sh = rng.integers(0, 1 << BITS, size=DIM, dtype=np.uint32)
    p = PTS if pts is None else pts
    u = ((p ^ sh).astype(np.float64) + 0.5) * (1.0 / float(1 << BITS))
    return norm.ppf(u).astype(np.float32)

def frames(seed, K=N // (2 * DIM)):
    """Haar-rotated orthogonal frame bank: +/- Q_k e_i on the sphere."""
    rng = np.random.default_rng(seed)
    out = np.empty((2 * DIM * K, DIM), dtype=np.float32)
    for k in range(K):
        Q, R = np.linalg.qr(rng.standard_normal((DIM, DIM)))
        Q = Q * np.sign(np.diag(R))
        out[2*DIM*k : 2*DIM*k + DIM] = (Q.T * E_R).astype(np.float32)
        out[2*DIM*k + DIM : 2*DIM*(k+1)] = (-Q.T * E_R).astype(np.float32)
    return out

def frames_gauss(seed, K=N // (2 * DIM)):
    """Same frames but with a chi-distributed radius per frame (Gaussian, not sphere)."""
    rng = np.random.default_rng(seed)
    X = frames(seed, K).astype(np.float64) / E_R
    r = rng.chisquare(DIM, size=(2 * DIM * K, 1)) ** 0.5
    return (X * r).astype(np.float32)

def anti(seed):
    X = sobol(seed, N // 2, PTS_H)
    return np.concatenate([X, -X], axis=0)

def sphere(X):
    r = np.linalg.norm(X.astype(np.float64), axis=1, keepdims=True)
    return (X * (E_R / r)).astype(np.float32)

ARMS = {
    "sobol (ours)":      lambda s: sobol(s),
    "sobol+anti":        lambda s: anti(s),
    "sobol+anti+sph":    lambda s: sphere(anti(s)),
    "haar frames (sph)": lambda s: frames(s),
    "haar frames (gau)": lambda s: frames_gauss(s),
    "iid":               lambda s: np.random.default_rng(s).standard_normal((N, DIM)).astype(np.float32),
}
print("building carriers ...", flush=True); t0 = time.time()
BANK = {a: [torch.from_numpy(f(900_000 + i)).to(dev) for i in range(N_REPS)]
        for a, f in ARMS.items()}
print(f"  {time.time()-t0:.0f}s", flush=True)

def fwd(X, W):
    h = X
    for li in range(DEPTH): h = torch.clamp(h @ W[li], min=0.0)
    return h.mean(dim=0, dtype=torch.float64)

def truth(W):
    acc = torch.zeros(DIM, dtype=torch.float64, device=dev)
    B = 1 << 17; g = torch.Generator(device=dev); g.manual_seed(12345)
    for _ in range(N_GT // B):
        h = torch.randn(B, DIM, generator=g, device=dev, dtype=torch.float32)
        for li in range(DEPTH): h = torch.clamp(h @ W[li], min=0.0)
        acc += h.sum(dim=0, dtype=torch.float64)
    return acc / N_GT

res = {a: [] for a in ARMS}
for mi in range(N_MLPS):
    Wn, _, _ = canonical_mint(DIM, DEPTH, SEED0 + mi)
    W = [torch.from_numpy(np.ascontiguousarray(x)).to(dev) for x in Wn]
    t = truth(W)
    for a in ARMS:
        for i in range(N_REPS):
            res[a].append(float(((fwd(BANK[a][i], W) - t) ** 2).mean()))

A = {a: np.array(res[a]).reshape(N_MLPS, N_REPS) for a in ARMS}
base = A["sobol (ours)"]
print(f"\n{N_MLPS} MLPs x {N_REPS} realisations, N={N}, GT 2^23\n")
print(f"{'carrier':<20}{'mean MSE':>12}{'vs sobol':>12}{'95% CI':>18}{'wins':>10}")
rng = np.random.default_rng(0)
for a in ARMS:
    r = base / A[a]; per = np.exp(np.log(r).mean(axis=1))
    bs = [np.exp(np.log(rng.choice(per, N_MLPS)).mean()) for _ in range(4000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"{a:<20}{A[a].mean():>12.4e}{np.exp(np.log(r).mean()):>11.4f}x"
          f"  [{lo:.3f},{hi:.3f}]{int((r>1).sum()):>7}/{r.size}")
json.dump({a: A[a].tolist() for a in ARMS}, open("design_probe.json", "w"))
print(f"\ntotal {time.time()-t0:.0f}s")
