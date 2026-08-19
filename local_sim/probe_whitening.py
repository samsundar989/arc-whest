"""DESIGN-TIER VALIDATION PROBE. No gate, no decision, estimator.py untouched.

Q: our Sobol net's off-diagonal 2nd moment is 12.6x worse than its diagonal.
   Does full ZCA whitening pay ON TOP of our carrier, and is the diagonal-only
   version (which we killed on 2026-08-05) really worth 1.000x?

All arms share the SAME base Sobol net and the SAME 24 digital shifts (paired
CRN). Point sets and whitening matrices are MLP-independent -> built once.
Seeds: D16 unused tail (30806016200+), design block, no gate attached.
"""
import sys, time, json
import numpy as np, torch
sys.path.insert(0, "local_sim")
from scipy.stats import norm
from mint_provenance import canonical_mint

torch.backends.cuda.matmul.allow_tf32 = False
dev = "cuda"
BITS, DIM, DEPTH, N = 30, 256, 32, 16384
N_MLPS, N_SHIFTS, N_GT = 24, 24, 1 << 23
SEED0 = 30_806_016_200

dn = np.load("local_sim/sobol_dirnum.npz"); dirnum = dn[dn.files[0]]
def base_points(n, w):
    idx = np.arange(n, dtype=np.uint32); idx = idx ^ (idx >> np.uint32(1))
    idx = idx.reshape((n, 1)); pts = np.zeros((n, w), dtype=np.uint32)
    for k in range(BITS):
        pts ^= (((idx >> np.uint32(k)) & np.uint32(1)) * dirnum[k][:w].astype(np.uint32))
    return pts
PTS = base_points(N, DIM)

def gauss(shift_seed, tent):
    rng = np.random.default_rng(shift_seed)
    sh = rng.integers(0, 1 << BITS, size=DIM, dtype=np.uint32)
    u = ((PTS ^ sh).astype(np.float64) + 0.5) * (1.0 / float(1 << BITS))
    if tent: u = 1.0 - np.abs(2.0 * u - 1.0)
    return norm.ppf(u).astype(np.float32)

def zca(X):
    S = (X.T.astype(np.float64) @ X.astype(np.float64)) / X.shape[0]
    w, V = np.linalg.eigh(S)
    return ((V * (np.maximum(w, 1e-12) ** -0.5)) @ V.T).astype(np.float32)

lg = lambda z: torch.lgamma(torch.tensor(z)).item()
E_R = float(np.exp(0.5*np.log(2.0) + lg((DIM+1)/2.) - lg(DIM/2.)))

ARMS = ["plain", "whitediag", "sphere", "white", "sphere+white",
        "tent", "tent+white", "iid", "iid+white"]

print("building point sets ...", flush=True); t0 = time.time()
BANK = {}   # arm -> list over shifts of (X_gpu, M_gpu or None)
for a in ARMS: BANK[a] = []
for si in range(N_SHIFTS):
    ss = 777_000 + si
    Xp, Xt = gauss(ss, False), gauss(ss, True)
    Xi = np.random.default_rng(555_000 + si).standard_normal((N, DIM)).astype(np.float32)
    r = np.linalg.norm(Xp.astype(np.float64), axis=1, keepdims=True)
    Xs = (Xp * (E_R / r)).astype(np.float32)
    m = Xp.astype(np.float64).mean(axis=0); s = Xp.astype(np.float64).std(axis=0)
    Xd = ((Xp - m) / s).astype(np.float32)
    g = lambda X: torch.from_numpy(X).to(dev)
    BANK["plain"].append((g(Xp), None))
    BANK["whitediag"].append((g(Xd), None))
    BANK["sphere"].append((g(Xs), None))
    BANK["white"].append((g(Xp), g(zca(Xp))))
    BANK["sphere+white"].append((g(Xs), g(zca(Xs))))
    BANK["tent"].append((g(Xt), None))
    BANK["tent+white"].append((g(Xt), g(zca(Xt))))
    BANK["iid"].append((g(Xi), None))
    BANK["iid+white"].append((g(Xi), g(zca(Xi))))
print(f"  built in {time.time()-t0:.0f}s", flush=True)

def fwd(X, W, M):
    h = X if M is None else X @ M
    for li in range(DEPTH): h = torch.clamp(h @ W[li], min=0.0)
    return h.mean(dim=0, dtype=torch.float64)

def truth(W):
    acc = torch.zeros(DIM, dtype=torch.float64, device=dev)
    acc2 = torch.zeros(DIM, dtype=torch.float64, device=dev)
    B = 1 << 17; g = torch.Generator(device=dev); g.manual_seed(12345)
    for _ in range(N_GT // B):
        h = torch.randn(B, DIM, generator=g, device=dev, dtype=torch.float32)
        for li in range(DEPTH): h = torch.clamp(h @ W[li], min=0.0)
        acc += h.sum(dim=0, dtype=torch.float64); acc2 += (h.double()**2).sum(dim=0)
    m = acc / N_GT
    return m, float(((acc2/N_GT - m*m) / N_GT).mean())

res = {a: [] for a in ARMS}; gtn = []
for mi in range(N_MLPS):
    Wn, _, _ = canonical_mint(DIM, DEPTH, SEED0 + mi)
    W = [torch.from_numpy(np.ascontiguousarray(x)).to(dev) for x in Wn]
    t, gn = truth(W); gtn.append(gn)
    for a in ARMS:
        for si in range(N_SHIFTS):
            X, M = BANK[a][si]
            res[a].append(float(((fwd(X, W, M) - t) ** 2).mean()))
    print(f"  mlp {mi+1}/{N_MLPS}  {time.time()-t0:.0f}s", flush=True)

sig = float(np.mean(gtn))
A = {a: np.array(res[a]).reshape(N_MLPS, N_SHIFTS) for a in ARMS}
base = A["plain"]
print(f"\nGT 2^23/MLP, sigma_gt^2 = {sig:.4e}  ({sig/base.mean()*100:.2f}% of plain MSE)")
print(f"{N_MLPS} MLPs x {N_SHIFTS} shifts, paired CRN\n")
print(f"{'arm':<15}{'mean MSE':>12}{'geo-mean vs plain':>20}{'95% CI':>18}{'wins':>10}")
rng = np.random.default_rng(0)
for a in ARMS:
    r = base / A[a]; per = np.exp(np.log(r).mean(axis=1))
    bs = [np.exp(np.log(rng.choice(per, N_MLPS)).mean()) for _ in range(4000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"{a:<15}{A[a].mean():>12.4e}{np.exp(np.log(r).mean()):>19.4f}x"
          f"  [{lo:.3f},{hi:.3f}]{int((r>1).sum()):>7}/{r.size}")
json.dump({a: A[a].tolist() for a in ARMS} | {"sigma_gt2": sig},
          open("white_probe.json", "w"))
print(f"\ntotal {time.time()-t0:.0f}s")
