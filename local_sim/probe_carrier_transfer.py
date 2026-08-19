"""DESIGN-TIER PROBE 3. Does the frame-bank advantage survive contact with the
SHIPPED stack (rank-32 J J^T rotation + dead-neuron pruning at tau=-2.5)?

The rotation is the obvious transfer risk: it is tuned to put Sobol's
best-equidistributed coordinates on the network's most sensitive directions.
A frame bank has no coordinate ordering, so the rotation may buy it nothing --
or may even hurt.  Also sweeps N to check the advantage is not an N=16384 artefact.
"""
import sys, time
import numpy as np, torch
sys.path.insert(0, "local_sim")
from scipy.stats import norm
from mint_provenance import canonical_mint
from scipy.special import ndtr

torch.backends.cuda.matmul.allow_tf32 = False
dev = "cuda"
BITS, DIM, DEPTH = 30, 256, 32
N_MLPS, N_REPS, N_GT = 20, 16, 1 << 23
SEED0 = 30_806_016_300          # fresh slice of the D16 tail

dn = np.load("local_sim/sobol_dirnum.npz"); dirnum = dn[dn.files[0]]
def base_points(n, w):
    idx = np.arange(n, dtype=np.uint32); idx = idx ^ (idx >> np.uint32(1))
    idx = idx.reshape((n, 1)); pts = np.zeros((n, w), dtype=np.uint32)
    for k in range(BITS):
        pts ^= (((idx >> np.uint32(k)) & np.uint32(1)) * dirnum[k][:w].astype(np.uint32))
    return pts
lg = lambda z: torch.lgamma(torch.tensor(z)).item()
E_R = float(np.exp(0.5*np.log(2.0) + lg((DIM+1)/2.) - lg(DIM/2.)))

def sobol(seed, n):
    rng = np.random.default_rng(seed)
    sh = rng.integers(0, 1 << BITS, size=DIM, dtype=np.uint32)
    u = ((base_points(n, DIM) ^ sh).astype(np.float64) + 0.5) * (1.0/float(1 << BITS))
    return norm.ppf(u).astype(np.float32)

def frames(seed, n):
    rng = np.random.default_rng(seed); K = n // (2 * DIM)
    out = np.empty((2 * DIM * K, DIM), dtype=np.float32)
    for k in range(K):
        Q, R = np.linalg.qr(rng.standard_normal((DIM, DIM)))
        Q = Q * np.sign(np.diag(R))
        out[2*DIM*k:2*DIM*k+DIM] = (Q.T * E_R).astype(np.float32)
        out[2*DIM*k+DIM:2*DIM*(k+1)] = (-Q.T * E_R).astype(np.float32)
    return out

def closure(w64):
    """estimator.py:_closure -- alphas and post-ReLU means."""
    mu = np.zeros(DIM); cov = np.eye(DIM)
    off = np.ones((DIM, DIM)) - np.eye(DIM)
    alphas, mus, phis = [], [], []
    for w in w64:
        mp = w.T @ mu; cp = w.T @ cov @ w
        vp = np.maximum(np.diag(cp), 1e-12); sg = np.sqrt(vp)
        a = mp / sg; Phi = ndtr(a); phi = np.exp(-0.5*a*a)/np.sqrt(2*np.pi)
        mu = mp * Phi + sg * phi
        ez2 = (mp*mp + vp) * Phi + mp * sg * phi
        var = np.maximum(ez2 - mu*mu, 0.0)
        cov = np.outer(Phi, Phi) * cp * off + np.diag(var)
        alphas.append(a); mus.append(mu.copy()); phis.append(Phi)
    return alphas, mus, phis

def basis(w64, phis):
    J = w64[0]
    for k in range(1, DEPTH): J = (J * phis[k-1]) @ w64[k]
    _, vecs = np.linalg.eigh(J @ J.T)
    top = vecs[:, DIM-32:][:, ::-1]
    hi, lo = top.max(axis=0), top.min(axis=0)
    top = top * np.where(hi >= -lo, 1.0, -1.0)
    q, r = np.linalg.qr(top, mode="complete")
    d = np.diag(r)
    return q * np.concatenate([np.where(d < 0, -1.0, 1.0), np.ones(DIM-32)])

def build(w64, rotate, prune):
    alphas, mus, phis = closure(w64)
    B = basis(w64, phis) if rotate else None
    alive, dead = [], []
    for li in range(DEPTH):
        if li == DEPTH-1 or not prune:
            alive.append(np.arange(DIM)); dead.append(None); continue
        keep = alphas[li] > -2.5; idx = np.flatnonzero(keep)
        if idx.size == 0: idx = np.flatnonzero(alphas[li] >= alphas[li].max())
        alive.append(idx); d = np.flatnonzero(~keep); dead.append(d if d.size else None)
    sl, offs, prev = [], [], None
    for li in range(DEPTH):
        wl = w64[li]
        if li == 0 and B is not None: wl = B.T @ wl
        w32 = wl.astype(np.float32); cur = alive[li]
        b = w32 if prev is None else w32[prev]
        sl.append(np.ascontiguousarray(b[:, cur]))
        dp = dead[li-1] if li > 0 else None
        offs.append(None if dp is None else (mus[li-1][dp].astype(np.float32) @ w32[np.ix_(dp, cur)]))
        prev = cur
    return sl, offs

def fwd(X, sl, offs):
    h = X
    for li in range(DEPTH):
        z = h @ sl[li]
        if offs[li] is not None: z = z + offs[li]
        h = torch.clamp(z, min=0.0)
    return h.mean(dim=0, dtype=torch.float64)

def truth(W):
    acc = torch.zeros(DIM, dtype=torch.float64, device=dev)
    B = 1 << 17; g = torch.Generator(device=dev); g.manual_seed(12345)
    for _ in range(N_GT // B):
        h = torch.randn(B, DIM, generator=g, device=dev, dtype=torch.float32)
        for li in range(DEPTH): h = torch.clamp(h @ W[li], min=0.0)
        acc += h.sum(dim=0, dtype=torch.float64)
    return acc / N_GT

CFG = [("bare",  False, False), ("rot",  True, False), ("rot+prune (SHIPPED)", True, True)]
t0 = time.time()
for Nn in (16384, 65536):
    CAR = {"sobol": [torch.from_numpy(sobol(900_000+i, Nn)).to(dev) for i in range(N_REPS)],
           "frames": [torch.from_numpy(frames(900_000+i, Nn)).to(dev) for i in range(N_REPS)]}
    res = {(c, k): [] for c, _, _ in CFG for k in CAR}
    for mi in range(N_MLPS):
        Wn, _, _ = canonical_mint(DIM, DEPTH, SEED0 + mi)
        W = [torch.from_numpy(np.ascontiguousarray(x)).to(dev) for x in Wn]
        w64 = [x.astype(np.float64) for x in Wn]
        t = truth(W)
        for cname, rot, pr in CFG:
            sl, offs = build(w64, rot, pr)
            slt = [torch.from_numpy(s).to(dev) for s in sl]
            oft = [None if o is None else torch.from_numpy(o).to(dev) for o in offs]
            for k in CAR:
                for i in range(N_REPS):
                    res[(cname, k)].append(float(((fwd(CAR[k][i], slt, oft) - t) ** 2).mean()))
    print(f"\nN = {Nn}   ({N_MLPS} MLPs x {N_REPS} realisations, GT 2^23)")
    print(f"{'stack':<24}{'sobol MSE':>13}{'frames MSE':>13}{'frames/sobol':>15}{'95% CI':>18}")
    rng = np.random.default_rng(0)
    for cname, _, _ in CFG:
        S = np.array(res[(cname,"sobol")]).reshape(N_MLPS, N_REPS)
        F = np.array(res[(cname,"frames")]).reshape(N_MLPS, N_REPS)
        r = S / F; per = np.exp(np.log(r).mean(axis=1))
        bs = [np.exp(np.log(rng.choice(per, N_MLPS)).mean()) for _ in range(4000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"{cname:<24}{S.mean():>13.4e}{F.mean():>13.4e}"
              f"{np.exp(np.log(r).mean()):>14.4f}x  [{lo:.3f},{hi:.3f}]")
print(f"\ntotal {time.time()-t0:.0f}s")
