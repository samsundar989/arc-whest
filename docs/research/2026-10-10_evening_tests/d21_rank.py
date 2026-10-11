# Is D21 (and its first-order covariance image E) effectively low-rank, in the metric that matters (next-layer variance)?
import numpy as np, sys, math
from math import erf
ROOT = __import__('os').environ.get('WHEST_ROOT') or str(__import__('pathlib').Path(__file__).resolve().parents[3])  # repo root
g = int(sys.argv[1]) if len(sys.argv) > 1 else 0
layers = [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [4, 8, 12, 14]
z = np.load(f'{ROOT}/cache/twin_labels/fit_{g:03d}.npz')
mu, var, D3 = z['mu'].astype(np.float64), z['var'].astype(np.float64), z['D3'].astype(np.float64)
D21all = z['D21']; C11all = z['C11']
W = np.memmap(f'{ROOT}/cache/twin/full_w.f32', dtype=np.float32, mode='r', shape=(1000, 16, 1024, 1024))
n = 1024
Phi = lambda x: 0.5*(1+np.vectorize(erf)(x/np.sqrt(2)))
phi = lambda x: np.exp(-x*x/2)/np.sqrt(2*np.pi)
def post_mean(m, v):
    s = np.sqrt(v); a = m/s
    return s*phi(a) + m*Phi(a)
# sanity / convention check: pre mean at l+1 vs W^T postmean(l) or W postmean(l)
l = 6
pm = post_mean(mu[l], var[l]) + 0  # Gaussian approx of post mean
Wl = np.asarray(W[g, l+1], dtype=np.float64)
c1 = np.corrcoef(Wl.T @ pm, mu[l+1])[0,1]; c2 = np.corrcoef(Wl @ pm, mu[l+1])[0,1]
print(f'convention check l={l}: corr(W^T pm, mu_next)={c1:.5f}  corr(W pm, mu_next)={c2:.5f}')
useT = c1 > c2
for l in layers:
    m, v = mu[l], var[l]; s = np.sqrt(v); a = m/s
    w1 = Phi(a); w2 = phi(a)/s
    D21 = D21all[l].astype(np.float64); d3 = np.diag(D21).copy(); np.fill_diagonal(D21, 0.0)
    E = 0.5*(w2[:, None]*D21*w1[None, :]); E = E + E.T          # first-order cov contribution (off-diag)
    C = C11all[l].astype(np.float64)                               # pre-activation off-diag covariance (labels)
    Wn = np.asarray(W[g, l+1], dtype=np.float64)
    M = Wn if useT else Wn.T                                       # S' = M^T C M
    # next-layer variance contribution of E and of the post-activation covariance proxy
    def dvar(X): return np.einsum('ia,ij,ja->a', M, X, M)
    vE = dvar(E)
    # reference: next-layer pre-activation variance (labels)
    vnext = var[l+1]
    # eigen-decomp of E (symmetric)
    ev, U = np.linalg.eigh(E); order = np.argsort(-np.abs(ev)); ev = ev[order]; U = U[:, order]
    fro2 = (ev**2).sum()
    out = []
    for r in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        Er = (U[:, :r]*ev[:r]) @ U[:, :r].T
        dv = dvar(E - Er)
        out.append((r, 1 - (ev[:r]**2).sum()/fro2, np.sqrt(np.mean(dv**2))/np.sqrt(np.mean(vE**2)), np.sqrt(np.mean((dv/vnext)**2))))
    # SVD of D21 itself
    sv = np.linalg.svd(D21, compute_uv=False); sfro = (sv**2).sum()
    print(f'\n== net {g} layer {l}: |alpha|<2.5 frac {np.mean(np.abs(a)<2.5):.2f}; rms E-induced var / var_next = {np.sqrt(np.mean((vE/vnext)**2)):.3e}')
    print('  D21 energy left after rank r:', ' '.join(f'{r}:{1-(sv[:r]**2).sum()/sfro:.3f}' for r in (1,2,4,8,16,32,64,128,256)))
    print('  r   E-energy-left   rel.rms(dvar)/rms(vE)   rms(dvar/var_next)')
    for r, el, rel, rv in out: print(f'  {r:4d}   {el:9.3f}        {rel:9.3f}             {rv:.3e}')
