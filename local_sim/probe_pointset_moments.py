"""Pure point-set property: how wrong is the empirical second moment of OUR
actual shipped point set?  No MLPs, no ground truth, no seed block involved.

Reproduces estimator.py's construction exactly:
  Sobol integers -> 30-bit digital XOR shift -> (+0.5)/2^30 -> norm.ppf -> float32
"""
import numpy as np, sys
sys.path.insert(0, "local_sim")
from scipy.stats import norm

d = np.load("local_sim/sobol_dirnum.npz")
dirnum = d[d.files[0]]
BITS, DIM = 30, 256

def base_points(n, width):
    idx = np.arange(n, dtype=np.uint32)
    idx = idx ^ (idx >> np.uint32(1))
    idx = idx.reshape((n, 1))
    pts = np.zeros((n, width), dtype=np.uint32)
    for k in range(BITS):
        bit = (idx >> np.uint32(k)) & np.uint32(1)
        pts ^= (bit * dirnum[k][:width].astype(np.uint32))
    return pts

def sample(n, width, seed):
    rng = np.random.default_rng(seed)
    shift = rng.integers(0, 1 << BITS, size=width, dtype=np.uint32)
    pts = base_points(n, width)
    u = ((pts ^ shift).astype(np.float64) + 0.5) * (1.0 / float(1 << BITS))
    return norm.ppf(u).astype(np.float32).astype(np.float64)

def iid(n, width, seed):
    return np.random.default_rng(seed).standard_normal((n, width))

N = 16384
print(f"N={N}, d={DIM}, 8 realisations each\n")
rows = []
for name, gen in (("sobol+digital shift (OURS)", sample), ("plain iid", iid)):
    m_err, offd, diag_err, cond = [], [], [], []
    for s in range(8):
        X = gen(N, DIM, 1000 + s)
        mu = X.mean(axis=0)
        S = (X.T @ X) / N                       # empirical 2nd moment
        E = S - np.eye(DIM)
        m_err.append(np.sqrt((mu**2).mean()))
        diag_err.append(np.sqrt((np.diag(E)**2).mean()))
        off = E - np.diag(np.diag(E))
        offd.append(np.sqrt((off**2).sum() / (DIM*(DIM-1))))
        cond.append(np.linalg.cond(S))
    rows.append((name, np.mean(m_err), np.mean(diag_err), np.mean(offd), np.mean(cond)))

print(f"{'point set':<28}{'rms mean err':>14}{'rms diag err':>14}{'rms offdiag':>14}{'cond(S)':>10}")
for r in rows:
    print(f"{r[0]:<28}{r[1]:>14.3e}{r[2]:>14.3e}{r[3]:>14.3e}{r[4]:>10.3f}")

a, b = rows[0], rows[1]
print(f"\nSobol vs iid   mean: {b[1]/a[1]:6.2f}x better")
print(f"Sobol vs iid   diag: {b[2]/a[2]:6.2f}x better")
print(f"Sobol vs iid offdiag: {b[3]/a[3]:6.2f}x better")
print(f"\ntheory: iid rms mean err = 1/sqrt(N) = {1/np.sqrt(N):.3e}")
print(f"theory: iid rms offdiag  = 1/sqrt(N) = {1/np.sqrt(N):.3e}")
print(f"theory: iid rms diag     = sqrt(2/N) = {np.sqrt(2/N):.3e}")
