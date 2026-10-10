import sys, math, numpy as np
sys.path.insert(0, '.')
from mm import member_memmap
p = sys.argv[1]; layers = [int(x) for x in sys.argv[2].split(',')]
M = {k: member_memmap(p, k + '.npy') for k in ('mu', 'var', 'C11', 'D21')}
erf = np.frompyfunc(math.erf, 1, 1)
for l in layers:
    mu = np.array(M['mu'][l], np.float64); s = np.sqrt(np.array(M['var'][l], np.float64))
    C = np.array(M['C11'][l], np.float64); D = np.array(M['D21'][l], np.float64)
    a = mu / s; phi = np.exp(-0.5*a*a)/math.sqrt(2*math.pi); Phi = 0.5*(1+erf(a/math.sqrt(2)).astype(np.float64))
    w1 = Phi; w2 = phi/s; r = -a/s
    X = w1[:, None] * C                       # undressed leg
    Hr = 0.5 * (w1[:, None] * D.T) * r[None, :]   # r_j-amplified dressing piece
    Hw = 0.5 * w2[:, None] * D
    on = a > 2.5; mid = np.abs(a) <= 2.5
    colE = lambda Y: (Y*Y).sum(0)
    hubE = (w1**2) * colE(X)                  # route-A hub column energy  ~ cA_j^2 |X_j|^2
    starE = (w2**2) * colE(X)**1              # star weight^2 * |X_j|^2 (proxy)
    print(f"L{l+1}: on={on.sum()} | hubA energy share on always-on cols={hubE[on].sum()/hubE.sum():.2f} (star proxy {starE[on].sum()/starE.sum():.3f})"
          f" | on-cols |r D21^T|/|w1 C|={math.sqrt(colE(Hr)[on].sum()/colE(X)[on].sum()):.2f}  |w2 D21|/|w1 C|={math.sqrt(colE(Hw)[on].sum()/colE(X)[on].sum()):.3f}"
          f" | mid-cols |r D21^T|/|w1 C|={math.sqrt(colE(Hr)[mid].sum()/colE(X)[mid].sum()):.2f}  median |r| on={np.median(np.abs(r[on])):.1f}")
