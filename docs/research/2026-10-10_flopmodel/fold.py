import sys, math, numpy as np
sys.path.insert(0, '.')
from mm import member_memmap
p = sys.argv[1]
layers = [int(x) for x in sys.argv[2].split(',')]
M = {k: member_memmap(p, k + '.npy') for k in ('mu', 'var', 'C11', 'D21', 'Q31')}
erf = np.frompyfunc(math.erf, 1, 1)
def nf(X, mask=None):
    X = X.copy(); np.fill_diagonal(X, 0.0)
    if mask is not None: X = X[:, mask]
    return float(np.sqrt((X * X).sum()))
for l in layers:
    mu = np.array(M['mu'][l], np.float64); s = np.sqrt(np.array(M['var'][l], np.float64))
    C = np.array(M['C11'][l], np.float64); D = np.array(M['D21'][l], np.float64); Q = np.array(M['Q31'][l], np.float64)
    a = mu / s
    phi = np.exp(-0.5 * a * a) / math.sqrt(2 * math.pi); Phi = 0.5 * (1 + erf(a / math.sqrt(2)).astype(np.float64))
    w0 = s * phi + mu * Phi; w1 = Phi; w2 = phi / s; w3 = -a * phi / s**2
    w12 = 2 * w0; w22 = 2 * Phi; w32 = 2 * phi / s
    r = -a / s; pk1 = w0
    I, J = (slice(None), None), (None, slice(None))
    FO = lambda X, wa: 0.5 * ((wa[:, None] * w1[None, :]) * X + ((wa[:, None] * w1[None, :]) * X).T)
    FOD = FO(D, w2)
    FOQ = (1/6) * ((w3[:, None] * w1[None, :]) * Q + ((w3[:, None] * w1[None, :]) * Q).T)
    E = (r[:, None] / 3.0) * Q
    chk = nf(FO(E, w2) - FOQ) / nf(FOQ)
    def L21(X):
        return (0.5 * (w22[:, None] * w1[None, :]) * X + 0.5 * (w12[:, None] * w2[None, :]) * X.T
                - pk1[:, None] * ((w2[:, None] * w1[None, :]) * X + (w1[:, None] * w2[None, :]) * X.T)
                - ((w1 * w1)[:, None] * w1[None, :]) * X)
    Lq = ((1/6) * ((w32[:, None] * w1[None, :]) * Q + (w12[:, None] * w3[None, :]) * Q.T)
          - (pk1[:, None] / 3.0) * ((w3[:, None] * w1[None, :]) * Q + (w1[:, None] * w3[None, :]) * Q.T))
    Hd = lambda X: 0.5 * (w2[:, None] * X + (w1[:, None] * X.T) * r[None, :])
    lead = w1[:, None] * C
    on = a > 2.5; dead = a < -2.5
    de = (D * D).sum(1); de2 = ((D + E) ** 2).sum(1)
    print(f"L{l+1}: on={on.sum()} dead={dead.sum()} chk={chk:.1e} | FO_Q/FO_D21={nf(FOQ)/nf(FOD):.3f}"
          f" | births: L21(D21)/FO_D21={nf(L21(D))/nf(FOD):.2f}  L21(E)/FO_Q={nf(L21(E))/nf(FOQ):.2f}"
          f"  Lq/FO_Q={nf(Lq)/nf(FOQ):.2f}  |L21(E)-Lq|/|Lq|={nf(L21(E)-Lq)/nf(Lq):.2f}"
          f" | dress: |Hd(D21)|/|w1C|={nf(Hd(D))/nf(lead):.3f} |Hd(E)|/|Hd(D21)|={nf(Hd(E))/nf(Hd(D)):.3f}"
          f" on-cols: |Hd(D21)|/|w1C|={nf(Hd(D),on)/max(nf(lead,on),1e-30):.3f} |Hd(E)|/|Hd(D21)|={nf(Hd(E),on)/max(nf(Hd(D),on),1e-30):.3f}"
          f" | d21e rel chg={np.linalg.norm(de2-de)/np.linalg.norm(de):.3f}")
