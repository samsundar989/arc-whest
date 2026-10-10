import sys, math, numpy as np
sys.path.insert(0, '.')
from mm import member_memmap
p = sys.argv[1]; layers = [int(x) for x in sys.argv[2].split(',')]; taus = [float(x) for x in sys.argv[3].split(',')]
M = {k: member_memmap(p, k + '.npy') for k in ('mu', 'var', 'C11', 'D21', 'Q31')}
erf = np.frompyfunc(math.erf, 1, 1)
def nf(X):
    X = X.copy(); np.fill_diagonal(X, 0.0); return float(np.sqrt((X*X).sum()))
for l in layers:
    mu = np.array(M['mu'][l], np.float64); s = np.sqrt(np.array(M['var'][l], np.float64))
    C = np.array(M['C11'][l], np.float64); D = np.array(M['D21'][l], np.float64); Q = np.array(M['Q31'][l], np.float64)
    a = mu/s; phi = np.exp(-0.5*a*a)/math.sqrt(2*math.pi); Phi = 0.5*(1+erf(a/math.sqrt(2)).astype(np.float64))
    w0 = s*phi+mu*Phi; w1 = Phi; w2 = phi/s; w3 = -a*phi/s**2; w12 = 2*w0; w22 = 2*Phi; w32 = 2*phi/s; r = -a/s; pk1 = w0
    FOQ = (1/6)*((w3[:,None]*w1[None,:])*Q + ((w3[:,None]*w1[None,:])*Q).T)
    Hd = lambda X: 0.5*(w2[:,None]*X + (w1[:,None]*X.T)*r[None,:])
    def L21(X):
        return (0.5*(w22[:,None]*w1[None,:])*X + 0.5*(w12[:,None]*w2[None,:])*X.T
                - pk1[:,None]*((w2[:,None]*w1[None,:])*X + (w1[:,None]*w2[None,:])*X.T) - ((w1*w1)[:,None]*w1[None,:])*X)
    Lq = ((1/6)*((w32[:,None]*w1[None,:])*Q + (w12[:,None]*w3[None,:])*Q.T)
          - (pk1[:,None]/3.0)*((w3[:,None]*w1[None,:])*Q + (w1[:,None]*w3[None,:])*Q.T))
    de = (D*D).sum(1)
    out = [f"L{l+1}:"]
    for t in taus:
        om = (np.abs(a) <= t).astype(float)
        E = (om*r/3.0)[:,None]*Q
        FOm = 0.5*((w2[:,None]*w1[None,:])*E + ((w2[:,None]*w1[None,:])*E).T)
        out.append(f"tau={t}: covkept={nf(FOm)/nf(FOQ):.3f} dress|Hd(E)|/|Hd(D)|={nf(Hd(E))/nf(Hd(D)):.3f} d21e={np.linalg.norm(((D+E)**2).sum(1)-de)/np.linalg.norm(de):.3f} births|L21(E)|/|FOQ|={nf(L21(E))/nf(FOQ):.2f}")
    print("  ".join(out))
