"""Monte-Carlo truth with per-layer cumulant accumulators. Usage:
   python3 mc.py n L seed Nsamples [sphere] -> results/mc_n{n}_L{L}_s{seed}[_sphere].npz
"""
import numpy as np, sys, time, os
from tower import make_net

def run_mc(W, N, seed_mc=12345, sphere=False, batch=1 << 16, dtype=np.float32):
    L = len(W); n = W[0].shape[0]
    Wf = [w.astype(dtype) for w in W]
    rng = np.random.default_rng(seed_mc)
    acc = [dict(S1=np.zeros(n), S2=np.zeros((n, n)), S3d=np.zeros(n), S21=np.zeros((n, n)),
                S4d=np.zeros(n), S22=np.zeros((n, n)), S3all=None) for _ in range(L)]
    Sh = np.zeros(n); Sh2 = np.zeros(n); Sh3 = np.zeros(n); Sh4=np.zeros(n)
    done = 0
    while done < N:
        B = min(batch, N - done)
        X = rng.standard_normal((B, n), dtype=dtype)
        if sphere:
            X *= (np.sqrt(n)/np.linalg.norm(X, axis=1))[:, None]
        H = X
        for l in range(L):
            Z = H@Wf[l].T
            if l > 0 or sphere:   # layer-1 Gaussian stats are exact; skip unless sphere
                Z2 = Z*Z
                a = acc[l]
                a['S1'] += Z.sum(0, dtype=np.float64); a['S2'] += (Z.T@Z).astype(np.float64)
                a['S3d'] += np.einsum('bi,bi->i', Z2, Z).astype(np.float64)
                a['S21'] += (Z2.T@Z).astype(np.float64); a['S4d'] += np.einsum('bi,bi->i', Z2, Z2).astype(np.float64)
                a['S22'] += (Z2.T@Z2).astype(np.float64)
            H = np.maximum(Z, 0)
        Hd = H.astype(np.float64)
        Sh += Hd.sum(0); Sh2 += (Hd*Hd).sum(0); Sh3 += (Hd**3).sum(0); Sh4 += (Hd**4).sum(0)
        done += B
    if not sphere:
        C1 = W[0]@W[0].T; acc[0]['S2'] = N*C1; acc[0]['S21'][:] = 0; acc[0]['S3d'][:] = 0
        acc[0]['S4d'] = N*3*np.diag(C1)**2; acc[0]['S22'] = N*(np.outer(np.diag(C1), np.diag(C1)) + 2*C1**2)
    out = dict(N=N, y=Sh/N, y_var=(Sh2/N - (Sh/N)**2)/N, hL_m2=Sh2/N, hL_m3=Sh3/N, hL_m4=Sh4/N)
    for l in range(L):
        a = acc[l]; m1 = a['S1']/N; M2 = a['S2']/N; C = M2 - np.outer(m1, m1)
        M3d = a['S3d']/N; M21 = a['S21']/N; M4d = a['S4d']/N; M22 = a['S22']/N
        # central: kappa3_iii, kappa3_iij (= E[(zi-mi)^2 (zj-mj)]), kappa4_iiii, kappa4_iijj
        k3d = M3d - 3*m1*np.diag(M2) + 2*m1**3
        # E[zi^2 zj] - 2 mi E[zi zj] - mj E[zi^2] + 2 mi^2 mj
        k3p = M21 - 2*m1[:, None]*M2 - m1[None, :]*np.diag(M2)[:, None] + 2*(m1**2)[:, None]*m1[None, :]
        V = np.diag(C)
        m4c = M4d - 4*m1*M3d + 6*m1**2*np.diag(M2) - 3*m1**4
        k4d = m4c - 3*V**2
        # E[(zi-mi)^2 (zj-mj)^2] = M22 - 2 mj E[zi^2 zj] - 2 mi E[zj^2 zi] + mj^2 E[zi^2] + mi^2 E[zj^2] + 4 mi mj E[zi zj] - 3 mi^2 mj^2
        m22 = M22 - 2*m1[None, :]*M21 - 2*m1[:, None]*M21.T + (m1**2)[None, :]*np.diag(M2)[:, None] \
              + (m1**2)[:, None]*np.diag(M2)[None, :] + 4*np.outer(m1, m1)*M2 - 3*np.outer(m1**2, m1**2)
        k4p = m22 - np.outer(V, V) - 2*C**2
        out[f'mu{l+1}'] = m1; out[f'C{l+1}'] = C; out[f'k3d{l+1}'] = k3d; out[f'k3p{l+1}'] = k3p
        out[f'k4d{l+1}'] = k4d; out[f'k4p{l+1}'] = k4p
    return out

if __name__ == '__main__':
    n, L, seed, N = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(float(sys.argv[4]))
    sphere = len(sys.argv) > 5 and sys.argv[5] == 'sphere'
    W = make_net(n, L, seed)
    t = time.time()
    out = run_mc(W, N, sphere=sphere)
    fn = f'/home/claude/sandbox/results/mc_n{n}_L{L}_s{seed}' + ('_sphere' if sphere else '') + '.npz'
    np.savez(fn, **out)
    print(fn, 'time', time.time()-t, 'noise RMS', np.sqrt(out['y_var'].mean()))
