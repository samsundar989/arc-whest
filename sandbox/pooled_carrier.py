"""Pooled-column kappa3 carrier vs exact-age/band split, on exact first-order hub-star lineages.

For a random He net (x ~ N(0,I)), run the Gaussian (mu, C) chain, build every hub-star lineage
(hub j born at layer s: u = Jacobian column, v = cross-covariance column / sigma_j, weight w = sigma_j phi(alpha_j)),
transport them with the gate-linearised propagator, and compare the kappa3 readouts D3 (diag) and D21 (pair slice)
of three representations against the full lineage sum:
  FULL   : all ages, all hubs
  SPLIT  : ages <= E exact (KAS hubs |alpha|<2.5), older ages projected on the best rank-r basis of their legs
  POOLED : one pooled set of columns under a cap N, lowest-importance columns deleted sequentially (3 importance rules)
Errors are sensitivity-weighted (D3: (alpha phi/sigma^2)^2 ; D21: (phi_i phi_k)^2) and unweighted; costs in units of n^2.
"""
import numpy as np, json, sys, time
from relu_hermite import relu_moments, phi, Phi, cov_relu

def k2_chain(W):
    n = W[0].shape[0]; L = len(W)
    mu = np.zeros(n); C = W[0] @ W[0].T
    st = []
    for l in range(L):
        sigma = np.sqrt(np.diag(C)); alpha = mu / sigma
        g = Phi(alpha); M1, V, _, _ = relu_moments(mu, sigma)
        rho = C / np.outer(sigma, sigma); np.fill_diagonal(rho, 1.0)
        Ch = cov_relu(mu, sigma, rho); np.fill_diagonal(Ch, V)
        st.append(dict(mu=mu, C=C, sigma=sigma, alpha=alpha, g=g, M1=M1, V=V, Ch=Ch))
        if l + 1 < L:
            mu = W[l + 1] @ M1; C = W[l + 1] @ Ch @ W[l + 1].T
    return st

def readouts(U, V, w):
    """D3_i = 3 sum_j w_j u_ij v_ij^2 ; D21_ik = sum_j w_j (2 u_ij v_ij v_kj + u_kj v_ij^2)"""
    UVw = (U * V) * w[None, :]
    D3 = 3.0 * np.sum(UVw * V, axis=1)
    D21 = 2.0 * UVw @ V.T + ((V * V) * w[None, :]) @ U.T
    np.fill_diagonal(D21, 0.0)
    return D3, D21

def werr(a, b, wt):
    d = a - b
    return float(np.sqrt(np.sum(wt * d * d) / max(np.sum(wt * b * b), 1e-300)))

def run(n, L, seed, E_list, r_list, N_list, kas=2.5):
    W = [np.random.default_rng(seed + 1000 * k).standard_normal((n, n)) * np.sqrt(2.0 / n) for k in range(L)]
    st = k2_chain(W)
    # --- full lineage pass: keep legs per birth layer ---
    legs = {}   # s -> dict(U, V, w, alpha_birth)
    full = {}   # L -> (D3, D21)
    sens = {}
    for l in range(L):
        s_ = st[l]
        # transport existing legs to layer l (they live at pre-activation z_l)
        if l > 0:
            G = W[l] * st[l - 1]['g'][None, :]
            for s in legs:
                legs[s]['U'] = G @ legs[s]['U']; legs[s]['V'] = G @ legs[s]['V']
            # births at layer l-1 (hubs of z_{l-1}), first seen at z_l
            p = st[l - 1]
            U0 = W[l].copy()                                       # column j = W[l][:, j]
            V0 = W[l] @ ((p['g'][:, None] * p['C']) / p['sigma'][None, :])  # W g C e_j / sigma_j
            w0 = p['sigma'] * phi(p['alpha'])
            legs[l - 1] = dict(U=U0, V=V0, w=w0, alpha=p['alpha'].copy())
        if legs:
            Uall = np.concatenate([legs[s]['U'] for s in sorted(legs)], axis=1)
            Vall = np.concatenate([legs[s]['V'] for s in sorted(legs)], axis=1)
            wall = np.concatenate([legs[s]['w'] for s in sorted(legs)])
            full[l] = readouts(Uall, Vall, wall)
        a, sg = s_['alpha'], s_['sigma']
        sens[l] = dict(s3=(a * phi(a) / sg**2)**2, t21=np.outer(phi(a), phi(a))**2)
    # --- SPLIT representations ---
    res = {'n': n, 'L': L, 'seed': seed, 'split': {}, 'pooled': {}, 'layers': list(range(4, L))}
    for E in E_list:
        for r in r_list:
            key = f'E{E}_r{r}'
            rows = []
            cost = []
            for l in range(4, L):
                Us, Vs, ws = [], [], []
                oldU, oldV, oldw = [], [], []
                for s in sorted(legs):
                    if s >= l: continue
                    age = l - s
                    Lg = legs[s]
                    # legs at layer l must be re-derived: recompute transported legs up to layer l
                    U, V = leg_at(W, st, s, l)
                    keep = np.abs(Lg['alpha']) < kas
                    if age <= E:
                        Us.append(U[:, keep]); Vs.append(V[:, keep]); ws.append(Lg['w'][keep])
                    else:
                        oldU.append(U[:, keep]); oldV.append(V[:, keep]); oldw.append(Lg['w'][keep])
                nex = sum(u.shape[1] for u in Us)
                if oldU:
                    oU = np.concatenate(oldU, axis=1); oV = np.concatenate(oldV, axis=1); ow = np.concatenate(oldw)
                    sw = np.sqrt(ow)[None, :]
                    M = np.concatenate([oU * sw, oV * sw], axis=1)
                    Q, _, _ = np.linalg.svd(M, full_matrices=False)
                    Q = Q[:, :r]
                    oU = Q @ (Q.T @ oU); oV = Q @ (Q.T @ oV)
                    Us.append(oU); Vs.append(oV); ws.append(ow)
                U = np.concatenate(Us, axis=1); V = np.concatenate(Vs, axis=1); w = np.concatenate(ws)
                D3, D21 = readouts(U, V, w)
                F3, F21 = full[l]
                rows.append(dict(l=l, e3w=werr(D3, F3, sens[l]['s3']), e3=werr(D3, F3, np.ones(n)),
                                 e21w=werr(D21, F21, sens[l]['t21']), e21=werr(D21, F21, np.ones((n, n))),
                                 ncols=int(nex), rank=int(r if oldU else 0)))
                cost.append(4.0 * nex + (6.0 * r if oldU else 0.0))   # units of n^2 per layer
            res['split'][key] = dict(rows=rows, cost=float(np.mean(cost)))
    # --- POOLED representations (sequential cap) ---
    for N in N_list:
        for rule in ('w_v2_u', 'w_v2', 'sens'):
            key = f'N{N}_{rule}'
            U = np.zeros((n, 0)); V = np.zeros((n, 0)); w = np.zeros(0)
            rows = []
            for l in range(1, L):
                G = W[l] * st[l - 1]['g'][None, :]
                U = G @ U; V = G @ V
                p = st[l - 1]
                keep = np.abs(p['alpha']) < kas
                U0 = W[l][:, keep]
                V0 = (W[l] @ ((p['g'][:, None] * p['C']) / p['sigma'][None, :]))[:, keep]
                w0 = (p['sigma'] * phi(p['alpha']))[keep]
                U = np.concatenate([U, U0], axis=1); V = np.concatenate([V, V0], axis=1); w = np.concatenate([w, w0])
                if U.shape[1] > N:
                    if rule == 'w_v2_u':
                        I = w * np.sum(V * V, axis=0) * np.sqrt(np.sum(U * U, axis=0))
                    elif rule == 'w_v2':
                        I = w * np.sum(V * V, axis=0)
                    else:
                        s3 = sens[l]['s3']
                        I = w * np.sum((s3[:, None] * np.abs(U)) * V * V, axis=0)
                    idx = np.argsort(-I)[:N]
                    idx.sort()
                    U = U[:, idx]; V = V[:, idx]; w = w[idx]
                if l >= 4:
                    D3, D21 = readouts(U, V, w)
                    F3, F21 = full[l]
                    rows.append(dict(l=l, e3w=werr(D3, F3, sens[l]['s3']), e3=werr(D3, F3, np.ones(n)),
                                     e21w=werr(D21, F21, sens[l]['t21']), e21=werr(D21, F21, np.ones((n, n))),
                                     ncols=int(U.shape[1])))
            res['pooled'][key] = dict(rows=rows, cost=float(4.0 * min(N, max(r['ncols'] for r in rows))))
    return res

_leg_cache = {}
def leg_at(W, st, s, l):
    """legs of birth layer s transported to layer l (cached)."""
    key = (id(W), s, l)
    if key in _leg_cache: return _leg_cache[key]
    if l == s + 1:
        p = st[s]
        U = W[l].copy(); V = W[l] @ ((p['g'][:, None] * p['C']) / p['sigma'][None, :])
    else:
        U0, V0 = leg_at(W, st, s, l - 1)
        G = W[l] * st[l - 1]['g'][None, :]
        U = G @ U0; V = G @ V0
    _leg_cache[key] = (U, V)
    return U, V

if __name__ == '__main__':
    n = int(sys.argv[1]); L = int(sys.argv[2]); seed = int(sys.argv[3]); out = sys.argv[4]
    rr = max(8, int(round(288 * n / 1024)))
    E_list = [6, 4, 8, 2]
    r_list = sorted(set([rr, 2 * rr, rr // 2]))
    N_list = sorted(set([int(round(f * n)) for f in (2.0, 3.0, 3.7, 4.6, 6.0, 8.0)]))
    t0 = time.time()
    res = run(n, L, seed, E_list, r_list, N_list)
    res['time'] = time.time() - t0
    json.dump(res, open(out, 'w'))
    print('done', n, L, seed, 'in %.0fs' % res['time'])
