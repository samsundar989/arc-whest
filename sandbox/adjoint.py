"""
Forward lineages vs backward adjoint for the direct final-layer kappa3 diagonal D3,
under gate-linearised transport and born patterns {diag, pair, hub(order-2 star), triangle}.
Also returns matmul counts.
"""
import numpy as np
from relu_hermite import *
from tower import make_net, run_tower

def k2_chain(W, pmax=40):
    """exact K2 chain; returns per-layer dicts for z_l (l=1..L) with mu, C, sigma, alpha, rho, g, born ingredients."""
    L = len(W); n = W[0].shape[0]
    mu = np.zeros(n); C = W[0]@W[0].T
    layers = []
    for l in range(1, L+1):
        sigma = np.sqrt(np.diag(C)); alpha = mu/sigma; rho = C/np.outer(sigma, sigma)
        A = coefA(mu, sigma, pmax); Bc = coefB(mu, sigma, pmax)
        m, V, k3, k4 = relu_moments(mu, sigma)
        P = pair_series(Bc, A, rho, pmax); np.fill_diagonal(P, 0.0)
        rt = rho.copy(); np.fill_diagonal(rt, 0.0)
        layers.append(dict(mu=mu, C=C, sigma=sigma, alpha=alpha, rho=rt, g=Phi(alpha), d=k3, P=P, A1=A[1], A2=A[2], A3=A[3], m=m))
        if l == L: break
        Cov = cov_relu(mu, sigma, rho, pmax)
        mu = W[l]@m; C = W[l]@Cov@W[l].T
    return layers

def contract_birth(V, lay, with_triangle=False):
    """Given V (n x n, V[i,a] = P_l[a,i]) and the born ingredients of layer lay, return the
    contributions to D3_a from diag, pair, hub(order 2), [triangle]. Counts matmuls."""
    d, P, rt, a1, a2 = lay['d'], lay['P'], lay['rho'], lay['A1'], lay['A2']
    V2 = V*V
    diag = (V2*V).T@d                                   # O(n^2)
    pair = 3*np.sum(V2*(P@V), axis=0)                   # 1 matmul
    S1 = rt@(a1[:, None]*V)                             # 1 matmul
    S2 = (rt*rt)@((a1*a1)[:, None]*V2)                  # 1 matmul
    hub = 3*(np.sum(a2[:, None]*V*S1*S1, axis=0) - np.sum(a2[:, None]*V*S2, axis=0))
    out = dict(diag=diag, pair=pair, hub=hub, matmuls=3)
    if with_triangle:
        # sum_{i,j,k distinct} a2_i a2_j a2_k V_ia V_ja V_ka rho_ij rho_jk rho_ki ; rt has zero diagonal so
        # i!=j, j!=k, k!=i automatically. cost O(n^4): per a, w = a2*V[:,a]; tr((diag(w) rt)^3)
        n = V.shape[0]; tri = np.zeros(n)
        for a in range(n):
            w = a2*V[:, a]; X = w[:, None]*rt
            tri[a] = np.einsum('ij,jk,ki->', X, X, X, optimize=True)
        out['tri'] = tri
    return out

def backward_adjoint(W, layers, with_triangle=False):
    L = len(W); n = W[0].shape[0]
    V = W[L-1].T.copy()          # V_{L-1}: legs of h_{L-1} to z_L
    D3 = {}; mm = 0
    for l in range(L-1, 0, -1):  # birth layer h_l
        D3[l] = contract_birth(V, layers[l-1], with_triangle); mm += D3[l]['matmuls']
        if l > 1:
            V = W[l-1].T@(layers[l-1]['g'][:, None]*V); mm += 1
    return D3, mm

def forward_lineages(W, layers, with_triangle=False):
    L = len(W); n = W[0].shape[0]
    D3 = {}; mm = 0
    for l in range(1, L):
        U = W[l].copy()                                  # legs of h_l at z_{l+1}: U[a,i]
        for k in range(l+1, L):
            U = W[k]@(layers[k-1]['g'][:, None]*U); mm += 1
        D3[l] = contract_birth(U.T, layers[l-1], with_triangle); mm += D3[l]['matmuls']
    return D3, mm

if __name__ == '__main__':
    import time
    n, L = 48, 6
    W = make_net(n, L, 3)
    layers = k2_chain(W)
    t = time.time(); Db, mb = backward_adjoint(W, layers, with_triangle=True); tb = time.time()-t
    t = time.time(); Df, mf = forward_lineages(W, layers, with_triangle=True); tf = time.time()-t
    tot_b = sum(Db[l]['diag']+Db[l]['pair']+Db[l]['hub'] for l in Db)
    tot_f = sum(Df[l]['diag']+Df[l]['pair']+Df[l]['hub'] for l in Df)
    print(f"n={n} L={L}: backward {mb} matmuls, forward {mf} matmuls; max|fwd-bwd| = {np.max(np.abs(tot_f-tot_b)):.2e} (scale {np.sqrt(np.mean(tot_b**2)):.3e})")
    # compare with full-tensor tower (gate transport, born diag+pair+hub order 2, no indirect)
    r = run_tower(W, dict(transport='gate', indirect_mean=False, indirect_cov=False, distinct_maxorder=2, readout_k33=False))
    d3_tower = r['rec'][-1]['d3']
    print(f"full tensor tower D3 vs adjoint (diag+pair+hub): max|diff| = {np.max(np.abs(d3_tower-tot_b)):.2e}")
    for b in sorted(r['rec'][-1]['d3_by_birth']):
        db = r['rec'][-1]['d3_by_birth'][b]; ab = Db[b]['diag']+Db[b]['pair']+Db[b]['hub']
        print(f"   birth {b}: max|diff| {np.max(np.abs(db-ab)):.2e}  rms diag {np.sqrt(np.mean(Db[b]['diag']**2)):.3e} pair {np.sqrt(np.mean(Db[b]['pair']**2)):.3e} hub {np.sqrt(np.mean(Db[b]['hub']**2)):.3e} tri {np.sqrt(np.mean(Db[b]['tri']**2)):.3e}")
    # tower with triangle (maxorder 3 includes (1,1,1) + order-3 stars) for reference
    r3 = run_tower(W, dict(transport='gate', indirect_mean=False, indirect_cov=False, distinct_maxorder=3, readout_k33=False))
    tot_tri = tot_b + sum(Db[l]['tri'] for l in Db)
    print(f"tower maxorder3 D3 - (adjoint+triangle): rms {np.sqrt(np.mean((r3['rec'][-1]['d3']-tot_tri)**2)):.3e} (order-3 star terms), rms D3 {np.sqrt(np.mean(tot_b**2)):.3e}")
