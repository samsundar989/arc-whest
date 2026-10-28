"""Section 7(c): template projections of exact (MC) and tower cumulants; Q4 self-averaging tests.
python3 radial.py n L seed"""
import numpy as np, sys, json
from tower import make_net, run_tower, sym_mC
from relu_hermite import e1_chi, readout_c3, readout_c4, relu_moments, transport_factors, Phi, phi

def proj_pairs(k3p, mu, C):
    """projection of kappa3 (pair entries i!=j and diag) onto Sym(mu (x) C)."""
    n = len(mu); V = np.diag(C); off = ~np.eye(n, dtype=bool)
    S = (2*mu[:, None]*C + mu[None, :]*V[:, None])/3.0           # S[i,j] = Sym_iij
    Sd = mu*V
    eps_p = np.sum((k3p*S)[off])/np.sum((S*S)[off]); cos2_p = np.sum((k3p*S)[off])**2/(np.sum((S*S)[off])*np.sum((k3p*k3p)[off]))
    kd = np.diag(k3p)
    eps_d = np.sum(kd*Sd)/np.sum(Sd*Sd); cos2_d = np.sum(kd*Sd)**2/(np.sum(Sd*Sd)*np.sum(kd*kd))
    return dict(eps_pair=float(eps_p), cos2_pair=float(cos2_p), eps_diag=float(eps_d), cos2_diag=float(cos2_d))

def proj4_pairs(k4p, mu, C):
    n = len(mu); V = np.diag(C); off = ~np.eye(n, dtype=bool)
    T1 = (np.outer(V, V) + 2*C**2)/3.0                           # Sym(CC)_iijj
    T2 = (np.outer(mu**2, V) + np.outer(V, mu**2) + 4*np.outer(mu, mu)*C)/6.0   # Sym(mu mu C)_iijj
    y = k4p[off]; M = np.stack([T1[off], T2[off]], 1)
    coef, *_ = np.linalg.lstsq(M, y, rcond=None)
    eps1 = np.sum(y*T1[off])/np.sum(T1[off]**2); cos2 = np.sum(y*T1[off])**2/(np.sum(T1[off]**2)*np.sum(y*y))
    return dict(eps_CC=float(eps1), cos2_CC=float(cos2), lsq_CC=float(coef[0]), lsq_mmC=float(coef[1]))

def analyse(n, L, seed):
    W = make_net(n, L, seed)
    mc = np.load(f'/home/claude/sandbox/results/mc_n{n}_L{L}_s{seed}.npz')
    out = dict(n=n, L=L, seed=seed, v_radius=float(1-e1_chi(n)**2), e1=float(e1_chi(n)))
    full = run_tower(W, dict(k4=(n <= 64)))
    diag_only = run_tower(W, dict(k4=False, born_pair=False, born_distinct=False, transport='gate'))
    dp = run_tower(W, dict(k4=False, born_distinct=False, transport='gate'))
    rows = []
    for l in range(2, L+1):
        mu, C = mc[f'mu{l}'], mc[f'C{l}']
        pm = proj_pairs(mc[f'k3p{l}'], mu, C)
        p4 = proj4_pairs(mc[f'k4p{l}'], mu, C)
        rt = full['rec'][l-1]; pt = proj_pairs(rt['P3'] + np.diag(rt['d3']), rt['mu'], rt['C'])
        rd = diag_only['rec'][l-1]; pd_ = proj_pairs(rd['P3'] + np.diag(rd['d3']), rd['mu'], rd['C'])
        rp = dp['rec'][l-1]; pp_ = proj_pairs(rp['P3'] + np.diag(rp['d3']), rp['mu'], rp['C'])
        row = dict(l=l, n_eps_pair_mc=n*pm['eps_pair'], cos2_pair_mc=pm['cos2_pair'], n_eps_diag_mc=n*pm['eps_diag'], cos2_diag_mc=pm['cos2_diag'],
                   n_eps_pair_tower=n*pt['eps_pair'], cos2_pair_tower=pt['cos2_pair'], n_eps_all_tower=n*rt['proj3']['eps_all'], cos2_all_tower=rt['proj3']['cos2_all'],
                   n_eps_pair_diagonly=n*pd_['eps_pair'], n_eps_pair_diagpair=n*pp_['eps_pair'],
                   n_eps_by_birth_tower={b: n*e for b, e in rt['proj3']['eps_by_birth'].items()},
                   n_eps4_CC_mc=n*p4['eps_CC'], cos2_4_mc=p4['cos2_CC'], n_lsq4_CC=n*p4['lsq_CC'], n_lsq4_mmC=n*p4['lsq_mmC'])
        if n <= 64:
            p4t = proj4_pairs(rt['K22'], rt['mu'], rt['C']); row['n_eps4_CC_tower'] = n*p4t['eps_CC']; row['cos2_4_tower'] = p4t['cos2_CC']
        rows.append(row)
    out['rows'] = rows
    # ---- Q4: self-averaging of per-birth diagonal kappa3 at each layer: regress on mu*sigma^2 (template) and on richer basis
    sa = []
    for l in range(2, L+1):
        rt = full['rec'][l-1]; mu, C = rt['mu'], rt['C']; sig = np.sqrt(np.diag(C)); al = mu/sig
        X1 = (mu*sig**2)[:, None]
        X2 = np.stack([np.ones(n), mu*sig**2, mu**3, sig**3, mu*sig**2*Phi(al), sig**3*phi(al)], 1)
        for b, d in rt['d3_by_birth'].items():
            def r2(X):
                c, *_ = np.linalg.lstsq(X, d, rcond=None); res = d - X@c
                return float(1 - np.sum(res**2)/np.sum((d-d.mean())**2)), c
            r2a, ca = r2(X1); r2b, cb = r2(X2)
            sa.append(dict(l=l, birth=int(b), age=l-int(b), rms=float(np.sqrt(np.mean(d**2))), R2_template=r2a, n_eps=float(n*ca[0]), R2_basis6=r2b))
        # kappa4 diag self-averaging (mean-field: ~ sigma^4)
        d4 = rt['d4']
        if np.any(d4 != 0):
            X = np.stack([np.ones(n), sig**4, mu**2*sig**2, mu**4], 1); c, *_ = np.linalg.lstsq(X, d4, rcond=None); res = d4 - X@c
            sa.append(dict(l=l, birth=-1, age=-1, rms=float(np.sqrt(np.mean(d4**2))), R2_template=float(1-np.sum(res**2)/np.sum((d4-d4.mean())**2)), note='kappa4 diag on [1,s^4,mu^2 s^2,mu^4]'))
            Xm = np.stack([np.ones(n), sig**4, mu**2*sig**2, mu**4], 1); c, *_ = np.linalg.lstsq(Xm, mc[f'k4d{l}'], rcond=None); res = mc[f'k4d{l}'] - Xm@c
            sa[-1]['R2_mc'] = float(1-np.sum(res**2)/np.sum((mc[f'k4d{l}']-mc[f'k4d{l}'].mean())**2))
    out['self_averaging'] = sa
    json.dump(out, open(f'/home/claude/sandbox/results/radial_n{n}_L{L}_s{seed}.json', 'w'), indent=1)
    return out

if __name__ == '__main__':
    n, L, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    out = analyse(n, L, seed)
    print(f"n={n} L={L} seed={seed}: radius prediction n*eps3 = {3*out['v_radius']*n*2:.3f}(6v n), n*eps4 = {12*out['v_radius']*n:.3f} (12 v n)")
    for r in out['rows']:
        print(f" z_{r['l']}: n*eps3 pair MC {r['n_eps_pair_mc']:.2f} (cos2 {r['cos2_pair_mc']:.2f}) tower {r['n_eps_pair_tower']:.2f} (cos2 {r['cos2_pair_tower']:.2f}) all-entries tower {r['n_eps_all_tower']:.2f} (cos2 {r['cos2_all_tower']:.2f}) | diag-only chain {r['n_eps_pair_diagonly']:.2f} diag+pair {r['n_eps_pair_diagpair']:.2f} | diag MC n*eps {r['n_eps_diag_mc']:.2f} cos2 {r['cos2_diag_mc']:.2f} | k4: n*eps4 MC {r['n_eps4_CC_mc']:.2f} cos2 {r['cos2_4_mc']:.2f} lsq[CC,mmC] {r['n_lsq4_CC']:.2f},{r['n_lsq4_mmC']:.2f}" + (f" tower {r['n_eps4_CC_tower']:.2f}" if 'n_eps4_CC_tower' in r else ''))
    print(" self-averaging (R2 of per-birth diag kappa3 on mu*sigma^2):")
    for s in out['self_averaging']:
        if s['birth'] > 0: print(f"  z_{s['l']} birth {s['birth']} age {s['age']}: rms {s['rms']:.3e} R2(template) {s['R2_template']:.3f} n*eps {s['n_eps']:.2f} R2(6-basis) {s['R2_basis6']:.3f}")
        else: print(f"  z_{s['l']} kappa4 diag: rms {s['rms']:.3e} R2 tower {s['R2_template']:.3f} R2 MC {s.get('R2_mc', float('nan')):.3f}")
