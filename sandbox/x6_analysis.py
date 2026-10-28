"""X6 sandbox, step 2: rank / transport analysis of the covariance error dC_l = C_l(truncated) - C_l(reference).
   python3 x6_analysis.py n L seed trunc [ref=full|mc] [k0=first layer for transport]
Reads results/x6_chains_n{n}_L{L}_s{seed}.npz (from x6_chains.py) and writes results/x6_an_{trunc}_vs_{ref}_n{n}_L{L}_s{seed}.json
and a human-readable .txt with the tables.

Quantities (pre-activation z_l, l = 1..L; post-ReLU h_l, l = 1..L-1; W_{l+1} maps h_l -> z_{l+1}):
  dC_l      = C_l^tr - C_l^ref                (n x n, symmetric),  dvar_l = diag(dC_l)
  dCovh_l   = Covh_l^tr - Covh_l^ref          (exact post-ReLU covariance error actually propagated by the chain)
  dC_{l+1}  = W dCovh_l W^T  exactly;  split dCovh_l = D_l (diag) + O_l (off-diag):
  dvar_{l+1} = (W*W) diag(D_l)  [diagonal channel, X1]  +  diag(W O_l W^T)  [off-diagonal channel, X6 target]
"""
import numpy as np, sys, json, time, os
from tower import make_net
from relu_hermite import transport_factors, cov_relu, phi, Phi

KS = [1, 2, 4, 8, 16, 32, 64, 128]

def eig_sorted(M):
    lam, U = np.linalg.eigh(0.5*(M + M.T))
    o = np.argsort(-np.abs(lam))
    return lam[o], U[:, o]

def frob_fractions(lam, ks):
    e = lam**2; tot = e.sum()
    if tot == 0: return {k: float('nan') for k in ks}
    c = np.cumsum(e)
    return {k: float(c[min(k, len(lam))-1]/tot) for k in ks}

def participation_ratio(lam):
    e = lam**2
    return float(e.sum()**2/np.sum(e**2)) if e.sum() > 0 else float('nan')

def r2(target, approx):
    tt = np.sum(target**2)
    return float(1.0 - np.sum((target-approx)**2)/tt) if tt > 0 else float('nan')

def sandwich_r2(lam, U, W, target, ks, remove_diag=False):
    """R^2 of diag(W T_k W^T) against target, T_k = rank-k eigen-truncation (sorted by |lambda|)."""
    WU = W @ U                                   # n x n, column j = W u_j
    contrib = lam[None, :]*WU**2                 # n x n: contribution of eigenpair j to each output i
    if remove_diag:
        W2 = W*W
        contrib = contrib - W2 @ (lam[None, :]*U**2)   # remove (W*W) diag(T_k) part
    c = np.cumsum(contrib, axis=1)
    return {k: r2(target, c[:, min(k, U.shape[1])-1]) for k in ks}

def goe_like(n, rng, zero_diag=True):
    R = rng.standard_normal((n, n)); R = (R + R.T)/np.sqrt(2)
    if zero_diag: np.fill_diagonal(R, 0.0)
    return R

def sign_scramble(M, rng):
    S = np.sign(rng.standard_normal(M.shape)); S = np.triu(S, 1); S = S + S.T + np.eye(M.shape[0])
    return M*S

def ols_r2(y, X):
    X1 = np.column_stack([np.ones(len(y))] + [x for x in X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return r2(y, X1 @ beta)

def analyse(n, L, seed, trunc, ref='full', l0=None, nsur=3, verbose=True):
    rng = np.random.default_rng(0)
    W = make_net(n, L, seed)
    d = np.load(f'/home/claude/sandbox/results/x6_chains_n{n}_L{L}_s{seed}.npz')
    ks = [k for k in KS if k <= n]
    mc = None
    if ref == 'mc':
        mc = np.load(f'/home/claude/sandbox/results/mc_n{n}_L{L}_s{seed}.npz')
    C_ref = {l: (mc[f'C{l}'] if ref == 'mc' else d[f'full_C{l}']) for l in range(1, L+1)}
    mu_ref = {l: (mc[f'mu{l}'] if ref == 'mc' else d[f'full_mu{l}']) for l in range(1, L+1)}
    C_tr = {l: d[f'{trunc}_C{l}'] for l in range(1, L+1)}
    dC = {l: C_tr[l] - C_ref[l] for l in range(1, L+1)}
    # post-ReLU covariance error (tower reference only; for MC ref use the gate-linearised G dC G)
    gates = {}; T2s = {}
    for l in range(1, L):
        sig = np.sqrt(np.diag(C_ref[l])); g, T2, *_ = transport_factors(mu_ref[l], sig)
        gates[l] = g; T2s[l] = T2
    if ref == 'full':
        dCovh = {l: d[f'{trunc}_Covh{l}'] - d[f'full_Covh{l}'] for l in range(1, L)}
        # sanity: W dCovh W^T == dC_{l+1}
        for l in range(1, L):
            err = np.abs(W[l] @ dCovh[l] @ W[l].T - dC[l+1]).max()
            assert err < 1e-10*max(1.0, np.abs(dC[l+1]).max()), (l, err)
    else:
        dCovh = {l: gates[l][:, None]*dC[l]*gates[l][None, :] for l in range(1, L)}
    out = dict(n=n, L=L, seed=seed, trunc=trunc, ref=ref, ks=ks, layers={})
    if ref == 'full':
        out['mse_tr'] = float(d[f'{trunc}_mse']) if f'{trunc}_mse' in d else None
        out['mse_full'] = float(d['full_mse']) if 'full_mse' in d else None
    txt = []
    def P(s=''):
        txt.append(s)
        if verbose: print(s, flush=True)
    P(f'# X6 analysis  n={n} L={L} seed={seed}  truncated chain = {trunc}  reference = {ref}')
    if out.get('mse_tr') and out.get('mse_full'):
        P(f'MSE vs MC: {trunc} {out["mse_tr"]:.3e}  full {out["mse_full"]:.3e}  ratio {out["mse_tr"]/out["mse_full"]:.3f}')
    if ref == 'mc':
        N = float(mc['N'])
    # ------------------------------------------------------------------ per-layer basic sizes
    P('\n## A. sizes.  rms(dvar_l) = rms of diag(dC_l); split of the energy of dvar_{l+1} into diagonal channel (W*W)dVarh_l and off-diagonal channel diag(W O_l W^T)')
    P('l   rms(var_l)  rms(dvar_l)  ||dC_l||_F  rms(offdiag dC_l)  | next-layer split of ||dvar_{l+1}||^2: diag-ch  off-ch  cross   [noise energy frac if MC]')
    for l in range(1, L+1):
        var = np.diag(C_ref[l]); dv = np.diag(dC[l]); off = dC[l] - np.diag(dv)
        row = dict(rms_var=float(np.sqrt(np.mean(var**2))), rms_dvar=float(np.sqrt(np.mean(dv**2))),
                   frob_dC=float(np.linalg.norm(dC[l])), rms_off=float(np.sqrt(np.sum(off**2)/(n*(n-1)))))
        s = f'{l:<3d} {row["rms_var"]:9.3e}  {row["rms_dvar"]:9.3e}  {row["frob_dC"]:9.3e}  {row["rms_off"]:9.3e}'
        if l < L:
            Dl = np.diag(np.diag(dCovh[l])); Ol = dCovh[l] - Dl
            fd = (W[l]*W[l]) @ np.diag(Dl); fo = np.einsum('ij,jk,ik->i', W[l], Ol, W[l])
            tot = np.sum((fd+fo)**2)
            row.update(diag_ch=float(np.sum(fd**2)/tot) if tot > 0 else float('nan'),
                       off_ch=float(np.sum(fo**2)/tot) if tot > 0 else float('nan'),
                       cross=float(2*np.sum(fd*fo)/tot) if tot > 0 else float('nan'))
            s += f'   | {row["diag_ch"]:6.3f}  {row["off_ch"]:6.3f}  {row["cross"]:+6.3f}'
        if ref == 'mc':
            noise = (np.outer(var, var) + C_ref[l]**2)/N
            row['noise_frac'] = float(noise.sum()/max(np.sum(dC[l]**2), 1e-300))
            s += f'   noise/|dC|^2 = {row["noise_frac"]:.3f}'
        out['layers'][l] = row
        P(s)
    # ------------------------------------------------------------------ B. spectrum / rank of O_l and sandwich R^2
    P('\n## B. rank of the off-diagonal post-ReLU covariance error O_l and the fraction of the induced next-layer variance error it reproduces')
    P('For each l: eigen-decompose O_l (sorted by |lambda|).  frobK = fraction of ||O_l||_F^2 in top-k;  R2K = R^2 of diag(W T_k W^T) against dvar^off_{l+1} = diag(W O_l W^T)')
    P('(R2K-nd = same with the diagonal of T_k removed, i.e. pure off-diagonal readout).  GOE / scr = same numbers for a GOE matrix / sign-scrambled O_l (coherence baseline).')
    hdr = 'l   PR    ' + ' '.join(f'fro{k:<3d}' for k in ks) + ' | ' + ' '.join(f'R2_{k:<3d}' for k in ks) + ' | ' + ' '.join(f'nd{k:<3d}' for k in ks)
    P(hdr)
    for l in range(1, L):
        Ol = dCovh[l] - np.diag(np.diag(dCovh[l]))
        if np.linalg.norm(Ol) == 0:
            out['layers'][l]['B'] = None; P(f'{l:<3d}  (dC = 0)'); continue
        target = np.einsum('ij,jk,ik->i', W[l], Ol, W[l])
        lam, U = eig_sorted(Ol)
        fr = frob_fractions(lam, ks); R2 = sandwich_r2(lam, U, W[l], target, ks); R2nd = sandwich_r2(lam, U, W[l], target, ks, remove_diag=True)
        pr = participation_ratio(lam)
        # surrogates
        sur = dict(goe_fro={k: 0.0 for k in ks}, goe_R2={k: 0.0 for k in ks}, goe_PR=0.0, scr_fro={k: 0.0 for k in ks}, scr_R2={k: 0.0 for k in ks}, scr_PR=0.0)
        for s_ in range(nsur):
            G = goe_like(n, rng); G *= np.linalg.norm(Ol)/np.linalg.norm(G)
            lg, Ug = eig_sorted(G); tg = np.einsum('ij,jk,ik->i', W[l], G, W[l])
            f_ = frob_fractions(lg, ks); r_ = sandwich_r2(lg, Ug, W[l], tg, ks)
            for k in ks: sur['goe_fro'][k] += f_[k]/nsur; sur['goe_R2'][k] += r_[k]/nsur
            sur['goe_PR'] += participation_ratio(lg)/nsur
            S = sign_scramble(Ol, rng); ls, Us = eig_sorted(S); ts = np.einsum('ij,jk,ik->i', W[l], S, W[l])
            f_ = frob_fractions(ls, ks); r_ = sandwich_r2(ls, Us, W[l], ts, ks)
            for k in ks: sur['scr_fro'][k] += f_[k]/nsur; sur['scr_R2'][k] += r_[k]/nsur
            sur['scr_PR'] += participation_ratio(ls)/nsur
        out['layers'][l]['B'] = dict(PR=pr, fro=fr, R2=R2, R2nd=R2nd, top_eigs=[float(x) for x in lam[:8]], **sur)
        P(f'{l:<3d} {pr:5.1f} ' + ' '.join(f'{fr[k]:6.3f}' for k in ks) + ' | ' + ' '.join(f'{R2[k]:6.3f}' for k in ks) + ' | ' + ' '.join(f'{R2nd[k]:6.3f}' for k in ks))
        P(f'    GOE {sur["goe_PR"]:5.1f} ' + ' '.join(f'{sur["goe_fro"][k]:6.3f}' for k in ks) + ' | ' + ' '.join(f'{sur["goe_R2"][k]:6.3f}' for k in ks))
        P(f'    scr {sur["scr_PR"]:5.1f} ' + ' '.join(f'{sur["scr_fro"][k]:6.3f}' for k in ks) + ' | ' + ' '.join(f'{sur["scr_R2"][k]:6.3f}' for k in ks))
    # same for the pre-activation dC_l (what a state in z-space would have to represent), Frobenius only + diag R^2 (readout = diag of truncation vs dvar_l)
    P('\n## B2. pre-activation dC_l itself: frobK and R^2 of diag(T_k) against dvar_l = diag(dC_l) (the in-layer quadratic readout of a rank-k state that represents dC_l)')
    P('l   PR    ' + ' '.join(f'fro{k:<3d}' for k in ks) + ' | ' + ' '.join(f'R2_{k:<3d}' for k in ks))
    for l in range(1, L+1):
        if np.linalg.norm(dC[l]) == 0: continue
        lam, U = eig_sorted(dC[l]); fr = frob_fractions(lam, ks); pr = participation_ratio(lam)
        dv = np.diag(dC[l]); contrib = np.cumsum(lam[None, :]*U**2, axis=1)
        R2 = {k: r2(dv, contrib[:, min(k, n)-1]) for k in ks}
        out['layers'][l]['B2'] = dict(PR=pr, fro=fr, R2=R2)
        P(f'{l:<3d} {pr:5.1f} ' + ' '.join(f'{fr[k]:6.3f}' for k in ks) + ' | ' + ' '.join(f'{R2[k]:6.3f}' for k in ks))
    # ------------------------------------------------------------------ C. transport persistence
    nz = [l for l in range(1, L+1) if np.linalg.norm(dC[l]) > 0]
    if l0 is None: l0 = nz[0] if nz else None
    if l0 is not None and l0 < L:
        P(f'\n## C. transport persistence from l0 = {l0}: top-k eigenpairs of dC_l0 transported as U <- W_(l+1) (gate_l * U) (and sqrtT2);')
        P('    pure = R^2 of diag(U_l Lam U_l^T) on dvar_l (no innovation);  span = ||Q^T dC_l Q||_F^2/||dC_l||_F^2 (Q = orth(U_l));  core = R^2 of diag(Q (Q^T dC_l Q) Q^T) on dvar_l (oracle k x k core refit);')
        P('    fullT = R^2 on dvar_l of the FULL-RANK transported dC_l0 (gate-linearised, no innovation) = how much of the later error is transported at all;  eigK = R^2 of the in-layer top-k truncation of dC_l (ceiling from B2).')
        lam0, U0 = eig_sorted(dC[l0])
        for gate_name in ['gate', 'sqrtT2']:
            P(f'  transport = {gate_name}')
            P('  l   ' + ' '.join(f'pure{k:<3d}' for k in [4, 16, 32, 64] if k <= n) + ' | ' + ' '.join(f'span{k:<3d}' for k in [4, 16, 32, 64] if k <= n)
              + ' | ' + ' '.join(f'core{k:<3d}' for k in [4, 16, 32, 64] if k <= n) + ' |  fullT   ' + ' '.join(f'eig{k:<3d}' for k in [16, 32] if k <= n))
            Us = {k: U0[:, :k].copy() for k in [4, 16, 32, 64] if k <= n}
            Sfull = dC[l0].copy()
            rowsC = {}
            for l in range(l0+1, L+1):
                f = gates[l-1] if gate_name == 'gate' else np.sqrt(np.maximum(T2s[l-1], 0))
                for k in Us: Us[k] = W[l-1] @ (f[:, None]*Us[k])
                Sfull = W[l-1] @ (np.outer(f, f)*Sfull) @ W[l-1].T
                dv = np.diag(dC[l]); nrm = np.linalg.norm(dC[l])**2
                rr = dict(pure={}, span={}, core={})
                for k, Uk in Us.items():
                    rr['pure'][k] = r2(dv, np.sum(lam0[None, :k]*Uk**2, axis=1))
                    Q, _ = np.linalg.qr(Uk)
                    core = Q.T @ dC[l] @ Q
                    rr['span'][k] = float(np.sum(core**2)/nrm) if nrm > 0 else float('nan')
                    rr['core'][k] = r2(dv, np.einsum('ik,kl,il->i', Q, core, Q))
                rr['fullT'] = r2(dv, np.diag(Sfull))
                rr['eig'] = {k: out['layers'][l]['B2']['R2'][k] for k in [16, 32] if k <= n} if 'B2' in out['layers'][l] else {}
                rowsC[l] = rr
                P(f'  {l:<3d} ' + ' '.join(f'{rr["pure"][k]:7.3f}' for k in Us) + ' | ' + ' '.join(f'{rr["span"][k]:7.3f}' for k in Us)
                  + ' | ' + ' '.join(f'{rr["core"][k]:7.3f}' for k in Us) + ' | ' + f'{rr["fullT"]:7.3f}  ' + ' '.join(f'{rr["eig"].get(k, float("nan")):6.3f}' for k in [16, 32] if k <= n))
            out[f'C_{gate_name}'] = {str(l): rr for l, rr in rowsC.items()}
    # ------------------------------------------------------------------ D. innovation rank + oracle-innovation rank-k state
    P('\n## D. innovation I_l = dC_l - W G dC_{l-1} G W^T (gate-linearised transport of the previous error; Iex = exact one-step propagation through cov_relu at mu_ref).')
    P('    |I|/|dC| = ||I_l||_F^2/||dC_l||_F^2;  diag share = ||diag I_l||^2/||dvar_l||^2;  R2tr = R^2 of diag(transported dC_{l-1}) on dvar_l (= 1 - residual, the "transported share");  PR, frobK of I_l; sandR2K = R^2 of rank-k I_l sandwich at l+1')
    P('l   |I|/|dC|  |Iex|/|dC|  diagI/dvar  R2tr   R2tr_ex   PR_I  ' + ' '.join(f'fro{k:<3d}' for k in [4, 16, 32, 64] if k <= n) + ' | ' + ' '.join(f'sR2_{k:<3d}' for k in [4, 16, 32, 64] if k <= n) + '   GOE_PR')
    Iinn = {}
    for l in range(2, L+1):
        if np.linalg.norm(dC[l]) == 0: continue
        g = gates[l-1]
        T_lin = W[l-1] @ (np.outer(g, g)*dC[l-1]) @ W[l-1].T
        sig = np.sqrt(np.diag(C_ref[l-1])); rho = C_ref[l-1]/np.outer(sig, sig)
        Ct = C_ref[l-1] + dC[l-1]; sigt = np.sqrt(np.diag(Ct)); rhot = Ct/np.outer(sigt, sigt)
        T_ex = W[l-1] @ (cov_relu(mu_ref[l-1], sigt, rhot, 40) - cov_relu(mu_ref[l-1], sig, rho, 40)) @ W[l-1].T
        I = dC[l] - T_lin; Iex = dC[l] - T_ex; Iinn[l] = I
        dv = np.diag(dC[l]); nrm = np.linalg.norm(dC[l])**2
        lam, U = eig_sorted(I); fr = frob_fractions(lam, [4, 16, 32, 64]); pr = participation_ratio(lam)
        row = dict(I_frac=float(np.linalg.norm(I)**2/nrm), Iex_frac=float(np.linalg.norm(Iex)**2/nrm), diagI_frac=float(np.sum(np.diag(I)**2)/np.sum(dv**2)),
                   R2tr=r2(dv, np.diag(T_lin)), R2tr_ex=r2(dv, np.diag(T_ex)), PR=pr, fro=fr)
        sR2 = {}
        if l < L:
            Gi = gates[l]; Ig = Gi[:, None]*I*Gi[None, :]
            tgt = np.einsum('ij,jk,ik->i', W[l], Ig, W[l]); lg, Ug = eig_sorted(Ig)
            sR2 = sandwich_r2(lg, Ug, W[l], tgt, [4, 16, 32, 64])
        row['sandR2'] = sR2
        G_ = goe_like(n, rng, zero_diag=False); gpr = participation_ratio(eig_sorted(G_)[0])
        out['layers'][l]['D'] = row
        P(f'{l:<3d} {row["I_frac"]:8.3f}  {row["Iex_frac"]:9.3f}  {row["diagI_frac"]:9.3f}  {row["R2tr"]:6.3f}  {row["R2tr_ex"]:7.3f}  {pr:5.1f}  '
          + ' '.join(f'{fr[k]:6.3f}' for k in fr) + ' | ' + ' '.join(f'{sR2.get(k, float("nan")):7.3f}' for k in [4, 16, 32, 64] if k <= n) + f'   {gpr:5.1f}')
    # oracle innovation, rank-k state bottleneck: S_l = T_k[ dC_l - W G (dC_{l-1} - S_{l-1}) G W^T ]
    P('\n## D2. oracle-innovation, rank-k state: S_l = Trunc_k[ transport(S_{l-1}) + I_l ]  (perfect per-layer innovation, but the carried state is re-truncated to rank k every layer).')
    P('    R^2 of diag(S_l) on dvar_l.  This is the ceiling of what an X6 state of rank k can reproduce of the variance error, whatever the MLP learns.')
    P('l   ' + ' '.join(f'k={k:<4d}' for k in [4, 8, 16, 32, 64, 128] if k <= n))
    kk = [k for k in [4, 8, 16, 32, 64, 128] if k <= n]
    S = {k: np.zeros((n, n)) for k in kk}
    rowsD2 = {}
    for l in range(1, L+1):
        if l > 1:
            g = gates[l-1]
            for k in kk:
                M = dC[l] - W[l-1] @ (np.outer(g, g)*(dC[l-1] - S[k])) @ W[l-1].T
                lam, U = eig_sorted(M); S[k] = (U[:, :k]*lam[None, :k]) @ U[:, :k].T
        else:
            for k in kk:
                lam, U = eig_sorted(dC[l]); S[k] = (U[:, :k]*lam[None, :k]) @ U[:, :k].T
        if np.linalg.norm(dC[l]) == 0: continue
        dv = np.diag(dC[l]); rr = {k: r2(dv, np.diag(S[k])) for k in kk}
        rowsD2[l] = rr
        P(f'{l:<3d} ' + ' '.join(f'{rr[k]:6.3f}' for k in kk))
    out['D2'] = {str(l): rr for l, rr in rowsD2.items()}
    # ------------------------------------------------------------------ E. coherence: what the diagonal channel / local features explain
    P('\n## E. what per-neuron / diagonal messages explain of dvar_{l+1} (OLS R^2 with intercept):')
    P('    fdiag = (W*W) dVarh_l (exact diagonal channel);  g2 = (W*W)(g_l^2 dvar_l) (X1-type message from the pre-act diagonal error);  loc = [var, mu, mu^2, g, phi] at l+1;  all = fdiag+loc')
    P('l+1  R2(fdiag)  R2(g2)  R2(loc)  R2(all)   | off-channel share of ||dvar_{l+1}||^2  (from A)')
    for l in range(1, L):
        if np.linalg.norm(dC[l+1]) == 0: continue
        dv = np.diag(dC[l+1]); W2 = W[l]*W[l]
        fdiag = W2 @ np.diag(dCovh[l]); g2 = W2 @ (gates[l]**2*np.diag(dC[l]))
        var = np.diag(C_ref[l+1]); mu = mu_ref[l+1]; a = mu/np.sqrt(var)
        loc = [var, mu, mu**2, Phi(a), phi(a)]
        row = dict(R2_fdiag=ols_r2(dv, [fdiag]), R2_g2=ols_r2(dv, [g2]), R2_loc=ols_r2(dv, loc), R2_all=ols_r2(dv, [fdiag]+loc))
        out['layers'][l+1]['E'] = row
        P(f'{l+1:<4d} {row["R2_fdiag"]:8.3f}  {row["R2_g2"]:7.3f}  {row["R2_loc"]:7.3f}  {row["R2_all"]:7.3f}   | {out["layers"][l]["off_ch"]:6.3f}')
    # ------------------------------------------------------------------ F. coherent (template) part vs incoherent remainder
    P('\n## F. coherent part of O_l: least-squares fit of the off-diagonal entries of O_l on the templates m_h m_h^T and Covh_ref (scale-mixture directions);')
    P('    cos2(u1,m_h) = squared cosine of the top eigenvector of O_l with the post-ReLU mean;  tpl = fraction of ||O_l||_F^2 explained by the 2 templates;')
    P('    remainder R_l = O_l - fit:  offE = ||diag(W R W^T)||^2/||diag(W O W^T)||^2 (share of the off-channel variance error that is NOT template);  PR, fro16/32, sandwich R2_16/32 of R_l, with GOE/scrambled baselines.')
    P('l   cos2(u1,m_h) cos2(u1,mu)  tpl   a_mm      b_C     | offE   PR_R  fro16  fro32  R2_16  R2_32 | GOE16 GOE32 | scr16 scr32')
    for l in range(1, L):
        Ol = dCovh[l] - np.diag(np.diag(dCovh[l]))
        if np.linalg.norm(Ol) == 0: continue
        mh = d[f'full_mh{l}'] if ref == 'full' else None
        if mh is None:
            sig = np.sqrt(np.diag(C_ref[l])); a = mu_ref[l]/sig; mh = sig*phi(a) + mu_ref[l]*Phi(a)
        Covh_ref = d[f'full_Covh{l}'] if ref == 'full' else gates[l][:, None]*C_ref[l]*gates[l][None, :]
        T1 = np.outer(mh, mh); T2 = Covh_ref.copy(); np.fill_diagonal(T1, 0.0); np.fill_diagonal(T2, 0.0)
        iu = np.triu_indices(n, 1)
        X = np.column_stack([T1[iu], T2[iu]]); coef, *_ = np.linalg.lstsq(X, Ol[iu], rcond=None)
        fit = coef[0]*T1 + coef[1]*T2; R = Ol - fit
        tpl = float(1 - np.sum(R**2)/np.sum(Ol**2))
        lam, U = eig_sorted(Ol); u1 = U[:, 0]
        c_mh = float((u1 @ mh)**2/(mh @ mh)); c_mu = float((u1 @ mu_ref[l])**2/(mu_ref[l] @ mu_ref[l]))
        tO = np.einsum('ij,jk,ik->i', W[l], Ol, W[l]); tR = np.einsum('ij,jk,ik->i', W[l], R, W[l])
        offE = float(np.sum(tR**2)/np.sum(tO**2))
        lr, Ur = eig_sorted(R); frR = frob_fractions(lr, [16, 32]); R2R = sandwich_r2(lr, Ur, W[l], tR, [16, 32]); prR = participation_ratio(lr)
        goe = {16: 0.0, 32: 0.0}; scr = {16: 0.0, 32: 0.0}
        for s_ in range(nsur):
            G = goe_like(n, rng); lg, Ug = eig_sorted(G); r_ = sandwich_r2(lg, Ug, W[l], np.einsum('ij,jk,ik->i', W[l], G, W[l]), [16, 32])
            S = sign_scramble(R, rng); ls, Us = eig_sorted(S); r2_ = sandwich_r2(ls, Us, W[l], np.einsum('ij,jk,ik->i', W[l], S, W[l]), [16, 32])
            for k in goe: goe[k] += r_[k]/nsur; scr[k] += r2_[k]/nsur
        out['layers'][l]['F'] = dict(cos2_u1_mh=c_mh, cos2_u1_mu=c_mu, tpl=tpl, a_mm=float(coef[0]), b_C=float(coef[1]), offE=offE, PR_R=prR, fro=frR, R2=R2R, goe_R2=goe, scr_R2=scr)
        P(f'{l:<3d}   {c_mh:6.3f}      {c_mu:6.3f}    {tpl:6.3f} {coef[0]:+9.2e} {coef[1]:+9.2e} | {offE:6.3f} {prR:5.1f}  {frR[16]:.3f}  {frR[32]:.3f}  {R2R[16]:.3f}  {R2R[32]:.3f} | {goe[16]:.3f} {goe[32]:.3f} | {scr[16]:.3f} {scr[32]:.3f}')
    # ------------------------------------------------------------------ G. one-step persistence of the error directions
    P('\n## G. one-step persistence: Q = orth( W G U_k(l-1) ) with U_k(l-1) = top-k eigenvectors of dC_{l-1} (oracle state at l-1, transported one layer).')
    P('    span1 = ||Q^T dC_l Q||_F^2/||dC_l||_F^2 ;  core1 = R^2 on dvar_l of diag(Q (Q^T dC_l Q) Q^T) ;  fro_k(l) = in-layer ceiling.  If span1 << fro_k the error directions are renewed every layer.')
    P('l   ' + ' '.join(f'span1_{k:<3d}' for k in [4, 16, 32] if k <= n) + ' | ' + ' '.join(f'core1_{k:<3d}' for k in [4, 16, 32] if k <= n) + ' | ' + ' '.join(f'fro_{k:<3d}' for k in [4, 16, 32] if k <= n))
    for l in range(2, L+1):
        if np.linalg.norm(dC[l]) == 0 or np.linalg.norm(dC[l-1]) == 0: continue
        lamp, Up = eig_sorted(dC[l-1]); g = gates[l-1]; dv = np.diag(dC[l]); nrm = np.linalg.norm(dC[l])**2
        lamc, Uc = eig_sorted(dC[l]); frc = frob_fractions(lamc, [4, 16, 32])
        row = dict(span1={}, core1={}, fro=frc)
        for k in [4, 16, 32]:
            if k > n: continue
            Q, _ = np.linalg.qr(W[l-1] @ (g[:, None]*Up[:, :k])); core = Q.T @ dC[l] @ Q
            row['span1'][k] = float(np.sum(core**2)/nrm); row['core1'][k] = r2(dv, np.einsum('ik,kl,il->i', Q, core, Q))
        out['layers'][l]['G'] = row
        P(f'{l:<3d} ' + ' '.join(f'{row["span1"][k]:9.3f}' for k in row['span1']) + ' | ' + ' '.join(f'{row["core1"][k]:9.3f}' for k in row['core1']) + ' | ' + ' '.join(f'{frc[k]:7.3f}' for k in row['span1']))
    # ------------------------------------------------------------------ H. local-feature-generated innovations (what an X6 MLP could emit) and their transported span
    P('\n## H. can the error be built from LOCAL vectors?  F_l = per-neuron feature matrix at z_l (p columns: 1, mu, sig2, mu^2, g, phi, m_h, V_h, alpha*phi, mu*m_h);')
    P('    loc  = fraction of ||M||_F^2 in the two-sided projection P_F M P_F (M = dC_l, and M = I_l the innovation) and R^2 of its diagonal on diag(M);')
    P('    tloc = same with the TRANSPORTED-LOCAL span F_aug(l) = [F_l, WG F_{l-1}, WGWG F_{l-2}, ...] (the span an X6 state can reach if every layer\'s innovation is a local vector; dimension p*(l-1));')
    P('    rnd  = random subspace of the same dimension (null).   p=4 uses (1, mu, sig2, m_h) only.')
    P('    win = transported-local span restricted to the last 6 layers (dimension <= 6p), wrnd = random null of that dimension.')
    P('l   dim4 dim10 | dC: loc4  loc10  tloc4  tloc10  rnd4   rnd10 | win4   win10  wrnd4  wrnd10 | dC diag-R2: loc10 tloc4 tloc10 win10 | I_l: loc10  tloc10 | fro_16(dC)')
    def feats(l, p):
        sig = np.sqrt(np.diag(C_ref[l])); mu = mu_ref[l]; a = mu/sig; g = Phi(a); ph = phi(a)
        mh = sig*ph + mu*g; Vh = (mu**2 + sig**2)*g + mu*sig*ph - mh**2
        F = np.column_stack([np.ones(n), mu, sig**2, mh]) if p == 4 else np.column_stack([np.ones(n), mu, sig**2, mu**2, g, ph, mh, Vh, a*ph, mu*mh])
        nr = np.linalg.norm(F, axis=0); F = F[:, nr > 1e-12]
        return F/np.linalg.norm(F, axis=0)
    def proj_frac(M, Q):
        core = Q.T @ M @ Q
        return float(np.sum(core**2)/np.sum(M**2)), r2(np.diag(M), np.einsum('ik,kl,il->i', Q, core, Q))
    Faug = {4: None, 10: None}; Fwin = {4: [], 10: []}
    for l in range(1, L+1):
        for p in (4, 10):
            F = feats(l, p)
            if Faug[p] is None: Faug[p] = F
            else:
                g = gates[l-1]; Faug[p] = np.column_stack([F, W[l-1] @ (g[:, None]*Faug[p])])
                Fwin[p] = [W[l-1] @ (g[:, None]*X) for X in Fwin[p]]
            Fwin[p] = ([F] + Fwin[p])[:6]
        if np.linalg.norm(dC[l]) == 0: continue
        row = {}
        s = f'{l:<3d} {Faug[4].shape[1]:4d} {Faug[10].shape[1]:5d} |'
        res = {}
        for p in (4, 10):
            Q, _ = np.linalg.qr(feats(l, p)); res[f'loc{p}'] = proj_frac(dC[l], Q)
            Qa, _ = np.linalg.qr(Faug[p]); Qa = Qa[:, :min(Qa.shape[1], n)]; res[f'tloc{p}'] = proj_frac(dC[l], Qa)
            Qr, _ = np.linalg.qr(rng.standard_normal((n, min(Faug[p].shape[1], n)))); res[f'rnd{p}'] = proj_frac(dC[l], Qr)
            Fw = np.column_stack(Fwin[p]); Qw, _ = np.linalg.qr(Fw); res[f'win{p}'] = proj_frac(dC[l], Qw)
            Qwr, _ = np.linalg.qr(rng.standard_normal((n, min(Fw.shape[1], n)))); res[f'wrnd{p}'] = proj_frac(dC[l], Qwr)
            if l in Iinn:
                res[f'I_loc{p}'] = proj_frac(Iinn[l], Q); res[f'I_tloc{p}'] = proj_frac(Iinn[l], Qa)
        out['layers'][l]['H'] = {k: list(v) for k, v in res.items()}
        s += f' {res["loc4"][0]:6.3f} {res["loc10"][0]:6.3f} {res["tloc4"][0]:6.3f} {res["tloc10"][0]:6.3f} {res["rnd4"][0]:6.3f} {res["rnd10"][0]:6.3f} |'
        s += f' {res["win4"][0]:6.3f} {res["win10"][0]:6.3f} {res["wrnd4"][0]:6.3f} {res["wrnd10"][0]:6.3f} |'
        s += f' {res["loc10"][1]:6.3f} {res["tloc4"][1]:6.3f} {res["tloc10"][1]:6.3f} {res["win10"][1]:6.3f} |'
        if l in Iinn: s += f' {res["I_loc10"][0]:6.3f} {res["I_tloc10"][0]:6.3f} |'
        else: s += '    -      -   |'
        s += f' {out["layers"][l]["B2"]["fro"][16]:6.3f}' if 'B2' in out['layers'][l] else ''
        P(s)
    # ------------------------------------------------------------------ D3. rank-k state with LOCAL-BORN innovations (oracle p x p core)
    P('\n## D3. X6-like ceiling with local-born innovations: S_l = Trunc_k[ transport(S_{l-1}) + P_F I_l P_F ]  (innovation restricted to the span of the p=10 local')
    P('    feature vectors at layer l with an oracle p x p core; state re-truncated to rank k).  R^2 of diag(S_l) on dvar_l; compare with D2 (unrestricted oracle innovation).')
    kk3 = [k for k in [8, 16, 32, 64] if k <= n]
    P('l   ' + ' '.join(f'k={k:<4d}' for k in kk3) + ' | D2: ' + ' '.join(f'k={k:<4d}' for k in kk3))
    S3 = {k: np.zeros((n, n)) for k in kk3}
    rowsD3 = {}
    for l in range(1, L+1):
        Fq, _ = np.linalg.qr(feats(l, 10))
        if l > 1:
            g = gates[l-1]
            for k in kk3:
                T = W[l-1] @ (np.outer(g, g)*S3[k]) @ W[l-1].T
                I = dC[l] - W[l-1] @ (np.outer(g, g)*dC[l-1]) @ W[l-1].T
                M = T + Fq @ (Fq.T @ I @ Fq) @ Fq.T
                lam, U = eig_sorted(M); S3[k] = (U[:, :k]*lam[None, :k]) @ U[:, :k].T
        else:
            for k in kk3:
                M = Fq @ (Fq.T @ dC[l] @ Fq) @ Fq.T
                lam, U = eig_sorted(M); S3[k] = (U[:, :k]*lam[None, :k]) @ U[:, :k].T
        if np.linalg.norm(dC[l]) == 0: continue
        dv = np.diag(dC[l]); rr = {k: r2(dv, np.diag(S3[k])) for k in kk3}
        rowsD3[l] = rr
        P(f'{l:<3d} ' + ' '.join(f'{rr[k]:6.3f}' for k in kk3) + ' |     ' + ' '.join(f'{out["D2"].get(str(l), {}).get(k, float("nan")):6.3f}' for k in kk3))
    out['D3'] = {str(l): rr for l, rr in rowsD3.items()}
    # ------------------------------------------------------------------ D4. rank-k state whose NEW directions can only enter through local vectors (oracle core)
    P('\n## D4. X6 direction ceiling: S_l = Trunc_k[ Q Q^T dC_l Q Q^T ],  Q = orth([F_l (p=10 local vectors), W G U_{l-1} (transported factors of S_{l-1})]).')
    P('    New directions enter only through local vectors; the (k+p)x(k+p) core is an oracle.  R^2 of diag(S_l) on dvar_l;  D4r = same with p=10 RANDOM vectors instead of local (null).')
    P('l   ' + ' '.join(f'k={k:<4d}' for k in kk3) + ' | D4r: ' + ' '.join(f'k={k:<4d}' for k in kk3) + ' | D2: ' + ' '.join(f'k={k:<4d}' for k in kk3))
    U4 = {k: None for k in kk3}; U4r = {k: None for k in kk3}
    rowsD4 = {}
    for l in range(1, L+1):
        F = feats(l, 10); Fr = rng.standard_normal((n, 10))
        rr = {}; rrr = {}
        for k in kk3:
            for tag_, Ud, Fx in (('loc', U4, F), ('rnd', U4r, Fr)):
                if Ud[k] is None: B = Fx
                else:
                    g = gates[l-1]; B = np.column_stack([Fx, W[l-1] @ (g[:, None]*Ud[k])])
                Q, _ = np.linalg.qr(B); core = Q.T @ dC[l] @ Q
                lam, V = np.linalg.eigh(core); o_ = np.argsort(-np.abs(lam)); lam = lam[o_]; V = V[:, o_]
                Uk = Q @ V[:, :k]; Ud[k] = Uk
                if np.linalg.norm(dC[l]) > 0:
                    (rr if tag_ == 'loc' else rrr)[k] = r2(np.diag(dC[l]), np.sum(lam[None, :k]*Uk**2, axis=1))
        if np.linalg.norm(dC[l]) == 0: continue
        rowsD4[l] = dict(loc=rr, rnd=rrr)
        P(f'{l:<3d} ' + ' '.join(f'{rr[k]:6.3f}' for k in kk3) + ' |      ' + ' '.join(f'{rrr[k]:6.3f}' for k in kk3) + ' |     ' + ' '.join(f'{out["D2"].get(str(l), {}).get(k, float("nan")):6.3f}' for k in kk3))
    out['D4'] = {str(l): rr for l, rr in rowsD4.items()}
    tag = f'x6_an_{trunc}_vs_{ref}_n{n}_L{L}_s{seed}'
    json.dump(out, open(f'/home/claude/sandbox/results/{tag}.json', 'w'), indent=1, default=lambda o: o if not isinstance(o, (np.floating, np.integer)) else float(o))
    open(f'/home/claude/sandbox/results/{tag}.txt', 'w').write('\n'.join(txt))
    return out

if __name__ == '__main__':
    n, L, seed, trunc = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    ref = sys.argv[5] if len(sys.argv) > 5 else 'full'
    l0 = int(sys.argv[6]) if len(sys.argv) > 6 else None
    analyse(n, L, seed, trunc, ref, l0)
