import json, numpy as np
D=[json.load(open(f'out_extra_{g:03d}.json')) for g in range(4)]
m=lambda l,k,i=None:np.mean([d[str(l)][k] if i is None else d[str(l)][k][i] for d in D])
print('nets 0-3. cols: R2(Y~alpha_c) R2(Y~alpha_a) | rho_next_rms | full D21 effect: var(rel) , offdiag cov (in units sigma_b sigma_d)')
for l in range(1,15):
    print('%3d  %.4f %.6f | %.4f | %.3g %.3g'%(l,m(l,'R2_alpha_c'),m(l,'R2_alpha_a'),m(l,'rho_next_rms'),m(l,'full',0),m(l,'full',1)))
for t in ('2.0','2.5','3.0'):
    print(f'\ntau={t}: per set: frac of offdiag entries pruned | var err ZERO, GAIN | offdiag-cov err (rms, units sigma_b sigma_d) ZERO, GAIN')
    print('  l '+''.join(('%s'%nm).center(52) for nm in ('rows |a|>tau','dead cols','rows+deadcols')))
    for l in range(1,15):
        print('%3d '%l+' | '.join('%.3f  %.2e %.2e  %.2e %.2e'%(m(l,f'{t}/{nm}/frac'),m(l,f'{t}/{nm}/zero',0),m(l,f'{t}/{nm}/gain',0),m(l,f'{t}/{nm}/zero',1),m(l,f'{t}/{nm}/gain',1)) for nm in ('rows','deadcols','rows+deadcols')))
