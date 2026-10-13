import json, numpy as np
mc=[json.load(open(f'out_mc_{g:03d}.json')) for g in range(4)]
ga=[json.load(open(f'out_gain_{g:03d}.json')) for g in range(4)]
print('input x: v_r=Var|x|/E|x|^2 x1e3:',[round(1e3*m['x']['v_r'],4) for m in mc],' theory 1/(2n)=',1e3/2048)
print('MC (458752 samples/net, nets 0-3) statistics of h_{l-1} (feeding z_l), x1e3; label-fit v at layer l for the same nets')
print('  l   v_r(norm)  v_p(proj m)  v_q(|h_perp|)  cov(r,p)  cov(q,p)  cov(r,q) |   v_a(D3)  v_c(D21)  v_b(D4)  v_spike_offdiag | v_a/cov(q,p)  v_a/cov(r,p) v_b/v_q  v_b/v_r  spike/v_p')
for l in range(1,16):
    k=str(l-1)
    f=lambda key:np.mean([m[k][key] for m in mc])*1e3
    gv=lambda key:np.mean([d[str(l)]['v'][key] for d in ga])*1e3
    print('%3d %10.3f %10.3f %12.3f %10.3f %9.3f %9.3f | %9.3f %9.3f %8.3f %10.3f       | %9.3f %12.3f %9.3f %8.3f %9.3f'%(l,f('v_r'),f('v_p'),f('v_q'),f('v_rp'),f('v_qp'),f('v_rq'),gv('a'),gv('c'),gv('b_mean'),gv('d_off_raw'),gv('a')/f('v_qp'),gv('a')/f('v_rp'),gv('b_mean')/f('v_q'),gv('b_mean')/f('v_r'),gv('d_off_raw')/f('v_p')))
