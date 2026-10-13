# Gain-mode analysis of exact pre-activation cumulant slices. READ-ONLY on repo.
import sys, json, time
from common import *
NMC=1e9
TAUS=(2.0,2.5,3.0)
OFF=~np.eye(N,dtype=bool)

def ls(Xs,y,mask=None,w=None):
    """least squares y ~ sum_k c_k X_k over entries in mask (bool array or None)."""
    if mask is None:
        A=np.stack([x.ravel() for x in Xs],1); b=y.ravel()
    else:
        A=np.stack([x[mask] for x in Xs],1); b=y[mask]
    if w is not None:
        ww=w.ravel() if mask is None else w[mask]
        A=A*ww[:,None]; b=b*ww
    G=A.T@A; r=A.T@b
    try: c=np.linalg.solve(G,r)
    except np.linalg.LinAlgError: c=np.linalg.lstsq(G,r,rcond=None)[0]
    return c
def pred(Xs,c):
    out=np.zeros_like(Xs[0])
    for x,ck in zip(Xs,c): out+=ck*x
    return out
def classes(alpha,tau):
    return {'dead':alpha<-tau,'kink':np.abs(alpha)<=tau,'on':alpha>tau}
def en_vec(res,y,alpha,wgt):
    """SSE/SST per class for vectors; returns dict."""
    o={}
    for tau in TAUS:
        for cn,m in classes(alpha,tau).items():
            o[f'{tau}/{cn}']=[float((res[m]**2).sum()),float((y[m]**2).sum()),float(((wgt*res)[m]**2).sum()),float(((wgt*y)[m]**2).sum())]
    o['all']=[float((res**2).sum()),float((y**2).sum()),float(((wgt*res)**2).sum()),float(((wgt*y)**2).sum())]
    return o
def en_mat(res,y,alpha,wr,wc):
    """SSE/SST by row class and by col class for off-diag matrices. res,y have zero diag."""
    r2=res**2; y2=y**2
    Wt=(wr[:,None]*wc[None,:])**2
    r2w=r2*Wt; y2w=y2*Wt
    rr=r2.sum(1); yr=y2.sum(1); rrw=r2w.sum(1); yrw=y2w.sum(1)
    rc=r2.sum(0); yc=y2.sum(0); rcw=r2w.sum(0); ycw=y2w.sum(0)
    o={'all':[float(rr.sum()),float(yr.sum()),float(rrw.sum()),float(yrw.sum())]}
    for tau in TAUS:
        for cn,m in classes(alpha,tau).items():
            o[f'{tau}/row/{cn}']=[float(rr[m].sum()),float(yr[m].sum()),float(rrw[m].sum()),float(yrw[m].sum())]
            o[f'{tau}/col/{cn}']=[float(rc[m].sum()),float(yc[m].sum()),float(rcw[m].sum()),float(ycw[m].sum())]
    return o
def r2c(y,p):
    return float(1-((y-p)**2).sum()/((y-y.mean())**2).sum())

def relu_k3(mu,sig):
    a=mu/sig; p=phi(a); P=Phi(a)
    m1=sig*(p+a*P); m2=sig**2*((a*a+1)*P+a*p); m3=sig**3*((a**3+3*a)*P+(a*a+2)*p)
    return m1, m2-m1**2, m3-3*m1*m2+2*m1**3
def corr(x,y):
    x=x-x.mean(); y=y-y.mean(); return float((x*y).sum()/np.sqrt((x*x).sum()*(y*y).sum()+1e-300))
def layer(L,g,l,mu,var,D3,dG,rng,tr):
    out={}
    s=np.sqrt(var[l]); m=mu[l]; al=m/s
    ph=phi(al); Ph=Phi(al)
    g3=D3[l]/s**3; g4=2*dG[l]/s**4      # kappa4 = 2*dG
    out['alpha_rms']=float(np.sqrt((al**2).mean())); out['alpha_mean']=float(al.mean()); out['sig_mean']=float(s.mean())
    out['frac']={f'{t}/{cn}':float(mk.mean()) for t in TAUS for cn,mk in classes(al,t).items()}
    C=L.mat('C11',l); out['C_sym']=float(np.abs(C-C.T).max()); out['C_diag']=float(np.abs(np.diag(C)).max())
    rho=C/np.outer(s,s); del C
    D21=L.mat('D21',l); out['D21_diag']=float(np.abs(np.diag(D21)).max())
    Y=D21/np.outer(s**2,s); del D21     # Y_ac = k3(a,a,c)/(s_a^2 s_c)
    aa=al[:,None]*np.ones((1,N)); ac=np.ones((N,1))*al[None,:]
    for X in (aa,ac): X[~OFF]=0
    # ---------- step 1: v estimates
    v={}
    v['a']=float((g3*al).sum()/(6*(al**2).sum()))
    A=np.stack([al,np.ones(N)],1); ca=np.linalg.lstsq(A,g3,rcond=None)[0]
    v['a_int']=float(ca[0]/6); v['a_int_c']=float(ca[1])
    out['g3_lin_R2c']=r2c(g3,A@ca); out['g3_rms']=float(np.sqrt((g3**2).mean())); out['g4_rms']=float(np.sqrt((g4**2).mean()))
    v['b_mean']=float(g4.mean()/12); v['b_med']=float(np.median(g4)/12)
    xb=12+12*al*g3; v['b_ls']=float((g4*xb).sum()/(xb**2).sum())
    X1=aa*rho; X2=ac; X3=aa*aa*ac
    Xfix=2*(2*X1+X2)
    v['c']=float(ls([Xfix],Y,OFF)[0])
    c2=ls([X1,X2],Y,OFF); c3=ls([X1,X2,X3],Y,OFF)
    v['c_free_a/4']=float(c2[0]/4); v['c_free_b/2']=float(c2[1]/2)
    out['d21_free2']=[float(x) for x in c2]; out['d21_free3']=[float(x) for x in c3]
    aaal=np.outer(al,al); aaal[~OFF]=0
    v['d_off_norm']=float((rho*aaal).sum()/(aaal**2).sum())
    mm=np.outer(m,m); mm[~OFF]=0; Craw=rho*np.outer(s,s)
    v['d_off_raw']=float((Craw*mm).sum()/(mm**2).sum())
    Cfull=Craw+np.diag(var[l]); v['d_full']=float(m@Cfull@m/(m@m)**2)
    # bias of the spike estimate from independent per-neuron fluctuations of h_{l-1}: sum_i Var(h_i) m_i^2/|m|^4
    if l>=1:
        m1,vh,k3h=relu_k3(mu[l-1],np.sqrt(var[l-1])); mt=tr[l-1]
        out['spike_bias']=float((vh*mt**2).sum()/(mt@mt)**2); v['d_off_raw_corr']=v['d_off_raw']-out['spike_bias']
        out['m1_vs_truth_rms']=float(np.sqrt(((m1-mt)**2).mean()))
        Wl=weights(g,l)
        Nn=((Wl*Wl).T@(k3h[:,None]*Wl))/np.outer(s**2,s); N3=((Wl**3).T@k3h)/s**3; Nn[~OFF]=0
        del Wl
    else:
        Nn=np.zeros((N,N)); N3=np.zeros(N)
    out['mu_norm2']=float(m@m); out['trC']=float(var[l].sum())
    del mm,Craw
    # D22 / Q31
    Q31=L.mat('Q31',l); Qn=Q31/np.outer(s**3,s); del Q31
    D22=L.mat('D22',l); Dn=D22/np.outer(s**2,s**2); del D22
    out['D22_asym']=float(np.abs(Dn-Dn.T).max())
    # normalized noise scale check (meaningful at layer 0)
    out['noise_rms']={'D21n':float(np.sqrt((Y[OFF]**2).mean())),'Q31n':float(np.sqrt((Qn[OFF]**2).mean())),'D22n':float(np.sqrt((Dn[OFF]**2).mean())),
                      'g3':out['g3_rms'],'g4':out['g4_rms'],'rho':float(np.sqrt((rho[OFF]**2).mean()))}
    # norm-fluctuation v of z_l from labels: Var(|z|^2)/(4 E|z|^2^2)
    covsq=Dn*np.outer(var[l],var[l])+2*(rho*np.outer(s,s))**2+4*np.outer(m,m)*rho*np.outer(s,s)+2*m[:,None]*(Y.T*np.outer(s,s**2))+2*m[None,:]*(Y*np.outer(s**2,s))
    dsq=2*dG[l]+2*var[l]**2+4*m**2*var[l]+4*m*D3[l]
    out['v_normz']=float((covsq.sum()+dsq.sum())/(4*(var[l]+m**2).sum()**2)); del covsq
    # Q31 fixed: 12 v rho + 3v(3 a_a Y_ac + a_c g3_a)
    g3r=g3[:,None]*np.ones((1,N)); g3r[~OFF]=0
    Qx1=rho; Qx2=aa*Y; Qx3=ac*g3r
    Qfix=12*Qx1+3*(3*Qx2+Qx3)
    v['q31']=float(ls([Qfix],Qn,OFF)[0])
    YT=Y.T
    Dx0=OFF.astype(float); Dx1=rho**2; Dx2=aa*YT+ac*Y
    Dfix=4*(Dx0+2*Dx1)+6*Dx2
    v['d22']=float(ls([Dfix],Dn,OFF)[0])
    out['v']=v
    # ---------- step 2: R2 tables
    R={}
    w3n=np.abs(al)*ph; w4n=np.abs(al**2-1)*ph
    # D3
    R['D3/zero']=en_vec(g3,g3,al,w3n)
    R['D3/fix_va']=en_vec(g3-6*v['a']*al,g3,al,w3n)
    R['D3/fix_vc']=en_vec(g3-6*v['c']*al,g3,al,w3n)
    R['D3/lin_int']=en_vec(g3-A@ca,g3,al,w3n)
    A2=np.stack([al,al**3],1); cc=np.linalg.lstsq(A2,g3,rcond=None)[0]; out['d3_free2']=[float(x) for x in cc]
    R['D3/free2']=en_vec(g3-A2@cc,g3,al,w3n)
    A3=np.stack([al,al**3,np.ones(N)],1); cc3=np.linalg.lstsq(A3,g3,rcond=None)[0]; out['d3_free3']=[float(x) for x in cc3]
    R['D3/free3']=en_vec(g3-A3@cc3,g3,al,w3n)
    A4=np.stack([al,al**3,N3],1); cc4=np.linalg.lstsq(A4,g3,rcond=None)[0]; out['d3_free2N']=[float(x) for x in cc4]
    R['D3/free2N']=en_vec(g3-A4@cc4,g3,al,w3n)
    # D4
    B=np.stack([np.ones(N),al**2,al*g3],1); cb=np.linalg.lstsq(B,g4,rcond=None)[0]; out['d4_freeclos2']=[float(x) for x in cb]
    R['D4/free_clos2']=en_vec(g4-B@cb,g4,al,w4n)
    vq=v['b_mean']
    for _ in range(20): vq=float((g4-12*vq*al*(g3-6*v['a']*al)).mean()-(18*vq*vq+24*vq*vq*(al**2).mean()))/12
    v['b_ln2']=vq
    R['D4/ln2_own']=en_vec(g4-(12*vq+18*vq*vq+24*vq*vq*al**2+12*vq*al*(g3-6*v['a']*al)),g4,al,w4n)
    R['D4/fix_va']=en_vec(g4-v['a']*xb,g4,al,w4n)
    R['D4/fix_own_const']=en_vec(g4-12*v['b_mean'],g4,al,w4n)
    R['D4/fix_own']=en_vec(g4-v['b_ls']*xb,g4,al,w4n)
    B=np.stack([np.ones(N),al*g3],1); cb=np.linalg.lstsq(B,g4,rcond=None)[0]; out['d4_freeclos']=[float(x) for x in cb]
    R['D4/free_clos']=en_vec(g4-B@cb,g4,al,w4n)
    Bp=np.stack([np.ones(N),al**2,al**4],1); cbp=np.linalg.lstsq(Bp,g4,rcond=None)[0]; out['d4_freepoly']=[float(x) for x in cbp]
    R['D4/free_poly']=en_vec(g4-Bp@cbp,g4,al,w4n)
    out['g4_poly_R2c']=r2c(g4,Bp@cbp)
    # D21
    res={}
    res['fix_va']=Y-v['a']*Xfix; res['fix_vc']=Y-v['c']*Xfix
    res['free2']=Y-pred([X1,X2],c2); res['free3']=Y-pred([X1,X2,X3],c3)
    c4=ls([X1,X2,X3,Nn],Y,OFF) if l>=1 else np.append(c3,0.0); out['d21_free3N']=[float(x) for x in c4]
    res['free3N']=Y-pred([X1,X2,X3,Nn],c4)
    vln=v['c']; res['ln2_vc']=Y-((2*vln+vln**2)*(2*X1+X2)-(3*vln**2)*X3)
    for k,r in res.items(): R['D21/'+k]=en_mat(r,Y,al,ph,Ph)
    # Q31: weights w3_a w1_c ; D22: w2_a w2_c
    for nm,vv in (('fix_va',v['a']),('fix_own',v['q31'])):
        R['Q31/'+nm]=en_mat(Qn-vv*Qfix,Qn,al,w3n,Ph)
    cq=ls([Qx1,Qx2,Qx3],Qn,OFF); out['q31_freeclos']=[float(x) for x in cq]
    R['Q31/free_clos']=en_mat(Qn-pred([Qx1,Qx2,Qx3],cq),Qn,al,w3n,Ph)
    Qp=[rho,aa*aa*rho,aa*ac,aa**3*ac]
    cq=ls(Qp,Qn,OFF); out['q31_freepoly']=[float(x) for x in cq]
    R['Q31/free_poly']=en_mat(Qn-pred(Qp,cq),Qn,al,w3n,Ph); del Qp
    for nm,vv in (('fix_va',v['a']),('fix_own',v['d22'])):
        R['D22/'+nm]=en_mat(Dn-vv*Dfix,Dn,al,ph,ph)
    cd=ls([Dx0,Dx1,Dx2],Dn,OFF); out['d22_freeclos']=[float(x) for x in cd]
    R['D22/free_clos']=en_mat(Dn-pred([Dx0,Dx1,Dx2],cd),Dn,al,ph,ph)
    Dp=[Dx0,Dx1,aa**2+ac**2*OFF,aa*ac*rho,aa**2*ac**2]
    cd=ls(Dp,Dn,OFF); out['d22_freepoly']=[float(x) for x in cd]
    R['D22/free_poly']=en_mat(Dn-pred(Dp,cd),Dn,al,ph,ph); del Dp
    del Qn,Dn,Qfix,Dfix,Qx2,Qx3,Dx1,Dx2,g3r
    out['R']=R
    # ---------- step 4: residual structure (free3 residual, normalised units)
    S={}
    for nm,M in (('Y',Y),('res_free3',res['free3']),('res_fix_vc',res['fix_vc'])):
        sv=np.linalg.svd(M,compute_uv=False); e=sv**2; tot=e.sum()
        S[nm]={'tot':float(tot),'top':[float(e[:k].sum()/tot) for k in (1,2,4,16,64)],'sv':[float(x) for x in sv[:8]]}
    U,sv,Vt=np.linalg.svd(res['free3'])
    for k in (1,4,16): res[f'free3_defl{k}']=res['free3']-(U[:,:k]*sv[:k])@Vt[:k]
    S['u1corr']={'1':float(abs(U[:,0].mean())*np.sqrt(N)),'al':corr(U[:,0],al),'al2':corr(U[:,0],al**2),'absal':corr(U[:,0],np.abs(al)),'phi':corr(U[:,0],ph),'Phi':corr(U[:,0],Ph)}
    S['v1corr']={'1':float(abs(Vt[0].mean())*np.sqrt(N)),'al':corr(Vt[0],al),'al2':corr(Vt[0],al**2),'absal':corr(Vt[0],np.abs(al)),'phi':corr(Vt[0],ph),'Phi':corr(Vt[0],Ph)}
    del U,Vt
    rn=np.sqrt((res['free3N']**2).sum(1)/(N-1))
    S['rowrmsN']={f'{t}/{cn}':float(np.sqrt((rn[mk]**2).mean())) if mk.any() else None for t in TAUS for cn,mk in classes(al,t).items()}
    sv=np.linalg.svd(res['free3N'],compute_uv=False); e=sv**2; tot=e.sum()
    S['res_free3N']={'tot':float(tot),'top':[float(e[:k].sum()/tot) for k in (1,2,4,16,64)],'sv':[float(x) for x in sv[:8]]}
    # consumption-weighted residual
    Mw=ph[:,None]*res['free3']*Ph[None,:]
    sv=np.linalg.svd(Mw,compute_uv=False); e=sv**2; tot=e.sum()
    S['res_free3_w']={'tot':float(tot),'top':[float(e[:k].sum()/tot) for k in (1,2,4,16,64)]}
    S['noise_tot']=float(2/NMC*N*(N-1))
    # per-row rms of residual & of Y per class
    rr=np.sqrt((res['free3']**2).sum(1)/(N-1)); yr=np.sqrt((Y**2).sum(1)/(N-1))
    S['rowrms']={f'{t}/{cn}':[float(np.sqrt((rr[mk]**2).mean())) if mk.any() else None,float(np.sqrt((yr[mk]**2).mean())) if mk.any() else None] for t in TAUS for cn,mk in classes(al,t).items()}
    out['S']=S
    # ---------- step 3: pruning test
    P={}
    if l<DEPTH-1:
        W=weights(g,l+1); varn=var[l+1]
        rowf=ph*s; colf=Ph*s
        def contrib(Rn):
            A_=rowf[:,None]*Rn*colf[None,:]
            return W*(A_@W)          # P_ab ; delta_b = sum_a
        def stat(d):
            x=d/varn
            return [float(np.sqrt((x**2).mean())),float(x.mean())]
        noise=rng.standard_normal((N,N))*np.sqrt(2/NMC); noise[~OFF]=0
        mats={'D21':Y,'noise':noise}; mats.update(res)
        # fits on kept (kink) rows only, per tau; and own-set fits
        for t in TAUS:
            cl=classes(al,t)
            mk=OFF&cl['kink'][:,None]
            ck=ls([X1,X2,X3],Y,mk); out[f'd21_free3_kinkfit/{t}']=[float(x) for x in ck]
            mats[f'free3_kinkfit/{t}']=Y-pred([X1,X2,X3],ck)
            ckN=ls([X1,X2,X3,Nn],Y,mk) if l>=1 else np.append(ck,0.0)
            mats[f'free3N_kinkfit/{t}']=Y-pred([X1,X2,X3,Nn],ckN)
            ck2=ls([Xfix],Y,mk); out[f'v_c_kinkfit/{t}']=float(ck2[0])
            mats[f'fix_vckink/{t}']=Y-ck2[0]*Xfix
            Ro=np.zeros((N,N))
            for cn,mrow in cl.items():
                if mrow.sum()>=4:
                    co=ls([X1,X2,X3],Y,OFF&mrow[:,None]); out[f'd21_free3_own/{t}/{cn}']=[float(x) for x in co]
                    Ro[mrow]=(Y-pred([X1,X2,X3],co))[mrow]
                else: Ro[mrow]=Y[mrow]
            mats[f'free3_own/{t}']=Ro
        for nm,M in mats.items():
            Pm=contrib(M)
            P[nm+'/all']=stat(Pm.sum(0))
            for t in TAUS:
                if '/' in nm and not nm.endswith(f'/{t}'): continue
                cl=classes(al,t)
                for cn,mrow in cl.items():
                    P[f'{nm}/rows/{t}/{cn}']=stat(Pm[mrow].sum(0))
                P[f'{nm}/rows/{t}/dead+on']=stat(Pm[~cl['kink']].sum(0))
            del Pm
        # column restrictions (dead cols, on cols)
        for nm in ('D21','noise','fix_va','fix_vc','free3'):
            M=mats[nm]
            for t in TAUS:
                cl=classes(al,t)
                for cn in ('dead','on'):
                    mc=cl[cn]
                    A_=rowf[:,None]*M[:,mc]*colf[None,mc]
                    d=(W*(A_@W[mc])).sum(0)
                    P[f'{nm}/cols/{t}/{cn}']=stat(d)
        # with diagonal (D3 on diag) full effect
        Yd=Y+np.diag(g3); P['D21+diag/all']=stat(contrib(Yd).sum(0)); del Yd
        P['varn_mean']=float(varn.mean())
        # mean channel D3
        base=-al*ph*s/6      # dm_a = base_a * d(g3_a)
        Mn={}
        def mstat(dm):
            return [float(np.sqrt((dm**2).mean())),float(np.sqrt(((W.T@dm)**2).mean()))]
        Mn['full']=mstat(base*g3)
        base4=(al**2-1)*ph*s/24
        Mn['full_k4']=mstat(base4*g4)
        A2=np.stack([al,al**3],1)
        for t in TAUS:
            cl=classes(al,t)
            sets={'dead':cl['dead'],'on':cl['on'],'dead+on':~cl['kink'],'all':np.ones(N,bool)}
            for sn,ms in sets.items():
                keep=~ms if sn!='all' else np.ones(N,bool)
                Mn[f'{t}/{sn}/n']=int(ms.sum())
                Mn[f'{t}/{sn}/zero']=mstat(np.where(ms,base*g3,0))
                Mn[f'{t}/{sn}/fix_va']=mstat(np.where(ms,base*(g3-6*v['a']*al),0))
                Mn[f'{t}/{sn}/fix_vc']=mstat(np.where(ms,base*(g3-6*v['c']*al),0))
                vk=(g3[keep]*al[keep]).sum()/(6*(al[keep]**2).sum())
                Mn[f'{t}/{sn}/fix_vkeep']=mstat(np.where(ms,base*(g3-6*vk*al),0))
                Mn[f'{t}/{sn}/free2']=mstat(np.where(ms,base*(g3-A2@cc),0))
                ckN=np.linalg.lstsq(A4[keep],g3[keep],rcond=None)[0]
                Mn[f'{t}/{sn}/free2N_keep']=mstat(np.where(ms,base*(g3-A4@ckN),0))
                ck=np.linalg.lstsq(A2[keep],g3[keep],rcond=None)[0]
                Mn[f'{t}/{sn}/free2_keep']=mstat(np.where(ms,base*(g3-A2@ck),0))
                # k4 mean channel
                Mn[f'{t}/{sn}/k4_zero']=mstat(np.where(ms,base4*g4,0))
                Mn[f'{t}/{sn}/k4_fixva']=mstat(np.where(ms,base4*(g4-v['a']*xb),0))
                Mn[f'{t}/{sn}/k4_poly']=mstat(np.where(ms,base4*(g4-Bp@cbp),0))
        P['mean']=Mn
    out['P']=P
    return out

if __name__=='__main__':
    g=int(sys.argv[1]); layers=[int(x) for x in sys.argv[2].split(',')] if len(sys.argv)>2 else list(range(DEPTH))
    tag=sys.argv[3] if len(sys.argv)>3 else ''
    L=Labels(g); mu=L.vec('mu'); var=L.vec('var'); D3=L.vec('D3'); dG=L.vec('dG')
    rng=np.random.default_rng(1000+g); tr=truth(g)
    res={}
    for l in layers:
        t0=time.time()
        res[l]=layer(L,g,l,mu,var,D3,dG,rng,tr)
        print(f'net {g} layer {l} done {time.time()-t0:.1f}s v={res[l]["v"]}',flush=True)
        json.dump(res,open(f'out_gain_{g:03d}{tag}.json','w'))
