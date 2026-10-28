"""Band with importance-capped OLD hub columns: E=6 exact (KAS 2.5), old ages projected on rank-r basis,
but only the top-f fraction of old hub columns (by importance at the time of entry = w*||v||^2*||u||) are kept."""
import numpy as np, json, sys
import pooled_carrier as pc
res={}
for n in (128,256):
    r=int(round(0.28*n))
    for seed in (1,2):
        W=[np.random.default_rng(seed+1000*k).standard_normal((n,n))*np.sqrt(2.0/n) for k in range(16)]
        st=pc.k2_chain(W); pc._leg_cache.clear()
        sens={l:(st[l]['alpha']*pc.phi(st[l]['alpha'])/st[l]['sigma']**2)**2 for l in range(16)}
        # importance at entry (age 7): computed from legs at layer s+7
        keepsets={}
        for f in (1.0,0.75,0.5,0.35,0.25):
            e3=[];cols=[]
            for l in range(9,16):
                Uf=[];Vf=[];wf=[];Us=[];Vs=[];ws=[];oU=[];oV=[];ow=[]
                for s in range(0,l):
                    U,V=pc.leg_at(W,st,s,l); w=st[s]['sigma']*pc.phi(st[s]['alpha']); kas=np.abs(st[s]['alpha'])<2.5
                    Uf.append(U);Vf.append(V);wf.append(w)
                    if l-s<=6: Us.append(U[:,kas]);Vs.append(V[:,kas]);ws.append(w[kas])
                    else:
                        key=(s,f)
                        if key not in keepsets:
                            Ue,Ve=pc.leg_at(W,st,s,s+7); imp=w*np.sum(Ve*Ve,0)*np.sqrt(np.sum(Ue*Ue,0)); imp=np.where(kas,imp,-1)
                            k=int(round(f*kas.sum())); idx=np.argsort(-imp)[:k]; m=np.zeros(n,bool); m[idx]=True; keepsets[key]=m
                        m=keepsets[key]; oU.append(U[:,m]);oV.append(V[:,m]);ow.append(w[m])
                if oU:
                    U_=np.concatenate(oU,1);V_=np.concatenate(oV,1);w_=np.concatenate(ow); sw=np.sqrt(w_)[None,:]
                    Q,_,_=np.linalg.svd(np.concatenate([U_*sw,V_*sw],1),full_matrices=False); Q=Q[:,:r]
                    Us.append(Q@(Q.T@U_));Vs.append(Q@(Q.T@V_));ws.append(w_); cols.append(U_.shape[1])
                F3,_=pc.readouts(np.concatenate(Uf,1),np.concatenate(Vf,1),np.concatenate(wf))
                D3,_=pc.readouts(np.concatenate(Us,1),np.concatenate(Vs,1),np.concatenate(ws))
                e3.append(pc.werr(D3,F3,sens[l]))
            res[f'n{n}_s{seed}_f{f}']=(float(np.sqrt(np.mean(np.square(e3)))), float(np.mean(cols))/n)
            print(n,seed,f,'e3w %.4f  old cols/n %.2f'%res[f'n{n}_s{seed}_f{f}'],flush=True)
json.dump(res,open('results/oldcap.json','w'),indent=1)
