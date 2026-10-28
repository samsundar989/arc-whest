import numpy as np, json
import pooled_carrier as pc
for n in (128,256):
    for seed in (1,2):
        W=[np.random.default_rng(seed+1000*k).standard_normal((n,n))*np.sqrt(2.0/n) for k in range(16)]
        st=pc.k2_chain(W); L=16
        pc._leg_cache.clear()
        # full reference
        sens={l:dict(s3=(st[l]['alpha']*pc.phi(st[l]['alpha'])/st[l]['sigma']**2)**2) for l in range(L)}
        rows={}
        for kas in (2.0,2.5,3.0,3.5,99.0):
            e3=[];cost=[]
            for l in range(9,16):
                Us=[];Vs=[];ws=[];Uf=[];Vf=[];wf=[]
                for s in range(0,l):
                    U,V=pc.leg_at(W,st,s,l); w=st[s]['sigma']*pc.phi(st[s]['alpha']); keep=np.abs(st[s]['alpha'])<kas
                    Uf.append(U);Vf.append(V);wf.append(w)
                    if l-s<=6: Us.append(U[:,keep]);Vs.append(V[:,keep]);ws.append(w[keep])
                    else: Us.append(U[:,keep]);Vs.append(V[:,keep]);ws.append(w[keep])  # band ignored: isolate the hub cut only
                F3,_=pc.readouts(np.concatenate(Uf,1),np.concatenate(Vf,1),np.concatenate(wf))
                D3,_=pc.readouts(np.concatenate(Us,1),np.concatenate(Vs,1),np.concatenate(ws))
                e3.append(pc.werr(D3,F3,sens[l]['s3'])); cost.append(sum(u.shape[1] for u in Us[-6:]))
            rows[kas]=(float(np.sqrt(np.mean(np.square(e3)))), float(np.mean(cost))/(6*n))
        print(n,seed,{k:(round(v[0],4),round(v[1],3)) for k,v in rows.items()})
