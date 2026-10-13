# Extra checks: (1) index convention of D21, (2) off-diagonal next-layer covariance error, (3) combined rows+dead-cols pruning.
import sys, json
from common import *
OFF=~np.eye(N,dtype=bool)
g=int(sys.argv[1]); L=Labels(g); mu=L.vec('mu'); var=L.vec('var')
out={}
for l in range(1,15):
    s=np.sqrt(var[l]); al=mu[l]/s; ph=phi(al); Ph=Phi(al)
    rho=L.mat('C11',l)/np.outer(s,s); Y=L.mat('D21',l)/np.outer(s**2,s)
    aa=al[:,None]*OFF; ac=al[None,:]*OFF
    o={}
    sst=(Y**2).sum()
    for nm,X in (('alpha_c',ac),('alpha_a',aa)):
        c=(X*Y).sum()/(X*X).sum(); o['R2_'+nm]=float(1-((Y-c*X)**2).sum()/sst); o['coef_'+nm]=float(c)
    W=weights(g,l+1); sn=np.sqrt(var[l+1]); rhon=L.mat('C11',l+1)/np.outer(sn,sn)
    o['rho_next_rms']=float(np.sqrt((rhon[OFF]**2).mean()))
    rowf=ph*s; colf=Ph*s
    Xfix=2*(2*aa*rho+ac)
    def eff(Rn):
        A=rowf[:,None]*Rn*colf[None,:]; E=0.5*(A+A.T); M=W.T@E@W
        M=M/np.outer(sn,sn)
        return [float(np.sqrt((np.diag(M)**2).mean())),float(np.sqrt((M[OFF]**2).mean()))]
    o['full']=eff(Y)
    for t in (2.0,2.5,3.0):
        kink=np.abs(al)<=t; mk=OFF&kink[:,None]
        v=(Xfix[mk]*Y[mk]).sum()/(Xfix[mk]**2).sum()
        R=Y-v*Xfix
        rows=~kink; deadc=al<-t
        U=rows[:,None]|deadc[None,:]
        for nm,mask in (('rows',np.broadcast_to(rows[:,None],(N,N))),('deadcols',np.broadcast_to(deadc[None,:],(N,N))),('rows+deadcols',U)):
            o[f'{t}/{nm}/zero']=eff(np.where(mask,Y,0)); o[f'{t}/{nm}/gain']=eff(np.where(mask,R,0))
            o[f'{t}/{nm}/frac']=float((mask&OFF).sum()/OFF.sum())
    out[l]=o
    print(g,l,o['R2_alpha_c'],o['R2_alpha_a'],o['full'],o['2.5/rows+deadcols/zero'],o['2.5/rows+deadcols/gain'],flush=True)
json.dump(out,open(f'out_extra_{g:03d}.json','w'))
