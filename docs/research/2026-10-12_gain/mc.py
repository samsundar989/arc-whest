# Monte-Carlo measurement of gain statistics of post-activations h_l (direct, model-free).
# For each layer l: r=|h_l|, p=h_l.m_l/|m_l|^2 (m_l = exact E h_l from full_truth), q=|h_l - p m_l|
import sys, json, time
from common import *
g=int(sys.argv[1]); M=int(sys.argv[2]) if len(sys.argv)>2 else 131072; B=4096
tr=truth(g).astype(np.float32)
Ws=[weights(g,l).astype(np.float32) for l in range(DEPTH)]
rng=np.random.default_rng(777+g)
acc=np.zeros((DEPTH+1,9))   # sums: r, r2, p, p2, q, q2, rp, qp, rq   (index 0 = input x: r only)
t0=time.time()
for b in range(M//B):
    h=rng.standard_normal((B,N),dtype=np.float32)
    r=np.sqrt((h.astype(np.float64)**2).sum(1)); acc[0,0]+=r.sum(); acc[0,1]+=(r*r).sum()
    for l in range(DEPTH):
        h=np.maximum(h@Ws[l],0)
        m=tr[l].astype(np.float64); hd=h.astype(np.float64)
        r2=(hd**2).sum(1); r=np.sqrt(r2); p=hd@m/(m@m); q=np.sqrt(np.maximum(r2-p*p*(m@m),0))
        acc[l+1]+=[r.sum(),r2.sum(),p.sum(),(p*p).sum(),q.sum(),(q*q).sum(),(r*p).sum(),(q*p).sum(),(r*q).sum()]
    if b%8==0: print(b,time.time()-t0,flush=True)
a=acc/M
out={}
out['x']={'v_r':float(a[0,1]/a[0,0]**2-1)}
for l in range(DEPTH):
    r,r2,p,p2,q,q2,rp,qp,rq=a[l+1]
    out[l]={'v_r':float(r2/r**2-1),'v_p':float(p2/p**2-1),'v_q':float(q2/q**2-1),'v_rp':float(rp/(r*p)-1),'v_qp':float(qp/(q*p)-1),'v_rq':float(rq/(r*q)-1),'Ep':float(p),'q_over_r':float(q/r)}
json.dump(out,open(f'out_mc_{g:03d}.json','w'),indent=1)
print('done',time.time()-t0)
