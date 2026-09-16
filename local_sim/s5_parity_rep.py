import sys, os, numpy as np
"""Stage-1b v34 parity gates (nesting, exact law, float32, equivariance, guard)."""
_HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.dirname(_HERE)
# never let local_sim/ shadow the real flopscope: drop it, append it last
sys.path[:] = [q for q in sys.path if os.path.abspath(q or ".") != _HERE] + [R, _HERE]
import flopscope as flops; flops.configure(symmetry_warnings=False)
import k4feed_oracle as K, s1_parity_v32 as P1, s5_build_rep as B
from mint_provenance import canonical_mint
import tempfile
ok=[]
def claim(n,c,d=""): ok.append(bool(c)); print(("PASS " if c else "FAIL ")+n, d)
m32=K._load(os.path.join(R,"estimator_k3_v32.py")); m34=K._load(os.path.join(R,"estimator_k3_v34.py"))
tmp=tempfile.mkdtemp(); off=os.path.join(tmp,"off.py")
open(off,"w",encoding="utf-8").write(B.build(open(os.path.join(R,"estimator_k3_v32.py"),encoding="utf-8").read(),()))
moff=K._load(off)
for wd,dp,sd in ((64,5,90000000801),(96,6,90000000802)):
    w,es,_=canonical_mint(wd,dp,sd)
    pb,fb=P1._predict(m32,w,wd,dp,es); po,fo=P1._predict(moff,w,wd,dp,es); pp,fp=P1._predict(m34,w,wd,dp,es)
    claim(f"{wd}x{dp} off==v32 bitwise+flops", np.array_equal(pb,po) and fb==fo)
    claim(f"{wd}x{dp} v34 not no-op", np.abs(pp-pb).max()/np.abs(pb).max()>1e-7, "rel %.2e cost x%.5f"%(np.abs(pp-pb).max()/np.abs(pb).max(), fp/fb))
    g=m34._projected_flops(wd,dp,m34._fit_generations(wd,dp,2**41)); claim(f"{wd}x{dp} guard over-predicts", g>=fp, "%.4f"%(g/fp))
# exact law with the dense arm
md=K._load(os.path.join(R,"estimator_k3_v34_dense.py"))
rng=np.random.default_rng(20260916); n=3; mu=np.array([0.35,-0.2,0.1]); C=rng.normal(size=(n,n)); C=(C+C.T)*.5; np.fill_diagonal(C,0)
def est(d,e):
    out=K.run_layer(md,np.eye(n),mu,np.eye(n)+e*C,np.full(n,1.5*d*d),3*d*d); T=K.tower_tensor(out[2],n)
    return {"k3_001":T[0,0,1],"cov01":float(np.asarray(out[1])[0,1]),"k3_012":T[0,1,2]}
def odd(fn,k,d=0.02,e=0.01): return ((fn(d,e)[k]-fn(0,e)[k])-(fn(d,-e)[k]-fn(0,-e)[k]))/(2*d*d*e)
exact={"cov01":3.52809e-03,"k3_001":-1.62096e-01,"k3_012":-1.12056e-02}
for k,v in exact.items():
    r=odd(est,k); claim(f"exact law {k}", abs(r-v)<=0.05*abs(v), "%+.5e vs %+.5e"%(r,v))
# float32 and equivariance
wd,dp=64,5; w,es,_=canonical_mint(wd,dp,90000000803)
import flopscope.numpy as fnp
errs={}
for name,mod in (("v32",m32),("v34",m34)):
    p32,_=P1._predict(mod,w,wd,dp,es); st=(np.zeros(wd),np.eye(wd),None,0.0,np.zeros(wd)); rows=[]
    for l in range(dp):
        with flops.BudgetContext(flop_budget=2**55,quiet=True):
            o=mod._layer(fnp.asarray(np.asarray(w[l],dtype=np.float64),dtype=fnp.float64),(fnp.asarray(st[0],dtype=fnp.float64),fnp.asarray(st[1],dtype=fnp.float64),st[2],st[3],fnp.asarray(st[4],dtype=fnp.float64)),mod._tables(),fnp.float64,mod.Estimator()._constants(wd,fnp.float64),2.0,last_layer=(l==dp-1),gens=None,layer_index=l)
        rows.append(np.asarray(o[0],dtype=np.float64))
        if l<dp-1: st=(rows[-1],np.asarray(o[1],dtype=np.float64),o[2],o[3],np.asarray(o[4],dtype=np.float64))
    f64=np.maximum(np.stack(rows),0); errs[name]=max(np.abs(p32[l]-f64[l]).max()/np.abs(f64[l]).max() for l in range(1,dp))
claim("float32 <= 2x v32 + 1e-5", errs["v34"]<=2*errs["v32"]+1e-5, str(errs))
wd,dp=96,6; w,es,_=canonical_mint(wd,dp,90000000804); rng=np.random.default_rng(1)
for lay in (1,4):
    perm=rng.permutation(wd); wp=[np.array(x,copy=True) for x in w]; wp[lay]=wp[lay][:,perm]; wp[lay+1]=wp[lay+1][perm,:]
    e={}
    for name,mod in (("v32",m32),("v34",m34)):
        b,_=P1._predict(mod,w,wd,dp,es); q,_=P1._predict(mod,wp,wd,dp,es); mv=b.copy(); mv[lay]=b[lay][perm]
        e[name]=np.mean((mv-q)**2)/np.mean(b**2)
    claim(f"equivariance layer {lay} <= 100x v32", e["v34"]<=100*e["v32"], str(e))
claim("1024 same gens", m34._fit_generations(1024,16,2**41)==m32._fit_generations(1024,16,2**41))
print(f"{sum(ok)}/{len(ok)}")
