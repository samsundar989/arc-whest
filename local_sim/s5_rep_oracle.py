import sys, os, numpy as np
"""Stage-1b exact-law oracle: bivariate kappa4 (x C) completion (q31, link) vs the scale mixture."""
_HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.dirname(_HERE)
# never let local_sim/ shadow the real flopscope: drop it, append it last
sys.path[:] = [q for q in sys.path if os.path.abspath(q or ".") != _HERE] + [R, _HERE]
import flopscope as flops; flops.configure(symmetry_warnings=False)
import k4feed_oracle as K
from scipy.special import ndtr
import tempfile; SCR=tempfile.mkdtemp(prefix="s5rep_")
src=open(os.path.join(R,"estimator_k3_v32_dense.py"),encoding="utf-8").read()
anchor="    # ---- the (1,1,1) block's repeated part -------------------------------"
assert src.count(anchor)==1
patch='''    if _REP_MODE and not last_layer:
        # first-order kappa4 (x C) completion of the bivariate moment slices
        _li_r = layer_index - 1
        _lam_r = float(_K4_LAM[_li_r]) if 0 <= _li_r < len(_K4_LAM) else 0.0
        _m = metric
        _a = alpha; _ph = fnp.exp(_a * _a * (-0.5)) * 0.3989422804014327
        _s4 = sigma * sigma * sigma * sigma
        w51 = ((_a * _a - 3.0) * _a * _ph / _s4) * (-1.0)
        W = {(1, 1): w[(1, 1)], (1, 2): w[(1, 2)], (3, 1): w[(3, 1)], (3, 2): w[(3, 2)],
             (5, 1): w51, (5, 2): w[(4, 1)] * 2.0}
        for (p, q) in ((1, 1), (2, 1), (2, 2)):
            add = None
            if "q31" in _REP_MODE:
                t = (W[(3, p)][:, None] * W[(1, q)][None, :] + W[(1, p)][:, None] * W[(3, q)][None, :]) * C11
                add = t * (_m * _lam_r / 12.0)
            if "link" in _REP_MODE:
                t2 = ((dG * W[(5, p)])[:, None] * W[(1, q)][None, :]
                      + W[(1, p)][:, None] * (dG * W[(5, q)])[None, :]
                      + (dG[:, None] + dG[None, :]) * (W[(3, p)][:, None] * W[(3, q)][None, :])) * C11
                t2 = t2 * (_m / 24.0)
                add = t2 if add is None else add + t2
            if add is not None:
                pK[(p, q)] = pK[(p, q)] + (add if p != q else (add + add.T) * 0.5)
'''
src=src.replace(anchor, patch+anchor,1).replace("_P2_RANK = 0","_P2_RANK = 0\n_REP_MODE = ()",1)
p=os.path.join(SCR,"v32d_rep.py"); open(p,"w",encoding="utf-8").write(src)
mod=K._load(p)
rng=np.random.default_rng(20260916); n=3
mu=np.array([0.35,-0.2,0.1]); C=rng.normal(size=(n,n)); C=(C+C.T)*.5; np.fill_diagonal(C,0)
# check w conventions: w(1,2)==2 w(0,1)?, w(3,2)==2 w(2,1)?, w(4,2)==2 w(3,1)?
import flopscope.numpy as fnp
def est(d,eps,mode):
    mod._REP_MODE=mode; d2=d*d
    out=K.run_layer(mod,np.eye(n),mu,np.eye(n)+eps*C,np.full(n,1.5*d2),3*d2)
    T=K.tower_tensor(out[2],n)
    return {"k3_001":T[0,0,1],"cov01":float(np.asarray(out[1])[0,1])}
def exact(d,eps):
    S=np.eye(n)+eps*C; acc={"0":0,"1":0,"01":0,"00":0,"001":0}
    for s2 in (1-d,1+d):
        Ss=S*s2
        acc["0"]+=.5*K._moment(mu[[0]],Ss[np.ix_([0],[0])],(1,)); acc["1"]+=.5*K._moment(mu[[1]],Ss[np.ix_([1],[1])],(1,))
        acc["01"]+=.5*K._moment(mu[[0,1]],Ss[np.ix_([0,1],[0,1])],(1,1)); acc["00"]+=.5*K._moment(mu[[0]],Ss[np.ix_([0],[0])],(2,))
        acc["001"]+=.5*K._moment(mu[[0,1]],Ss[np.ix_([0,1],[0,1])],(2,1))
    m0,m1=acc["0"],acc["1"]
    return {"cov01":acc["01"]-m0*m1,"k3_001":acc["001"]-acc["00"]*m1-2*acc["01"]*m0+2*m0*m0*m1}
def odd(fn,k,d=0.02,e=0.01):
    return ((fn(d,e)[k]-fn(0,e)[k])-(fn(d,-e)[k]-fn(0,-e)[k]))/(2*d*d*e)
for k in ("cov01","k3_001"):
    print(k,"exact %+.5e"%odd(exact,k), *["%s %+.5e"%(mm or "v32",odd(lambda a,b:est(a,b,mm),k)) for mm in ((),("q31",),("q31","link"))])
