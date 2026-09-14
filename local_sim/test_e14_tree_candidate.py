"""Label-free tiny controls for the isolated E14 tree candidate."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import flopscope.numpy as fnp
from flopscope import BudgetContext

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import estimator_e14_tree as candidate
import estimator_k3_strassen_L4_f21 as frozen
import e14_tree_oracle as oracle


def tree_case(n):
    ix = np.arange(n, dtype=np.float64)
    m = np.fromfunction(lambda i, j: (0.11 + 0.07*i - 0.03*j) /
                        (1 + abs(i-j)), (n,n), dtype=float)
    c = (m+m.T)*0.5 + np.diag(0.9+0.13*ix)
    b, a, d = 0.33+0.08*ix, -0.17+0.09*ix, 0.06-0.03*ix
    k4 = 0.12+0.07*ix
    k22 = np.fromfunction(lambda i,j: 0.03+0.02*(i+j), (n,n))
    np.fill_diagonal(k22, 0)
    cout = c + 0.05 * np.outer(a,a)
    with BudgetContext(10**10):
        g, lam, cout = candidate._tree_core(
            fnp.array(c), fnp.array(cout), fnp.array(k4), fnp.array(k22), fnp.array(b),
            fnp.array(a), fnp.array(d), fnp.array(1-np.eye(n)))
    dense = oracle.dense_tree(c,b,a,d)
    h = np.einsum('abcc->ab',oracle.anchored(dense,k4,k22))
    _, gref, lref, _ = oracle.projected(h,c + 0.05 * np.outer(a,a))
    return max(float(np.max(abs(np.asarray(g)-gref))),
               abs(float(lam)-lref), float(np.max(abs(np.asarray(cout)-
                                         (c + 0.05 * np.outer(a,a))))))


def feed_case(n, k):
    rng=np.random.default_rng(94+n*10+k)
    A=fnp.array(rng.normal(size=(n,k*n)).astype('float32')*0.1)
    P=fnp.array(rng.normal(size=(n,k*n)).astype('float32')*0.1)
    D=fnp.array(rng.normal(size=(n,k*n)).astype('float32')*0.1)
    Y=fnp.array(rng.normal(size=(n,k)).astype('float32')*0.1)
    R=fnp.array(rng.normal(size=(n,k)).astype('float32')*0.1)
    w=fnp.array(rng.normal(size=k*n).astype('float32')*0.1)
    tw=candidate._Tower(D,A,P,w,k)
    mask=fnp.array(1-np.eye(n,dtype='float32'))
    with BudgetContext(10**10):
        fused=np.asarray(candidate._dslice21_tree(tw,Y,R,mask,True))
        ref=np.asarray(candidate._dslice21_tree(tw,Y,R,mask,False))
        d3=np.asarray(candidate._feed_slices(A,P,Y,R,mask,False)[0])
    return float(np.max(abs(fused-ref))),float(np.max(abs(d3)))


class MLP:
    def __init__(self,n,depth):
        rng=np.random.default_rng(818+n*10+depth)
        self.width=n; self.seed=1
        self.weights=[fnp.array((rng.normal(size=(n,n))*
                       np.sqrt(2/n)).astype('float32')) for _ in range(depth)]


def run():
    result={'tree':{},'feed':{},'disabled':{},'active':{}}
    for n in (3,4):
        result['tree'][str(n)]=tree_case(n)
        assert result['tree'][str(n)]<2e-12
        for k in (1,2,3):
            err,signal=feed_case(n,k)
            result['feed'][f'{n}:{k}']={'fused_gap':err,'signal':signal}
            assert err<1e-5 and signal>0
        for depth in (2,3,4):
            mlp=MLP(n,depth)
            os.environ['ARC_WHEST_DISABLE_E14_TREE']='1'
            with BudgetContext(10**11):
                disabled=np.asarray(candidate.Estimator().predict(mlp,10**11))
            with BudgetContext(10**11):
                control=np.asarray(frozen.Estimator().predict(mlp,10**11))
            gap=float(np.max(abs(disabled-control)))
            result['disabled'][f'{n}:{depth}']=gap
            assert np.array_equal(disabled,control)
            del os.environ['ARC_WHEST_DISABLE_E14_TREE']
            with BudgetContext(10**11):
                active=np.asarray(candidate.Estimator().predict(mlp,10**11))
            result['active'][f'{n}:{depth}']={'finite':bool(np.isfinite(active).all()),
                                            'drift':float(np.max(abs(active-control)))}
            assert np.isfinite(active).all()
    return result


def trace_first_drift():
    """Retain the first large active drift and its actual state operands."""
    n, depth = 4, 3
    mlp = MLP(n, depth)
    est = candidate.Estimator(); est.setup(None)
    dtype = fnp.float32
    const = est._constants(n, dtype)
    table = candidate._tables()
    base = (fnp.zeros(n,dtype=dtype),fnp.eye(n,dtype=dtype),None,0.0)
    tree = base + (None,None,None)
    rows=[]
    def arr(x):
        return None if x is None else np.asarray(x).tolist()
    with BudgetContext(10**11):
        for l, w in enumerate(mlp.weights):
            base = frozen._layer(w,base,table,dtype,const,2.0,l==depth-1,None)
            tree = candidate._layer_tree(w,tree,table,dtype,const,2.0,l==depth-1,None)
            gap=float(np.max(abs(np.asarray(base[0])-np.asarray(tree[0]))))
            rows.append({'layer':l,'W':arr(w),'f21_mean':arr(base[0]),
                         'tree_mean':arr(tree[0]),'mean_max_gap':gap,
                         'f21_cov':arr(base[1]),'tree_cov':arr(tree[1]),
                         'f21_c4':float(base[3]),'tree_c4':float(tree[3]),
                         'g':arr(None if tree[4] is None else tree[4][0]),
                         'lambda':None if tree[4] is None else float(tree[4][1]),
                         'Cout_s':arr(None if tree[4] is None else tree[4][2]),
                         'Y':arr(tree[5]),'R':arr(tree[6]),
                         'source_ages':None if tree[2] is None else tree[2].k})
    first=next((r for r in rows if r['mean_max_gap']>0.1),None)
    return {'case':'n4_depth3_seed861','first_large_layer':None if first is None else first['layer'],
            'rows':rows,'scope':'actual F21/tree operands; no truth or accuracy conclusion'}


if __name__=='__main__':
    out=ROOT/'experiments'/'e14_tree_candidate'/'tiny_results.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    try:
        result=run(); result['drift_trace']=trace_first_drift(); result['passed']=True
    except Exception as exc:
        result=locals().get('result',{}); result['passed']=False
        result['first_exception']=repr(exc)
        out.write_text(json.dumps(result,indent=2),encoding='utf-8')
        raise
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
