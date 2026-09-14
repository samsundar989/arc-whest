"""Label-free active fused/unfused and generation-cap recurrence checks."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import flopscope.numpy as fnp
from flopscope import BudgetContext

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import estimator_e14_tree as e
import estimator_k3_strassen_L4_f21 as frozen


class MLP:
    def __init__(self, weights):
        self.width=weights[0].shape[0]
        self.weights=[fnp.array(x.astype('float32')) for x in weights]
        self.seed=1


def weights(n,kind):
    if kind=='admissible_controlled':
        x=np.eye(n)+0.03*np.ones((n,n))
        return [x.copy() for _ in range(4)]
    rng=np.random.default_rng(818+n*10+4)
    return [rng.normal(size=(n,n))*np.sqrt(2/n) for _ in range(4)]


def evaluate(mlp,fused,gens=None):
    e._E14_QUERY_FUSED=fused
    est=e.Estimator(); est.setup(None)
    if gens is None:
        with BudgetContext(10**11):
            out=np.asarray(est.predict(mlp,10**11))
        return out,est.last_run,None
    table=e._tables(); const=est._constants(mlp.width,fnp.float32)
    n=mlp.width
    state=(fnp.zeros(n,dtype=fnp.float32),fnp.eye(n,dtype=fnp.float32),
           None,0.0,None,None,None)
    metadata=[]
    with BudgetContext(10**11):
        for l,w in enumerate(mlp.weights):
            state=e._layer_tree(w,state,table,fnp.float32,const,2.0,
                                 l==len(mlp.weights)-1,gens)
            tw,Y,R=state[2],state[5],state[6]
            metadata.append({'layer':l,'source_ages':None if tw is None else tw.k,
                             'y_columns':None if Y is None else Y.shape[1],
                             'r_columns':None if R is None else R.shape[1],
                             'cov_diagonal':None if state[1] is None else
                                 np.diag(np.asarray(state[1])).tolist(),
                             'mean':np.asarray(state[0]).tolist()})
            if l<len(mlp.weights)-1:
                assert tw.k==Y.shape[1]==R.shape[1]
                assert tw.k<=gens
    return np.asarray(state[0]),{'mode':'forced_direct_cap','gens':gens},metadata


def run():
    result={'cases':[],'scope':'same active path f32 fused/unfused; no labels or truth'}
    for n in (3,4):
        for kind in ('admissible_controlled','negative_variance'):
            mlp=MLP(weights(n,kind))
            fused,mode,_=evaluate(mlp,True)
            unfused,_,_=evaluate(mlp,False)
            diff=fused-unfused
            with BudgetContext(10**11):
                f21=np.asarray(frozen.Estimator().predict(mlp,10**11))
            caps={}
            forced_last={}
            for g in (1,2):
                final,_,meta=evaluate(mlp,True,g)
                caps[str(g)]=meta
                forced_last[str(g)]=final
            reserve=(4*(4-1)*n*n*(2*n-1)+(5000*4+100*4*4)*n*n)
            budget=reserve+int(np.ceil(e._projected_flops(n,4,1)/0.95))+1
            e._E14_QUERY_FUSED=True
            chooser=e.Estimator(); chooser.setup(None)
            with BudgetContext(10**11):
                budget_out=np.asarray(chooser.predict(mlp,budget))
            budget_mode=dict(chooser.last_run)
            assert budget_mode['mode']=='e14_tree' and budget_mode['generations']==1
            assert budget_mode['projected_flops']<=budget
            forced_gap=float(np.max(abs(budget_out[-1]-np.maximum(forced_last['1'],0))))
            assert forced_gap<1e-6
            regen_budget=reserve+int(np.ceil(e._projected_flops(n,4,1)/0.95))-1000
            fallback=e.Estimator(); fallback.setup(None)
            with BudgetContext(10**11):
                regen_out=np.asarray(fallback.predict(mlp,regen_budget))
            f21_regen=frozen.Estimator(); f21_regen.setup(None)
            with BudgetContext(10**11):
                f21_regen_out=np.asarray(f21_regen.predict(mlp,regen_budget-reserve))
            regen_gap=float(np.max(abs(regen_out-f21_regen_out)))
            assert fallback.last_run['mode']=='regen' and regen_gap==0.0
            result['cases'].append({'n':n,'kind':kind,'weights':[x.tolist() for x in weights(n,kind)],
                'fused':fused.tolist(),'unfused':unfused.tolist(),'f21':f21.tolist(),
                'max_fused_unfused_gap':float(np.max(abs(diff))),
                'difference':diff.tolist(),'mode':mode,'caps':caps,
                'budget_chooser':{'budget':budget,'mode':budget_mode,
                                  'output':budget_out.tolist(),
                                  'forced_gen1_final_gap':forced_gap},
                'regen_fallback':{'budget':regen_budget,
                                  'mode':dict(fallback.last_run),
                                  'f21_mode':dict(f21_regen.last_run),
                                  'max_output_gap':regen_gap,
                                  'output':regen_out.tolist()},
                'finite':bool(np.isfinite(fused).all() and np.isfinite(unfused).all())})
    e._E14_QUERY_FUSED=True
    result['max_fused_unfused_gap']=max(c['max_fused_unfused_gap'] for c in result['cases'])
    result['passed']=all(c['finite'] for c in result['cases'])
    return result


if __name__=='__main__':
    out=ROOT/'experiments'/'e14_tree_candidate'/'recurrence_v2.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    try: result=run()
    except Exception as exc:
        result={'passed':False,'first_exception':repr(exc)}
        out.write_text(json.dumps(result,indent=2),encoding='utf-8')
        raise
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'max_fused_unfused_gap':result['max_fused_unfused_gap'],
                      'passed':result['passed']},indent=2))
