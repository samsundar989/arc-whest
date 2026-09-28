"""Consistent downstream oracle for known-basis symmetric AD corrections.

Offline research only. Recomputes error/core on the CURRENT intervention state;
no label-derived basis or fitted per-net core is a deployable predictor. Hooks
see both jointly fitted slices, prepared before the next layer starts. Direct
consumers use the correction, while compensation uses actual carried slices.

Resource review: existing labels/weights, one network/arm at a time, no autograd,
no layer-state cache or dense n^3 tensor, CPU2, per-net atomic checkpoint. q32
screen measured4.44GiB GPU; request RAM16/VRAM10/CPU2, initial FIT0-3~1-3minutes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def layer_set(spec):
    out=set()
    for s in spec.split(','):
        if '-' in s:
            lo,hi=map(int,s.split('-'));out.update(range(lo,hi+1))
        else:out.add(int(s))
    return out


def run(a):
    import torch
    from k111_diag import Diag,rep3,rep21
    from k111_consistent_probe import patch_layer,write_json
    from k111_core_probe import fit_core
    torch.set_num_threads(2)
    args=SimpleNamespace(cand=str(ROOT/'estimator_k3_v62_s1.py'),params=str(ROOT/a.params),
                         h=.05,max_layer=99,gain_file='')
    d=Diag(args)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    legacy,consistent=patch_layer(d.mod,out)
    layers=layer_set(a.layers)
    arms=a.arms.split(',')
    allowed={'ADorth','scalar','q32w1','q32w1024'}
    if not set(arms)<=allowed:raise ValueError(arms)
    off=d.off64;off32=off.float();eye=torch.eye(d.n,dtype=torch.float64,device=d.dev)
    write_json(out/'metadata.json',{'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'params':a.params,'params_sha256':hashlib.sha256(Path(args.params).read_bytes()).hexdigest(),
               'candidate_sha256':hashlib.sha256(Path(args.cand).read_bytes()).hexdigest(),
               'core_fitter_sha256':hashlib.sha256((ROOT/'tools/k111_core_probe.py').read_bytes()).hexdigest(),
               'nets':a.nets,'layers':sorted(layers),'arms':arms,'cg_steps':a.cg_steps,
               'semantics':'normalized AD correction, one scaling direction removed; saved actual slices compensate; fitted cores oracle-only',
               'final_layer_noise_floor_approx':7.7e-11})
    lo,hi=map(int,a.nets.split('-'))
    def finite(x,where):
        if not bool(torch.isfinite(x).all()):raise FloatingPointError(where)
    for g in range(lo,hi+1):
        dest=out/f'net_{g:03d}.json'
        if dest.exists():continue
        t0=time.time();torch.cuda.reset_peak_memory_stats()
        w,truth=d.data.net('full',g);labels=d.labels(g)
        results={'net':g,'arms':{}}
        references={}
        for arm in ['base_legacy','base_consistent']+arms:
            d.mod._layer=legacy if arm=='base_legacy' else consistent
            ctx={};records={};matches={}
            def prepare(l,state):
                ctx.clear()
                c=l+1
                if arm.startswith('base') or c not in layers:return
                W=w[c];WT=W.T
                t,S=d.slices(d.mod._advance(d.I32,state[2]),off32)
                # Production passes the transposed view, not a contiguous copy.
                # CUDA may select a different GEMM reduction for that copy;
                # duplicate research reads must preserve layout as well as values.
                ad=d.mod._advance(WT,state[2])
                D3,D21=d.slices(ad,off32);del ad
                lp=d.label_parts(labels,l,W.double())
                md3=D3-rep3(t,S*off,W.double())
                md21=(D21-rep21(t,S*off,W.double()))*off
                # Match production sigma including the final variance-only path.
                if d.mod._P4_FINAL_VAR and c==d.depth-1:
                    X=d.mod._mm(WT,state[1])
                    var=((X*WT).sum(1)).clamp_min(1e-12)
                    C=(X@W).double()
                else:
                    C=(d.mod._p4_sym_sandwich(WT,state[1],W) if d.mod._P4_BLOCKSYM
                       else d.mod._mm(d.mod._mm(WT,state[1]),W))
                    var=torch.diagonal(C).clamp_min(1e-12)
                    C=C.double()
                sig=var.double().sqrt()
                scale=sig[:,None]**2*sig[None,:]
                E=(lp['AD21']-md21)/scale*off+torch.diag((lp['AD3']-md3)/sig**3)
                Mod=md21/scale+torch.diag(md3/sig**3)
                den=(Mod*Mod).sum()
                beta=(E*Mod).sum()/den if float(den)>1e-30 else den*0
                Eorth=E-beta*Mod
                if arm=='scalar':
                    correction=beta*Mod
                    info={'scalar':float(beta)}
                elif arm=='ADorth':
                    correction=Eorth
                    info={'scalar_removed':float(beta)}
                else:
                    R=(C*off)/(sig[:,None]*sig[None,:])+eye
                    _,Q=torch.linalg.eigh(R)
                    U=Q[:,-32:].flip(1).contiguous()
                    weight=1024. if arm.endswith('1024') else 1.
                    _,correction,info=fit_core(U,Eorth,weight,steps=a.cg_steps,tol=1e-7)
                    info['scalar_removed']=float(beta)
                finite(correction,f'{g}/{arm}/{c} correction')
                ctx.update(layer=c,D3=D3,D21=D21,var=var,
                           delta3=torch.diagonal(correction)*sig**3,
                           delta21=correction*scale*off)
                records[c]=info

            def hook(l,name,value,*aux):
                if name=='K':
                    for key,x in value.items():finite(x,f'{g}/{arm}/{l} K{key}')
                elif name in ('mu','var','C11','D3','D21','dG'):
                    finite(value,f'{g}/{arm}/{l} {name}')
                if ctx and ctx['layer']==l:
                    if name in ('D3','D21','var'):
                        # The duplicate research readout must match the actual consumer.
                        ref=ctx[name].to(value.dtype)
                        delta=float((value-ref).abs().max())
                        matches[f'{l}:{name}']=delta
                        if not torch.equal(value,ref):
                            raise AssertionError(f'prepared {name} differs at {g}/{arm}/{l}: {delta}')
                    if name in ('D3','D21'):
                        correction=ctx['delta3' if name=='D3' else 'delta21']
                        changed=(value.double()+correction).to(value.dtype)
                        finite(changed,f'{g}/{arm}/{l} injected {name}')
                        return changed
                return value

            mus,_=d.chain_run(w,hook,prepare)
            mse=d.mse(mus,truth)
            if arm.startswith('base'):
                references[arm]={l:m.clone() for l,m in mus.items()}
            results['arms'][arm]={'final':mse[d.depth-1],'rows':mse,'core_fits':records,
                                  'prepared_consumer_max_abs':max(matches.values(),default=0.)}
            print(f'net={g} arm={arm} final={mse[d.depth-1]:.10e} elapsed={time.time()-t0:.1f}s',flush=True)
            del mus
        same=all(torch.equal(references['base_legacy'][l],references['base_consistent'][l]) for l in references['base_legacy'])
        if not same:raise AssertionError('baseline mismatch under consistent compensation')
        results['baseline_all_layers_bit_equal']=same
        base=results['arms']['base_consistent']['final']
        for v in results['arms'].values():v['relative_final_change']=v['final']/base-1
        results['seconds']=time.time()-t0;results['peak_vram_gib']=torch.cuda.max_memory_allocated()/2**30
        write_json(dest,results)
        progress=os.environ.get('WORKERQ_PROGRESS')
        if progress:Path(progress).write_text(str((g-lo+1)/(hi-lo+1)))
        print(f'COMPLETE net={g} peak={results["peak_vram_gib"]:.2f}GiB',flush=True)
        del w,labels,references


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--nets',default='0-3');p.add_argument('--layers',default='1-15')
    p.add_argument('--arms',default='ADorth,q32w1,q32w1024')
    p.add_argument('--cg-steps',type=int,default=100)
    p.add_argument('--params',default='experiments/2026-09-27_S2t3/params_step736.npz')
    p.add_argument('--out',default=str(ROOT/'experiments/2026-09-28_review_arch/core_oracle_fit4'))
    run(p.parse_args())
