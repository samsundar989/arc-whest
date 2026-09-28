"""Known-state feature/basis screen for all-distinct error. Diagnostic, never a score meter.

Streams one network and layer. Stores only small Gram matrices and subspace energies.
No fitting on DEV/VAL, no dense cubic tensor, no oracle directions used as predictors.
Run substantial work through workerq. See experiments/2026-09-28_ad_features/REVIEW.md.
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

ROOT = Path(__file__).resolve().parents[1]


def solve(A, b):
    sc = np.sqrt(np.maximum(np.diag(A), 1e-30))
    z = np.linalg.lstsq(A / sc[:, None] / sc[None, :], b / sc, rcond=1e-7)[0]
    return z / sc


def report(out):
    rows = [json.loads(p.read_text()) for p in sorted(out.glob('net_*.json'))]
    if not rows:
        raise SystemExit('No completed networks')
    lines = ['# Known-state all-distinct feature screen', '',
             'Prediction target is the normalized AD error on previously trained FIT networks. '
             'Leave-one-network-out fitting below is diagnostic, not independent final validation. '
             'Basis projection energies use oracle coefficients, whereas Gram fits use only other networks. '
             'No downstream score benefit follows from residual prediction alone.', '']
    for layer in rows[0]['layers']:
        rr = [r['layers'][layer] for r in rows]
        names = rr[0]['names']
        lines += [f'## Consumer layer {layer}', '', '| target | feature family | LONO residual energy removed |', '|---|---|---:|']
        groups = {'local': [i for i,n in enumerate(names) if not n.startswith('global:')],
                  'all': list(range(len(names)))}
        for target in ('E', 'Eorth'):
            for grp, ix in groups.items():
                removed = []
                for i,r in enumerate(rr):
                    others = [v for j,v in enumerate(rr) if j != i]
                    if not others:
                        continue
                    A = sum(np.array(v['gram']['XtX']) for v in others)[np.ix_(ix, ix)]
                    b = sum(np.array(v['gram'][target]['Xty']) for v in others)[ix]
                    th = solve(A,b)
                    At = np.array(r['gram']['XtX'])[np.ix_(ix,ix)]
                    bt = np.array(r['gram'][target]['Xty'])[ix]
                    yt = r['gram'][target]['yty']
                    removed.append(float((2*th@bt-th@At@th)/yt))
                lines.append(f'| {target} | {grp} | {np.median(removed):.4f} |')
        lines += ['', '| known basis | rank | median right capture | median left capture | median bilateral capture |', '|---|---:|---:|---:|---:|']
        for basis in rr[0]['subspaces']:
            for rank in rr[0]['subspaces'][basis]:
                vals=[r['subspaces'][basis][rank] for r in rr]
                lines.append(f'| {basis} | {rank} | '+ ' | '.join(f'{np.median([v[k] for v in vals]):.4f}' for k in ('right','left','both'))+' |')
    lines += ['', 'Final-layer reference floor approximately 7.7e-11; moment-target uncertainty is distinct. '
              'Current probe baseline final MSE per net: '+str({r['net']:r['baseline_final_mse'] for r in rows})]
    (out/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


def run(a):
    from k111_diag import Diag, rep3, rep21
    import torch
    torch.set_num_threads(2)
    args=SimpleNamespace(cand=str(ROOT/'estimator_k3_v62_s1.py'), params=str(ROOT/a.params),
                         h=.05,max_layer=99,gain_file='')
    d=Diag(args)
    layers={int(x) for x in a.layers.split(',')}
    off=d.off64
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    meta={'params':a.params,'params_sha256':hashlib.sha256(Path(args.params).read_bytes()).hexdigest(),
          'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'layers':sorted(layers),'nets':a.nets,'source':args.cand}
    (out/'metadata.json').write_text(json.dumps(meta,indent=2))
    start,end=map(int,a.nets.split('-'))
    for g in range(start,end+1):
        dest=out/f'net_{g:03d}.json'
        if dest.exists():
            continue
        t0=time.time(); torch.cuda.reset_peak_memory_stats()
        w,truth=d.data.net('full',g); labs=d.labels(g)
        result={'net':g,'layers':{}}
        rec={}
        def hook(l,name,value,*aux):
            if name=='K':
                if not all(bool(torch.isfinite(x).all()) for x in value.values()):
                    raise FloatingPointError(f'nonfinite K at {g}/{l}')
            elif name in ('mu','var','C11','D3','D21'):
                if not bool(torch.isfinite(value).all()):
                    raise FloatingPointError(f'nonfinite {name} at {g}/{l}')
                rec[name]=value.detach()
            return value
        state=(d.T.fnp.zeros(d.n),d.T.fnp.eye(d.n),None,0.0,d.T.fnp.zeros(d.n))
        d.mod._A2=hook
        with torch.no_grad():
            for l in range(d.depth):
                prev=state; rec.clear()
                state=d.tw._step(l,w[l],state)
                if l not in layers:
                    continue
                # The production final layer intentionally omits pairwise state. Reconstruct
                # it for this diagnostic only; this extra research work is never a cost claim.
                if 'C11' not in rec:
                    rec['C11']=(w[l].T@prev[1]@w[l])*off.float()
                if 'D21' not in rec:
                    ad=d.mod._advance(w[l].T.contiguous(),prev[2])
                    rec['D21']=d.mod._dslice21(ad,off.float())
                    del ad
                W=w[l].double()
                t,S=d.slices(d.mod._advance(d.I32,prev[2]),off.float())
                lp=d.label_parts(labs,l-1,W)
                md3=rec['D3'].double()-rep3(t,S*off,W)
                md21=(rec['D21'].double()-rep21(t,S*off,W))*off
                sig=rec['var'].double().clamp_min(1e-20).sqrt()
                scale=sig[:,None]**2*sig[None,:]
                E=(lp['AD21']-md21)/scale
                e3=(lp['AD3']-md3)/(sig**3)
                E=E*off+torch.diag(e3)
                Mod=md21/scale+torch.diag(md3/(sig**3))
                Eorth=E-((E*Mod).sum()/(Mod*Mod).sum())*Mod
                R=rec['C11'].double()/(sig[:,None]*sig[None,:])+torch.eye(d.n,device=d.dev,dtype=torch.float64)
                D=rec['D21'].double()/scale+torch.diag(rec['D3'].double()/(sig**3))
                skew=rec['D3'].double()/sig**3
                alpha=(rec['mu'].double()/sig).clamp(-8,8)
                B=[D,D.T,R*skew[:,None],R*skew[None,:],D*alpha[:,None],D*alpha[None,:]]
                names=['D','DT','R_skew_i','R_skew_j','D_alpha_i','D_alpha_j']
                n=d.n
                R2=R@R/n
                glob=[D@R/n,R@D/n,D.T@R/n,R@D.T/n,
                      R2*skew[:,None],R2*skew[None,:],
                      (R*R)@D/n,D@(R*R)/n,
                      ((R*R)@R/n)*skew[:,None],(R@(R*R)/n)*skew[None,:]]
                B+=glob
                names+=['global:'+s for s in ('DR','RD','DTR','RDT','R2_skew_i','R2_skew_j','RRhad_D','D_RRhad','RRhad_R_skew_i','R_RRhad_skew_j')]
                X=torch.stack([b.flatten() for b in B])
                gram={'XtX':(X@X.T).cpu().tolist()}
                for key,Y in (('E',E),('Eorth',Eorth)):
                    gram[key]={'Xty':(X@Y.flatten()).cpu().tolist(),'yty':float((Y*Y).sum())}
                del X,B,glob,R2
                # Known right modes from covariance. Left modes may be quadratic functions of them.
                ev,Q=torch.linalg.eigh(R)
                Q=Q.flip(1)
                q=Q[:,:16]
                ij=torch.triu_indices(16,16,device=d.dev)
                lift=q[:,ij[0]]*q[:,ij[1]]
                lift=lift-lift.mean(0,keepdim=True)
                Qlift=torch.linalg.qr(lift,mode='reduced').Q
                sub={}
                den=float((Eorth*Eorth).sum())
                for label,QL,QR in (('covariance',Q,Q),('quadratic_left_covariance_right',Qlift,Q)):
                    vals={}
                    for rk in (8,16,32,64,128):
                        left=QL[:,:rk].T@Eorth; right=Eorth@QR[:,:rk]
                        core=QL[:,:rk].T@right
                        vals[rk]={'left':float((left*left).sum())/den,'right':float((right*right).sum())/den,
                                  'both':float((core*core).sum())/den}
                    sub[label]=vals
                result['layers'][l]={'names':names,'gram':gram,'subspaces':sub,
                                      'AD_relative_error':float((lp['AD21']-md21).norm()/lp['AD21'].norm())}
                print(f'net={g} layer={l} elapsed={time.time()-t0:.1f}s',flush=True)
                del E,Eorth,Mod,R,D,Q,Qlift,prev,lift
        d.mod._A2=None
        m=state[0].clamp_min(0).double()
        result['baseline_final_mse']=float(((m-torch.from_numpy(truth[-1]).to(d.dev))**2).mean())
        result['seconds']=time.time()-t0
        result['peak_vram_gib']=torch.cuda.max_memory_allocated()/2**30
        dest.write_text(json.dumps(result))
        progress=os.environ.get('WORKERQ_PROGRESS')
        if progress:
            Path(progress).write_text(str((g-start+1)/(end-start+1)))
        print(f'COMPLETE net={g} seconds={result["seconds"]:.1f} peak={result["peak_vram_gib"]:.2f}',flush=True)
        del labs,w,state


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('cmd',choices=['run','report'])
    p.add_argument('--nets',default='0-7')
    p.add_argument('--layers',default='3,6,9,12,15')
    p.add_argument('--params',default='experiments/2026-09-27_S2t3/params_step736.npz')
    p.add_argument('--out',default=str(ROOT/'experiments/2026-09-28_ad_features'))
    a=p.parse_args()
    run(a) if a.cmd=='run' else report(Path(a.out))
