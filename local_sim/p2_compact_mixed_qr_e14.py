"""E14 mixed-f64 QR representation remedy, prediction commit 832c0d6.

Only QR/basis/source representation uses f64. Decoded source arrays are cast
back to f32 before the unchanged frozen reference-order nonlinear operator.
No zero masking, rank reduction, labels, or production-shape claim.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import p2_compact_fullrank_f32_e14 as f32  # Real-package bootstrap.
import flopscope as flops
import flopscope.numpy as fnp
import p2_compact_f32_backend_e14 as compact
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]


def source_views(tower):
    """Source views from wrapped f32 tower; no unmetered casting in encoding."""
    if tower is None:
        return []
    n, k = tower.A_D.shape[0], tower.k
    sources=[]
    for j in range(k):
        part=slice(j*n,(j+1)*n)
        P=fnp.eye(n,dtype=fnp.float32) if j==k-1 else tower.M[:,part]
        sources.append({'A':tower.A_L[:,part], 'P':P,
                        'D':tower.A_D[:,part], 'w2':tower.w2b[part], 'age':k-1-j})
    return sources


def encode64(state):
    mu,cov,tower,c4=state
    if tower is None:
        return (mu,cov,compact.Shared(),c4)
    n=len(mu)
    eye=fnp.eye(n,dtype=fnp.float64)
    zero=fnp.zeros(n,dtype=fnp.float64)
    one=fnp.ones(n,dtype=fnp.float64)
    sources=[]
    for view in source_views(tower):
        sources.append(compact.Source(
            A=view['A'].astype(fnp.float64),P=view['P'].astype(fnp.float64),
            Z=view['D'].astype(fnp.float64),L=eye,s=zero,e=zero,
            w2=view['w2'].astype(fnp.float64),dA=one,dP=one,
            c1=None,c2=None,Y=None,age=view['age']))
    gram=fnp.zeros((n,n),dtype=fnp.float64)
    for source in sources:
        gram=gram+(source.A*source.dA[None,:])@source.A.T
        gram=gram+(source.P*source.dP[None,:])@source.P.T
    Q=fnp.linalg.qr(gram,mode='reduced')[0]
    factors=[compact.Factors(Q.T@s.A,Q.T@s.P,s.Z,s.L,s.s,s.e,s.w2,
                             s.dA,s.dP,s.c1,s.c2,s.Y,s.age) for s in sources]
    carried=fnp.zeros((n,n),dtype=fnp.float64)
    for factor in factors:
        carried=carried+(factor.FA*factor.dA[None,:])@factor.FA.T
        carried=carried+(factor.FP*factor.dP[None,:])@factor.FP.T
    return mu,cov,compact.Shared(Q,factors,carried),c4


def decode32(frozen,state):
    mu,cov,shared,c4=state
    if shared.Q is None:
        return (mu,cov,None,c4),[]
    sources=sorted(compact.reconstruct(shared),key=lambda s:-s.age)
    k=len(sources)
    D=[compact.materialize(s) for s in sources]
    tower=frozen._Tower(
        fnp.concatenate(D,axis=1).astype(fnp.float32),
        fnp.concatenate([s.A for s in sources],axis=1).astype(fnp.float32),
        None if k==1 else fnp.concatenate([s.P for s in sources[:-1]],axis=1).astype(fnp.float32),
        fnp.concatenate([s.w2 for s in sources]).astype(fnp.float32),k)
    return (mu,cov,tower,c4),sources


def exact_field(name,got,want):
    if got is None or want is None:
        return {'field':name,'passed':got is want,'none':True,'zero_mismatches':0,'bit_mismatches':0}
    g,w=f32.arr(got).astype(np.float32),f32.arr(want).astype(np.float32)
    if g.shape != w.shape:
        raise RuntimeError('Roundtrip shape mismatch')
    unequal=g.view(np.uint32)!=w.view(np.uint32)
    zeros=(w==0)&(g!=0)
    return {'field':name,'got':g.tolist(),'want':w.tolist(),
            'bit_mismatches':int(unequal.sum()),'zero_mismatches':int(zeros.sum()),
            'max_abs':float(np.max(np.abs(g.astype(float)-w.astype(float)))) if g.size else 0,
            'passed':bool(not unequal.any()),'got_dtype':str(got.dtype),'want_dtype':str(want.dtype)}


def payload_inventory(shared):
    """Unique-wrapper array payload, not allocator/RSS/process peak memory."""
    rows=[]; seen=set()
    if shared.Q is None:
        return rows
    def add(name,value):
        if value is not None and hasattr(value,'dtype') and id(value) not in seen:
            seen.add(id(value));rows.append({'field':name,'dtype':str(value.dtype),
                                            'shape':list(value.shape),'bytes':int(value.nbytes)})
    add('Q',shared.Q);add('gram',shared.gram)
    for i,factor in enumerate(shared.factors):
        for name,value in vars(factor).items():
            add(str(i)+'.'+name,value)
    return rows


def run_case(weights,model,forced):
    frozen=f32.load_reference(model,forced)
    n,depth=weights.shape[1],len(weights);rows=[]
    with flops.BudgetContext(flop_budget=2**42,quiet=True) as budget:
        dtype=fnp.float32;table=frozen._tables()
        const={'off_mask':fnp.ones((n,n),dtype=dtype)-fnp.eye(n,dtype=dtype),
               'ones_n':fnp.ones(n,dtype=dtype),'wick':frozen._wick_program(table,dtype)}
        ref=(fnp.zeros(n,dtype=dtype),fnp.eye(n,dtype=dtype),None,0.0);candidate=ref
        for layer,weight in enumerate(weights):
            W=fnp.array(weight.astype(np.float32).astype(np.float64),dtype=dtype)
            before=int(budget.flops_used);encoded=encode64(candidate)
            encoded_flops=int(budget.flops_used)-before
            before=int(budget.flops_used);decoded,sources=decode32(frozen,encoded)
            decoded_flops=int(budget.flops_used)-before
            fields=[]
            if candidate[2] is not None:
                for field in ('A_D','A_L','M','w2b'):
                    fields.append(exact_field(field,getattr(decoded[2],field),getattr(candidate[2],field)))
                # Youngest P remains symbolically I in the frozen layout. Its
                # reconstructed value is an additional diagnostic, not masked.
                fields.append(exact_field('youngest_P_unconsumed',sources[-1].P,
                                          fnp.eye(n,dtype=dtype)))
            inventory=payload_inventory(encoded[2])
            before=int(budget.flops_used)
            candidate,candidate_trace=f32.observe_layer(frozen,W,decoded,table,dtype,const,layer==depth-1)
            candidate_flops=int(budget.flops_used)-before
            before=int(budget.flops_used)
            ref,ref_trace=f32.observe_layer(frozen,W,ref,table,dtype,const,layer==depth-1)
            reference_flops=int(budget.flops_used)-before
            checks=[]
            for checkpoint in ('before_pk','before_K','after_K'):
                for name in ('mu_pre','cov_pre','var','sigma','alpha','S','w','pK','K'):
                    g,w=candidate_trace[checkpoint].get(name),ref_trace[checkpoint].get(name)
                    if g is None or w is None:continue
                    pairs=((str(key),g[key],w[key]) for key in g) if isinstance(g,dict) else [(name,g,w)]
                    for key,got,want in pairs:
                        f32.record(checks,layer,checkpoint+'/'+name+'/'+key,got,want)
            rows.append({'layer':layer,'roundtrip_fields':fields,
                         'encoded':f32.snapshot(encoded),'decoded':f32.snapshot(decoded),
                         'operator_checks':checks,'output':f32.output_check(candidate[0],ref[0]),
                         'payload_inventory':inventory,'unique_wrapper_payload_bytes':sum(r['bytes'] for r in inventory),
                         'encoder_counted_flops':encoded_flops,'decoder_counted_flops':decoded_flops,
                         'candidate_operator_counted_flops':candidate_flops,
                         'reference_operator_counted_flops':reference_flops})
    consumed=[f for row in rows for f in row['roundtrip_fields'] if f['field']!='youngest_P_unconsumed']
    return {'rows':rows,'final_output':rows[-1]['output'],
            'exact_consumed_roundtrip':all(f['passed'] for f in consumed),
            'consumed_bit_mismatches':sum(f['bit_mismatches'] for f in consumed),
            'consumed_zero_mismatches':sum(f['zero_mismatches'] for f in consumed),
            'first_roundtrip_mismatch':next(({'layer':r['layer'],**f} for r in rows for f in r['roundtrip_fields']
                                              if f['field']!='youngest_P_unconsumed' and not f['passed']),None),
            'original_tolerance_failures':sum(not c['passed'] for r in rows for c in r['operator_checks']),
            'encoder_counted_flops':sum(r['encoder_counted_flops'] for r in rows),
            'decoder_counted_flops':sum(r['decoder_counted_flops'] for r in rows),
            'candidate_operator_counted_flops':sum(r['candidate_operator_counted_flops'] for r in rows),
            'reference_operator_counted_flops':sum(r['reference_operator_counted_flops'] for r in rows),
            'max_unique_wrapper_payload_bytes':max(r['unique_wrapper_payload_bytes'] for r in rows)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out-dir',type=Path,
        default=ROOT/'experiments'/'2026-09-14_e14'/'b4_mixed_qr')
    args=parser.parse_args();out=args.out_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    summary=[]
    with threadpool_limits(limits=1):
        backend=threadpool_info()
        for fixture,weights in [('b4',f32.fixture_weights()),('b3',f32.b1.WEIGHTS)]:
            for control,model,forced in [('l2',f32.b1.FROZEN,False),('f21_forced',f32.F21,True)]:
                result=run_case(weights,model,forced)
                result.update(fixture=fixture,control=control,reference_sha256=f32.sha(model),
                              driver_sha256=f32.sha(__file__),f32_driver_sha256=f32.sha(f32.__file__),
                              compact_backend_sha256=f32.sha(compact.__file__))
                path=out/(fixture+'_'+control+'_mixed_qr.json')
                path.write_text(json.dumps(f32.json_safe(result),indent=2,allow_nan=False)+'\n',encoding='utf-8')
                row={key:result[key] for key in ('fixture','control','final_output','exact_consumed_roundtrip',
                    'consumed_bit_mismatches','consumed_zero_mismatches','original_tolerance_failures',
                    'encoder_counted_flops','decoder_counted_flops','candidate_operator_counted_flops',
                    'reference_operator_counted_flops','max_unique_wrapper_payload_bytes')}
                row.update(artifact=str(path.relative_to(ROOT)),sha256=f32.sha(path));summary.append(row)
                print(json.dumps(row),flush=True)
    result={'schema':'e14.b4_mixed_qr.v1','prediction_commit':'832c0d6','cases':summary,
            'flopscope_import_path':flops.__file__,'fnp_import_path':fnp.__file__,'threadpools':backend,
            'scope':'Fullrank mixed-f64 QR/dense-reference wrapper only; no zero masking, compression, production L4 or accuracy claim.',
            'original_B4_unchanged':f32.sha(f32.ORIGINAL)==f32.ORIGINAL_SHA256}
    (out/'b4_mixed_qr_summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
