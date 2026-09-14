"""E14 bounded localization controls, prediction committed c2bcfba.

Identity and QR representation roundtrips retain full rank/all ages and execute
the frozen reference _layer itself. The independent one-step control resets its
input to the exact frozen reference state at each layer. These are diagnostics,
not a new compressed candidate or a price claim at the production shape.
"""
from __future__ import annotations
import argparse
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import p2_compact_fullrank_f32_e14 as f32  # Bootstrap installed package first.
import flopscope as flops
import flopscope.numpy as fnp
from threadpoolctl import threadpool_limits
import p2_compact_fullrank_bridge_b1 as b1
import p2_compact_f32_backend_e14 as compact

ROOT = Path(__file__).resolve().parents[1]


def copy_source(source):
    values = {key: (None if value is None else int(value) if key == 'age'
                    else fnp.array(value, dtype=fnp.float32))
              for key, value in vars(source).items()}
    return compact.Source(**values)


def encode_state(reference, mode='identity'):
    mu, covariance, tower, c4 = reference
    if tower is None:
        return (mu, covariance, compact.Shared(), c4)
    sources = [copy_source(source) for source in b1.tower_sources(tower)]
    n = len(mu)
    if mode == 'identity':
        Q = fnp.eye(n, dtype=fnp.float32)
    elif mode == 'qr':
        # A single complete weighted-Gram sketch and full-rank QR. All source
        # legs are then projected directly in that same gauge. This isolates
        # roundtrip conditioning; it is not the candidate's cached rotation order.
        gram = fnp.zeros((n, n), dtype=fnp.float32)
        for source in sources:
            gram = gram + (source.A * source.dA[None, :]) @ source.A.T
            gram = gram + (source.P * source.dP[None, :]) @ source.P.T
        Q = fnp.linalg.qr(gram, mode='reduced')[0]
    else:
        raise ValueError(mode)
    factors = [compact.Factors(
        Q.T @ source.A, Q.T @ source.P, source.Z, source.L, source.s,
        source.e, source.w2, source.dA, source.dP, source.c1, source.c2,
        source.Y, source.age) for source in sources]
    gram = fnp.zeros((n, n), dtype=fnp.float32)
    for factor in factors:
        gram = gram + (factor.FA * factor.dA[None, :]) @ factor.FA.T
        gram = gram + (factor.FP * factor.dP[None, :]) @ factor.FP.T
    return mu, covariance, compact.Shared(Q, factors, gram), c4


def decode_state(frozen, state):
    mu, covariance, shared, c4 = state
    sources = sorted(compact.reconstruct(shared), key=lambda s: -s.age)
    if not sources:
        return mu, covariance, None, c4
    n, k = len(mu), len(sources)
    if [s.age for s in sources] != list(range(k - 1, -1, -1)):
        raise RuntimeError('Age inventory changed')
    tower = frozen._Tower(
        fnp.concatenate([compact.materialize(s) for s in sources], axis=1),
        fnp.concatenate([s.A for s in sources], axis=1),
        None if k == 1 else fnp.concatenate([s.P for s in sources[:-1]], axis=1),
        fnp.concatenate([s.w2 for s in sources]), k)
    # The youngest P is mathematically I. Frozen's symbolic layout intentionally
    # ignores its numerically reconstructed perturbation, which is recorded here.
    return mu, covariance, tower, c4


def measure_reference_order(weights, model, forced, mode):
    frozen = f32.load_reference(model, forced)
    n, depth = weights.shape[1], len(weights)
    rows = []
    with flops.BudgetContext(flop_budget=2**42, quiet=True) as budget:
        dtype = fnp.float32
        table = frozen._tables()
        const = {'off_mask': fnp.ones((n,n), dtype=dtype) - fnp.eye(n, dtype=dtype),
                 'ones_n': fnp.ones(n, dtype=dtype),
                 'wick': frozen._wick_program(table, dtype)}
        reference = (fnp.zeros(n,dtype=dtype), fnp.eye(n,dtype=dtype), None, 0.0)
        candidate = reference
        for layer, weight in enumerate(weights):
            W = fnp.array(weight.astype(np.float32).astype(np.float64), dtype=dtype)
            before = int(budget.flops_used)
            encoded = encode_state(candidate, mode)
            decoded = decode_state(frozen, encoded)
            roundtrip_flops = int(budget.flops_used) - before
            candidate, candidate_trace = f32.observe_layer(
                frozen, W, decoded, table, dtype, const, layer == depth-1)
            reference, reference_trace = f32.observe_layer(
                frozen, W, reference, table, dtype, const, layer == depth-1)
            values = []
            for checkpoint in ('before_pk', 'before_K', 'after_K'):
                for name in ('mu_pre','cov_pre','var','sigma','alpha','D3','D21',
                             'D4','D22','S','w','pK','K'):
                    got = candidate_trace[checkpoint].get(name)
                    want = reference_trace[checkpoint].get(name)
                    if got is None or want is None:
                        continue
                    pairs = ((str(key), got[key], want[key]) for key in got) if isinstance(got,dict) else [(name,got,want)]
                    for key,g,w in pairs:
                        ga,wa=f32.arr(g),f32.arr(w)
                        values.append({'checkpoint':checkpoint,'field':name+'/'+key,
                                       'got':ga.tolist(),'want':wa.tolist(),
                                       'max_abs':float(np.max(np.abs(ga-wa))),
                                       'bit_equal':bool(np.array_equal(ga,wa))})
            rows.append({'layer':layer, 'output':f32.output_check(candidate[0],reference[0]),
                         'encoded':f32.snapshot(encoded),'decoded':f32.snapshot(decoded),
                         'candidate':f32.snapshot(candidate),'reference':f32.snapshot(reference),
                         'values':values,'roundtrip_counted_flops':roundtrip_flops})
    return {'mode':mode,'rows':rows,'final_output':rows[-1]['output'],
            'all_observed_equal':all(v['bit_equal'] for row in rows for v in row['values']),
            'roundtrip_counted_flops':sum(r['roundtrip_counted_flops'] for r in rows),
            'roundtrip_cost_scope':'Tiny extra encoding/reconstruction operations; not production cost or a saving.',
            'qr_map_scope':'One complete Gram QR per layer; direct fullrank reprojection, not cached-rotation candidate order.'}


def measure_same_input(weights, model, forced, outdir, label):
    n = weights.shape[1]
    frozen = f32.load_reference(model, forced)
    bridge = SimpleNamespace(**vars(b1))
    bridge.FROZEN = model
    bridge.load_frozen = lambda: frozen
    namespace = dict(vars(f32))
    namespace.update(N=n,RANK=n,RESIDUAL_RANK=n,CHECKS=[],ACTIVITY=[],CONTEXTS=[],
                     fixture_weights=lambda:weights.copy(),b1=bridge,encode_state=encode_state)
    source = inspect.getsource(f32.run)
    source = source.replace('assert weights.shape == (6, N, N)',
                            'assert weights.shape == (len(weights), N, N)')
    source = source.replace('            mu, cov, shared, c4 = compact_state',
                            '            compact_state = encode_state(reference)\n            mu, cov, shared, c4 = compact_state')
    source = source.replace('activity[4]', 'activity[len(weights)-2]')
    source = source.replace('positive(checks, 4,', 'positive(checks, len(weights)-2,')
    source = source.replace('record(checks, 4,', 'record(checks, len(weights)-2,')
    source = source.replace('>= 4', '>= len(weights)-2')
    source = source.replace('activity.four_ages_nonfinal','activity.all_ages_nonfinal')
    source = source.replace('record(checks, 5, "final.output"', 'record(checks, len(weights)-1, "final.output"')
    path=outdir/(label+'_same_input_adapter.py')
    path.write_text(source,encoding='utf-8')
    exec(compile(source,str(path),'exec'),namespace)
    result=namespace['run']()
    result['checkpoints']=namespace['CONTEXTS']
    result['adapter_sha256']=f32.sha(path)
    result['first_nonzero_by_layer']=[]
    for layer in range(len(weights)):
        relevant=[r for r in result['checks'] if r['layer']==layer and 'normalized_error' in r and r['normalized_error']>0]
        result['first_nonzero_by_layer'].append(relevant[0] if relevant else None)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out-dir',type=Path,default=ROOT/'experiments'/'2026-09-14_e14'/'b4_real_backend')
    args=parser.parse_args(); outdir=args.out_dir.resolve(); outdir.mkdir(parents=True,exist_ok=True)
    summaries=[]
    with threadpool_limits(limits=1):
        for fixture,weights in [('b4',f32.fixture_weights()),('b3',b1.WEIGHTS)]:
            for control,model,forced in [('l2',b1.FROZEN,False),('f21_forced',f32.F21,True)]:
                label=fixture+'_'+control
                modes={mode:measure_reference_order(weights,model,forced,mode) for mode in ('identity','qr')}
                same=measure_same_input(weights,model,forced,outdir,label)
                result={'fixture':fixture,'control':control,'source_sha256':f32.sha(__file__),
                        'reference_sha256':f32.sha(model),'f32_driver_sha256':f32.sha(f32.__file__),
                        'backend_sha256':f32.sha(compact.__file__),'reference_order_controls':modes,
                        'same_input_independent':same}
                path=outdir/(label+'_localization.json')
                path.write_text(json.dumps(f32.json_safe(result),indent=2,allow_nan=False)+'\n',encoding='utf-8')
                final=next(r for r in same['checks'] if r['field']=='final.output')
                summary={'fixture':fixture,'control':control,'artifact':str(path.relative_to(ROOT)),
                         'sha256':f32.sha(path),'identity':modes['identity']['final_output'],
                         'identity_all_observed_equal':modes['identity']['all_observed_equal'],
                         'qr':modes['qr']['final_output'],
                         'same_input_independent':f32.output_check(final['got'],final['want'])}
                summaries.append(summary);print(json.dumps(summary),flush=True)
    result={'schema':'e14.b4_localization.v1','prediction_commit':'c2bcfba','cases':summaries,
            'scope':'Fixed fullrank diagnostics. Identity passing cannot grant QR/backend acceptance or compression.',
            'source_sha256':f32.sha(__file__),'original_B4_unchanged':f32.sha(f32.ORIGINAL)==f32.ORIGINAL_SHA256}
    (outdir/'b4_localization_summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
