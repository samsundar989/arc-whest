"""Exposed B3/B4 full-rank mixed-kernel discriminator; no labels or seed block."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

import p2_compact_fullrank_f32_e14 as fixture
import p2_mixed_residual_e14 as mixed
import flopscope as flops
import flopscope.numpy as fnp

ROOT = Path(__file__).resolve().parents[1]


def row_diff(layer, name, got, want):
    a, b = fixture.arr(got), fixture.arr(want)
    delta = np.abs(a-b)
    ij = np.unravel_index(np.argmax(delta), delta.shape) if delta.size else ()
    return {'layer':layer, 'field':name, 'max_abs':float(np.max(delta)) if delta.size else 0.0,
            'first_max_index':[int(i) for i in ij], 'got_operand':float(a[ij]) if delta.size else None,
            'want_operand':float(b[ij]) if delta.size else None,
            'got':a.tolist(), 'want':b.tolist()}


def case(weights, model, forced, mode):
    frozen = fixture.load_reference(model, forced)
    if mode == 'mixed':
        mixed.install(frozen)
    n, depth = weights.shape[1], len(weights)
    rows=[]
    with flops.BudgetContext(flop_budget=2**42,quiet=True) as budget:
        dtype=fnp.float32
        table=frozen._tables()
        const={'off_mask':fnp.ones((n,n),dtype=dtype)-fnp.eye(n,dtype=dtype),
               'ones_n':fnp.ones(n,dtype=dtype),
               'wick':frozen._wick_program(table,dtype)}
        state=(fnp.zeros(n,dtype=dtype),fnp.eye(n,dtype=dtype),None,0.0)
        for layer, weight in enumerate(weights):
            W=fnp.array(weight.astype(np.float32).astype(np.float64),dtype=dtype)
            state,trace=fixture.observe_layer(frozen,W,state,table,dtype,const,layer==depth-1)
            fields={}
            for checkpoint in ('before_pk','before_K','after_K'):
                fields[checkpoint]={}
                for name in ('mu_pre','cov_pre','var','sigma','alpha','S','w','pK','K'):
                    value=trace[checkpoint].get(name)
                    if value is not None:
                        fields[checkpoint][name]=fixture.snapshot(value)
            rows.append({'layer':layer,'mu':fixture.arr(state[0]).tolist(),
                         'trace':fields,'counted_flops_cumulative':int(budget.flops_used),
                         'tower':fixture.snapshot(state[2]) if mode!='mixed' else
                         tower_snapshot(state[2])})
    return {'mode':mode,'rows':rows,'final_mu':rows[-1]['mu'],
            'counted_flops':rows[-1]['counted_flops_cumulative']}


def tower_snapshot(tw):
    if tw is None:return None
    return {'k':tw.k,'D':fixture.snapshot(tw.D),'Q':fixture.snapshot(tw.Q),
            'FA':fixture.snapshot(tw.FA),'FP':fixture.snapshot(tw.FP),
            'G':fixture.snapshot(tw.G),'young_A':fixture.snapshot(tw.young_A),
            'young_P':fixture.snapshot(tw.young_P),'w':fixture.snapshot(tw.w)}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out-dir',type=Path,
        default=ROOT/'experiments'/'e14_slice_a');args=ap.parse_args()
    args.out_dir.mkdir(parents=True,exist_ok=True)
    summary=[]
    with threadpool_limits(limits=1):
        for label,weights in [('b3',fixture.b1.WEIGHTS),('b4',fixture.fixture_weights())]:
            for control,model,forced in [('l2',fixture.b1.FROZEN,False),
                                         ('f21_forced',fixture.F21,True)]:
                # L2 remains an exposed operational fixture, but its helper
                # internals differ; the direct mixed kernel targets F21.
                if control != 'f21_forced':continue
                reference=case(weights,model,forced,'control')
                no_basis=case(weights,model,forced,'no_basis')
                candidate=case(weights,model,forced,'mixed')
                refpath=args.out_dir/f'{label}_{control}_control.json'
                candpath=args.out_dir/f'{label}_{control}_mixed.json'
                nop= args.out_dir/f'{label}_{control}_no_basis.json'
                refpath.write_text(json.dumps(fixture.json_safe(reference),indent=2,allow_nan=False)+'\n',encoding='utf-8')
                candpath.write_text(json.dumps(fixture.json_safe(candidate),indent=2,allow_nan=False)+'\n',encoding='utf-8')
                nop.write_text(json.dumps(fixture.json_safe(no_basis),indent=2,allow_nan=False)+'\n',encoding='utf-8')
                output=row_diff(len(weights)-1,'final_mu',candidate['final_mu'],reference['final_mu'])
                first=None
                for c,r in zip(candidate['rows'],reference['rows']):
                    diff=row_diff(c['layer'],'mu',c['mu'],r['mu'])
                    if diff['max_abs']>0:
                        first=diff;break
                summary.append({'fixture':label,'control':control,'final_output':output,
                                'no_basis_exact_output':np.array_equal(np.asarray(no_basis['final_mu']),np.asarray(reference['final_mu'])),
                                'first_output_difference':first,
                                'reference_counted_flops':reference['counted_flops'],
                                'mixed_counted_flops':candidate['counted_flops'],
                                'reference_path':str(refpath.relative_to(ROOT)),
                                'no_basis_path':str(nop.relative_to(ROOT)),
                                'mixed_path':str(candpath.relative_to(ROOT))})
    (args.out_dir/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps([{'fixture':r['fixture'],'max_abs':r['final_output']['max_abs'],
                       'first_layer':None if r['first_output_difference'] is None else r['first_output_difference']['layer']}
                      for r in summary]))


if __name__=='__main__':main()
