"""Same saved E14 n3 fixture: unchanged active f64 path versus dense f64 recurrence.

This is a label-free precision discriminator, not a proposed estimator regime.
"""
import json
from pathlib import Path

import numpy as np
import flopscope.numpy as fnp
from flopscope import BudgetContext

import e14_tree_full_oracle as dense
from e14_tree_layer_oracle import dense_tower

ROOT = Path(__file__).resolve().parents[1]
SAVED = ROOT / 'experiments/e14_tree_candidate/full_oracle_v4.json'
OUT = ROOT / 'experiments/e14_tree_candidate/f64_diagnostic.json'


def wick64(mu, cov, table, const):
    var = np.maximum(np.diag(cov), 1e-12)
    sigma = np.sqrt(var)
    alpha = np.clip(mu / sigma, -dense.active.ALPHA_CAP,
                    dense.active.ALPHA_CAP)
    with BudgetContext(10**8):
        got = dense.active._wick(fnp.array(mu, dtype=fnp.float64),
                                 fnp.array(sigma, dtype=fnp.float64),
                                 fnp.array(alpha, dtype=fnp.float64),
                                 table, fnp.float64, const['wick'])
    return {key: dense.arr(value) for key, value in got.items()}


def compare(left, right):
    a, b = dense.arr(left), dense.arr(right)
    ix = np.unravel_index(np.argmax(np.abs(a - b)), a.shape)
    return {'max_abs': float(np.max(np.abs(a - b))),
            'index': [int(i) for i in ix],
            'active': float(a[ix]), 'dense': float(b[ix])}


def main():
    saved = json.loads(SAVED.read_text(encoding='utf-8'))
    case = next(c for c in saved['cases'] if c['n'] == 3 and
                c['kind'] == 'negative_variance')
    weights = [np.asarray(w, dtype=np.float64) for w in case['weights']]
    table = dense.active._tables()
    est = dense.active.Estimator()
    est.setup(None)
    const = est._constants(3, fnp.float64)
    old_wick = dense.wick
    dense.wick = wick64
    active_state = (fnp.zeros(3, dtype=fnp.float64),
                    fnp.eye(3, dtype=fnp.float64), None, 0.0,
                    None, None, None)
    oracle_state = (np.zeros(3), np.eye(3), np.zeros((3, 3, 3)),
                    0.0, None)
    rows = []
    dense.active._E14_DEBUG_CAPTURE = []
    dense.active._E14_QUERY_FUSED = True
    try:
        with BudgetContext(10**11):
            for layer, W in enumerate(weights):
                active_state = dense.active._layer_tree(
                    fnp.array(W, dtype=fnp.float64), active_state, table,
                    fnp.float64, const, 2.0, layer == 3, None)
                oracle_state, detail = dense.oracle_layer(
                    W, oracle_state, table, const, layer == 3)
                trace = dense.active._E14_DEBUG_CAPTURE[-1]
                fields = {}
                for key, value in detail['pK'].items():
                    fields['pK' + str(key)] = compare(trace['pK'][key], value)
                for key, value in detail['K'].items():
                    fields['K' + str(key)] = compare(trace['K'][key], value)
                if layer < 3:
                    fields['K3_all'] = compare(
                        dense_tower(active_state[2], active_state[5],
                                    active_state[6]), oracle_state[2])
                field = max(fields, key=lambda k: fields[k]['max_abs'])
                rows.append({'layer': layer, 'fields': fields,
                             'max_field': field,
                             'max_abs': fields[field]['max_abs']})
    finally:
        dense.wick = old_wick
        dense.active._E14_DEBUG_CAPTURE = None
    out = {'scope': 'same saved n3 negative fixture; unchanged active f64 path '
                    'versus independent dense f64 recurrence; no labels',
           'candidate_sha256': '5127a55ccd99a9516e271c8363153b86916684f470465d02c02dcaa413debb7d',
           'declared_original_f32_bound': 2e-5,
           'source_weights': case['weights'], 'rows': rows,
           'max_abs': max(r['max_abs'] for r in rows)}
    OUT.write_text(json.dumps(out, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'max_abs': out['max_abs'],
                      'rows': [{'layer': r['layer'], 'max_field': r['max_field'],
                                'max_abs': r['max_abs']} for r in rows]}))


if __name__ == '__main__':
    main()
