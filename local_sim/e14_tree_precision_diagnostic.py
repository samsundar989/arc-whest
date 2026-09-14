"""Read-only, tiny precision trace for the saved E14 n3 negative fixture."""
import json
from pathlib import Path

import numpy as np
import flopscope.numpy as fnp
from flopscope import BudgetContext

import e14_tree_full_oracle as ref


OUT = ref.ROOT / 'experiments' / 'e14_tree_candidate' / 'precision_diagnostic.json'
SAVED = OUT.parent / 'full_oracle_v4.json'


def quant(x):
    if x is None:
        return None
    if isinstance(x, tuple):
        return tuple(quant(v) for v in x)
    if isinstance(x, np.ndarray):
        return x.astype('float32').astype('float64')
    return float(np.float32(x))


def val(x):
    return np.asarray(x, dtype=np.float64)


def comparison(a, b):
    a, b = val(a), val(b)
    delta = np.abs(a-b)
    j = np.unravel_index(np.argmax(delta), delta.shape)
    return {'index': [int(k) for k in j], 'left': float(a[j]), 'right': float(b[j]),
            'absolute_gap': float(delta[j])}


def main():
    saved = json.loads(SAVED.read_text(encoding='utf-8'))
    fixture = next(c for c in saved['cases'] if c['n'] == 3 and c['kind'] == 'negative_variance')
    weights = [np.array(w, dtype=np.float64) for w in fixture['weights']]
    table = ref.active._tables()
    est = ref.active.Estimator(); est.setup(None)
    const = est._constants(3, fnp.float32)
    fs = (fnp.zeros(3, dtype=fnp.float32), fnp.eye(3, dtype=fnp.float32),
          None, 0.0, None, None, None)
    ref.active._E14_DEBUG_CAPTURE = []
    ref.active._E14_QUERY_FUSED = True
    try:
        with BudgetContext(10**11):
            for i, W in enumerate(weights):
                fs = ref.active._layer_tree(fnp.array(W.astype('float32')), fs,
                                            table, fnp.float32, const, 2.0,
                                            i == 3, None)
        traces = ref.active._E14_DEBUG_CAPTURE
    finally:
        ref.active._E14_DEBUG_CAPTURE = None
    full = (np.zeros(3), np.eye(3), np.zeros((3,3,3)), 0.0, None)
    matched = full
    layers = []
    for i, W in enumerate(weights[:3]):
        full, d64 = ref.oracle_layer(W, full, table, const, i == 3)
        matched, dm = ref.oracle_layer(W, quant(matched), table, const, i == 3)
        tr = traces[i]
        fields = {}
        for name, key in [('pre','mu_pre'), ('cpre','cov_pre'), ('D3','D3')]:
            src = d64['S'][(3,)] if name == 'D3' else d64[name]
            fields[name] = {'active_vs_full': comparison(tr[key], src),
                            'matched_vs_full': comparison(dm['S'][(3,)] if name == 'D3' else dm[name], src)}
        for key in ((1,), (3,), (4,)):
            name = 'S'+str(key)
            fields[name] = {'active_vs_full': comparison(tr['S'][key], d64['S'][key]),
                            'matched_vs_full': comparison(dm['S'][key], d64['S'][key])}
        base64 = ref.moment_table(d64['S'], d64['w'], table, i == 3)
        basem = ref.moment_table(dm['S'], dm['w'], table, i == 3)
        active_operands_base = ref.moment_table(
            {k: val(v) for k,v in tr['S'].items()},
            {k: val(v) for k,v in tr['w'].items()}, table, False)
        fields['pK_base(4,)'] = {'active_vs_full': comparison(tr['pK_base'][(4,)], base64[(4,)]),
                                  'matched_vs_full': comparison(basem[(4,)], base64[(4,)]),
                                  'active_operands_f64_vs_active': comparison(active_operands_base[(4,)],tr['pK_base'][(4,)]),
                                  'active_operands_f64_vs_full': comparison(active_operands_base[(4,)],base64[(4,)])}
        fields['pK(4,)'] = {'active_vs_full': comparison(tr['pK'][(4,)], d64['pK'][(4,)]),
                             'matched_vs_full': comparison(dm['pK'][(4,)], d64['pK'][(4,)])}
        layers.append({'layer':i, 'fields':fields})
    out = {'scope':'saved n3 negative fixture; same f32-rounded weights and Wick evaluator; f64 arithmetic with f32 state quantization at layer boundaries',
           'finding':'first traced operand difference is layer0 cov_pre[1,1], at the f32 WT @ I @ W boundary; the layer2 pK(4,) gap is already present in the base moment before the fourth-hyperedge addition',
           'limitations':'boundary quantization is not an emulation of every f32 matmul, reduction, fused kernel, or rounding operation; this is a no-label diagnostic, not repair, gate change, or score evidence',
           'declared_f32_bound':2e-5, 'saved_first_falsifier':saved['first_falsifier'],
           'layers':layers}
    OUT.write_text(json.dumps(out, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'layer2':layers[2]['fields']}, indent=2))


if __name__ == '__main__':
    main()
