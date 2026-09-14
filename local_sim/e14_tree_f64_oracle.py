"""Independent Gaussian-start dense comparison for the full-f64 E14 tree arm.

Tiny, label-free n=3/4 fixtures only. The oracle carries dense K3 and a dense
connected-tree fourth tensor, never reading prior active state. Wick coefficients
come from the frozen table evaluator; this shared boundary is disclosed.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import flopscope.numpy as fnp
from flopscope import BudgetContext

import e14_tree_full_oracle as dense
from e14_tree_layer_oracle import dense_tower
import estimator_e14_tree_f64 as candidate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/e14_tree_f64/oracle_v2.json'
BOUND = 1e-8


def _arr(x):
    return np.asarray(x, dtype=np.float64)


def _compare(a, b):
    aa, bb = _arr(a), _arr(b)
    if aa.shape != bb.shape:
        return {'shape_mismatch': [list(aa.shape), list(bb.shape)], 'max_abs': None}
    if not np.all(np.isfinite(aa)) or not np.all(np.isfinite(bb)):
        return {'nonfinite': True, 'max_abs': None}
    gap = np.abs(aa - bb)
    index = np.unravel_index(np.argmax(gap), gap.shape)
    return {'max_abs': float(gap[index]), 'index': [int(i) for i in index],
            'active': float(aa[index]), 'dense': float(bb[index])}


def wick64(mu, cov, table, const):
    var = np.maximum(np.diag(cov), 1e-12)
    sigma = np.sqrt(var)
    alpha = np.clip(mu / sigma, -candidate.ALPHA_CAP,
                    candidate.ALPHA_CAP)
    with BudgetContext(10**8):
        w = candidate._wick(fnp.array(mu, dtype=fnp.float64),
                            fnp.array(sigma, dtype=fnp.float64),
                            fnp.array(alpha, dtype=fnp.float64),
                            table, fnp.float64, const['wick'])
    return {key: _arr(val) for key, val in w.items()}


def one_case(n, kind):
    weights = [np.asarray(W, dtype=np.float64)
               for W in dense.controlled_weights(n, kind)]
    table = candidate._tables()
    est = candidate.Estimator()
    est.setup(None)
    const = est._constants(n, fnp.float64)
    active_state = (fnp.zeros(n, dtype=fnp.float64),
                    fnp.eye(n, dtype=fnp.float64), None, 0.0,
                    None, None, None)
    oracle_state = (np.zeros(n), np.eye(n), np.zeros((n, n, n)),
                    0.0, None)
    rows = []
    candidate._E14_DEBUG_CAPTURE = []
    candidate._E14_QUERY_FUSED = True
    try:
        with BudgetContext(10**11):
            for layer, W in enumerate(weights):
                active_state = candidate._layer_tree(
                    fnp.array(W, dtype=fnp.float64), active_state, table,
                    fnp.float64, const, 2.0, layer == 3, None)
                oracle_state, detail = dense.oracle_layer(
                    W, oracle_state, table, const, layer == 3)
                trace = candidate._E14_DEBUG_CAPTURE[-1]
                fields = {
                    'pre': _compare(trace['mu_pre'], detail['pre']),
                    'cpre': _compare(trace['cov_pre'], detail['cpre']),
                    'q4': _compare(trace['q4'], np.einsum('iiii->i', detail['Q'])),
                    'q31': _compare(trace['q31'], np.asarray([
                        [detail['Q'][i, i, i, j] if i != j else 0.0
                         for j in range(n)] for i in range(n)])),
                    'q22': _compare(trace['q22'], np.asarray([
                        [detail['Q'][i, i, j, j] if i != j else 0.0
                         for j in range(n)] for i in range(n)])),
                    'mean': _compare(active_state[0], oracle_state[0]),
                    'pK': {str(k): _compare(trace['pK'][k], v)
                           for k, v in detail['pK'].items()},
                    'K': {str(k): _compare(trace['K'][k], v)
                          for k, v in detail['K'].items()},
                    'w': {str(k): _compare(trace['w'][k], v)
                          for k, v in detail['w'].items()},
                    'S': {str(k): _compare(trace['S'][k], v)
                          for k, v in detail['S'].items()}
                    if 'S' in trace else None,
                }
                if layer < 3:
                    fields['cov'] = _compare(active_state[1], oracle_state[1])
                    fields['K3_all'] = _compare(
                        dense_tower(active_state[2], active_state[5],
                                    active_state[6]), oracle_state[2])
                    fields['scalar'] = _compare(active_state[3], oracle_state[3])
                    fields['core_g'] = _compare(active_state[4][0], oracle_state[4][0])
                    fields['core_lambda'] = _compare(active_state[4][1], oracle_state[4][1])
                    fields['core_Cout'] = _compare(active_state[4][2], oracle_state[4][2])
                    fields['age'] = {'tower': int(active_state[2].k),
                                     'Y': int(active_state[5].shape[1]),
                                     'R': int(active_state[6].shape[1])}
                rows.append({'layer': layer, 'fields': fields})
    finally:
        candidate._E14_DEBUG_CAPTURE = None
    return {'n': n, 'kind': kind, 'weights': [W.tolist() for W in weights],
            'rows': rows}


def _walk(prefix, obj):
    if isinstance(obj, dict):
        if 'max_abs' in obj:
            yield prefix, obj
        else:
            for k, v in obj.items():
                yield from _walk(prefix + '/' + str(k), v)


def main():
    original_active, original_wick = dense.active, dense.wick
    dense.active, dense.wick = candidate, wick64
    try:
        cases = [one_case(n, kind) for n in (3, 4)
                 for kind in ('admissible_controlled', 'negative_variance')]
    finally:
        dense.active, dense.wick = original_active, original_wick
    comparisons = [(case['n'], case['kind'], row['layer'], field, rec)
                   for case in cases for row in case['rows']
                   for field, rec in _walk('', row['fields'])]
    failures = [x for x in comparisons if x[4]['max_abs'] is None or
                x[4]['max_abs'] > BOUND]
    age_failures = [(case['n'], case['kind'], row['layer'], row['fields']['age'])
                    for case in cases for row in case['rows']
                    if 'age' in row['fields'] and len(set(row['fields']['age'].values())) != 1]
    first = min(failures, key=lambda x: (x[0], x[1], x[2], x[3])) if failures else None
    out = {'scope': 'full-f64 Gaussian-start dense recurrence, n3/4, four layers, '
                    'controlled and preserved negative fixtures; no labels',
           'bound_abs': BOUND, 'candidate_sha256': hashlib.sha256(
               (ROOT / 'estimator_e14_tree_f64.py').read_bytes()).hexdigest(),
           'oracle_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'shared_boundary': 'Wick coefficient evaluator/table and constants; '
                              'dense state, trees, moments, centering, birth independent',
           'cases': cases, 'n_comparisons': len(comparisons),
           'max_abs': max((x[4]['max_abs'] or 0) for x in comparisons),
           'age_failures': age_failures,
           'passed': not failures and not age_failures,
           'first_falsifier': None if first is None else {
               'n': first[0], 'kind': first[1], 'layer': first[2],
               'field': first[3], 'operand': first[4]}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'passed': out['passed'], 'max_abs': out['max_abs'],
                      'n_comparisons': out['n_comparisons'],
                      'first_falsifier': out['first_falsifier']}))


if __name__ == '__main__':
    main()
