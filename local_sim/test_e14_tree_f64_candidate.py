"""Tiny no-label checks for the separate full-f64 tree candidate."""
import os
import sys
from pathlib import Path

import numpy as np
import flopscope.numpy as fnp
from flopscope import BudgetContext

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import estimator_e14_tree_f64 as e
import estimator_k3_strassen_L4_f21 as frozen


class MLP:
    def __init__(self, weights):
        self.width = weights[0].shape[0]
        self.weights = [fnp.array(x.astype('float64')) for x in weights]
        self.seed = 1


def weights(n, kind):
    if kind == 'controlled':
        x = np.eye(n) + 0.03 * np.ones((n, n))
        return [x.copy() for _ in range(4)]
    rng = np.random.default_rng(818 + n * 10 + 4)
    return [rng.normal(size=(n, n)) * np.sqrt(2 / n) for _ in range(4)]


def predict(est, mlp, budget=10**11):
    est.setup(None)
    with BudgetContext(10**11):
        return np.asarray(est.predict(mlp, budget))


def run():
    results = []
    old = os.environ.get('ARC_WHEST_DISABLE_E14_TREE')
    try:
        for n in (3, 4):
            for kind in ('controlled', 'negative_variance'):
                mlp = MLP(weights(n, kind))
                e._E14_QUERY_FUSED = True
                active = e.Estimator()
                trace = []
                e._E14_DEBUG_CAPTURE = trace
                out = predict(active, mlp)
                e._E14_DEBUG_CAPTURE = None
                assert active.last_run['mode'] == 'e14_tree'
                assert out.dtype == np.float64 and np.isfinite(out).all()
                assert len(trace) == 4
                assert all(t['W'].dtype == fnp.float64 for t in trace)
                assert all(t['cov_pre'].dtype == fnp.float64 for t in trace)
                e._E14_QUERY_FUSED = False
                unfused = predict(e.Estimator(), mlp)
                gap = float(np.max(np.abs(out - unfused)))
                assert gap < 1e-8, (n, kind, gap)
                e._E14_QUERY_FUSED = True
                os.environ['ARC_WHEST_DISABLE_E14_TREE'] = '1'
                disabled = predict(e.Estimator(), mlp)
                baseline = predict(frozen.Estimator(), mlp)
                assert np.array_equal(disabled, baseline), (n, kind)
                os.environ.pop('ARC_WHEST_DISABLE_E14_TREE', None)
                est = e.Estimator(); est.setup(None); est._tree_active = True
                table = e._tables(); const = est._constants(n, fnp.float64)
                state = (fnp.zeros(n, dtype=fnp.float64),
                         fnp.eye(n, dtype=fnp.float64), None, 0.0,
                         None, None, None)
                ages = []
                with BudgetContext(10**11):
                    for layer, w in enumerate(mlp.weights):
                        state = e._layer_tree(w, state, table, fnp.float64,
                                              const, 2.0, layer == 3, 1)
                        tw, y, r = state[2], state[5], state[6]
                        ages.append(None if tw is None else tw.k)
                        if layer < 3:
                            assert tw.k == y.shape[1] == r.shape[1] == 1
                reserve = (4 * 3 * n * n * (2 * n - 1)
                           + (5000 * 4 + 100 * 16) * n * n)
                budget = reserve + int(np.ceil(e._projected_flops(n, 4, 1) / .95)) - 1000
                fallback = e.Estimator()
                regen = predict(fallback, mlp, budget)
                assert fallback.last_run['mode'] == 'regen'
                assert np.isfinite(regen).all()
                results.append(dict(n=n, kind=kind, fused_gap=gap,
                                    disabled_exact=True, forced_cap1_ages=ages,
                                    fallback_mode=fallback.last_run['mode']))
    finally:
        if old is None:
            os.environ.pop('ARC_WHEST_DISABLE_E14_TREE', None)
        else:
            os.environ['ARC_WHEST_DISABLE_E14_TREE'] = old
        e._E14_QUERY_FUSED = True
        e._E14_DEBUG_CAPTURE = None
    return results


if __name__ == '__main__':
    import json
    print(json.dumps(run(), indent=2))
