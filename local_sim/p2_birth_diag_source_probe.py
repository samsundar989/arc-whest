"""Source-bound, label-free lifecycle probe for C10 versus v45 P2 birth.

Run: uv run python local_sim/p2_birth_diag_source_probe.py

The modules execute their real predict/layer/birth path on a deterministic tiny
weight fixture. Rank is set to one in memory to expose truncation at width six;
the frozen source files are not modified. No labels or seed block are used.
"""

from pathlib import Path
import importlib.util

import numpy as np


ROOT = Path(__file__).resolve().parent.parent


def load(name):
    path = ROOT / name
    spec = importlib.util.spec_from_file_location(name[:-3], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._HUB_ROUTES == ('i', 'ii')
    assert module._HUB_COMPENSATE and not module._PROTECT_FEED
    assert not module._P3_ON
    module._P2_RANK = 1
    return module


def run_source(module, weights):
    from flopscope import BudgetContext
    from harness import MLP

    births = []
    original = module._append

    def capture(*args, **kwargs):
        tower = original(*args, **kwargs)
        if len(births) < 1:
            births.append((args, kwargs, tower))
        return tower

    module._append = capture
    try:
        mlp = MLP(weights=weights, width=len(weights[0]), depth=len(weights), seed=0)
        with BudgetContext(flop_budget=2**41, quiet=True):
            prediction = module.Estimator().predict(mlp, 2**41)
    finally:
        module._append = original
    assert births, 'candidate did not enter the P2 birth path'
    return np.asarray(prediction), births[0]


def close(got, want, name, rtol=5e-5, atol=5e-6):
    np.testing.assert_allclose(np.asarray(got), np.asarray(want),
                               rtol=rtol, atol=atol, err_msg=name)


def main():
    n, depth = 6, 3
    # Fixed arithmetic fixture with nonuniform signed weights and no RNG.
    weights = []
    for layer in range(depth):
        w = np.fromfunction(lambda i, j: (
            .10 * np.sin((i+1) * (j+2) + layer)
            + .05 * np.cos((i+3) * (j+1) - 2*layer)
            + 1.0 * (i == j)), (n, n), dtype=float)
        weights.append(np.asarray(w, dtype=np.float64))

    c10 = load('estimator_k3_v43_complete_c10.py')
    v45 = load('estimator_k3_v45_p2_diag.py')
    pred_old, (old_args, old_kw, old_tw) = run_source(c10, weights)
    pred_new, (new_args, new_kw, new_tw) = run_source(v45, weights)
    assert not old_kw and not new_kw
    assert old_args[0] is None and new_args[0] is None
    assert old_tw.Y is not None and new_tw.Y is not None
    assert old_tw.A_D is None and new_tw.A_D is None
    assert old_tw.k == new_tw.k == 1

    # _append(tw,A_D,A_L,w2b,n,Zb,Lb,sb,eb,Y,CA,CB): only sb may
    # differ at the first birth. This checks the active hub branch's actual
    # arguments after its compensation and the real compressor.
    for i in (2, 3, 5, 6, 8, 9, 10, 11):
        close(old_args[i], new_args[i], f'first birth argument {i}')
    z, leg, sb_old, sb_new = [np.asarray(x) for x in (
        old_args[5], old_args[6], old_args[7], new_args[7])]
    leak = (z * leg).sum(1)
    assert np.max(np.abs(leak)) > 1e-5
    close(sb_old - sb_new, leak, 'source sb delta equals factor diagonal')
    # At this first birth M is the source's symbolic identity (M=None), so
    # slice functions are intentionally called only after _advance. Direct
    # birth reconstruction uses its returned factors and the source's D block.
    born3_old = sb_old + leak
    born3_new = sb_new + leak
    close(born3_old - born3_new, leak, 'newborn represented D3 delta')
    close(born3_new, sb_old, 'newborn D3 restored to specified S3')
    off = np.ones((n, n), dtype=np.float32) - np.eye(n, dtype=np.float32)
    eye = np.eye(n)
    ad_old = np.diag(sb_old) + 3 * np.asarray(old_tw.A_L) * np.asarray(old_tw.eb)[None, :] + z @ leg.T
    ad_new = np.diag(sb_new) + 3 * np.asarray(new_tw.A_L) * np.asarray(new_tw.eb)[None, :] + z @ leg.T
    def d_block(ad):
        return (np.einsum('aj,bj,cj->abc', ad, eye, eye)
                + np.einsum('bj,aj,cj->abc', ad, eye, eye)
                + np.einsum('cj,aj,bj->abc', ad, eye, eye)) / 3
    birth_delta = d_block(ad_old) - d_block(ad_new)
    close(np.einsum('iij->ij', birth_delta) * off, np.zeros((n, n)),
          'birth oriented off-D21 unchanged')
    close(np.einsum('iij->ij', birth_delta).T * off, np.zeros((n, n)),
          'birth reverse off-D21 unchanged')

    # Use the actual source transport and slice functions for the next linear
    # step. The birth correction must survive in recurrent D3 and both D21
    # orientations even though no new rank/history source was added.
    wt = weights[2].T.astype(np.float32)
    advanced_old = c10._advance(wt, old_tw)
    advanced_new = v45._advance(wt, new_tw)
    close(advanced_old.Z, advanced_new.Z, 'transported residual factors')
    close(advanced_old.Y, advanced_new.Y, 'transported hubs')
    close(advanced_old.sb - advanced_new.sb, leak, 'static sb survives transport')
    delta3 = np.asarray(c10._dslice3(advanced_old)
                        - v45._dslice3(advanced_new))
    expected3 = (wt ** 3) @ leak
    close(delta3, expected3, 'transported D3 correction')
    delta21 = np.asarray(c10._dslice21(advanced_old, off)
                         - v45._dslice21(advanced_new, off))
    expected21 = ((wt ** 2) * leak[None, :]) @ wt.T * off
    close(delta21, expected21, 'transported oriented D21 correction')
    close(delta21.T, expected21.T, 'transported reverse D21 orientation')

    assert np.isfinite(pred_old).all() and np.isfinite(pred_new).all()
    print('source-bound P2 birth probe: PASS')
    print(f'active hub birth; max factor leak {np.max(np.abs(leak)):.6g}')
    print(f'transported max |D3 correction| {np.max(np.abs(delta3)):.6g}')
    print(f'transported max |D21 correction| {np.max(np.abs(delta21)):.6g}')
    print(f'full tiny trajectory max prediction change '
          f'{np.max(np.abs(pred_old-pred_new)):.6g} (truth-free)')


if __name__ == '__main__':
    main()
