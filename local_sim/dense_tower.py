"""Run a candidate's REAL `_layer` with the third cumulant carried as a DENSE tensor.

Tiny widths only (n^3 memory). The candidate module is loaded fresh and its tower
primitives (`_advance`, `_wick_tower`, `_dslice3`, `_dslice21`, `_append`,
`_truncate`) are swapped for dense equivalents, so every Wick coefficient, term
table row, birth, compensation and fourth-cumulant line is the file's own code,
while the carried state is an explicit (n,n,n) float64 tensor that an oracle can
read, perturb or project.

Newborn sources are still built by the file's own `_append(None, ...)` and then
expanded to a full tensor with `k4feed_oracle.tower_tensor`, so the dense chain
equals the factored chain at full rank (checked by `self_check`).

Used by: `k3feed_oracle.py` (P3 derivation) and `n1_relevance_oracle.py` (N1).
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]

import numpy as np  # noqa: E402

import k4feed_oracle as K  # noqa: E402


class DenseTower(object):
    __slots__ = ("T", "k")

    def __init__(self, T, k=1):
        self.T = T
        self.k = k


def load_dense(path):
    """Fresh module with dense tower primitives. Returns (module, originals)."""
    import flopscope.numpy as fnp

    mod = K._load(path)
    orig = {name: getattr(mod, name) for name in
            ("_advance", "_wick_tower", "_dslice3", "_dslice21", "_append", "_truncate")}

    def _np(x):
        return np.asarray(x, dtype=np.float64)

    def advance(WT, tw):
        W = _np(WT)
        return DenseTower(np.einsum("ai,bj,ck,ijk->abc", W, W, W, tw.T, optimize=True), tw.k)

    def wick_tower(col, tw):
        c = _np(col).reshape(-1)
        return DenseTower(tw.T * c[:, None, None] * c[None, :, None] * c[None, None, :], tw.k)

    def dslice3(tw):
        n = tw.T.shape[0]
        i = np.arange(n)
        return fnp.asarray(tw.T[i, i, i].copy(), dtype=fnp.float64)

    def dslice21(tw, off_mask):
        d21 = np.einsum("aab->ab", tw.T).copy()
        return fnp.asarray(d21, dtype=fnp.float64) * off_mask

    def append(tw, *args, **kwargs):
        n = args[3]
        newborn = orig["_append"](None, *args, **kwargs)
        Tn = K.tower_tensor(newborn, n)
        if tw is None:
            return DenseTower(Tn, 1)
        return DenseTower(tw.T + Tn, tw.k + 1)

    def truncate(tw, gens, n):
        return tw

    mod._advance = advance
    mod._wick_tower = wick_tower
    mod._dslice3 = dslice3
    mod._dslice21 = dslice21
    mod._append = append
    mod._truncate = truncate
    return mod, orig


def run_chain(mod, Ws, *, inject=None, capture=None, lam=None):
    """Float64 chain over weights `Ws` (list of (n,n)), dense tower.

    `inject(l, T)` may return a replacement incoming tensor at layer l (before
    `_layer`); `capture(l, state_in, state_out)` sees each step. `lam`, if
    given, patches `_K4_LAM` for the run. Returns the list of layer means,
    clamped at 0 exactly as `predict` does for rows >= 1.
    """
    import flopscope as flops
    import flopscope.numpy as fnp

    n = Ws[0].shape[0]
    dt = fnp.float64
    old_lam = list(mod._K4_LAM)
    if lam is not None:
        mod._K4_LAM = list(lam)
    try:
        table = mod._tables()
        const = mod.Estimator()._constants(n, dt)
        state = (np.zeros(n), np.eye(n), None, 0.0, np.zeros(n))
        mus = []
        depth = len(Ws)
        for l in range(depth):
            mu, cov, tw, c4, gk4 = state
            if inject is not None and tw is not None:
                rep = inject(l, tw.T)
                if rep is not None:
                    tw = DenseTower(rep, tw.k)
            with flops.BudgetContext(flop_budget=2 ** 55, quiet=True):
                out = mod._layer(fnp.asarray(Ws[l], dtype=dt),
                                 (fnp.asarray(mu, dtype=dt), fnp.asarray(cov, dtype=dt), tw, c4,
                                  fnp.asarray(gk4, dtype=dt)),
                                 table, dt, const, 2.0, last_layer=(l == depth - 1),
                                 gens=None, layer_index=l)
            m = np.asarray(out[0], dtype=np.float64)
            mus.append(m)
            if capture is not None:
                capture(l, state, out)
            if l < depth - 1:
                state = (m, np.asarray(out[1], dtype=np.float64), out[2], out[3],
                         np.asarray(out[4], dtype=np.float64))
        return mus
    finally:
        mod._K4_LAM = old_lam


def self_check(path, n=10, depth=5, seed=3):
    """Dense chain == the file's factored chain (float64), every layer mean."""
    import flopscope as flops
    import flopscope.numpy as fnp

    rng = np.random.default_rng(seed)
    Ws = [rng.normal(size=(n, n)) * np.sqrt(2.0 / n) for _ in range(depth)]
    dmod, _ = load_dense(path)
    fmod = K._load(path)
    dense = run_chain(dmod, Ws)
    # factored run, same loop
    dt = fnp.float64
    table = fmod._tables()
    const = fmod.Estimator()._constants(n, dt)
    state = (np.zeros(n), np.eye(n), None, 0.0, np.zeros(n))
    worst = 0.0
    for l in range(depth):
        mu, cov, tw, c4, gk4 = state
        with flops.BudgetContext(flop_budget=2 ** 55, quiet=True):
            out = fmod._layer(fnp.asarray(Ws[l], dtype=dt),
                              (fnp.asarray(mu, dtype=dt), fnp.asarray(cov, dtype=dt), tw, c4,
                               fnp.asarray(gk4, dtype=dt)),
                              table, dt, const, 2.0, last_layer=(l == depth - 1), gens=None,
                              layer_index=l)
        m = np.asarray(out[0], dtype=np.float64)
        worst = max(worst, float(np.abs(m - dense[l]).max() / max(np.abs(m).max(), 1e-300)))
        if l < depth - 1:
            state = (m, np.asarray(out[1], dtype=np.float64), out[2], out[3],
                     np.asarray(out[4], dtype=np.float64))
    return worst


if __name__ == "__main__":
    import flopscope as flops
    flops.configure(symmetry_warnings=False)
    for f in ("estimator_k3_v31_dense.py", "estimator_k3_v32_dense.py"):
        w = self_check(os.path.join(_ROOT, f))
        print(f"{f}: dense chain vs factored chain, worst rel mean diff {w:.2e}  "
              f"{'OK' if w < 1e-10 else 'FAILED'}")
