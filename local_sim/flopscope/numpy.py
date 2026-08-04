"""
LOCAL SHIM for `flopscope.numpy` (fnp) — re-exports real NumPy.

One behavioral tweak: einsum defaults to optimize=True so contraction paths
match flopscope's pairwise scheduling (otherwise numpy's naive einsum would be
O(n^4) for our covariance contraction and painfully slow).
"""

import numpy as _np
from numpy import *  # noqa: F401,F403 — zeros, eye, ones, sqrt, maximum, outer, diag, stack, float32, ...
from numpy import random  # noqa: F401  (fnp.random.default_rng)


def einsum(*args, **kwargs):
    kwargs.setdefault("optimize", True)
    return _np.einsum(*args, **kwargs)


def __getattr__(name):  # fallback for anything not pulled in by star-import
    return getattr(_np, name)
