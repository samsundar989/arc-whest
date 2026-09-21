"""`flopscope.stats` — `norm.cdf` / `.pdf` / `.ppf`, on the GPU.

`cupyx.scipy.special.ndtr` is the same standard-normal CDF scipy's `ndtr`
computes, and the numpy shim calls scipy's directly, so the two paths agree to
the last float32 bit on the values the chain actually feeds them. That equality
is what `tools/_predict_worker.py --backend cupy` is checked against.

The real flopscope promotes a float32 input to float64 here "to match
scipy.stats" and warns about it; the shipped file casts straight back with
`.astype(dtype)`, so the promotion is invisible downstream and neither shim
reproduces it.
"""

import cupy as _cp
from cupyx.scipy import special as _sp

_INV_SQRT_2PI = 1.0 / _cp.sqrt(_cp.asarray(2.0 * _cp.pi))


class norm:
    @staticmethod
    def cdf(x):
        return _sp.ndtr(x)

    @staticmethod
    def pdf(x):
        x = _cp.asarray(x)
        return _INV_SQRT_2PI.astype(x.dtype) * _cp.exp(-0.5 * x * x)

    @staticmethod
    def ppf(p):
        return _sp.ndtri(_cp.asarray(p, dtype=_cp.float64))
