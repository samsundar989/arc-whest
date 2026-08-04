"""LOCAL SHIM for `flopscope.stats` — provides stats.norm.cdf / stats.norm.pdf."""

import numpy as _np

try:
    from scipy.special import ndtr as _ndtr  # exact, fast standard-normal CDF
except ImportError:  # pragma: no cover
    from math import erf as _erf

    def _ndtr(x):
        x = _np.asarray(x, dtype=_np.float64)
        return 0.5 * (1.0 + _np.vectorize(_erf)(x / _np.sqrt(2.0)))

_INV_SQRT_2PI = 1.0 / _np.sqrt(2.0 * _np.pi)


class norm:
    @staticmethod
    def cdf(x):
        return _ndtr(x)

    @staticmethod
    def pdf(x):
        x = _np.asarray(x)
        return _INV_SQRT_2PI * _np.exp(-0.5 * x * x)
