"""LOCAL SHIM for `flopscope.stats` — stats.norm.cdf / .pdf / .ppf.

Real-flopscope pricing (measured 2026-08-04, per element):
    cdf 96   pdf 54   ppf 168
"""

import numpy as _np

try:
    from scipy.special import ndtr as _ndtr  # exact, fast standard-normal CDF
    from scipy.special import ndtri as _ndtri  # its inverse
except ImportError:  # pragma: no cover
    from math import erf as _erf

    def _ndtr(x):
        x = _np.asarray(x, dtype=_np.float64)
        return 0.5 * (1.0 + _np.vectorize(_erf)(x / _np.sqrt(2.0)))

    _ndtri = None

_INV_SQRT_2PI = 1.0 / _np.sqrt(2.0 * _np.pi)


class norm:
    @staticmethod
    def cdf(x):
        return _ndtr(x)

    @staticmethod
    def pdf(x):
        x = _np.asarray(x)
        return _INV_SQRT_2PI * _np.exp(-0.5 * x * x)

    @staticmethod
    def ppf(p):
        if _ndtri is None:  # pragma: no cover
            raise RuntimeError("scipy required for the ppf shim")
        return _ndtri(_np.asarray(p, dtype=_np.float64))
