"""`flopscope.stats.norm` on torch. `ndtr` is the same standard-normal CDF."""
import math

import torch as _t

_C = 1.0 / math.sqrt(2.0 * math.pi)


class norm:
    @staticmethod
    def cdf(x):
        return _t.special.ndtr(x)

    @staticmethod
    def pdf(x):
        return _t.exp(-0.5 * x * x) * _C
