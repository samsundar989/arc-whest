"""
LOCAL SHIM for the `flopscope` package — cloud/offline development only.

The real flopscope (github.com/AIcrowd/flopscope) is a FLOP-counting NumPy
wrapper used by the WHEST grader. This shim lets the *exact* submission file
(estimator.py) run in a plain-Python environment for fast iteration:

    sys.path.insert(0, "local_sim")   # then `import flopscope` resolves here

It does NOT count FLOPs. Analytical cost of our estimator is computed by hand
in local_sim/validate_estimator.py. Never ship this shim in a submission
(single-file `whest submit --estimator estimator.py` ships only estimator.py).
"""

from . import numpy  # noqa: F401  (flopscope.numpy as fnp)
from . import stats  # noqa: F401  (flops.stats.norm.cdf / .pdf)


class BudgetContext:
    """No-op stand-in for flopscope.BudgetContext."""

    def __init__(self, flop_budget=None, **kwargs):
        self.flop_budget = flop_budget

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def as_symmetric(a, symmetry=(0, 1)):
    """Real flopscope re-tags an array as symmetric so symmetric einsums are
    billed at ~half cost. Numerically it is the identity."""
    return a
