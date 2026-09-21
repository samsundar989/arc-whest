"""NON-COUNTING `flopscope` shim backed by **cupy** on the RTX 5090.

PLAN §4's accuracy path. Same contract as `local_sim/flopscope` — it lets the
exact submission file run unchanged — but `fnp` is cupy rather than numpy, so
the chain's 1024x1024 products run on the GPU instead of two CPU cores.

**It does not count FLOPs and it must never be asked to.** Every cost figure in
this project comes from `tools/_cost_worker.py`, which runs the genuine
flopscope package in a process where this directory has never been on the path.
`BudgetContext` here is a no-op that records nothing; there is deliberately no
`flops_used` attribute to read, so code that tries to price something against
this shim fails loudly instead of quietly reporting a zero.

`as_symmetric` is the identity, exactly as in the numpy shim: on the real
package it is a billing tag, and numerically it changes nothing.
"""

from . import numpy  # noqa: F401  (flopscope.numpy as fnp)
from . import stats  # noqa: F401  (flops.stats.norm.cdf / .pdf)

__all__ = ["numpy", "stats", "BudgetContext", "as_symmetric",
           "BudgetExhaustedError", "TimeExhaustedError"]


class BudgetExhaustedError(RuntimeError):
    pass


class TimeExhaustedError(RuntimeError):
    pass


class BudgetContext:
    """No-op stand-in. Enters, exits, counts nothing, and says so."""

    def __init__(self, flop_budget=None, **kwargs):
        self.flop_budget = flop_budget

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __getattr__(self, name):
        raise AttributeError(
            f"{name!r} is not available on the cupy accuracy shim: this backend "
            f"does not count FLOPs. Take cost from tools/_cost_worker.py, which "
            f"runs the real flopscope package.")


def as_symmetric(a, symmetry=(0, 1)):
    """A billing tag on the real package; numerically the identity."""
    return a
