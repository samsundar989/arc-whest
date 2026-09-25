"""NON-COUNTING `flopscope` shim backed by **torch** (CUDA), for the differentiable twin.

HANDOFF 2026-09-25 Step 2. Same contract as `tools/gpu_shim` (the cupy accuracy
shim): the shipped estimator source runs UNCHANGED, but every array is a
`torch.Tensor`, so autograd flows through the exact chain -- the same Wick
program, closure tables, tower ops, kappa4 regeneration and corrector -- rather
than through a hand re-implementation that could drift from it.

It counts nothing. `BudgetContext` is a no-op with no `flops_used`.
"""

from . import numpy  # noqa: F401
from . import stats  # noqa: F401

__all__ = ["numpy", "stats", "BudgetContext", "as_symmetric",
           "BudgetExhaustedError", "TimeExhaustedError"]


class BudgetExhaustedError(RuntimeError):
    pass


class TimeExhaustedError(RuntimeError):
    pass


class BudgetContext:
    def __init__(self, flop_budget=None, **kwargs):
        self.flop_budget = flop_budget

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __getattr__(self, name):
        raise AttributeError(f"{name!r}: the torch twin shim does not count FLOPs")


def as_symmetric(a, symmetry=(0, 1)):
    return a
