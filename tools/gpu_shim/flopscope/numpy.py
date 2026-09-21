"""`flopscope.numpy` (fnp) backed by cupy.

Re-exports cupy under the names the shipped estimator uses. The surface it
actually touches is small — array creation, `einsum`, `matmul`, `concatenate`,
`stack`, `reshape`, `diag`/`diagonal`, `eye`, `ones`/`zeros`/`ones_like`,
`clip`, `maximum`, `where`, `sqrt`, `isfinite`, `argsort`, `linalg.qr`,
`random.default_rng`, and the two float dtypes — and cupy provides every one.

Two deliberate differences from `local_sim/flopscope/numpy.py`:

* `einsum` does NOT force `optimize=True`. cupy's einsum already contracts
  pairwise, and passing the numpy keyword through changes nothing but can
  reject paths cupy computes itself.
* `asnumpy` is exported so the caller can bring a result back to the host. The
  estimator never calls it; `tools/_predict_worker.py` does, once per net.
"""

import cupy as _cp
from cupy import *  # noqa: F401,F403
from cupy import random  # noqa: F401  (fnp.random.default_rng)
from cupy import linalg  # noqa: F401  (fnp.linalg.qr)
from cupy import asnumpy  # noqa: F401

ndarray = _cp.ndarray
float32 = _cp.float32
float64 = _cp.float64


def einsum(*args, **kwargs):
    # numpy's `optimize` keyword is meaningless here and cupy's signature does
    # not promise to accept it; drop it rather than pass it through.
    kwargs.pop("optimize", None)
    return _cp.einsum(*args, **kwargs)


def __getattr__(name):
    """Anything not pulled in by the star-import falls through to cupy.

    A genuinely missing name raises AttributeError naming cupy, which is the
    signal that this backend cannot run a particular candidate — not a reason
    to silently substitute numpy and quietly move the computation to the CPU.
    """
    return getattr(_cp, name)
