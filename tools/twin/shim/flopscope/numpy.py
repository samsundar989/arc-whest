"""`flopscope.numpy` (fnp) on torch tensors -- the surface the chain uses, no more.

Numpy idioms the estimator writes on arrays are given to `torch.Tensor` here:
`.astype(dtype)` and a many-axis `.transpose(1, 0, 2)` (torch's own takes two).
`fnp.float32` is both a dtype and a scalar constructor (`dtype(0.0)`), exactly as
numpy's is, so it is a small callable wrapper around `torch.float32`.

Anything the chain calls that is not defined here raises AttributeError -- the
signal that a candidate uses a function this backend has not been checked on.
"""
import torch as _t

import os as _os

#: TWIN_DEVICE=cpu forces CPU (CPU-only queue jobs, e.g. the forward-only closure bench)
DEVICE = _t.device(_os.environ.get("TWIN_DEVICE") or ("cuda" if _t.cuda.is_available() else "cpu"))
ndarray = _t.Tensor


class _DType:
    def __init__(self, td):
        self.t = td

    def __call__(self, v):
        if isinstance(v, _t.Tensor):
            return v.to(self.t)
        return _t.tensor(v, dtype=self.t, device=DEVICE)

    def __eq__(self, o):
        if isinstance(o, _DType):
            return o.t == self.t
        return o == self.t

    def __hash__(self):
        return hash(self.t)

    def __repr__(self):
        return f"fnp.{self.t}"


float32 = _DType(_t.float32)
float64 = _DType(_t.float64)


def _td(d):
    return d.t if isinstance(d, _DType) else d


# ---- numpy idioms on the tensor type ---------------------------------------
_orig_transpose = _t.Tensor.transpose


def _astype(self, d):
    return self.to(_td(d))


def _transpose(self, *dims):
    if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
        dims = tuple(dims[0])
    if len(dims) == 2:
        return _orig_transpose(self, *dims)       # torch semantics == numpy's for 2 axes
    if not dims:
        return self.permute(*reversed(range(self.dim())))
    return self.permute(*dims)


_t.Tensor.astype = _astype
_t.Tensor.transpose = _transpose


def _tens(x, like=None):
    if isinstance(x, _t.Tensor):
        return x
    dt = like.dtype if isinstance(like, _t.Tensor) else _t.float32
    return _t.tensor(x, dtype=dt, device=DEVICE)


# ---- creation ---------------------------------------------------------------
def zeros(shape, dtype=float32):
    return _t.zeros(shape, dtype=_td(dtype), device=DEVICE)


def arange(n):
    """Integer index range (int64), as the KAS hcols concatenation uses it."""
    return _t.arange(n, device=DEVICE)


def ones(shape, dtype=float32):
    return _t.ones(shape, dtype=_td(dtype), device=DEVICE)


def ones_like(x):
    return _t.ones_like(x)


def eye(n, dtype=float32):
    return _t.eye(n, dtype=_td(dtype), device=DEVICE)


def array(obj, dtype=float32):
    if isinstance(obj, _t.Tensor):
        return obj.to(_td(dtype))
    return _t.tensor(obj, dtype=_td(dtype), device=DEVICE)


def asarray(obj, dtype=None):
    if isinstance(obj, _t.Tensor):
        return obj if dtype is None else obj.to(_td(dtype))
    return _t.tensor(obj, dtype=_td(dtype) if dtype is not None else None, device=DEVICE)


# ---- shape ------------------------------------------------------------------
def concatenate(seq, axis=0):
    return _t.cat(list(seq), dim=axis)


def stack(seq, axis=0):
    return _t.stack(list(seq), dim=axis)


def reshape(x, shape):
    return x.reshape(shape)


def diagonal(x):
    return _t.diagonal(x)


def diag(x):
    return _t.diag(x)


# ---- elementwise / reductions -------------------------------------------------
def where(c, a, b):
    if not isinstance(a, _t.Tensor):
        a = _tens(a, b)
    if not isinstance(b, _t.Tensor):
        b = _tens(b, a)
    return _t.where(c, a, b)


def maximum(a, b):
    if not isinstance(b, _t.Tensor):
        return _t.clamp_min(a, b)
    if not isinstance(a, _t.Tensor):
        return _t.clamp_min(b, a)
    return _t.maximum(a, b)


def clip(a, lo, hi):
    return _t.clamp(a, lo, hi)


def sqrt(x):
    return _t.sqrt(x)


def tanh(x):
    return _t.tanh(x)


def isfinite(x):
    return _t.isfinite(x)


def divide(a, b):
    return _t.div(_tens(a, b), b)


def argsort(x):
    return _t.argsort(x)


def take(x, idx, axis=0):
    """numpy.take with an integer index tensor (P2 compaction gathers)."""
    return _t.index_select(x, axis, idx.long())


def pad(x, widths):
    """numpy.pad (constant zero) with numpy's per-axis ((lo, hi), ...) order."""
    flat = []
    for lo, hi in reversed(widths):
        flat += [int(lo), int(hi)]
    return _t.nn.functional.pad(x, flat)


def matmul(a, b):
    return _t.matmul(a, b)


def einsum(spec, *ops, **kw):
    kw.pop("optimize", None)
    return _t.einsum(spec, *ops)


class linalg:
    @staticmethod
    def qr(a):
        q, r = _t.linalg.qr(a)
        return q, r

    @staticmethod
    def inv(a):
        return _t.linalg.inv(a)
