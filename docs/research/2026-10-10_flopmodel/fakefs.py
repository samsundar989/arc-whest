"""Shape-only abstract interpreter of flopscope 0.12.1 billing (float32 world).
Billing rules verified on the real flopscope at small shapes (probe_bill.py, probe2.py, probe3.py)."""
import sys, types, math
import numpy as _np

EST_FILES = set()          # basenames of estimator files to attribute to
LEDGER = []                # one record per metered op
STATE = {"hooks": []}

class DT:
    def __init__(self, name): self.name = name
    def __repr__(self): return "fnp." + self.name
    def __eq__(self, o): return isinstance(o, DT) and o.name == self.name
    def __ne__(self, o): return not self.__eq__(o)
    def __hash__(self): return hash(self.name)
    def __call__(self, v=0.0): return FS(float(v) if self.name.startswith("float") else int(v), self)
float32 = DT("float32"); float64 = DT("float64"); int64 = DT("int64"); bool_ = DT("bool")
RATE = {"float32": 1, "float64": 2, "int64": 2, "bool": 1}

def promote(*dts):
    names = {d.name for d in dts if d is not None}
    if "float64" in names: return float64
    if "float32" in names: return float64 if "int64" in names else float32
    if "int64" in names: return int64
    return bool_

class FS:
    """A strongly typed numpy scalar (dtype(...))."""
    def __init__(self, v, dt): self.v, self.dt = v, dt
    def __float__(self): return float(self.v)
    def __int__(self): return int(self.v)
    def __repr__(self): return "FS(%r,%s)" % (self.v, self.dt)
    def _b(self, o, f, rev=False):
        if isinstance(o, FA): return NotImplemented
        ov = o.v if isinstance(o, FS) else o
        odt = o.dt if isinstance(o, FS) else None
        r = f(ov, self.v) if rev else f(self.v, ov)
        return FS(r, promote(self.dt, odt) if odt else self.dt)
    def __add__(s, o): return s._b(o, lambda a, b: a + b)
    def __radd__(s, o): return s._b(o, lambda a, b: a + b, True)
    def __sub__(s, o): return s._b(o, lambda a, b: a - b)
    def __rsub__(s, o): return s._b(o, lambda a, b: a - b, True)
    def __mul__(s, o): return s._b(o, lambda a, b: a * b)
    def __rmul__(s, o): return s._b(o, lambda a, b: a * b, True)
    def __truediv__(s, o): return s._b(o, lambda a, b: a / b)
    def __rtruediv__(s, o): return s._b(o, lambda a, b: a / b, True)
    def __neg__(s): return FS(-s.v, s.dt)
    def __eq__(s, o): return float(s.v) == (o.v if isinstance(o, FS) else o)
    def __ne__(s, o): return not s.__eq__(o)
    def __hash__(s): return hash(s.v)
    def __lt__(s, o): return s.v < (o.v if isinstance(o, FS) else o)
    def __gt__(s, o): return s.v > (o.v if isinstance(o, FS) else o)

def _numel(shape):
    p = 1
    for s in shape: p *= int(s)
    return p

def _site():
    """(layer, site_key, stack) for attribution."""
    f = sys._getframe(2)
    frames = []
    while f is not None:
        fn = f.f_code.co_filename.replace("\\", "/").rsplit("/", 1)[-1]
        if fn in EST_FILES:
            frames.append(f)
        f = f.f_back
    layer = -1
    for fr in frames:
        if fr.f_code.co_name == "_layer":
            layer = fr.f_locals.get("layer_index", -1)
            break
    names = [fr.f_code.co_name for fr in frames]
    if "_mm" in names:
        i = max(j for j, nm in enumerate(names) if nm in ("_mm",))
        caller = frames[i + 1] if i + 1 < len(frames) else frames[i]
        key = ("_mm", caller.f_code.co_name, caller.f_lineno)
    elif "_sb" in names:  # direct _sb call (not through _mm)
        i = max(j for j, nm in enumerate(names) if nm == "_sb")
        caller = frames[i + 1] if i + 1 < len(frames) else frames[i]
        key = ("_sb", caller.f_code.co_name, caller.f_lineno)
    elif frames:
        key = (frames[0].f_code.co_name, frames[0].f_code.co_name, frames[0].f_lineno)
    else:
        key = ("harness", "harness", 0)
    return layer, key

def bill(op, flops, out_shape=()):
    layer, key = _site()
    LEDGER.append((op, int(flops), layer, key, tuple(out_shape)))

class FA:
    __array_priority__ = 10000
    def __init__(self, shape, dtype=float32, sym=False):
        self.shape = tuple(int(s) for s in shape)
        self.dtype = dtype
        self.sym = sym
    def __repr__(self): return "FA(%s,%s)" % (self.shape, self.dtype)
    @property
    def ndim(self): return len(self.shape)
    @property
    def size(self): return _numel(self.shape)
    def __len__(self): return self.shape[0]
    def __bool__(self): raise RuntimeError("materialization __bool__")
    def __float__(self): raise RuntimeError("materialization __float__")
    def __int__(self): raise RuntimeError("materialization __int__")
    def __index__(self): raise RuntimeError("materialization __index__")
    @property
    def T(self): return FA(self.shape[::-1], self.dtype, self.sym)
    def transpose(self, *axes):
        if not axes: return self.T
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)): axes = axes[0]
        return FA(tuple(self.shape[a] for a in axes), self.dtype)
    def __getitem__(self, key):
        parts = key if isinstance(key, tuple) else (key,)
        adv = False
        conv = []
        for p in parts:
            if isinstance(p, FA):
                adv = True
                conv.append(_np.zeros(p.shape, dtype=_np.bool_ if p.dtype.name == "bool" else _np.intp))
            elif isinstance(p, (list, tuple, _np.ndarray)):
                adv = True
                conv.append(_np.zeros(_np.asarray(p).shape, dtype=_np.intp))
            elif isinstance(p, FS):
                conv.append(int(p.v))
            else:
                conv.append(p)
        dummy = _np.broadcast_to(_np.zeros((), dtype=_np.int8), self.shape)
        out_shape = dummy[tuple(conv)].shape
        if adv:
            bill("getitem", 4 * _numel(out_shape) * RATE[self.dtype.name], out_shape)
        return FA(out_shape, self.dtype)
    # ---- elementwise -----------------------------------------------------
    def _bin(self, other, op, rev=False, cmp=False):
        if isinstance(other, FA):
            out = _np.broadcast_shapes(self.shape, other.shape)
            dt = promote(self.dtype, other.dtype)
            sym = self.sym and other.sym and self.shape == other.shape
        elif isinstance(other, FS):
            out = self.shape; dt = promote(self.dtype, other.dt); sym = self.sym
        elif isinstance(other, (bool, int, float, _np.integer, _np.floating)):
            out = self.shape
            if isinstance(other, float) and self.dtype.name in ("int64", "bool"):
                dt = float64
            else:
                dt = self.dtype
            sym = self.sym
        else:
            return NotImplemented
        cost = _numel(out)
        if sym and len(out) == 2 and out[0] == out[1]:
            cost = out[0] * (out[0] + 1) // 2
        bill(op, cost * RATE[dt.name], out)
        return FA(out, bool_ if cmp else dt, sym and not cmp)
    def __add__(s, o): return s._bin(o, "add")
    def __radd__(s, o): return s._bin(o, "add", True)
    def __sub__(s, o): return s._bin(o, "subtract")
    def __rsub__(s, o): return s._bin(o, "subtract", True)
    def __mul__(s, o): return s._bin(o, "multiply")
    def __rmul__(s, o): return s._bin(o, "multiply", True)
    def __truediv__(s, o): return s._bin(o, "divide")
    def __rtruediv__(s, o): return s._bin(o, "divide", True)
    def __ge__(s, o): return s._bin(o, "greater_equal", cmp=True)
    def __gt__(s, o): return s._bin(o, "greater", cmp=True)
    def __le__(s, o): return s._bin(o, "less_equal", cmp=True)
    def __lt__(s, o): return s._bin(o, "less", cmp=True)
    def __neg__(s):
        bill("negative", s.size * RATE[s.dtype.name], s.shape); return FA(s.shape, s.dtype)
    def __matmul__(s, o): return matmul(s, o)
    def __rmatmul__(s, o): return matmul(o, s)
    # ---- reductions / misc -------------------------------------------------
    def sum(self, axis=None, keepdims=False):
        if axis is None:
            out = ()
        else:
            ax = axis % self.ndim
            out = tuple(d for i, d in enumerate(self.shape) if i != ax)
        bill("sum", max(self.size - _numel(out), 1) * RATE[self.dtype.name], out)
        return FA(out, self.dtype)
    def mean(self, axis=None):
        out = () if axis is None else tuple(d for i, d in enumerate(self.shape) if i != axis % self.ndim)
        bill("mean", self.size * RATE[self.dtype.name], out)
        return FA(out, self.dtype)
    def astype(self, dt):
        if dt == self.dtype:
            bill("astype", self.size * RATE[dt.name], self.shape)   # copy=True default bills
            return FA(self.shape, dt)
        bill("astype", self.size * RATE[promote(self.dtype, dt).name], self.shape)
        return FA(self.shape, dt)
    def reshape(self, *shape):
        return reshape(self, shape[0] if len(shape) == 1 else shape)

def _shape_of(x):
    if isinstance(x, FA): return x.shape
    return ()
def _dt_of(x):
    if isinstance(x, FA): return x.dtype
    if isinstance(x, FS): return x.dt
    return None

def matmul(a, b):
    if not isinstance(a, FA): a = FA(_np.shape(a), float32)
    if not isinstance(b, FA): b = FA(_np.shape(b), float32)
    dt = promote(a.dtype, b.dtype)
    if a.ndim == 1 and b.ndim == 1:
        k = a.shape[0]; out = ()
    elif a.ndim == 1:
        k = a.shape[0]; out = b.shape[:-2] + b.shape[-1:]
    elif b.ndim == 1:
        k = b.shape[0]; out = a.shape[:-1]
    else:
        assert a.shape[-1] == b.shape[-2], (a.shape, b.shape)
        k = a.shape[-1]
        batch = _np.broadcast_shapes(a.shape[:-2], b.shape[:-2])
        out = tuple(batch) + (a.shape[-2], b.shape[-1])
    cost = _numel(out) * (2 * k - 1)
    bill("matmul", cost * RATE[dt.name], out)
    return FA(out, dt)

def concatenate(arrs, axis=0):
    arrs = list(arrs)
    dt = promote(*[a.dtype for a in arrs])
    shp = list(arrs[0].shape)
    ax = axis % len(shp)
    shp[ax] = sum(a.shape[ax] for a in arrs)
    bill("concatenate", sum(a.size for a in arrs) * RATE[dt.name], shp)
    return FA(shp, dt)

def stack(arrs, axis=0):
    arrs = list(arrs)
    dt = promote(*[a.dtype for a in arrs])
    shp = list(arrs[0].shape)
    ax = axis % (len(shp) + 1)
    shp.insert(ax, len(arrs))
    bill("stack", sum(a.size for a in arrs) * RATE[dt.name], shp)
    return FA(shp, dt)

def block(nested):
    # assemble shape via numpy.block on dummies
    def conv(x):
        if isinstance(x, list): return [conv(y) for y in x]
        return _np.broadcast_to(_np.zeros((), _np.int8), x.shape)
    def dts(x):
        if isinstance(x, list):
            out = []
            for y in x: out += dts(y)
            return out
        return [x.dtype]
    shp = _np.block(conv(nested)).shape
    dt = promote(*dts(nested))
    bill("block", _numel(shp) * RATE[dt.name], shp)
    return FA(shp, dt)

def reshape(a, shape):
    if isinstance(shape, int): shape = (shape,)
    shape = list(shape)
    if -1 in shape:
        i = shape.index(-1); rest = _numel([s for s in shape if s != -1])
        shape[i] = a.size // rest
    assert _numel(shape) == a.size, (a.shape, shape)
    bill("reshape", max(a.size, 1) * RATE[a.dtype.name], shape)
    return FA(shape, a.dtype)

def take(a, idx, axis=None):
    ishape = idx.shape if isinstance(idx, FA) else _np.shape(idx)
    if axis is None:
        out = tuple(ishape)
    else:
        ax = axis % a.ndim
        out = a.shape[:ax] + tuple(ishape) + a.shape[ax + 1:]
    bill("take", 4 * _numel(out) * RATE[a.dtype.name], out)
    return FA(out, a.dtype)

def where(c, x, y):
    shapes = [_shape_of(v) for v in (c, x, y)]
    out = _np.broadcast_shapes(*shapes)
    dts = [d for d in (_dt_of(x), _dt_of(y)) if d is not None]
    if not dts:
        dt = float64 if isinstance(x, float) or isinstance(y, float) else int64
    else:
        dt = promote(*dts)
        if dt.name in ("int64", "bool") and (isinstance(x, float) or isinstance(y, float)):
            dt = float64
    bill("where", 4 * _numel(out) * RATE[dt.name], out)
    return FA(out, dt)

def _unary(name, w=1, out_dt=None):
    def f(x, *a, **k):
        bill(name, w * x.size * RATE[x.dtype.name], x.shape)
        return FA(x.shape, out_dt or x.dtype)
    return f
sqrt = _unary("sqrt"); tanh = _unary("tanh", 16); isfinite = _unary("isfinite", 1, bool_)
def nan_to_num(x, **k):
    bill("nan_to_num", x.size * RATE[x.dtype.name], x.shape); return FA(x.shape, x.dtype)
def maximum(x, y): return x._bin(y, "maximum") if isinstance(x, FA) else y._bin(x, "maximum", True)
def divide(x, y):
    if isinstance(y, FA) and not isinstance(x, FA): return y._bin(x, "divide", True)
    return x._bin(y, "divide")
def clip(x, lo, hi):
    dt = promote(x.dtype, _dt_of(lo), _dt_of(hi)) if _dt_of(lo) else x.dtype
    bill("clip", 2 * x.size * RATE[dt.name], x.shape); return FA(x.shape, dt)
def argsort(x, axis=-1):
    m = x.shape[-1]
    rows = x.size // max(m, 1)
    lg = 1 if m <= 1 else max(math.ceil(math.log2(m)), 1)
    bill("argsort", 4 * rows * m * lg * RATE[x.dtype.name], x.shape)
    return FA(x.shape, int64)
def zeros(shape, dtype=float32):
    if isinstance(shape, int): shape = (shape,)
    bill("zeros", 0, shape); return FA(shape, dtype, sym=(len(shape) == 2 and shape[0] == shape[1]))
def ones(shape, dtype=float32):
    if isinstance(shape, int): shape = (shape,)
    bill("ones", _numel(shape) * RATE[dtype.name], shape)
    return FA(shape, dtype, sym=(len(shape) == 2 and shape[0] == shape[1]))
def ones_like(x):
    bill("ones_like", x.size * RATE[x.dtype.name], x.shape); return FA(x.shape, x.dtype)
def eye(n, dtype=float32):
    bill("eye", n * RATE[dtype.name], (n, n)); return FA((n, n), dtype, sym=True)
def diag(v):
    if v.ndim == 1:
        bill("diag", v.size * RATE[v.dtype.name], (v.shape[0],) * 2)
        return FA((v.shape[0], v.shape[0]), v.dtype, sym=True)
    bill("diag", min(v.shape) * RATE[v.dtype.name], (min(v.shape),))
    return FA((min(v.shape),), v.dtype)
def diagonal(x):
    bill("diagonal", 0, (min(x.shape),)); return FA((min(x.shape),), x.dtype)
def pad(a, pads):
    out = tuple(d + p[0] + p[1] for d, p in zip(a.shape, pads))
    bill("pad", _numel(out) * RATE[a.dtype.name], out); return FA(out, a.dtype)
def array(v, dtype=float32):
    shp = _np.shape(v)
    bill("array", _numel(shp) * RATE[dtype.name], shp); return FA(shp, dtype)
def asarray(v, dtype=float32):
    if isinstance(v, FA): return v
    shp = _np.shape(v)
    bill("asarray", 2 * _numel(shp), shp); return FA(shp, dtype)
def einsum(subs, *ops):
    shapes = [o.shape for o in ops]
    dt = promote(*[o.dtype for o in ops])
    ins, out = subs.split("->")
    sizes = {}
    for part, s in zip(ins.split(","), shapes):
        for ch, d in zip(part, s): sizes[ch] = d
    oshape = tuple(sizes[c] for c in out)
    alpha = _numel(list(sizes.values())); M = _numel(oshape)
    cost = (len(ops) - 1) * alpha + (alpha - M)
    bill("einsum", cost * RATE[dt.name], oshape)
    return FA(oshape, dt)

class _Linalg: pass
linalg = _Linalg()
def _qr(a):
    m, n = a.shape[-2], a.shape[-1]
    k = min(m, n)
    cost = 2 * (2 * m * n * k - 2 * k ** 3 // 3)
    bill("linalg.qr", cost * RATE[a.dtype.name], (m, k))
    return FA((m, k), a.dtype), FA((k, n), a.dtype)
linalg.qr = _qr

class _Norm:
    @staticmethod
    def cdf(x):
        bill("stats.norm.cdf", 96 * x.size, x.shape); return FA(x.shape, float64)
    @staticmethod
    def pdf(x):
        bill("stats.norm.pdf", 54 * x.size, x.shape); return FA(x.shape, float64)

def install():
    fnp = types.ModuleType("flopscope.numpy")
    for name in ["concatenate", "stack", "block", "reshape", "take", "where", "sqrt", "tanh",
                 "isfinite", "nan_to_num", "maximum", "divide", "clip", "argsort", "zeros", "ones",
                 "ones_like", "eye", "diag", "diagonal", "pad", "array", "asarray", "einsum",
                 "linalg", "matmul"]:
        setattr(fnp, name, globals()[name])
    fnp.float32, fnp.float64, fnp.int64 = float32, float64, int64
    fnp.ndarray = FA
    flops = types.ModuleType("flopscope")
    flops.numpy = fnp
    st = types.SimpleNamespace(norm=_Norm())
    flops.stats = st
    wb = types.ModuleType("whestbench")
    class MLP:
        def __init__(self, width, weights): self.width, self.weights = width, weights
    class BaseEstimator:
        pass
    wb.MLP, wb.BaseEstimator = MLP, BaseEstimator
    sys.modules["flopscope"] = flops
    sys.modules["flopscope.numpy"] = fnp
    sys.modules["whestbench"] = wb
    return flops, fnp, wb
