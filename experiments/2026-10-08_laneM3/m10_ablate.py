"""Lane M3 / M10.1 -- which K=4 SIMPLE births carry the fourth-order gain?  ARC's sweep, patched.

Copy into ARC's `scripts/` next to m9_sweep.py and run from there after M9 has filled the
sample cache (same nets and labels, so every arm is paired with M9's runs by (n, trial)):

    cd mlp_cumulant_propagation/scripts && python m10_ablate.py [arm ...]

The patch rewrites only `factor_k4.factored_nonlin_kprop_k4` (factor_k4.py:478-506 birth loops,
:434 memory), scaling each birth class by a flag:

    path  alpha=12 path births       star  alpha=4 star births      p32  alpha=12 κ3×edge (3+2)
    mem   0 -> the input κ4 factors are not carried: κ4 = this layer's births + correction only
"""
import inspect
import os
import sys
import textwrap

import torch

import m9_sweep as m9
from kprop_by_width import HKind, KPropByWidthCfg, kprop_by_width
import mlp_kprop.factor_k4 as fk4

ARMS = {
    "full": dict(path=1, star=1, p32=1, mem=1),        # == k4simple_fac (sanity)
    "star": dict(path=0, star=1, p32=0, mem=1),
    "path": dict(path=1, star=0, p32=0, mem=1),
    "p32": dict(path=0, star=0, p32=1, mem=1),
    "starpath": dict(path=1, star=1, p32=0, mem=1),
    "none": dict(path=0, star=0, p32=0, mem=1),        # top-slice births off: κ4 from corrections only
    "nomem": dict(path=1, star=1, p32=1, mem=0),
}
REPL = [
    ("B_jkl.add_(tmp, alpha=12)  # number of path graphs", "B_jkl.add_(tmp, alpha=12 * _ABL['path'])"),
    ("B_jkl.add_(tmp, alpha=4)  # number of star graphs", "B_jkl.add_(tmp, alpha=4 * _ABL['star'])"),
    ("B_jkl.add_(tmp, alpha=12)  # number of 3+2 graphs", "B_jkl.add_(tmp, alpha=12 * _ABL['p32'])"),
    ("pK_1111 = WK[4].clone()",
     "pK_1111 = WK[4].clone() if _ABL['mem'] else FactoredTensor4(n=n, device=mean.device, dtype=mean.dtype)"),
]
_SRC = textwrap.dedent(inspect.getsource(fk4.factored_nonlin_kprop_k4))
for old, new in REPL:
    assert _SRC.count(old) == 1, old
    _SRC = _SRC.replace(old, new)
exec(compile(_SRC, fk4.__file__ + "<m10>", "exec"), fk4.__dict__)
NS = (96, 128) if not os.environ.get("M9_SMOKE") else (16, 24)

if __name__ == "__main__":
    for arm in sys.argv[1:] or list(ARMS):
        fk4._ABL = ARMS[arm]
        tag = "relu_w2_m9smoke" if os.environ.get("M9_SMOKE") else "relu_w2_m9"
        cfg = KPropByWidthCfg(k_maxs=(4,), ns=NS, name=f"{tag}/m10_k4s_{arm}/kprop_by_width",
                              kprop_kwargs=dict(kind=HKind.SIMPLE, factor=True, use_avg_metric=True), **m9.COMMON)
        print(f"=== m10 {arm} {fk4._ABL}", flush=True)
        kprop_by_width(cfg)
        print(f"=== m10 {arm} done", flush=True)
