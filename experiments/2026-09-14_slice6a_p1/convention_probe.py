"""Measure the convention factor between F21's scalar c4 and v30's per-neuron g.

At lambda = 0 the two carry the same physical object by different conventions.
The ratio mean(g_out) / c4_out per layer IS the factor, measured rather than
guessed at. Three guesses gave three different residuals; this gives a number.
"""
import importlib.util
import io
import os
import sys
import tempfile

sys.path[:] = [p for p in sys.path
               if os.path.abspath(p or ".") != os.path.abspath("local_sim")] \
    + [".", "local_sim"]

import numpy as np
import flopscope as flops
import harness
from mint_provenance import canonical_mint

flops.configure(symmetry_warnings=False)


def instrument(path, anchor, expr, name):
    src = io.open(path, encoding="utf-8").read()
    assert src.count(anchor) == 1, (path, anchor)
    hook = "    _CONV.append(float(np.asarray(%s).mean()))\n" % expr
    src = src.replace(anchor, hook + anchor, 1)
    src = src.replace("import flopscope as flops",
                      "import numpy as np\nimport flopscope as flops", 1)
    src = src.replace("_TABLES = None", "_CONV = []\n_TABLES = None", 1)
    d = tempfile.mkdtemp()
    fp = os.path.join(d, name + ".py")
    io.open(fp, "w", encoding="utf-8", newline="\n").write(src)
    sp = importlib.util.spec_from_file_location(name, fp)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


f21 = instrument("estimator_k3_strassen_L4_f21.py",
                 "    return (mu_out, cov_out, out_tw, c4_out)",
                 "c4_out", "f21c")
v30 = instrument("estimator_k3_v30.py",
                 "    return (mu_out, cov_out, out_tw, c4_out, g_out)",
                 "g_out", "v30c")

W, D, B = 96, 6, 2 ** 41


class C:
    width, depth, flop_budget, seed, n_mlps = W, D, B, 0, 1


for m in (f21, v30):
    m.Estimator().setup(C())

for wname, seed in (("net A", 90000000401), ("net B", 90000000402)):
    w, es, _ = canonical_mint(W, D, seed)
    vals = {}
    for m, lab in ((f21, "f21"), (v30, "v30")):
        m._CONV.clear()
        mlp = harness.MLP(weights=[np.asarray(x) for x in w], width=W,
                          depth=D, seed=es)
        with flops.BudgetContext(flop_budget=B, quiet=True):
            m.Estimator().predict(mlp, B)
        vals[lab] = list(m._CONV)
    print(wname)
    print("   F21 c4_out      ", ["%.4e" % x for x in vals["f21"]])
    print("   v30 mean(g_out) ", ["%.4e" % x for x in vals["v30"]])
    r = [v / f if f else float("nan")
         for f, v in zip(vals["f21"], vals["v30"])]
    print("   ratio v30/F21   ", ["%.5f" % x for x in r])
print()
print("n = %d, metric = 2.0, so candidate factors: n=%d, metric=2, n*metric=%d, "
      "1/n=%.5f" % (W, W, 2 * W, 1.0 / W))
