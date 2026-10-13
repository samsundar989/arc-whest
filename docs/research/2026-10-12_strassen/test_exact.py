import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "site"))
import numpy as np
import flopscope as flops, flopscope.numpy as fnp
import swproto as sp
rng = np.random.default_rng(1)
for (m, k, n, d) in [(256, 256, 256, 5), (128, 104, 1312, 2), (512, 448, 320, 4), (216, 2784, 216, 3), (256, 64, 1408, 4)]:
    A = fnp.asarray(rng.standard_normal((m, k)).astype(np.float32)); B = fnp.asarray(rng.standard_normal((k, n)).astype(np.float32))
    with flops.BudgetContext(flop_budget=2**60, quiet=True):
        r = np.asarray(sp.mm_ref(A, B, d)); o = np.asarray(sp.sw_out(A, B, d)); f = np.asarray(sp.sw_out(A, B, d, True)); c = np.asarray(sp.sw_ks(A, B, d))
    print((m, k, n, d), "out bit-identical to _sb:", np.array_equal(r, o), " fused bit-identical:", np.array_equal(r, f),
          " ks bit-identical:", np.array_equal(r, c), " max|ks-ref|/max|ref| = %.2e" % (np.abs(c - r).max() / np.abs(r).max()))
