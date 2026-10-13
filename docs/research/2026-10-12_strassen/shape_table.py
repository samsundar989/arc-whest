"""Fraction-of-dense table for the shapes named in the task (exact models; every model row that could be
run was checked equal to the real meter -- see run_*.log / bench.log)."""
import sys, types
sys.modules.setdefault("flopscope", types.ModuleType("flopscope")); sys.modules.setdefault("flopscope.numpy", types.ModuleType("flopscope.numpy"))
sys.modules["flopscope"].numpy = sys.modules["flopscope.numpy"]
import swproto as sp
def v2(x):
    d = 0
    while x % 2 == 0: x //= 2; d += 1
    return d
shapes = [(1024, 1024, 1024, "square"), (1024, 832, 10592, "_advance"), (1024, 832, 10624, "_advance, n padded to 64"),
          (864, 11136, 864, "contract 787"), (896, 11136, 896, "contract, H multiple of 128"),
          (1024, 256, 5600, "thin-k"), (1024, 256, 5632, "thin-k, n multiple of 512")]
print("%-44s %2s  %-22s %-22s %-22s %-22s" % ("shape", "d", "ref (_sb)  frac  ops", "A out=", "B fused leaf", "C KS/Morton"))
for m, k, n, tag in shapes:
    dense = m * n * (2 * k - 1)
    for d in (4, 5, 6, 7):
        if d > min(v2(m), v2(k), v2(n)):
            print("%-44s %2d  not divisible by 2^%d (ref falls back to depth %d)" % ("(%d,%d)@(%d,%d) %s" % (m, k, k, n, tag), d, d, min(v2(m), v2(k), v2(n)))); continue
        cells = []
        for f, o in (sp.cost_ref(m, k, n, d), sp.cost_out(m, k, n, d), sp.cost_out(m, k, n, d, True), sp.cost_ks(m, k, n, d)):
            cells.append("%13d %.4f %3d" % (f, f / dense, o))
        print("%-44s %2d  %s  %s  %s  %s" % ("(%d,%d)@(%d,%d) %s" % (m, k, k, n, tag), d, *cells))
