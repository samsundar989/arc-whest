import sys, os
sys.path.insert(0, os.path.expanduser("~/scratch"))
from variants6 import *
from collections import defaultdict
def prox2():
    L = fakefs.LEDGER
    glue = 0; leaf = 0; calls = 0; elem = 0; kleaf = defaultdict(int)
    for op, fl, layer, key, shp in L:
        if key[0] in ("_mm", "_sb"):
            if op == "matmul":
                leaf += fl; b = shp[0] if len(shp) == 3 else 1; calls += b
            else: glue += fl
        else: elem += fl
    return glue, leaf, calls, elem
for patches, label in [([], "v81d"), ([set_depth(5, 20, 16, 2**62)], "v81d no bigvol"),
                       ([p_block_sandwich], "block"), ([p_block_sandwich, lambda m: fused_advance(m)], "block+fused"),
                       ([p_block_sandwich, lambda m: fused_advance(m), join_mm, lambda m: skip_join_at(m, {13, 14}), sandwich_split2], "C1s"),
                       ([p_block_sandwich, lambda m: fused_advance(m), join_mm, lambda m: skip_join_at(m, {13, 14}), sandwich_split2, set_depth(6, 20, 12)], "C2")]:
    mod, tot, ops = rc.run("estimator_k3_v81d_starall2.py", patch=lambda m, P=patches: [p(m) for p in P])
    g, l, c, e = prox2()
    print("%-16s F=%.4e ops=%5d glue=%.3e leaf=%.3e leafcalls=%8d elem=%.3e" % (label, tot+32768, ops+1, g, l, c, e))
