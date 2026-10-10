import sys, os
sys.path.insert(0, os.path.expanduser("~/scratch"))
from variants2 import *
from collections import defaultdict
def proxies(label):
    L = fakefs.LEDGER
    glue = 0; leaf = defaultdict(int); gemms = defaultdict(int); elem = 0; mmops = 0
    for op, fl, layer, key, shp in L:
        if key[0] in ("_mm", "_sb"):
            if op == "matmul":
                # leaf: out shape (b, m, n) or (m, n)
                if len(shp) == 3:
                    b, m, n = shp
                else:
                    b, (m, n) = 1, shp[-2:] if len(shp) >= 2 else (1, shp[0] if shp else 1)
                cls = min(m, n)
                cls = 16 if cls <= 16 else (24 if cls <= 24 else (32 if cls <= 32 else (64 if cls <= 64 else 999)))
                leaf[cls] += fl; gemms[cls] += b
            else:
                glue += fl
        else:
            elem += fl
    print("   %-40s glue=%.2f u  elem=%.2f u  leaf by min(m,n) class: %s  | GEMM calls: %s" % (label, glue/U, elem/U,
          {k: round(v/U, 2) for k, v in sorted(leaf.items())}, {k: v for k, v in sorted(gemms.items())}))
if __name__ == "__main__":
    B = [p_block_sandwich, lambda m: fused_advance(m)]
    for patches, label in [([], "baseline"), ([p_block_sandwich], "V1 block"), (B, "V2 block+fused"),
                           (B + [join_mm, lambda m: skip_join_at(m, {14})], "V4 +join_mm+skip15"),
                           (B + [lambda m: skip_join_at(m, {13,14})], "V4' block+fused+skip14,15 (no join_mm)"),
                           (B + [set_depth(6, 20, 12)], "V5b block+fused+depth6 big"),
                           (B + [set_depth(5, 20, 16, 2**62)], "V5c block+fused no bigvol")]:
        run(patches, label)
        proxies(label)
