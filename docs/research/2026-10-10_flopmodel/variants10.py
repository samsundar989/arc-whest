import sys, os
sys.path.insert(0, os.path.expanduser("~/scratch"))
from variants9 import *
from proxies import proxies
if __name__ == "__main__":
    C1s = [p_block_sandwich, lambda m: fused_advance(m), join_mm, lambda m: skip_join_at(m, {13, 14}), sandwich_split2]
    run(C1s + [optimal_depth(8, 8)], "C3d C1s + FLOP-opt depth (leaf>=8)"); proxies("C3d")
    run(C1s + [optimal_depth(7, 12), last_layer_rows(800), old_rebuild_kept_rows], "C3c + V6 + V7")
    run(C1s + [optimal_depth(7, 12), v8_plus_v6(800)], "C3c + V8 + V6")
