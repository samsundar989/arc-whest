import sys, os
sys.path.insert(0, os.path.expanduser("~/scratch"))
from variants6 import *
if __name__ == "__main__":
    C1s = [p_block_sandwich, lambda m: fused_advance(m), join_mm, lambda m: skip_join_at(m, {13, 14}), sandwich_split2]
    run(C1s + [last_layer_rows(800), old_rebuild_kept_rows], "C1s + V6(800) + V7")
    run(C1s + [set_depth(6, 20, 12), last_layer_rows(800), old_rebuild_kept_rows], "C2 + V6(800) + V7")
    run(C1s + [two_sided_D, last_layer_rows(800)], "C1s + V8 + V6")
    run(C1s + [set_kas("H", T["2.5"]["active"])], "C1s + 2a (H kink-active)")
    # block + fused + skip penultimate joins only (the 'clean' bundle without join_mm/sandwich)
    run([p_block_sandwich, lambda m: fused_advance(m), lambda m: skip_join_at(m, {14})], "block+fused+skip14 only")
