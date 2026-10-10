import sys, os, json, functools
sys.path.insert(0, os.path.expanduser("~/scratch"))
from variants5 import *
from proxies import proxies
from sbmodel import sb_cost

def sandwich_split2(mod):
    """second product of W^T C W with a two-level symmetric split (block assembly)."""
    def _sym_rest(X, W):
        n = X.shape[0]; h = n // 2; q = h // 2
        top = mod._mm(X[:h], W)                       # rows 0:h, all columns
        t2 = mod._mm(X[h:h+q], W[:, h:])              # rows h:h+q, columns h:
        b22 = mod._mm(X[h+q:], W[:, h+q:])            # rows h+q:, columns h+q:
        # rows h+q: , columns h:h+q  = t2[:, :q].T
        lower = fnp.block([[top[:, h:].T, fnp.block([[t2], [t2[:, :q].T, b22]])]])
        return fnp.block([[top], [lower]])
    def _p4_sym_sandwich(WT, cov, W):
        return _sym_rest(mod._mm(WT, cov), W)
    mod._p4_sym_sandwich = _p4_sym_sandwich; mod._p4_sym_rest = _sym_rest

def optimal_depth(maxd=7, min_leaf=16):
    def p(mod):
        @functools.lru_cache(maxsize=None)
        def _depth_for(m, k, n):
            best = None
            for d in range(0, maxd + 1):
                if d and (min(m, k, n) >> d) < min_leaf: break
                c, o = sb_cost(1, m, k, n, d, True)
                if best is None or c < best[0]: best = (c, d)
            return best[1]
        mod._depth_for = _depth_for
        mod._STRASSEN_DEPTH = maxd
    return p

if __name__ == "__main__":
    B = [p_block_sandwich, lambda m: fused_advance(m)]
    run([p_block_sandwich, sandwich_split2], "block + sandwich split2")
    for patches, label in [
        (B + [lambda m: skip_join_at(m, {13, 14})], "C0 block+fused+skip13,14"),
        (B + [join_mm, lambda m: skip_join_at(m, {13, 14})], "C1 C0+join_mm"),
        (B + [join_mm, lambda m: skip_join_at(m, {13, 14}), sandwich_split2], "C1s C1+sandwich split2"),
        (B + [join_mm, lambda m: skip_join_at(m, {13, 14}), sandwich_split2, set_depth(6, 20, 12)], "C2 C1s+depth6(big, leaf>=12)"),
        (B + [join_mm, lambda m: skip_join_at(m, {13, 14}), sandwich_split2, optimal_depth(7, 16)], "C3 C1s+FLOP-opt depth (leaf>=16)"),
        (B + [join_mm, lambda m: skip_join_at(m, {13, 14}), sandwich_split2, optimal_depth(7, 20)], "C3b C1s+FLOP-opt depth (leaf>=20)"),
        (B + [join_mm, lambda m: skip_join_at(m, {13, 14}), sandwich_split2, optimal_depth(7, 12)], "C3c C1s+FLOP-opt depth (leaf>=12)"),
    ]:
        run(patches, label); proxies(label)
