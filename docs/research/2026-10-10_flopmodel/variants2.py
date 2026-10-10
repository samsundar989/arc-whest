import sys, os
sys.path.insert(0, os.path.expanduser("~/scratch"))
from variants import *

def join_mm(mod):
    """route the join's plain `@` products (839, 845, 854) through _mm."""
    import types
    src = open(mod.__file__).read()
    # rebuild _shared_join_lean with _mm spellings via exec of a modified source
    start = src.index("def _shared_join_lean(tw):")
    end = src.index("def _proj_coef_d4_r2(n):")
    body = src[start:end]
    body = body.replace("Y = Y + tw.Q @ (tw.Sg @ (tw.Q.T @ Om))", "Y = Y + _mm(tw.Q, _mm(tw.Sg, _mm(tw.Q.T, Om)))")
    body = body.replace("T = Qn.T @ tw.Q", "T = _mm(Qn.T, tw.Q)")
    body = body.replace("Sg = None if _QUERY_BASIS else (T @ tw.Sg) @ T.T + Sj", "Sg = None if _QUERY_BASIS else _mm(_mm(T, tw.Sg), T.T) + Sj")
    exec(compile("\n" * src[:start].count("\n") + body, mod.__file__, "exec"), mod.__dict__)

def skip_join_at(mod, layers):
    """skip the shared join at the given layer indices (0-based)."""
    orig = mod._shared_join_lean
    def wrapped(tw):
        f = sys._getframe(1)
        li = f.f_locals.get("layer_index", None)
        if li in layers:
            return tw
        return orig(tw)
    mod._shared_join_lean = wrapped

def set_depth(D, MB, BIGBLOCK=None, BIGVOL=None):
    def p(mod):
        mod._STRASSEN_DEPTH = D; mod._STRASSEN_MIN_BLOCK = MB
        if BIGBLOCK is not None: mod._STRASSEN_BIG_BLOCK = BIGBLOCK
        if BIGVOL is not None: mod._STRASSEN_BIG_VOL = BIGVOL
    return p

if __name__ == "__main__":
    B = [p_block_sandwich, lambda m: fused_advance(m)]
    run([join_mm], "V3a join products via _mm (alone)")
    run(B + [join_mm], "V3 block+fused+join_mm")
    run([lambda m: skip_join_at(m, {14})], "V4a skip join at layer idx14 (L15) alone")
    run(B + [join_mm, lambda m: skip_join_at(m, {14})], "V4 block+fused+join_mm+skip L15 join")
    run(B + [join_mm, lambda m: skip_join_at(m, {13, 14})], "V4b ... skip L14,L15 joins")
    run(B + [join_mm, lambda m: skip_join_at(m, {12, 13, 14})], "V4c ... skip L13-15 joins")
    run(B + [set_depth(6, 16)], "V5a block+fused depth6/min16")
    run(B + [set_depth(6, 20, 12)], "V5b block+fused depth6/min20/bigblock12")
    run(B + [set_depth(5, 20, 16, 2**62)], "V5c block+fused, no bigvol level")
