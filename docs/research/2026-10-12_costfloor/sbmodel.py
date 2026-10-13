def sb_cost(b, m, k, n, depth, block_out=False, glue="std"):
    """Return (flops, ops) of _sb((b,m,k),(b,k,n),depth) under flopscope 0.12.1 (float32).

    glue = 'std'  shipped spelling: 8 pre-adds, 2 seven-block concatenates, 7 post-adds, block/concat reassembly
                  (block_out=True is v82+'s single `fnp.block`: 33 quarter-areas = 8.25 x block area per level)
           'out'  x3 S6 model, out=-view spelling: the 4 computed operand sums per side are written straight into
                  the stacked operand (only the 3 raw blocks per side are copied), post-adds write into views of
                  the result (no reassembly): 21 quarter-areas = 5.25 x block area per level
           'min'  lower bound: no raw copies either: 15 quarter-areas = 3.75 x
    """
    if depth <= 0 or (m % 2) or (k % 2) or (n % 2):
        return b*m*n*(2*k-1), 1
    mh, kh, nh = m//2, k//2, n//2
    if glue == "out":
        g = 7*b*mh*kh + 7*b*kh*nh + 7*b*mh*nh; ops = 8+6+7
    elif glue == "min":
        g = 4*b*mh*kh + 4*b*kh*nh + 7*b*mh*nh; ops = 8+7
    else:
        g = 4*b*mh*kh + 4*b*kh*nh + 7*b*mh*kh + 7*b*kh*nh + 7*b*mh*nh
        if block_out:
            g += b*m*n; ops = 8+2+7+1
        else:
            g += 2*b*m*n; ops = 8+2+7+3
    f, o = sb_cost(7*b, mh, kh, nh, depth-1, block_out, glue)
    return g + f, ops + o


def best_depth(m, k, n, min_leaf=16, maxd=8, glue="std"):
    """FLOP-optimal depth for one shape (leaf >= min_leaf), as the x3 `depth=('opt', leaf)` patch chooses it."""
    best = None
    for d in range(0, maxd + 1):
        if d and (min(m, k, n) >> d) < min_leaf:
            break
        c = sb_cost(1, m, k, n, d, True, glue)[0]
        if best is None or c < best[0]:
            best = (c, d)
    return best[1]


if __name__ == "__main__":
    for K in (1024, 992, 928):
        f, o = sb_cost(1, 1024, K, 4096, 5)
        print(K, 2*f, o)
