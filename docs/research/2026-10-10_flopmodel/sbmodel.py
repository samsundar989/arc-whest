def sb_cost(b, m, k, n, depth, block_out=False):
    """Return (flops, ops) of _sb((b,m,k),(b,k,n),depth) under flopscope 0.12.1 (float32)."""
    if depth <= 0 or (m % 2) or (k % 2) or (n % 2):
        return b*m*n*(2*k-1), 1
    mh, kh, nh = m//2, k//2, n//2
    glue = 4*b*mh*kh + 4*b*kh*nh + 7*b*mh*kh + 7*b*kh*nh + 7*b*mh*nh
    if block_out:
        glue += b*m*n; ops = 8+2+7+1
    else:
        glue += 2*b*m*n; ops = 8+2+7+3
    f, o = sb_cost(7*b, mh, kh, nh, depth-1, block_out)
    return glue + f, ops + o

if __name__ == "__main__":
    for K in (1024, 992, 928):
        f, o = sb_cost(1, 1024, K, 4096, 5)
        print(K, 2*f, o)
