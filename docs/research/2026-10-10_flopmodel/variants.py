import sys, os
sys.path.insert(0, os.path.expanduser("~/scratch"))
import run_census as rc, fakefs
fnp = sys.modules["flopscope.numpy"]
U = 2 * 1024**3 - 1024**2
BASE = 387664679697

def make_sb(mod, block_out=True):
    def _sb(A, B, depth):
        b = A.shape[0]; m = A.shape[1]; k = A.shape[2]; n = B.shape[2]
        if depth <= 0 or (m % 2) or (k % 2) or (n % 2):
            return A @ B
        mh, kh, nh = m // 2, k // 2, n // 2
        a11 = A[:, :mh, :kh]; a12 = A[:, :mh, kh:]; a21 = A[:, mh:, :kh]; a22 = A[:, mh:, kh:]
        b11 = B[:, :kh, :nh]; b12 = B[:, :kh, nh:]; b21 = B[:, kh:, :nh]; b22 = B[:, kh:, nh:]
        s1 = a21 + a22; s2 = s1 - a11; s3 = a11 - a21; s4 = a12 - s2
        t1 = b12 - b11; t2 = b22 - t1; t3 = b22 - b12; t4 = t2 - b21
        SA = fnp.concatenate([a11, a12, s4, a22, s1, s2, s3], axis=0)
        SB = fnp.concatenate([b11, b21, b22, t4, t1, t2, t3], axis=0)
        P = mod._sb(SA, SB, depth - 1)
        p1 = P[0:b]; p2 = P[b:2*b]; p3 = P[2*b:3*b]; p4 = P[3*b:4*b]; p5 = P[4*b:5*b]; p6 = P[5*b:6*b]; p7 = P[6*b:7*b]
        u2 = p1 + p6; u3 = u2 + p7; c11 = p1 + p2; c12 = (u2 + p5) + p3; c21 = u3 - p4; c22 = u3 + p5
        if block_out:
            return fnp.block([[c11, c12], [c21, c22]])
        return fnp.concatenate([fnp.concatenate([c11, c12], axis=2), fnp.concatenate([c21, c22], axis=2)], axis=1)
    return _sb

def p_block(mod):
    mod._sb = make_sb(mod, True)

def p_block_sandwich(mod):
    p_block(mod)
    _mm = lambda A, B: mod._mm(A, B)
    def _p4_sym_sandwich(WT, cov, W):
        X = mod._mm(WT, cov); n = X.shape[0]; h = n // 2
        top = mod._mm(X[:h], W); b22 = mod._mm(X[h:], W[:, h:])
        return fnp.block([[top], [top[:, h:].T, b22]])
    def _p4_sym_rest(X, W):
        n = X.shape[0]; h = n // 2
        top = mod._mm(X[:h], W); b22 = mod._mm(X[h:], W[:, h:])
        return fnp.block([[top], [top[:, h:].T, b22]])
    mod._p4_sym_sandwich = _p4_sym_sandwich; mod._p4_sym_rest = _p4_sym_rest

def fused_advance(mod, include=("A_L", "M", "Z", "Y", "Q"), pad_to=32):
    def _advance(WT, tw):
        WTr = WT if tw.rows is None else fnp.take(WT, tw.rows, axis=1)
        parts = []
        if "A_L" in include: parts.append(("A_L", tw.A_L))
        if tw.M is not None and "M" in include: parts.append(("M", tw.M))
        if tw.Z is not None and "Z" in include: parts.append(("Z", tw.Z))
        if tw.Y is not None and "Y" in include: parts.append(("Y", tw.Y))
        if tw.Q is not None and "Q" in include: parts.append(("Q", tw.Q))
        widths = [p[1].shape[1] for p in parts]
        tot = sum(widths)
        padw = (-tot) % pad_to
        Bs = [p[1] for p in parts]
        if padw:
            Bs.append(fnp.zeros((WTr.shape[1], padw), dtype=fnp.float32))
        Bf = fnp.concatenate(Bs, axis=1)
        Of = mod._mm(WTr, Bf)
        res = {}; off = 0
        for (nm, _), w in zip(parts, widths):
            res[nm] = Of[:, off:off + w]; off += w
        A_L = res.get("A_L", None) if "A_L" in include else mod._mm(WTr, tw.A_L)
        WTh = WT if tw.hcols is None else fnp.take(WT, tw.hcols, axis=1)
        if tw.M is None:
            M = WTh
        else:
            Mm = res["M"] if "M" in include else mod._mm(WTr, tw.M)
            M = fnp.concatenate([Mm, WTh], axis=1)
        Z = None if tw.Z is None else (res["Z"] if "Z" in include else mod._mm(WTr, tw.Z))
        out = mod._Tower(None, A_L, M, tw.w2b, tw.k, Z, tw.L, tw.sb, tw.eb)
        if tw.Y is not None:
            out.Y = res["Y"] if "Y" in include else mod._mm(WTr, tw.Y)
            out.CA, out.CB = tw.CA, tw.CB
        Q = None if tw.Q is None else (res["Q"] if "Q" in include else mod._mm(WTr, tw.Q))
        return mod._carry_shared(out, tw, Q)
    mod._advance = _advance

def run(patches, label, path="estimator_k3_v81d_starall2.py"):
    def patch(mod):
        for p in patches: p(mod)
    mod, tot, ops = rc.run(path, patch=patch)
    tot += 32768; ops += 1
    print("%-48s F=%15d  dF=%+8.3f%%  (%+7.2f u)  ops=%6d (%+5d)  F/B=%.5f" % (label, tot, 100*(tot-BASE)/BASE, (tot-BASE)/U, ops, ops-25741, tot/2**41))
    return mod, tot, ops

if __name__ == "__main__":
    run([], "baseline v81d")
    run([p_block], "V1 block in _sb")
    run([p_block_sandwich], "V1b block in _sb + sandwich")
    run([lambda m: fused_advance(m)], "V2a fused advance (A_L,M,Z,Y,Q) only")
    run([p_block_sandwich, lambda m: fused_advance(m)], "V2 block + fused advance (all 5)")
    run([p_block_sandwich, lambda m: fused_advance(m, ("A_L","M"))], "V2' block + fused (A_L,M)")
    run([p_block_sandwich, lambda m: fused_advance(m, ("A_L","M","Q"))], "V2'' block + fused (A_L,M,Q)")
