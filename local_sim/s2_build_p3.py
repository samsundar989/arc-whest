"""Generate P3 candidates: the kappa3 x covariance DISTINCT-INDEX birth feed (beta1 + beta2).

Layered on the stage-1 generator's output (`s1_build_v32.build`), so the same
patches produce "v31 + P3" (hub routes ()) or "v32 + P3" (routes (i)+(ii)).

DERIVATION AND EXACT CHECK: `local_sim/k3feed_oracle.py` (label-free). To first
order in kappa3 and in the covariance C, an all-distinct output entry receives,
summed over the 6 role assignments (d doubled, s single, t third):

    (beta1)  1/2 kappa3_dds C_dt w3_d w1_s w1_t
    (beta2)  1/2 kappa3_dds C_st w2_d w2_s w1_t
    (gamma)  kappa3_abc C_pq w2_p w2_q w1_r          (distinct input x link)

v32 has none of them; beta1+beta2+gamma closes the exact response to 1.5e-4,
beta1+beta2 alone leaves 20% of it. gamma is a Hadamard product of the whole
incoming tensor with C and has no cheap factored form; it is NOT implemented.

REPRESENTATION. Each beta route is exactly a two-leg (L) block,
T = (1/2) sum_perms A (x) M (x) C, with the carried mode M (= I at birth):

    beta1:  A = d(w1) D21^T ,  C = fac1 d(w3)      (fac1 = d(w1) C11, the existing A_L leg)
    beta2:  A = d(w2) D21   ,  C = fac1 d(w2)      (= C_L)

`D21` is the consumed (2,1) slice at this birth, compressed D21 ~ U V^T by the
same equivariant range finder as `_p2_compress` (rank `_P3_RANK`, 0 = full), so

    A1 = F1 R3^T, F1 = d(w1) V (transported), R3 = U (static)
    A2 = F2 R4^T, F2 = d(w2) U (transported), R4 = V (static)

Consumption of an L block with C = A_L d(c):
    D3  += 3 sum_j A . M . A_L . c
    D21 += (A.M.c) A_L^T  +  (A.A_L.c) M^T  +  (M.A_L.c) R F^T
The first two FOLD into the existing `LA A_L^T` / `LP M^T` contractions; the
third is thin. New products: F transport, A reconstruction O(k n^2 r), one thin
contraction per layer.

COMPENSATION. At birth the blocks' repeated content is T_aai = A_ai C_ai
(T_aaa = 0); a compensated arm subtracts it from S21 (it enters the residual).

Run:  uv run python local_sim/s2_build_p3.py [--emit-all]
"""

from __future__ import annotations

import hashlib
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]

import s1_build_v32 as S1  # noqa: E402

_P3_FUNCS = '''
def _p3_compress(D):
    """`D ~= U V^T` at rank `_P3_RANK` (0 = full) by the equivariant range finder
    `_p2_compress` uses: the sketch is `D`'s own strongest columns by norm."""
    n = D.shape[0]
    r = n if _P3_RANK <= 0 else min(int(_P3_RANK), n)
    Om = D if r >= n else D[:, fnp.argsort((D * D).sum(axis=0))[-r:]]
    Q, _ = fnp.linalg.qr(D @ Om)
    Q, _ = fnp.linalg.qr(D @ (D.T @ Q))
    return Q, (Q.T @ D).T


def _p3_parts(tw):
    """P3's shared per-layer pieces `(Gp, TT, d3p)`, cached on the tower.

    `Gp = A1 . w3b + A2 . w2b` in the wide `(n, k n)` layout, `TT` the thin
    left factors `(M.A_L.c) R` interleaved like `F`, `d3p` the D3 addition.
    """
    if tw._p3 is not None:
        return tw._p3
    n, k = tw.M.shape[0], tw.k
    r = tw.R3.shape[2]
    F4 = fnp.reshape(tw.F, (n, k, 2 * r)).transpose(1, 0, 2)
    A1 = fnp.reshape((F4[:, :, :r] @ tw.R3.transpose(0, 2, 1)).transpose(1, 0, 2), (n, k * n))
    A2 = fnp.reshape((F4[:, :, r:] @ tw.R4.transpose(0, 2, 1)).transpose(1, 0, 2), (n, k * n))
    Gp = A1 * tw.w3b + A2 * tw.w2b
    AM = tw.A_L * tw.M
    MC3 = fnp.reshape(AM * tw.w3b, (n, k, n)).transpose(1, 0, 2) @ tw.R3
    MC2 = fnp.reshape(AM * tw.w2b, (n, k, n)).transpose(1, 0, 2) @ tw.R4
    TT = fnp.reshape(fnp.concatenate([MC3, MC2], axis=2).transpose(1, 0, 2), (n, 2 * k * r))
    d3p = (Gp * AM).sum(axis=1) * 3.0
    tw._p3 = (Gp, TT, d3p)
    return tw._p3


'''


def _patches():
    P = []

    def add(old, new, why, count=1):
        P.append((old, new, why, count))

    add('''                 "Y", "CA", "CB", "_hub")''',
        '''                 "Y", "CA", "CB", "_hub",
                 # P3: kappa3 x C distinct feed. `F` is `(n, 2 k r)` TRANSPORTED
                 # (per lineage [F1 | F2]); `R3`/`R4` are `(k, n, r)` STATIC;
                 # `w3b` is `(k n,)` STATIC. `_p3` caches `_p3_parts`.
                 "F", "R3", "R4", "w3b", "_p3")''', "slots")
    add('''        self._hub = None
''', '''        self._hub = None
        self.F = self.R3 = self.R4 = self.w3b = None
        self._p3 = None
''', "init")
    add('''        out.Y, out.CA, out.CB = _mm(WT, tw.Y), tw.CA, tw.CB
    return out''', '''        out.Y, out.CA, out.CB = _mm(WT, tw.Y), tw.CA, tw.CB
    if tw.F is not None:
        out.F, out.R3, out.R4, out.w3b = _mm(WT, tw.F), tw.R3, tw.R4, tw.w3b
    return out''', "advance")
    add('''        out.Y, out.CA, out.CB = tw.Y * col, tw.CA, tw.CB
    return out''', '''        out.Y, out.CA, out.CB = tw.Y * col, tw.CA, tw.CB
    if tw.F is not None:
        out.F, out.R3, out.R4, out.w3b = tw.F * col, tw.R3, tw.R4, tw.w3b
    return out''', "wick_tower")
    add('''            d3 = d3 + (RY * tw.Y).sum(axis=1) * 2.0
        return d3''', '''            d3 = d3 + (RY * tw.Y).sum(axis=1) * 2.0
        if tw.F is not None:
            d3 = d3 + _p3_parts(tw)[2]
        return d3''', "dslice3")
    add('''        acc = _mm(LA, tw.A_L.T) + _mm(LP, tw.M.T)
        if tw.Y is not None:
            acc = acc + _mm(RY * (2.0 / 3.0), tw.Y.T)''',
        '''        if tw.F is not None:
            # P3's two L blocks fold into the same two contractions
            Gp, TT, _d3p = _p3_parts(tw)
            LA = LA + Gp * tw.M
            LP = LP + Gp * tw.A_L
        acc = _mm(LA, tw.A_L.T) + _mm(LP, tw.M.T)
        if tw.Y is not None:
            acc = acc + _mm(RY * (2.0 / 3.0), tw.Y.T)
        if tw.F is not None:
            acc = acc + _mm(TT, tw.F.T)''', "dslice21")
    add('''            out.Y, out.CA, out.CB = tw.Y[:, -(2 * g):], tw.CA[-keep:], tw.CB[-keep:]
        return out''', '''            out.Y, out.CA, out.CB = tw.Y[:, -(2 * g):], tw.CA[-keep:], tw.CB[-keep:]
        if tw.F is not None:
            r3 = tw.R3.shape[2]
            out.F, out.R3, out.R4 = tw.F[:, -(2 * g * r3):], tw.R3[-g:], tw.R4[-g:]
            out.w3b = tw.w3b[-keep:]
        return out''', "truncate")
    add('''        e_b = w[(1, 2)] - 2.0 * pK[(1,)] * w1
        S_sep = (e_b[:, None] * C11) * w1[None, :]
        if _HUB_ROUTES:''', '''        if _P3_ON:
            # P3 -- kappa3 x C distinct-index feed, see `local_sim/s2_build_p3.py`
            w3_p3 = w[(3, 1)]
            if adv is None:
                _r3 = n if _P3_RANK <= 0 else min(int(_P3_RANK), n)
                U3 = fnp.zeros((n, _r3), dtype=dtype)
                V3 = U3
            else:
                U3, V3 = _p3_compress(D21)
            F3b = fnp.concatenate([w1[:, None] * V3, w2[:, None] * U3], axis=1)
            if _P3_COMPENSATE and adv is not None:
                D21c = U3 @ V3.T
                S21 = (S21 - ((w1_sq[:, None] * D21c.T) * C11) * w3_p3[None, :]
                       - (((w1 * w2)[:, None] * D21c) * C11) * w2[None, :])
        e_b = w[(1, 2)] - 2.0 * pK[(1,)] * w1
        S_sep = (e_b[:, None] * C11) * w1[None, :]
        if _HUB_ROUTES:''', "birth")
    add('''    out_tw = _truncate(out_tw, gens, n)''', '''    if _P3_ON and _P2_SEPARABLE:
        if adv is None or adv.F is None:
            out_tw.F, out_tw.R3, out_tw.R4, out_tw.w3b = F3b, U3[None], V3[None], w3_p3
        else:
            out_tw.F = fnp.concatenate([adv.F, F3b], axis=1)
            out_tw.R3 = fnp.concatenate([adv.R3, U3[None]], axis=0)
            out_tw.R4 = fnp.concatenate([adv.R4, V3[None]], axis=0)
            out_tw.w3b = fnp.concatenate([adv.w3b, w3_p3])
    out_tw = _truncate(out_tw, gens, n)''', "attach")
    add('''        if _HUB_ROUTES and l < depth - 1:
            thin += 12 * n * n                     # birth-time hub construction
    return base + thin''', '''        if _HUB_ROUTES and l < depth - 1:
            thin += 12 * n * n                     # birth-time hub construction
        if _P3_ON:
            r3 = n if _P3_RANK <= 0 else min(int(_P3_RANK), n)
            if k:
                thin += n * (2 * k * r3) * (2 * n - 1)       # F transport
                thin += 2 * k * n * n * (2 * r3 - 1)         # A1, A2 reconstruction
                thin += 2 * k * n * r3 * (2 * n - 1)         # (M.A_L.c) R
                thin += n * n * (4 * k * r3 - 1)             # TT F^T
                thin += 20 * k * n * n                       # elementwise
            if l < depth - 1:
                thin += 4 * n * r3 * (2 * n - 1) + 4 * n * r3 * r3 + 4 * n * n
                thin += n * n * (2 * r3 - 1) + 10 * n * n    # D21c, compensation
    return base + thin''', "cost-model")
    return P


def build(routes, hub_comp, rank, p3_rank, p3_comp, title):
    with open(S1.SOURCE, "rb") as fh:
        raw = fh.read()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != S1.V31_SHA:
        raise SystemExit("v31 is not the frozen graded bytes")
    body = S1.build(routes, hub_comp, rank, title, raw.decode("utf-8"), sha)
    for old, new, why, count in _patches():
        found = body.count(old)
        if found != count:
            raise SystemExit("P3 PATCH ANCHOR COUNT WRONG (%s): expected %d, found %d\n%s"
                             % (why, count, found, old[:300]))
        body = body.replace(old, new, count)
    anchor = "_HUB_COMPENSATE = "
    i = body.index(anchor)
    j = body.index("\n", i)
    body = (body[:j + 1] + "\n#: P3 kappa3 x C distinct-index feed (beta1 + beta2).\n"
            "_P3_ON = True\n#: D21 compression rank; 0 = full.\n_P3_RANK = %d\n"
            "#: Subtract the P3 blocks' birth repeated content.\n_P3_COMPENSATE = %r\n"
            % (p3_rank, bool(p3_comp)) + body[j + 1:])
    fa = "def _dslice3(tw):"
    body = body.replace(fa, _P3_FUNCS.lstrip("\n") + fa, 1)
    body = body.replace("GENERATED by `local_sim/s1_build_v32.py`",
                        "GENERATED by `local_sim/s2_build_p3.py` (on `s1_build_v32.py`)", 1)
    return body


ARMS = {
    "estimator_k3_v33.py": (("i", "ii"), True, 32, 32, True,
                            "v33: v32 plus P3 (kappa3 x C distinct feed, beta1+beta2), rank 32, compensated."),
    "estimator_k3_v33_uncomp.py": (("i", "ii"), True, 32, 32, False,
                                   "v33_uncomp DIAGNOSTIC: v32 plus P3, P3 repeated content not compensated."),
    "estimator_k3_v33_dense.py": (("i", "ii"), True, 0, 0, True,
                                  "v33_dense PARITY ARM: v32 dense plus P3 at full rank."),
    "estimator_k3_v31p3.py": ((), True, 32, 32, True,
                              "v31p3: v31 plus P3 only (no stage-1 hubs), rank 32, compensated."),
    "estimator_k3_v31p3_dense.py": ((), True, 0, 0, True,
                                    "v31p3_dense PARITY ARM: v31 dense plus P3 at full rank."),
}


def emit(names=None):
    for name, args in ARMS.items():
        if names and name not in names:
            continue
        text = build(*args)
        with open(os.path.join(_ROOT, name), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        print("  wrote %-30s sha256 %s" % (name, hashlib.sha256(text.encode()).hexdigest()[:16]))


if __name__ == "__main__":
    emit([a for a in sys.argv[1:] if a.endswith(".py")] or None)
