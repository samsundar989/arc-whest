"""Generate the stage-1 candidates from frozen v31: the DISTINCT-INDEX kappa4 birth feed.

Contract: `experiments/LOG.md`, 2026-09-16, *EXECUTION CONTRACT AMENDMENT
(pre-data)* ("Changed D2/D3 discriminator") and the stage-1 contract entry
that follows the oracle result. Oracle: `local_sim/k4feed_oracle.py`.

WHAT THE ORACLE ESTABLISHED (label-free, exact reference)
---------------------------------------------------------
At one birth from an exact non-Gaussian law whose fourth cumulant is the
estimator's own model `kappa4 = m Sym(G (x) I)`, `G = diag(dG) + lam C_off`:

  * v31's newborn source has ZERO kappa4 content at all-distinct indices;
  * the exact first-order content there has two routes,
      (i)  kappa4_ppxy touching all three indices, coefficient 1/2:
           (m/12) lam sum_cyc w2_c w1_a C_ab w1_b
      (ii) kappa4_aabb linked to the third index by one covariance, 1/4:
           (1/4) D22_ab [w2_a w3_b C_bc + w3_a w2_b C_ac] w1_c
    and (i)+(ii) matches the exact response to 6e-5 while (i) alone -- the
    public v17 feed's form -- overshoots it by 53%.

THE REPRESENTATION (no new dense product)
-----------------------------------------
Both routes are sums of HUBS  `Sym3(y (x) S)`  with a transported vector `y`
and a symmetric matrix `S = X M^T + M X^T`, `X = A_L d(c)`, where `A_L`, `M`
are legs every lineage already carries and `c` is a static column scale:

    hub A:  y = w2,        c = (m/8)(lam w1 + dG w3)      routes (i)+(ii)
    hub B:  y = dG w2,     c = (m/8) w3                   route (ii)

(route (i): S = lam d(w1) C d(w1), y = (m/4) w2  ==  y = w2, c = (m/8) lam w1;
 route (ii): (1/4) sum_perms (v (x) M) = (3/4) Sym3(v (x) (M + M^T)).)

Consumed slices of one hub, with R = (A_L . M) c:

    D3  += 2 y . R
    D21 += (2/3) (y . A_L . c) M^T  +  (2/3) (y . M . c) A_L^T  +  (2/3) R y^T

The first two FOLD INTO the existing `LP M^T` and `LA A_L^T` contractions, so
the only new products are one `(n,n)@(n,2k)` transport of `Y` and one thin
`(n,2k)@(2k,n)` contraction per layer.

COMPENSATION. At birth (M = I) a hub's repeated content is
`F_aai = (2/3) y_a (w1_a C_ai c_i + c_a C_ai w1_i)`, `F_aaa = 0`. The table's
repeated slices already carry whatever the table carries, so a COMPENSATED arm
subtracts it: the `d(y c) C d(w1)` half is separable and moves `e_b` for free;
the `d(y w1) C d(c)` half enters the rank-r residual. An UNCOMPENSATED arm
keeps it (diagnostic only).

ARMS
----
    estimator_k3_v32.py          routes (i)+(ii), compensated, rank 32   PRIMARY
    estimator_k3_v32_i.py        route (i) only,  compensated, rank 32   diagnostic
    estimator_k3_v32_uncomp.py   routes (i)+(ii), uncompensated, rank 32 diagnostic
    estimator_k3_v32_dense.py    routes (i)+(ii), compensated, rank 0    parity only

With `_HUB_ROUTES = ()` every hub branch is skipped at Python level, so the
generated file is v31 bit-for-bit in output and counted FLOPs (`--parity`).

Run:  uv run python local_sim/s1_build_v32.py            (emit)
      uv run python local_sim/s1_build_v32.py --parity   (fast gates)
"""

from __future__ import annotations

import hashlib
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
SOURCE = os.path.join(_ROOT, "estimator_k3_v31.py")
V31_SHA = "6dde5fd43ad41e8ec7199b742dd554d22d506c9cf85a532027e86a590d0a787c"

ARMS = (
    ("estimator_k3_v32.py", ("i", "ii"), True, 32,
     "v32: v31 plus the DISTINCT-INDEX kappa4 birth feed, routes (i)+(ii), compensated."),
    ("estimator_k3_v32_i.py", ("i",), True, 32,
     "v32_i DIAGNOSTIC: route (i) only (the public-v17 form), compensated. Never packaged."),
    ("estimator_k3_v32_uncomp.py", ("i", "ii"), False, 32,
     "v32_uncomp DIAGNOSTIC: routes (i)+(ii), repeated content NOT compensated. Never packaged."),
    ("estimator_k3_v32_dense.py", ("i", "ii"), True, 0,
     "v32_dense PARITY ARM: routes (i)+(ii), compensated, full-rank residual. Not a cost candidate."),
)


_HUB_FUNCS = '''
def _hub_parts(tw):
    """The distinct-index kappa4 hubs' shared per-layer pieces: `(G, RY)`.

    `G = sum_h y_h . c_h` in the wide `(n, k n)` layout (y broadcast over its
    lineage's columns) and `RY` the `(n, 2k)` stack of `R_h = (A_L . M) c_h`,
    interleaved `[A_1, B_1, A_2, B_2, ...]` like `Y`. Cached on the tower
    because `_dslice3` and `_dslice21` read the same object in one layer.
    All `O(k n^2)` elementwise work; no matmul.
    """
    if tw._hub is not None:
        return tw._hub
    n, k = tw.M.shape[0], tw.k
    Y3 = fnp.reshape(tw.Y, (n, k, 2))
    cA = fnp.reshape(tw.CA, (k, n))
    cB = fnp.reshape(tw.CB, (k, n))
    yA = Y3[:, :, 0]
    yB = Y3[:, :, 1]
    G = fnp.reshape(yA[:, :, None] * cA[None, :, :] + yB[:, :, None] * cB[None, :, :],
                    (n, k * n))
    AM3 = fnp.reshape(tw.A_L * tw.M, (n, k, n))
    RA = (AM3 * cA[None, :, :]).sum(axis=2)
    RB = (AM3 * cB[None, :, :]).sum(axis=2)
    RY = fnp.reshape(fnp.stack([RA, RB], axis=2), (n, 2 * k))
    tw._hub = (G, RY)
    return tw._hub


'''


def _patches(routes, compensate):
    P = []

    def add(old, new, why, count=1):
        P.append((old, new, why, count))

    add('''                 "Z", "L", "sb", "eb", "_p2")''',
        '''                 "Z", "L", "sb", "eb", "_p2",
                 # STAGE 1: the distinct-index kappa4 hubs. `Y` is `(n, 2k)`
                 # and TRANSPORTED (interleaved A/B per lineage); `CA`/`CB` are
                 # `(k n,)` STATIC column scales. `_hub` caches `_hub_parts`.
                 "Y", "CA", "CB", "_hub")''', "slots")
    add('''        self._p2 = None
''', '''        self._p2 = None
        self.Y = self.CA = self.CB = None
        self._hub = None
''', "init")
    add('''    Z = None if tw.Z is None else _mm(WT, tw.Z)
    return _Tower(A_D, A_L, M, tw.w2b, tw.k, Z, tw.L, tw.sb, tw.eb)''',
        '''    Z = None if tw.Z is None else _mm(WT, tw.Z)
    out = _Tower(A_D, A_L, M, tw.w2b, tw.k, Z, tw.L, tw.sb, tw.eb)
    if tw.Y is not None:
        # a hub's `y` is a LEFT factor: transported like a leg, one thin product
        out.Y, out.CA, out.CB = _mm(WT, tw.Y), tw.CA, tw.CB
    return out''', "advance")
    add('''    return _Tower(None if tw.A_D is None else tw.A_D * col,
                  tw.A_L * col, tw.M * col, tw.w2b, tw.k,
                  None if tw.Z is None else tw.Z * col,
                  tw.L, tw.sb, tw.eb)''',
        '''    out = _Tower(None if tw.A_D is None else tw.A_D * col,
                 tw.A_L * col, tw.M * col, tw.w2b, tw.k,
                 None if tw.Z is None else tw.Z * col,
                 tw.L, tw.sb, tw.eb)
    if tw.Y is not None:
        # left scale on every mode of Sym3(y (x) S): y, and both legs of S
        out.Y, out.CA, out.CB = tw.Y * col, tw.CA, tw.CB
    return out''', "wick_tower")
    add('''        MP, _PP, _PPL = _p2_recon(tw)
        return (MP * tw.M).sum(axis=1) + 3.0 * (tw.A_L * tw.M * C_L).sum(axis=1)''',
        '''        MP, _PP, _PPL = _p2_recon(tw)
        d3 = (MP * tw.M).sum(axis=1) + 3.0 * (tw.A_L * tw.M * C_L).sum(axis=1)
        if tw.Y is not None:
            # D3 += 2 sum_h y_h . R_h
            _G, RY = _hub_parts(tw)
            d3 = d3 + (RY * tw.Y).sum(axis=1) * 2.0
        return d3''', "dslice3")
    add('''        LP = (tw.A_L * tw.A_L) * tw.w2b + PP * (tw.sb / 3.0) + (2.0 / 3.0) * MP
        acc = _mm(LA, tw.A_L.T) + _mm(LP, tw.M.T)''',
        '''        LP = (tw.A_L * tw.A_L) * tw.w2b + PP * (tw.sb / 3.0) + (2.0 / 3.0) * MP
        if tw.Y is not None:
            # the hubs FOLD INTO the two existing contractions -- no new dense
            # product: (2/3)(y.A_L.c) M^T joins LP, (2/3)(y.M.c) A_L^T joins LA
            G, RY = _hub_parts(tw)
            G23 = G * (2.0 / 3.0)
            LA = LA + G23 * tw.M
            LP = LP + G23 * tw.A_L
        acc = _mm(LA, tw.A_L.T) + _mm(LP, tw.M.T)
        if tw.Y is not None:
            acc = acc + _mm(RY * (2.0 / 3.0), tw.Y.T)''', "dslice21")
    add('''    if tw is None:
        return _Tower(A_D, A_L, None, w2b, 1, Zb,
                      None if Lb is None else Lb[None], sb, eb)
    return _Tower(None if A_D is None else fnp.concatenate([tw.A_D, A_D], axis=1),
                  fnp.concatenate([tw.A_L, A_L], axis=1),
                  tw.M, fnp.concatenate([tw.w2b, w2b]), tw.k + 1,
                  None if Zb is None else fnp.concatenate([tw.Z, Zb], axis=1),
                  None if Lb is None else fnp.concatenate([tw.L, Lb[None]], axis=0),
                  None if sb is None else fnp.concatenate([tw.sb, sb]),
                  None if eb is None else fnp.concatenate([tw.eb, eb]))''',
        '''    if tw is None:
        out = _Tower(A_D, A_L, None, w2b, 1, Zb,
                     None if Lb is None else Lb[None], sb, eb)
        if Y is not None:
            out.Y, out.CA, out.CB = Y, CA, CB
        return out
    out = _Tower(None if A_D is None else fnp.concatenate([tw.A_D, A_D], axis=1),
                 fnp.concatenate([tw.A_L, A_L], axis=1),
                 tw.M, fnp.concatenate([tw.w2b, w2b]), tw.k + 1,
                 None if Zb is None else fnp.concatenate([tw.Z, Zb], axis=1),
                 None if Lb is None else fnp.concatenate([tw.L, Lb[None]], axis=0),
                 None if sb is None else fnp.concatenate([tw.sb, sb]),
                 None if eb is None else fnp.concatenate([tw.eb, eb]))
    if Y is not None:
        out.Y = fnp.concatenate([tw.Y, Y], axis=1)
        out.CA = fnp.concatenate([tw.CA, CA])
        out.CB = fnp.concatenate([tw.CB, CB])
    return out''', "append")
    add('''def _append(tw, A_D, A_L, w2b, n, Zb=None, Lb=None, sb=None, eb=None):''',
        '''def _append(tw, A_D, A_L, w2b, n, Zb=None, Lb=None, sb=None, eb=None,
            Y=None, CA=None, CB=None):''', "append-sig")
    add('''    if tw.A_D is None:
        r = tw.L.shape[2]
        return _Tower(None, tw.A_L[:, -keep:], M, tw.w2b[-keep:], g,
                      tw.Z[:, -(g * r):], tw.L[-g:],
                      tw.sb[-keep:], tw.eb[-keep:])''',
        '''    if tw.A_D is None:
        r = tw.L.shape[2]
        out = _Tower(None, tw.A_L[:, -keep:], M, tw.w2b[-keep:], g,
                     tw.Z[:, -(g * r):], tw.L[-g:],
                     tw.sb[-keep:], tw.eb[-keep:])
        if tw.Y is not None:
            out.Y, out.CA, out.CB = tw.Y[:, -(2 * g):], tw.CA[-keep:], tw.CB[-keep:]
        return out''', "truncate")

    add('''        e_b = w[(1, 2)] - 2.0 * pK[(1,)] * w1
        S_sep = (e_b[:, None] * C11) * w1[None, :]
        Zb, Lb = _p2_compress(S21 - S_sep)
        out_tw = _append(adv, None, fac1, w2b_vec, n, Zb, Lb, S3, e_b)''',
        '''        e_b = w[(1, 2)] - 2.0 * pK[(1,)] * w1
        S_sep = (e_b[:, None] * C11) * w1[None, :]
        if _HUB_ROUTES:
            # STAGE 1 -- the distinct-index kappa4 birth. See the generator's
            # header for the derivation and `local_sim/k4feed_oracle.py` for
            # the exact reference it was checked against.
            _li_h = layer_index - 1
            _lam_h = float(_K4_LAM[_li_h]) if 0 <= _li_h < len(_K4_LAM) else 0.0
            _hm = metric / 8.0
            w3 = w[(3, 1)]
            yA = w2
            if "i" in _HUB_ROUTES:
                cA = w1 * (_hm * _lam_h)
            else:
                cA = w1 * 0.0
            if "ii" in _HUB_ROUTES:
                cA = cA + (dG * w3) * _hm
                yB = dG * w2
                cB = w3 * _hm
            else:
                yB = w2 * 0.0
                cB = w3 * 0.0
            resid = S21 - S_sep
            if _HUB_COMPENSATE:
                # the hubs' repeated content is already in the table's slices:
                # d(y c) C d(w1) moves e_b for free, d(y w1) C d(c) joins the residual
                _t = 2.0 / 3.0
                resid = (resid - ((yA * w1)[:, None] * C11) * (cA * _t)[None, :]
                         - ((yB * w1)[:, None] * C11) * (cB * _t)[None, :])
                e_b = e_b - (yA * cA + yB * cB) * _t
            Zb, Lb = _p2_compress(resid)
            out_tw = _append(adv, None, fac1, w2b_vec, n, Zb, Lb, S3, e_b,
                             fnp.stack([yA, yB], axis=1), cA, cB)
        else:
            Zb, Lb = _p2_compress(S21 - S_sep)
            out_tw = _append(adv, None, fac1, w2b_vec, n, Zb, Lb, S3, e_b)''', "birth")

    add('''            thin += 4 * n * r * (2 * n - 1) + 4 * n * r * r + 2 * n * n
    return base + thin''',
        '''            thin += 4 * n * r * (2 * n - 1) + 4 * n * r * r + 2 * n * n
        if _HUB_ROUTES and k:
            # STAGE 1 hubs, over-charged on purpose (this is the budget guard):
            # Y transport, the (n,2k)@(2k,n) contraction, and ~16 k n^2 of
            # elementwise G / R / LA / LP / D3 work.
            thin += n * (2 * k) * (2 * n - 1) + n * n * (4 * k - 1) + 16 * k * n * n
        if _HUB_ROUTES and l < depth - 1:
            thin += 12 * n * n                     # birth-time hub construction
    return base + thin''', "cost-model")
    return P


def build(routes, compensate, rank, title, src, sha):
    body = src
    for old, new, why, count in _patches(routes, compensate):
        found = body.count(old)
        if found != count:
            raise SystemExit("PATCH ANCHOR COUNT WRONG (%s): expected %d, found %d.\n%s"
                             % (why, count, found, old[:300]))
        body = body.replace(old, new, count)
    anchor = "_P2_RANK = 32\n"
    if body.count(anchor) != 1:
        raise SystemExit("rank anchor not unique")
    body = body.replace(anchor, "_P2_RANK = %d\n\n"
                        "#: STAGE 1 distinct-index kappa4 feed routes; () is v31 exactly.\n"
                        "_HUB_ROUTES = %r\n"
                        "#: Subtract the hubs' repeated content already carried by the table.\n"
                        "_HUB_COMPENSATE = %r\n" % (rank, tuple(routes), bool(compensate)), 1)
    fa = "def _dslice3(tw):"
    if body.count(fa) != 1:
        raise SystemExit("cannot place _hub_parts")
    body = body.replace(fa, _HUB_FUNCS.lstrip("\n") + fa, 1)
    i = body.find('"""')
    j = body.find('"""', i + 3)
    v31doc = body[i + 3:j]
    head = ('"""%s\n\nGENERATED by `local_sim/s1_build_v32.py` from `estimator_k3_v31.py`\n'
            '(sha256 %s). DO NOT EDIT; edit the generator and re-emit.\n'
            'Contract and oracle: see the generator header. Not approved for submission.\n\n'
            '--------------------------------------------------------------------------\n'
            'INHERITED FROM v31\n\n%s\n"""' % (title, sha, v31doc.strip()))
    return head + body[j + 3:]


def emit(out_dir=_ROOT, arms=ARMS):
    with open(SOURCE, "rb") as fh:
        raw = fh.read()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != V31_SHA:
        raise SystemExit("estimator_k3_v31.py is not the frozen graded v31 (%s)" % sha)
    src = raw.decode("utf-8")
    paths = []
    for name, routes, comp, rank, title in arms:
        text = build(routes, comp, rank, title, src, sha)
        path = os.path.join(out_dir, name)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        paths.append(path)
        print("  wrote %-28s routes=%-12s compensate=%-5s rank=%d sha256 %s"
              % (name, routes, comp, rank,
                 hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]))
    return paths


if __name__ == "__main__":
    if "--parity" in sys.argv:
        sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]
        import s1_parity_v32
        raise SystemExit(s1_parity_v32.main())
    emit()
