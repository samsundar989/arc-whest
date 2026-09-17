"""Stage 3 (handoff 2026-09-17b): the shared-coordinate basis chosen by the CONSUMED QUERY.

Generated from an already-emitted lean shared-coordinate arm by asserted
patches. **Only basis selection changes.** Age, rank, power passes, the
initialiser, the recent-lineage exactness, P4, Strassen, the closure, lambda and
every other algorithm choice are untouched, so a paired comparison against the
same chain isolates the objective and nothing else.

THE DEFECT BEING REPAIRED
-------------------------
The live join selects its basis from the range of `J diag(dJ) J^T`, where `dJ`
holds the joiner's column energies in this layer's `(2,1)` contraction. Write
the frozen left factors as `U = [LA | LP]` and the legs as `J = [A | M]`. The
thing the shared basis must preserve is the CONSUMED contraction

    E = U J^T = LA A^T + LP M^T,

and the exact frozen-query error of an orthogonal projector `P = Q Q^T` acting
on the legs is

    ||E (I - P)||_F^2 = tr( (I - P) J U^T U J^T (I - P) ).

The live objective is that expression with `U^T U` replaced by its DIAGONAL. It
therefore drops every cross-column term and every A/M interaction. It is exact
only when U's columns are orthogonal and equal-normed, and it is not an upper
bound in general: if both U columns equal `u` and J's columns are `e1`, `e2`,
the true query is rank one along `e1 + e2` while the surrogate is isotropic and
cannot see the direction at all. The weights are also captured BEFORE the Wick
column scale while the join happens after, and the older lineages' weights come
from their own earlier joins.

WHAT REPLACES IT
----------------
A randomized range finder for the RIGHT singular space of the query itself:

    Q = orth( E_oj^T Omega ),
    E_oj = X_old Q_old^T + U_j J_j^T          (never formed),
    E_oj^T Omega = Q_old (X_old^T Omega) + J_j (U_j^T Omega).

`E_oj` is the OLD-plus-JOINING contribution at ONE consistent temporal boundary.
The recent lineages we deliberately keep exact are NOT in it: they are never
projected, so including them would tune the basis to content that does not pass
through it. `X_old` is `[LA_old | LP_old] @ [FA | FM]^T`, which
`_shared_contract_lean` ALREADY forms as the inner half of its old-leg product,
so it is stashed rather than recomputed.

COST, COUNTED HONESTLY
----------------------
Per power pass, wide products at width n and rank r:

    live objective   J^T Om + J(...)                                 = 4 n^2 r
    query objective  Z = U (J^T Om) + X_old (Q^T Om)                 = 4 n^2 r + 2 n r^2
                     Y = J (U^T Z) + Q (X_old^T Z)                   = 4 n^2 r + 2 n r^2

The query objective runs ONE POWER ITERATION of `E^T E`, because the live
objective already is one -- `J D J^T Om` is `(J sqrt D)(J sqrt D)^T Om` -- and
the sketch `Om` is deterministic in both (norm-selected columns; hard rule 2
forbids sampling and equivariance forbids a fixed weight slice). Applying `E^T`
once instead would have compared two different ALGORITHMS and attributed the
difference to the objective. Check (c) of `s10_parity.py` measured that mistake
before any truth was read: single application, rank-2 query error 0.2174 against
the surrogate's 0.0622.

Against that, the query objective DROPS the live one's carried Gram `Sg`
entirely: `Sj = (FJ d) FJ^T` at `2 n r^2`, `T Sg T^T` at `2 r^3`, and
`Q (Sg (Q^T Om))` at `2 n r^2 + r^3` inside the loop. The carried Gram exists
only so the diagonal surrogate can remember older lineages' weights; the query
objective remembers them exactly through `X_old`.

Net, at n = 1024 and r = 256, about `+4 n^2 r - 4 n r^2 - 3 r^3` per join,
roughly `+7.5e8` counted FLOPs, times about eleven joins at depth 16 -- on the
order of `+1.7%` of the base's `C`. That is an ARGUMENT. `s2_cost_meter.py` on
the emitted file is the number of record, and the screen is on the adjusted
score, which pays for it.

THE D3 PROTECTION ARM (`qEd`), PREREGISTERED
--------------------------------------------
`E` is the `(2,1)` query. It is not the only consumer: `_dslice3` reads the legs
ELEMENTWISE,

    d3 += 3 * (A_L . M . C_L).sum(axis=1),      C_L = A_L d(w2b),

which a projection of the legs also perturbs and which no right-singular
subspace of `E` is chosen to protect. `qEd` reserves `_QB_D3` of the `r`
columns for the range of the joiner's own D3 summand `G_j = A_j . M_j . C_j`,
so the total rank is UNCHANGED and the comparison stays fixed-rank. Declared in
the contract before any measurement; it is not added after an unfavourable
result, and it is not a second free parameter -- `_QB_D3` is fixed at 16 here.

Run:  uv run python local_sim/s10_query_basis.py --base estimator_k3_v37_a4r256_p4d5.py --emit
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]

# --------------------------------------------------------------------------- #
# the anchors, asserted by exact count
# --------------------------------------------------------------------------- #

_CONTRACT_OLD = '''    FF = fnp.concatenate([tw.FA, tw.FM], axis=1)
    old = _mm(_mm(fnp.concatenate([LA[:, :c], LP[:, :c]], axis=1), FF.T), tw.Q.T)
    return old + rec'''

_CONTRACT_NEW = '''    FF = fnp.concatenate([tw.FA, tw.FM], axis=1)
    # QUERY BASIS: `X_old` is this product's inner half -- the OLD lineages'
    # actual contribution to the consumed contraction in the current layer's
    # coordinates. It is stashed, not recomputed: the join reads it instead of
    # a carried Gram of stale weights.
    Xold = _mm(fnp.concatenate([LA[:, :c], LP[:, :c]], axis=1), FF.T)
    tw._qold = Xold
    old = _mm(Xold, tw.Q.T)
    return old + rec'''

_WTS_OLD = '''    if _SHARED_ON and _shared_rank(n) and (tw.k - tw.ko) >= _SHARED_AGE:
        tw._wts = ((LA[:, c:c + n] * LA[:, c:c + n]).sum(axis=0),
                   (LP[:, c:c + n] * LP[:, c:c + n]).sum(axis=0))
    rec = _mm(fnp.concatenate([LA[:, c:], LP[:, c:]], axis=1),'''

_WTS_NEW = '''    if _SHARED_ON and _shared_rank(n) and (tw.k - tw.ko) >= _SHARED_AGE:
        tw._wts = ((LA[:, c:c + n] * LA[:, c:c + n]).sum(axis=0),
                   (LP[:, c:c + n] * LP[:, c:c + n]).sum(axis=0))
        # QUERY BASIS: the joiner's own frozen left factors, kept as VIEWS of
        # arrays this layer already holds. No product, no copy.
        tw._qU = (LA[:, c:c + n], LP[:, c:c + n])
    rec = _mm(fnp.concatenate([LA[:, c:], LP[:, c:]], axis=1),'''

_JOIN_OLD = '''    for _p in range(int(_SHARED_PASSES)):
        Y = _mm(J, (J.T @ Om) * dJ[:, None])
        if tw.ko:
            Y = Y + tw.Q @ (tw.Sg @ (tw.Q.T @ Om))
        Om, _ = fnp.linalg.qr(Y)
    Qn = Om
    FJ = Qn.T @ J
    Sj = (FJ * dJ) @ FJ.T
    if tw.ko:
        T = Qn.T @ tw.Q
        c = tw.FA.shape[1]
        FF = T @ fnp.concatenate([tw.FA, tw.FM], axis=1)
        FA = fnp.concatenate([FF[:, :c], FJ[:, :n]], axis=1)
        FM = fnp.concatenate([FF[:, c:], FJ[:, n:]], axis=1)
        Sg = (T @ tw.Sg) @ T.T + Sj
    else:
        FA, FM, Sg = FJ[:, :n], FJ[:, n:], Sj'''

_JOIN_NEW = '''    if _QUERY_BASIS:
        # ---- the CONSUMED-QUERY range finder ---------------------------- #
        # ONE power iteration of the query's own Gram,
        #     Y = E^T (E Om),   E = X_old Q_old^T + U_j J_j^T,
        # with E never formed. A SINGLE application `E^T Om` is not enough and
        # this is not a tuning choice: the sketch `Om` is deterministic (the
        # norm-selected columns, unchanged from the base, because hard rule 2
        # forbids sampling and equivariance forbids a fixed weight slice), and a
        # deterministic sketch applied once has no singular-value decay working
        # for it. The live Gram objective is ALREADY a power iteration --
        # `J D J^T Om` is `(J sqrt D)(J sqrt D)^T Om` -- so a single application
        # here would have compared two different algorithms and called the
        # difference an objective. Check (c) of `s10_parity.py` measured exactly
        # that and refused the first spelling: rank-2 query error 0.2174 against
        # the surrogate's 0.0622. Priced in the header; metered on the file.
        UA, UP = tw._qU
        U = fnp.concatenate([UA, UP], axis=1)
        rq = r - int(_QB_D3) if _QB_D3 else r
        for _p in range(int(_SHARED_PASSES)):
            Z = _mm(U, _mm(J.T, Om))
            if tw.ko and tw._qold is not None:
                Z = Z + _mm(tw._qold, _mm(tw.Q.T, Om))
            Y = _mm(J, _mm(U.T, Z))
            if tw.ko and tw._qold is not None:
                Y = Y + _mm(tw.Q, _mm(tw._qold.T, Z))
            if _QB_D3:
                # PREREGISTERED D3 protection: `_dslice3` reads the legs
                # elementwise, so `rq` columns go to the (2,1) query's range
                # and `_QB_D3` to the range of the joiner's own D3 summand.
                # The TOTAL rank is unchanged, so the compare stays fixed-rank.
                G = (tw.A_L[:, :n] * tw.M[:, :n]) * tw.w2b[:n]
                Yd = _mm(G, _mm(G.T, Om[:, :int(_QB_D3)]))
                Y = fnp.concatenate([Y[:, :rq], Yd], axis=1)
            Om, _ = fnp.linalg.qr(Y)
        Qn = Om
        FJ = Qn.T @ J
        Sj = None
    else:
        for _p in range(int(_SHARED_PASSES)):
            Y = _mm(J, (J.T @ Om) * dJ[:, None])
            if tw.ko:
                Y = Y + tw.Q @ (tw.Sg @ (tw.Q.T @ Om))
            Om, _ = fnp.linalg.qr(Y)
        Qn = Om
        FJ = Qn.T @ J
        Sj = (FJ * dJ) @ FJ.T
    if tw.ko:
        T = Qn.T @ tw.Q
        c = tw.FA.shape[1]
        FF = T @ fnp.concatenate([tw.FA, tw.FM], axis=1)
        FA = fnp.concatenate([FF[:, :c], FJ[:, :n]], axis=1)
        FM = fnp.concatenate([FF[:, c:], FJ[:, n:]], axis=1)
        # The carried Gram exists ONLY for the diagonal surrogate's memory of
        # older lineages. The query basis remembers them exactly through
        # `X_old`, so it is not formed and not carried.
        Sg = None if _QUERY_BASIS else (T @ tw.Sg) @ T.T + Sj
    else:
        FA, FM, Sg = FJ[:, :n], FJ[:, n:], Sj'''

_JOIN_GUARD_OLD = '''    if not r or (tw.k - tw.ko) < _SHARED_AGE or tw._wts is None:
        return tw
    J = fnp.concatenate([tw.A_L[:, :n], tw.M[:, :n]], axis=1)'''

_JOIN_GUARD_NEW = '''    if not r or (tw.k - tw.ko) < _SHARED_AGE or tw._wts is None:
        return tw
    if _QUERY_BASIS and tw._qU is None:
        return tw
    J = fnp.concatenate([tw.A_L[:, :n], tw.M[:, :n]], axis=1)'''

_SLOTS_OLD = '''                 "Q", "FA", "FM", "ko", "Sg", "_legs", "_wts")'''
_SLOTS_NEW = '''                 "Q", "FA", "FM", "ko", "Sg", "_legs", "_wts",
                 # STAGE 3: the consumed-query basis's two stashes
                 "_qU", "_qold")'''

_INIT_OLD = '''        self._legs = self._wts = None'''
_INIT_NEW = '''        self._legs = self._wts = None
        self._qU = self._qold = None'''

_FLAGS_OLD = '''_SHARED_WEIGHTS = "cons"'''
_FLAGS_NEW = '''_SHARED_WEIGHTS = "cons"
#: STAGE 3 (2026-09-17b): choose the shared basis from the CONSUMED QUERY's right
#: singular space instead of the weighted Gram's diagonal surrogate. `_QB_D3` is
#: the number of the `r` columns reserved for the D3 elementwise consumer; 0
#: spends every column on the (2,1) query. Both values are preregistered.
_QUERY_BASIS = %r
_QB_D3 = %d'''


def _patch(body, old, new, why, count=1):
    found = body.count(old)
    if found != count:
        raise SystemExit("PATCH ANCHOR COUNT WRONG (%s): expected %d, found %d\n%s"
                         % (why, count, found, old[:240]))
    return body.replace(old, new)


def _carry_stashes(body):
    """`_carry_shared` must forward the two stashes, or a transported tower loses them."""
    i = body.index("def _carry_shared(")
    j = body.find("\ndef ", i + 1)
    seg = body[i:j]
    anchor = "    return dst"
    if seg.count(anchor) != 1:
        raise SystemExit("_carry_shared has no single `return dst`")
    seg = seg.replace(anchor, "    dst._qU, dst._qold = src._qU, src._qold\n"
                              "    return dst")
    return body[:i] + seg + body[j:]


def build(src, query_basis=True, d3=0):
    b = src
    b = _patch(b, _SLOTS_OLD, _SLOTS_NEW, "slots")
    b = _patch(b, _INIT_OLD, _INIT_NEW, "init")
    b = _patch(b, _FLAGS_OLD, _FLAGS_NEW % (bool(query_basis), int(d3)), "flags")
    b = _patch(b, _WTS_OLD, _WTS_NEW, "stash-U")
    b = _patch(b, _CONTRACT_OLD, _CONTRACT_NEW, "stash-Xold")
    b = _patch(b, _JOIN_GUARD_OLD, _JOIN_GUARD_NEW, "join-guard")
    b = _patch(b, _JOIN_OLD, _JOIN_NEW, "join")
    b = _carry_stashes(b)
    return b


ARMS = (
    # tag, query_basis, d3-reserved columns
    ("qoff", False, 0),   # PARITY: must equal the base bit for bit and in FLOPs
    ("qE", True, 0),      # the consumed-query basis
    ("qEd", True, 16),    # + the preregistered D3 protection component
)


def emit(base, out_dir=_ROOT, arms=ARMS):
    path = os.path.join(_ROOT, base)
    raw = open(path, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    src = raw.decode("utf-8")
    stem = os.path.splitext(base)[0].replace("estimator_k3_", "")
    outs = []
    for tag, qb, d3 in arms:
        text = build(src, qb, d3)
        i = text.find('"""')
        head = ('"""estimator_k3_v39_%s_%s.py: STAGE 3 consumed-query basis '
                '(query_basis=%s, d3_reserved=%d).\n\n'
                'GENERATED by `local_sim/s10_query_basis.py` from `%s`\n'
                '(sha256 %s). DO NOT EDIT. Not approved for submission.\n'
                'ONLY basis selection differs from the base: age, rank, passes,\n'
                'initialiser, recent-lineage exactness, P4, closure and lambda are\n'
                'identical, which is what makes the paired compare isolate the\n'
                'objective.\n\n' % (stem, tag, qb, d3, base, sha))
        text = head + text[i + 3:]
        name = "estimator_k3_v39_%s_%s.py" % (stem, tag)
        p = os.path.join(out_dir, name)
        with open(p, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        outs.append(p)
        print("  wrote %-46s sha256 %s"
              % (name, hashlib.sha256(text.encode()).hexdigest()[:16]))
    return outs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="estimator_k3_v37_a4r256_p4d5.py")
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--out-dir", default=_ROOT)
    a = ap.parse_args()
    if a.emit:
        emit(a.base, a.out_dir)
