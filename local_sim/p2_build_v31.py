"""Generate `estimator_k3_v31.py` from v30 — P2, the 7→4 separable leg.

WHY A GENERATOR. Same reason as `p2_build_v30.py`: v30 is ~1,150 lines and a
hand-edited copy drifts from its control. Every difference between candidate
and control is one asserted-count patch in this file, so the two files cannot
disagree about anything except the mechanism.

WHY IT READS v30 AND NOT F21. v30's patches touch `_layer`'s κ4 channel, the
birth's fourth-order lines and the tail. P2's patches touch `_Tower`,
`_advance`, `_wick_tower`, `_dslice3`, `_dslice21`, `_append`, `_truncate`, the
birth's *third*-order line and the cost model. The two sets are DISJOINT
(verified by diffing v30 against F21), so stacking them is layering, not
merging, and v31 is P1 **and** P2.

--------------------------------------------------------------------------
THE MECHANISM

F21 carries a dense `A_D` per lineage and transports it. `A_D` is born as

    A_D = diag(S3) + 3 S21^T ,   S21 = S_sep + R_res
    S_sep = zero_diag( e_i C_off[i,j] w1_j ),  e = w(1,2) - 2 pK1 w(1,1)

and the middle piece is FREE, because `a_b = fac1 = w1[:,None] * C11` is a leg
the tower already carries and `3 S_sep^T = 3 a_b d(e)` is a column scale on it.
`diag(S3)` is likewise free: at birth `M == I`, so `diag(S3) = M d(S3)`, and
because every later operation applies the same matrix to every mode, that
stays true forever. Only the residual needs a leg of its own, and it is
numerically low-rank. So

    A_D  =  M d(s)  +  3 A_L d(e)  +  Z L^T          (`_p2_recon`)

with `s = S3`, `e` as above, `Z` an `(n, r)` transported leg and `L` an
`(n, r)` STATIC basis. `A_D` is never materialised again.

Both halves of this were proved before any estimator code was written, in
`local_sim/p2_p2_identity_test.py` (11/11):

  - the reconstruction, to 1.9e-16 on float64 with an ARBITRARY residual and
    3.7 float32 ulps on real F21 state at its own birth site;
  - the `(2,1)` REGROUPING, to 6.8e-16 — substituting the reconstruction on the
    right of F21's contractions sends two of its three pieces into groups that
    already exist:

        D21 = LA A_L^T + LP M^T + [(M.M) L] Z^T / 3
        LA  = 2 A_L.M.w2 + M.M.e
        LP  = A_L.A_L.w2 + M.M.(s/3) + (2/3) A_D.M

THE COUNT, read off the source rather than off the plan. Per layer, the
products that scale with `k`:

    F21   transports  A_D, A_L, M          = 3        (`_advance`)
          contractions  3 `_mm` calls      = 3        (`_dslice21`)
                                             ---
                                               6
    P2    transports  A_L, M               = 2
          contractions  LA A_L^T, LP M^T   = 2
                                             ---
                                               4      + O(k n^2 r) thin

so **6 → 4**, a 32.3% saving on the tower, which is 92.6% of F21's bill.

NOTE A DISCREPANCY WE ARE NOT FIXING. F21's own `_matmuls` cost model bills an
aged lineage `7`, not the `6` the code issues — it predates the fusion in
`_dslice21` where `{0,1}|{2}` and `{0,2}|{1}` turned out to be identically the
same product. That over-pricing is why "7→4" was read as a term count twice
before it was read off the code. `_matmuls` is used ONLY to choose `gens`, and
at the live shape `_fit_generations` chooses `None` either way (asserted in
`--parity`), so correcting it would change no behaviour and would break the
control's bit-identity with v30. The candidate's own branch is set to the true
`4`/`3`; the control's `7`/`6` is left exactly as it stands.

--------------------------------------------------------------------------
THE RANK, AND WHY NO RANDOMNESS ENTERS

The compression is a randomized range finder with one power iteration — but the
sketch is `W[:, :r]`, the first `r` columns of THIS layer's weight matrix. At
He init that is an i.i.d. Gaussian matrix, which is exactly what a random
sketch would be, and it makes the estimator **deterministic by construction**:
hard rule 2 is satisfied without seeding anything, because nothing is sampled.
(Read off `external/504aldo/estimators/estimator_v16b.py:561`, logged in
`docs/PROVENANCE_504aldo.md`. Our own design note had assumed this would need
`mlp.seed`; it does not.)

`_P2_RANK = 0` means FULL rank: the sketch is the whole of `W`, the range
finder is exact, and the reconstruction reproduces the dense `A_D` to float32.
That is the algebraic-parity control — it is *more* expensive than v30, and it
exists to separate "did the regrouping break anything" from "does truncation
cost accuracy".

Run:
    uv run python local_sim/p2_build_v31.py --emit
    uv run python local_sim/p2_build_v31.py --parity
"""

from __future__ import annotations

import argparse
import hashlib
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
SOURCE = os.path.join(_ROOT, "estimator_k3_v30.py")
TARGET = os.path.join(_ROOT, "estimator_k3_v31.py")
CONTROL = os.path.join(_ROOT, "estimator_k3_v31_control.py")
DENSE = os.path.join(_ROOT, "estimator_k3_v31_dense.py")

DEFAULT_RANK = 32


# ==========================================================================
# the new module-level machinery, inserted ahead of `_dslice3`

_P2_FUNCS = '''
def _p2_compress(R, W):
    """`3 R^T ~= Z L^T` by a randomized range finder with one power iteration.

    THE SKETCH IS NOT RANDOM. `W[:, :r]` is the first `r` columns of this
    layer's weight matrix, which at He init is an i.i.d. Gaussian matrix --
    statistically what a random sketch would be, and deterministic, so hard
    rule 2 is satisfied by construction and nothing is ever seeded.

    `_P2_RANK <= 0` takes the full width, which makes the range finder exact
    and the reconstruction a float32 copy of the dense `A_D`.

    Returns `(Z, L)` with the factor 3 already folded into `Z`, so the
    consumers never carry it.
    """
    n = R.shape[0]
    r = n if _P2_RANK <= 0 else min(int(_P2_RANK), n)
    Om = W[:, :r]
    if Om.dtype != R.dtype:
        Om = Om.astype(R.dtype)
    Q, _ = fnp.linalg.qr(R @ Om)
    Q, _ = fnp.linalg.qr(R @ (R.T @ Q))
    return (Q.T @ R).T * 3.0, Q


def _p2_recon(tw):
    """The shared per-layer reconstruction: `(A_D.M, M.M, (M.M)L, r)`.

    `A_D` is NOT materialised. What both slices actually read is `A_D . M`, so
    only that product is formed:

        A_D . M = M.M.s + 3 A_L.M.e + Zres.M ,   Zres = Z L^T

    `Zres` is the one dense `(n, k n)` array P2 adds, at `O(k n^2 r)` -- and P2
    removes three of F21's: the `_advance` output for `A_D`, `A_D . M`, and the
    `M.M @ A_D^T` product. Cached on the tower because `_dslice3` and
    `_dslice21` are both called on the same object in one layer.

    The `(n, k n) <-> (k, n, n)` shuttles are the price of F21's WIDE layout
    against the per-lineage blocking the thin terms need. They are memory
    traffic, not FLOPs, and they are what the residual measurement has to
    clear.
    """
    if tw._p2 is not None:
        return tw._p2
    n, k = tw.M.shape[0], tw.k
    r = tw.L.shape[2]
    # (n, k r) -> (k, n, r): SMALL, `k n r` not `k n^2`.
    Z3 = fnp.reshape(tw.Z, (n, k, r)).transpose(1, 0, 2)
    Zres = fnp.reshape((Z3 @ tw.L.transpose(0, 2, 1)).transpose(1, 0, 2),
                       (n, k * n))
    PP = tw.M * tw.M
    MP = PP * tw.sb + (tw.A_L * tw.M) * (3.0 * tw.eb) + Zres * tw.M
    PPL = fnp.reshape(PP, (n, k, n)).transpose(1, 0, 2) @ tw.L
    tw._p2 = (MP, PP, fnp.reshape(PPL.transpose(1, 0, 2), (n, k * r)))
    return tw._p2


'''


def _patches(separable: bool):
    """`(anchor, replacement, why, expected_count)`; count is ASSERTED."""
    P = []

    # ---- 1. the tower carries the factored leg ---------------------------
    P.append((
        '''    __slots__ = ("A_D", "A_L", "M", "w2b", "k")

    def __init__(self, A_D, A_L, M, w2b, k):
        self.A_D, self.A_L, self.M, self.w2b, self.k = A_D, A_L, M, w2b, k''',
        '''    __slots__ = ("A_D", "A_L", "M", "w2b", "k",
                 # P2: the FACTORED leg. When `_P2_SEPARABLE`, `A_D` is None
                 # and these carry it exactly: `A_D = M d(sb) + 3 A_L d(eb)
                 # + Z L^T`. `Z` is `(n, k r)` and transported; `L` is
                 # `(k, n, r)` and STATIC; `sb`/`eb` are `(k n,)` column
                 # scales. `_p2` caches the per-layer reconstruction.
                 "Z", "L", "sb", "eb", "_p2")

    def __init__(self, A_D, A_L, M, w2b, k,
                 Z=None, L=None, sb=None, eb=None):
        self.A_D, self.A_L, self.M, self.w2b, self.k = A_D, A_L, M, w2b, k
        self.Z, self.L, self.sb, self.eb = Z, L, sb, eb
        self._p2 = None''',
        "the tower gains the factored leg", 1))

    # ---- 2. transport: the dense `A_D` matmul goes away ------------------
    P.append((
        '''    A_D = _mm(WT, tw.A_D)
    A_L = _mm(WT, tw.A_L)
    M = WT if tw.M is None else fnp.concatenate([_mm(WT, tw.M), WT], axis=1)
    return _Tower(A_D, A_L, M, tw.w2b, tw.k)''',
        '''    A_D = None if tw.A_D is None else _mm(WT, tw.A_D)
    A_L = _mm(WT, tw.A_L)
    M = WT if tw.M is None else fnp.concatenate([_mm(WT, tw.M), WT], axis=1)
    # P2: `Z` is `(n, k r)`, so this is thin -- and it is the ONLY thing the
    # residual needs, because `L`, `sb` and `eb` are static under transport.
    # `A_D = M d(sb) + 3 A_L d(eb) + Z L^T` maps to
    # `W^T A_D = (W^T M) d(sb) + 3 (W^T A_L) d(eb) + (W^T Z) L^T`, which is the
    # same identity in the transported legs. That is why the leg survives.
    Z = None if tw.Z is None else _mm(WT, tw.Z)
    return _Tower(A_D, A_L, M, tw.w2b, tw.k, Z, tw.L, tw.sb, tw.eb)''',
        "transport drops A_D and gains a thin Z", 1))

    # ---- 3. the Wick column scale reaches Z too --------------------------
    P.append((
        '''    return _Tower(tw.A_D * col, tw.A_L * col, tw.M * col, tw.w2b, tw.k)''',
        '''    # A left column scale distributes over the factored leg exactly as it does
    # over the dense one: `(A_D)*col = (M*col) d(sb) + 3 (A_L*col) d(eb)
    # + (Z*col) L^T`. `sb`/`eb` are RIGHT diagonals and are untouched.
    return _Tower(None if tw.A_D is None else tw.A_D * col,
                  tw.A_L * col, tw.M * col, tw.w2b, tw.k,
                  None if tw.Z is None else tw.Z * col,
                  tw.L, tw.sb, tw.eb)''',
        "the Wick scale reaches Z", 1))

    # ---- 4. the (3,) slice reads the reconstruction ----------------------
    P.append((
        '''    C_L = tw.A_L * tw.w2b
    t = (tw.A_D * tw.M * tw.M).sum(axis=1)
    return t + 3.0 * (tw.A_L * tw.M * C_L).sum(axis=1)''',
        '''    C_L = tw.A_L * tw.w2b
    if tw.A_D is None:
        # `A_D . M` is already formed and shared; the second term is F21's own,
        # unchanged, because `C_L = A_L d(w2)` never involved `A_D`.
        MP, _PP, _PPL = _p2_recon(tw)
        return (MP * tw.M).sum(axis=1) + 3.0 * (tw.A_L * tw.M * C_L).sum(axis=1)
    t = (tw.A_D * tw.M * tw.M).sum(axis=1)
    return t + 3.0 * (tw.A_L * tw.M * C_L).sum(axis=1)''',
        "the (3,) slice reads the reconstruction", 1))

    # ---- 5. the (2,1) slice: four dense contractions become two ----------
    P.append((
        '''    C_L = tw.A_L * tw.w2b
    acc = _mm(2.0 * (tw.A_D * tw.M) + 3.0 * (tw.A_L * C_L), tw.M.T)''',
        '''    C_L = tw.A_L * tw.w2b
    if tw.A_D is None:
        # THE REGROUPING, proved in `local_sim/p2_p2_identity_test.py` to
        # 6.8e-16 with an arbitrary residual. Substituting
        # `A_D = M d(sb) + 3 A_L d(eb) + Z L^T` on the RIGHT of F21's three
        # contractions sends two of its pieces into groups that already exist,
        # leaving right factors `A_L`, `M` and a THIN `Z`. The `1/3` and the
        # diagonal kill are still applied once, at the end.
        MP, PP, PPL = _p2_recon(tw)
        LA = 2.0 * (tw.A_L * tw.M) * tw.w2b + PP * tw.eb
        LP = (tw.A_L * tw.A_L) * tw.w2b + PP * (tw.sb / 3.0) + (2.0 / 3.0) * MP
        acc = _mm(LA, tw.A_L.T) + _mm(LP, tw.M.T)
        # the thin leg: one `(n, k r) @ (k r, n)`, and the contraction over the
        # `k r` columns IS the sum over lineages -- the same trick the wide
        # layout already uses for the `k n` columns.
        return (acc + _mm(PPL, tw.Z.T) * (1.0 / 3.0)) * off_mask
    acc = _mm(2.0 * (tw.A_D * tw.M) + 3.0 * (tw.A_L * C_L), tw.M.T)''',
        "the (2,1) regrouping", 1))

    # ---- 6. birth: append the factored leg -------------------------------
    P.append((
        '''def _append(tw, A_D, A_L, w2b, n):
    """Add the block born at this layer. Its `M` is `I` and is not stored."""
    if tw is None:
        return _Tower(A_D, A_L, None, w2b, 1)
    return _Tower(fnp.concatenate([tw.A_D, A_D], axis=1),
                  fnp.concatenate([tw.A_L, A_L], axis=1),
                  tw.M, fnp.concatenate([tw.w2b, w2b]), tw.k + 1)''',
        '''def _append(tw, A_D, A_L, w2b, n, Zb=None, Lb=None, sb=None, eb=None):
    """Add the block born at this layer. Its `M` is `I` and is not stored.

    P2 appends the FACTORED leg instead of a dense `A_D`: `Z` grows by `r`
    columns, `L` by one `(n, r)` page, and `sb`/`eb` by `n` entries each.
    """
    if tw is None:
        return _Tower(A_D, A_L, None, w2b, 1, Zb,
                      None if Lb is None else Lb[None], sb, eb)
    return _Tower(None if A_D is None else fnp.concatenate([tw.A_D, A_D], axis=1),
                  fnp.concatenate([tw.A_L, A_L], axis=1),
                  tw.M, fnp.concatenate([tw.w2b, w2b]), tw.k + 1,
                  None if Zb is None else fnp.concatenate([tw.Z, Zb], axis=1),
                  None if Lb is None else fnp.concatenate([tw.L, Lb[None]], axis=0),
                  None if sb is None else fnp.concatenate([tw.sb, sb]),
                  None if eb is None else fnp.concatenate([tw.eb, eb]))''',
        "append the factored leg", 1))

    # ---- 7. truncation drops the factored leg's oldest pages -------------
    P.append((
        '''    M = None if (tw.M is None or g <= 1) else tw.M[:, -((g - 1) * n):]
    return _Tower(tw.A_D[:, -keep:], tw.A_L[:, -keep:], M, tw.w2b[-keep:], g)''',
        '''    M = None if (tw.M is None or g <= 1) else tw.M[:, -((g - 1) * n):]
    if tw.A_D is None:
        r = tw.L.shape[2]
        return _Tower(None, tw.A_L[:, -keep:], M, tw.w2b[-keep:], g,
                      tw.Z[:, -(g * r):], tw.L[-g:],
                      tw.sb[-keep:], tw.eb[-keep:])
    return _Tower(tw.A_D[:, -keep:], tw.A_L[:, -keep:], M, tw.w2b[-keep:], g)''',
        "truncation drops the leg's oldest pages", 1))

    # ---- 8. the birth site ----------------------------------------------
    P.append((
        '''    out_tw = _append(adv, fnp.diag(S3) + 3.0 * S21.T, fac1, w2b_vec, n)''',
        '''    if _P2_SEPARABLE:
        # P2: the leg is born FACTORED and `A_D` is never built. `diag(S3)`
        # rides on `M` (which is `I` at birth), the separable part of `S21`
        # rides on `A_L` as a column scale -- `3 S_sep^T = 3 fac1 d(e)`,
        # because `C11` is symmetric and `fac1 = w1[:,None] * C11` -- and only
        # the residual gets a leg. Proved on real F21 state at this very site
        # to 3.7 float32 ulps (`local_sim/p2_p2_identity_test.py`).
        e_b = w[(1, 2)] - 2.0 * pK[(1,)] * w1
        S_sep = (e_b[:, None] * C11) * w1[None, :]
        Zb, Lb = _p2_compress(S21 - S_sep, W)
        out_tw = _append(adv, None, fac1, w2b_vec, n, Zb, Lb, S3, e_b)
    else:
        out_tw = _append(adv, fnp.diag(S3) + 3.0 * S21.T, fac1, w2b_vec, n)''',
        "the factored birth", 1))

    # ---- 9. the cost model --------------------------------------------
    P.append((
        '''        if l == depth - 1:
            total += 2 + aged * 3 + young * 2
        else:
            total += 2 + aged * 7 + young * 6
    return total''',
        '''        if l == depth - 1:
            # last layer: transports only, no birth and no slices.
            total += 2 + (aged * 2 + young * 1 if _P2_SEPARABLE
                          else aged * 3 + young * 2)
        else:
            # P2's `4`/`3` is READ OFF THE CODE: 2 transports (A_L, M; the
            # youngest M is `I` and free) + 2 contractions (LA A_L^T, LP M^T).
            # F21's `7`/`6` is left exactly as it stands even though the code
            # issues 6/5 -- see this generator's header. It only picks `gens`.
            total += 2 + (aged * 4 + young * 3 if _P2_SEPARABLE
                          else aged * 7 + young * 6)
    return total''',
        "the cost model learns the new count", 1))

    # ---- 10. the thin FLOPs enter the projection -------------------------
    P.append((
        '''    return (_matmuls(depth, gens) * n * n * (2 * n - 1)
            + _ELEM_PER_LAYER * n * n * depth)''',
        '''    base = (_matmuls(depth, gens) * n * n * (2 * n - 1)
            + _ELEM_PER_LAYER * n * n * depth)
    if not _P2_SEPARABLE:
        return base
    # The thin terms: transport `W^T Z`, the dense reconstruction `Z L^T`,
    # `(M.M) L`, and `[(M.M)L] Z^T` -- four `O(k n^2 r)` products a layer.
    r = n if _P2_RANK <= 0 else min(int(_P2_RANK), n)
    thin = 0
    for l in range(depth):
        k = l if gens is None else min(l, gens)
        thin += 4 * k * n * n * r
    return base + thin''',
        "the thin FLOPs enter the projection", 1))

    if not separable:
        # The control reaches every patch site, so the two files cannot drift
        # apart through separate editing -- but `_P2_SEPARABLE = False` makes
        # every new branch dead, and `A_D` is never None.
        pass
    return P


# ==========================================================================

_HEADER = '''"""%(title)s

%(blurb)s

GENERATED by `local_sim/p2_build_v31.py` from `estimator_k3_v30.py`
(sha256 %(sha)s).
DO NOT EDIT THIS FILE. Edit the generator and re-emit, or the candidate and
its control drift apart and the comparison stops meaning anything.

--------------------------------------------------------------------------
INHERITED FROM v30 (P1, the per-neuron fourth-order core)

%(v30doc)s
"""
'''


def build(separable, rank, title, blurb, src, sha, v30doc):
    body = src
    for old, new, why, count in _patches(separable):
        found = body.count(old)
        if found != count:
            raise SystemExit(
                "PATCH ANCHOR COUNT WRONG (%s): expected %d, found %d.\n"
                "v30 has moved under this generator. A generator that silently "
                "skips or over-applies a patch emits a CONTROL wearing the "
                "CANDIDATE's name, so this is fatal.\nAnchor:\n%s"
                % (why, count, found, old[:280]))
        body = body.replace(old, new, count)

    # strip v30's docstring; it is carried into the new header instead
    i = body.find('"""')
    j = body.find('"""', i + 3)
    body = body[j + 3:].lstrip("\n")

    head = _HEADER % {"title": title, "blurb": blurb, "sha": sha,
                      "v30doc": v30doc.strip()}
    consts = ("_P2_SEPARABLE = %r\n"
              "#: Residual rank. 0 means FULL width -- an exact range finder\n"
              "#: and a float32 copy of the dense `A_D`, which is the\n"
              "#: algebraic-parity control, not a cost candidate.\n"
              "_P2_RANK = %r\n\n" % (separable, rank))
    lines = body.splitlines(keepends=True)
    last_import = 0
    for k, ln in enumerate(lines):
        if ln.startswith(("import ", "from ")):
            last_import = k
    lines.insert(last_import + 1, "\n" + consts)
    body = "".join(lines)

    # the new machinery goes in ahead of its first consumer
    anchor = "def _dslice3(tw):"
    if body.count(anchor) != 1:
        raise SystemExit("cannot place _p2 helpers: `%s` not unique" % anchor)
    body = body.replace(anchor, _P2_FUNCS.lstrip("\n") + anchor, 1)
    return head + body


def emit(rank=DEFAULT_RANK):
    with open(SOURCE, "rb") as fh:
        raw = fh.read()
    src, sha = raw.decode("utf-8"), hashlib.sha256(raw).hexdigest()
    i = src.find('"""')
    j = src.find('"""', i + 3)
    v30doc = src[i + 3:j]
    print("source  estimator_k3_v30.py  sha256 %s..." % sha[:16])
    for path, sep, rk, title, blurb in (
            (TARGET, True, rank,
             "v31: v30 plus the SEPARABLE (2,1) leg (P2).",
             "`A_D` is never materialised or transported. It is carried as\n"
             "`M d(sb) + 3 A_L d(eb) + Z L^T`, which turns six dense products\n"
             "a layer into four plus O(k n^2 r) thin work."),
            (CONTROL, False, rank,
             "v31 CONTROL: v30, reached through the generator.",
             "_P2_SEPARABLE = False. Bit-identical to v30 in output and in\n"
             "counted FLOPs; it exists so the candidate and its control cannot\n"
             "drift apart through separate editing."),
            (DENSE, True, 0,
             "v31 DENSE: the separable leg at FULL rank.",
             "_P2_RANK = 0. Algebraically identical to v30 and MORE expensive\n"
             "than it. This arm answers 'did the regrouping break anything',\n"
             "separately from 'does truncation cost accuracy'."),
    ):
        text = build(sep, rk, title, blurb, src, sha, v30doc)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        print("  wrote %-32s separable=%-5s rank=%-3s %d bytes"
              % (os.path.basename(path), sep, rk, len(text)))
    return 0


def parity():
    """Four claims, each of which can fail on its own.

    1. the CONTROL is v30 bit for bit, at the same counted FLOPs;
    2. the DENSE arm is v30 to float32 -- the regrouping is exact ALGEBRA, so
       anything beyond reassociation noise here is a bug in the identity, not
       a cost of the rank;
    3. the CANDIDATE differs from v30 (a no-op cannot test P2) and costs LESS;
    4. `_fit_generations` picks the same `gens` for control and candidate at
       the LIVE shape, so the two arms are structurally comparable despite
       their different `_matmuls` branches.
    """
    import sys
    import numpy as np
    sys.path[:] = [p for p in sys.path
                   if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]
    import flopscope as flops
    import harness
    import lifecycle_eval as le
    from mint_provenance import canonical_mint

    flops.configure(symmetry_warnings=False)
    width, depth, budget = 96, 6, 2 ** 41
    spec = le.LifecycleSpec.for_round("v2-phase2", seed=0)
    order = (("v30", SOURCE), ("control", CONTROL),
             ("dense", DENSE), ("v31", TARGET))
    mods = {}
    for name, path in order:
        m, _ = le.load_candidate(path)
        e = m.Estimator()
        e.setup(le.SetupContext(spec))
        mods[name] = m

    ok = True

    # ---- 4. the budget guard agrees at the LIVE shape --------------------
    lw, ld = 1024, 16
    g_ctl = mods["control"]._fit_generations(lw, ld, 2 ** 41)
    g_can = mods["v31"]._fit_generations(lw, ld, 2 ** 41)
    f_ctl = mods["control"]._projected_flops(lw, ld, g_ctl)
    f_can = mods["v31"]._projected_flops(lw, ld, g_can)
    print("budget guard at %dx%d" % (lw, ld))
    print("  gens  control %-6s candidate %-6s   %s"
          % (g_ctl, g_can, "AGREE" if g_ctl == g_can else "*** DIFFER ***"))
    print("  projected tower FLOPs  %.4e -> %.4e   %.4fx"
          % (f_ctl, f_can, f_can / f_ctl))
    if g_ctl != g_can:
        ok = False
        print("  *** the arms would carry different towers ***")

    print("\nparity at %dx%d" % (width, depth))
    for seed in (90000000401, 90000000402, 90000000403):
        w, es, _ = canonical_mint(width, depth, seed)
        out, cost = {}, {}
        for name, m in mods.items():
            mlp = harness.MLP(weights=[np.asarray(x) for x in w], width=width,
                              depth=depth, seed=es)
            ctx = flops.BudgetContext(flop_budget=budget, quiet=True)
            with ctx:
                p = m.Estimator().predict(mlp, budget)
            out[name] = np.asarray(p, dtype=np.float64)
            cost[name] = float(ctx.flops_used)
        s = max(float(np.max(np.abs(out["v30"]))), 1e-300)
        rel = lambda k: float(np.max(np.abs(out["v30"] - out[k]))) / s
        same = np.array_equal(out["v30"], out["control"])
        same_cost = cost["v30"] == cost["control"]
        print("  seed %d" % seed)
        print("    control == v30 bitwise: %-5s   same counted FLOPs: %-5s"
              % (same, same_cost))
        print("    dense  differs by %.3e relative   cost %.4fx"
              % (rel("dense"), cost["dense"] / cost["v30"]))
        print("    v31    differs by %.3e relative   cost %.4fx"
              % (rel("v31"), cost["v31"] / cost["v30"]))
        if not same or not same_cost:
            ok = False
            print("    *** THE CONTROL IS NOT v30 ***")
        if rel("dense") > 1e-4:
            ok = False
            print("    *** THE REGROUPING IS NOT EXACT -- identity bug ***")
        if rel("v31") < 1e-12:
            ok = False
            print("    *** v31 CHANGED NOTHING -- a no-op cannot test P2 ***")
    print("\nPARITY OK" if ok else "\nPARITY FAILED")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--rank", type=int, default=DEFAULT_RANK)
    args = ap.parse_args(argv)
    if args.emit:
        return emit(args.rank)
    if args.parity:
        return parity()
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
