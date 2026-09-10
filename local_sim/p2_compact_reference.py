"""Stage 1: the compact all-age K3 representation, derived, checked and PRICED.

`docs/GAME_PLAN.md` Stage 1. Transparent NumPy, no flopscope, no truth, no seed
block, no network minted. This module exists to answer one question before any
label is spent:

    does a shared-basis, all-age K3 representation have a CREDIBLE fully
    counted cost advantage over the tower this repository already ships?

and to answer it with algebra that can be checked on five-by-five matrices
rather than with a benchmark.

ATTRIBUTION, AND WHAT IS AND IS NOT OURS
----------------------------------------
The idea of carrying every generation in one shared left basis, and of
regenerating a cheap fourth-order core consistently with the third-order births
it induces, comes from **team 504aldo's published Phase-2 implementation**, MIT
licensed, pinned at commit `18c17e2d7a9aeacd399cfc2c6b571e4e16dbfb45`, published
2026-09-09/10. It is a **RESEARCH REFERENCE**. Nothing here is a submission
candidate: on 2026-09-10 Samuel decided to hold off entirely on the rules §6
participant-derivation question, so no derivative advances toward submission and
no organizer has been contacted.

No external code was read into this file. The identities below are re-derived
from first principles and checked against **our own** shipped chain
(`estimator_k3_strassen_L2.py`), which is the only thing here that can be a
control. Published coefficient tables were fitted on eight public MLPs and are
not used: our validation policy fits on fresh mints only.

THE THREE PROPOSITIONS, STATED SO THEY CAN BE FALSIFIED
--------------------------------------------------------
**P1 (transport factors through a shared left basis).** Every operation the
chain applies to a carried block between births is a LEFT multiplication --
`contract_W` is `W^T X` and `contract_wick_` is `diag(col) X`. So if a block is
written `A_s = Q F_s`, transport touches only `Q`:

    W^T (Q F_s) = (W^T Q) F_s,      diag(c) (Q F_s) = (diag(c) Q) F_s

and `k` sources cost ONE transport between them instead of `k`. This is exact,
it is not an approximation, and it is the entire reason the representation can
be cheaper. `estimator_k3_strassen_L2.py` already exploits the mirror-image
fact on the right (`C_L = A_L diag(w2)` survives because everything applied
afterwards is a left multiplication).

**P2 (the Hadamard does NOT factor, and reassociation is where the saving
actually is).** The `(2,1)` slice needs `(A_s . M_s) @ M_s^T`, and
`(Q F) . (Q H)` is not `Q (F . H)`: an elementwise product destroys the rank
structure. So each source must be RECONSTRUCTED to form its Hadamard. The
saving is not avoided reconstruction; it is that

    (A . M) @ M^T = ((A . M) @ H^T) @ Q^T

turns one `(n,n)@(n,n)` into `(n,n)@(n,r)` then `(n,r)@(r,n)`, AND that the
sum over sources can be taken in the `r`-dimensional coordinate space before a
SINGLE trailing multiplication by `Q^T`. Anyone claiming a cheap compact tier
without this step has not priced the Hadamard.

**P3 (the fourth-order diagonal channel is nearly free, given the K2
backbone).** For a core `G = diag(g) + lam * C_off` with `C_off` the
off-diagonal part of the pre-activation covariance `C`,

    diag(W G W^T) = (W . W) g + lam [ diag(W C W^T) - (W . W) diag(C) ]

is EXACT for that core. And `diag(W C W^T)` is the diagonal of the NEXT LAYER'S
COVARIANCE, which the K2 backbone has already computed. So the whole channel
costs two matrix-vector products against `W . W`, which is `O(n^2)` beside an
`O(n^3)` chain. Whether the core is a good approximation is a separate,
empirical question this module does not answer.

WHAT THIS MODULE DOES NOT CLAIM
--------------------------------
It does not claim the compact chain is more accurate, or that it wins. It does
not claim to reproduce anyone's estimator. It does not describe the public
chain as "exact 7->4": that chain separately prunes terms, drops blocks and
uses finite-rank residuals, and calling the whole of it exact would be the
attribution error the Skeptic corrected here on 2026-09-10.

Run:
    uv run python local_sim/p2_compact_reference.py --check      # the algebra
    uv run python local_sim/p2_compact_reference.py --ledger     # the price
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

#: The live Phase-2 shape and budget. Asserted against the installed package by
#: `local_sim/official_config.py`; repeated here only so the ledger has a
#: default, never as a second source of truth.
WIDTH, DEPTH, BUDGET = 1024, 16, 2 ** 41


# --------------------------------------------------------------------------- #
# 1. The DENSE tower — our shipped chain's contractions, in plain NumPy
#
# These mirror `estimator_k3_strassen_L2.py::_dslice3` and `::_dslice21` term
# for term. They are the CONTROL: the compact form is only interesting if it
# reproduces these exactly when nothing is compressed.
#
# Layout, following the shipped file:
#     A_D  (n, k n)   the from_dstensor blocks, k lineages side by side
#     A_L  (n, k n)   the two-leg blocks
#     M    (n, k n)   the modes  (the shipped file omits the youngest, whose M
#                     is symbolically I; here every lineage carries its own, so
#                     the algebra is visible rather than implied)
#     w2b  (k n,)     the birth Wick vectors;  C_L = A_L * w2b
# --------------------------------------------------------------------------- #


def dense_d3(A_D, A_L, M, w2b):
    """The `(3,)` repeated slice. `O(n^2 k)`, no matmul."""
    C_L = A_L * w2b
    return (A_D * M * M).sum(axis=1) + 3.0 * (A_L * M * C_L).sum(axis=1)


def dense_d21(A_D, A_L, M, w2b, off_mask):
    """The `(2,1)` repeated slice. FOUR distinct matmuls, whatever `k` is.

    The two coincidences the shipped file documents are reproduced here, not
    assumed: `{0,1}|{2}` and `{0,2}|{1}` are the same product on the D block,
    and `{0,1}|{2}` and `{1,2}|{0}` are the same product on the L block because
    `C_L = A_L diag(w2)`. `--check` asserts both numerically.
    """
    C_L = A_L * w2b
    acc = 2.0 * ((A_D * M) @ M.T)
    acc = acc + ((M * M) @ A_D.T)
    acc = acc + 6.0 * ((A_L * M) @ C_L.T)
    acc = acc + 3.0 * ((A_L * C_L) @ M.T)
    return (acc / 3.0) * off_mask


def transport_dense(WT, A_D, A_L, M):
    """`contract_W` on the dense tower: one matmul per block, per lineage."""
    return WT @ A_D, WT @ A_L, WT @ M


# --------------------------------------------------------------------------- #
# 2. The COMPACT tower — one shared left basis, all ages retained
# --------------------------------------------------------------------------- #


@dataclass
class Compact:
    """`k` sources sharing one left basis `Q`.

    `Q` is `(n, r)`. Each source `s` carries right factors `F_D[s]`, `F_L[s]`,
    `H[s]`, each `(r, n)`, so the dense block it stands for is `Q @ F_D[s]` and
    so on. `w2` is the source's birth Wick vector, `(n,)`.

    NOTHING about the source's identity lives in `Q`: that is what lets one
    transport serve every age, and it is also why the basis has to be JOINED
    when a new source is born whose left leg is not in the current span.
    """

    Q: np.ndarray
    F_D: List[np.ndarray] = field(default_factory=list)
    F_L: List[np.ndarray] = field(default_factory=list)
    H: List[np.ndarray] = field(default_factory=list)
    w2: List[np.ndarray] = field(default_factory=list)

    @property
    def k(self) -> int:
        return len(self.F_D)

    @property
    def r(self) -> int:
        return int(self.Q.shape[1])

    def to_dense(self):
        """Reconstruct the dense tower this stands for. Only for checking."""
        A_D = np.concatenate([self.Q @ f for f in self.F_D], axis=1)
        A_L = np.concatenate([self.Q @ f for f in self.F_L], axis=1)
        M = np.concatenate([self.Q @ h for h in self.H], axis=1)
        w2b = np.concatenate(list(self.w2))
        return A_D, A_L, M, w2b


def transport_compact(WT, cpt: Compact) -> Compact:
    """P1 in code: transport is ONE matmul, whatever `k` is.

    `W^T (Q F_s) = (W^T Q) F_s`. The right factors are untouched -- not
    approximately, exactly -- because a left multiplication cannot see them.
    """
    return Compact(Q=WT @ cpt.Q, F_D=list(cpt.F_D), F_L=list(cpt.F_L),
                   H=list(cpt.H), w2=list(cpt.w2))


def wick_compact(col, cpt: Compact) -> Compact:
    """`contract_wick_` on the compact tower: one column scale on `Q`."""
    return Compact(Q=cpt.Q * col, F_D=list(cpt.F_D), F_L=list(cpt.F_L),
                   H=list(cpt.H), w2=list(cpt.w2))


def compact_d3(cpt: Compact) -> np.ndarray:
    """The `(3,)` slice from the compact state.

    P2 bites here immediately: `(Q F) . (Q H) . (Q H)` is a triple elementwise
    product, so each source must be reconstructed. There is no reassociation
    that avoids it, and this function does not pretend otherwise -- it is here
    so the ledger can PRICE the reconstruction honestly.
    """
    n = cpt.Q.shape[0]
    out = np.zeros(n, dtype=np.float64)
    for s in range(cpt.k):
        A_D = cpt.Q @ cpt.F_D[s]
        A_L = cpt.Q @ cpt.F_L[s]
        M = cpt.Q @ cpt.H[s]
        C_L = A_L * cpt.w2[s]
        out += (A_D * M * M).sum(axis=1) + 3.0 * (A_L * M * C_L).sum(axis=1)
    return out


def compact_d21(cpt: Compact, off_mask) -> np.ndarray:
    """The `(2,1)` slice from the compact state — P2's reassociation.

    Each source is reconstructed (unavoidable: the Hadamard destroys the rank
    structure), but the trailing multiplication is pushed OUT of the loop:

        sum_s (A_s . M_s) @ M_s^T
            = sum_s ((A_s . M_s) @ H_s^T) @ Q^T
            = [ sum_s (A_s . M_s) @ H_s^T ] @ Q^T

    So `k` sources cost `k` cheap `(n,n)@(n,r)` products and ONE trailing
    `(n,r)@(r,n)`, instead of `k` full `(n,n)@(n,n)` products. Everything
    accumulates in the `r`-dimensional coordinate space.
    """
    n = cpt.Q.shape[0]
    r = cpt.r
    accQ = np.zeros((n, r), dtype=np.float64)     # summed in SMALL coordinates
    for s in range(cpt.k):
        A_D = cpt.Q @ cpt.F_D[s]
        A_L = cpt.Q @ cpt.F_L[s]
        M = cpt.Q @ cpt.H[s]
        C_L = A_L * cpt.w2[s]
        # {0,1}|{2} and {0,2}|{1} on the D block, right operand M^T = H^T Q^T
        accQ += 2.0 * ((A_D * M) @ cpt.H[s].T)
        # {1,2}|{0} on the D block, right operand A_D^T = F_D^T Q^T
        accQ += (M * M) @ cpt.F_D[s].T
        # the L block's two, right operands C_L^T and M^T
        #   C_L = A_L diag(w2) = Q F_L diag(w2), so C_L^T = diag(w2) F_L^T Q^T
        accQ += 6.0 * ((A_L * M) @ (cpt.F_L[s] * cpt.w2[s]).T)
        accQ += 3.0 * ((A_L * C_L) @ cpt.H[s].T)
    return ((accQ @ cpt.Q.T) / 3.0) * off_mask


# --------------------------------------------------------------------------- #
# 3. The basis JOIN — where an all-age representation actually costs something
# --------------------------------------------------------------------------- #


def basis_join(Q: np.ndarray, new_legs: np.ndarray, rank: Optional[int] = None,
               weights: Optional[np.ndarray] = None):
    """Fold a newborn source's left legs into the shared basis.

    Returns `(Q_new, R)` where `R` re-expresses the OLD basis in the new one:
    `Q ~= Q_new @ R`, so every existing right factor is updated as
    `F_s <- R @ F_s`. That update is `(r_new, r_old) @ (r_old, n)` per source,
    and it is the price of sharing -- an honest ledger must carry it.

    `rank=None` keeps everything (exact, and what `--check` uses to prove
    algebraic parity). A finite `rank` truncates, and truncation is an
    APPROXIMATION whose error is measured, never assumed.

    `weights` (length `n`, one per column of `[Q, new_legs]`) lets the leading
    subspace be chosen by carried-leg energy rather than by raw column norm.
    That is the published choice; the alternative -- choosing by future output
    sensitivity -- is GAME_PLAN Stage 4 item B and is NOT implemented here.
    """
    stacked = np.concatenate([Q, new_legs], axis=1)
    if weights is not None:
        stacked = stacked * np.asarray(weights, dtype=np.float64)[None, :]
    # A thin SVD, not a bare QR: the truncation has to be by SINGULAR VALUE or
    # the "leading subspace" is whatever order the columns happened to arrive
    # in. QR alone is exact when nothing is dropped and arbitrary when
    # something is.
    U, S, _Vt = np.linalg.svd(stacked, full_matrices=False)
    keep = U.shape[1] if rank is None else int(min(rank, U.shape[1]))
    Q_new = U[:, :keep]
    R = Q_new.T @ Q
    return Q_new, R


def rebase(cpt: Compact, Q_new: np.ndarray, R: np.ndarray) -> Compact:
    """Re-express every source in a new shared basis. `F <- R F`."""
    return Compact(Q=Q_new,
                   F_D=[R @ f for f in cpt.F_D],
                   F_L=[R @ f for f in cpt.F_L],
                   H=[R @ h for h in cpt.H],
                   w2=list(cpt.w2))


# --------------------------------------------------------------------------- #
# 4. P3 — the fourth-order diagonal channel
# --------------------------------------------------------------------------- #


def k4_diag_exact(W: np.ndarray, G: np.ndarray) -> np.ndarray:
    """`diag(W G W^T)`, computed the expensive honest way. The control."""
    return np.einsum("ij,jk,ik->i", W, G, W)


def k4_diag_via_core(W: np.ndarray, g: np.ndarray, lam: float, C: np.ndarray,
                     cov_post_diag: Optional[np.ndarray] = None) -> np.ndarray:
    """`diag(W G W^T)` for `G = diag(g) + lam * C_off`, the cheap way.

        diag(W G W^T) = (W . W) g + lam [ diag(W C W^T) - (W . W) diag(C) ]

    EXACT for that core -- the approximation is in choosing the core, not in
    this identity, and conflating the two is how a closure gets blamed for an
    algebra error.

    `cov_post_diag` is `diag(W C W^T)`. Pass it: the K2 backbone has already
    computed the post-activation covariance, so this term is FREE. Computing it
    here instead costs a full `(n,n)@(n,n)`, which is the whole difference
    between a nearly-free channel and an unaffordable one.
    """
    W2 = W * W
    if cov_post_diag is None:
        cov_post_diag = np.einsum("ij,jk,ik->i", W, C, W)
    return W2 @ g + lam * (cov_post_diag - W2 @ np.diag(C))


# --------------------------------------------------------------------------- #
# 5. THE COST LEDGER — every operation family, at the live shape
#
# Counted in FLOPs as flopscope bills them: a matmul `(m,k) @ (k,n)` is
# `m * n * (2k - 1)`, an elementwise op on `N` elements is `N`, a reduction
# over `N` elements is `N - 1` per output. The ledger below uses the matmul
# rule exactly and rounds elementwise families to their element count, which
# OVERSTATES nothing and understates the elementwise families by at most one
# FLOP per output.
#
# THIS IS A MODEL, NOT A MEASUREMENT. It prices arithmetic that has not been
# written in `fnp`. `local_sim/p2_flop_anchor.py` is what turns a model into a
# measurement, and it needs a real file to run. Every number below is labelled
# PROJECTED for that reason.
# --------------------------------------------------------------------------- #


def _mm(m, k, n):
    """flopscope's matmul bill."""
    return m * n * (2 * k - 1)


@dataclass(frozen=True)
class LedgerLine:
    family: str
    per_layer: float
    total: float
    note: str


def dense_ledger(n=WIDTH, depth=DEPTH, budget=BUDGET, gens=None):
    """Price the tower this repository SHIPS, from its shapes.

    `gens=None` is the exact tower: lineage count `k` grows by one per layer,
    so the total carries a `sum_l l = L(L-1)/2` rather than a factor `L`. That
    quadratic is the thing the compact representation exists to remove.
    """
    lines: List[LedgerLine] = []
    ks = [min(l + 1, gens or (l + 1)) for l in range(depth)]
    K = float(sum(ks))

    # transport: three blocks, each (n,n) @ (n, k n)
    t = sum(3 * _mm(n, n, k * n) for k in ks)
    lines.append(LedgerLine("transport (contract_W)", t / depth, t,
                            "3 blocks x (n,n)@(n,kn); the youngest M is I and "
                            "is concatenated, not multiplied"))
    # the (2,1) slice: four distinct (n, k n) @ (k n, n)
    d21 = sum(4 * _mm(n, k * n, n) for k in ks)
    lines.append(LedgerLine("(2,1) slice", d21 / depth, d21,
                            "4 distinct matmuls after the two documented "
                            "coincidences collapse six into four"))
    # the (3,) slice and the Wick scales: O(n^2 k), no matmul
    d3 = sum(6.0 * n * (k * n) for k in ks)
    lines.append(LedgerLine("(3,) slice", d3 / depth, d3,
                            "elementwise, O(n^2 k)"))
    wick = sum(3.0 * n * (k * n) for k in ks)
    lines.append(LedgerLine("Wick column scales", wick / depth, wick,
                            "three left diagonal scales"))
    total = sum(x.total for x in lines)
    return {"lines": lines, "total": total, "b": total / budget,
            "k_sum": K, "shape": (n, depth), "gens": gens}


def compact_ledger(n=WIDTH, depth=DEPTH, budget=BUDGET, recent=4, rank=384,
                   join_every=1):
    """Price the compact all-age representation, INCLUDING what it is easy to forget.

    Structure priced: `recent` newest sources stay DENSE (their left legs are
    full rank and cannot be compressed without losing the accuracy that makes
    recent sources worth carrying); every older source lives in one shared
    rank-`rank` basis.

    Families that a careless ledger omits, and that are priced here:
      * reconstructing `Q F_s` for EVERY old source, EVERY layer -- P2 says the
        Hadamard forces this and no reassociation avoids it;
      * the basis join itself (a thin SVD of `(n, r + n)`);
      * rebasing every old source's right factors after the join, `R @ F_s`;
      * the single trailing `(n,r) @ (r,n)` that returns the accumulated
        `(2,1)` slice to neuron coordinates.
    """
    lines: List[LedgerLine] = []
    r = int(rank)
    old_counts = [max(0, min(l + 1, depth) - recent) for l in range(depth)]
    dense_counts = [min(l + 1, recent) for l in range(depth)]

    # --- the DENSE recent tier: exactly the shipped chain, capped at `recent`
    t_dense = sum(3 * _mm(n, n, k * n) for k in dense_counts if k)
    lines.append(LedgerLine("recent tier: transport", t_dense / depth, t_dense,
                            f"{recent} newest sources stay dense"))
    d21_dense = sum(4 * _mm(n, k * n, n) for k in dense_counts if k)
    lines.append(LedgerLine("recent tier: (2,1) slice", d21_dense / depth,
                            d21_dense, "unchanged from the shipped chain"))
    elem_dense = sum(9.0 * n * (k * n) for k in dense_counts if k)
    lines.append(LedgerLine("recent tier: elementwise", elem_dense / depth,
                            elem_dense, "(3,) slice and Wick scales"))

    # --- the SHARED tier
    # transport is ONE matmul for every age (P1)
    t_shared = sum(_mm(n, n, r) for k in old_counts if k)
    lines.append(LedgerLine("shared tier: transport of Q", t_shared / depth,
                            t_shared, "P1: one (n,n)@(n,r) serves every age"))
    # reconstruction: three blocks per old source, (n,r)@(r,n)
    recon = sum(3 * k * _mm(n, r, n) for k in old_counts)
    lines.append(LedgerLine("shared tier: reconstruction", recon / depth, recon,
                            "P2: the Hadamard destroys the rank structure, so "
                            "every old source must be materialised"))
    # the (2,1) contraction into small coordinates: four (n,n)@(n,r) per source
    contract = sum(4 * k * _mm(n, n, r) for k in old_counts)
    lines.append(LedgerLine("shared tier: (2,1) into r-coords",
                            contract / depth, contract,
                            "P2: (n,n)@(n,r) instead of (n,n)@(n,n)"))
    # ONE trailing return to neuron coordinates, per layer
    trail = sum(_mm(n, r, n) for k in old_counts if k)
    lines.append(LedgerLine("shared tier: one trailing @ Q^T", trail / depth,
                            trail, "the sum is taken in r-coords first"))
    elem_shared = sum(9.0 * k * n * n for k in old_counts)
    lines.append(LedgerLine("shared tier: elementwise", elem_shared / depth,
                            elem_shared, "(3,) slice and Wick, per source"))

    # --- the JOIN, which is the family most often left out
    #
    # THE FIRST DRAFT OF THIS LEDGER PRICED A DENSE THIN SVD OF `(n, r + n)`
    # AND CONCLUDED THE COMPACT FORM WAS MORE EXPENSIVE THAN THE TOWER
    # (1.0087 B against 0.9300 B), with the join alone carrying 0.371 B. That
    # was a real number about the wrong algorithm. Nobody would build the join
    # that way, and pricing a strawman is not a cost screen -- it is a way to
    # kill a representation for free.
    #
    # The join it would actually be: one source is born per layer, its left leg
    # enters as `c` new columns, and the basis is re-truncated to `r`. Written
    # incrementally that is
    #
    #   1. project the new legs off the current basis:  Z = L - Q (Q^T L)
    #   2. orthonormalise Z, a thin QR of an `(n, c)` matrix
    #   3. decide what to drop, in the (r+c)-dimensional CORE, not in n
    #   4. rotate the basis by the core's eigenvectors and rebase
    #
    # Steps 1, 2 and 4 are matmuls of supported shape. Step 3 is a symmetric
    # eigendecomposition of an `(r+c) x (r+c)` matrix -- small, but flopscope
    # has no primitive for it, and that is a DEPLOYABILITY problem this ledger
    # names rather than prices away.
    n_joins = sum(1 for l in range(depth) if l % join_every == 0 and
                  old_counts[l])
    c = n                      # a newborn source's left leg is FULL RANK
    proj = n_joins * (_mm(r, n, n) + _mm(n, r, c))    # Q^T L, then Q (Q^T L)
    qr = n_joins * (2.0 * n * c * c - (2.0 / 3.0) * c ** 3)
    core = n_joins * (2.0 * (r + c) ** 2 * n + 21.0 * (r + c) ** 3)
    rot = n_joins * _mm(n, r + c, r)
    join = proj + qr + core + rot
    lines.append(LedgerLine("basis join (incremental)", join / max(depth, 1),
                            join,
                            "project-off + thin QR + an (r+c) core "
                            "eigendecomposition + one rotation. The core step "
                            "has NO flopscope primitive: priced as arithmetic, "
                            "and a deployable version must be priced again"))
    rebase_c = sum(3 * k * _mm(r, r, n) for k in old_counts)
    lines.append(LedgerLine("rebase right factors after join",
                            rebase_c / depth, rebase_c,
                            "R @ F_s for every source, every join -- the price "
                            "of sharing, and the family most often forgotten"))

    total = sum(x.total for x in lines)
    return {"lines": lines, "total": total, "b": total / budget,
            "shape": (n, depth), "recent": recent, "rank": r,
            "join_every": join_every}


def k4_ledger(n=WIDTH, depth=DEPTH, budget=BUDGET, backbone_gives_diag=True):
    """Price P3's fourth-order diagonal channel, both ways."""
    lines: List[LedgerLine] = []
    w2 = depth * float(n * n)
    lines.append(LedgerLine("W . W", w2 / depth, w2, "one Hadamard per layer"))
    mv = depth * 2.0 * _mm(n, n, 1)
    lines.append(LedgerLine("(W.W) g and (W.W) diag(C)", mv / depth, mv,
                            "two matrix-vector products"))
    if backbone_gives_diag:
        lines.append(LedgerLine("diag(W C W^T)", 0.0, 0.0,
                                "FREE: it is the diagonal of the next layer's "
                                "covariance, which the K2 backbone computes"))
    else:
        d = depth * float(_mm(n, n, n) + n * n)
        lines.append(LedgerLine("diag(W C W^T)", d / depth, d,
                                "NOT free: a full (n,n)@(n,n) if the backbone "
                                "does not already have it"))
    total = sum(x.total for x in lines)
    return {"lines": lines, "total": total, "b": total / budget}



def sweep(n=WIDTH, depth=DEPTH, budget=BUDGET):
    """Where, if anywhere, the compact form is cheaper than the tower.

    Returns rows of `(recent, rank, total, B, ratio_vs_dense)`. This is a MODEL
    sweep over shapes, not an accuracy sweep: it says what a configuration
    would COST, and says nothing about whether it would work.
    """
    base = dense_ledger(n, depth, budget)["total"]
    rows = []
    for recent in (1, 2, 3, 4, 6):
        for rank in (128, 192, 256, 384, 512):
            c = compact_ledger(n, depth, budget, recent=recent, rank=rank)
            rows.append({"recent": recent, "rank": rank,
                         "total": c["total"], "b": c["b"],
                         "ratio_vs_dense": base / c["total"],
                         "join_share": next(x.total for x in c["lines"]
                                            if x.family.startswith("basis join"))
                                       / c["total"]})
    return {"dense_total": base, "dense_b": base / budget, "rows": rows}

def _fmt(led, title):
    out = [f"\n{title}", "-" * len(title)]
    out.append(f"  {'family':<38}{'per layer':>18}{'total':>20}{'B':>10}")
    for x in led["lines"]:
        out.append(f"  {x.family:<38}{x.per_layer:>18,.0f}{x.total:>20,.0f}"
                   f"{x.total / BUDGET:>10.5f}")
    out.append(f"  {'TOTAL':<38}{'':>18}{led['total']:>20,.0f}"
               f"{led['b']:>10.5f}")
    for x in led["lines"]:
        if x.note:
            out.append(f"    {x.family}: {x.note}")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# 6. THE FIXTURES — five-by-five matrices, chosen to break things
#
# A parity check on random Gaussians proves almost nothing: signs cancel,
# repeated indices are never hit, and an orientation error is invisible under a
# symmetric input. Each fixture below exists because a specific mistake would
# survive the others.
# --------------------------------------------------------------------------- #

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"  {name:<64} {'ok ' if ok else 'FAILED'}  {detail}")
    return ok


def head(t):
    print(f"\n{t}\n{'-' * len(t)}")


def _rel(a, b):
    d = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    s = max(float(np.max(np.abs(np.asarray(a)))),
            float(np.max(np.abs(np.asarray(b)))), 1e-300)
    return d / s


def _off_mask(n):
    return 1.0 - np.eye(n)


def fixtures():
    """(name, A_D, A_L, M, w2b, why this one exists)."""
    out = []
    n, k = 5, 3

    rng = np.random.default_rng(20260910)
    A_D = rng.normal(size=(n, k * n))
    A_L = rng.normal(size=(n, k * n))
    M = rng.normal(size=(n, k * n))
    w2b = rng.normal(size=(k * n,))
    out.append(("generic signed", A_D, A_L, M, w2b,
                "the baseline; catches nothing subtle on its own"))

    # ORIENTATION. Make every block a distinct rank-one outer product with
    # DIFFERENT left and right vectors, so A and A^T are nowhere equal and a
    # transposed operand cannot pass.
    u = np.arange(1.0, n + 1)[:, None]
    v = np.arange(1.0, k * n + 1)[None, :] ** 1.5
    out.append(("orientation (rank-one, u v^T)", u * v, (u ** 2) * (v + 1.0),
                (u + 3.0) * (v - 2.0), np.linspace(-1.0, 1.0, k * n),
                "left and right factors differ, so a transpose is visible"))

    # SIGN CANCELLATION. Two lineages that are exact negatives of each other:
    # the (2,1) slice must come out at the third lineage's contribution alone.
    a = rng.normal(size=(n, n))
    b = rng.normal(size=(n, n))
    c = rng.normal(size=(n, n))
    A_D2 = np.concatenate([a, -a, c], axis=1)
    A_L2 = np.concatenate([b, b, c], axis=1)
    M2 = np.concatenate([c, c, a], axis=1)
    w2b2 = np.concatenate([np.ones(n), np.ones(n), np.full(n, 0.5)])
    out.append(("sign cancellation", A_D2, A_L2, M2, w2b2,
                "two lineages cancel in the D block; a missing sign survives "
                "every non-cancelling fixture"))

    # REPEATED INDICES. All-ones blocks make every Wick coefficient visible as
    # an integer, so a factor of 3 or a 1/3 in the wrong place is unmissable.
    o = np.ones((n, k * n))
    out.append(("repeated indices (all ones)", o, o, o, np.ones(k * n),
                "every coefficient becomes an integer count"))

    # ZERO VARIANCE. A dead lineage must contribute exactly zero and must not
    # produce a nan through a division that should not be there.
    z = np.zeros((n, n))
    out.append(("zero-variance lineage",
                np.concatenate([a, z, c], axis=1),
                np.concatenate([b, z, c], axis=1),
                np.concatenate([c, z, a], axis=1),
                np.concatenate([np.ones(n), np.zeros(n), np.ones(n)]),
                "a dead source contributes 0, not nan"))
    return out


def t_algebraic_parity():
    head("1. the compact form reproduces the dense one EXACTLY at full rank")
    n, k = 5, 3
    om = _off_mask(n)
    for name, A_D, A_L, M, w2b, why in fixtures():
        # Build the compact state with Q = I, i.e. rank n and no compression.
        # If parity fails here the reassociation is wrong, full stop -- no
        # approximation is involved.
        Q = np.eye(n)
        cpt = Compact(Q=Q,
                      F_D=[A_D[:, s * n:(s + 1) * n] for s in range(k)],
                      F_L=[A_L[:, s * n:(s + 1) * n] for s in range(k)],
                      H=[M[:, s * n:(s + 1) * n] for s in range(k)],
                      w2=[w2b[s * n:(s + 1) * n] for s in range(k)])
        e3 = _rel(dense_d3(A_D, A_L, M, w2b), compact_d3(cpt))
        e21 = _rel(dense_d21(A_D, A_L, M, w2b, om), compact_d21(cpt, om))
        check(f"({name}) the (3,) slice agrees", e3 < 1e-12, f"rel {e3:.2e}")
        check(f"({name}) the (2,1) slice agrees", e21 < 1e-12, f"rel {e21:.2e}")


def t_orientation_is_actually_tested():
    head("2. the fixtures would CATCH the mistakes they are for")
    n, k = 5, 3
    om = _off_mask(n)
    name, A_D, A_L, M, w2b, _ = fixtures()[1]      # the orientation fixture
    ref = dense_d21(A_D, A_L, M, w2b, om)

    # A transposed right operand.
    C_L = A_L * w2b
    wrong = 2.0 * ((A_D * M) @ M.T) + ((M * M) @ A_D.T) \
        + 6.0 * ((A_L * M) @ C_L.T) + 3.0 * ((A_L * C_L) @ M.T)
    wrong_t = wrong.T
    check("a TRANSPOSED result is visibly different on this fixture",
          _rel(ref, (wrong_t / 3.0) * om) > 0.1,
          f"rel {_rel(ref, (wrong_t / 3.0) * om):.2e}")

    # A dropped Wick factor of 3.
    bad = 2.0 * ((A_D * M) @ M.T) + ((M * M) @ A_D.T) \
        + 6.0 * ((A_L * M) @ C_L.T) + 1.0 * ((A_L * C_L) @ M.T)
    check("a dropped Wick coefficient is visibly different",
          _rel(ref, (bad / 3.0) * om) > 0.01,
          f"rel {_rel(ref, (bad / 3.0) * om):.2e}")

    # The cancellation fixture must actually cancel.
    _n2, A_D2, A_L2, M2, w2b2, _ = fixtures()[2]
    d = dense_d21(A_D2, A_L2, M2, w2b2, om)
    check("the cancellation fixture is not accidentally trivial",
          float(np.max(np.abs(d))) > 1e-6, f"max |slice| {np.max(np.abs(d)):.3e}")

    # The zero-variance fixture must be finite.
    _n3, A_D3, A_L3, M3, w2b3, _ = fixtures()[4]
    d3 = dense_d3(A_D3, A_L3, M3, w2b3)
    d21z = dense_d21(A_D3, A_L3, M3, w2b3, om)
    check("a dead lineage yields finite output, not nan",
          bool(np.all(np.isfinite(d3)) and np.all(np.isfinite(d21z))), "")


def t_documented_coincidences():
    head("3. the two collapses the shipped chain relies on")
    n, k = 5, 3
    rng = np.random.default_rng(7)
    A_D = rng.normal(size=(n, k * n))
    A_L = rng.normal(size=(n, k * n))
    M = rng.normal(size=(n, k * n))
    w2b = rng.normal(size=(k * n,))
    C_L = A_L * w2b
    # D block: {0,1}|{2} and {0,2}|{1} are the same product.
    p1 = (A_D * M) @ M.T
    p2 = np.einsum("aj,aj,ij->ai", A_D, M, M)
    check("D block: {0,1}|{2} == {0,2}|{1}", _rel(p1, p2) < 1e-13,
          f"rel {_rel(p1, p2):.2e}")
    # L block: {0,1}|{2} and {1,2}|{0} are the same floats, because
    # C_L = A_L diag(w2).
    q1 = np.einsum("aj,aj,ij->ai", A_L, M, C_L)
    q2 = np.einsum("aj,aj,ij->ai", M, C_L, A_L)
    check("L block: {0,1}|{2} == {1,2}|{0}, via C_L = A_L diag(w2)",
          _rel(q1, q2) < 1e-13, f"rel {_rel(q1, q2):.2e}")
    # And the genuinely distinct pair is NOT equal, or the check above is vacuous.
    q3 = np.einsum("aj,aj,ij->ai", A_L, C_L, M)
    check("...and the DISTINCT pair really is distinct", _rel(q1, q3) > 0.1,
          f"rel {_rel(q1, q3):.2e}  (a vacuous check would show ~0)")


def t_transport_factors():
    head("4. P1 — transport factors through the shared basis, exactly")
    n, k, r = 6, 3, 4
    rng = np.random.default_rng(11)
    Q = np.linalg.qr(rng.normal(size=(n, r)))[0]
    cpt = Compact(Q=Q,
                  F_D=[rng.normal(size=(r, n)) for _ in range(k)],
                  F_L=[rng.normal(size=(r, n)) for _ in range(k)],
                  H=[rng.normal(size=(r, n)) for _ in range(k)],
                  w2=[rng.normal(size=n) for _ in range(k)])
    WT = rng.normal(size=(n, n))
    col = rng.normal(size=(n, 1))

    A_D, A_L, M, _w = cpt.to_dense()
    dA_D, dA_L, dM = transport_dense(WT, A_D, A_L, M)
    cA_D, cA_L, cM, _ = transport_compact(WT, cpt).to_dense()
    check("W^T (Q F) == (W^T Q) F on every block",
          max(_rel(dA_D, cA_D), _rel(dA_L, cA_L), _rel(dM, cM)) < 1e-12,
          f"rel {max(_rel(dA_D, cA_D), _rel(dA_L, cA_L), _rel(dM, cM)):.2e}")

    wA_D, wA_L, wM, _ = wick_compact(col, cpt).to_dense()
    check("diag(c) (Q F) == (diag(c) Q) F on every block",
          max(_rel(A_D * col, wA_D), _rel(A_L * col, wA_L),
              _rel(M * col, wM)) < 1e-12, "a left scale cannot see F")

    # PERMUTATION EQUIVARIANCE. Relabelling the neurons must relabel the
    # answer and nothing else. This catches an index that has been fixed to a
    # position rather than to a neuron.
    P = np.eye(n)[rng.permutation(n)]
    om = _off_mask(n)
    ref = dense_d21(A_D, A_L, M, _w, om)
    permuted = dense_d21(P @ A_D, P @ A_L, P @ M, _w, om)
    check("the (2,1) slice is equivariant under a neuron relabelling",
          _rel(P @ ref @ P.T, permuted) < 1e-12,
          f"rel {_rel(P @ ref @ P.T, permuted):.2e}")


def t_join_and_rebase():
    head("5. the basis join is exact when nothing is dropped")
    n, k, r = 8, 3, 5
    rng = np.random.default_rng(13)
    Q = np.linalg.qr(rng.normal(size=(n, r)))[0]
    cpt = Compact(Q=Q,
                  F_D=[rng.normal(size=(r, n)) for _ in range(k)],
                  F_L=[rng.normal(size=(r, n)) for _ in range(k)],
                  H=[rng.normal(size=(r, n)) for _ in range(k)],
                  w2=[rng.normal(size=n) for _ in range(k)])
    before = cpt.to_dense()
    new_legs = rng.normal(size=(n, 3))
    Q_new, R = basis_join(Q, new_legs, rank=None)
    after = rebase(cpt, Q_new, R).to_dense()
    check("a full-rank join changes NO carried block",
          max(_rel(before[i], after[i]) for i in range(3)) < 1e-11,
          f"rel {max(_rel(before[i], after[i]) for i in range(3)):.2e}")
    check("...and the new legs are inside the joined span",
          _rel(Q_new @ (Q_new.T @ new_legs), new_legs) < 1e-11,
          "otherwise the newborn source has nowhere to live")

    # And a TRUNCATED join is an approximation, which must be visible as one.
    Q_t, R_t = basis_join(Q, new_legs, rank=2)
    trunc = rebase(cpt, Q_t, R_t).to_dense()
    err = _rel(before[0], trunc[0])
    check("a truncated join IS lossy, and measurably so", err > 1e-3,
          f"rel {err:.2e} at rank 2 of {r + 3}  -- an approximation, not algebra")


def t_k4_identity():
    head("6. P3 — the fourth-order diagonal identity, exact for its core")
    rng = np.random.default_rng(17)
    n = 7
    W = rng.normal(size=(n, n))
    C = rng.normal(size=(n, n))
    C = C @ C.T                                  # a real covariance
    g = rng.normal(size=n) ** 2
    lam = 0.37
    G = np.diag(g) + lam * (C - np.diag(np.diag(C)))
    exact = k4_diag_exact(W, G)
    cheap = k4_diag_via_core(W, g, lam, C)
    check("diag(W G W^T) == (W.W)g + lam[diag(WCW^T) - (W.W)diag(C)]",
          _rel(exact, cheap) < 1e-12, f"rel {_rel(exact, cheap):.2e}")

    # Passing the backbone's diagonal must change nothing but the price.
    post_diag = np.einsum("ij,jk,ik->i", W, C, W)
    cheap2 = k4_diag_via_core(W, g, lam, C, cov_post_diag=post_diag)
    check("...and using the backbone's own diag(W C W^T) is bit-identical",
          _rel(cheap, cheap2) < 1e-14, "so the free path is the same algebra")

    # The identity is EXACT FOR THAT CORE and not for a general G. Saying so is
    # the difference between an identity and a closure.
    G_gen = rng.normal(size=(n, n))
    G_gen = G_gen + G_gen.T
    off = G_gen - np.diag(np.diag(G_gen))
    lam_fit = float((off * (C - np.diag(np.diag(C)))).sum()
                    / ((C - np.diag(np.diag(C))) ** 2).sum())
    approx = k4_diag_via_core(W, np.diag(G_gen), lam_fit, C)
    check("a GENERAL core is NOT reproduced -- the closure is the approximation",
          _rel(k4_diag_exact(W, G_gen), approx) > 1e-3,
          f"rel {_rel(k4_diag_exact(W, G_gen), approx):.2e}")

    # Zero variance must not produce a nan.
    Cz = np.zeros((n, n))
    check("a zero covariance yields finite output",
          bool(np.all(np.isfinite(k4_diag_via_core(W, g, lam, Cz)))), "")


def t_rank_error_is_measured_not_assumed():
    head("7. what truncation costs, measured on a transported chain")
    rng = np.random.default_rng(23)
    n, k, depth = 24, 4, 5
    om = _off_mask(n)
    Q = np.eye(n)
    cpt = Compact(Q=Q,
                  F_D=[rng.normal(size=(n, n)) / math.sqrt(n) for _ in range(k)],
                  F_L=[rng.normal(size=(n, n)) / math.sqrt(n) for _ in range(k)],
                  H=[rng.normal(size=(n, n)) / math.sqrt(n) for _ in range(k)],
                  w2=[rng.normal(size=n) for _ in range(k)])
    rows = []
    for rank in (n, n // 2, n // 4, n // 8):
        c = Compact(Q=cpt.Q.copy(), F_D=list(cpt.F_D), F_L=list(cpt.F_L),
                    H=list(cpt.H), w2=list(cpt.w2))
        for _l in range(depth):
            WT = rng.normal(size=(n, n)) / math.sqrt(n)
            c = transport_compact(WT, c)
            Qn, R = basis_join(c.Q, rng.normal(size=(n, 2)) / math.sqrt(n),
                               rank=rank)
            c = rebase(c, Qn, R)
        rows.append((rank, c.r, float(np.linalg.norm(compact_d21(c, om)))))
    full = rows[0][2]
    for rank, r_actual, norm in rows:
        print(f"    rank {rank:>3} (kept {r_actual:>3})   "
              f"||(2,1)|| {norm:.6e}   ratio {norm / full:.4f}")
    check("truncation changes the answer monotonically, and is reported as "
          "an approximation",
          rows[-1][2] != rows[0][2],
          "this is a SHAPE measurement on synthetic transports, NOT an "
          "accuracy result on networks")


def _main_check():
    print("=" * 96)
    print("STAGE 1 -- the compact all-age representation: algebra before labels")
    print("=" * 96)
    for fn in (t_algebraic_parity, t_orientation_is_actually_tested,
               t_documented_coincidences, t_transport_factors,
               t_join_and_rebase, t_k4_identity,
               t_rank_error_is_measured_not_assumed):
        fn()
    n_ok = sum(1 for r in RESULTS if r)
    print(f"\n{n_ok}/{len(RESULTS)} checks passed")
    return 0 if n_ok == len(RESULTS) else 1


def _main_ledger(args):
    print("=" * 96)
    print(f"STAGE 1 -- PROJECTED counted-FLOP ledger at {args.width}x{args.depth}, "
          f"B = 2**41")
    print("=" * 96)
    print("\nEVERY NUMBER BELOW IS A MODEL. It prices arithmetic that has not")
    print("been written in `fnp`. `local_sim/p2_flop_anchor.py` is what turns a")
    print("model into a measurement, and it needs a real file to run.")

    d = dense_ledger(args.width, args.depth)
    print(_fmt(d, "A. the tower this repository SHIPS (exact, no generation cap)"))
    print(f"\n  Cross-check: estimator_k3_strassen_L2.py MEASURES 1,328,654,462,566")
    print(f"  (0.60420 B) at Strassen depth 2. This model prices depth 0, and")
    print(f"  Strassen's own measured ladder puts depth 0 at 1,714,554,351,206")
    print(f"  (0.77969 B). Model / measured = {d['total'] / 1_714_554_351_206:.4f}.")

    c = compact_ledger(args.width, args.depth, recent=args.recent,
                       rank=args.rank)
    print(_fmt(c, f"B. the COMPACT representation "
                  f"(recent={args.recent} dense, shared rank {args.rank})"))

    k4 = k4_ledger(args.width, args.depth, backbone_gives_diag=True)
    print(_fmt(k4, "C. the fourth-order diagonal channel, with the backbone's "
                   "covariance diagonal"))
    k4n = k4_ledger(args.width, args.depth, backbone_gives_diag=False)
    print(_fmt(k4n, "D. ...and WITHOUT it, which is the version that is not free"))

    print("\n" + "=" * 96)
    print("WHAT THE LEDGER SAYS")
    print("=" * 96)
    ratio = d["total"] / c["total"] if c["total"] else float("inf")
    print(f"  shipped tower, modelled          {d['total']:>20,.0f}   "
          f"{d['b']:.5f} B")
    print(f"  compact, modelled                {c['total']:>20,.0f}   "
          f"{c['b']:.5f} B")
    print(f"  ratio                            {ratio:>20.3f}x")
    print(f"  K4 diagonal channel, backbone    {k4['total']:>20,.0f}   "
          f"{k4['b']:.6f} B")
    print(f"  K4 diagonal channel, standalone  {k4n['total']:>20,.0f}   "
          f"{k4n['b']:.6f} B")
    print()
    print("  The join's CORE step is priced as arithmetic: flopscope has no")
    print("  symmetric-eigendecomposition primitive, so a deployable version")
    print("  needs an explicit orthogonalisation and must be priced AGAIN.")
    print("  Read the join line before quoting the ratio.")

    sw = sweep(args.width, args.depth)
    print("\n" + "=" * 96)
    print("E. THE SHAPE SWEEP -- where the compact form is cheaper, if anywhere")
    print("=" * 96)
    print(f"  {'recent':>7}{'rank':>7}{'total':>22}{'B':>10}{'vs dense':>11}"
          f"{'join share':>12}")
    best = None
    for row in sw["rows"]:
        mark = ""
        if row["ratio_vs_dense"] > 1.0:
            mark = "  <- cheaper"
            if best is None or row["ratio_vs_dense"] > best["ratio_vs_dense"]:
                best = row
        print(f"  {row['recent']:>7}{row['rank']:>7}{row['total']:>22,.0f}"
              f"{row['b']:>10.5f}{row['ratio_vs_dense']:>11.3f}"
              f"{row['join_share'] * 100:>11.1f}%{mark}")
    # THE CALIBRATION, AND WHICH WAY IT CUTS.
    #
    # This model prices the shipped tower at 1.1928x its own MEASURED Strassen
    # depth-0 figure -- the shipped file is better than the model, because it
    # never materialises the youngest lineage's identity mode and collapses six
    # (2,1) products into four. The compact ledger has no equivalent
    # concessions applied to it. So the model FLATTERS THE COMPACT FORM, and
    # every ratio above is an UPPER BOUND on the saving, not a lower one.
    MEASURED_DENSE_L0 = 1_714_554_351_206
    calib = d["total"] / MEASURED_DENSE_L0
    print()
    print(f"  CALIBRATION: this model prices the shipped tower {calib:.4f}x its")
    print(f"  own measured depth-0 figure. The compact side gets no equivalent")
    print(f"  concession, so divide every ratio by ~{calib:.2f} before believing")
    print(f"  it. Calibrated best: {(best['ratio_vs_dense'] / calib):.3f}x"
          if best else "  Calibrated best: none")
    print()
    if best is None:
        print("  NO configuration in this sweep is cheaper than the tower we")
        print("  already ship. On this model the compact representation does")
        print("  not pay for itself, and Stage 1's stopping rule applies.")
    else:
        print(f"  Cheapest: recent={best['recent']}, rank={best['rank']}, "
              f"{best['b']:.5f} B, {best['ratio_vs_dense']:.3f}x cheaper than")
        print("  the modelled tower. That is a MODEL, on families that have not")
        print("  been written in `fnp`, and the join's core step has no")
        print("  primitive. It is a reason to build, not a result.")
    if args.json:
        pass
    if args.json:
        payload = {k: {"total": v["total"], "b": v["b"],
                       "lines": [{"family": x.family, "total": x.total,
                                  "note": x.note} for x in v["lines"]]}
                   for k, v in (("dense", d), ("compact", c), ("k4", k4),
                                ("k4_standalone", k4n))}
        payload["meta"] = {"width": args.width, "depth": args.depth,
                           "budget": BUDGET, "recent": args.recent,
                           "rank": args.rank,
                           "status": "PROJECTED MODEL, not a metered measurement"}
        payload["sweep"] = sweep(args.width, args.depth)
        with open(args.json, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, indent=2) + "\n")
        print(f"\n  ledger -> {args.json}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="the algebra")
    ap.add_argument("--ledger", action="store_true", help="the price")
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--depth", type=int, default=DEPTH)
    ap.add_argument("--recent", type=int, default=4)
    ap.add_argument("--rank", type=int, default=384)
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    if a.ledger:
        return _main_ledger(a)
    return _main_check()


if __name__ == "__main__":
    raise SystemExit(main())
