"""Executable end-to-end reference for the Stage-1B compact architecture.

Each source has two shared transported legs ``A=Q FA`` and ``P=Q FP``. Its
third leg is reconstructed exactly as ``M=P diag(s)+3 A diag(e)+Z L.T``.
``Z L.T`` is an explicit residual; changing its rank is an approximation.

Old sources share ``Q``. A join uses their carried weighted Gram core, a
deterministic sketch, thin QR, factor rotation, and an exact carried-Gram
update. The dense reference keeps identical sources without compression.
The schedule starts empty, admits one source after each non-final layer, moves
at most one source across the age boundary per layer, and skips joins and D21
on the final layer. No source is deleted.

The metered path also runs a mean/full-covariance Gaussian ReLU backbone, an
analytic D3/D21 closure, the fixed-lam coherent K4 transport and D4/D22/D31
consumers, every induced structured K3 feed, source construction, and a final
mean-only readout.  It is a label-free research reference and cost instrument,
not a submission estimator.  Its omitted higher cross-cumulant nonlinear terms
are reported explicitly; no accuracy conclusion follows from this program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np

WIDTH, DEPTH, BUDGET = 1024, 16, 2 ** 41


def _source_sha256():
    with open(__file__, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return float("inf")
    scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), 1e-300)
    return float(np.max(np.abs(a / scale - b / scale)))


@dataclass
class Source:
    A: object
    P: object
    Z: object
    L: object
    s: object
    e: object
    w2: object
    dA: object
    dP: object
    c1: object
    c2: object
    Y: object
    age: int = 0


@dataclass
class Factors:
    FA: object
    FP: object
    Z: object
    L: object
    s: object
    e: object
    w2: object
    dA: object
    dP: object
    c1: object
    c2: object
    Y: object
    age: int


@dataclass
class Shared:
    Q: object | None = None
    factors: list[Factors] = field(default_factory=list)
    gram: object | None = None


@dataclass
class Birth:
    a: object
    R: object
    L: object
    s: object
    e: object
    w2: object
    dA: object
    dP: object
    c1: object
    c2: object
    y: object


def materialize(source):
    return (source.P * source.s[None, :]
            + 3.0 * source.A * source.e[None, :]
            + source.Z @ source.L.T)


def dense_slices(sources, need_d21=True):
    if not sources:
        return None, None
    n = sources[0].A.shape[0]
    d3 = np.zeros(n)
    d21 = np.zeros((n, n)) if need_d21 else None
    for source in sources:
        M = materialize(source)
        AP = source.A * source.P
        PP = source.P * source.P
        MP = M * source.P
        d3 += (3.0 * source.A * AP * source.w2[None, :] + MP * source.P).sum(axis=1)
        rf = AP @ source.c1 + PP @ source.c2
        d3 += rf * source.Y
        if need_d21:
            LA = 2.0 * AP * source.w2[None, :] + PP * source.e[None, :]
            LP = (source.A * source.A) * source.w2[None, :]
            LP += PP * (source.s[None, :] / 3.0) + (2.0 / 3.0) * MP
            LP += (source.A * source.c1[None, :]
                   + source.P * source.c2[None, :]) * (
                       (2.0 / 3.0) * source.Y[:, None])
            d21 += (LA @ source.A.T + LP @ source.P.T
                    + ((PP @ source.L) @ source.Z.T) / 3.0
                    + (rf[:, None] * source.Y[None, :]) / 3.0)
    return d3, d21


def reconstruct(shared):
    if shared.Q is None:
        return []
    return [Source(shared.Q @ f.FA, shared.Q @ f.FP, f.Z, f.L,
                   f.s, f.e, f.w2, f.dA, f.dP, f.c1, f.c2, f.Y,
                   f.age) for f in shared.factors]


def compact_slices(shared, need_d21=True):
    """Reassociate every Q-trailing product before one shared return."""
    if shared.Q is None or not shared.factors:
        return None, None
    n, r = shared.Q.shape
    d3 = np.zeros(n)
    coord = np.zeros((n, r)) if need_d21 else None
    residual = np.zeros((n, n)) if need_d21 else None
    for f in shared.factors:
        A, P = shared.Q @ f.FA, shared.Q @ f.FP
        H = f.FP * f.s[None, :] + 3.0 * f.FA * f.e[None, :]
        R = f.Z @ f.L.T
        M = shared.Q @ H + R
        AP, PP, MP = A * P, P * P, M * P
        d3 += (3.0 * A * AP * f.w2[None, :] + MP * P).sum(axis=1)
        rf = AP @ f.c1 + PP @ f.c2
        d3 += rf * f.Y
        if need_d21:
            LA = 2.0 * AP * f.w2[None, :] + PP * f.e[None, :]
            LP = (A * A) * f.w2[None, :]
            LP += PP * (f.s[None, :] / 3.0) + (2.0 / 3.0) * MP
            LP += (A * f.c1[None, :] + P * f.c2[None, :]) * (
                (2.0 / 3.0) * f.Y[:, None])
            coord += LA @ f.FA.T + LP @ f.FP.T
            residual += (((PP @ f.L) @ f.Z.T) / 3.0
                         + (rf[:, None] * f.Y[None, :]) / 3.0)
    return d3, None if not need_d21 else coord @ shared.Q.T + residual


def direct_weighted_gram(sources):
    if not sources:
        return None
    n = sources[0].A.shape[0]
    G = np.zeros((n, n))
    for s in sources:
        G += (s.A * s.dA[None, :]) @ s.A.T
        G += (s.P * s.dP[None, :]) @ s.P.T
    return G


def weighted_gram_join(shared, source, rank, omega):
    """Join one dense source using weighted-Gram sketch and thin QR."""
    n = source.A.shape[0]
    rank = min(int(rank), n, omega.shape[1])
    Y = source.A @ (source.dA[:, None] * (source.A.T @ omega))
    Y += source.P @ (source.dP[:, None] * (source.P.T @ omega))
    if shared.Q is not None:
        Y += shared.Q @ (shared.gram @ (shared.Q.T @ omega))
    Qn, _ = np.linalg.qr(Y, mode="reduced")
    Qn = Qn[:, :rank]
    FAj, FPj = Qn.T @ source.A, Qn.T @ source.P
    if shared.Q is not None:
        T = Qn.T @ shared.Q
        rotated = [Factors(T @ f.FA, T @ f.FP, f.Z, f.L, f.s, f.e, f.w2,
                           f.dA, f.dP, f.c1, f.c2, f.Y, f.age)
                   for f in shared.factors]
        gram = T @ shared.gram @ T.T
    else:
        rotated, gram = [], np.zeros((rank, rank))
    fj = Factors(FAj, FPj, source.Z, source.L, source.s, source.e, source.w2,
                 source.dA, source.dP, source.c1, source.c2, source.Y,
                 source.age)
    gram += (FAj * source.dA[None, :]) @ FAj.T
    gram += (FPj * source.dP[None, :]) @ FPj.T
    return Shared(Qn, rotated + [fj], gram)


def transport_dense(source, W, wick):
    return Source(W @ (wick[:, None] * source.A),
                  W @ (wick[:, None] * source.P),
                  W @ (wick[:, None] * source.Z), source.L,
                  source.s, source.e, source.w2, source.dA, source.dP,
                  source.c1, source.c2, W @ (wick * source.Y),
                  source.age + 1)


def transport_shared(shared, W, wick):
    if shared.Q is None:
        return shared
    return Shared(W @ (wick[:, None] * shared.Q),
                  [Factors(f.FA, f.FP, W @ (wick[:, None] * f.Z), f.L,
                           f.s, f.e, f.w2, f.dA, f.dP, f.c1, f.c2,
                           W @ (wick * f.Y), f.age + 1)
                   for f in shared.factors], shared.gram)


def make_source(rng, n, residual_rank):
    scale = 1.0 / math.sqrt(n)
    A = rng.normal(size=(n, n)) * scale
    P = rng.normal(size=(n, n)) * scale
    Z = rng.normal(size=(n, residual_rank)) * scale
    L = rng.normal(size=(n, residual_rank)) * scale
    s, e, w2 = (rng.normal(size=n) for _ in range(3))
    dA = 9.0 + w2 * w2 + 9.0 * e * e
    dP = 1.0 + s * s
    c1, c2, Y = (rng.normal(size=n) for _ in range(3))
    return Source(A, P, Z, L, s, e, w2, dA, dP, c1, c2, Y)


def make_birth(rng, n, residual_rank):
    """Synthetic latent birth; insertion applies the next layer's W."""
    s = make_source(rng, n, residual_rank)
    return Birth(s.A, s.Z, s.L, s.s, s.e, s.w2, s.dA, s.dP,
                 s.c1, s.c2, s.Y)


def insert_birth(birth, W):
    return Source(W @ birth.a, W.copy(), W @ birth.R, birth.L,
                  birth.s, birth.e, birth.w2, birth.dA, birth.dP,
                  birth.c1, birth.c2, W @ birth.y, 0)


def run_schedule(n=12, depth=7, recent=2, rank=None, residual_rank=3,
                 seed=20260910):
    """Run dense and compact histories side by side."""
    rank = n if rank is None else min(rank, n)
    rng = np.random.default_rng(seed)
    dense, young, shared = [], [], Shared()
    newborn = None
    rows = []
    for layer in range(depth):
        last = layer == depth - 1
        W = rng.normal(size=(n, n)) / math.sqrt(n)
        wick = 0.5 + rng.random(n)
        dense = [transport_dense(s, W, wick) for s in dense]
        joined = 0
        if not last and young and young[0].age + 1 >= recent:
            joiner, young = young[0], young[1:]
            scaled = Source(wick[:, None] * joiner.A,
                            wick[:, None] * joiner.P, joiner.Z, joiner.L,
                            joiner.s, joiner.e, joiner.w2, joiner.dA,
                            joiner.dP, joiner.c1, joiner.c2, joiner.Y,
                            joiner.age)
            pre = Shared(None if shared.Q is None else wick[:, None] * shared.Q,
                         shared.factors, shared.gram)
            shared = weighted_gram_join(pre, scaled, rank, W[:, :rank])
            shared = Shared(W @ shared.Q,
                            [Factors(f.FA, f.FP,
                                     W @ (wick[:, None] * f.Z), f.L,
                                     f.s, f.e, f.w2, f.dA, f.dP,
                                     f.c1, f.c2, W @ (wick * f.Y),
                                     f.age + 1)
                             for f in shared.factors], shared.gram)
            joined = 1
        else:
            shared = transport_shared(shared, W, wick)
        young = [transport_dense(s, W, wick) for s in young]
        if newborn is not None:
            inserted = insert_birth(newborn, W)
            dense.append(inserted)
            young.append(inserted)
            newborn = None
        d3, d21 = dense_slices(dense, need_d21=not last)
        y3, y21 = dense_slices(young, need_d21=not last)
        o3, o21 = compact_slices(shared, need_d21=not last)
        z3 = sum((x for x in (y3, o3) if x is not None), np.zeros(n))
        z21 = None if last else sum((x for x in (y21, o21) if x is not None),
                                    np.zeros((n, n)))
        rows.append({"layer": layer, "last": last, "joined": joined,
                     "n_dense": len(dense), "n_young": len(young),
                     "n_shared": len(shared.factors),
                     "d3_rel": 0.0 if d3 is None else _rel(d3, z3),
                     "d21_rel": None if last or d21 is None else _rel(d21, z21)})
        if not last:
            newborn = make_birth(rng, n, residual_rank)
    return rows


class Meter:
    def __init__(self, ctx):
        self.ctx = ctx
        self.families = {}
        self.calls = {}
        self.peak_bytes = 0
        self.scratch = {}

    def run(self, family, fn, live_bytes=0, scratch_bytes=0):
        before = int(self.ctx.flops_used)
        out = fn()
        delta = int(self.ctx.flops_used) - before
        self.families[family] = self.families.get(family, 0) + delta
        self.calls[family] = self.calls.get(family, 0) + 1
        result_bytes = int(getattr(out, "nbytes", 0))
        accounted_scratch = max(scratch_bytes, result_bytes)
        self.scratch[family] = max(self.scratch.get(family, 0),
                                   accounted_scratch)
        self.peak_bytes = max(self.peak_bytes, live_bytes + accounted_scratch)
        return out


def _real_flopscope():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    sys.path = [p for p in sys.path if os.path.abspath(p or ".") != here]
    for p in (here, root):
        if p not in sys.path:
            sys.path.append(p)
    import flopscope as flops
    import flopscope.numpy as fnp
    if "site-packages" not in os.path.abspath(flops.__file__):
        raise RuntimeError("real flopscope required, got %s" % flops.__file__)
    return flops, fnp


def metered_program(n, depth, recent, rank, residual_rank, seed=20260910):
    """Execute the actual schedule and return an exact metered family ledger."""
    flops, fnp = _real_flopscope()
    rank = min(rank, n)
    residual_rank = min(residual_rank, n)
    rng = np.random.default_rng(seed)
    # Match the grader-facing boundary: incoming weights are float64 and each
    # layer is explicitly converted before counted arithmetic.
    Ws = [rng.normal(size=(n, n)) / math.sqrt(n) for _ in range(depth)]
    young, shared, newborn = [], None, None
    bpe = 4
    oracle_trace = []
    with flops.BudgetContext(flop_budget=100 * BUDGET, quiet=True) as ctx:
        meter = Meter(ctx)
        mu = meter.run("backbone_mean_init",
                       lambda: fnp.zeros(n, dtype=fnp.float32))
        C = meter.run("backbone_cov_init",
                      lambda: fnp.eye(n, dtype=fnp.float32))
        wick_prev = meter.run("backbone_wick_init",
                              lambda: fnp.ones(n, dtype=fnp.float32))
        gcore = None
        outputs = []
        for layer in range(depth):
            last = layer == depth - 1
            W = meter.run("cast_weight_f32",
                          lambda layer=layer: fnp.array(Ws[layer], dtype=fnp.float32),
                          scratch_bytes=bpe * n * n)
            wick = wick_prev
            C_old = C
            mu_pre = meter.run("backbone_mean_linear", lambda: W @ mu)
            WC = meter.run("backbone_cov_linear_left", lambda: W @ C)
            C_pre = meter.run("backbone_cov_linear_right", lambda: WC @ W.T)
            var_pre = meter.run("backbone_variance_extract",
                                lambda: fnp.diagonal(C_pre))
            joining = bool(not last and young and young[0][12] + 1 >= recent)
            if joining:
                src, young = young[0], young[1:]
                A, P, Z, L, s, e, w2, dA, dP, c1, c2, Yfeed, age = src
                A = meter.run("joiner_wick_A", lambda: wick[:, None] * A)
                P = meter.run("joiner_wick_P", lambda: wick[:, None] * P)
                Om = W[:, :rank]
                live = bpe * (2 * n * n + 2 * n * residual_rank)
                Qp = None
                if shared is not None:
                    Qp = meter.run("shared_wick_Q",
                                   lambda: wick[:, None] * shared[0])
                def sketch():
                    Y = A @ (dA[:, None] * (A.T @ Om))
                    Y = Y + P @ (dP[:, None] * (P.T @ Om))
                    if shared is not None:
                        _Q0, _f0, S0 = shared
                        Y = Y + Qp @ (S0 @ (Qp.T @ Om))
                    return Y
                Y = meter.run("weighted_gram_sketch", sketch, live, bpe * n * rank)
                Qn = meter.run("thin_qr", lambda: fnp.linalg.qr(Y)[0], live,
                               bpe * (n * rank + rank * rank))[:, :rank]
                FAj = meter.run("project_joiner_A", lambda: Qn.T @ A)
                FPj = meter.run("project_joiner_P", lambda: Qn.T @ P)
                if shared is None:
                    facts = []
                    Sg = meter.run("carried_gram_init",
                                   lambda: fnp.zeros((rank, rank), dtype=fnp.float32))
                else:
                    Q0, facts0, S0 = shared
                    T = meter.run("factor_rotation_matrix", lambda: Qn.T @ Qp)
                    facts = []
                    for (FA, FP, Zi, Li, si, ei, w2i, dAi, dPi,
                         c1i, c2i, Yi, ai) in facts0:
                        FAr = meter.run("rotate_factor_A", lambda FA=FA: T @ FA)
                        FPr = meter.run("rotate_factor_P", lambda FP=FP: T @ FP)
                        facts.append((FAr, FPr, Zi, Li, si, ei, w2i,
                                      dAi, dPi, c1i, c2i, Yi, ai))
                    Sg = meter.run("rotate_carried_gram", lambda: T @ (S0 @ T.T))
                Sj = meter.run("carried_gram_update",
                               lambda: ((FAj * dA[None, :]) @ FAj.T
                                        + (FPj * dP[None, :]) @ FPj.T))
                Sg = meter.run("carried_gram_add", lambda: Sg + Sj)
                facts.append((FAj, FPj, Z, L, s, e, w2, dA, dP,
                              c1, c2, Yfeed, age))
                Qout = meter.run("transport_shared_Q", lambda: W @ Qn)
                nf = []
                for (FA, FP, Zi, Li, si, ei, w2i, dAi, dPi,
                     c1i, c2i, Yi, ai) in facts:
                    Zi = meter.run("transport_residual_Z",
                                   lambda Zi=Zi: W @ (wick[:, None] * Zi))
                    Yi = meter.run("transport_k4_birth_Y",
                                   lambda Yi=Yi: W @ (wick * Yi))
                    nf.append((FA, FP, Zi, Li, si, ei, w2i, dAi, dPi,
                               c1i, c2i, Yi, ai + 1))
                shared = (Qout, nf, Sg)
            elif shared is not None:
                Q, facts, Sg = shared
                Q = meter.run("transport_shared_Q",
                              lambda: W @ (wick[:, None] * Q))
                nf = []
                for (FA, FP, Z, L, s, e, w2, dA, dP,
                     c1, c2, Yfeed, age) in facts:
                    Z = meter.run("transport_residual_Z",
                                  lambda Z=Z: W @ (wick[:, None] * Z))
                    Yfeed = meter.run("transport_k4_birth_Y",
                                      lambda Yfeed=Yfeed: W @ (wick * Yfeed))
                    nf.append((FA, FP, Z, L, s, e, w2, dA, dP,
                               c1, c2, Yfeed, age + 1))
                shared = (Q, nf, Sg)
            transported = []
            for src in young:
                A, P, Z, L, s, e, w2, dA, dP, c1, c2, Yfeed, age = src
                A1 = meter.run("transport_young_A", lambda A=A: W @ (wick[:, None] * A))
                P1 = meter.run("transport_young_P", lambda P=P: W @ (wick[:, None] * P))
                Z1 = meter.run("transport_residual_Z", lambda Z=Z: W @ (wick[:, None] * Z))
                Y1 = meter.run("transport_k4_birth_Y",
                               lambda: W @ (wick * Yfeed))
                transported.append((A1, P1, Z1, L, s, e, w2, dA, dP,
                                    c1, c2, Y1, age + 1))
            young = transported
            if newborn is not None:
                a, Rr, Lr, sb, eb, w2b, dAb, dPb, c1b, c2b, yb = newborn
                A = meter.run("source_insert_A", lambda: W @ a)
                P = meter.run("source_insert_P_copy", lambda: fnp.copy(W))
                Z = meter.run("source_insert_Z", lambda: W @ Rr)
                Yb = meter.run("source_insert_k4_birth_Y", lambda: W @ yb)
                young.append((A, P, Z, Lr, sb, eb, w2b, dAb, dPb,
                              c1b, c2b, Yb, 0))
                newborn = None
            all_sources = [("young", x) for x in young]
            if shared is not None:
                Q, facts, _ = shared
                for f in facts:
                    (FA, FP, Z, L, s, e, w2, dA, dP,
                     c1, c2, Yfeed, age) = f
                    A = meter.run("materialize_shared_A", lambda FA=FA: Q @ FA)
                    P = meter.run("materialize_shared_P", lambda FP=FP: Q @ FP)
                    all_sources.append(("shared", (A, P, Z, L, s, e, w2,
                                                    dA, dP, c1, c2, Yfeed,
                                                    age, FA, FP)))
            coord = (meter.run("D21_coord_init",
                               lambda: fnp.zeros((n, rank), dtype=fnp.float32))
                     if shared is not None and not last else None)
            residual = (meter.run("D21_residual_init",
                                  lambda: fnp.zeros((n, n), dtype=fnp.float32))
                        if shared is not None and not last else None)
            D3_total = meter.run("history_D3_init",
                                 lambda: fnp.zeros(n, dtype=fnp.float32))
            D21_total = (meter.run("history_D21_init",
                                   lambda: fnp.zeros((n, n), dtype=fnp.float32))
                         if not last else None)
            for kind, src in all_sources:
                A, P, Z, L, s, e, w2, _dA, _dP, c1, c2, Yfeed = src[:12]
                AP = meter.run("materialize_AP", lambda: A * P)
                PP = meter.run("materialize_PP", lambda: P * P)
                R = meter.run("materialize_residual_M", lambda: Z @ L.T)
                MP = meter.run("materialize_MP", lambda: (R * P
                               + PP * s[None, :] + 3.0 * AP * e[None, :]))
                LA = meter.run("materialize_LA", lambda: (2.0 * AP * w2[None, :]
                               + PP * e[None, :]))
                LP = meter.run("materialize_LP", lambda: ((A * A) * w2[None, :]
                               + PP * (s[None, :] / 3.0) + (2.0 / 3.0) * MP))
                RF = meter.run("materialize_k4_birth_R",
                               lambda: AP @ c1 + PP @ c2)
                LP = meter.run("materialize_k4_birth_LP", lambda: LP
                               + (A * c1[None, :] + P * c2[None, :])
                               * ((2.0 / 3.0) * Yfeed[:, None]))
                d3part = meter.run("D3_contraction", lambda: (
                    (3.0 * A * AP * w2[None, :] + MP * P).sum(axis=1)
                    + RF * Yfeed))
                D3_total = meter.run("history_D3_add",
                                     lambda: D3_total + d3part)
                if not last:
                    if kind == "young":
                        d21part = meter.run("D21_young", lambda: (
                            LA @ A.T + LP @ P.T
                            + ((PP @ L) @ Z.T) / 3.0
                            + (RF[:, None] * Yfeed[None, :]) / 3.0))
                        D21_total = meter.run("history_D21_add",
                                             lambda: D21_total + d21part)
                    else:
                        FA, FP = src[13], src[14]
                        coord = meter.run("D21_shared_coords",
                                          lambda: coord + LA @ FA.T + LP @ FP.T)
                        residual = meter.run("D21_residual",
                                             lambda: residual + ((PP @ L) @ Z.T) / 3.0
                                             + (RF[:, None] * Yfeed[None, :]) / 3.0)
            if shared is not None and not last:
                d21old = meter.run("D21_shared_return",
                                   lambda: coord @ shared[0].T + residual)
                D21_total = meter.run("history_D21_add",
                                      lambda: D21_total + d21old)

            # Preserve the actual tuple-program state for a separately executed
            # dense oracle.  This is enabled only at tiny full rank, so oracle
            # arithmetic is outside the counted reference and cannot affect its
            # output or ledger.
            if n <= 16 and rank == n:
                oracle_trace.append((list(all_sources), D3_total, D21_total, last))

            # Analytic ReLU closure.  This is deliberately coefficient-free:
            # derivatives of the Gaussian ReLU mean supply the Wick factors.
            safe_var = meter.run("nonlinear_safe_variance",
                                 lambda: fnp.maximum(var_pre, 1.0e-8))
            sigma = meter.run("nonlinear_sigma", lambda: fnp.sqrt(safe_var))
            alpha = meter.run("nonlinear_alpha", lambda: mu_pre / sigma)
            phi64 = meter.run("nonlinear_phi",
                              lambda: flops.stats.norm.pdf(alpha))
            phi = meter.run("cast_phi_f32", lambda: phi64.astype(fnp.float32))
            Phi64 = meter.run("nonlinear_Phi",
                              lambda: flops.stats.norm.cdf(alpha))
            Phi = meter.run("cast_Phi_f32", lambda: Phi64.astype(fnp.float32))
            w1 = Phi
            w2now = meter.run("nonlinear_w2", lambda: phi / sigma)
            w3 = meter.run("nonlinear_w3", lambda: -alpha * phi / safe_var)
            w4 = meter.run("nonlinear_w4",
                           lambda: (alpha * alpha - 1.0) * phi / (safe_var * sigma))
            base = meter.run("nonlinear_gaussian_mean",
                             lambda: sigma * phi + mu_pre * Phi)
            C_off = meter.run("backbone_cov_offdiag",
                              lambda: C_pre - fnp.diag(var_pre))
            if gcore is None:
                dG = meter.run("k4_transport_initial",
                               lambda: fnp.zeros(n, dtype=fnp.float32))
            else:
                WW = meter.run("k4_transport_square", lambda: W * W)
                old_var = meter.run("k4_old_variance",
                                    lambda: fnp.diagonal(C_old))
                dG = meter.run("k4_transport_core", lambda: (
                    WW @ (gcore - (1.0 / 3.0) * old_var)
                    + (1.0 / 3.0) * var_pre))
            D4 = meter.run("k4_consumer_D4", lambda: 2.0 * dG)
            D22 = meter.run("k4_consumer_D22", lambda: (
                (dG[:, None] + dG[None, :]) / 3.0))
            D31 = meter.run("k4_consumer_D31", lambda: C_off / 3.0)
            mean3 = meter.run("nonlinear_mean_k3",
                              lambda: D3_total * w3 / 6.0)
            mean4 = meter.run("nonlinear_mean_k4",
                              lambda: D4 * w4 / 24.0)
            mu = meter.run("nonlinear_mean_add", lambda: base + mean3 + mean4)
            outputs.append(mu)
            if last:
                continue

            e2 = meter.run("nonlinear_gaussian_e2", lambda: (
                (safe_var + mu_pre * mu_pre) * Phi + mu_pre * sigma * phi))
            var_out = meter.run("nonlinear_gaussian_var",
                                lambda: fnp.maximum(e2 - base * base, 1.0e-8))
            K11 = meter.run("backbone_cov_delta2", lambda: (
                C_off * (w1[:, None] * w1[None, :])
                + 0.5 * (C_off * C_off)
                * (w2now[:, None] * w2now[None, :])))
            K11 = meter.run("k3_consumer_cov", lambda: K11 + 0.25 * (
                D21_total * (w2now[:, None] * w1[None, :])
                + D21_total.T * (w1[:, None] * w2now[None, :])))
            K11 = meter.run("k4_consumer_cov", lambda: K11
                            + D22 * (w2now[:, None] * w2now[None, :]) / 4.0
                            + (D31 + D31.T)
                            * (w3[:, None] * w1[None, :]) / 12.0)
            off_mask = meter.run("backbone_off_mask",
                                 lambda: fnp.ones((n, n), dtype=fnp.float32)
                                 - fnp.eye(n, dtype=fnp.float32))
            C = meter.run("backbone_cov_assemble",
                          lambda: fnp.diag(var_out) + K11 * off_mask)

            # Exact one-dimensional Gaussian ReLU moments seed the diagonal
            # K3/K4 closure; cross-neuron K21 uses second-order delta algebra.
            e3 = meter.run("nonlinear_gaussian_e3", lambda: (
                (mu_pre * mu_pre * mu_pre + 3.0 * mu_pre * safe_var) * Phi
                + (mu_pre * mu_pre * sigma + 2.0 * sigma * safe_var) * phi))
            k3v = meter.run("nonlinear_gaussian_k3",
                            lambda: e3 - 3.0 * base * e2 + 2.0 * base * base * base)
            e4 = meter.run("nonlinear_gaussian_e4", lambda: (
                (mu_pre**4 + 6.0 * mu_pre * mu_pre * safe_var
                 + 3.0 * safe_var * safe_var) * Phi
                + (mu_pre**3 * sigma + 5.0 * mu_pre * sigma * safe_var) * phi))
            k4v = meter.run("nonlinear_gaussian_k4", lambda: (
                e4 - 4.0 * base * e3 + 6.0 * base * base * e2
                - 3.0 * base**4 + D4 * w1**4))
            k22 = meter.run("k4_project_K22", lambda: (
                ((k4v[:, None] + k4v[None, :]) / 6.0
                 + D22 * (w1[:, None] * w1[None, :])) * off_mask))
            h4 = meter.run("k4_project_trace",
                           lambda: k4v + k22.sum(axis=1))
            gcore = meter.run("k4_project_core", lambda: (
                (6.0 / (n + 4.0)) * h4
                - (3.0 / ((n + 2.0) * (n + 4.0))) * h4.sum()) / 2.0)
            wick_prev = w1

            if not last:
                a = meter.run("source_a", lambda: w1[:, None] * C_off)
                local21 = meter.run("source_local_K21", lambda: (
                    (w1[:, None] * w1[:, None]) * w2now[None, :] * (C_off * C_off)
                    + 2.0 * (w2now * w1 * var_pre)[:, None]
                    * w1[None, :] * C_off))
                old21 = meter.run("source_transport_K21", lambda: (
                    (w1[:, None] * w1[:, None]) * D21_total * w1[None, :]))
                w1sq = meter.run("k4_birth_w1sq", lambda: w1 * w1)
                yb = meter.run("k4_birth_y", lambda: 0.5 * w2now)
                dgw = meter.run("k4_birth_dgw", lambda: w1sq * dG)
                ub = meter.run("k4_birth_u", lambda: yb * dG)
                c1b = meter.run("k4_birth_c1", lambda: w1 / 3.0)
                c2b = dgw
                feed21 = meter.run("k4_birth_D21", lambda: (
                    dgw[:, None] * yb[None, :] / 3.0
                    + (2.0 / 3.0) * yb[:, None]
                    * (a * c1b[None, :])
                    + w1sq[:, None] * ub[None, :] / 3.0))
                feed3 = meter.run("k4_birth_D3",
                                  lambda: dgw * yb + ub * w1sq)
                rep21 = meter.run("source_repeated_K21",
                                  lambda: old21 + (a * a) * w2now[None, :] + feed21)
                K21 = meter.run("source_K21_closure",
                                lambda: old21 + local21 + feed21)
                s0 = meter.run("source_s", lambda: k3v - D3_total * (w1**3) - feed3)
                e0 = meter.run("source_e", lambda: w2now - 2.0 * mu * w1)
                Ssep = meter.run("source_separable_K21", lambda: (
                    e0[:, None] * C_off * w1[None, :]))
                Rres = meter.run("source_residual", lambda: K21 - rep21 - Ssep)
                Om = W[:, :residual_rank]
                Y = meter.run("source_residual_sketch", lambda: Rres @ Om)
                Qr = meter.run("source_residual_qr1",
                               lambda: fnp.linalg.qr(Y)[0])[:, :residual_rank]
                Y = meter.run("source_residual_power",
                              lambda: Rres @ (Rres.T @ Qr))
                Qr = meter.run("source_residual_qr2",
                               lambda: fnp.linalg.qr(Y)[0])[:, :residual_rank]
                Bm = meter.run("source_residual_project", lambda: Qr.T @ Rres)
                Rr = meter.run("source_R_scale", lambda: Bm.T * 3.0)
                Lr = meter.run("source_L_alias", lambda: Qr)
                dA0 = meter.run("source_dA", lambda: 9.0 + w2now * w2now + 9.0 * e0 * e0)
                dP0 = meter.run("source_dP", lambda: 1.0 + s0 * s0)
                newborn = (a, Rr, Lr, s0, e0, w2now, dA0, dP0,
                            c1b, c2b, yb)
        prediction = meter.run("output_stack", lambda: fnp.stack(outputs, axis=0))
        used = int(ctx.flops_used)
        family_sum = sum(meter.families.values())
        if family_sum != used:
            raise AssertionError("metered family ledger mismatch: %d != %d" %
                                 (family_sum, used))
        ops = len(ctx.op_log)
    oracle_d3_rel = 0.0
    oracle_d21_rel = 0.0
    for source_rows, got_d3, got_d21, was_last in oracle_trace:
        dense_sources = []
        for _kind, row in source_rows:
            dense_sources.append(Source(*[np.asarray(x) for x in row[:12]],
                                        age=int(row[12])))
        want_d3, want_d21 = dense_slices(dense_sources, need_d21=not was_last)
        want_d3 = np.zeros(n) if want_d3 is None else want_d3
        oracle_d3_rel = max(oracle_d3_rel, _rel(np.asarray(got_d3), want_d3))
        if not was_last:
            want_d21 = np.zeros((n, n)) if want_d21 is None else want_d21
            oracle_d21_rel = max(oracle_d21_rel,
                                 _rel(np.asarray(got_d21), want_d21))
    if oracle_d3_rel > 2e-5 or oracle_d21_rel > 2e-5:
        raise AssertionError("metered tuple/dense oracle mismatch: %.3e %.3e" %
                             (oracle_d3_rel, oracle_d21_rel))

    retained = len(young) + (0 if shared is None else len(shared[1]))
    if retained != max(depth - 1, 0):
        raise AssertionError("all-history lifecycle lost a source: %d != %d" %
                             (retained, max(depth - 1, 0)))
    prediction_np = np.asarray(prediction)
    if prediction_np.shape != (depth, n) or not np.isfinite(prediction_np).all():
        raise AssertionError("invalid end-to-end output")
    persistent = bpe * (
        len(young) * (2 * n * n + 2 * n * residual_rank + 8 * n)
        + (0 if shared is None else n * rank + rank * rank
           + len(shared[1]) * (2 * rank * n + 2 * n * residual_rank + 8 * n)))
    births = max(depth - 1, 0)
    young_peak = min(recent + 1, births)
    old_peak = births - young_peak
    weights_bytes = 8 * depth * n * n
    weight_cast_scratch = bpe * n * n
    backbone_state_bytes = bpe * (2 * n * n + (depth + 12) * n)
    young_payload = young_peak * (
        2 * n * n + 2 * n * residual_rank + 8 * n)
    shared_payload = (0 if old_peak == 0 else
                      n * rank + rank * rank
                      + old_peak * (2 * rank * n + 2 * n * residual_rank + 8 * n))
    source_state_bytes = bpe * (young_payload + shared_payload)
    source_construction_scratch = (0 if births == 0 else bpe * (
        9 * n * n + 5 * n * residual_rank
        + residual_rank * residual_rank + 12 * n))
    join_scratch = (0 if old_peak == 0 else
                    bpe * (2 * n * n + 5 * n * rank + 3 * rank * rank))
    shared_dense_leg_stack = bpe * 2 * old_peak * n * n
    materialize_scratch = (0 if births == 0 else
                           shared_dense_leg_stack + bpe * 6 * n * n)
    contraction_scratch = (0 if births == 0 else
                           bpe * (3 * n * n + 2 * n * rank + n * residual_rank))
    peak_algorithm = weights_bytes + backbone_state_bytes + source_state_bytes + max(
        weight_cast_scratch, source_construction_scratch, join_scratch,
        materialize_scratch + contraction_scratch)
    groups = {"casts": {}, "backbone": {}, "nonlinear": {}, "k4": {},
              "source_construction": {}, "transport": {}, "join_and_rebase": {},
              "materialization": {}, "contractions": {}, "output": {},
              "initialization": {}}
    for family, value in meter.families.items():
        if family.startswith("cast_"):
            group = "casts"
        elif family.startswith("backbone_") or family.startswith("k3_consumer_"):
            group = "backbone"
        elif family.startswith("nonlinear_"):
            group = "nonlinear"
        elif family.startswith("k4_"):
            group = "k4"
        elif family.startswith("source_"):
            group = "source_construction"
        elif family.startswith("transport_"):
            group = "transport"
        elif family.startswith(("weighted_", "thin_", "project_joiner_",
                                "factor_rotation_", "rotate_", "carried_",
                                "joiner_", "shared_wick_")):
            group = "join_and_rebase"
        elif family.startswith("materialize_"):
            group = "materialization"
        elif family.startswith(("D3_", "D21_")) and not family.endswith("_init"):
            group = "contractions"
        elif family.startswith("output_"):
            group = "output"
        else:
            group = "initialization"
        groups[group][family] = value
    group_totals = {name: sum(values.values()) for name, values in groups.items()}
    if sum(group_totals.values()) != used:
        raise AssertionError("operation-group ledger mismatch")
    return {"kind": "label-free end-to-end compact all-history reference ledger",
            "evidence_note": ("Counted-cost and lifecycle evidence for this "
                              "research reference only; it makes no accuracy or "
                              "competitiveness claim."),
            "source_sha256": _source_sha256(),
            "shape": [n, depth], "recent": recent, "rank": rank,
            "residual_rank": residual_rank, "flops": used,
            "budget_fraction": used / BUDGET, "operations": ops,
            "operations_kind": "atomic flopscope op_log entries",
            "families": meter.families, "family_calls": meter.calls,
            "family_calls_kind": "explicit program-family wrapper invocations",
            "family_groups": groups, "family_group_totals": group_totals,
            "memory": {"persistent_source_bytes_final": persistent,
                       "external_weight_bytes": weights_bytes,
                       "weight_cast_scratch_bytes": weight_cast_scratch,
                       "backbone_state_and_output_bytes": backbone_state_bytes,
                       "persistent_source_state_peak_bytes": source_state_bytes,
                       "source_construction_scratch_bytes": source_construction_scratch,
                       "join_scratch_bytes": join_scratch,
                       "shared_dense_leg_stack_bytes": shared_dense_leg_stack,
                       "materialization_scratch_bytes": materialize_scratch,
                       "contraction_scratch_bytes": contraction_scratch,
                       "estimated_peak_bytes": peak_algorithm,
                       "single_array_max_bytes": bpe * n * n,
                       "result_or_declared_scratch_by_family": meter.scratch,
                       "basis": ("array payloads only; each family records at least its "
                                 "result allocation, with larger explicitly declared "
                                 "temporary workspace where modeled; Python/flopscope "
                                 "metadata excluded")},
            "casts": {"counted_flops": sum(
                          value for name, value in meter.families.items()
                          if name.startswith("cast_")),
                      "calls": sum(value for name, value in meter.calls.items()
                                   if name.startswith("cast_")),
                      "reason": "incoming float64 weights and normal CDF/PDF outputs are explicitly cast to float32"},
            "source_factors": {
                "latent_constructed": ["a", "R", "L", "s", "e", "w2", "dA", "dP", "c1", "c2", "y"],
                "inserted_next_layer": ["A=W@a", "P=copy(W)", "Z=W@R", "Y=W@y"],
                "shared_per_source": ["FA", "FP", "Z", "L", "s", "e", "w2", "dA", "dP", "c1", "c2", "Y"],
                "third_leg_identity": "M = P diag(s) + 3 A diag(e) + Z L.T"},
            "schedule": {"initial_sources": 0, "births": births,
                         "young_peak": young_peak, "shared_peak": old_peak,
                         "final_join": False, "final_D21": False},
            "lifecycle": {"output_shape": list(prediction_np.shape),
                          "all_finite": bool(np.isfinite(prediction_np).all()),
                          "output_sha256": hashlib.sha256(
                              prediction_np.astype(np.float32, copy=False).tobytes()).hexdigest(),
                          "output_max_abs": float(np.max(np.abs(prediction_np))),
                          "retained_sources": retained,
                          "all_births_retained": retained == births,
                          "initial_layer_has_history": False,
                          "final_layer_mean_only": True},
            "scope": {
                "exact_algebra": ["two-leg M reassociation", "explicit residual contractions", "shared-Q transport and factor rotation", "full-rank tiny old-source parity"],
                "compression": ["residual rank %d" % residual_rank, "old-source shared rank %d" % rank],
                "term_deletion": "the analytic reference closure omits higher cross-cumulant nonlinear terms outside D3/D21 and represented D4/D22/D31",
                "coefficient_policy": "fixed lam=1/3 structural fixture; no fitted coefficients or labels"},
            "parity": {"family_sum": family_sum, "context_flops": used,
                       "exact": family_sum == used,
                       "tiny_dense_oracle_executed": bool(oracle_trace),
                       "tiny_dense_D3_max_relative": oracle_d3_rel,
                       "tiny_dense_D21_max_relative": oracle_d21_rel}}


def lifecycle_check(n=12, depth=7, recent=2, rank=12, residual_rank=3):
    """Run twice and make deterministic lifecycle/cost drift fatal."""
    first = metered_program(n, depth, recent, rank, residual_rank)
    second = metered_program(n, depth, recent, rank, residual_rank)
    checks_ = {
        "deterministic_output": (first["lifecycle"]["output_sha256"]
                                 == second["lifecycle"]["output_sha256"]),
        "deterministic_flops": first["flops"] == second["flops"],
        "deterministic_calls": first["family_calls"] == second["family_calls"],
        "family_context_parity": first["parity"]["exact"],
        "tiny_dense_oracle": (not first["parity"]["tiny_dense_oracle_executed"]
                              or (first["parity"]["tiny_dense_D3_max_relative"] <= 2e-5
                                  and first["parity"]["tiny_dense_D21_max_relative"] <= 2e-5)),
        "finite_shape": (first["lifecycle"]["all_finite"]
                         and first["lifecycle"]["output_shape"] == [depth, n]),
        "all_history_retained": first["lifecycle"]["all_births_retained"],
        "initial_empty": not first["lifecycle"]["initial_layer_has_history"],
        "final_mean_only": first["lifecycle"]["final_layer_mean_only"],
    }
    if not all(checks_.values()):
        raise AssertionError("lifecycle failure: %r" % checks_)
    return {"kind": "label-free end-to-end compact lifecycle",
            "source_sha256": _source_sha256(), "shape": [n, depth],
            "checks": checks_, "pass": True,
            "output_sha256": first["lifecycle"]["output_sha256"],
            "flops": first["flops"], "operations": first["operations"]}


def checks():
    rows = run_schedule()
    out = []
    for row in rows:
        ok = row["d3_rel"] < 2e-11 and (row["d21_rel"] is None or row["d21_rel"] < 2e-11)
        out.append({"check": "layer-%d exact-full-rank" % row["layer"],
                    "pass": ok, **row})
    rng = np.random.default_rng(17)
    n = 9
    shared = Shared()
    srcs = []
    for j in range(3):
        s = make_source(rng, n, 2)
        srcs.append(s)
        shared = weighted_gram_join(shared, s, n, np.eye(n))
        e = _rel(direct_weighted_gram(srcs), shared.Q @ shared.gram @ shared.Q.T)
        out.append({"check": "carried-gram-%d" % j, "pass": e < 2e-11,
                    "error": e})
    s = make_source(rng, n, 3)
    full = materialize(s)
    nores = s.P * s.s[None, :] + 3.0 * s.A * s.e[None, :]
    e = _rel(full, nores)
    out.append({"check": "residual-is-explicit", "pass": e > 1e-4,
                "error": e})
    return {"kind": "label-free compact reference conformance",
            "source_sha256": _source_sha256(),
            "checks": out, "n_checks": len(out),
            "n_failed": sum(not x["pass"] for x in out),
            "pass": all(x["pass"] for x in out)}


def write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--meter", action="store_true")
    ap.add_argument("--lifecycle", action="store_true")
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--depth", type=int, default=DEPTH)
    ap.add_argument("--recent", type=int, default=4)
    ap.add_argument("--rank", type=int, default=384)
    ap.add_argument("--residual-rank", type=int, default=16)
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    result = (lifecycle_check(a.width, a.depth, a.recent, a.rank,
                              a.residual_rank) if a.lifecycle else
              metered_program(a.width, a.depth, a.recent, a.rank,
                              a.residual_rank) if a.meter else checks())
    if a.json:
        write_json(a.json, result)
    if a.lifecycle:
        for name, ok in result["checks"].items():
            print("%-36s %s" % (name, "PASS" if ok else "FAIL"))
        print("lifecycle deterministic at %d FLOPs" % result["flops"])
        return 0
    if a.meter:
        print("metered %d FLOPs (%.6f B), %d operations" %
              (result["flops"], result["budget_fraction"], result["operations"]))
        for k, v in sorted(result["families"].items()):
            print("  %-32s %16d" % (k, v))
        print("family/context parity: %s" % result["parity"]["exact"])
        return 0
    for row in result["checks"]:
        print("%-36s %s" % (row["check"], "PASS" if row["pass"] else "FAIL"))
    print("%d/%d passed" %
          (result["n_checks"] - result["n_failed"], result["n_checks"]))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
