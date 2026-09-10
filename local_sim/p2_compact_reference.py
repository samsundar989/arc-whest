"""Executable reference for the Stage-1B compact all-history K3 state.

Each source has two shared transported legs ``A=Q FA`` and ``P=Q FP``. Its
third leg is reconstructed exactly as ``M=P diag(s)+3 A diag(e)+Z L.T``.
``Z L.T`` is an explicit residual; changing its rank is an approximation.

Old sources share ``Q``. A join uses their carried weighted Gram core, a
deterministic sketch, thin QR, factor rotation, and an exact carried-Gram
update. The dense reference keeps identical sources without compression.
The schedule starts empty, admits one source after each non-final layer, moves
at most one source across the age boundary per layer, and skips joins and D21
on the final layer. No source is deleted.

This is a label-free reference and cost instrument, not a submission estimator.
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
    return float(np.max(np.abs(a - b)) /
                 max(np.max(np.abs(a)), np.max(np.abs(b)), 1e-300))


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
        if need_d21:
            LA = 2.0 * AP * source.w2[None, :] + PP * source.e[None, :]
            LP = (source.A * source.A) * source.w2[None, :]
            LP += PP * (source.s[None, :] / 3.0) + (2.0 / 3.0) * MP
            d21 += (LA @ source.A.T + LP @ source.P.T
                    + ((PP @ source.L) @ source.Z.T) / 3.0)
    return d3, d21


def reconstruct(shared):
    if shared.Q is None:
        return []
    return [Source(shared.Q @ f.FA, shared.Q @ f.FP, f.Z, f.L,
                   f.s, f.e, f.w2, f.dA, f.dP, f.age) for f in shared.factors]


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
        if need_d21:
            LA = 2.0 * AP * f.w2[None, :] + PP * f.e[None, :]
            LP = (A * A) * f.w2[None, :]
            LP += PP * (f.s[None, :] / 3.0) + (2.0 / 3.0) * MP
            coord += LA @ f.FA.T + LP @ f.FP.T
            residual += ((PP @ f.L) @ f.Z.T) / 3.0
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
                           f.dA, f.dP, f.age) for f in shared.factors]
        gram = T @ shared.gram @ T.T
    else:
        rotated, gram = [], np.zeros((rank, rank))
    fj = Factors(FAj, FPj, source.Z, source.L, source.s, source.e, source.w2,
                 source.dA, source.dP, source.age)
    gram += (FAj * source.dA[None, :]) @ FAj.T
    gram += (FPj * source.dP[None, :]) @ FPj.T
    return Shared(Qn, rotated + [fj], gram)


def transport_dense(source, W, wick):
    return Source(W @ (wick[:, None] * source.A),
                  W @ (wick[:, None] * source.P),
                  W @ (wick[:, None] * source.Z), source.L,
                  source.s, source.e, source.w2, source.dA, source.dP,
                  source.age + 1)


def transport_shared(shared, W, wick):
    if shared.Q is None:
        return shared
    return Shared(W @ (wick[:, None] * shared.Q),
                  [Factors(f.FA, f.FP, W @ (wick[:, None] * f.Z), f.L,
                           f.s, f.e, f.w2, f.dA, f.dP, f.age + 1)
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
    return Source(A, P, Z, L, s, e, w2, dA, dP)


def make_birth(rng, n, residual_rank):
    """Synthetic latent birth; insertion applies the next layer's W."""
    s = make_source(rng, n, residual_rank)
    return Birth(s.A, s.Z, s.L, s.s, s.e, s.w2, s.dA, s.dP)


def insert_birth(birth, W):
    return Source(W @ birth.a, W.copy(), W @ birth.R, birth.L,
                  birth.s, birth.e, birth.w2, birth.dA, birth.dP, 0)


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
                            joiner.dP, joiner.age)
            pre = Shared(None if shared.Q is None else wick[:, None] * shared.Q,
                         shared.factors, shared.gram)
            shared = weighted_gram_join(pre, scaled, rank, W[:, :rank])
            shared = Shared(W @ shared.Q,
                            [Factors(f.FA, f.FP,
                                     W @ (wick[:, None] * f.Z), f.L,
                                     f.s, f.e, f.w2, f.dA, f.dP, f.age + 1)
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
        self.scratch[family] = max(self.scratch.get(family, 0), scratch_bytes)
        self.peak_bytes = max(self.peak_bytes, live_bytes + scratch_bytes)
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
    mk = lambda shape: fnp.array((rng.normal(size=shape) / math.sqrt(n)).astype(np.float32))
    Ws = [mk((n, n)) for _ in range(depth)]
    wicks = [fnp.array((0.5 + rng.random(n)).astype(np.float32)) for _ in range(depth)]
    young, shared, newborn = [], None, None
    bpe = 4
    with flops.BudgetContext(flop_budget=100 * BUDGET, quiet=True) as ctx:
        meter = Meter(ctx)
        for layer in range(depth):
            last = layer == depth - 1
            W, wick = Ws[layer], wicks[layer]
            joining = bool(not last and young and young[0][9] + 1 >= recent)
            if joining:
                src, young = young[0], young[1:]
                A, P, Z, L, s, e, w2, dA, dP, age = src
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
                    for FA, FP, Zi, Li, si, ei, w2i, dAi, dPi, ai in facts0:
                        FAr = meter.run("rotate_factor_A", lambda FA=FA: T @ FA)
                        FPr = meter.run("rotate_factor_P", lambda FP=FP: T @ FP)
                        facts.append((FAr, FPr, Zi, Li, si, ei, w2i,
                                      dAi, dPi, ai))
                    Sg = meter.run("rotate_carried_gram", lambda: T @ (S0 @ T.T))
                Sj = meter.run("carried_gram_update",
                               lambda: ((FAj * dA[None, :]) @ FAj.T
                                        + (FPj * dP[None, :]) @ FPj.T))
                Sg = meter.run("carried_gram_add", lambda: Sg + Sj)
                facts.append((FAj, FPj, Z, L, s, e, w2, dA, dP, age))
                Qout = meter.run("transport_shared_Q", lambda: W @ Qn)
                nf = []
                for FA, FP, Zi, Li, si, ei, w2i, dAi, dPi, ai in facts:
                    Zi = meter.run("transport_residual_Z",
                                   lambda Zi=Zi: W @ (wick[:, None] * Zi))
                    nf.append((FA, FP, Zi, Li, si, ei, w2i, dAi, dPi, ai + 1))
                shared = (Qout, nf, Sg)
            elif shared is not None:
                Q, facts, Sg = shared
                Q = meter.run("transport_shared_Q",
                              lambda: W @ (wick[:, None] * Q))
                nf = []
                for FA, FP, Z, L, s, e, w2, dA, dP, age in facts:
                    Z = meter.run("transport_residual_Z",
                                  lambda Z=Z: W @ (wick[:, None] * Z))
                    nf.append((FA, FP, Z, L, s, e, w2, dA, dP, age + 1))
                shared = (Q, nf, Sg)
            transported = []
            for src in young:
                A, P, Z, L, s, e, w2, dA, dP, age = src
                A1 = meter.run("transport_young_A", lambda A=A: W @ (wick[:, None] * A))
                P1 = meter.run("transport_young_P", lambda P=P: W @ (wick[:, None] * P))
                Z1 = meter.run("transport_residual_Z", lambda Z=Z: W @ (wick[:, None] * Z))
                transported.append((A1, P1, Z1, L, s, e, w2, dA, dP, age + 1))
            young = transported
            if newborn is not None:
                a, Rr, Lr, sb, eb, w2b, dAb, dPb = newborn
                A = meter.run("source_insert_A", lambda: W @ a)
                P = meter.run("source_insert_P_copy", lambda: fnp.copy(W))
                Z = meter.run("source_insert_Z", lambda: W @ Rr)
                young.append((A, P, Z, Lr, sb, eb, w2b, dAb, dPb, 0))
                newborn = None
            all_sources = [("young", x) for x in young]
            if shared is not None:
                Q, facts, _ = shared
                for f in facts:
                    FA, FP, Z, L, s, e, w2, dA, dP, age = f
                    A = meter.run("materialize_shared_A", lambda FA=FA: Q @ FA)
                    P = meter.run("materialize_shared_P", lambda FP=FP: Q @ FP)
                    all_sources.append(("shared", (A, P, Z, L, s, e, w2,
                                                    dA, dP, age, FA, FP)))
            coord = (meter.run("D21_coord_init",
                               lambda: fnp.zeros((n, rank), dtype=fnp.float32))
                     if shared is not None and not last else None)
            residual = (meter.run("D21_residual_init",
                                  lambda: fnp.zeros((n, n), dtype=fnp.float32))
                        if shared is not None and not last else None)
            for kind, src in all_sources:
                A, P, Z, L, s, e, w2 = src[:7]
                AP = meter.run("materialize_AP", lambda: A * P)
                PP = meter.run("materialize_PP", lambda: P * P)
                R = meter.run("materialize_residual_M", lambda: Z @ L.T)
                MP = meter.run("materialize_MP", lambda: (R * P
                               + PP * s[None, :] + 3.0 * AP * e[None, :]))
                LA = meter.run("materialize_LA", lambda: (2.0 * AP * w2[None, :]
                               + PP * e[None, :]))
                LP = meter.run("materialize_LP", lambda: ((A * A) * w2[None, :]
                               + PP * (s[None, :] / 3.0) + (2.0 / 3.0) * MP))
                meter.run("D3_contraction", lambda: (3.0 * A * AP * w2[None, :]
                                                       + MP * P).sum(axis=1))
                if not last:
                    if kind == "young":
                        meter.run("D21_young", lambda: (LA @ A.T + LP @ P.T
                                  + ((PP @ L) @ Z.T) / 3.0))
                    else:
                        FA, FP = src[10], src[11]
                        coord = meter.run("D21_shared_coords",
                                          lambda: coord + LA @ FA.T + LP @ FP.T)
                        residual = meter.run("D21_residual",
                                             lambda: residual + ((PP @ L) @ Z.T) / 3.0)
            if shared is not None and not last:
                meter.run("D21_shared_return", lambda: coord @ shared[0].T + residual)
            if not last:
                # The fixture supplies one layer's upstream factors at a time,
                # just as a full estimator would. Generating the label-free
                # fixture is outside the algorithmic bill; conversion and all
                # factor construction below are inside the context.
                srng = np.random.default_rng(seed + 1_000_003 * (layer + 1))
                shapes = ((n, n), (n, n), (n,), (n,), (n,), (n,))
                raw = tuple((srng.normal(size=shape) / math.sqrt(n)).astype(np.float32)
                            for shape in shapes)
                converted = []
                for value in raw:
                    converted.append(meter.run(
                        "source_input_cast", lambda value=value: fnp.array(value)))
                C0, Rres, w10, w20, s0, e0 = converted
                a = meter.run("source_a", lambda: w10[:, None] * C0)
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
                dA0 = meter.run("source_dA", lambda: 9.0 + w20 * w20 + 9.0 * e0 * e0)
                dP0 = meter.run("source_dP", lambda: 1.0 + s0 * s0)
                newborn = (a, Rr, Lr, s0, e0, w20, dA0, dP0)
        used = int(ctx.flops_used)
        family_sum = sum(meter.families.values())
        if family_sum != used:
            raise AssertionError("metered family ledger mismatch: %d != %d" %
                                 (family_sum, used))
        ops = len(ctx.op_log)
    persistent = bpe * (
        len(young) * (2 * n * n + 2 * n * residual_rank + 5 * n)
        + (0 if shared is None else n * rank + rank * rank
           + len(shared[1]) * (2 * rank * n + 2 * n * residual_rank + 5 * n)))
    births = max(depth - 1, 0)
    young_peak = min(recent + 1, births)
    old_peak = births - young_peak
    weights_bytes = bpe * depth * n * n
    wick_input_bytes = bpe * depth * n
    young_payload = young_peak * (
        2 * n * n + 2 * n * residual_rank + 5 * n)
    shared_payload = (0 if old_peak == 0 else
                      n * rank + rank * rank
                      + old_peak * (2 * rank * n + 2 * n * residual_rank + 5 * n))
    source_state_bytes = bpe * (young_payload + shared_payload)
    source_construction_scratch = (0 if births == 0 else bpe * (
        3 * n * n + 5 * n * residual_rank
        + residual_rank * residual_rank + 5 * n))
    join_scratch = (0 if old_peak == 0 else
                    bpe * (2 * n * n + 5 * n * rank + 3 * rank * rank))
    shared_dense_leg_stack = bpe * 2 * old_peak * n * n
    materialize_scratch = (0 if births == 0 else
                           shared_dense_leg_stack + bpe * 6 * n * n)
    contraction_scratch = (0 if births == 0 else
                           bpe * (3 * n * n + 2 * n * rank + n * residual_rank))
    peak_algorithm = weights_bytes + wick_input_bytes + source_state_bytes + max(
        source_construction_scratch, join_scratch,
        materialize_scratch + contraction_scratch)
    groups = {"source_construction": {}, "transport": {}, "join_and_rebase": {},
              "materialization": {}, "contractions": {}, "initialization": {}}
    for family, value in meter.families.items():
        if family.startswith("source_"):
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
        else:
            group = "initialization"
        groups[group][family] = value
    group_totals = {name: sum(values.values()) for name, values in groups.items()}
    if sum(group_totals.values()) != used:
        raise AssertionError("operation-group ledger mismatch")
    return {"kind": "label-free executable compact all-history ledger",
            "source_sha256": _source_sha256(),
            "shape": [n, depth], "recent": recent, "rank": rank,
            "residual_rank": residual_rank, "flops": used,
            "budget_fraction": used / BUDGET, "operations": ops,
            "families": meter.families, "family_calls": meter.calls,
            "family_groups": groups, "family_group_totals": group_totals,
            "memory": {"persistent_bytes_final_measured_schedule": persistent,
                       "external_weight_bytes": weights_bytes,
                       "external_wick_input_bytes": wick_input_bytes,
                       "persistent_source_state_peak_bytes": source_state_bytes,
                       "source_construction_scratch_bytes": source_construction_scratch,
                       "join_scratch_bytes": join_scratch,
                       "shared_dense_leg_stack_bytes": shared_dense_leg_stack,
                       "materialization_scratch_bytes": materialize_scratch,
                       "contraction_scratch_bytes": contraction_scratch,
                       "estimated_peak_bytes": peak_algorithm,
                       "single_array_max_bytes": bpe * n * n,
                       "scratch_by_metered_family": meter.scratch,
                       "basis": "float32 array payloads; Python/flopscope metadata excluded"},
            "casts": {"counted_flops": meter.families.get("source_input_cast", 0),
                      "calls": meter.calls.get("source_input_cast", 0),
                      "reason": "label-free NumPy fixture inputs are converted to float32 flopscope arrays inside the measured program"},
            "source_factors": {
                "latent_constructed": ["a", "R", "L", "dA", "dP"],
                "inserted_next_layer": ["A=W@a", "P=copy(W)", "Z=W@R"],
                "upstream_aliases": ["s", "e", "w2"],
                "shared_per_source": ["FA", "FP", "Z", "L", "s", "e", "w2", "dA", "dP"],
                "third_leg_identity": "M = P diag(s) + 3 A diag(e) + Z L.T"},
            "schedule": {"initial_sources": 0, "births": births,
                         "young_peak": young_peak, "shared_peak": old_peak,
                         "final_join": False, "final_D21": False},
            "parity": {"family_sum": family_sum, "context_flops": used,
                       "exact": family_sum == used}}


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
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--depth", type=int, default=DEPTH)
    ap.add_argument("--recent", type=int, default=4)
    ap.add_argument("--rank", type=int, default=384)
    ap.add_argument("--residual-rank", type=int, default=16)
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    result = (metered_program(a.width, a.depth, a.recent, a.rank,
                              a.residual_rank) if a.meter else checks())
    if a.json:
        write_json(a.json, result)
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
