"""N1 merged-old-tier CYCLE PRICE at n = 1024 under the real flopscope meter.

COST ONLY. No labels, no truth, no seed block, no estimator edit. Stage-3 N1
economic precondition (`docs/research/2026-09-16_architecture_audit.md` "N1",
`docs/research/2026-09-16_strategy_and_implementation_plan.md` "Two research
directions"): before any accuracy/rank sweep, price a COMPLETE old-history cycle
-- join + basis update + transport + D3/D21 query -- billing every operation.

Representation priced. Lineages older than the age gate are merged into one
symmetric Tucker core G (r,r,r) in a transported basis V (n,r):
    T_old ~= G(V, V, V)
Every leg of every lineage is LEFT-multiplied by the same per-layer map
(W^T, then diag(w1)), so V is transported and G is unchanged between rebases.

Age-gate convention used everywhere here: the recent tier holds at most `a`
dense lineages (v31 run with `gens = a`, so a lineage is queried densely at `a`
layers). At the end of layer l the lineage born at l - a is JOINED instead of
dropped. With depth 16 (layers 0..15, births at 0..14):
    joins           l = a .. 14         (15 - a, the first is the basis init)
    old D21 queries l = a+1 .. 14       (14 - a)
    old D3-only     l = 15              (1)
    transports      l = a+1 .. 15       (15 - a; Wick scale on the 14 - a non-last)
The other reading ("merged when it reaches age a") is this table's a - 1 row.

Tensor of one lineage with carried mode P, legs A_L, Z (n,r0), L (n,r0),
column scales sb, eb, w2 (all (n,)):
    A_D = P d(sb) + 3 A_L d(eb) + Z L^T,  C_L = A_L d(w2)
    T   = (1/3) sum_3 placements A_D (x) P (x) P + (1/2) sum_6 perms A_L (x) P (x) C_L
Because C_L = A_L d(w2), the L block collapses to 3 placements of
H_ikl = sum_j A_ij A_kj w2_j P_lj, so T = sym3(E/3 + H) with
E_ikl = sum_j P_ij P_kj A_D_lj and sym3(X) = X + X_(0,2,1) + X_(2,0,1).

    uv run python local_sim/n1_cycle_meter.py --selfcheck
    uv run python local_sim/n1_cycle_meter.py --out experiments/2026-09-16_stage3_n1/cycle_cost.json

# seed-ledger: no block named -- every array here is a price operand (values
# irrelevant to the bill) or a self-check operand at n = 12; nothing is scored.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import tracemalloc

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# local_sim must never shadow flopscope: strip it, never insert it.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]

import numpy as np  # noqa: E402  (lab script; the estimator never sees numpy)
import flopscope  # noqa: E402
import flopscope.numpy as fnp  # noqa: E402

if os.path.abspath(flopscope.__file__).startswith(_HERE + os.sep):
    raise SystemExit("REFUSED: flopscope resolved to the local shim")
try:
    flopscope.configure(symmetry_warnings=False)
except Exception:
    pass
import warnings  # noqa: E402
warnings.filterwarnings("ignore")

N, DEPTH, B = 1024, 16, 2 ** 41
RANKS = [96, 128, 150, 192, 256]
AGES = [2, 3, 4, 5]
R0 = 32                      # v31 _P2_RANK: the thin residual leg
PRICE_SEED = 7_331_024       # price operands only; not a registered block
V31 = "estimator_k3_v31.py"
V31_GRADED_CB = 0.3010


def _load_v31():
    path = os.path.join(_ROOT, V31)
    spec = importlib.util.spec_from_file_location("v31_ro", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with open(path, "rb") as fh:
        return mod, hashlib.sha256(fh.read()).hexdigest()


# ==========================================================================
# the operations being priced -- generic over dtype, used by the self-check
# AND by the meter, so the metered spelling is the verified spelling
# ==========================================================================

def transport(WT, V, col, mm):
    """V <- diag(w1) W^T V (col None at the last layer)."""
    X = mm(WT, V)
    return X if col is None else X * col


def rebase_einsum(V, G):
    Q, R = fnp.linalg.qr(V)
    return Q, fnp.einsum("abc,ia,jb,kc->ijk", G, R, R, R)


def rebase_tdot(V, G):
    Q, R = fnp.linalg.qr(V)
    X = fnp.tensordot(R, G, axes=([1], [0]))      # (i, b, c)
    X = fnp.tensordot(R, X, axes=([1], [1]))      # (j, i, c)
    X = fnp.tensordot(R, X, axes=([1], [2]))      # (k, j, i)
    return Q, X.transpose(2, 1, 0)


def query_einsum(G, V, off_mask, last):
    X = fnp.einsum("ijk,ai,aj->ak", G, V, V)       # (n, r): G(V_a, V_a, .)
    d3 = (X * V).sum(axis=1)
    if last:
        return d3, None
    return d3, (X @ V.T) * off_mask


def query_kr(G, V, off_mask, last):
    n, r = V.shape
    VV = fnp.reshape(V[:, :, None] * V[:, None, :], (n, r * r))
    X = VV @ fnp.reshape(G, (r * r, r))
    d3 = (X * V).sum(axis=1)
    if last:
        return d3, None
    return d3, (X @ V.T) * off_mask


def legs_ortho(V, P, AL, Z):
    VT = V.T
    return VT @ P, VT @ AL, VT @ Z


def legs_oblique(V, P, AL, Z):
    VT = V.T
    Vp = fnp.linalg.solve(VT @ V, VT)              # (V^T V)^-1 V^T, (r, n)
    return Vp @ P, Vp @ AL, Vp @ Z


def core_einsum(p, aL, z, L, sb, eb, w2):
    aD = p * sb + aL * (eb * 3.0) + z @ L.T
    E = fnp.einsum("ij,kj,lj->ikl", p, p, aD)
    H = fnp.einsum("ij,kj,lj->ikl", aL, aL, p * w2)
    X = E * (1.0 / 3.0) + H
    return X + X.transpose(0, 2, 1) + X.transpose(2, 0, 1)


def core_kr(p, aL, z, L, sb, eb, w2):
    r, n = p.shape
    aD = p * sb + aL * (eb * 3.0) + z @ L.T
    PP = fnp.reshape(p[:, None, :] * p[None, :, :], (r * r, n))
    AA = fnp.reshape(aL[:, None, :] * aL[None, :, :], (r * r, n))
    X = fnp.reshape(PP @ (aD.T * (1.0 / 3.0)) + AA @ (p * w2).T, (r, r, r))
    return X + X.transpose(0, 2, 1) + X.transpose(2, 0, 1)


def join_a(V, G, P, AL, Z, L, sb, eb, w2, legs, core):
    p, aL, z = legs(V, P, AL, Z)
    return G + core(p, aL, z, L, sb, eb, w2)


def init_basis(P, AL, OmP, OmA):
    Q, _R = fnp.linalg.qr(P @ OmP + AL @ OmA)
    return Q


def join_b(V, G, P, AL, Z, L, sb, eb, w2, OmP, OmA, r_out, core):
    """Sketch, enlarge to m = r + s, rotate+extend, add, HOSVD-truncate to r_out."""
    r = V.shape[1]
    Y = P @ OmP + AL @ OmA                              # (n, s) range finder
    U, Rf = fnp.linalg.qr(fnp.concatenate([V, Y], axis=1))
    Rv = Rf[:, :r]                                      # V = U Rv exactly
    m = Rf.shape[0]
    Gx = fnp.einsum("abc,ia,jb,kc->ijk", G, Rv, Rv, Rv)  # (m, m, m)
    p, aL, z = legs_ortho(U, P, AL, Z)
    Gx = Gx + core(p, aL, z, L, sb, eb, w2)
    Gu = fnp.reshape(Gx, (m, m * m))
    _ev, Ev = fnp.linalg.eigh(Gu @ Gu.T)                # symmetric core: one mode Gram
    E = Ev[:, m - r_out:]
    Vn = U @ E
    Gn = fnp.einsum("abc,ai,bj,ck->ijk", Gx, E, E, E)
    return Vn, Gn


# ==========================================================================
# self-check, float64, n = 12
# ==========================================================================

def _rel(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def _dense_lineage(P, AL, Z, L, sb, eb, w2):
    """The caller's DEFINITION, spelled out placement by placement in numpy."""
    AD = P * sb + 3.0 * AL * eb + Z @ L.T
    C = AL * w2
    ein = np.einsum
    TD = (ein("aj,bj,cj->abc", AD, P, P) + ein("aj,bj,cj->abc", P, AD, P)
          + ein("aj,bj,cj->abc", P, P, AD)) / 3.0
    F = [AL, P, C]
    TL = 0.0
    import itertools
    for s in itertools.permutations(range(3)):
        TL = TL + ein("aj,bj,cj->abc", F[s[0]], F[s[1]], F[s[2]])
    return TD + TL / 2.0


def selfcheck(v31):
    rng = np.random.default_rng(12)
    n, r0 = 12, 4
    f = lambda *s: rng.standard_normal(s)
    P, AL, W = f(n, n), f(n, n), f(n, n)
    Z, L = f(n, r0), f(n, r0)
    sb, eb, w2, col = f(n), f(n), f(n), f(n)
    off = 1.0 - np.eye(n)
    A = fnp.asarray
    out = {}

    T = _dense_lineage(P, AL, Z, L, sb, eb, w2)
    out["dense_tensor_symmetric"] = max(_rel(T.transpose(p), T) for p in
                                        [(0, 2, 1), (1, 0, 2), (2, 1, 0)])
    d3_T = np.einsum("aaa->a", T)
    d21_T = np.einsum("aab->ab", T) * off

    # (A) the definition agrees with v31's own slice code on a 1-lineage tower
    tw = v31._Tower(None, A(AL), A(P), A(w2), 1, A(Z), A(L[None]), A(sb), A(eb))
    v3 = np.asarray(v31._dslice3(tw))
    v21 = np.asarray(v31._dslice21(tw, A(off)))
    out["A_definition_vs_v31_dslice3"] = _rel(d3_T, v3)
    out["A_definition_vs_v31_dslice21"] = _rel(d21_T, v21)

    # (B) query spellings == dense einsum of G(V,V,V), random symmetric G, oblique V
    Gr = f(n, n, n)
    Gs = sum(Gr.transpose(p) for p in
             [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]) / 6.0
    V = f(n, n)
    TG = np.einsum("ijk,ai,bj,ck->abc", Gs, V, V, V)
    for name, q in (("einsum", query_einsum), ("kr", query_kr)):
        d3, d21 = q(A(Gs), A(V), A(off), False)
        d3l, _ = q(A(Gs), A(V), A(off), True)
        out[f"B_query_{name}_d3"] = _rel(d3, np.einsum("aaa->a", TG))
        out[f"B_query_{name}_d21"] = _rel(d21, np.einsum("aab->ab", TG) * off)
        out[f"B_query_{name}_d3_lastlayer"] = _rel(d3l, np.einsum("aaa->a", TG))

    # (C) join (a) at full rank reproduces the lineage tensor and v31's slices
    Qo = np.linalg.qr(f(n, n))[0]
    for lname, legs, Vb in (("ortho", legs_ortho, Qo), ("oblique", legs_oblique, f(n, n))):
        for cname, core in (("einsum", core_einsum), ("kr", core_kr)):
            G = join_a(A(Vb), A(np.zeros((n, n, n))), A(P), A(AL), A(Z), A(L),
                       A(sb), A(eb), A(w2), legs, core)
            TGj = np.einsum("ijk,ai,bj,ck->abc", np.asarray(G), Vb, Vb, Vb)
            d3, d21 = query_einsum(G, A(Vb), A(off), False)
            out[f"C_join_{lname}_{cname}_full_tensor"] = _rel(TGj, T)
            out[f"C_join_{lname}_{cname}_d3_vs_v31"] = _rel(d3, v3)
            out[f"C_join_{lname}_{cname}_d21_vs_v31"] = _rel(d21, v21)

    # (D) transport: (V, G) transported == v31 _advance + _wick_tower, then slices
    G = join_a(A(Qo), A(np.zeros((n, n, n))), A(P), A(AL), A(Z), A(L),
               A(sb), A(eb), A(w2), legs_ortho, core_einsum)
    Vt = transport(A(W.T), A(Qo), A(col[:, None]), lambda x, y: x @ y)
    # pre-advance v31 layout stores M for the OLDER k-1 blocks only, so carry our
    # lineage as the older block of a k=2 tower whose youngest block is all zero.
    zn, zr = np.zeros((n, n)), np.zeros((n, r0))
    pre = v31._Tower(None, A(np.concatenate([AL, zn], 1)), A(P),
                     A(np.concatenate([w2, w2])), 2,
                     A(np.concatenate([Z, zr], 1)), A(np.stack([L, zr])),
                     A(np.concatenate([sb, np.zeros(n)])), A(np.concatenate([eb, np.zeros(n)])))
    tw2 = v31._wick_tower(A(col[:, None]), v31._advance(A(W.T), pre))
    d3, d21 = query_einsum(G, Vt, A(off), False)
    out["D_transport_d3_vs_v31_advance_wick"] = _rel(d3, v31._dslice3(tw2))
    out["D_transport_d21_vs_v31_advance_wick"] = _rel(d21, v31._dslice21(tw2, A(off)))

    # (E) rebase: G'(Q,Q,Q) == G(V,V,V), both spellings, on the transported basis
    TV = np.einsum("ijk,ai,bj,ck->abc", np.asarray(G), np.asarray(Vt), np.asarray(Vt),
                   np.asarray(Vt))
    for name, rb in (("einsum", rebase_einsum), ("tdot", rebase_tdot)):
        Q, G2 = rb(Vt, G)
        Qn = np.asarray(Q)
        out[f"E_rebase_{name}"] = _rel(np.einsum("ijk,ai,bj,ck->abc", np.asarray(G2), Qn, Qn, Qn), TV)

    # (F) join (b): old (V0 orthonormal rank 6, G0), new lineage, s = 6 -> m = 12 = n
    r_old, s = 6, 6
    V0 = np.linalg.qr(f(n, r_old))[0]
    Gr = f(r_old, r_old, r_old)
    G0 = sum(Gr.transpose(p) for p in
             [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]) / 6.0
    Ttot = np.einsum("ijk,ai,bj,ck->abc", G0, V0, V0, V0) + T
    OmP, OmA = f(n, s), f(n, s)
    for r_out in (12, 8):
        Vn, Gn = join_b(A(V0), A(G0), A(P), A(AL), A(Z), A(L), A(sb), A(eb), A(w2),
                        A(OmP), A(OmA), r_out, core_einsum)
        Vn = np.asarray(Vn)
        Tn = np.einsum("ijk,ai,bj,ck->abc", np.asarray(Gn), Vn, Vn, Vn)
        Pi = Vn @ Vn.T
        ref = np.einsum("ijk,ai,bj,ck->abc", Ttot, Pi, Pi, Pi)
        out[f"F_join_b_r{r_out}_vs_projected_total"] = _rel(Tn, ref)
        out[f"F_join_b_r{r_out}_basis_orthonormal"] = _rel(Vn.T @ Vn, np.eye(r_out))
    out["F_join_b_r12_vs_exact_total"] = _rel(
        np.einsum("ijk,ai,bj,ck->abc", *(lambda VG: (np.asarray(VG[1]),) + (np.asarray(VG[0]),) * 3)(
            join_b(A(V0), A(G0), A(P), A(AL), A(Z), A(L), A(sb), A(eb), A(w2),
                   A(OmP), A(OmA), 12, core_einsum))), Ttot)
    worst = max(v for k, v in out.items())
    out["worst"] = worst
    out["pass_1e-10"] = bool(worst < 1e-10)
    return out


# ==========================================================================
# the meter
# ==========================================================================

def meter(fn):
    """Run fn() under a fresh BudgetContext; FLOPs, ops, names, temp bytes, wall."""
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    tracemalloc.reset_peak()
    with flopscope.BudgetContext(2 ** 60, quiet=True) as ctx:
        t0 = time.perf_counter()
        res = fn()
        wall = time.perf_counter() - t0
        log = list(ctx.op_log)
        flops = int(ctx.flops_used)
    peak = tracemalloc.get_traced_memory()[1] - base
    tracemalloc.stop()
    names = {}
    for rec in log:
        key = rec.op_name
        names[key] = names.get(key, 0) + 1
    detail = [{"op": rec.op_name, "shapes": [list(s) for s in rec.shapes],
               "flops": int(rec.flop_cost)} for rec in log]
    return res, {"flops": flops, "c_over_b": flops / B, "operations": len(log),
                 "op_names": names, "ops": detail,
                 "tracemalloc_peak_increment_bytes": int(peak),
                 "flopscope_overhead_s": float(sum(r.flopscope_overhead_duration_s for r in log)),
                 "local_wall_s_advisory": wall}


def _best(steps, names):
    cands = [(steps[k]["flops"], k) for k in names if k in steps and "flops" in steps[k]]
    return min(cands)[1]


def meter_steps(v31, ranks, s_list):
    rng = np.random.default_rng(PRICE_SEED)
    n = N

    def arr(*s):
        return fnp.asarray(rng.standard_normal(s).astype(np.float32))

    WT = arr(n, n)
    col = arr(n, 1)
    off = fnp.asarray((1.0 - np.eye(n)).astype(np.float32))
    P, AL = arr(n, n), arr(n, n)
    Z, L = arr(n, R0), arr(n, R0)
    sb, eb, w2 = arr(n), arr(n), arr(n)
    D3r, D21r = arr(n), arr(n, n)
    out = {}
    for r in ranks:
        print(f"[steps] r={r}", flush=True)
        st = {}
        V = fnp.asarray(np.linalg.qr(rng.standard_normal((n, r)))[0].astype(np.float32))
        g = rng.standard_normal((r, r, r)).astype(np.float32)
        g = (g + g.transpose(0, 2, 1) + g.transpose(1, 0, 2) + g.transpose(1, 2, 0)
             + g.transpose(2, 0, 1) + g.transpose(2, 1, 0)) / np.float32(6.0)
        G = fnp.asarray(g)
        plain = lambda x, y: x @ y
        _, st["transport_plain"] = meter(lambda: transport(WT, V, col, plain))
        _, st["transport_mm"] = meter(lambda: transport(WT, V, col, v31._mm))
        _, st["transport_last_plain"] = meter(lambda: transport(WT, V, None, plain))
        _, st["transport_last_mm"] = meter(lambda: transport(WT, V, None, v31._mm))
        _, st["rebase_einsum"] = meter(lambda: rebase_einsum(V, G))
        _, st["rebase_tdot"] = meter(lambda: rebase_tdot(V, G))

        def q_add(q, last):
            def f():
                d3, d21 = q(G, V, off, last)
                a3 = D3r + d3
                return (a3,) if last else (a3, D21r + d21)
            return f
        _, st["query_d21_einsum"] = meter(q_add(query_einsum, False))
        _, st["query_d21_kr"] = meter(q_add(query_kr, False))
        _, st["query_d3_einsum"] = meter(q_add(query_einsum, True))
        _, st["query_d3_kr"] = meter(q_add(query_kr, True))
        # declared-symmetric core: the wrap (with its validation) is inside the step
        try:
            def qsym(last):
                def f():
                    Gs = flopscope.as_symmetric(G, symmetry=(0, 1, 2))
                    return q_add(lambda _G, V_, o, l: query_einsum(Gs, V_, o, l), last)()
                return f
            _, st["query_d21_einsum_symdeclared"] = meter(qsym(False))
            _, st["query_d3_einsum_symdeclared"] = meter(qsym(True))
            _, st["rebase_einsum_symdeclared"] = meter(
                lambda: rebase_einsum(V, flopscope.as_symmetric(G, symmetry=(0, 1, 2))))
        except Exception as exc:  # pragma: no cover - API probe
            st["query_symdeclared_error"] = repr(exc)[:300]

        for lname, legs in (("ortho", legs_ortho), ("oblique", legs_oblique)):
            for cname, core in (("einsum", core_einsum), ("kr", core_kr)):
                _, st[f"join_a_{lname}_{cname}"] = meter(
                    lambda: join_a(V, G, P, AL, Z, L, sb, eb, w2, legs, core))
        OmP_r, OmA_r = arr(n, r), arr(n, r)
        Gz = fnp.asarray(np.zeros((r, r, r), dtype=np.float32))
        for cname, core in (("einsum", core_einsum), ("kr", core_kr)):
            def _init(core=core):
                Vi = init_basis(P, AL, OmP_r, OmA_r)
                return join_a(Vi, Gz, P, AL, Z, L, sb, eb, w2, legs_ortho, core)
            _, st[f"init_{cname}"] = meter(_init)
        for s in s_list:
            sv = r // 2 if s == "half" else int(s)
            OmP, OmA = arr(n, sv), arr(n, sv)
            print(f"[steps] r={r} join_b s={sv}", flush=True)
            _, st[f"join_b_s{s}"] = meter(
                lambda: join_b(V, G, P, AL, Z, L, sb, eb, w2, OmP, OmA, r, core_einsum))
            st[f"join_b_s{s}"]["s"] = sv
        # the chosen (cheapest metered) spelling of each role
        st["_chosen"] = {
            "transport": _best(st, ["transport_plain", "transport_mm"]),
            "transport_last": _best(st, ["transport_last_plain", "transport_last_mm"]),
            "rebase": _best(st, ["rebase_einsum", "rebase_tdot"]),
            "query_d21": _best(st, ["query_d21_einsum", "query_d21_kr"]),
            "query_d3": _best(st, ["query_d3_einsum", "query_d3_kr"]),
            "join_ortho": _best(st, ["join_a_ortho_einsum", "join_a_ortho_kr"]),
            "join_oblique": _best(st, ["join_a_oblique_einsum", "join_a_oblique_kr"]),
            "init": _best(st, ["init_einsum", "init_kr"]),
        }
        out[str(r)] = st
    return out


# ---- v31 whole-chain meter (fresh subprocess per generation cap) ---------

class _Ctx:
    seed = 0
    scratch_dir = _ROOT
    submission_dir = _ROOT
    width, depth, flop_budget, n_mlps = N, DEPTH, B, 2


def _v31_child(gens):
    import ctypes
    import whestbench
    v31, sha = _load_v31()
    if gens != "none":
        g = int(gens)
        v31._fit_generations = lambda n, d, b: g
    est = v31.Estimator()
    est.setup(_Ctx())
    rng = np.random.default_rng(PRICE_SEED + 1)
    res = None
    for i in range(2):
        weights = []
        for _ in range(DEPTH):
            a = rng.standard_normal((N, N)) * np.sqrt(2.0 / N)
            a.setflags(write=False)
            weights.append(fnp.asarray(a))
        mlp = whestbench.MLP(weights=weights, width=N, depth=DEPTH, seed=i)
        with flopscope.BudgetContext(B, quiet=True) as ctx:
            t0 = time.perf_counter()
            est.predict(mlp, B)
            wall = time.perf_counter() - t0
            log = list(ctx.op_log)
            flops = int(ctx.flops_used)
        names = {}
        for rec in log:
            names[rec.op_name] = names.get(rec.op_name, 0) + 1
        res = {"gens": gens, "flops": flops, "c_over_b": flops / B,
               "operations": len(log), "op_names": names,
               "flopscope_overhead_s": float(sum(r.flopscope_overhead_duration_s for r in log)),
               "local_wall_s_advisory": wall, "candidate_sha256": sha,
               "mode": (est.last_run or {}).get("mode")}
    if os.name == "nt":
        class PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("a", ctypes.c_size_t), ("b", ctypes.c_size_t), ("c", ctypes.c_size_t),
                        ("d", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
        k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        res["process_peak_working_set_bytes"] = int(pmc.PeakWorkingSetSize)
        res["process_peak_commit_bytes"] = int(pmc.PeakPagefileUsage)
    print("RESULT " + json.dumps(res))


def meter_v31(gens_list):
    out = {}
    for g in gens_list:
        print(f"[v31] gens={g}", flush=True)
        cp = subprocess.run([sys.executable, "-u", os.path.abspath(__file__), "--v31-child", str(g)],
                            capture_output=True, text=True, cwd=_ROOT)
        line = [ln for ln in cp.stdout.splitlines() if ln.startswith("RESULT ")]
        if cp.returncode != 0 or not line:
            out[str(g)] = {"failed": True, "stderr_tail": cp.stderr[-3000:]}
        else:
            out[str(g)] = json.loads(line[0][7:])
        print({k: v for k, v in out[str(g)].items() if k != "op_names"}, flush=True)
    return out


# ---- whole-chain projection ---------------------------------------------

def chain(r, a, sched, steps, v31m, sym=False):
    st = steps[str(r)]
    ch = st["_chosen"]
    base = v31m[str(a)]
    c = {"recent_tier_and_rest_v31_gens": (base["flops"], base["operations"])}

    def add(term, key, times):
        f, o = c.get(term, (0, 0))
        c[term] = (f + st[key]["flops"] * times, o + st[key]["operations"] * times)

    rb = ch["rebase"]
    if sym and "rebase_einsum_symdeclared" in st and st["rebase_einsum_symdeclared"]["flops"] < st[rb]["flops"]:
        rb = "rebase_einsum_symdeclared"
    add("transport", ch["transport"], 14 - a)
    add("transport", ch["transport_last"], 1)
    qd21 = "query_d21_einsum_symdeclared" if sym and "query_d21_einsum_symdeclared" in st else ch["query_d21"]
    qd3 = "query_d3_einsum_symdeclared" if sym and "query_d3_einsum_symdeclared" in st else ch["query_d3"]
    if sym:
        qd21 = min([qd21, ch["query_d21"]], key=lambda k: st[k]["flops"])
        qd3 = min([qd3, ch["query_d3"]], key=lambda k: st[k]["flops"])
    add("query", qd21, 14 - a)
    add("query", qd3, 1)
    add("join", ch["init"], 1)
    later = list(range(a + 1, 15))                 # layers whose end carries a non-init join
    if sched.startswith("join_b"):
        add("join", sched, len(later))
    else:
        k = {"qr_every_1": 1, "qr_every_2": 2, "qr_every_4": 4, "qr_never": None}[sched]
        rebased = [l for l in later if k is not None and (l - a) % k == 0]
        add("rebase", rb, len(rebased))
        add("join", ch["join_ortho"], len(rebased))
        add("join", ch["join_oblique"], len(later) - len(rebased))
    total_f = sum(v[0] for v in c.values())
    total_o = sum(v[1] for v in c.values())
    old = {k: v for k, v in c.items() if k != "recent_tier_and_rest_v31_gens"}
    return {"r": r, "a": a, "schedule": sched, "symdeclared_query": sym,
            "flops": total_f, "c_over_b": total_f / B, "operations": total_o,
            "terms_flops": {k: v[0] for k, v in c.items()},
            "terms_c_over_b": {k: v[0] / B for k, v in c.items()},
            "terms_operations": {k: v[1] for k, v in c.items()},
            "dominant_term": max(c, key=lambda k: c[k][0]),
            "dominant_old_tier_term": max(old, key=lambda k: old[k][0]) if old else None,
            "old_tier_c_over_b": sum(v[0] for v in old.values()) / B,
            "gate_cb_le_0p20": total_f / B <= 0.20, "gate_ops_le_22000": total_o <= 22000}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--v31-child", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ranks", default=",".join(map(str, RANKS)))
    a = ap.parse_args()
    if a.v31_child is not None:
        _v31_child(a.v31_child)
        return 0
    v31, sha = _load_v31()
    sc = selfcheck(v31)
    print(json.dumps(sc, indent=1), flush=True)
    if a.selfcheck:
        return 0 if sc["pass_1e-10"] else 1
    if not sc["pass_1e-10"]:
        raise SystemExit("self-check failed; refusing to price an unverified spelling")
    ranks = [int(x) for x in a.ranks.split(",")]
    s_list = [32, 64, "half"]
    steps = meter_steps(v31, ranks, s_list)
    v31m = meter_v31(["none"] + AGES)
    scheds = ["qr_every_1", "qr_every_2", "qr_every_4", "qr_never"] + [f"join_b_s{s}" for s in s_list]
    chains = []
    for r in ranks:
        for age in AGES:
            for sc_name in scheds:
                chains.append(chain(r, age, sc_name, steps, v31m))
                if not sc_name.startswith("join_b"):
                    chains.append(chain(r, age, sc_name, steps, v31m, sym=True))
    full = v31m["none"]
    formula = v31._projected_flops(N, DEPTH, None)
    doc = {
        "what": "N1 merged old-tier cycle price, n=1024, depth 16, real flopscope meter. COST ONLY.",
        "flopscope": getattr(flopscope, "__version__", "?"),
        "flopscope_file": flopscope.__file__,
        "v31": V31, "v31_sha256": sha,
        "age_convention": "recent tier holds at most a dense lineages (v31 gens=a); "
                          "joins at end of layers a..14; old queries at layers a+1..15",
        "selfcheck_f64_n12": sc,
        "v31_metered_chain": v31m,
        "no_merge_reference": {
            "metered_flops": full.get("flops"), "metered_c_over_b": full.get("c_over_b"),
            "graded_c_over_b": V31_GRADED_CB,
            "discrepancy_c_over_b": (full.get("c_over_b") or 0) - V31_GRADED_CB,
            "v31_own_formula_flops": formula, "v31_own_formula_c_over_b": formula / B},
        "steps_by_r": steps,
        "chains": chains,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
