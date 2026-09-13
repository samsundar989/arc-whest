"""B2: tiny full-rank weighted-Gram QR storage discriminator.

The frozen L2 layer remains the nonlinear engine.  This probe tests only
whether the existing compact QR join and factor rotation preserve all exact
Source ages when the shared rank and residual rank equal width four.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np

import p2_compact_fullrank_bridge_b1 as b1
import p2_compact_reference as compact


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "experiments" / "2026-09-12_b2_fullrank_qr.json"
TOL = 1e-5
RANK = RESIDUAL_RANK = 4


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(checks, layer, name, got, want):
    g, w = b1.arr(got), b1.arr(want)
    shape_equal = g.shape == w.shape
    finite = bool(np.isfinite(g).all() and np.isfinite(w).all())
    error = (float(np.max(np.abs(g - w) / np.maximum(1.0, np.abs(w))))
             if shape_equal and finite and g.size else
             0.0 if shape_equal and finite else float("inf"))
    checks.append({"layer": layer, "field": name, "shape": list(w.shape),
                   "finite": finite, "normalized_error": error,
                   "passed": shape_equal and finite and error <= TOL})


def compare_sources(checks, layer, got, want, prefix):
    got = sorted(got, key=lambda s: -s.age)
    want = sorted(want, key=lambda s: -s.age)
    record(checks, layer, prefix + ".age_count", len(got), len(want))
    for g, w in zip(got, want):
        tag = f"{prefix}.age{w.age}"
        record(checks, layer, tag + ".age", g.age, w.age)
        for field in ("A", "P", "Z", "L", "s", "e", "w2", "dA", "dP"):
            record(checks, layer, tag + "." + field,
                   getattr(g, field), getattr(w, field))
        record(checks, layer, tag + ".feed_absent",
               int(g.c1 is g.c2 is g.Y is None), 1)
        record(checks, layer, tag + ".materialized_A_D",
               compact.materialize(g), compact.materialize(w))


def decode_qr(frozen, shared, checks, layer):
    """Decode sources into L2's symbolic-youngest-I tower layout."""
    sources = sorted(compact.reconstruct(shared), key=lambda s: -s.age)
    if not sources:
        return None
    n, k = sources[0].A.shape[0], len(sources)
    record(checks, layer, "decode.age_sequence",
           [s.age for s in sources], list(range(k - 1, -1, -1)))
    record(checks, layer, "decode.youngest_P_identity", sources[-1].P, np.eye(n))
    for s in sources:
        record(checks, layer, f"decode.age{s.age}.s_zero", s.s, np.zeros(n))
        record(checks, layer, f"decode.age{s.age}.e_zero", s.e, np.zeros(n))
        record(checks, layer, f"decode.age{s.age}.L_identity", s.L, np.eye(n))
    A_D = fnp.array(np.concatenate([compact.materialize(s) for s in sources], axis=1),
                    dtype=fnp.float32)
    A_L = fnp.array(np.concatenate([s.A for s in sources], axis=1), dtype=fnp.float32)
    M = (None if k == 1 else
         fnp.array(np.concatenate([s.P for s in sources[:-1]], axis=1),
                   dtype=fnp.float32))
    w2b = fnp.array(np.concatenate([s.w2 for s in sources]), dtype=fnp.float32)
    return frozen._Tower(A_D, A_L, M, w2b, k)


def scale_shared_rows(shared, wick):
    """Apply the frozen post-nonlinear Wick scaling without changing age."""
    if shared.Q is None:
        return shared
    col = wick[:, None]
    factors = [compact.Factors(
        f.FA, f.FP, col * f.Z, f.L, f.s, f.e, f.w2,
        f.dA, f.dP, f.c1, f.c2,
        None if f.Y is None else wick * f.Y, f.age)
        for f in shared.factors]
    return compact.Shared(col * shared.Q, factors, shared.gram)


def wick1(frozen, W, state, table, const):
    mu, cov = state[:2]
    pre_mu = W.T @ mu
    pre_cov = frozen._mm(frozen._mm(W.T, cov), W)
    var = fnp.maximum(fnp.diagonal(pre_cov), fnp.float32(1e-12))
    sigma = fnp.sqrt(var)
    alpha = fnp.clip(pre_mu / sigma, fnp.float32(-frozen.ALPHA_CAP),
                     fnp.float32(frozen.ALPHA_CAP))
    return b1.arr(frozen._wick(pre_mu, sigma, alpha, table, fnp.float32,
                               const["wick"])[(1, 1)])


def run():
    frozen = b1.load_frozen()
    weights = b1.WEIGHTS
    depth, n = weights.shape[:2]
    assert (n, depth, RANK, RESIDUAL_RANK) == (4, 4, 4, 4)
    checks, activity = [], []
    identity_means, qr_means, reference_means = [], [], []
    with flops.BudgetContext(flop_budget=2 ** 42, quiet=True):
        dtype = fnp.float32
        table = frozen._tables()
        const = {"off_mask": fnp.ones((n, n), dtype=dtype) - fnp.eye(n, dtype=dtype),
                 "ones_n": fnp.ones(n, dtype=dtype),
                 "wick": frozen._wick_program(table, dtype)}
        initial = (fnp.zeros(n, dtype=dtype), fnp.eye(n, dtype=dtype), None, 0.0)
        reference = initial
        identity = (initial[0], initial[1], compact.Shared(), initial[3])
        qr = (initial[0], initial[1], compact.Shared(), initial[3])
        for layer, fixed_weight in enumerate(weights):
            last = layer == depth - 1
            W = fnp.array(fixed_weight, dtype=dtype)
            Wc = fixed_weight.astype(np.float32).T
            ref_mu, ref_cov, ref_tw, ref_c4 = reference
            id_mu, id_cov, id_shared, id_c4 = identity
            qr_mu, qr_cov, qr_shared, qr_c4 = qr
            id_tw = b1.decode_shared(frozen, id_shared)
            qr_tw = decode_qr(frozen, qr_shared, checks, layer)
            record(checks, layer, "incoming.mu", qr_mu, ref_mu)
            record(checks, layer, "incoming.cov", qr_cov, ref_cov)
            record(checks, layer, "incoming.c4", qr_c4, ref_c4)
            record(checks, layer, "identity.incoming.mu", id_mu, ref_mu)
            record(checks, layer, "identity.incoming.cov", id_cov, ref_cov)
            record(checks, layer, "identity.incoming.c4", id_c4, ref_c4)
            if ref_tw is None:
                record(checks, layer, "incoming.empty", int(qr_tw is None), 1)
            else:
                compare_sources(checks, layer, b1.tower_sources(qr_tw),
                                b1.tower_sources(ref_tw), "incoming")

            ref_pre_mu = W.T @ ref_mu
            ref_pre_cov = frozen._mm(frozen._mm(W.T, ref_cov), W)
            qr_pre_mu = W.T @ qr_mu
            qr_pre_cov = frozen._mm(frozen._mm(W.T, qr_cov), W)
            record(checks, layer, "preactivation.mu", qr_pre_mu, ref_pre_mu)
            record(checks, layer, "preactivation.cov", qr_pre_cov, ref_pre_cov)
            record(checks, layer, "preactivation.c4", qr_c4, ref_c4)

            ref_adv = None if ref_tw is None else frozen._advance(W.T, ref_tw)
            want_advanced = b1.tower_sources(ref_adv, age_offset=1)
            qr_advanced = compact.transport_shared(qr_shared, Wc, np.ones(n))
            compare_sources(checks, layer, compact.reconstruct(qr_advanced),
                            want_advanced, "linear_transport")
            d3, d21 = compact.compact_slices(qr_advanced, need_d21=not last)
            want_d3 = (frozen._dslice3(ref_adv) if ref_adv is not None
                       else fnp.zeros(n, dtype=dtype))
            if d3 is None:
                d3 = np.zeros(n)
            record(checks, layer, "D3_compact_QR", d3, want_d3)
            if not last:
                want_d21 = (frozen._dslice21(ref_adv, const["off_mask"])
                            if ref_adv is not None else
                            fnp.zeros((n, n), dtype=dtype))
                if d21 is None:
                    d21 = np.zeros((n, n))
                record(checks, layer, "D21_compact_QR_masked",
                       d21 * (np.ones((n, n)) - np.eye(n)), want_d21)

            reference = frozen._layer(W, reference, table, dtype, const, 2.0,
                                      last_layer=last, gens=None)
            id_next = frozen._layer(W, (id_mu, id_cov, id_tw, id_c4),
                                    table, dtype, const, 2.0,
                                    last_layer=last, gens=None)
            qr_next = frozen._layer(W, (qr_mu, qr_cov, qr_tw, qr_c4),
                                    table, dtype, const, 2.0,
                                    last_layer=last, gens=None)
            reference_means.append(reference[0])
            identity_means.append(id_next[0])
            qr_means.append(qr_next[0])
            for arm, state in (("identity", id_next), ("qr", qr_next)):
                record(checks, layer, arm + ".post_mu", state[0], reference[0])
                record(checks, layer, arm + ".post_c4", state[3], reference[3])
                if not last:
                    record(checks, layer, arm + ".post_cov", state[1], reference[1])
            if not last:
                want_sources = b1.tower_sources(reference[2])
                id_sources = b1.tower_sources(id_next[2])
                qr_sources = b1.tower_sources(qr_next[2])
                id_shared = b1.identity_shared(id_sources)
                compare_sources(checks, layer, compact.reconstruct(id_shared),
                                want_sources, "identity_post")
                # _layer has already applied current Wick to old ages.  Keep
                # those ages in the QR basis, then join this layer's newborn.
                w1 = wick1(frozen, W, (qr_mu, qr_cov), table, const)
                qr_scaled = scale_shared_rows(qr_advanced, w1)
                compare_sources(checks, layer, compact.reconstruct(qr_scaled),
                                qr_sources[:-1], "old_after_wick")
                newborn = qr_sources[-1]
                qr_shared = compact.weighted_gram_join(
                    qr_scaled, newborn, RANK, np.eye(n))
                reconstructed = compact.reconstruct(qr_shared)
                compare_sources(checks, layer, reconstructed, want_sources,
                                "joined_post")
                compare_sources(checks, layer, [reconstructed[-1]],
                                [want_sources[-1]], "newborn")
                qrank = int(np.linalg.matrix_rank(qr_shared.Q))
                record(checks, layer, "joined_basis_rank", qrank, n)
                activity.append({"layer": layer, "joined_ages": len(reconstructed),
                                 "basis_rank": qrank,
                                 "basis_min_singular": float(
                                     np.linalg.svd(qr_shared.Q, compute_uv=False)[-1]),
                                 "scalar_c4_abs": float(np.abs(b1.arr(qr_next[3]))),
                                 "D3_max_abs": float(np.max(np.abs(b1.arr(want_d3)))),
                                 "D21_max_abs": float(np.max(np.abs(b1.arr(want_d21))))})
            else:
                id_shared = qr_shared = compact.Shared()
                record(checks, layer, "final_tower_absent", int(qr_next[2] is None), 1)
                activity.append({"layer": layer, "joined_ages": 0,
                                 "basis_rank": None,
                                 "D3_max_abs": float(np.max(np.abs(b1.arr(want_d3)))),
                                 "D21_max_abs": None})
            identity = (id_next[0], id_next[1], id_shared, id_next[3])
            qr = (qr_next[0], qr_next[1], qr_shared, qr_next[3])
        record(checks, depth - 1, "qr.all_means", fnp.stack(qr_means),
               fnp.stack(reference_means))
        record(checks, depth - 1, "identity.all_means", fnp.stack(identity_means),
               fnp.stack(reference_means))
        record(checks, depth - 1, "qr.vs_identity.final_output",
               qr_means[-1], identity_means[-1])
        record(checks, 2, "two_ages_at_nonfinal_join",
               int(activity[1]["joined_ages"] == 2), 1)
    return {
        "fixture": "B1_explicit_asymmetric_n4_depth4_fullrank_QR",
        "scope": "weighted-Gram QR storage only; frozen L2 nonlinear engine; no compression/score/truth",
        "rank": RANK, "residual_rank": RESIDUAL_RANK,
        "omega": "identity_4x4", "tolerance": TOL,
        "normalization": "abs(got-want)/max(1,abs(want)) elementwise",
        "source_sha256": {
            "probe": sha(Path(__file__)), "B1_probe": sha(Path(b1.__file__)),
            "frozen_L2": sha(b1.FROZEN), "compact_reference": sha(b1.COMPACT)},
        "weights": weights.tolist(), "activity": activity, "checks": checks,
        "check_count": len(checks),
        "failed_count": sum(not c["passed"] for c in checks),
        "max_normalized_error": max(c["normalized_error"] for c in checks),
        "final_D21": "intentionally absent under frozen final mean-only shortcut",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    result = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                        encoding="utf-8")
    print(json.dumps({"artifact": str(args.out), "checks": result["check_count"],
                      "failed": result["failed_count"],
                      "max_error": result["max_normalized_error"]}))
    if result["failed_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
