"""B1: tiny, label-free identity-storage bridge against frozen L2.

This is not a compact nonlinear candidate or a cost test.  Every nonlinear
transition below is executed by the unchanged frozen ``_layer``.  Only its
third-cumulant tower is encoded as full-rank compact Sources and decoded before
the next transition.  The shared basis and residual are both full identity
matrices, so this test introduces no compression or QR approximation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np

import p2_compact_reference as compact


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "estimator_k3_strassen_L2.py"
COMPACT = ROOT / "local_sim" / "p2_compact_reference.py"
DEFAULT_OUT = ROOT / "experiments" / "2026-09-12_b1_fullrank_bridge.json"
TOL = 1e-6

# Explicit, asymmetric input-to-output matrices.  There is no network mint,
# random generator, registered seed, protected truth, or fitted parameter.
WEIGHTS = np.array([
    [[0.85, -0.31, 0.22, 0.14], [0.16, 0.74, -0.28, 0.39],
     [-0.24, 0.18, 0.91, -0.17], [0.37, -0.13, 0.26, 0.68]],
    [[0.72, 0.25, -0.34, 0.19], [-0.21, 0.88, 0.17, -0.27],
     [0.33, -0.16, 0.79, 0.23], [0.11, 0.29, -0.25, 0.84]],
    [[0.93, -0.19, 0.13, -0.22], [0.27, 0.67, -0.31, 0.12],
     [-0.14, 0.36, 0.82, 0.21], [0.25, -0.23, 0.18, 0.77]],
    [[0.76, 0.17, -0.29, 0.31], [-0.26, 0.94, 0.16, -0.11],
     [0.21, -0.24, 0.71, 0.38], [0.18, 0.32, -0.13, 0.86]],
], dtype=np.float64)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arr(value):
    return np.array(value, dtype=np.float64, copy=True)


def load_frozen():
    spec = importlib.util.spec_from_file_location("frozen_l2_b1", FROZEN)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def tower_sources(tower, age_offset=0):
    """Represent every L2 age as an exact full-residual compact Source."""
    if tower is None:
        return []
    n, k = tower.A_D.shape[0], tower.k
    assert tower.A_D.shape == tower.A_L.shape == (n, n * k)
    assert tower.w2b.shape == (n * k,)
    # Post-nonlinearity the youngest P is symbolic I.  Immediately after the
    # next linear step, _advance has materialized that same block as W.T.
    if tower.M is None:
        assert k == 1
        P = np.eye(n)
    elif tower.M.shape == (n, n * k):
        P = arr(tower.M)
    else:
        assert tower.M.shape == (n, n * (k - 1))
        P = np.concatenate((arr(tower.M), np.eye(n)), axis=1)
    eye, zero, one = np.eye(n), np.zeros(n), np.ones(n)
    out = []
    for j in range(k):
        sl = slice(j * n, (j + 1) * n)
        out.append(compact.Source(
            A=arr(tower.A_L[:, sl]), P=P[:, sl].copy(),
            Z=arr(tower.A_D[:, sl]), L=eye.copy(),
            s=zero.copy(), e=zero.copy(), w2=arr(tower.w2b[sl]),
            dA=one.copy(), dP=one.copy(), c1=None, c2=None, Y=None,
            age=k - 1 - j + age_offset))
    return out


def identity_shared(sources):
    if not sources:
        return compact.Shared()
    n = sources[0].A.shape[0]
    return compact.Shared(
        Q=np.eye(n),
        factors=[compact.Factors(
            FA=s.A.copy(), FP=s.P.copy(), Z=s.Z.copy(), L=s.L.copy(),
            s=s.s.copy(), e=s.e.copy(), w2=s.w2.copy(),
            dA=s.dA.copy(), dP=s.dP.copy(),
            c1=None, c2=None, Y=None, age=s.age) for s in sources],
        gram=None)


def decode_shared(frozen, shared):
    """Recover L2's symbolic youngest-I layout from full-rank Sources."""
    sources = sorted(compact.reconstruct(shared), key=lambda s: -s.age)
    if not sources:
        return None
    n, k = sources[0].A.shape[0], len(sources)
    assert [s.age for s in sources] == list(range(k - 1, -1, -1))
    assert np.array_equal(sources[-1].P, np.eye(n))
    for s in sources:
        assert np.array_equal(s.s, np.zeros(n))
        assert np.array_equal(s.e, np.zeros(n))
        assert np.array_equal(s.L, np.eye(n))
        assert s.c1 is s.c2 is s.Y is None
    A_D = fnp.array(np.concatenate([compact.materialize(s) for s in sources], axis=1),
                    dtype=fnp.float32)
    A_L = fnp.array(np.concatenate([s.A for s in sources], axis=1), dtype=fnp.float32)
    M = (None if k == 1 else
         fnp.array(np.concatenate([s.P for s in sources[:-1]], axis=1),
                   dtype=fnp.float32))
    w2b = fnp.array(np.concatenate([s.w2 for s in sources]), dtype=fnp.float32)
    return frozen._Tower(A_D, A_L, M, w2b, k)


def record(checks, layer, name, got, want):
    g, w = arr(got), arr(want)
    shape_equal = g.shape == w.shape
    finite = bool(np.isfinite(g).all() and np.isfinite(w).all())
    error = (float(np.max(np.abs(g - w) / np.maximum(1.0, np.abs(w))))
             if shape_equal and finite and g.size else
             0.0 if shape_equal and finite else float("inf"))
    checks.append({"layer": layer, "field": name, "shape": list(w.shape),
                   "finite": finite, "normalized_error": error,
                   "passed": shape_equal and finite and error <= TOL})


def check_towers(checks, layer, got, want, prefix):
    if got is None or want is None:
        record(checks, layer, prefix + ".presence",
               int(got is not None), int(want is not None))
        return
    record(checks, layer, prefix + ".k", got.k, want.k)
    record(checks, layer, prefix + ".A_D", got.A_D, want.A_D)
    record(checks, layer, prefix + ".A_L", got.A_L, want.A_L)
    record(checks, layer, prefix + ".w2b", got.w2b, want.w2b)
    if got.M is None or want.M is None:
        record(checks, layer, prefix + ".M_presence",
               int(got.M is not None), int(want.M is not None))
    else:
        record(checks, layer, prefix + ".M", got.M, want.M)
    # A source is one age, whereas the frozen tower concatenates all ages.
    gs, ws = tower_sources(got), tower_sources(want)
    record(checks, layer, prefix + ".age_count", len(gs), len(ws))
    for j, (g, w) in enumerate(zip(gs, ws)):
        tag = f"{prefix}.age{w.age}"
        record(checks, layer, tag + ".age", g.age, w.age)
        for field in ("A", "P", "Z", "L", "s", "e", "w2"):
            record(checks, layer, tag + "." + field,
                   getattr(g, field), getattr(w, field))


def check_sources(checks, layer, got, want, prefix):
    record(checks, layer, prefix + ".count", len(got), len(want))
    for g, w in zip(got, want):
        tag = f"{prefix}.age{w.age}"
        record(checks, layer, tag + ".age", g.age, w.age)
        for field in ("A", "P", "Z", "L", "s", "e", "w2"):
            record(checks, layer, tag + "." + field,
                   getattr(g, field), getattr(w, field))


def run():
    frozen = load_frozen()
    n, depth = WEIGHTS.shape[1], WEIGHTS.shape[0]
    assert (n, depth) == (4, 4)
    checks = []
    reference_means, bridge_means = [], []
    activity = []
    with flops.BudgetContext(flop_budget=2 ** 42, quiet=True):
        dtype = fnp.float32
        table = frozen._tables()
        program = frozen._wick_program(table, dtype)
        const = {"off_mask": fnp.ones((n, n), dtype=dtype) - fnp.eye(n, dtype=dtype),
                 "ones_n": fnp.ones(n, dtype=dtype), "wick": program}
        reference = (fnp.zeros(n, dtype=dtype), fnp.eye(n, dtype=dtype), None, 0.0)
        bridge = (fnp.zeros(n, dtype=dtype), fnp.eye(n, dtype=dtype),
                  identity_shared([]), 0.0)
        for layer, fixed_weight in enumerate(WEIGHTS):
            W = fnp.array(fixed_weight, dtype=dtype)
            Wc = fixed_weight.astype(np.float32).T
            ref_mu, ref_cov, ref_tw, ref_c4 = reference
            br_mu, br_cov, br_shared, br_c4 = bridge
            br_tw = decode_shared(frozen, br_shared)
            check_towers(checks, layer, br_tw, ref_tw, "incoming_tower")
            record(checks, layer, "incoming_mu", br_mu, ref_mu)
            record(checks, layer, "incoming_cov", br_cov, ref_cov)
            record(checks, layer, "incoming_c4", br_c4, ref_c4)
            ref_pre_mu = W.T @ ref_mu
            br_pre_mu = W.T @ br_mu
            ref_pre_cov = frozen._mm(frozen._mm(W.T, ref_cov), W)
            br_pre_cov = frozen._mm(frozen._mm(W.T, br_cov), W)
            record(checks, layer, "preactivation_mu", br_pre_mu, ref_pre_mu)
            record(checks, layer, "preactivation_cov", br_pre_cov, ref_pre_cov)
            record(checks, layer, "preactivation_c4", br_c4, ref_c4)

            ref_adv = None if ref_tw is None else frozen._advance(W.T, ref_tw)
            advanced_sources = tower_sources(ref_adv, age_offset=1)
            compact_advanced = [compact.transport_dense(s, Wc, np.ones(n))
                                for s in compact.reconstruct(br_shared)]
            check_sources(checks, layer, compact_advanced, advanced_sources,
                          "linear_transport")
            shared_adv = identity_shared(advanced_sources)
            compact_d3, compact_d21 = compact.compact_slices(
                shared_adv, need_d21=layer != depth - 1)
            dense_d3, dense_d21 = compact.dense_slices(
                compact.reconstruct(shared_adv), need_d21=layer != depth - 1)
            frozen_d3 = (frozen._dslice3(ref_adv) if ref_adv is not None
                         else fnp.zeros(n, dtype=dtype))
            if compact_d3 is None:
                compact_d3 = dense_d3 = np.zeros(n)
            record(checks, layer, "D3_compact", compact_d3, frozen_d3)
            record(checks, layer, "D3_dense", dense_d3, frozen_d3)
            if layer != depth - 1:
                frozen_d21 = (frozen._dslice21(ref_adv, const["off_mask"])
                              if ref_adv is not None else
                              fnp.zeros((n, n), dtype=dtype))
                off_mask = np.ones((n, n)) - np.eye(n)
                if compact_d21 is None:
                    compact_d21 = dense_d21 = np.zeros((n, n))
                record(checks, layer, "D21_compact_masked",
                       compact_d21 * off_mask, frozen_d21)
                record(checks, layer, "D21_dense_masked",
                       dense_d21 * off_mask, frozen_d21)
            activity.append({
                "layer": layer, "incoming_ages": len(advanced_sources),
                "incoming_c4_abs": float(np.max(np.abs(arr(ref_c4)))),
                "D3_max_abs": float(np.max(np.abs(arr(frozen_d3)))),
                "D21_max_abs": (None if layer == depth - 1 else
                                 float(np.max(np.abs(arr(frozen_d21))))),
            })

            reference = frozen._layer(W, reference, table, dtype, const, 2.0,
                                      last_layer=layer == depth - 1, gens=None)
            next_bridge = frozen._layer(W, (br_mu, br_cov, br_tw, br_c4),
                                        table, dtype, const, 2.0,
                                        last_layer=layer == depth - 1, gens=None)
            reference_means.append(reference[0])
            bridge_means.append(next_bridge[0])
            record(checks, layer, "postactivation_mu", next_bridge[0], reference[0])
            record(checks, layer, "postactivation_c4", next_bridge[3], reference[3])
            if layer != depth - 1:
                record(checks, layer, "postactivation_cov", next_bridge[1], reference[1])
                check_towers(checks, layer, next_bridge[2], reference[2],
                             "postactivation_tower")
                newborn = tower_sources(next_bridge[2])[-1]
                ref_newborn = tower_sources(reference[2])[-1]
                for field in ("A", "P", "Z", "w2"):
                    record(checks, layer, "birth." + field,
                           getattr(newborn, field), getattr(ref_newborn, field))
                stored = identity_shared(tower_sources(next_bridge[2]))
                decoded = decode_shared(frozen, stored)
                check_towers(checks, layer, decoded, next_bridge[2],
                             "storage_roundtrip")
            else:
                stored = identity_shared([])
                record(checks, layer, "final_tower_absent",
                       int(next_bridge[2] is None), 1)
            bridge = (next_bridge[0], next_bridge[1], stored, next_bridge[3])
        record(checks, depth - 1, "all_layer_means",
               fnp.stack(bridge_means), fnp.stack(reference_means))
        record(checks, depth - 1, "final_output", bridge_means[-1],
               reference_means[-1])
        record(checks, 2, "two_ages_at_nonfinal_D21",
               int(activity[2]["incoming_ages"] == 2), 1)
        record(checks, 2, "nonzero_two_age_D21",
               int(activity[2]["D21_max_abs"] > 1e-10), 1)
        record(checks, 1, "nonzero_carried_scalar_K4",
               int(activity[1]["incoming_c4_abs"] > 1e-10), 1)
    return {
        "fixture": "explicit_asymmetric_n4_depth4_fullrank_identity_source",
        "scope": "storage bridge only; frozen L2 nonlinear engine; no QR/compression/score/truth",
        "tolerance": TOL,
        "normalization": "abs(got-want)/max(1,abs(want)) elementwise",
        "source_sha256": {"probe": sha(Path(__file__)), "frozen_L2": sha(FROZEN),
                          "compact_reference": sha(COMPACT)},
        "weights": WEIGHTS.tolist(),
        "activity": activity,
        "checks": checks,
        "check_count": len(checks),
        "failed_count": sum(not c["passed"] for c in checks),
        "max_normalized_error": max(c["normalized_error"] for c in checks),
        "last_layer_D21": "intentionally absent under frozen final mean-only shortcut",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    result = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                        encoding="utf-8")
    print(json.dumps({"artifact": str(args.out),
                      "checks": result["check_count"],
                      "failed": result["failed_count"],
                      "max_error": result["max_normalized_error"]}))
    if result["failed_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
