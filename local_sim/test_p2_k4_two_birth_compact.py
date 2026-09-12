"""Truth-free two-birth integration of abstract discrete K4-to-K3 feeds.

The control enumerates raw fourth moments and all Wick subtractions without
using compact Source, dense_slices, or compact_slices.  Only the system under
test uses the previously accepted abstract feed Source constructor.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from p2_compact_reference import (Shared, compact_slices, reconstruct,
                                  transport_shared, weighted_gram_join)
from test_p2_k4_feed_embedding import compact_sources


ATOL = 1e-10
CONTRACT_COMMIT = "424477e"


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def banks_and_maps():
    weights = [np.array([0.1, 0.2, 0.3, 0.4]),
               np.array([0.31, 0.13, 0.37, 0.19])]
    raw = [np.array([[1.2, -0.8, 0.4, 1.7],
                     [-0.7, 1.1, 0.9, -1.3],
                     [0.5, -1.5, 1.4, 0.3],
                     [1.0, 0.6, -0.9, -0.2]]),
           np.array([[-0.45, 1.65, -1.15, 0.75],
                     [1.4, -0.35, 0.6, -1.2],
                     [0.55, 1.1, -0.85, -0.4],
                     [-1.3, 0.25, 1.45, 0.8]])]
    v = [x - (x * p[None, :]).sum(axis=1)[:, None]
         for x, p in zip(raw, weights)]
    w1 = [np.array([0.85, 1.1, 0.65, 0.9]),
          np.array([1.25, 0.72, 0.95, 0.58])]
    w2 = [np.array([0.35, -0.42, 0.28, 0.53]),
          np.array([-0.47, 0.62, 0.31, -0.26])]
    maps = [np.array([[0.8, -0.3, 0.2, 0.1],
                      [-0.25, 0.7, 0.4, -0.2],
                      [0.15, -0.1, 0.9, 0.35],
                      [-0.2, 0.3, -0.15, 0.75]]),
            np.array([[0.6, 0.2, -0.3, 0.4],
                      [0.25, -0.8, 0.1, 0.2],
                      [-0.15, 0.45, 0.7, -0.25],
                      [0.3, -0.1, 0.2, 0.65]])]
    return list(zip(v, weights, w1, w2)), maps


def direct_birth_feed(v, p, w1, w2):
    """Raw weighted fourth moment minus the three covariance pairings."""
    n, count = v.shape
    c = np.zeros((n, n))
    fourth = np.zeros((n, n, n, n))
    for h in range(count):
        for i in range(n):
            for j in range(n):
                c[i, j] += p[h] * v[i, h] * v[j, h]
                for k in range(n):
                    for ell in range(n):
                        fourth[i, j, k, ell] += (
                            p[h] * v[i, h] * v[j, h] * v[k, h] * v[ell, h])
    out = np.zeros((n, n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                def cumulant(a, b, d, e):
                    return (fourth[a, b, d, e] - c[a, b] * c[d, e]
                            - c[a, d] * c[b, e] - c[a, e] * c[b, d])
                out[i, j, k] = 0.5 * (
                    w2[i] * w1[j] * w1[k] * cumulant(i, i, j, k)
                    + w1[i] * w2[j] * w1[k] * cumulant(i, j, j, k)
                    + w1[i] * w1[j] * w2[k] * cumulant(i, j, k, k))
    return out


def direct_carry(tensor, transform):
    n = transform.shape[0]
    out = np.zeros((n, n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for a in range(n):
                    for b in range(n):
                        for c in range(n):
                            out[i, j, k] += (transform[i, a] * transform[j, b]
                                             * transform[k, c] * tensor[a, b, c])
    return out


def source_tensor(source):
    """Expand the *reconstructed* compact factors without using its slice API."""
    n = source.A.shape[0]
    out = np.zeros((n, n, n))
    m = (source.P * source.s[None, :]
         + 3.0 * source.A * source.e[None, :]
         + source.Z @ source.L.T)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for h in range(source.P.shape[1]):
                    a, p = source.A[:, h], source.P[:, h]
                    out[i, j, k] += source.w2[h] * (
                        p[i] * a[j] * a[k] + a[i] * p[j] * a[k]
                        + a[i] * a[j] * p[k])
                    out[i, j, k] += (
                        m[i, h] * p[j] * p[k] + p[i] * m[j, h] * p[k]
                        + p[i] * p[j] * m[k, h]) / 3.0
                if source.c1 is not None:
                    # This fixture's Sources leave c1/c2/Y unset, but preserve
                    # the reference's optional feed term in the expansion.
                    for h in range(source.P.shape[1]):
                        out[i, j, k] += (source.c1[h] * (
                            source.Y[i] * source.A[j, h] * source.P[k, h]
                            + source.P[i, h] * source.Y[j] * source.A[k, h]
                            + source.A[i, h] * source.P[j, h] * source.Y[k])
                            + source.c2[h] * (
                                source.Y[i] * source.P[j, h] * source.P[k, h]
                                + source.P[i, h] * source.Y[j] * source.P[k, h]
                                + source.P[i, h] * source.P[j, h] * source.Y[k])) / 3.0
    return out


def direct_gram(legs):
    n = legs[0][0].shape[0]
    out = np.zeros((n, n))
    for a, p, da, dp in legs:
        for h in range(a.shape[1]):
            for i in range(n):
                for j in range(n):
                    out[i, j] += da[h] * a[i, h] * a[j, h]
        for h in range(p.shape[1]):
            for i in range(n):
                for j in range(n):
                    out[i, j] += dp[h] * p[i, h] * p[j, h]
    return out


def record(checks, name, got, want, limit=ATOL):
    delta = np.asarray(got, dtype=np.float64) - np.asarray(want, dtype=np.float64)
    error = float(np.max(np.abs(delta))) if delta.size else 0.0
    passed = bool(np.isfinite(delta).all() and error <= limit)
    checks.append({"name": name, "pass": passed, "max_abs_error": error,
                   "limit": limit})
    return error


def evaluate_schedule(rank, banks, maps, checks=None):
    shared = Shared()
    direct = np.zeros((4, 4, 4))
    legs = []
    expected_ages = []
    rows = []
    omega = np.eye(4)[:, :rank]
    for step in range(4):
        if step in (0, 2):
            birth = step // 2
            v, p, w1, w2 = banks[birth]
            direct += direct_birth_feed(v, p, w1, w2)
            new_sources = compact_sources(v, p, w1, w2)
            for source in new_sources:
                shared = weighted_gram_join(shared, source, rank, omega)
                legs.append((source.A.copy(), source.P.copy(),
                             source.dA.copy(), source.dP.copy()))
                expected_ages.append(0)
                if checks is not None:
                    join_tag = "birth%d_subsource%d" % (birth + 1, len(new_sources)
                                                           if len(legs) % 2 == 0 else 1)
                    record(checks, join_tag + "_gram_symmetric",
                           shared.gram, shared.gram.T)
                    record(checks, join_tag + "_weighted_gram",
                           shared.Q @ shared.gram @ shared.Q.T,
                           direct_gram(legs))
        else:
            transform = maps[0 if step == 1 else 1]
            direct = direct_carry(direct, transform)
            shared = transport_shared(shared, transform, np.ones(4))
            legs = [(transform @ a, transform @ p, da, dp)
                    for a, p, da, dp in legs]
            expected_ages = [age + 1 for age in expected_ages]
        tag = ("birth1", "carry1", "birth2_join", "carry2")[step]
        recovered = reconstruct(shared)
        cubic = sum((source_tensor(s) for s in recovered), np.zeros((4, 4, 4)))
        got_d3, got_d21 = compact_slices(shared)
        oracle_d3 = np.array([direct[i, i, i] for i in range(4)])
        oracle_d21 = np.array([[direct[i, i, j] for j in range(4)]
                               for i in range(4)])
        tensor_error = float(np.max(np.abs(cubic - direct)))
        d3_error = float(np.max(np.abs(got_d3 - oracle_d3)))
        d21_error = float(np.max(np.abs(got_d21 - oracle_d21)))
        rows.append({"stage": tag, "retained_sources": len(recovered),
                     "ages": [s.age for s in recovered],
                     "full_tensor_error": tensor_error,
                     "D3_error": d3_error, "D21_error": d21_error,
                     "full_tensor_max_abs": float(np.max(np.abs(cubic))),
                     "D3_max_abs": float(np.max(np.abs(got_d3))),
                     "D21_max_abs": float(np.max(np.abs(got_d21)))})
        if checks is not None:
            record(checks, tag + "_full_cubic", cubic, direct)
            record(checks, tag + "_D3", got_d3, oracle_d3)
            record(checks, tag + "_oriented_D21", got_d21, oracle_d21)
            checks.append({"name": tag + "_birth_age_count",
                           "pass": ([s.age for s in recovered] == expected_ages
                                    and len(recovered) == (2 if step < 2 else 4)),
                           "ages": [s.age for s in recovered],
                           "expected_ages": expected_ages.copy(),
                           "retained_sources": len(recovered)})
            record(checks, tag + "_gram_symmetric", shared.gram, shared.gram.T)
            record(checks, tag + "_weighted_gram", shared.Q @ shared.gram @ shared.Q.T,
                   direct_gram(legs))
            if step in (0, 2):
                record(checks, tag + "_basis_orthogonality_at_join",
                       shared.Q.T @ shared.Q, np.eye(4))
            for idx, (source, (a, p, _, _)) in enumerate(zip(recovered, legs)):
                record(checks, tag + "_factor_rotation_A%d" % idx, source.A, a)
                record(checks, tag + "_factor_rotation_P%d" % idx, source.P, p)
            if step == 3:
                mean_only, skipped = compact_slices(shared, need_d21=False)
                record(checks, "final_mean_only_D3", mean_only, oracle_d3)
                checks.append({"name": "final_mean_only_skips_D21",
                               "pass": skipped is None})
    return rows


def evaluate():
    checks = []
    banks, maps = banks_and_maps()
    for idx, (v, p, w1, w2) in enumerate(banks, 1):
        record(checks, "bank%d_total_mass" % idx, p.sum(), 1.0)
        record(checks, "bank%d_centered" % idx, v @ p, np.zeros(4))
        checks.append({"name": "bank%d_positive_weights" % idx,
                       "pass": bool((p > 0).all())})
        checks.append({"name": "bank%d_nonzero_quadratic" % idx,
                       "pass": bool(np.any(w2 > 0) and np.any(w2 < 0))})
    checks.append({"name": "distinct_banks_and_signed_maps", "pass": bool(
        not np.array_equal(banks[0][0], banks[1][0])
        and not np.array_equal(banks[0][1], banks[1][1])
        and not np.array_equal(maps[0], maps[1])
        and all(np.min(b) < 0 < np.max(b) and np.ptp(b) > 0.5 for b in maps))})
    full_rows = evaluate_schedule(4, banks, maps, checks)
    rank2_rows = evaluate_schedule(2, banks, maps)
    # Dropping both covariance-negative pair legs is an intentionally wrong
    # approximation, checked at the second birth before the final carry.
    v1, p1, w11, w21 = banks[0]
    v2, p2, w12, w22 = banks[1]
    direct_at_birth2 = direct_carry(direct_birth_feed(v1, p1, w11, w21), maps[0])
    direct_at_birth2 += direct_birth_feed(v2, p2, w12, w22)
    mutant = Shared()
    omega = np.eye(4)
    for s in compact_sources(v1, p1, w11, w21, "positive"):
        mutant = weighted_gram_join(mutant, s, 4, omega)
    mutant = transport_shared(mutant, maps[0], np.ones(4))
    for s in compact_sources(v2, p2, w12, w22, "positive"):
        mutant = weighted_gram_join(mutant, s, 4, omega)
    mutant_tensor = sum((source_tensor(s) for s in reconstruct(mutant)),
                        np.zeros((4, 4, 4)))
    mutant_gap = float(np.max(np.abs(mutant_tensor - direct_at_birth2)))
    checks.append({"name": "dropped_negative_pair_mutant_refused",
                   "pass": bool(np.isfinite(mutant_gap) and mutant_gap > 1e-3),
                   "gap": mutant_gap, "minimum": 1e-3})
    zero_banks = [(v, p, w1, np.zeros_like(w2)) for v, p, w1, w2 in banks]
    zero_rows = evaluate_schedule(4, zero_banks, maps)
    for row in zero_rows:
        checks.append({"name": row["stage"] + "_zero_source",
                       "pass": bool(max(row["full_tensor_max_abs"],
                                        row["D3_max_abs"], row["D21_max_abs"])
                                    <= ATOL),
                       "full_tensor_error": row["full_tensor_error"],
                       "D3_error": row["D3_error"],
                       "D21_error": row["D21_error"],
                       "full_tensor_max_abs": row["full_tensor_max_abs"],
                       "D3_max_abs": row["D3_max_abs"],
                       "D21_max_abs": row["D21_max_abs"], "limit": ATOL})
    rank2_loss = max(row["full_tensor_error"] for row in rank2_rows)
    return {"kind": "truth-free two-birth abstract compact K3-feed integration",
            "source_sha256": file_sha(__file__), "contract_commit": CONTRACT_COMMIT,
            "feed_constructor_sha256": file_sha(Path(__file__).with_name(
                "test_p2_k4_feed_embedding.py")),
            "compact_reference_sha256": file_sha(Path(__file__).with_name(
                "p2_compact_reference.py")),
            "checks": checks, "n_checks": len(checks),
            "n_failed": sum(not x["pass"] for x in checks),
            "pass": all(x["pass"] for x in checks), "full_rank": full_rows,
            "rank2_diagnostic": {"rows": rank2_rows,
                                 "max_full_tensor_loss": rank2_loss,
                                 "predicted_material_loss": rank2_loss > 1e-6,
                                 "minimum_material_loss": 1e-6,
                                 "is_exactness_gate": False},
            "mutant_gap": mutant_gap,
            "scope": "two abstract discrete feeds, not physical nonlinear source, live rank or challenge score"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    args = ap.parse_args()
    target = Path(args.json)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = evaluate()
    except Exception as exc:
        result = {"kind": "truth-free two-birth abstract compact K3-feed integration",
                  "source_sha256": file_sha(__file__),
                  "contract_commit": CONTRACT_COMMIT, "pass": False,
                  "error_type": type(exc).__name__, "error": str(exc)}
    data = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    target.write_bytes(data)
    target.with_name("result_receipt.json").write_text(
        json.dumps({"source_sha256": file_sha(__file__),
                    "result_sha256": hashlib.sha256(data).hexdigest(),
                    "pass": result["pass"]}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print("%d/%d checks passed" %
          (result.get("n_checks", 0) - result.get("n_failed", 0),
           result.get("n_checks", 0)))
    if not result["pass"]:
        print("FAIL: " + result.get("error", repr(
            [x for x in result.get("checks", []) if not x["pass"]])))
        return 1
    print("rank-2 max full-tensor loss %.12g; dropped-pair mutant gap %.12g" %
          (result["rank2_diagnostic"]["max_full_tensor_loss"],
           result["mutant_gap"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
