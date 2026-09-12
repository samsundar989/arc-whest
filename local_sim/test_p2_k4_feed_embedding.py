"""Truth-free dense oracle for an abstract two-subsource K4-to-K3 embedding.

This exercises the existing compact reference's Source/materialize/slice/join
algebra. It does not create a physical ReLU source or measure challenge cost.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from p2_compact_reference import (Shared, Source, compact_slices, dense_slices,
                                  materialize, transport_dense, transport_shared,
                                  weighted_gram_join)


ATOL = 1e-10
N_LIVE = 1024
BIRTHS = 14
USES = 105
FROZEN_BYTES = 112 * 2**20
FROZEN_TRANSPORT = 450971566080


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raw_cumulant_parts(v, p):
    """Independently enumerate weighted raw moments and three Wick pairings."""
    n, count = v.shape
    c = np.zeros((n, n), dtype=np.float64)
    positive = np.zeros((n, n, n, n), dtype=np.float64)
    for h in range(count):
        for i in range(n):
            for j in range(n):
                c[i, j] += p[h] * v[i, h] * v[j, h]
                for k in range(n):
                    for ell in range(n):
                        positive[i, j, k, ell] += (
                            p[h] * v[i, h] * v[j, h] * v[k, h] * v[ell, h])
    negative = np.zeros_like(positive)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for ell in range(n):
                    negative[i, j, k, ell] = -(
                        c[i, j] * c[k, ell] + c[i, k] * c[j, ell]
                        + c[i, ell] * c[j, k])
    return c, positive, negative


def direct_feed(k4, w1, w2):
    """Enumerate all three quadratic-response placements, no factor algebra."""
    n = len(w1)
    out = np.zeros((n, n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                out[i, j, k] = 0.5 * (
                    w2[i] * w1[j] * w1[k] * k4[i, i, j, k]
                    + w1[i] * w2[j] * w1[k] * k4[i, j, j, k]
                    + w1[i] * w1[j] * w2[k] * k4[i, j, k, k])
    return out


def compact_tensor(source):
    """Enumerate the reference Source's entire implied symmetric cubic."""
    n = source.A.shape[0]
    m = materialize(source)
    out = np.zeros((n, n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                value = 0.0
                for h in range(source.P.shape[1]):
                    ah, ph, mh = source.A[:, h], source.P[:, h], m[:, h]
                    value += source.w2[h] * (
                        ph[i] * ah[j] * ah[k]
                        + ah[i] * ph[j] * ah[k]
                        + ah[i] * ah[j] * ph[k])
                    value += (mh[i] * ph[j] * ph[k]
                              + ph[i] * mh[j] * ph[k]
                              + ph[i] * ph[j] * mh[k]) / 3.0
                out[i, j, k] = value
    return out


def compact_sources(v, p, w1, w2, part="all"):
    """Build two abstract Source objects with the exact reference state fields."""
    n, count = v.shape
    if count > n:
        raise ValueError("fixture uses at most n weighted columns")
    c, _, _ = raw_cumulant_parts(v, p)
    birth_p = w1[:, None] * v
    a = w2 * np.diag(c)
    birth_a = np.diag(w2)
    q = (birth_p * p[None, :]) @ v.T
    aplus = birth_a @ (v * v)
    zeros = np.zeros((n, n))
    pad_p = np.zeros((n, n))
    pad_p[:, :count] = birth_p
    z = np.zeros((n, count))
    if part in ("all", "positive"):
        z += 1.5 * (aplus - (a[:, None] if part == "all" else 0.0)) * p[None, :]
    elif part == "a_only":
        z -= 1.5 * a[:, None] * p[None, :]
    elif part == "negative_pair":
        z *= 0.0
        pad_p *= 0.0
    else:
        raise ValueError(part)
    selectors = np.zeros((n, count))
    for h in range(count):
        selectors[h, h] = 1.0
    plus_a = Source(zeros.copy(), pad_p, z, selectors,
                    np.zeros(n), np.zeros(n), np.zeros(n),
                    np.ones(n), np.ones(n), None, None, None)
    negative = Source(birth_a, q, np.zeros((n, 0)), np.zeros((n, 0)),
                      np.zeros(n), -np.ones(n), np.zeros(n),
                      np.ones(n), np.ones(n), None, None, None)
    if part in ("positive", "a_only"):
        return [plus_a]
    if part == "negative_pair":
        return [negative]
    return [plus_a, negative]


def _fixed_n4():
    p = np.array([0.1, 0.2, 0.3, 0.4])
    raw = np.array([[1.2, -0.8, 0.4, 1.7],
                    [-0.7, 1.1, 0.9, -1.3],
                    [0.5, -1.5, 1.4, 0.3],
                    [1.0, 0.6, -0.9, -0.2]])
    v = raw - (raw * p[None, :]).sum(axis=1)[:, None]
    w1 = np.array([0.85, 1.1, 0.65, 0.9])
    w2 = np.array([0.35, -0.42, 0.28, 0.53])
    b1 = np.array([[0.8, -0.3, 0.2, 0.1],
                   [-0.25, 0.7, 0.4, -0.2],
                   [0.15, -0.1, 0.9, 0.35],
                   [-0.2, 0.3, -0.15, 0.75]])
    b2 = np.array([[0.6, 0.2, -0.3, 0.4],
                   [0.25, -0.8, 0.1, 0.2],
                   [-0.15, 0.45, 0.7, -0.25],
                   [0.3, -0.1, 0.2, 0.65]])
    return v, p, w1, w2, (b1, b2)


def _centered_equilateral():
    z = np.array([[np.sqrt(2), -1 / np.sqrt(2), -1 / np.sqrt(2)],
                  [0.0, np.sqrt(3 / 2), -np.sqrt(3 / 2)]])
    g = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    return g @ z, np.ones(3) / 3.0


def _tensor_carry(t, b):
    n = b.shape[0]
    out = np.zeros((n, n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for a in range(n):
                    for c in range(n):
                        for d in range(n):
                            out[i, j, k] += b[i, a] * b[j, c] * b[k, d] * t[a, c, d]
    return out


def _record(checks, name, got, want, limit=ATOL):
    delta = np.asarray(got, dtype=np.float64) - np.asarray(want, dtype=np.float64)
    error = float(np.max(np.abs(delta))) if delta.size else 0.0
    passed = bool(np.isfinite(delta).all() and error <= limit)
    checks.append({"name": name, "pass": passed, "max_abs_error": error,
                   "limit": limit})


def evaluate():
    checks = []
    v, p, w1, w2, maps = _fixed_n4()
    n, count = v.shape
    if not all(np.min(b) < 0 < np.max(b) and np.ptp(b) > 0.5 for b in maps):
        raise AssertionError("fixed maps lost signed/nonconstant property")
    c, kpos, kneg = raw_cumulant_parts(v, p)
    fpos = direct_feed(kpos, w1, w2)
    fneg = direct_feed(kneg, w1, w2)
    direct = fpos + fneg
    p_birth = w1[:, None] * v
    h = (p_birth * p[None, :]) @ p_birth.T
    a = w2 * np.diag(c)
    fneg_a = np.zeros((n, n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                fneg_a[i, j, k] = -0.5 * (a[i] * h[j, k]
                                           + a[j] * h[i, k]
                                           + a[k] * h[i, j])
    fneg_pair = fneg - fneg_a
    sources = compact_sources(v, p, w1, w2)
    pos_source = compact_sources(v, p, w1, w2, "positive")
    a_source = compact_sources(v, p, w1, w2, "a_only")
    neg_source = compact_sources(v, p, w1, w2, "negative_pair")
    stages = []
    for step in range(3):
        tag = "birth" if step == 0 else "carry%d" % step
        pos_t = sum((compact_tensor(x) for x in pos_source), np.zeros((n, n, n)))
        a_t = sum((compact_tensor(x) for x in a_source), np.zeros((n, n, n)))
        neg_t = sum((compact_tensor(x) for x in neg_source), np.zeros((n, n, n)))
        total = sum((compact_tensor(x) for x in sources), np.zeros((n, n, n)))
        _record(checks, tag + "_positive", pos_t, fpos)
        _record(checks, tag + "_negative_a", a_t, fneg_a)
        _record(checks, tag + "_negative_pair", neg_t, fneg_pair)
        _record(checks, tag + "_negative_total", a_t + neg_t, fneg)
        _record(checks, tag + "_total", total, direct)
        d3, d21 = dense_slices(sources)
        _record(checks, tag + "_D3", d3, np.einsum("iii->i", direct))
        _record(checks, tag + "_D21", d21, np.einsum("iij->ij", direct))
        stages.append({"stage": tag,
                       "max_abs_feed": float(np.max(np.abs(direct))),
                       "max_D21_asymmetry": float(np.max(np.abs(d21 - d21.T)))})
        if step < 2:
            b = maps[step]
            fpos, fneg_a, fneg_pair = (_tensor_carry(t, b)
                                       for t in (fpos, fneg_a, fneg_pair))
            fneg, direct = fneg_a + fneg_pair, fpos + fneg_a + fneg_pair
            sources = [transport_dense(s, b, np.ones(n)) for s in sources]
            pos_source = [transport_dense(s, b, np.ones(n)) for s in pos_source]
            a_source = [transport_dense(s, b, np.ones(n)) for s in a_source]
            neg_source = [transport_dense(s, b, np.ones(n)) for s in neg_source]
    mutant_gap = float(np.max(np.abs(fpos + fneg_a - direct)))
    checks.append({"name": "dropped_negative_pair_mutant_refused",
                   "pass": mutant_gap > 1e-3, "gap": mutant_gap,
                   "minimum": 1e-3})
    # Exact rank-n shared join is a distinct check of the existing reference.
    birth_sources = compact_sources(v, p, w1, w2)
    shared = Shared()
    for s in birth_sources:
        shared = weighted_gram_join(shared, s, n, np.eye(n))
    joined_d3, joined_d21 = compact_slices(shared)
    _, birth_pos, birth_neg = raw_cumulant_parts(v, p)
    birth_feed = direct_feed(birth_pos + birth_neg, w1, w2)
    _record(checks, "full_rank_join_D3", joined_d3, np.einsum("iii->i", birth_feed))
    _record(checks, "full_rank_join_D21", joined_d21, np.einsum("iij->ij", birth_feed))
    for step, b in enumerate(maps, 1):
        shared = transport_shared(shared, b, np.ones(n))
        birth_feed = _tensor_carry(birth_feed, b)
        joined_d3, joined_d21 = compact_slices(shared)
        _record(checks, "full_rank_join_carry%d_D3" % step,
                joined_d3, np.einsum("iii->i", birth_feed))
        _record(checks, "full_rank_join_carry%d_D21" % step,
                joined_d21, np.einsum("iij->ij", birth_feed))
    for control, vc, wc in (("zero_w2", v, np.zeros(n)),
                            ("zero_source", np.zeros_like(v), w2)):
        _, kp, kn = raw_cumulant_parts(vc, p)
        oracle = direct_feed(kp + kn, w1, wc)
        srcs = compact_sources(vc, p, w1, wc)
        for step in range(3):
            name = control + ("_birth" if step == 0 else "_carry%d" % step)
            got = sum((compact_tensor(x) for x in srcs), np.zeros((n, n, n)))
            _record(checks, name + "_oracle", got, oracle)
            _record(checks, name + "_zero", got, np.zeros_like(got))
            if step < 2:
                b = maps[step]
                oracle = _tensor_carry(oracle, b)
                srcs = [transport_dense(s, b, np.ones(n)) for s in srcs]
    v3, p3 = _centered_equilateral()
    c3, kp3, kn3 = raw_cumulant_parts(v3, p3)
    f3 = direct_feed(kp3 + kn3, np.ones(3), np.ones(3))
    unfolding = f3[:, (0, 1, 0), (0, 1, 1)]
    sigma_c = np.linalg.svd(c3, compute_uv=False)
    sigma_f = np.linalg.svd(unfolding, compute_uv=False)
    cov_rank = int(np.sum(sigma_c > 1e-10))
    feed_rank = int(np.sum(sigma_f > 1e-10))
    det = float(np.linalg.det(unfolding))
    checks.append({"name": "equilateral_covariance_rank_2",
                   "pass": cov_rank == 2, "rank": cov_rank,
                   "singular_values": sigma_c.tolist(), "tolerance": 1e-10})
    checks.append({"name": "equilateral_feed_unfolding_rank_3",
                   "pass": feed_rank == 3, "rank": feed_rank,
                   "singular_values": sigma_f.tolist(), "tolerance": 1e-10})
    checks.append({"name": "equilateral_determinant_over_one",
                   "pass": abs(det) > 1.0, "determinant": det,
                   "minimum_abs": 1.0})
    _record(checks, "equilateral_fixed_C", c3,
            np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 2.0]]))
    _record(checks, "equilateral_fixed_unfolding", unfolding,
            np.array([[-2.25, -0.25, -0.25],
                      [-0.25, -2.25, -0.25],
                      [-2.5, -2.5, -1.0]]))
    two_dense_arrays = BIRTHS * 2 * N_LIVE * N_LIVE * 4
    transport_flops = USES * 4 * N_LIVE**3
    checks.append({"name": "literal_negative_storage_112MiB",
                   "pass": two_dense_arrays == FROZEN_BYTES,
                   "bytes": two_dense_arrays, "frozen_bytes": FROZEN_BYTES})
    checks.append({"name": "literal_negative_transport_450971566080",
                   "pass": transport_flops == FROZEN_TRANSPORT,
                   "conventional_flops": transport_flops,
                   "frozen_conventional_flops": FROZEN_TRANSPORT})
    checks.append({"name": "embedding_dimensions",
                   "pass": (birth_sources[1].A.shape == (n, n)
                            and birth_sources[1].P.shape == (n, n)
                            and birth_sources[0].Z.shape == (n, count)),
                   "negative_A_shape": list(birth_sources[1].A.shape),
                   "negative_Q_shape": list(birth_sources[1].P.shape),
                   "positive_residual_shape": list(birth_sources[0].Z.shape)})
    return {"kind": "truth-free abstract compact feed embedding and rank counterexample",
            "source_sha256": file_sha(__file__), "contract_commit": "60ab638",
            "checks": checks, "n_checks": len(checks),
            "n_failed": sum(not x["pass"] for x in checks),
            "pass": all(x["pass"] for x in checks), "stages": stages,
            "equilateral": {"covariance": c3.tolist(),
                             "unfolding": unfolding.tolist(),
                             "covariance_rank": cov_rank, "feed_rank": feed_rank,
                             "determinant": det},
            "conditional_cost": {"negative_dense_A_Q_bytes": two_dense_arrays,
                                 "negative_dense_transport_conventional_flops": transport_flops,
                                 "note": "conditional arithmetic only; excludes joins, QR, K4 bank, backbone and server residual"},
            "scope": "induced K3 feed for one discrete K4 source; no physical ReLU source or later increment"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    args = ap.parse_args()
    target = Path(args.json)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = evaluate()
    except Exception as exc:
        result = {"kind": "truth-free abstract compact feed embedding and rank counterexample",
                  "source_sha256": file_sha(__file__), "contract_commit": "60ab638",
                  "pass": False, "error_type": type(exc).__name__, "error": str(exc)}
    data = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    target.write_bytes(data)
    receipt = {"source_sha256": file_sha(__file__),
               "result_sha256": hashlib.sha256(data).hexdigest(),
               "pass": result["pass"]}
    target.with_name("result_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("%d/%d checks passed" %
          (result.get("n_checks", 0) - result.get("n_failed", 0),
           result.get("n_checks", 0)))
    if not result["pass"]:
        print("FAIL: " + result.get("error", repr([x for x in result.get("checks", []) if not x["pass"]])))
        return 1
    print("equilateral covariance rank=%d, feed rank=%d, determinant=%.12g" %
          (result["equilateral"]["covariance_rank"],
           result["equilateral"]["feed_rank"],
           result["equilateral"]["determinant"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
