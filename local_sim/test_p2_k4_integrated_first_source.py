"""Independent tiny dense oracle for the integrated first-source instrument."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from p2_k4_integrated_first_source import integrated_metered_program
from test_p2_k4_two_birth_compact import direct_birth_feed, direct_carry


LIMIT = 1e-8


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(checks, name, got, want, limit=LIMIT):
    got, want = np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64)
    delta = got - want
    scale = np.maximum(1.0, np.abs(want))
    error = float(np.max(np.abs(delta) / scale)) if delta.size else 0.0
    checks.append({"name": name, "pass": bool(np.isfinite(delta).all()
                    and error <= limit), "material_relative_error": error,
                   "limit": limit})


def dense_k4(v, p):
    n, count = v.shape
    covariance = np.zeros((n, n))
    fourth = np.zeros((n, n, n, n))
    for h in range(count):
        for i in range(n):
            for j in range(n):
                covariance[i, j] += p[h] * v[i, h] * v[j, h]
                for k in range(n):
                    for ell in range(n):
                        fourth[i, j, k, ell] += (
                            p[h] * v[i, h] * v[j, h] * v[k, h] * v[ell, h])
    k4 = np.zeros_like(fourth)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for ell in range(n):
                    k4[i, j, k, ell] = (
                        fourth[i, j, k, ell]
                        - covariance[i, j] * covariance[k, ell]
                        - covariance[i, k] * covariance[j, ell]
                        - covariance[i, ell] * covariance[j, k])
    d4 = np.array([k4[i, i, i, i] for i in range(n)])
    d22 = np.array([[k4[i, i, j, j] for j in range(n)] for i in range(n)])
    d31 = np.array([[k4[i, i, i, j] for j in range(n)] for i in range(n)])
    return d4, d22, d31, k4


def ordinary_tensor(source):
    """Dense index expansion of an ordinary compact Source's represented cubic."""
    a = np.asarray(source.A)
    p = np.asarray(source.P)
    z = np.asarray(source.Z)
    loading = np.asarray(source.L)
    s = np.asarray(source.s)
    e = np.asarray(source.e)
    w2 = np.asarray(source.w2)
    n, count = a.shape
    m = p * s[None, :] + 3.0 * a * e[None, :] + z @ loading.T
    out = np.zeros((n, n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for h in range(count):
                    out[i, j, k] += w2[h] * (
                        p[i, h] * a[j, h] * a[k, h]
                        + a[i, h] * p[j, h] * a[k, h]
                        + a[i, h] * a[j, h] * p[k, h])
                    out[i, j, k] += (
                        m[i, h] * p[j, h] * p[k, h]
                        + p[i, h] * m[j, h] * p[k, h]
                        + p[i, h] * p[j, h] * m[k, h]) / 3.0
    return out


def evaluate():
    raw = integrated_metered_program(
        n=4, depth=16, source_rank=3, rank=4, recent=4,
        ordinary_residual=4, capture_tiny=True)
    checks = []
    traces = raw.pop("_trace")
    weights, _, _ = raw.pop("_inputs")
    expected_bases = []
    for layer, row in enumerate(traces):
        full_k4 = np.zeros((4, 4, 4, 4))
        if layer == 0:
            record(checks, "L0_zero_D4", row["D4"], np.zeros(4))
        else:
            d4, d22, d31, full_k4 = dense_k4(row["V"], row["p"])
            record(checks, "L%d_D4" % layer, row["D4"], d4)
            if layer < 15:
                record(checks, "L%d_D22" % layer, row["D22"], d22)
                record(checks, "L%d_D31" % layer, row["D31"], d31)
                record(checks, "L%d_D31_other_orientation" % layer,
                       row["D31"].T, d31.T)
        ordinary = sum((ordinary_tensor(source) for kind, source in row["sources"]
                        if kind == "ordinary"), np.zeros((4, 4, 4)))
        direct_feed = np.zeros((4, 4, 4))
        for birth in range(1, layer):
            source_row = traces[birth]
            feed = direct_birth_feed(source_row["V"], source_row["p"],
                                     source_row["w1"], source_row["w2"])
            for later in range(birth + 1, layer + 1):
                W = np.asarray(weights[later], dtype=np.float32)
                if later > birth + 1:
                    W = W * traces[later - 1]["w1"][None, :]
                feed = direct_carry(feed, W)
            direct_feed += feed
        full = ordinary + direct_feed
        want_d3 = np.array([full[i, i, i] for i in range(4)])
        record(checks, "L%d_D3" % layer, row["D3"], want_d3)
        want_mu = (row["base"] + want_d3 * row["w3"] / math.factorial(3)
                   + np.array([full_k4[i, i, i, i] for i in range(4)])
                   * row["w4"] / math.factorial(4))
        record(checks, "L%d_mean_consumers" % layer, row["mu"], want_mu)
        if layer < 15:
            want_d21 = np.array([[full[i, i, j] for j in range(4)]
                                 for i in range(4)])
            record(checks, "L%d_oriented_D21" % layer,
                   row["D21"], want_d21)
            w = {1: row["w1"], 2: row["w2"], 3: row["w3"]}
            want_k11 = np.zeros((4, 4))
            for i in range(4):
                for j in range(4):
                    co = row["C_off"][i, j]
                    value = (co * w[1][i] * w[1][j]
                             + 0.5 * co * co * w[2][i] * w[2][j])
                    for degree, tensor in ((3, full), (4, full_k4)):
                        for left in range(1, degree):
                            right = degree - left
                            if left <= 3 and right <= 3:
                                indices = (i,) * left + (j,) * right
                                value += (w[left][i] * w[right][j]
                                          * tensor[indices]
                                          / (math.factorial(left)
                                             * math.factorial(right)))
                    want_k11[i, j] = value
            record(checks, "L%d_cov_Taylor_consumers" % layer,
                   row["K11"], want_k11)
            want_c = np.diag(row["var_out"]) + want_k11 * (
                np.ones((4, 4)) - np.eye(4))
            record(checks, "L%d_cov_assemble" % layer, row["C"], want_c)
        ordinary_count = sum(kind == "ordinary" for kind, _ in row["sources"])
        positive_count = sum(kind == "feed_positive" for kind, _ in row["sources"])
        negative_count = sum(kind == "feed_negative" for kind, _ in row["sources"])
        expected = (layer, max(0, layer - 1), max(0, layer - 1))
        checks.append({"name": "L%d_source_ages_counts" % layer,
                       "pass": ((ordinary_count, positive_count,
                                 negative_count) == expected
                                and all(source.age >= 0 for _, source in row["sources"])),
                       "got": [ordinary_count, positive_count, negative_count],
                       "expected": list(expected)})
    schedule = raw["schedule"]
    fixed = {"k4_uses": 15, "k4_full": 14, "k4_final": 1,
             "feed_births": 14, "ordinary_births": 15,
             "joins": 28, "join_ordinary": 10,
             "join_feed_positive": 9, "join_feed_negative": 9,
             "source_D3_uses": 330, "source_D21_uses": 287,
             "ordinary_D3_uses": 120, "ordinary_D21_uses": 105,
             "feed_positive_D3_uses": 105, "feed_negative_D3_uses": 105,
             "feed_positive_D21_uses": 91, "feed_negative_D21_uses": 91}
    for key, want in fixed.items():
        checks.append({"name": "schedule_" + key,
                       "pass": schedule[key] == want,
                       "got": schedule[key], "expected": want})
    forbidden = [key for key in raw["families"] if
                 key.startswith("dense_feed_D21") or
                 key in ("dense_shared_A_transport", "dense_feed_P_transport",
                         "dense_feed_a_transport", "dense_feed_Q_reconstruct")]
    checks.append({"name": "no_old_direct_feed_families",
                   "pass": not forbidden, "forbidden": forbidden})
    checks.append({"name": "one_common_weight_cast_per_layer",
                   "pass": raw["family_calls"].get("cast_weight_f32") == 16,
                   "calls": raw["family_calls"].get("cast_weight_f32")})
    checks.append({"name": "one_C_gram_per_full_K4_use",
                   "pass": raw["family_calls"].get("dense_k4_C_gram") == 14,
                   "calls": raw["family_calls"].get("dense_k4_C_gram")})
    checks.append({"name": "family_context_parity",
                   "pass": sum(raw["families"].values()) == raw["flops"],
                   "family_sum": sum(raw["families"].values()),
                   "context_flops": raw["flops"]})
    return {"kind": "integrated first-source independent tiny dense oracle",
            "instrument_sha256": sha(Path(__file__).with_name(
                "p2_k4_integrated_first_source.py")),
            "source_sha256": sha(__file__),
            "input_sha256": raw["input_sha256"],
            "checks": checks, "n_checks": len(checks),
            "n_failed": sum(not c["pass"] for c in checks),
            "pass": all(c["pass"] for c in checks),
            "meter": {k: v for k, v in raw.items() if k != "families"},
            "max_material_relative_error": max(
                (c["material_relative_error"] for c in checks
                 if "material_relative_error" in c), default=0.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    args = ap.parse_args()
    target = Path(args.json)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = evaluate()
    except Exception as exc:
        result = {"kind": "integrated first-source independent tiny dense oracle",
                  "source_sha256": sha(__file__), "pass": False,
                  "error_type": type(exc).__name__, "error": str(exc)}
    data = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    target.write_bytes(data)
    target.with_name("tiny_receipt.json").write_text(
        json.dumps({"source_sha256": sha(__file__),
                    "result_sha256": hashlib.sha256(data).hexdigest(),
                    "pass": result["pass"]}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print("%d/%d checks passed" %
          (result.get("n_checks", 0) - result.get("n_failed", 0),
           result.get("n_checks", 0)))
    if not result["pass"]:
        print("FAIL: " + result.get("error", repr(
            [c for c in result.get("checks", []) if not c["pass"]])))
        return 1
    print("tiny integrated FLOPs %d" % result["meter"]["flops"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
