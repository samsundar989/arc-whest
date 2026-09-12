"""Tiny, truth-free dense oracle for fourth-cumulant component identities.

This checks exact linear transport and a declared quadratic-response truncation.
It does not implement a nonlinear ReLU closure, a scored estimator, or a cost model.
No registry seeds, protected labels, or frozen estimator arms are read.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np


TOL = 1e-12


def check(name: str, actual: np.ndarray, expected: np.ndarray, checks: dict) -> None:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape:
        raise AssertionError(f"{name}: shapes {actual.shape} != {expected.shape}")
    error = np.abs(actual - expected)
    scale = np.maximum(1.0, np.abs(expected))
    max_scaled = float(np.max(error / scale))
    max_absolute = float(np.max(error))
    nonzero = np.abs(expected) > TOL
    zero = ~nonzero
    max_relative = float(np.max(error[nonzero] / np.abs(expected[nonzero]))) if np.any(nonzero) else 0.0
    max_zero_absolute = float(np.max(error[zero])) if np.any(zero) else 0.0
    checks[name] = {"max_absolute_error": max_absolute, "max_scaled_error": max_scaled,
                    "max_relative_nonzero": max_relative,
                    "max_absolute_zero_or_near_zero": max_zero_absolute}
    if not np.all(np.isfinite(actual)) or max_relative > TOL or max_zero_absolute > TOL:
        raise AssertionError(
            f"{name}: nonzero relative {max_relative:.17g}, near-zero absolute "
            f"{max_zero_absolute:.17g}; tolerance {TOL}"
        )


def relu_fourth_cumulant(sigma: float) -> float:
    # Raw moments of max(0, N(0, sigma**2)); these are not central moments.
    m1 = sigma / math.sqrt(2.0 * math.pi)
    m2 = sigma**2 / 2.0
    m3 = sigma**3 * math.sqrt(2.0 / math.pi)
    m4 = 3.0 * sigma**4 / 2.0
    central4 = m4 - 4 * m1 * m3 + 6 * m1 * m1 * m2 - 3 * m1**4
    variance = m2 - m1 * m1
    return central4 - 3 * variance**2


def tensor_from_independent_coordinates(k: np.ndarray) -> np.ndarray:
    """Dense source, constructed separately from the transported factor path."""
    n = len(k)
    tensor = np.zeros((n,) * 4)
    for indices in itertools.product(range(n), repeat=4):
        if indices[0] == indices[1] == indices[2] == indices[3]:
            tensor[indices] = k[indices[0]]
    return tensor


def transport_dense(tensor: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = len(b)
    out = np.zeros((n,) * 4)
    for i, j, k, l in itertools.product(range(n), repeat=4):
        value = 0.0
        for a, c, d, e in itertools.product(range(n), repeat=4):
            value += b[i, a] * b[j, c] * b[k, d] * b[l, e] * tensor[a, c, d, e]
        out[i, j, k, l] = value
    return out


def factor_tensor(q: np.ndarray, k: np.ndarray) -> np.ndarray:
    n = len(q)
    out = np.zeros((n,) * 4)
    for i, j, h, l in itertools.product(range(n), repeat=4):
        out[i, j, h, l] = sum(
            k[a] * q[i, a] * q[j, a] * q[h, a] * q[l, a]
            for a in range(n)
        )
    return out


def dense_slices(tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(tensor)
    d4 = np.array([tensor[i, i, i, i] for i in range(n)])
    d22 = np.array([[tensor[i, i, j, j] for j in range(n)] for i in range(n)])
    d31 = np.array([[tensor[i, i, i, j] for j in range(n)] for i in range(n)])
    return d4, d22, d31


def factor_slices(q: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q2 = q * q
    return (
        (q2 * q2) @ k,
        (q2 * k) @ q2.T,
        ((q2 * q) * k) @ q.T,
    )


def covariance_from_dense(tensor: np.ndarray, w1: np.ndarray, w2: np.ndarray, w3: np.ndarray) -> np.ndarray:
    n = len(w1)
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                out[i, j] = (
                    w2[i] * w2[j] * tensor[i, i, j, j] / 4
                    + w3[i] * w1[j] * tensor[i, i, i, j] / 6
                    + w1[i] * w3[j] * tensor[i, j, j, j] / 6
                )
    return out


def covariance_from_slices(d22: np.ndarray, d31: np.ndarray, w1: np.ndarray, w2: np.ndarray, w3: np.ndarray) -> np.ndarray:
    out = (
        np.outer(w2, w2) * d22 / 4
        + np.outer(w3, w1) * d31 / 6
        + np.outer(w1, w3) * d31.T / 6
    )
    np.fill_diagonal(out, 0.0)  # The stated covariance formula is off-diagonal only.
    return out


def feed_from_dense(tensor: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    n = len(w1)
    out = np.zeros((n,) * 3)
    for i, j, k in itertools.product(range(n), repeat=3):
        out[i, j, k] = (
            w2[i] * w1[j] * w1[k] * tensor[i, i, j, k]
            + w1[i] * w2[j] * w1[k] * tensor[i, j, j, k]
            + w1[i] * w1[j] * w2[k] * tensor[i, j, k, k]
        ) / 2
    return out


def feed_from_factors(q: np.ndarray, cumulants: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    a = w2[:, None] * q**2
    p = w1[:, None] * q
    n = len(q)
    out = np.zeros((n,) * 3)
    for i, j, k in itertools.product(range(n), repeat=3):
        out[i, j, k] = sum(
            cumulants[h] * (a[i, h] * p[j, h] * p[k, h]
                            + p[i, h] * a[j, h] * p[k, h]
                            + p[i, h] * p[j, h] * a[k, h]) / 2
            for h in range(n)
        )
    return out


def metric_tensor(g: np.ndarray, m: np.ndarray) -> np.ndarray:
    n = len(g)
    out = np.zeros((n,) * 4)
    for i, j, k, l in itertools.product(range(n), repeat=4):
        out[i, j, k, l] = (
            g[i, j] * m[k, l] + g[i, k] * m[j, l] + g[i, l] * m[j, k]
            + g[j, k] * m[i, l] + g[j, l] * m[i, k] + g[k, l] * m[i, j]
        ) / 6
    return out


def metric_slices(g: np.ndarray, m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gd, md = np.diag(g), np.diag(m)
    return (
        gd * md,
        (np.outer(gd, md) + np.outer(md, gd) + 4 * g * m) / 6,
        (gd[:, None] * m + md[:, None] * g) / 2,
    )


def metric_feed_factors(g: np.ndarray, m: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    """The memo's S and S6 decomposition, independent of dense K indexing."""
    a = np.diag(w2)
    p = w1[:, None] * g
    r = w1[:, None] * m
    hm = w1[:, None] * m * w1[None, :]
    hg = w1[:, None] * g * w1[None, :]
    vg = w2 * np.diag(g)
    vm = w2 * np.diag(m)
    n = len(g)
    out = np.zeros((n,) * 3)
    for i, j, k in itertools.product(range(n), repeat=3):
        first = (
            vg[i] * hm[j, k] + vg[j] * hm[i, k] + vg[k] * hm[i, j]
            + vm[i] * hg[j, k] + vm[j] * hg[i, k] + vm[k] * hg[i, j]
        ) / 12
        second = sum(
            a[i, h] * p[j, h] * r[k, h]
            + a[i, h] * r[j, h] * p[k, h]
            + p[i, h] * a[j, h] * r[k, h]
            + r[i, h] * a[j, h] * p[k, h]
            + p[i, h] * r[j, h] * a[k, h]
            + r[i, h] * p[j, h] * a[k, h]
            for h in range(n)
        ) / 6
        out[i, j, k] = first + second
    return out


def signed_matrix(n: int) -> np.ndarray:
    return np.array([
        [(((-1) ** (i + 2 * j + i * j)) * (2 + i + 2 * j)) / (n + 3)
         + (0.37 if i == j else 0.0) for j in range(n)]
        for i in range(n)
    ], dtype=np.float64)


def independent_source_case(n: int, checks: dict) -> dict:
    # W0 is orthogonal row by row. Gaussian input therefore gives genuinely
    # independent preactivations; coordinatewise ReLU preserves independence.
    w0 = np.diag(np.sqrt(np.linspace(0.7, 1.8, n)))
    sigma = np.sqrt(np.diag(w0 @ w0.T))
    cumulants = np.array([relu_fourth_cumulant(float(s)) for s in sigma])
    source = tensor_from_independent_coordinates(cumulants)
    b = signed_matrix(n)
    q = b.copy()
    dense = transport_dense(source, b)
    check(f"source_n{n}_full_tensor", factor_tensor(q, cumulants), dense, checks)
    for label, actual, expected in zip(
        ("d4", "d22", "d31"), factor_slices(q, cumulants), dense_slices(dense)
    ):
        check(f"source_n{n}_{label}", actual, expected, checks)
    w1 = np.array([(-1) ** i * (0.2 + 0.11 * i) for i in range(n)])
    w2 = np.array([(-1) ** (i + 1) * (0.6 + 0.07 * i) for i in range(n)])
    w3 = np.array([(-1) ** (i // 2) * (0.15 + 0.09 * i) for i in range(n)])
    covariance = covariance_from_slices(*factor_slices(q, cumulants)[1:], w1, w2, w3)
    check(f"source_n{n}_oriented_covariance", covariance,
          covariance_from_dense(dense, w1, w2, w3), checks)
    check(f"source_n{n}_covariance_symmetric", covariance, covariance.T, checks)
    check(f"source_n{n}_quadratic_feed", feed_from_factors(q, cumulants, w1, w2),
          feed_from_dense(dense, w1, w2), checks)
    d4, d22, d31 = factor_slices(q, cumulants)
    feed = feed_from_dense(dense, w1, w2)
    repeated_iij = np.array([
        [w2[i] * w1[i] * w1[j] * d31[i, j]
         + w1[i]**2 * w2[j] * d22[i, j] / 2 for j in range(n)]
        for i in range(n)
    ])
    check(f"source_n{n}_feed_iij", repeated_iij,
          np.array([[feed[i, i, j] for j in range(n)] for i in range(n)]), checks)
    check(f"source_n{n}_feed_iii", 1.5 * w2 * w1**2 * d4,
          np.array([feed[i, i, i] for i in range(n)]), checks)
    # A second signed linear map confirms transport of a carried source, not
    # merely construction from Q=I at the first consumer.
    b2 = signed_matrix(n).T / 1.7
    check(f"source_n{n}_two_transports", factor_tensor(b2 @ q, cumulants),
          transport_dense(dense, b2), checks)
    p = np.eye(n)[::-1]
    check(f"source_n{n}_permutation", factor_tensor(p @ q, cumulants),
          transport_dense(dense, p), checks)
    t = -1.3
    check(f"source_n{n}_signed_scaling", factor_tensor(t * q, cumulants),
          t**4 * dense, checks)
    d31 = dense_slices(dense)[2]
    if np.max(np.abs(d31 - d31.T)) < 1e-5:
        raise AssertionError("signed source fixture failed to expose D31 orientation")
    wrong_covariance = (
        np.outer(w2, w2) * d22 / 4
        + np.outer(w3, w1) * d31 / 6
        + np.outer(w1, w3) * d31 / 6
    )
    np.fill_diagonal(wrong_covariance, 0.0)
    wrong_orientation_gap = float(np.max(np.abs(wrong_covariance - covariance)))
    if wrong_orientation_gap < 1e-5:
        raise AssertionError("signed source fixture did not expose wrong covariance orientation")
    return {"n": n, "cumulants": cumulants.tolist(),
            "d31_orientation_gap": float(np.max(np.abs(d31 - d31.T))),
            "wrong_covariance_orientation_gap": wrong_orientation_gap}


def metric_case(checks: dict) -> dict:
    g = np.array([[1.0, .2, -.4, .15], [.2, -.3, .7, -.11],
                  [-.4, .7, 2.0, .31], [.15, -.11, .31, .6]])
    m = np.array([[2.0, .1, .3, -.2], [.1, 1.3, -.2, .1],
                  [.3, -.2, .8, .05], [-.2, .1, .05, 1.1]])
    w1 = np.array([.2, .8, -.5, -.3])
    w2 = np.array([1.2, -.1, .4, .7])
    core = metric_tensor(g, m)
    for label, actual, expected in zip(("d4", "d22", "d31"), metric_slices(g, m), dense_slices(core)):
        check(f"metric_{label}", actual, expected, checks)
    check("metric_quadratic_feed", metric_feed_factors(g, m, w1, w2),
          feed_from_dense(core, w1, w2), checks)
    b = signed_matrix(4)
    gp, mp = b @ g @ b.T, b @ m @ b.T
    dense = transport_dense(core, b)
    check("metric_both_legs_full_tensor", metric_tensor(gp, mp), dense, checks)
    for label, actual, expected in zip(("d4", "d22", "d31"), metric_slices(gp, mp), dense_slices(dense)):
        check(f"metric_transported_{label}", actual, expected, checks)
    check("metric_transported_feed", metric_feed_factors(gp, mp, w1, w2),
          feed_from_dense(dense, w1, w2), checks)
    p = np.eye(4)[[2, 0, 3, 1]]
    check("metric_permutation", metric_tensor(p @ g @ p.T, p @ m @ p.T),
          transport_dense(core, p), checks)
    t = -1.3
    check("metric_signed_scaling", metric_tensor(t*t*g, t*t*m),
          t**4 * core, checks)
    fixed_metric_error = float(np.max(np.abs(metric_tensor(gp, m) - dense)))
    shortcut = np.diag(np.diag(gp)) + .27 * (b @ m @ b.T - np.diag(np.diag(b @ m @ b.T)))
    shortcut_error = float(np.max(np.abs(metric_tensor(shortcut, mp) - dense)))
    if fixed_metric_error < 1e-3 or shortcut_error < 1e-3:
        raise AssertionError("signed metric fixture did not expose either invalid shortcut")
    zero = np.zeros_like(g)
    check("metric_zero_g_tensor", metric_tensor(zero, mp), np.zeros_like(core), checks)
    check("metric_zero_g_feed", metric_feed_factors(zero, mp, w1, w2),
          np.zeros((4, 4, 4)), checks)
    return {"fixed_metric_error": fixed_metric_error,
            "offdiagonal_shortcut_error": shortcut_error}


def correlated_gaussian_zero(checks: dict) -> None:
    # Isserlis identity control: this is exact Gaussian algebra, not an
    # independent sample or empirical test of a nonlinear source model.
    a = signed_matrix(4)
    c = a @ a.T + np.eye(4) * .2
    cumulant = np.zeros((4,) * 4)
    for i, j, k, l in itertools.product(range(4), repeat=4):
        raw4 = c[i, j] * c[k, l] + c[i, k] * c[j, l] + c[i, l] * c[j, k]
        cumulant[i, j, k, l] = raw4 - (
            c[i, j] * c[k, l] + c[i, k] * c[j, l] + c[i, l] * c[j, k]
        )
    if abs(c[0, 1]) < .01:
        raise AssertionError("Gaussian fixture is insufficiently correlated")
    q = signed_matrix(4).T
    w1 = np.array([.2, .8, -.5, .4])
    w2 = np.array([1.2, -.1, .4, .7])
    check("correlated_gaussian_zero_full_k4", cumulant, np.zeros_like(cumulant), checks)
    check("correlated_gaussian_zero_factor", factor_tensor(q, np.zeros(4)), cumulant, checks)
    check("correlated_gaussian_zero_feed", feed_from_dense(cumulant, w1, w2),
          np.zeros((4, 4, 4)), checks)


def all_distinct_residual(checks: dict) -> dict:
    residual = np.zeros((4,) * 4)
    for indices in itertools.permutations(range(4)):
        residual[indices] = 1.0
    for label, current in zip(("d4", "d22", "d31"), dense_slices(residual)):
        check(f"all_distinct_present_{label}", current, np.zeros_like(current), checks)
    b = np.eye(4)
    b[0] = .5
    future = transport_dense(residual, b)
    check("all_distinct_future_d4", np.array([future[0, 0, 0, 0]]),
          np.array([1.5]), checks)
    check("all_distinct_dropped_slices_future", np.array([future[0, 0, 0, 0] - 0.0]),
          np.array([1.5]), checks)
    return {"present_max_slice": 0.0, "future_d4": float(future[0, 0, 0, 0]),
            "dropped_slice_prediction": 0.0}


def singular_support_case(checks: dict) -> dict:
    m = np.diag([2.0, 1.2, 0.0, 0.0])
    g = np.array([[.8, -.3, .4, 0], [-.3, 1.1, 0, -.2],
                  [.4, 0, 0, 0], [0, -.2, 0, 0]])
    source = metric_tensor(g, m)
    source[2, 2, 2, 2] += .4  # An unsupported source component is residual too.
    support = np.array([0, 1])
    ms = m[np.ix_(support, support)]
    ss = source[np.ix_(support, support, support, support)]
    inv = np.linalg.inv(ms)
    h = np.einsum("ab,ijab->ij", inv, ss)
    rank = 2
    recovered = 6 * h / (rank + 4) - 3 * np.trace(inv @ h) * ms / ((rank + 2) * (rank + 4))
    check("singular_support_recovered_g", recovered, g[np.ix_(support, support)], checks)
    g_projected = np.zeros((4, 4))
    g_projected[np.ix_(support, support)] = recovered
    projected = metric_tensor(g_projected, m)
    residual = source - projected
    check("singular_support_residual_zero_on_support",
          residual[np.ix_(support, support, support, support)], np.zeros((2,) * 4), checks)
    cross = float(residual[0, 0, 0, 2])
    if abs(cross) < .05 or abs(residual[1, 1, 1, 3]) < .03:
        raise AssertionError("singular fixture did not retain cross-support residual")
    check("singular_unsupported_component", np.array([residual[2, 2, 2, 2]]),
          np.array([.4]), checks)
    b = np.eye(4)
    b[0] = [.5, 0, .5, 0]
    full = transport_dense(source, b)
    carried = transport_dense(projected, b) + transport_dense(residual, b)
    check("singular_full_with_carried_residual", carried, full, checks)
    if abs(full[0, 0, 0, 0] - transport_dense(projected, b)[0, 0, 0, 0]) < .01:
        raise AssertionError("singular cross-support residual did not affect future D4")
    return {"rank": rank, "cross_support_residual_0002": cross,
            "future_d4_residual": float((transport_dense(residual, b))[0, 0, 0, 0])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write deterministic UTF-8 JSON result")
    args = parser.parse_args()
    checks: dict = {}
    result = {
        "scope": "truth-free deterministic algebra; exact linear components and declared quadratic truncation only",
        "gaussian_control": "Isserlis fourth-moment identity, not empirical evidence",
        "tolerance": TOL,
        "independent_sources": [independent_source_case(n, checks) for n in (4, 5)],
        "metric": metric_case(checks),
        "all_distinct_residual": all_distinct_residual(checks),
        "singular_metric": singular_support_case(checks),
    }
    correlated_gaussian_zero(checks)
    result["checks"] = checks
    result["check_count"] = len(checks)
    result["max_scaled_error"] = max(x["max_scaled_error"] for x in checks.values())
    result["max_relative_nonzero"] = max(x["max_relative_nonzero"] for x in checks.values())
    result["max_absolute_zero_or_near_zero"] = max(
        x["max_absolute_zero_or_near_zero"] for x in checks.values()
    )
    result["status"] = "PASS"
    serialized = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8", newline="\n")
    print(f"{result['status']}: {result['check_count']} checks, max scaled error {result['max_scaled_error']:.3g}")


if __name__ == "__main__":
    main()
