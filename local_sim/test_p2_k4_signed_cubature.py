"""Tiny truth-free signed Gaussian cubature fourth-cumulant factor fixture.

The factor identity is exact for its finite weighted measure. Gaussian/ReLU
quadrature accuracy is checked separately against the prior continuous-pair
reference. No estimator, labels, seed registry, or protected data are used.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np

from test_p2_k4_mixed_source import cumulant_partition, integrate_raw_moments, tensor


TOL_IDENTITY = 1e-10
TOL_QUADRATURE = 1e-8
MATERIAL = 1e-8
HALF_ORDERS = (32, 48)
RHOS = (0.0, 0.5)
HADAMARD = np.array([[1.0, 1.0], [1.0, -1.0]]) / math.sqrt(2.0)
SECOND = np.array([[0.8, -0.3], [-0.55, 1.4]])


def signed_legendre(rho: float, half_order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(half_order)
    left = 5.0 * (nodes - 1.0)
    right = 5.0 * (nodes + 1.0)
    x = np.concatenate((left, right))
    dx = np.concatenate((5.0 * weights, 5.0 * weights))
    x0, x1 = np.meshgrid(x, x, indexing="ij")
    density = np.exp(-(x0*x0 + x1*x1 - 2.0*rho*x0*x1) / (2.0*(1.0-rho*rho)))
    density /= 2.0 * math.pi * math.sqrt(1.0-rho*rho)
    masses = (dx[:, None] * dx[None, :] * density).ravel()
    probabilities = masses / np.sum(masses)
    rectified = np.stack((np.maximum(x0.ravel(), 0.0), np.maximum(x1.ravel(), 0.0)))
    return rectified, probabilities


def centered_factors(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = y @ p
    v = y - mean[:, None]
    covariance = (v * p[None, :]) @ v.T
    return v, covariance


def factor_entry(v: np.ndarray, p: np.ndarray, c: np.ndarray,
                 i: int, j: int, k: int, l: int) -> float:
    fourth = float(np.dot(p, v[i] * v[j] * v[k] * v[l]))
    return fourth - c[i, j]*c[k, l] - c[i, k]*c[j, l] - c[i, l]*c[j, k]


def factor_tensor(v: np.ndarray, p: np.ndarray, c: np.ndarray) -> np.ndarray:
    out = np.empty((2,)*4)
    for indices in itertools.product(range(2), repeat=4):
        out[indices] = factor_entry(v, p, c, *indices)
    return out


def raw_moments(y: np.ndarray, p: np.ndarray) -> dict[tuple[int, int], float]:
    raw = {(0, 0): 1.0}
    for total in range(1, 5):
        for a in range(total + 1):
            b = total-a
            raw[a, b] = float(np.dot(p, y[0]**a * y[1]**b))
    return raw


def partitions(items: tuple[int, ...]) -> list[tuple[tuple[int, ...], ...]]:
    if not items:
        return [()]
    first, rest = items[0], items[1:]
    result = []
    for partition in partitions(rest):
        result.append(((first,),) + partition)
        for location in range(len(partition)):
            updated = list(partition)
            updated[location] = (first,) + updated[location]
            result.append(tuple(updated))
    return result


FOUR_PARTITIONS = partitions((0, 1, 2, 3))


def partition_tensor(raw: dict[tuple[int, int], float]) -> np.ndarray:
    """Independent raw-moment set partitions of the weighted discrete measure."""
    out = np.empty((2,)*4)
    for labels in itertools.product(range(2), repeat=4):
        value = 0.0
        for blocks in FOUR_PARTITIONS:
            term = (-1.0)**(len(blocks)-1) * math.factorial(len(blocks)-1)
            for block in blocks:
                selected = [labels[position] for position in block]
                term *= raw[selected.count(0), selected.count(1)]
            value += term
        out[labels] = value
    return out


def transport_dense(source: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("ia,jb,kc,ld,abcd->ijkl", b, b, b, b, source)


def metric_tensor(g: np.ndarray, m: np.ndarray) -> np.ndarray:
    out = np.empty((2,)*4)
    for i, j, k, l in itertools.product(range(2), repeat=4):
        out[i, j, k, l] = (
            g[i, j]*m[k, l] + g[i, k]*m[j, l] + g[i, l]*m[j, k]
            + g[j, k]*m[i, l] + g[j, l]*m[i, k] + g[k, l]*m[i, j]
        ) / 6.0
    return out


def metric_split(source: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = 2.0*np.eye(2)
    h = np.einsum("ab,ijab->ij", np.linalg.inv(m), source)
    g = h - np.trace(np.linalg.inv(m) @ h)*m/8.0
    core = metric_tensor(g, m)
    return g, m, source-core


def factor_slices(v: np.ndarray, p: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v2 = v*v
    d4 = (v2*v2) @ p - 3.0*np.diag(c)**2
    d22 = (v2*p[None, :]) @ v2.T - np.outer(np.diag(c), np.diag(c)) - 2.0*c*c
    d31 = ((v2*v)*p[None, :]) @ v.T - 3.0*np.diag(c)[:, None]*c
    return d4, d22, d31


def dense_slices(source: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.array([source[i, i, i, i] for i in range(2)]),
        np.array([[source[i, i, j, j] for j in range(2)] for i in range(2)]),
        np.array([[source[i, i, i, j] for j in range(2)] for i in range(2)]),
    )


def factor_covariance(d22: np.ndarray, d31: np.ndarray,
                      w1: np.ndarray, w2: np.ndarray, w3: np.ndarray) -> np.ndarray:
    result = np.outer(w2, w2)*d22/4.0 + np.outer(w3, w1)*d31/6.0
    result += np.outer(w1, w3)*d31.T/6.0
    np.fill_diagonal(result, 0.0)
    return result


def dense_covariance(source: np.ndarray, w1: np.ndarray,
                     w2: np.ndarray, w3: np.ndarray) -> np.ndarray:
    result = np.zeros((2, 2))
    for i, j in ((0, 1), (1, 0)):
        result[i, j] = (
            w2[i]*w2[j]*source[i, i, j, j]/4.0
            + w3[i]*w1[j]*source[i, i, i, j]/6.0
            + w1[i]*w3[j]*source[i, j, j, j]/6.0
        )
    return result


def factor_feed(v: np.ndarray, p: np.ndarray, c: np.ndarray,
                w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    out = np.empty((2,)*3)
    for i, j, k in itertools.product(range(2), repeat=3):
        kiijk = float(np.dot(p, v[i]**2 * v[j] * v[k])) - c[i, i]*c[j, k] - 2*c[i, j]*c[i, k]
        kijjk = float(np.dot(p, v[i] * v[j]**2 * v[k])) - c[j, j]*c[i, k] - 2*c[i, j]*c[j, k]
        kijkk = float(np.dot(p, v[i] * v[j] * v[k]**2)) - c[k, k]*c[i, j] - 2*c[i, k]*c[j, k]
        out[i, j, k] = (
            w2[i]*w1[j]*w1[k]*kiijk
            + w1[i]*w2[j]*w1[k]*kijjk
            + w1[i]*w1[j]*w2[k]*kijkk
        ) / 2.0
    return out


def dense_feed(source: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    out = np.empty((2,)*3)
    for i, j, k in itertools.product(range(2), repeat=3):
        out[i, j, k] = (
            w2[i]*w1[j]*w1[k]*source[i, i, j, k]
            + w1[i]*w2[j]*w1[k]*source[i, j, j, k]
            + w1[i]*w1[j]*w2[k]*source[i, j, k, k]
        ) / 2.0
    return out


def near(name: str, actual: np.ndarray | float, expected: np.ndarray | float,
         tolerance: float, checks: dict) -> None:
    a, e = np.asarray(actual), np.asarray(expected)
    if a.shape != e.shape:
        raise ValueError(f"{name}: shape mismatch {a.shape} versus {e.shape}")
    error = np.abs(a-e)
    relative = np.where(np.abs(e) > MATERIAL, error/np.maximum(np.abs(e), MATERIAL), 0.0)
    nearzero = np.where(np.abs(e) > MATERIAL, 0.0, error)
    max_relative = float(np.max(relative))
    max_nearzero = float(np.max(nearzero))
    checks[name] = {"max_absolute_error": float(np.max(error)),
                    "max_relative_material": max_relative,
                    "max_absolute_nearzero": max_nearzero,
                    "pass": bool(np.all(np.isfinite(a)) and max_relative <= tolerance
                                 and max_nearzero <= tolerance)}


def lower(name: str, value: float, threshold: float, checks: dict) -> None:
    checks[name] = {"actual": value, "strict_lower_bound": threshold,
                    "pass": bool(value > threshold)}


def run() -> dict:
    checks: dict = {}
    cases: dict = {}
    w1 = np.array([0.23, -0.61])
    w2 = np.array([1.13, -0.47])
    w3 = np.array([-0.31, 0.79])
    for rho in RHOS:
        key = f"rho_{rho:g}"
        y32, p32 = signed_legendre(rho, HALF_ORDERS[0])
        y48, p48 = signed_legendre(rho, HALF_ORDERS[1])
        v32, c32 = centered_factors(y32, p32)
        v48, c48 = centered_factors(y48, p48)
        raw32, raw48 = raw_moments(y32, p32), raw_moments(y48, p48)
        source32 = factor_tensor(v32, p32, c32)
        source48 = factor_tensor(v48, p48, c48)
        partition32 = partition_tensor(raw32)
        partition48 = partition_tensor(raw48)
        near(f"{key}_same_measure_partition_order32", source32, partition32,
             TOL_IDENTITY, checks)
        near(f"{key}_same_measure_partition", source48, partition48, TOL_IDENTITY, checks)
        for moment in sorted(raw48):
            near(f"{key}_raw_{moment[0]}{moment[1]}_convergence",
                 raw32[moment], raw48[moment], TOL_QUADRATURE, checks)
        near(f"{key}_cumulant_convergence", source32, source48, TOL_QUADRATURE, checks)
        reference = tensor(integrate_raw_moments(rho, 96), cumulant_partition)
        for a in range(5):
            indices = (0,)*a + (1,)*(4-a)
            near(f"{key}_k{a}{4-a}_continuous_reference", source48[indices],
                 reference[indices], TOL_QUADRATURE, checks)

        g, m, residual = metric_split(source48)
        near(f"{key}_metric_source_reconstruction", metric_tensor(g, m)+residual,
             source48, TOL_IDENTITY, checks)
        maps = (("hadamard", HADAMARD), ("second", SECOND))
        v, c, dense = v48, c48, partition48
        carried_g, carried_m, carried_residual = g, m, residual
        map_results = {}
        wrong_orientation_gap = 0.0
        for map_name, b in maps:
            v = b @ v
            c = b @ c @ b.T
            dense = transport_dense(dense, b)
            carried_g = b @ carried_g @ b.T
            carried_m = b @ carried_m @ b.T
            carried_residual = transport_dense(carried_residual, b)
            prefix = f"{key}_{map_name}"
            near(f"{prefix}_full_tensor", factor_tensor(v, p48, c), dense,
                 TOL_IDENTITY, checks)
            for label, actual, expected in zip(("d4", "d22", "d31"),
                                               factor_slices(v, p48, c), dense_slices(dense)):
                near(f"{prefix}_{label}", actual, expected, TOL_IDENTITY, checks)
            covariance = factor_covariance(*factor_slices(v, p48, c)[1:], w1, w2, w3)
            near(f"{prefix}_oriented_covariance", covariance,
                 dense_covariance(dense, w1, w2, w3), TOL_IDENTITY, checks)
            near(f"{prefix}_covariance_symmetric", covariance, covariance.T,
                 TOL_IDENTITY, checks)
            d22, d31 = factor_slices(v, p48, c)[1:]
            wrong_covariance = (np.outer(w2, w2)*d22/4.0
                                + np.outer(w3, w1)*d31/6.0
                                + np.outer(w1, w3)*d31/6.0)
            np.fill_diagonal(wrong_covariance, 0.0)
            orientation_gap = float(np.max(np.abs(wrong_covariance-covariance)))
            wrong_orientation_gap = max(wrong_orientation_gap, orientation_gap)
            near(f"{prefix}_quadratic_k3_feed", factor_feed(v, p48, c, w1, w2),
                 dense_feed(dense, w1, w2), TOL_IDENTITY, checks)
            near(f"{prefix}_metric_with_residual", metric_tensor(carried_g, carried_m)
                 + carried_residual, dense, TOL_IDENTITY, checks)
            map_results[map_name] = {
                "matrix": b.tolist(), "d4": dense_slices(dense)[0].tolist(),
                "d22": dense_slices(dense)[1].tolist(),
                "d31": dense_slices(dense)[2].tolist(),
                "wrong_d31_orientation_covariance_gap": orientation_gap,
                "metric_residual_d4": dense_slices(carried_residual)[0].tolist(),
                "metric_residual_max_absolute": float(np.max(np.abs(carried_residual))),
            }

        # A central fourth moment is not a cumulant.
        raw_central = np.empty((2,)*4)
        for indices in itertools.product(range(2), repeat=4):
            raw_central[indices] = np.dot(p48, math.prod(v48[i] for i in indices))
        no_subtraction_gap = float(np.max(np.abs(raw_central-source48)))
        # Conditioning on the positive quadrant discards the axis/origin mass
        # created by rectification. It is a different probability measure.
        positive = np.logical_and(y48[0] > 0.0, y48[1] > 0.0)
        positive_p = p48[positive] / np.sum(p48[positive])
        positive_y = y48[:, positive]
        positive_v, positive_c = centered_factors(positive_y, positive_p)
        positive_source = factor_tensor(positive_v, positive_p, positive_c)
        omitted_source_gap = float(np.max(np.abs(positive_source-source48)))
        omitted_future_gap = float(np.max(np.abs(dense_slices(transport_dense(
            positive_source-source48, HADAMARD))[0])))
        if rho == 0.5:
            lower(f"{key}_no_covariance_subtraction_source_gap", no_subtraction_gap, 1e-3, checks)
            lower(f"{key}_axis_origin_omission_source_gap", omitted_source_gap, 1e-3, checks)
            lower(f"{key}_axis_origin_omission_future_d4_gap", omitted_future_gap, 1e-3, checks)
        cases[key] = {
            "rho": rho, "half_orders": list(HALF_ORDERS),
            "full_node_counts": [int((2*n)**2) for n in HALF_ORDERS],
            "raw_moments_order32": {f"{a}{b}": value for (a,b), value in raw32.items()},
            "raw_moments_order48": {f"{a}{b}": value for (a,b), value in raw48.items()},
            "cumulant_classes_order48": {f"k{a}{4-a}": float(source48[(0,)*a+(1,)*(4-a)])
                                         for a in range(5)},
            "normalization_order48": float(np.sum(p48)),
            "positive_quadrant_probability_order48": float(np.sum(p48[positive])),
            "discrete_covariance_order48": c48.tolist(),
            "metric_G": g.tolist(), "metric_M": m.tolist(),
            "metric_source_residual_max_absolute": float(np.max(np.abs(residual))),
            "maps": map_results,
            "negative_controls": {"no_covariance_subtraction_source_gap": no_subtraction_gap,
                                  "axis_origin_omission_source_gap": omitted_source_gap,
                                  "axis_origin_omission_future_d4_gap": omitted_future_gap,
                                  "wrong_d31_orientation_covariance_gap": wrong_orientation_gap},
        }
    failures = [name for name, check in checks.items() if not check["pass"]]
    return {"status": "PASS" if not failures else "FAIL", "checks_count": len(checks),
            "failures": failures, "checks": checks, "cases": cases,
            "identity_tolerance": TOL_IDENTITY, "quadrature_tolerance": TOL_QUADRATURE,
            "material_threshold": MATERIAL,
            "scope": "n=2 Gaussian/ReLU weighted discrete cumulant and two linear maps; no score, cost, or deployment claim"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run()
    serialized = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8", newline="\n")
    else:
        print(serialized, end="")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
