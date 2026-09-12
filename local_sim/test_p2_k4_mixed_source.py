"""Truth-free bivariate Gaussian/ReLU fourth-cumulant source integration.

Only two analytic Gaussian pairs are used. This does not read challenge labels,
seed registries, estimators, or the earlier dense oracle's source construction.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np


TOL = 1e-8
MATERIAL = 1e-8
RHO_VALUES = (0.0, 0.5)
ORDERS = (64, 96)
MAX_X = 10.0


def univariate_relu_moment(order: int) -> float:
    if order == 0:
        return 1.0
    return 2.0 ** (order / 2.0 - 1.0) * math.gamma((order + 1.0) / 2.0) / math.sqrt(math.pi)


def integrate_raw_moments(rho: float, order: int) -> dict[tuple[int, int], float]:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    x = (nodes + 1.0) * (MAX_X / 2.0)
    w = weights * (MAX_X / 2.0)
    marginal_density = np.exp(-x * x / 2.0) / math.sqrt(2.0 * math.pi)
    marginal = [1.0] + [float(np.dot(w, x**power * marginal_density)) for power in range(1, 5)]
    xy = x[:, None] * x[None, :]
    density = np.exp(-(x[:, None] ** 2 + x[None, :] ** 2 - 2.0 * rho * xy)
                     / (2.0 * (1.0 - rho * rho))) / (2.0 * math.pi * math.sqrt(1.0 - rho * rho))
    weighted = density * w[:, None] * w[None, :]
    moments: dict[tuple[int, int], float] = {(0, 0): 1.0}
    for total in range(1, 5):
        for a in range(total + 1):
            b = total - a
            if a == 0:
                value = marginal[b]
            elif b == 0:
                value = marginal[a]
            else:
                value = float(np.sum(weighted * x[:, None] ** a * x[None, :] ** b))
            moments[a, b] = value
    return moments


def raw(moments: dict[tuple[int, int], float], labels: tuple[int, ...]) -> float:
    return moments[labels.count(0), labels.count(1)]


def partitions(items: tuple[int, ...]) -> list[tuple[tuple[int, ...], ...]]:
    if not items:
        return [()]
    first, rest = items[0], items[1:]
    result = []
    for partition in partitions(rest):
        result.append(((first,),) + partition)
        for position in range(len(partition)):
            copy = list(partition)
            copy[position] = (first,) + copy[position]
            result.append(tuple(copy))
    return result


PARTITIONS_FOUR = partitions((0, 1, 2, 3))


def cumulant_partition(moments: dict[tuple[int, int], float], labels: tuple[int, ...]) -> float:
    total = 0.0
    for partition in PARTITIONS_FOUR:
        coefficient = (-1.0) ** (len(partition) - 1) * math.factorial(len(partition) - 1)
        term = coefficient
        for block in partition:
            term *= raw(moments, tuple(labels[position] for position in block))
        total += term
    return total


def cumulant_central(moments: dict[tuple[int, int], float], labels: tuple[int, ...]) -> float:
    means = (moments[1, 0], moments[0, 1])
    central = 0.0
    for mask in range(16):
        retained = tuple(labels[position] for position in range(4) if mask & (1 << position))
        omitted = tuple(labels[position] for position in range(4) if not mask & (1 << position))
        central += (-1.0) ** len(omitted) * raw(moments, retained) * math.prod(means[index] for index in omitted)

    def covariance(i: int, j: int) -> float:
        return raw(moments, (labels[i], labels[j])) - means[labels[i]] * means[labels[j]]

    return central - covariance(0, 1) * covariance(2, 3) - covariance(0, 2) * covariance(1, 3) - covariance(0, 3) * covariance(1, 2)


def tensor(moments: dict[tuple[int, int], float], method) -> np.ndarray:
    result = np.empty((2, 2, 2, 2), dtype=np.float64)
    for labels in itertools.product(range(2), repeat=4):
        result[labels] = method(moments, labels)
    return result


def transport(source: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.einsum("ia,jb,kc,ld,abcd->ijkl", matrix, matrix, matrix, matrix, source)


def metric_tensor(g: np.ndarray, m: np.ndarray) -> np.ndarray:
    result = np.empty((2, 2, 2, 2), dtype=np.float64)
    for i, j, k, l in itertools.product(range(2), repeat=4):
        result[i, j, k, l] = (
            g[i, j] * m[k, l] + g[i, k] * m[j, l] + g[i, l] * m[j, k]
            + g[j, k] * m[i, l] + g[j, l] * m[i, k] + g[k, l] * m[i, j]
        ) / 6.0
    return result


def contraction_matched(source: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = 2.0 * np.eye(2)
    inverse = np.linalg.inv(m)
    h = np.einsum("ab,ijab->ij", inverse, source)
    g = 6.0 * h / 6.0 - 3.0 * np.trace(inverse @ h) * m / (4.0 * 6.0)
    core = metric_tensor(g, m)
    return g, core, source - core


def near(name: str, actual: float, expected: float, checks: dict) -> None:
    error = abs(actual - expected)
    relative = error / abs(expected) if abs(expected) > MATERIAL else None
    passed = bool(np.isfinite(actual) and (relative <= TOL if relative is not None else error <= TOL))
    checks[name] = {"actual": float(actual), "expected": float(expected), "absolute_error": float(error),
                    "relative_error": float(relative) if relative is not None else None, "pass": passed}


def lower_bound(name: str, actual: float, threshold: float, checks: dict) -> None:
    checks[name] = {"actual": float(actual), "strict_lower_bound": threshold, "pass": bool(actual > threshold)}


def run() -> dict:
    checks: dict = {}
    cases: dict = {}
    b = np.array([[1.0, 1.0], [1.0, -1.0]]) / math.sqrt(2.0)
    for rho in RHO_VALUES:
        label = f"rho_{rho:g}"
        low = integrate_raw_moments(rho, ORDERS[0])
        high = integrate_raw_moments(rho, ORDERS[1])
        k_low = tensor(low, cumulant_partition)
        k_high = tensor(high, cumulant_partition)
        k_central = tensor(high, cumulant_central)
        for moment in high:
            if moment == (0, 0):
                continue
            near(f"{label}_raw_{moment[0]}{moment[1]}_convergence", low[moment], high[moment], checks)
            if 0 in moment:
                power = max(moment)
                near(f"{label}_marginal_{moment[0]}{moment[1]}", high[moment], univariate_relu_moment(power), checks)
        for a in range(5):
            indices = (0,) * a + (1,) * (4 - a)
            near(f"{label}_k_{a}{4-a}_convergence", k_low[indices], k_high[indices], checks)
            near(f"{label}_k_{a}{4-a}_central", k_central[indices], k_high[indices], checks)
        marginal_m = [univariate_relu_moment(power) for power in range(1, 5)]
        m1, m2, m3, m4 = marginal_m
        marginal_k4 = m4 - 4 * m3 * m1 - 3 * m2 * m2 + 12 * m2 * m1 * m1 - 6 * m1**4
        near(f"{label}_marginal_k4_0", k_high[0, 0, 0, 0], marginal_k4, checks)
        near(f"{label}_marginal_k4_1", k_high[1, 1, 1, 1], marginal_k4, checks)
        diagonal = np.zeros_like(k_high)
        diagonal[0, 0, 0, 0] = k_high[0, 0, 0, 0]
        diagonal[1, 1, 1, 1] = k_high[1, 1, 1, 1]
        g, core, residual = contraction_matched(k_high)
        near(f"{label}_core_plus_residual", float(np.max(np.abs(core + residual - k_high))), 0.0, checks)
        h_source = np.einsum("ab,ijab->ij", np.eye(2) / 2.0, k_high)
        h_core = np.einsum("ab,ijab->ij", np.eye(2) / 2.0, core)
        near(f"{label}_matched_contraction", float(np.max(np.abs(h_core - h_source))), 0.0, checks)
        full_future = transport(k_high, b)
        diagonal_future = transport(diagonal, b)
        core_future = transport(core, b)
        residual_future = transport(residual, b)
        near(f"{label}_future_core_plus_residual", float(np.max(np.abs(core_future + residual_future - full_future))), 0.0, checks)
        mixed = {"k1112": float(k_high[0, 0, 0, 1]), "k1122": float(k_high[0, 0, 1, 1]),
                 "k1222": float(k_high[0, 1, 1, 1])}
        d4_full = np.array([full_future[i, i, i, i] for i in range(2)])
        d4_diagonal = np.array([diagonal_future[i, i, i, i] for i in range(2)])
        d4_core = np.array([core_future[i, i, i, i] for i in range(2)])
        d4_residual = np.array([residual_future[i, i, i, i] for i in range(2)])
        if rho == 0.0:
            for name, value in mixed.items():
                near(f"{label}_{name}_zero", value, 0.0, checks)
            near(f"{label}_future_d4_diagonal_control", float(np.max(np.abs(d4_full - d4_diagonal))), 0.0, checks)
        else:
            lower_bound(f"{label}_k1122_positive", mixed["k1122"], 1e-3, checks)
            lower_bound(f"{label}_future_d4_diagonal_gap", float(np.max(np.abs(d4_full - d4_diagonal))), 1e-3, checks)
        cases[label] = {
            "rho": rho, "quadrature_orders": list(ORDERS), "domain": [0.0, MAX_X],
            "raw_moments_order96": {f"{a}{c}": value for (a, c), value in high.items()},
            "cumulants_order96": {f"k{a}{4-a}": float(k_high[(0,) * a + (1,) * (4-a)]) for a in range(5)},
            "mixed": mixed, "exact_univariate_k4": marginal_k4,
            "source_metric_M": (2.0 * np.eye(2)).tolist(), "contraction_matched_G": g.tolist(),
            "source_core_max_abs": float(np.max(np.abs(core))),
            "source_residual_max_abs": float(np.max(np.abs(residual))),
            "future_B": b.tolist(), "future_d4_full": d4_full.tolist(),
            "future_d4_diagonal_only": d4_diagonal.tolist(),
            "future_d4_full_minus_diagonal": (d4_full - d4_diagonal).tolist(),
            "future_d4_core": d4_core.tolist(), "future_d4_residual": d4_residual.tolist(),
            "future_d4_core_plus_residual": (d4_core + d4_residual).tolist(),
        }
    failures = [name for name, value in checks.items() if not value["pass"]]
    return {"status": "PASS" if not failures else "FAIL", "checks_count": len(checks),
            "failures": failures, "tolerance": TOL, "material_threshold": MATERIAL,
            "cases": cases, "checks": checks,
            "scope": "truth-free Gaussian-pair ReLU source and fixed linear transport; no score or deployment claim"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run()
    content = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    else:
        print(content, end="")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
