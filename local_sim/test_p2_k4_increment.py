"""Fixed truth-free two-layer fourth-cumulant increment fixture.

This checks an *accounting identity* for one finite signed Gaussian cubature
measure.  The declared old carry is exactly linear, B^4 K(X); no nonlinear
closure, low-rank approximation, score, or challenge cost is inferred.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np

from test_p2_k4_signed_cubature import signed_legendre


RHO = 0.5
ORDERS = (32, 48)
FIRST = np.array([[0.8, -0.3], [-0.55, 1.4]])
IDENTITY = np.eye(2)
NEGATIVE_IDENTITY = -np.eye(2)
SECOND_MAPS = (
    ("hadamard", np.array([[1.0, 1.0], [1.0, -1.0]]) / math.sqrt(2.0)),
    ("signed", np.array([[0.7, -0.4], [-0.25, 1.2]])),
)
W1 = np.array([0.23, -0.61])
W2 = np.array([1.13, -0.47])
W3 = np.array([-0.31, 0.79])
ABS_GATE = 1e-10
CONVERGENCE_GATE = 1e-8
MATERIAL = 1e-6
MAGNITUDE_GATE = 1e-3


def set_partitions(indices: tuple[int, ...]) -> list[tuple[tuple[int, ...], ...]]:
    """Enumerate the 15 set partitions of four *positions*."""
    if not indices:
        return [()]
    head, tail = indices[0], indices[1:]
    answer = []
    for prior in set_partitions(tail):
        answer.append(((head,),) + prior)
        for position in range(len(prior)):
            changed = list(prior)
            changed[position] = (head,) + changed[position]
            answer.append(tuple(changed))
    return answer


PARTITIONS = set_partitions((0, 1, 2, 3))


def raw_partition_tensor(samples: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Fourth cumulant using uncentered moments and independent set partitions."""
    moment_cache: dict[tuple[int, ...], float] = {(): 1.0}

    def moment(labels: tuple[int, ...]) -> float:
        key = tuple(sorted(labels))
        if key not in moment_cache:
            product = np.ones(len(probabilities))
            for label in key:
                product = product * samples[label]
            moment_cache[key] = float(probabilities @ product)
        return moment_cache[key]

    out = np.empty((2, 2, 2, 2))
    for labels in itertools.product(range(2), repeat=4):
        total = 0.0
        for partition in PARTITIONS:
            term = (-1) ** (len(partition) - 1) * math.factorial(len(partition) - 1)
            for block in partition:
                term *= moment(tuple(labels[position] for position in block))
            total += term
        out[labels] = total
    return out


def centered(samples: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = samples - (samples @ probabilities)[:, None]
    c = (v * probabilities[None, :]) @ v.T
    return v, c


def fourth_factors(v: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    return np.einsum("s,is,js,ks,ls->ijkl", probabilities, v, v, v, v)


def pairings(c: np.ndarray) -> np.ndarray:
    return (np.einsum("ij,kl->ijkl", c, c)
            + np.einsum("ik,jl->ijkl", c, c)
            + np.einsum("il,jk->ijkl", c, c))


def transform(tensor: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.einsum("ia,jb,kc,ld,abcd->ijkl", matrix, matrix, matrix, matrix, tensor)


def component_tensor(components: tuple[tuple[float, np.ndarray, np.ndarray], ...],
                     probabilities: np.ndarray) -> np.ndarray:
    """Signed fourth factors AND signed same-law covariance pairings.

    Each component is (sign, centered factors, covariance). The covariance
    belongs to that component's measure; no mixed-product independence is
    assumed. The negative old component flips the sign of both terms.
    """
    result = np.zeros((2, 2, 2, 2))
    for sign, v, c in components:
        result += sign * fourth_factors(v, probabilities)
        result -= sign * pairings(c)
    return result


def transported_components(components: tuple[tuple[float, np.ndarray, np.ndarray], ...],
                           matrix: np.ndarray) -> tuple[tuple[float, np.ndarray, np.ndarray], ...]:
    return tuple((sign, matrix @ v, matrix @ c @ matrix.T) for sign, v, c in components)


def slices(tensor: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "D4": np.array([tensor[i, i, i, i] for i in range(2)]),
        "D22": np.array([[tensor[i, i, j, j] for j in range(2)] for i in range(2)]),
        "D31": np.array([[tensor[i, i, i, j] for j in range(2)] for i in range(2)]),
    }


def component_slices(components: tuple[tuple[float, np.ndarray, np.ndarray], ...],
                     probabilities: np.ndarray) -> dict[str, np.ndarray]:
    d4 = np.zeros(2)
    d22 = np.zeros((2, 2))
    d31 = np.zeros((2, 2))
    for sign, v, c in components:
        v2 = v * v
        d4 += sign * ((v2 * v2) @ probabilities - 3 * np.diag(c) ** 2)
        d22 += sign * ((v2 * probabilities[None, :]) @ v2.T
                         - np.outer(np.diag(c), np.diag(c)) - 2 * c * c)
        d31 += sign * (((v2 * v) * probabilities[None, :]) @ v.T
                         - 3 * np.diag(c)[:, None] * c)
    return {"D4": d4, "D22": d22, "D31": d31}


def oriented_covariance(d22: np.ndarray, d31: np.ndarray) -> np.ndarray:
    result = np.outer(W2, W2) * d22 / 4
    result += np.outer(W3, W1) * d31 / 6
    result += np.outer(W1, W3) * d31.T / 6
    np.fill_diagonal(result, 0)
    return result


def dense_covariance(tensor: np.ndarray) -> np.ndarray:
    result = np.zeros((2, 2))
    for i, j in ((0, 1), (1, 0)):
        result[i, j] = (W2[i] * W2[j] * tensor[i, i, j, j] / 4
                        + W3[i] * W1[j] * tensor[i, i, i, j] / 6
                        + W1[i] * W3[j] * tensor[i, j, j, j] / 6)
    return result


def factor_feed(components: tuple[tuple[float, np.ndarray, np.ndarray], ...],
                probabilities: np.ndarray) -> np.ndarray:
    """The declared one-quadratic/two-linear K4-to-K3 feed, not full response."""
    out = np.zeros((2, 2, 2))
    for sign, v, c in components:
        for i, j, k in itertools.product(range(2), repeat=3):
            placements = (
                (W2[i] * W1[j] * W1[k], i, i, j, k),
                (W1[i] * W2[j] * W1[k], i, j, j, k),
                (W1[i] * W1[j] * W2[k], i, j, k, k),
            )
            for coefficient, a, b, d, e in placements:
                fourth = float(probabilities @ (v[a] * v[b] * v[d] * v[e]))
                covariance = c[a, b] * c[d, e] + c[a, d] * c[b, e] + c[a, e] * c[b, d]
                out[i, j, k] += sign * coefficient * (fourth - covariance) / 2
    return out


def dense_feed(tensor: np.ndarray) -> np.ndarray:
    out = np.empty((2, 2, 2))
    for i, j, k in itertools.product(range(2), repeat=3):
        out[i, j, k] = (W2[i] * W1[j] * W1[k] * tensor[i, i, j, k]
                        + W1[i] * W2[j] * W1[k] * tensor[i, j, j, k]
                        + W1[i] * W1[j] * W2[k] * tensor[i, j, k, k]) / 2
    return out


def check_near(checks: dict, name: str, actual: np.ndarray, expected: np.ndarray) -> None:
    a, e = np.asarray(actual), np.asarray(expected)
    if a.shape != e.shape:
        raise AssertionError(f"{name}: shape {a.shape} != {e.shape}")
    error = float(np.max(np.abs(a - e)))
    checks[name] = {"absolute_error": error,
                    "pass": bool(np.all(np.isfinite(a)) and error <= ABS_GATE)}


def check_convergence(checks: dict, name: str, coarse: np.ndarray, fine: np.ndarray) -> None:
    a, e = np.asarray(coarse), np.asarray(fine)
    if a.shape != e.shape:
        raise AssertionError(f"{name}: shape {a.shape} != {e.shape}")
    absolute = np.abs(a - e)
    material = np.abs(e) > MATERIAL
    relative = np.where(material, absolute / np.maximum(np.abs(e), MATERIAL), 0.0)
    nearzero = np.where(material, 0.0, absolute)
    r, z = float(np.max(relative)), float(np.max(nearzero))
    checks[name] = {"max_absolute_error": float(np.max(absolute)),
                    "max_relative_material": r, "max_absolute_nearzero": z,
                    "pass": bool(np.all(np.isfinite(a)) and r <= CONVERGENCE_GATE
                                 and z <= CONVERGENCE_GATE)}


def check_lower(checks: dict, name: str, value: float) -> None:
    checks[name] = {"value": value, "strict_lower_bound": MAGNITUDE_GATE,
                    "pass": bool(np.isfinite(value) and value > MAGNITUDE_GATE)}


def maxabs(value: np.ndarray) -> float:
    return float(np.max(np.abs(value)))


def evaluate_order(order: int, checks: dict) -> dict:
    x, p = signed_legendre(RHO, order)
    if x.shape != (2, (2 * order) ** 2) or p.shape != ((2 * order) ** 2,):
        raise AssertionError("signed full-grid shape changed")
    check_near(checks, f"order{order}_normalized", np.sum(p), 1.0)
    if np.any(p <= 0) or np.any(x < 0):
        raise AssertionError("full-sign weights must be positive and X rectified")

    vx, cx = centered(x, p)
    kx_raw = raw_partition_tensor(x, p)
    kx_factor = component_tensor(((1, vx, cx),), p)
    check_near(checks, f"order{order}_source_partition", kx_factor, kx_raw)
    cases = {}
    for name, b in (("nonconstant", FIRST), ("identity", IDENTITY),
                    ("negative_identity", NEGATIVE_IDENTITY)):
        linear = b @ x
        y = np.maximum(linear, 0)
        vlinear, clinear = centered(linear, p)
        vy, cy = centered(y, p)
        old_raw = transform(kx_raw, b)
        linear_raw = raw_partition_tensor(linear, p)
        total_raw = raw_partition_tensor(y, p)
        delta_raw = total_raw - old_raw
        old_component = ((1, vlinear, clinear),)
        total_component = ((1, vy, cy),)
        delta_component = ((1, vy, cy), (-1, vlinear, clinear))
        old_factor = component_tensor(old_component, p)
        total_factor = component_tensor(total_component, p)
        delta_factor = component_tensor(delta_component, p)
        prefix = f"order{order}_{name}"
        check_near(checks, f"{prefix}_linear_partition", linear_raw, old_raw)
        check_near(checks, f"{prefix}_old_factors", old_factor, old_raw)
        check_near(checks, f"{prefix}_later_partition", total_factor, total_raw)
        check_near(checks, f"{prefix}_delta_factors", delta_factor, delta_raw)
        check_near(checks, f"{prefix}_additive", old_factor + delta_factor, total_raw)

        if name == "identity":
            check_near(checks, f"{prefix}_zero_increment", delta_raw, np.zeros_like(delta_raw))
            check_lower(checks, f"{prefix}_naive_old_plus_whole_later_error", maxabs(old_raw))
        elif name == "negative_identity":
            check_near(checks, f"{prefix}_zero_later", total_raw, np.zeros_like(total_raw))
            check_near(checks, f"{prefix}_minus_old", delta_raw, -old_raw)
        else:
            check_lower(checks, f"{prefix}_nonzero_delta", maxabs(delta_raw))
            naive_error = maxabs(old_raw + total_raw - total_raw)
            check_lower(checks, f"{prefix}_naive_old_plus_whole_later_error", naive_error)

        map_cases = {}
        carried_delta = delta_component
        carried_total = total_component
        dense_delta = delta_raw
        dense_total = total_raw
        cumulative = np.eye(2)
        for map_name, matrix in SECOND_MAPS:
            cumulative = matrix @ cumulative
            carried_delta = transported_components(carried_delta, matrix)
            carried_total = transported_components(carried_total, matrix)
            dense_delta = transform(dense_delta, matrix)
            dense_total = transform(dense_total, matrix)
            map_prefix = f"{prefix}_{map_name}"
            check_near(checks, f"{map_prefix}_delta_tensor",
                       component_tensor(carried_delta, p), dense_delta)
            check_near(checks, f"{map_prefix}_total_tensor",
                       component_tensor(carried_total, p), dense_total)
            direct_total = raw_partition_tensor(cumulative @ y, p)
            direct_old = raw_partition_tensor(cumulative @ linear, p)
            check_near(checks, f"{map_prefix}_sequential_later_partition",
                       dense_total, direct_total)
            check_near(checks, f"{map_prefix}_sequential_delta_partition",
                       dense_delta, direct_total - direct_old)
            check_near(checks, f"{map_prefix}_additive",
                       direct_old + dense_delta, dense_total)
            for sign, v, c in carried_delta:
                check_near(checks, f"{map_prefix}_metric_{'later' if sign > 0 else 'old'}",
                           c, (v * p[None, :]) @ v.T)
            for kind, parts, target in (("delta", carried_delta, dense_delta),
                                        ("total", carried_total, dense_total)):
                readouts = component_slices(parts, p)
                direct = slices(target)
                for family in ("D4", "D22", "D31"):
                    check_near(checks, f"{map_prefix}_{kind}_{family}",
                               readouts[family], direct[family])
                check_near(checks, f"{map_prefix}_{kind}_oriented_covariance",
                           oriented_covariance(readouts["D22"], readouts["D31"]),
                           dense_covariance(target))
                check_near(checks, f"{map_prefix}_{kind}_quadratic_K3_feed",
                           factor_feed(parts, p), dense_feed(target))
            map_cases[map_name] = {
                "delta_D4": slices(dense_delta)["D4"].tolist(),
                "delta_D22": slices(dense_delta)["D22"].tolist(),
                "delta_D31": slices(dense_delta)["D31"].tolist(),
                "delta_oriented_covariance": dense_covariance(dense_delta).tolist(),
                "delta_quadratic_K3_feed": dense_feed(dense_delta).tolist(),
                "total_D4": slices(dense_total)["D4"].tolist(),
                "total_D22": slices(dense_total)["D22"].tolist(),
                "total_D31": slices(dense_total)["D31"].tolist(),
                "total_oriented_covariance": dense_covariance(dense_total).tolist(),
                "total_quadratic_K3_feed": dense_feed(dense_total).tolist(),
            }
        cases[name] = {"mean_later": (y @ p).tolist(), "old_tensor": old_raw.tolist(),
                       "total_tensor": total_raw.tolist(), "delta_tensor": delta_raw.tolist(),
                       "maps": map_cases}
    return {"source_tensor": kx_raw.tolist(), "cases": cases, "node_count": len(p)}


def run() -> dict:
    checks: dict = {}
    by_order = {order: evaluate_order(order, checks) for order in ORDERS}
    coarse, fine = (by_order[order] for order in ORDERS)
    check_convergence(checks, "source_cumulant_convergence",
                      coarse["source_tensor"], fine["source_tensor"])
    for name in ("nonconstant", "identity", "negative_identity"):
        a, b = coarse["cases"][name], fine["cases"][name]
        for quantity in ("mean_later", "old_tensor", "total_tensor", "delta_tensor"):
            check_convergence(checks, f"{name}_{quantity}_convergence", a[quantity], b[quantity])
        for map_name, _ in SECOND_MAPS:
            for quantity in a["maps"][map_name]:
                check_convergence(checks, f"{name}_{map_name}_{quantity}_convergence",
                                  a["maps"][map_name][quantity],
                                  b["maps"][map_name][quantity])
    failures = [name for name, entry in checks.items() if not entry["pass"]]
    return {"status": "PASS" if not failures else "FAIL", "checks_count": len(checks),
            "failures": failures, "checks": checks, "orders": by_order,
            "gates": {"absolute_identity": ABS_GATE, "convergence": CONVERGENCE_GATE,
                      "material": MATERIAL, "strict_magnitude": MAGNITUDE_GATE},
            "scope": "fixed n=2 full-sign finite measure, linear old carry, and declared quadratic feed only"}


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
