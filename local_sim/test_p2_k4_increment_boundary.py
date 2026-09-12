"""Boundary-partitioned, truth-free two-layer K4 increment diagnostic.

The old carry is the declared linear B^4 K(X). This tests a two-coordinate
continuous source and exact signed accounting, not an estimator or score.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr

import test_p2_k4_increment as finite


ORDERS = (32, 48, 64)
RHO = finite.RHO
B = finite.FIRST
SIGMA = math.sqrt(1 - RHO * RHO)
LOW = 11 / 28
HIGH = 8 / 3
ANGLES = (0.0, math.atan(LOW), math.atan(HIGH), math.pi / 2)
MASSES = (1 / 3, 1 / 6, 1 / 6, 1 / 3)
DEGREES = tuple((a, degree - a) for degree in range(1, 5)
                for a in range(degree + 1))
MATERIAL = 1e-6
NUMERIC_GATE = 1e-8
IDENTITY_GATE = 1e-10


def radial_rule() -> tuple[np.ndarray, np.ndarray, dict]:
    mu = np.array([2 ** (k / 2) * math.gamma(1 + k / 2)
                   for k in range(6)])
    h0 = np.array([[mu[i + j] for j in range(3)] for i in range(3)])
    h1 = np.array([[mu[i + j + 1] for j in range(3)] for i in range(3)])
    l = np.linalg.cholesky(h0)
    inverse = np.linalg.inv(l)
    nodes, eigenvectors = np.linalg.eigh(inverse @ h1 @ inverse.T)
    weights = eigenvectors[0] ** 2
    defects = [abs(float(weights @ nodes**k) - mu[k]) for k in range(6)]
    return nodes, weights, {"nodes": nodes.tolist(), "weights": weights.tolist(),
                            "moment_defects": defects, "max_moment_defect": max(defects)}


def bank(order: int, radial: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray, dict]:
    gl_x, gl_w = np.polynomial.legendre.leggauss(order)
    u = (gl_x + 1) / 2
    axis_x = u / (1 - u)
    axis_w = gl_w / 2 * np.exp(-axis_x**2 / 2) / math.sqrt(2 * math.pi)
    axis_w *= ndtr(-RHO * axis_x / SIGMA) / (1 - u)**2
    # The mapped tail has positive density in real arithmetic; count weights
    # that underflow to zero in float64 separately from negative weights.
    underflow_count = int(np.count_nonzero(axis_w == 0))
    pre_axis = float(np.sum(axis_w))
    points = [np.array([[0.0], [0.0]])]
    weights = [np.array([MASSES[0]])]
    for i in range(2):
        coords = np.zeros((2, order))
        coords[i] = axis_x
        points.append(coords)
        weights.append(axis_w * (MASSES[1 + i] / pre_axis))
    radial_x, radial_w = radial
    quadrant_points = []
    quadrant_weights = []
    sector_masses = []
    for left, right in zip(ANGLES[:-1], ANGLES[1:]):
        theta = left + (gl_x + 1) * (right - left) / 2
        wtheta = gl_w * (right - left) / 2
        q = (1 - RHO * np.sin(2 * theta)) / (1 - RHO**2)
        radius = radial_x[None, :] / np.sqrt(q[:, None])
        quadrant_points.append(np.stack((radius * np.cos(theta[:, None]),
                                         radius * np.sin(theta[:, None])), axis=0).reshape(2, -1))
        qw = wtheta[:, None] * radial_w[None, :] / (2 * math.pi * SIGMA * q[:, None])
        quadrant_weights.append(qw.ravel())
        sector_masses.append(float(np.sum(qw)))
    pre_quadrant = sum(sector_masses)
    points.extend(quadrant_points)
    weights.extend(w * MASSES[3] / pre_quadrant for w in quadrant_weights)
    x = np.concatenate(points, axis=1)
    p = np.concatenate(weights)
    diagnostics = {"node_count": len(p), "pre_scaling_mass_error":
                   {"axis_0": pre_axis - MASSES[1], "axis_1": pre_axis - MASSES[2],
                    "quadrant": pre_quadrant - MASSES[3]},
                   "axis_tail_weight_underflows_per_axis": underflow_count,
                   "sector_pre_scaling_masses": sector_masses,
                   "support_masses": [float(weights[0].sum()), float(weights[1].sum()),
                                      float(weights[2].sum()),
                                      float(sum(w.sum() for w in weights[3:]))]}
    return x, p, diagnostics


def raw_moments(samples: np.ndarray, p: np.ndarray) -> dict[tuple[int, int], float]:
    return {(0, 0): 1.0, **{(a, b): float(p @ (samples[0]**a * samples[1]**b))
                          for a, b in DEGREES}}


def independent_partitions(items: tuple[int, ...]):
    if not items:
        return [()]
    result = []
    for prior in independent_partitions(items[1:]):
        result.append(((items[0],),) + prior)
        for i in range(len(prior)):
            copy = list(prior)
            copy[i] = (items[0],) + copy[i]
            result.append(tuple(copy))
    return result


PARTITIONS = independent_partitions((0, 1, 2, 3))


def partition_cumulant(raw: dict[tuple[int, int], float]) -> np.ndarray:
    result = np.empty((2, 2, 2, 2))
    for labels in itertools.product(range(2), repeat=4):
        total = 0.0
        for partition in PARTITIONS:
            term = (-1)**(len(partition) - 1) * math.factorial(len(partition) - 1)
            for block in partition:
                selected = [labels[i] for i in block]
                term *= raw[(selected.count(0), selected.count(1))]
            total += term
        result[labels] = total
    return result


def phi(value: float) -> float:
    return math.exp(-value * value / 2) / math.sqrt(2 * math.pi)


def truncated_powers(mu: float, left: float, right: float) -> np.ndarray:
    a, b = (left - mu) / SIGMA, (right - mu) / SIGMA
    pa = 0.0 if math.isinf(a) else phi(a)
    pb = 0.0 if math.isinf(b) else phi(b)
    if a > 0:
        j0 = ndtr(-a) - ndtr(-b)
    else:
        j0 = ndtr(b) - ndtr(a)
    j = [j0, pa - pb]
    for k in range(2, 5):
        edge_a = 0.0 if math.isinf(a) else a**(k - 1) * pa
        edge_b = 0.0 if math.isinf(b) else b**(k - 1) * pb
        j.append((k - 1) * j[k - 2] + edge_a - edge_b)
    return np.array([sum(math.comb(k, n) * mu**(k - n) * SIGMA**n * j[n]
                         for n in range(k + 1)) for k in range(5)])


def polynomial_coefficients(rows: np.ndarray, a: int, b: int) -> np.ndarray:
    out = np.array([1.0])
    for row, power in ((rows[0], a), (rows[1], b)):
        for _ in range(power):
            out = np.convolve(out, [row[0], row[1]])
    return out


def conditional_reference(tolerance: float) -> tuple[dict[str, dict[tuple[int, int], float]], float]:
    """One-dimensional Gaussian conditioning; no radial or discrete-bank calls."""
    first_axis = np.array([[1., 0.], [0., 0.]])
    second_axis = np.array([[0., 0.], [0., 1.]])
    sectors = ((-math.inf, 0.0, first_axis), (0.0, LOW, np.eye(2)),
               (LOW, HIGH, np.eye(2)), (HIGH, math.inf, np.eye(2)))
    laws = {}
    max_quad_error = 0.0
    for law in ("X", "U", "Y"):
        raw = {(0, 0): 1.0}
        for a, b in DEGREES:
            total = 0.0
            for sector, (low, high, xrows) in enumerate(sectors):
                urows = B @ xrows
                if law == "X":
                    rows = xrows
                elif law == "U":
                    rows = urows
                else:
                    active = ([1., 0.] if sector == 0 else
                              [1., 0.] if sector == 1 else
                              [1., 1.] if sector == 2 else [0., 1.])
                    rows = np.diag(active) @ urows
                coeff = polynomial_coefficients(rows, a, b)

                def integrand(z):
                    lower = low * z if math.isfinite(low) else -math.inf
                    upper = high * z if math.isfinite(high) else math.inf
                    powers = truncated_powers(RHO * z, lower, upper)
                    return phi(z) * sum(coeff[k] * z**(a + b - k) * powers[k]
                                        for k in range(len(coeff)))

                value, error = quad(integrand, 0, 12, epsabs=tolerance,
                                    epsrel=tolerance, limit=200)
                total += value
                max_quad_error = max(max_quad_error, error)
            xrows = second_axis
            urows = B @ xrows
            rows = xrows if law == "X" else urows if law == "U" else np.diag([0., 1.]) @ urows
            coefficient = rows[0, 1]**a * rows[1, 1]**b
            value, error = quad(lambda z: coefficient * z**(a + b) * phi(z)
                                * ndtr(-RHO * z / SIGMA), 0, 12,
                                epsabs=tolerance, epsrel=tolerance, limit=200)
            raw[(a, b)] = total + value
            max_quad_error = max(max_quad_error, error)
        laws[law] = raw
    return laws, max_quad_error


def numeric(checks: dict, name: str, actual, expected):
    a, e = np.asarray(actual), np.asarray(expected)
    absolute = np.abs(a - e)
    material = np.abs(e) > MATERIAL
    relative = np.where(material, absolute / np.maximum(np.abs(e), MATERIAL), 0.)
    nearzero = np.where(material, 0., absolute)
    checks[name] = {"max_absolute_error": float(np.max(absolute)),
                    "max_relative_material": float(np.max(relative)),
                    "max_absolute_nearzero": float(np.max(nearzero)),
                    "pass": bool(np.all(np.isfinite(a)) and
                                 np.max(relative) <= NUMERIC_GATE and
                                 np.max(nearzero) <= NUMERIC_GATE)}


def identity(checks: dict, name: str, actual, expected, gate=IDENTITY_GATE):
    error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
    checks[name] = {"max_absolute_error": error, "gate": gate,
                    "pass": bool(math.isfinite(error) and error <= gate)}


def lower(checks: dict, name: str, value: float):
    checks[name] = {"value": value, "strict_lower_bound": 1e-3,
                    "pass": bool(math.isfinite(value) and value > 1e-3)}


def evaluate(order: int, radial, checks: dict) -> dict:
    x, p, diagnostics = bank(order, radial)
    identity(checks, f"N{order}_node_count", len(p), 11 * order + 1, 0.)
    identity(checks, f"N{order}_normalization", p.sum(), 1.)
    for name, error in diagnostics["pre_scaling_mass_error"].items():
        identity(checks, f"N{order}_{name}_pre_mass", error, 0., 1e-8)
    for i, mass in enumerate(diagnostics["support_masses"]):
        identity(checks, f"N{order}_support_{i}", mass, MASSES[i], 1e-12)
    checks[f"N{order}_positive"] = {
        "underflow_count_per_axis": diagnostics["axis_tail_weight_underflows_per_axis"],
        "pass": bool(np.all(np.isfinite(p)) and np.all(p >= 0) and
                     np.all(x >= 0) and np.any(p > 0))}
    laws = {"X": x, "U": B @ x}
    laws["Y"] = np.maximum(laws["U"], 0)
    raw = {name: raw_moments(samples, p) for name, samples in laws.items()}
    cumulants = {name: partition_cumulant(moments) for name, moments in raw.items()}
    factors = {}
    for name, samples in laws.items():
        v, c = finite.centered(samples, p)
        factors[name] = ((1., v, c),)
        identity(checks, f"N{order}_{name}_partition", finite.component_tensor(factors[name], p), cumulants[name])
    identity(checks, f"N{order}_linear_carry", cumulants["U"], finite.transform(cumulants["X"], B))
    delta = cumulants["Y"] - cumulants["U"]
    delta_parts = (factors["Y"][0], (-1., *factors["U"][0][1:]))
    identity(checks, f"N{order}_delta_factors", finite.component_tensor(delta_parts, p), delta)
    identity(checks, f"N{order}_additive", cumulants["U"] + delta, cumulants["Y"])
    lower(checks, f"N{order}_nonzero_delta", finite.maxabs(delta))
    lower(checks, f"N{order}_naive_double_count", finite.maxabs(cumulants["U"]))

    maps = {}
    carried = {"total": factors["Y"], "increment": delta_parts}
    dense = {"total": cumulants["Y"], "increment": delta}
    cumulative = np.eye(2)
    for map_name, matrix in finite.SECOND_MAPS:
        cumulative = matrix @ cumulative
        block = {}
        for kind in ("total", "increment"):
            carried[kind] = finite.transported_components(carried[kind], matrix)
            dense[kind] = finite.transform(dense[kind], matrix)
            parts, tensor = carried[kind], dense[kind]
            prefix = f"N{order}_{map_name}_{kind}"
            identity(checks, prefix + "_tensor", finite.component_tensor(parts, p), tensor)
            direct = finite.raw_partition_tensor(cumulative @ laws["Y"], p)
            if kind == "increment":
                direct -= finite.raw_partition_tensor(cumulative @ laws["U"], p)
            identity(checks, prefix + "_direct_partition", tensor, direct)
            for sign, v, c in parts:
                identity(checks, prefix + f"_metric_{sign}", c, (v * p[None, :]) @ v.T)
            readout = finite.component_slices(parts, p)
            dense_slices = finite.slices(tensor)
            for family in ("D4", "D22", "D31"):
                identity(checks, prefix + "_" + family, readout[family], dense_slices[family])
            oriented = finite.oriented_covariance(readout["D22"], readout["D31"])
            feed = finite.factor_feed(parts, p)
            identity(checks, prefix + "_oriented_covariance", oriented, finite.dense_covariance(tensor))
            identity(checks, prefix + "_quadratic_K3_feed", feed, finite.dense_feed(tensor))
            block[kind] = {"D4": dense_slices["D4"].tolist(),
                           "D22": dense_slices["D22"].tolist(),
                           "D31": dense_slices["D31"].tolist(),
                           "oriented_covariance": finite.dense_covariance(tensor).tolist(),
                           "quadratic_K3_feed": finite.dense_feed(tensor).tolist()}
        identity(checks, f"N{order}_{map_name}_old_plus_delta", dense["total"],
                 finite.transform(cumulants["U"], cumulative) + dense["increment"])
        maps[map_name] = block
    return {"diagnostics": diagnostics,
            "raw": {name: {f"{a}{b}": value for (a, b), value in moments.items()}
                    for name, moments in raw.items()},
            "cumulants": {name: value.tolist() for name, value in cumulants.items()},
            "increment": delta.tolist(), "mean_later": (laws["Y"] @ p).tolist(),
            "maps": maps}


def run() -> dict:
    checks = {}
    nodes, weights, radial_diagnostics = radial_rule()
    for k, error in enumerate(radial_diagnostics["moment_defects"]):
        identity(checks, f"radial_moment_{k}", error, 0., 1e-12)
    checks["radial_positive"] = {"pass": bool(np.all(nodes > 0) and np.all(weights > 0))}
    cases = {n: evaluate(n, (nodes, weights), checks) for n in ORDERS}
    reference, ref_error = conditional_reference(1e-12)
    reference_loose, _ = conditional_reference(1e-10)
    ref_tensors = {name: partition_cumulant(raw) for name, raw in reference.items()}
    ref_delta = ref_tensors["Y"] - ref_tensors["U"]
    for name in reference:
        for moment in DEGREES:
            numeric(checks, f"reference_precision_{name}_{moment}",
                    reference[name][moment], reference_loose[name][moment])
    identity(checks, "reference_X_mean_0", reference["X"][(1, 0)], 1 / math.sqrt(2 * math.pi))
    identity(checks, "reference_X_mean_1", reference["X"][(0, 1)], 1 / math.sqrt(2 * math.pi))
    identity(checks, "reference_X_second_0", reference["X"][(2, 0)], 0.5)
    identity(checks, "reference_X_second_1", reference["X"][(0, 2)], 0.5)
    identity(checks, "reference_linear_carry", ref_tensors["U"], finite.transform(ref_tensors["X"], B))

    def compare(label, left, right):
        for name in ("X", "U", "Y"):
            for moment in DEGREES:
                key = f"{moment[0]}{moment[1]}"
                numeric(checks, f"{label}_raw_{name}_{key}", left["raw"][name][key], right["raw"][name][key])
            numeric(checks, f"{label}_K_{name}", left["cumulants"][name], right["cumulants"][name])
        numeric(checks, f"{label}_increment", left["increment"], right["increment"])
        numeric(checks, f"{label}_mean_later", left["mean_later"], right["mean_later"])
        for map_name, _ in finite.SECOND_MAPS:
            for kind in ("total", "increment"):
                for quantity in left["maps"][map_name][kind]:
                    numeric(checks, f"{label}_{map_name}_{kind}_{quantity}",
                            left["maps"][map_name][kind][quantity],
                            right["maps"][map_name][kind][quantity])

    for coarse, fine in zip(ORDERS[:-1], ORDERS[1:]):
        compare(f"N{coarse}_N{fine}", cases[coarse], cases[fine])
    ref_maps = {}
    cumulative = np.eye(2)
    for map_name, matrix in finite.SECOND_MAPS:
        cumulative = matrix @ cumulative
        ref_maps[map_name] = {}
        for kind, tensor in (("total", ref_tensors["Y"]), ("increment", ref_delta)):
            transported = finite.transform(tensor, cumulative)
            readout = finite.slices(transported)
            ref_maps[map_name][kind] = {**{key: value.tolist() for key, value in readout.items()},
                                        "oriented_covariance": finite.dense_covariance(transported).tolist(),
                                        "quadratic_K3_feed": finite.dense_feed(transported).tolist()}
    ref_case = {"raw": {name: {f"{a}{b}": value for (a, b), value in raw.items()}
                        for name, raw in reference.items()},
                "cumulants": {name: tensor.tolist() for name, tensor in ref_tensors.items()},
                "increment": ref_delta.tolist(),
                "mean_later": [reference["Y"][(1, 0)], reference["Y"][(0, 1)]],
                "maps": ref_maps}
    for n in (48, 64):
        compare(f"N{n}_reference", cases[n], ref_case)
    for n in ORDERS:
        x, p, _ = bank(n, (nodes, weights))
        for name, matrix in (("identity", np.eye(2)), ("negative_identity", -np.eye(2))):
            u = matrix @ x
            y = np.maximum(u, 0)
            old = finite.raw_partition_tensor(u, p)
            later = finite.raw_partition_tensor(y, p)
            delta = later - old
            vu, cu = finite.centered(u, p)
            vy, cy = finite.centered(y, p)
            old_parts = ((1., vu, cu),)
            total_parts = ((1., vy, cy),)
            increment_parts = (total_parts[0], (-1., vu, cu))
            prefix = f"N{n}_{name}"
            identity(checks, prefix + "_old_factors", finite.component_tensor(old_parts, p), old)
            identity(checks, prefix + "_later_factors", finite.component_tensor(total_parts, p), later)
            identity(checks, prefix + "_increment_factors", finite.component_tensor(increment_parts, p), delta)
            identity(checks, prefix + "_additive", old + delta, later)
            if name == "identity":
                identity(checks, f"N{n}_identity_zero_increment", delta, 0.)
                lower(checks, f"N{n}_identity_naive_double_count", finite.maxabs(old))
            else:
                identity(checks, f"N{n}_negative_identity_zero_later", later, 0.)
                identity(checks, f"N{n}_negative_identity_minus_old", delta, -old)
            cumulative = np.eye(2)
            for map_name, next_map in finite.SECOND_MAPS:
                cumulative = next_map @ cumulative
                for kind, parts, tensor in (("total", total_parts, later),
                                            ("increment", increment_parts, delta)):
                    carried = finite.transported_components(parts, cumulative)
                    dense = finite.transform(tensor, cumulative)
                    label = f"{prefix}_{map_name}_{kind}"
                    identity(checks, label + "_tensor", finite.component_tensor(carried, p), dense)
                    direct = finite.raw_partition_tensor(cumulative @ y, p)
                    if kind == "increment":
                        direct -= finite.raw_partition_tensor(cumulative @ u, p)
                    identity(checks, label + "_direct", dense, direct)
                    readout, slices = finite.component_slices(carried, p), finite.slices(dense)
                    for family in ("D4", "D22", "D31"):
                        identity(checks, label + "_" + family, readout[family], slices[family])
                    identity(checks, label + "_covariance",
                             finite.oriented_covariance(readout["D22"], readout["D31"]),
                             finite.dense_covariance(dense))
                    identity(checks, label + "_feed", finite.factor_feed(carried, p),
                             finite.dense_feed(dense))
    failures = [name for name, entry in checks.items() if not entry["pass"]]
    return {"status": "PASS" if not failures else "FAIL", "checks_count": len(checks),
            "failures": failures, "checks": checks, "orders": cases,
            "radial": radial_diagnostics, "reference": ref_case,
            "reference_quad_reported_error": ref_error,
            "gates": {"numeric": NUMERIC_GATE, "material": MATERIAL,
                      "identity": IDENTITY_GATE, "pre_scaling_mass": 1e-8,
                      "support_and_radial": 1e-12, "discriminator_strict": 1e-3},
            "scope": "fixed two-coordinate Gaussian/ReLU law, linear old carry and declared feed only"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run()
    content = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8", newline="\n")
    else:
        print(content, end="")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
