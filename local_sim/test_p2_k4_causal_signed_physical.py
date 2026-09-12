"""Preregistered physical n=2 causal signed-law fixture (LOG 12d0dad).

This is a truth-free numerical oracle, not a compressed source or an estimator.
The scalar continuous reference does not call the finite-bank or cumulant paths.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


S0 = np.array([[1.0, 0.5], [0.5, 1.0]])
B = np.array([[0.8, -0.3], [-0.55, 1.4]])
D = np.array([[-0.65, 1.1], [0.9, -0.4]])
ORDERS = (32, 48, 64)
EXPS = tuple((a, k - a) for k in range(5) for a in range(k + 1))
LOW_EXPS = tuple(key for key in EXPS if 1 <= sum(key) <= 3)
CHAINS = {"P1": (None,), "P2": (None, B), "R2": (None,),
          "Q3": (None, D), "P3": (None, B, D), "R3": (None,)}


def branches(chain: tuple[np.ndarray | None, ...]) -> tuple[list[float], list[dict]]:
    """Find every intermediate ReLU row root from piecewise linear maps."""
    boundaries = [0.0, 2 * math.pi]
    records = []
    for stage, matrix in enumerate(chain):
        new = []
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            middle = (left + right) / 2
            direction = np.array([math.cos(middle), math.sin(middle)])
            current = np.eye(2)
            for previous in chain[:stage]:
                op = np.eye(2) if previous is None else previous
                pre = op @ current @ direction
                current = np.diag(pre > 0) @ op @ current
            op = np.eye(2) if matrix is None else matrix
            pre_rows = op @ current
            for row_index, row in enumerate(pre_rows):
                if np.linalg.norm(row) < 1e-14:
                    continue
                base = math.atan2(-row[0], row[1])
                for offset in range(-2, 4):
                    root = base + offset * math.pi
                    if left + 1e-13 < root < right - 1e-13:
                        new.append(root)
                        records.append({"stage": stage, "row": row_index,
                                        "root": root, "bracket": [left, right]})
        boundaries = sorted(set(boundaries + new))
    return boundaries, records


def scalar_map(theta: float, chain: tuple[np.ndarray | None, ...]) -> tuple[float, float]:
    """Scalar reference map, intentionally separate from vectorized bank mapping."""
    x, y = math.cos(theta), math.sin(theta)
    for matrix in chain:
        if matrix is not None:
            x, y = (float(matrix[0, 0] * x + matrix[0, 1] * y),
                    float(matrix[1, 0] * x + matrix[1, 1] * y))
        x, y = max(x, 0.0), max(y, 0.0)
    return x, y


def independent_roots(chain: tuple[np.ndarray | None, ...], records: list[dict]) -> dict:
    """Numerically solve each registered row zero, without the analytic root formula."""
    errors = []
    for item in records:
        stage, row = item["stage"], item["row"]
        matrix = np.eye(2) if chain[stage] is None else chain[stage]

        def preactivation(theta: float) -> float:
            x, y = scalar_map(theta, chain[:stage]) if stage else (math.cos(theta), math.sin(theta))
            return float(matrix[row, 0] * x + matrix[row, 1] * y)

        left, right = item["bracket"]
        # Stay inside the previous stage's one linear sector. Its endpoints
        # can belong to a zero plateau after a preceding ReLU.
        radius = min(item["root"] - left, right - item["root"]) / 4
        numerical = brentq(preactivation, item["root"] - radius,
                           item["root"] + radius,
                           xtol=5e-15, rtol=9e-16)
        errors.append(abs(numerical - item["root"]))
    return {"max_error": max(errors, default=0.0), "root_count": len(records),
            "analytic_roots": [item["root"] for item in records],
            "errors": errors}


def continuous_law(sigma: np.ndarray, chain: tuple[np.ndarray | None, ...],
                   sectors: list[float]) -> tuple[dict, dict]:
    """Independent one-dimensional adaptive polar integration, radial part exact."""
    determinant = float(np.linalg.det(sigma))
    inverse = np.linalg.inv(sigma)
    raw = {}
    bounds = {}
    calls = 0
    for a, b in EXPS:
        degree = a + b
        coefficient = (2 ** (degree / 2) * math.gamma(1 + degree / 2)
                       / (2 * math.pi * math.sqrt(determinant)))
        total, reported = 0.0, 0.0
        for left, right in zip(sectors[:-1], sectors[1:]):
            def integrand(theta: float) -> float:
                nonlocal calls
                calls += 1
                x, y = scalar_map(theta, chain)
                c, s = math.cos(theta), math.sin(theta)
                q = (inverse[0, 0] * c * c + 2 * inverse[0, 1] * c * s
                     + inverse[1, 1] * s * s)
                return (x ** a) * (y ** b) * q ** (-(degree + 2) / 2)
            value, error = quad(integrand, left, right, epsabs=2e-13,
                                epsrel=2e-13, limit=200)
            total += value
            reported += error
        raw[a, b] = coefficient * total
        bounds[a, b] = abs(coefficient) * reported
    return raw, {"reported_absolute_errors": keyed(bounds),
                 "maximum_reported_absolute_error": max(bounds.values()),
                 "integrand_evaluations": calls}


def radial_rule() -> tuple[np.ndarray, np.ndarray, dict]:
    moments = np.array([2 ** (k / 2) * math.gamma(1 + k / 2) for k in range(6)])
    h0 = np.array([[moments[i + j] for j in range(3)] for i in range(3)])
    h1 = np.array([[moments[i + j + 1] for j in range(3)] for i in range(3)])
    lower = np.linalg.cholesky(h0)
    whitening = np.linalg.inv(lower)
    nodes, eigenvectors = np.linalg.eigh(whitening @ h1 @ whitening.T)
    weights = eigenvectors[0] ** 2
    defects = [abs(float(weights @ nodes ** k) - moments[k]) for k in range(6)]
    return nodes, weights, {"nodes": nodes.tolist(), "weights": weights.tolist(),
                            "moment_defects": defects, "max_moment_defect": max(defects)}


def bank(sigma: np.ndarray, chain: tuple[np.ndarray | None, ...],
         sectors: list[float], order: int, nodes: np.ndarray,
         radial_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    abscissa, legendre_weights = np.polynomial.legendre.leggauss(order)
    inverse = np.linalg.inv(sigma)
    determinant = float(np.linalg.det(sigma))
    points, weights = [], []
    for left, right in zip(sectors[:-1], sectors[1:]):
        theta = left + (abscissa + 1) * (right - left) / 2
        angular_weights = legendre_weights * (right - left) / 2
        direction = np.stack((np.cos(theta), np.sin(theta)))
        q = np.einsum("in,ij,jn->n", direction, inverse, direction)
        gaussian = (direction[:, :, None] * nodes[None, None, :]
                    / np.sqrt(q)[None, :, None]).reshape(2, -1)
        p = (angular_weights[:, None] * radial_weights[None, :]
             / (2 * math.pi * math.sqrt(determinant) * q[:, None]))
        output = gaussian
        for matrix in chain:
            if matrix is not None:
                output = matrix @ output
            output = np.maximum(output, 0.0)
        points.append(output)
        weights.append(p.ravel())
    x, w = np.concatenate(points, axis=1), np.concatenate(weights)
    return x, w, {"mass": float(np.sum(w)), "mass_error": abs(float(np.sum(w)) - 1),
                  "minimum_weight": float(np.min(w)), "node_count": len(w),
                  "sector_count": len(sectors) - 1}


def keyed(raw: dict) -> dict[str, float]:
    return {f"{a}{b}": float(value) for (a, b), value in raw.items()}


def raw_vectorized(points: np.ndarray, weights: np.ndarray) -> dict:
    return {(a, b): float(weights @ (points[0] ** a * points[1] ** b))
            for a, b in EXPS}


def raw_atomwise(parts: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    """Independent direct enumeration; signed pieces remain on disjoint supports."""
    out = {}
    for a, b in EXPS:
        terms = []
        for points, weights in parts:
            terms.extend(float(weight) * float(points[0, i]) ** a
                         * float(points[1, i]) ** b
                         for i, weight in enumerate(weights))
        out[a, b] = math.fsum(terms)
    return out


def partitions(items: tuple[int, ...]) -> list[tuple[tuple[int, ...], ...]]:
    if not items:
        return [()]
    out = []
    for prior in partitions(items[1:]):
        out.append(((items[0],),) + prior)
        for index in range(len(prior)):
            changed = list(prior)
            changed[index] = (items[0],) + changed[index]
            out.append(tuple(changed))
    return out


PARTITIONS = {degree: partitions(tuple(range(degree))) for degree in (3, 4)}


def cumulant(raw: dict, degree: int) -> np.ndarray:
    tensor = np.empty((2,) * degree)
    for labels in itertools.product(range(2), repeat=degree):
        terms = []
        for partition in PARTITIONS[degree]:
            product = (-1) ** (len(partition) - 1) * math.factorial(len(partition) - 1)
            for block in partition:
                zeros = sum(labels[slot] == 0 for slot in block)
                product *= raw[zeros, len(block) - zeros]
            terms.append(product)
        tensor[labels] = math.fsum(terms)
    return tensor


def conditional(raw_q: dict, raw_delta: dict, degree: int) -> np.ndarray:
    """Set-partition polynomial difference, retaining all lower raw cross terms."""
    tensor = np.empty((2,) * degree)
    for labels in itertools.product(range(2), repeat=degree):
        terms = []
        for partition in PARTITIONS[degree]:
            coefficient = (-1) ** (len(partition) - 1) * math.factorial(len(partition) - 1)
            before = after = 1.0
            for block in partition:
                zeros = sum(labels[slot] == 0 for slot in block)
                key = zeros, len(block) - zeros
                before *= raw_q[key]
                after *= raw_q[key] + raw_delta[key]
            terms.append(coefficient * (after - before))
        tensor[labels] = math.fsum(terms)
    return tensor


def max_abs(values: np.ndarray | list[float]) -> float:
    return float(np.max(np.abs(values)))


def transport(tensor: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.einsum("ia,jb,kc,ld,abcd->ijkl", matrix, matrix, matrix, matrix, tensor)


class Gates:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def upper(self, name: str, value: float, threshold: float, *, strict: bool = False) -> None:
        passed = bool(math.isfinite(value) and (value < threshold if strict else value <= threshold))
        self.rows.append({"name": name, "value": float(value), "operator": "<" if strict else "<=",
                          "threshold": threshold, "pass": passed})

    def lower(self, name: str, value: float, threshold: float) -> None:
        self.rows.append({"name": name, "value": float(value), "operator": ">",
                          "threshold": threshold, "pass": bool(math.isfinite(value) and value > threshold)})


def run() -> dict:
    gates = Gates()
    root_data = {}
    # P2 and R2 deliberately include the later D roots in their finite banks.
    # Their continuous references still integrate the exact corresponding laws.
    for name, chain in CHAINS.items():
        roots, records = branches(chain)
        root_data[name] = {"boundaries": roots, "numerical": independent_roots(chain, records)}
        gates.upper(f"{name}_independent_root", root_data[name]["numerical"]["max_error"], 1e-12)
    c = 1 / 6 + math.sqrt(3) / (4 * math.pi)
    m1_exact = np.array([[0.5, c], [c, 0.5]])
    s2_exact = np.array([[0.365 - 0.48 * c, -0.43 + 1.285 * c],
                         [-0.43 + 1.285 * c, 1.13125 - 1.54 * c]])
    s2 = B @ m1_exact @ B.T
    gates.upper("S2_analytic", max_abs(s2 - s2_exact), 1e-12)
    gates.lower("S2_min_eigenvalue", float(np.linalg.eigvalsh(s2)[0]), 0.0)
    sigmas = {"P1": S0, "P2": S0, "R2": s2, "Q3": s2, "P3": S0}
    continuous = {}
    integration = {}
    for name in ("P1", "P2", "R2", "Q3", "P3"):
        continuous[name], integration[name] = continuous_law(sigmas[name], CHAINS[name],
                                                              root_data[name]["boundaries"])
        gates.upper(f"{name}_scalar_bound", integration[name]["maximum_reported_absolute_error"],
                    1e-10, strict=True)
    m1 = np.array([[continuous["P1"][2, 0], continuous["P1"][1, 1]],
                   [continuous["P1"][1, 1], continuous["P1"][0, 2]]])
    gates.upper("M1_analytic", max_abs(m1 - m1_exact), 1e-12)
    gates.upper("S2_from_integrated_M1", max_abs(B @ m1 @ B.T - s2_exact), 1e-12)
    m2 = np.array([[continuous["P2"][2, 0], continuous["P2"][1, 1]],
                   [continuous["P2"][1, 1], continuous["P2"][0, 2]]])
    s3 = D @ m2 @ D.T
    gates.lower("S3_min_eigenvalue", float(np.linalg.eigvalsh(s3)[0]), 0.0)
    sigmas["R3"] = s3
    continuous["R3"], integration["R3"] = continuous_law(s3, CHAINS["R3"],
                                                            root_data["R3"]["boundaries"])
    gates.upper("R3_scalar_bound", integration["R3"]["maximum_reported_absolute_error"],
                1e-10, strict=True)
    nodes, radial_weights, radial = radial_rule()
    gates.lower("radial_positive_node", float(np.min(nodes)), 0.0)
    gates.lower("radial_positive_weight", float(np.min(radial_weights)), 0.0)
    gates.upper("radial_degree_0_to_5", radial["max_moment_defect"], 1e-12)
    banks, diagnostics, numeric = {}, {}, {}
    for name in CHAINS:
        # Future-aware common sectors make independently built P2/P3 and R2/Q3
        # atoms align for an exact finite diagnostic. Each bank is created anew.
        donor = {"P2": "P3", "R2": "Q3"}.get(name, name)
        sectors = root_data[donor]["boundaries"]
        per_order = {}
        diagnostics[name] = {}
        for order in ORDERS:
            x, w, diag = bank(sigmas[name], CHAINS[name], sectors, order,
                              nodes, radial_weights)
            banks[name, order] = x, w
            diagnostics[name][str(order)] = diag
            per_order[order] = raw_vectorized(x, w)
            gates.lower(f"{name}_order{order}_positive_weight", diag["minimum_weight"], 0.0)
            gates.upper(f"{name}_order{order}_unrescaled_unit_mass", diag["mass_error"], 1e-10)
        reference = continuous[name]
        def gaps(left: dict, right: dict) -> dict:
            return {f"{a}{b}": abs(left[a, b] - right[a, b]) / max(1.0, abs(reference[a, b]))
                    for a, b in EXPS}
        comparisons = {"32_48": gaps(per_order[32], per_order[48]),
                       "48_64": gaps(per_order[48], per_order[64]),
                       "48_continuous": gaps(per_order[48], reference)}
        for comparison, values in comparisons.items():
            gates.upper(f"{name}_{comparison}_all_15", max(values.values()), 1e-8)
        numeric[name] = {"bank_raw": {str(n): keyed(raw) for n, raw in per_order.items()},
                         "continuous_raw": keyed(reference), "gaps": comparisons}
    # Exact references are derived from P2 independently of the bank. Check
    # that the finite, separately generated P2 bank reproduces M2 and S3.
    p2_raw = raw_vectorized(*banks["P2", 48])
    m2_bank = np.array([[p2_raw[2, 0], p2_raw[1, 1]],
                        [p2_raw[1, 1], p2_raw[0, 2]]])
    gates.upper("M2_bank_vs_scalar", max_abs(m2_bank - m2), 1e-10)
    gates.upper("S3_bank_vs_scalar", max_abs(D @ m2_bank @ D.T - s3), 1e-10)

    p2, wp2 = banks["P2", 48]
    r2, wr2 = banks["R2", 48]
    p3, wp3 = banks["P3", 48]
    q3, wq3 = banks["Q3", 48]
    r3, wr3 = banks["R3", 48]
    mapped_p2 = np.maximum(D @ p2, 0.0)
    mapped_r2 = np.maximum(D @ r2, 0.0)
    sources = {
        "Delta2": [(p2, wp2), (r2, -wr2)],
        "Delta3": [(q3, wq3), (r3, -wr3)],
        "O3": [(mapped_p2, wp2), (mapped_r2, -wr2)],
    }
    source_raw = {name: raw_atomwise(parts) for name, parts in sources.items()}
    for name, raw in source_raw.items():
        gates.upper(f"{name}_zero_mass", abs(raw[0, 0]), 1e-10)
    positive_parts = {"P2": [(p2, wp2)], "R2": [(r2, wr2)],
                      "P3": [(p3, wp3)], "Q3": [(q3, wq3)],
                      "R3": [(r3, wr3)]}
    oracle_raw = {name: raw_atomwise(parts) for name, parts in positive_parts.items()}
    fast_raw = {name: raw_vectorized(*banks[name, 48]) for name in positive_parts}
    direct_errors = {name: max(abs(fast_raw[name][key] - oracle_raw[name][key]) for key in EXPS)
                     for name in positive_parts}
    for name, error in direct_errors.items():
        gates.upper(f"{name}_direct_atom_raw", error, 1e-10)
    identity_errors = {
        "Delta2": max(abs(source_raw["Delta2"][key] -
                          (oracle_raw["P2"][key] - oracle_raw["R2"][key])) for key in EXPS),
        "Delta3": max(abs(source_raw["Delta3"][key] -
                          (oracle_raw["Q3"][key] - oracle_raw["R3"][key])) for key in EXPS),
        "O3": max(abs(source_raw["O3"][key] -
                     (oracle_raw["P3"][key] - oracle_raw["Q3"][key])) for key in EXPS),
        "P3_telescope": max(abs(oracle_raw["P3"][key] - oracle_raw["R3"][key]
                                 - source_raw["Delta3"][key] - source_raw["O3"][key])
                            for key in EXPS),
    }
    for name, error in identity_errors.items():
        gates.upper(f"{name}_all_raw_identity", error, 1e-10)
    contrasts = {}
    conditional_errors = {}
    for label, baseline, target, source in (
        ("new", "R3", "Q3", "Delta3"),
        ("old_given_new", "Q3", "P3", "O3"),
        ("layer2_old", "R2", "P2", "Delta2"),
    ):
        contrasts[label] = {}
        conditional_errors[label] = {}
        for degree in (3, 4):
            oracle = cumulant(oracle_raw[target], degree) - cumulant(oracle_raw[baseline], degree)
            fast = cumulant(fast_raw[target], degree) - cumulant(fast_raw[baseline], degree)
            increment = conditional(oracle_raw[baseline], source_raw[source], degree)
            error = max(max_abs(oracle - fast), max_abs(oracle - increment))
            gates.upper(f"{label}_K{degree}_all_oriented", error, 1e-10)
            contrasts[label][f"K{degree}"] = oracle.tolist()
            conditional_errors[label][f"K{degree}"] = error
    telescope = cumulant(oracle_raw["P3"], 4) - (cumulant(oracle_raw["R3"], 4)
                    + np.asarray(contrasts["new"]["K4"])
                    + np.asarray(contrasts["old_given_new"]["K4"]))
    gates.upper("K4_conditional_telescope", max_abs(telescope), 1e-10)
    old_k4 = np.asarray(contrasts["old_given_new"]["K4"])
    prior_k4 = np.asarray(contrasts["layer2_old"]["K4"])
    nonlinear_gap = max_abs(old_k4 - transport(prior_k4, D))
    prior_norm = max_abs(prior_k4)
    gates.lower("nonlinear_gap_absolute", nonlinear_gap, 1e-6)
    gates.lower("nonlinear_gap_relative_to_prior", nonlinear_gap, 1e-3 * prior_norm)
    linear_p2 = D @ p2
    linear_r2 = D @ r2
    linear_contrast = (cumulant(raw_atomwise([(linear_p2, wp2)]), 4)
                       - cumulant(raw_atomwise([(linear_r2, wr2)]), 4))
    linear_gap = max_abs(linear_contrast - transport(prior_k4, D))
    gates.upper("signed_linear_D_control", linear_gap, 1e-10)
    lower_leakage = max(abs(source_raw["O3"][key]) for key in LOW_EXPS)
    raw4_gap = max(abs(old_k4[labels] - source_raw["O3"][labels.count(0), labels.count(1)])
                   for labels in itertools.product(range(2), repeat=4))
    gates.lower("O3_lower_raw_nonzero", lower_leakage, 1e-8)
    gates.lower("O3_K4_differs_from_raw4", raw4_gap, 1e-6)
    zero_controls = {"Delta2": max_abs(conditional(oracle_raw["R2"],
                                                     {key: 0.0 for key in EXPS}, 4)),
                     "Delta3": max_abs(conditional(oracle_raw["R3"],
                                                     {key: 0.0 for key in EXPS}, 4))}
    for name, error in zero_controls.items():
        gates.upper(f"zeroed_{name}_contrast", error, 1e-10)
    return {
        "contract_commit": "12d0dad", "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "fixed physical n2 Gaussian/ReLU cascade; no labels, seed block, estimator or vault",
        "matrices": {"S0": S0.tolist(), "B": B.tolist(), "D": D.tolist(),
                     "M1_analytic": m1_exact.tolist(), "M1_scalar": m1.tolist(),
                     "S2": s2.tolist(), "M2_scalar": m2.tolist(), "S3": s3.tolist()},
        "roots": root_data, "radial": radial, "integration": integration,
        "positive_bank_diagnostics": diagnostics, "numeric": numeric,
        "positive_atom_raw": {name: keyed(raw) for name, raw in oracle_raw.items()},
        "signed_source_raw": {name: keyed(raw) for name, raw in source_raw.items()},
        "contrasts": contrasts, "conditional_errors": conditional_errors,
        "identity_errors": identity_errors, "nonlinear": {
            "old_gap": nonlinear_gap, "prior_K4_norm": prior_norm,
            "relative_gap": nonlinear_gap / prior_norm if prior_norm else math.inf,
            "linear_D_gap": linear_gap, "O3_lower_raw_max": lower_leakage,
            "O3_K4_vs_raw4_gap": raw4_gap, "zeroed_controls": zero_controls},
        "operation_counts": {"scalar_integrand_evaluations": sum(
            item["integrand_evaluations"] for item in integration.values()),
            "finite_bank_atoms_all_orders": sum(diag["node_count"]
                for law in diagnostics.values() for diag in law.values())},
        "gates": gates.rows, "n_pass": sum(row["pass"] for row in gates.rows),
        "n_fail": sum(not row["pass"] for row in gates.rows),
        "failures": [row for row in gates.rows if not row["pass"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "n_pass": result["n_pass"],
                      "n_fail": result["n_fail"], "failures": result["failures"]}))
    if result["n_fail"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
