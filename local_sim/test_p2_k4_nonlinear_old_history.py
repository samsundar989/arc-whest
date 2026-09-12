"""Frozen truth-free three-layer/two-source nonlinear old-history fixture.

The older source is a signed *law*, not a K4-only closure. This executable
checks the preregistered contract in experiments/LOG.md at dca4e7f.
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

import test_p2_k4_increment_axis_gauss as axis


RHO = 0.5
B = np.array([[0.8, -0.3], [-0.55, 1.4]], dtype=np.float64)
D = np.array([[-0.65, 1.1], [0.9, -0.4]], dtype=np.float64)
W1 = np.array([0.7, -0.4])
W2 = np.array([1.1, 0.3])
W3 = np.array([-0.2, 0.9])
W4 = np.array([0.6, -0.5])
EXPS = tuple((a, degree - a) for degree in range(5)
             for a in range(degree + 1))
LOW_EXPS = tuple(pair for pair in EXPS if sum(pair) <= 3)
IDENTITY_GATE = 1e-10
MOMENT_GATE = 1e-11
NUMERIC_GATE = 1e-8


def maxabs(a: np.ndarray | list[float]) -> float:
    return float(np.max(np.abs(np.asarray(a))))


def push(matrix: np.ndarray, points: np.ndarray, *, relu: bool) -> np.ndarray:
    result = matrix @ points
    return np.maximum(result, 0.0) if relu else result


def raw_fast(points: np.ndarray, weights: np.ndarray) -> dict[tuple[int, int], float]:
    return {(a, b): float(weights @ (points[0] ** a * points[1] ** b))
            for a, b in EXPS}


def cumulants_fast(points: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = weights / np.sum(weights)
    mean = points @ p
    centered = points - mean[:, None]
    cov = np.einsum("in,jn,n->ij", centered, centered, p)
    k3 = np.einsum("in,jn,kn,n->ijk", centered, centered, centered, p)
    fourth = np.einsum("in,jn,kn,ln,n->ijkl", centered, centered,
                       centered, centered, p)
    pairings = (np.einsum("ij,kl->ijkl", cov, cov)
                + np.einsum("ik,jl->ijkl", cov, cov)
                + np.einsum("il,jk->ijkl", cov, cov))
    return k3, fourth - pairings


def partitions(items: tuple[int, ...]) -> list[tuple[tuple[int, ...], ...]]:
    if not items:
        return [()]
    out = []
    for prior in partitions(items[1:]):
        out.append(((items[0],),) + prior)
        for pos in range(len(prior)):
            copy = list(prior)
            copy[pos] = (items[0],) + copy[pos]
            out.append(tuple(copy))
    return out


PARTITIONS = {degree: partitions(tuple(range(degree))) for degree in (3, 4)}


def raw_atomwise(points: np.ndarray, weights: np.ndarray) -> dict[tuple[int, int], float]:
    """Oracle path: no vectorized moment or compact-source helper."""
    result = {}
    for a, b in EXPS:
        total = 0.0
        for index in range(len(weights)):
            total += float(weights[index]) * float(points[0, index]) ** a * float(points[1, index]) ** b
        result[a, b] = total
    return result


def cumulant_partitions(raw: dict[tuple[int, int], float], degree: int) -> np.ndarray:
    out = np.empty((2,) * degree)
    mass = raw[0, 0]
    for labels in itertools.product(range(2), repeat=degree):
        total = 0.0
        for partition in PARTITIONS[degree]:
            term = (-1) ** (len(partition) - 1) * math.factorial(len(partition) - 1)
            for block in partition:
                count0 = sum(labels[slot] == 0 for slot in block)
                term *= raw[count0, len(block) - count0] / mass
            total += term
        out[labels] = total
    return out


def project_source(points: np.ndarray, weights: np.ndarray, which: int) -> tuple[np.ndarray, dict]:
    """Weighted degree<=3 null projection, with no adaptive contrast choice."""
    columns = []
    for a, b in LOW_EXPS:
        column = points[0] ** a * points[1] ** b
        rms = math.sqrt(float(weights @ (column * column)))
        columns.append(column / rms)
    moments = np.stack(columns)
    root = np.sqrt(weights)
    design = root[:, None] * moments.T
    u, singular, _ = np.linalg.svd(design, full_matrices=False)
    rank = int(np.sum(singular > singular[0] * 1e-12))
    condition = float(singular[0] / singular[-1])
    if which == 1:
        numerator = points[0] * points[1] ** 3
    else:
        numerator = points[0] ** 2 * points[1] ** 2
    radius_four = (points[0] ** 2 + points[1] ** 2) ** 2
    contrast = numerator / (1.0 + radius_four)
    target = root * contrast
    residual = target - u[:, :rank] @ (u[:, :rank].T @ target)
    h = root * residual
    max_ratio = maxabs(h / weights)
    source = h / max_ratio if max_ratio > 0 else np.zeros_like(h)
    reference = weights + 0.2 * source
    raw_before = raw_fast(points, weights)
    raw_after = raw_fast(points, reference)
    matched = max(abs(raw_before[key] - raw_after[key]) for key in LOW_EXPS)
    diagnostics = {
        "rank": rank, "condition": condition, "singular_values": singular.tolist(),
        "projected_contrast_max_ratio": max_ratio,
        "scaled_null_max": maxabs(moments @ source),
        "raw_degree_le3_match_max": matched,
        "source_mass": float(np.sum(source)),
        "minimum_reference_weight": float(np.min(reference)),
        "minimum_relative_reference_weight": float(np.min(reference / weights)),
        "maximum_relative_change": maxabs(0.2 * source / weights),
    }
    return reference, diagnostics


def tensor_transport(tensor: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.einsum("ia,jb,kc,ld,abcd->ijkl", matrix, matrix, matrix, matrix, tensor)


def consumers_fast(k: np.ndarray) -> dict[str, np.ndarray]:
    d4 = np.array([k[i, i, i, i] for i in range(2)])
    d22 = np.array([[k[i, i, j, j] for j in range(2)] for i in range(2)])
    d31 = np.array([[k[i, i, i, j] for j in range(2)] for i in range(2)])
    mean = W4 * d4 / 24.0
    covariance = (np.outer(W2, W2) * d22 / 4.0
                  + np.outer(W3, W1) * d31 / 6.0
                  + np.outer(W1, W3) * d31.T / 6.0)
    feed = np.empty((2, 2, 2))
    for i, j, z in itertools.product(range(2), repeat=3):
        feed[i, j, z] = (W2[i] * W1[j] * W1[z] * k[i, i, j, z]
                         + W1[i] * W2[j] * W1[z] * k[i, j, j, z]
                         + W1[i] * W1[j] * W2[z] * k[i, j, z, z]) / 2.0
    return {"d4": d4, "d22": d22, "d31": d31, "mean": mean,
            "covariance": covariance, "feed": feed}


def consumers_oracle(k: np.ndarray) -> dict[str, np.ndarray]:
    """Explicit dense indexing; orientation is checked before symmetry."""
    result = {key: np.zeros(shape) for key, shape in
              (("d4", (2,)), ("d22", (2, 2)), ("d31", (2, 2)),
               ("mean", (2,)), ("covariance", (2, 2)), ("feed", (2, 2, 2)))}
    for i in range(2):
        result["d4"][i] = k[i, i, i, i]
        result["mean"][i] = W4[i] * k[i, i, i, i] / 24.0
        for j in range(2):
            result["d22"][i, j] = k[i, i, j, j]
            result["d31"][i, j] = k[i, i, i, j]
            result["covariance"][i, j] = (
                W2[i] * W2[j] * k[i, i, j, j] / 4.0
                + W3[i] * W1[j] * k[i, i, i, j] / 6.0
                + W1[i] * W3[j] * k[j, j, j, i] / 6.0)
            for z in range(2):
                result["feed"][i, j, z] = (
                    W2[i] * W1[j] * W1[z] * k[i, i, j, z]
                    + W1[i] * W2[j] * W1[z] * k[i, j, j, z]
                    + W1[i] * W1[j] * W2[z] * k[i, j, z, z]) / 2.0
    return result


def angular_roots(row: np.ndarray, lo: float, hi: float) -> list[float]:
    a, b = float(row[0]), float(row[1])
    if math.hypot(a, b) < 1e-14:
        return []
    base = math.atan2(-a, b)
    return [base + k * math.pi for k in range(-3, 5)
            if lo + 1e-13 < base + k * math.pi < hi - 1e-13]


def angular_sectors() -> list[tuple[float, float, np.ndarray, np.ndarray]]:
    """Independent polar oracle: split every nested preactivation zero."""
    base = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi]
    result = []
    for start, end in zip(base[:-1], base[1:]):
        mid = (start + end) / 2
        e = np.diag(np.maximum(np.array([math.cos(mid), math.sin(mid)]), 0.0) > 0)
        cuts_b = sorted([start, end] + [root for row in B @ e
                                        for root in angular_roots(row, start, end)])
        for lo, hi in zip(cuts_b[:-1], cuts_b[1:]):
            mid_b = (lo + hi) / 2
            direction = np.array([math.cos(mid_b), math.sin(mid_b)])
            y_map = np.diag((B @ e @ direction) > 0) @ B @ e
            cuts_d = sorted([lo, hi] + [root for row in D @ y_map
                                       for root in angular_roots(row, lo, hi)])
            for left, right in zip(cuts_d[:-1], cuts_d[1:]):
                mid_d = (left + right) / 2
                direction_d = np.array([math.cos(mid_d), math.sin(mid_d)])
                t_map = np.diag((D @ y_map @ direction_d) > 0) @ D @ y_map
                result.append((left, right, y_map, t_map))
    return result


def continuous_raw(sectors: list[tuple[float, float, np.ndarray, np.ndarray]],
                   layer: int) -> tuple[dict[tuple[int, int], float], dict]:
    """Original correlated Gaussian, analytic radius and scalar angle quad."""
    raw = {}
    bounds = {}
    normalized = 1.0 / (2.0 * math.pi * math.sqrt(1.0 - RHO * RHO))
    for a, b in EXPS:
        degree = a + b
        radial = 2.0 ** (degree / 2.0) * math.gamma((degree + 2.0) / 2.0)
        total = 0.0
        reported = 0.0
        doubled = 0.0
        for lo, hi, y_map, t_map in sectors:
            mapping = y_map if layer == 2 else t_map

            def integrand(theta: float) -> float:
                direction = np.array([math.cos(theta), math.sin(theta)])
                q = (1.0 - RHO * math.sin(2.0 * theta)) / (1.0 - RHO * RHO)
                transformed = mapping @ direction
                return (normalized * radial * transformed[0] ** a
                        * transformed[1] ** b * q ** (-(degree + 2.0) / 2.0))

            value, error = quad(integrand, lo, hi, epsabs=1e-13, epsrel=1e-13,
                                limit=200)
            loose, _ = quad(integrand, lo, hi, epsabs=1e-11, epsrel=1e-11,
                            limit=200)
            total += value
            reported += error
            doubled += abs(value - loose)
        raw[a, b] = total
        bounds[a, b] = max(reported, doubled)
    return raw, {"max_reported_or_doubling_bound": max(bounds.values()),
                 "per_moment_bounds": {f"{a}{b}": value for (a, b), value in bounds.items()}}


class Gates:
    def __init__(self) -> None:
        self.checks: list[dict] = []

    def upper(self, name: str, value: float, threshold: float, *, strict: bool = False) -> None:
        passed = value < threshold if strict else value <= threshold
        self.checks.append({"name": name, "value": float(value), "relation": "<" if strict else "<=",
                            "threshold": float(threshold), "pass": bool(passed)})

    def lower(self, name: str, value: float, threshold: float, *, strict: bool = True) -> None:
        passed = value > threshold if strict else value >= threshold
        self.checks.append({"name": name, "value": float(value), "relation": ">" if strict else ">=",
                            "threshold": float(threshold), "pass": bool(passed)})


def make_state(points: np.ndarray, weights: np.ndarray) -> dict:
    raw = raw_fast(points, weights)
    k3, k4 = cumulants_fast(points, weights)
    oracle_raw = raw_atomwise(points, weights)
    oracle_k3 = cumulant_partitions(oracle_raw, 3)
    oracle_k4 = cumulant_partitions(oracle_raw, 4)
    consumers = consumers_fast(k4)
    oracle_consumers = consumers_oracle(oracle_k4)
    return {"raw": raw, "k3": k3, "k4": k4, "oracle_raw": oracle_raw,
            "oracle_k3": oracle_k3, "oracle_k4": oracle_k4,
            "consumers": consumers, "oracle_consumers": oracle_consumers}


def contrasts(states: dict[str, dict], key: str) -> dict[str, np.ndarray]:
    values = {label: states[label][key] for label in ("00", "10", "01", "11")}
    return {"new": values["01"] - values["00"],
            "old_given_new": values["11"] - values["01"],
            "old_given_none": values["10"] - values["00"],
            "interaction": values["11"] - values["10"] - values["01"] + values["00"],
            "total": values["11"] - values["00"]}


def raw_contrasts(states: dict[str, dict], key: str) -> dict[str, np.ndarray]:
    values = {label: np.array([states[label][key][exp] for exp in EXPS])
              for label in ("00", "10", "01", "11")}
    return {"new": values["01"] - values["00"],
            "old_given_new": values["11"] - values["01"],
            "old_given_none": values["10"] - values["00"],
            "interaction": values["11"] - values["10"] - values["01"] + values["00"],
            "total": values["11"] - values["00"]}


def serialize_state(state: dict) -> dict:
    return {"raw": {f"{a}{b}": value for (a, b), value in state["raw"].items()},
            "k3": state["k3"].tolist(), "k4": state["k4"].tolist(),
            "consumers": {key: value.tolist() for key, value in state["consumers"].items()}}


def run() -> dict:
    gates = Gates()
    radial_x, radial_w, radial_diag = axis.radial_rule()
    axis_x, axis_w, axis_diag = axis.axis_rule()
    x48, p48, bank48 = axis.bank(48, (radial_x, radial_w), (axis_x, axis_w))
    x64, p64, bank64 = axis.bank(64, (radial_x, radial_w), (axis_x, axis_w))
    gates.upper("order48_node_count_defect", abs(len(p48) - 439), 0)
    gates.upper("order64_node_count_defect", abs(len(p64) - 583), 0)
    gates.lower("positive_bank_min_weight", float(min(np.min(p48), np.min(p64))), 0)
    r1, projection1 = project_source(x48, p48, 1)
    y48 = push(B, x48, relu=True)
    v = r1.copy()
    old2 = p48 - r1
    r2, projection2 = project_source(y48, v, 2)
    new2 = v - r2
    for name, projection in (("old", projection1), ("new", projection2)):
        gates.lower(f"{name}_rank", projection["rank"], 10, strict=False)
        gates.upper(f"{name}_condition", projection["condition"], 1e12)
        gates.lower(f"{name}_projected_contrast", projection["projected_contrast_max_ratio"], 0)
        gates.upper(f"{name}_scaled_null", projection["scaled_null_max"], MOMENT_GATE)
        gates.upper(f"{name}_moment_match", projection["raw_degree_le3_match_max"], MOMENT_GATE)
        gates.lower(f"{name}_reference_weight", projection["minimum_reference_weight"], 0)
        gates.upper(f"{name}_max_relative_change", projection["maximum_relative_change"], 0.2 + 1e-14)
    q = {"00": r2, "10": r2 + old2, "01": r2 + new2,
         "11": r2 + old2 + new2}
    gates.upper("q01_equals_v", maxabs(q["01"] - v), IDENTITY_GATE)
    gates.upper("q11_equals_p2", maxabs(q["11"] - p48), IDENTITY_GATE)
    for label, weights in q.items():
        gates.lower(f"q{label}_positive", float(np.min(weights)), 0)
        gates.upper(f"q{label}_mass", abs(float(np.sum(weights)) - 1.0), MOMENT_GATE)
    t48 = push(D, y48, relu=True)
    states = {"layer2": {label: make_state(y48, weights) for label, weights in q.items()},
              "layer3": {label: make_state(t48, weights) for label, weights in q.items()}}
    max_identity = 0.0
    max_order_identity = 0.0
    max_true_feed_difference = 0.0
    contrast_output = {}
    for layer_name, layer in states.items():
        layer_max = {"raw": 0.0, "k3": 0.0, "k4": 0.0, "consumers": 0.0}
        for state in layer.values():
            layer_max["raw"] = max(layer_max["raw"], max(abs(state["raw"][exp] - state["oracle_raw"][exp]) for exp in EXPS))
            layer_max["k3"] = max(layer_max["k3"], maxabs(state["k3"] - state["oracle_k3"]))
            layer_max["k4"] = max(layer_max["k4"], maxabs(state["k4"] - state["oracle_k4"]))
            for key in state["consumers"]:
                layer_max["consumers"] = max(layer_max["consumers"],
                                             maxabs(state["consumers"][key] - state["oracle_consumers"][key]))
        for key, error in layer_max.items():
            gates.upper(f"{layer_name}_{key}_oracle", error, IDENTITY_GATE)
        fast_raw = raw_contrasts(layer, "raw")
        oracle_raw = raw_contrasts(layer, "oracle_raw")
        ck3 = contrasts(layer, "k3")
        ok3 = contrasts(layer, "oracle_k3")
        ck4 = contrasts(layer, "k4")
        ok4 = contrasts(layer, "oracle_k4")
        max_contrast = 0.0
        max_consumers = 0.0
        per_source = {}
        for name in ck4:
            max_contrast = max(max_contrast, maxabs(fast_raw[name] - oracle_raw[name]),
                               maxabs(ck3[name] - ok3[name]), maxabs(ck4[name] - ok4[name]))
            from_tensor = consumers_fast(ck4[name])
            direct = consumers_oracle(ok4[name])
            for key in from_tensor:
                max_consumers = max(max_consumers, maxabs(from_tensor[key] - direct[key]))
            max_true_feed_difference = max(max_true_feed_difference,
                                           maxabs(ck3[name] - from_tensor["feed"]))
            per_source[name] = {"k3": ck3[name].tolist(), "k4": ck4[name].tolist(),
                                "consumers": {key: value.tolist() for key, value in from_tensor.items()}}
        gates.upper(f"{layer_name}_contrast_oracle", max_contrast, IDENTITY_GATE)
        gates.upper(f"{layer_name}_consumer_feed_oracle", max_consumers, IDENTITY_GATE)
        ordering = maxabs(ck4["old_given_new"] - ck4["old_given_none"] - ck4["interaction"])
        total = maxabs(ck4["total"] - ck4["new"] - ck4["old_given_new"])
        max_order_identity = max(max_order_identity, ordering, total)
        gates.upper(f"{layer_name}_source_assignment", max(ordering, total), IDENTITY_GATE)
        max_identity = max(max_identity, *layer_max.values(), max_contrast, max_consumers)
        contrast_output[layer_name] = per_source
    gates.upper("all_identity_max", max(max_identity, max_order_identity), IDENTITY_GATE)
    y64 = push(B, x64, relu=True)
    t64 = push(D, y64, relu=True)
    sectors = angular_sectors()
    continuous2, contdiag2 = continuous_raw(sectors, 2)
    continuous3, contdiag3 = continuous_raw(sectors, 3)
    numeric = {}
    for name, samples48, samples64, continuous in (
        ("layer2", y48, y64, continuous2), ("layer3", t48, t64, continuous3)):
        raw48 = raw_fast(samples48, p48)
        raw64 = raw_fast(samples64, p64)
        errors = {"48_64": {}, "48_continuous": {}}
        for exp in EXPS:
            denom = max(1.0, abs(continuous[exp]))
            errors["48_64"][f"{exp[0]}{exp[1]}"] = abs(raw48[exp] - raw64[exp]) / denom
            errors["48_continuous"][f"{exp[0]}{exp[1]}"] = abs(raw48[exp] - continuous[exp]) / denom
        for pair, values in errors.items():
            gates.upper(f"{name}_{pair}_raw", max(values.values()), NUMERIC_GATE)
        numeric[name] = {"raw48": {f"{a}{b}": value for (a, b), value in raw48.items()},
                         "raw64": {f"{a}{b}": value for (a, b), value in raw64.items()},
                         "continuous": {f"{a}{b}": value for (a, b), value in continuous.items()},
                         "errors": errors}
    gates.upper("continuous_layer2_bound", contdiag2["max_reported_or_doubling_bound"], 1e-10, strict=True)
    gates.upper("continuous_layer3_bound", contdiag3["max_reported_or_doubling_bound"], 1e-10, strict=True)
    before = contrasts(states["layer2"], "k4")["old_given_new"]
    after = contrasts(states["layer3"], "k4")["old_given_new"]
    linear_prediction = tensor_transport(before, D)
    g3 = maxabs(after - linear_prediction)
    old_scale = maxabs(before)
    gates.lower("nonlinear_gap_absolute", g3, 1e-6)
    gates.lower("nonlinear_gap_relative", g3, 1e-3 * old_scale)
    k1p = cumulants_fast(x48, p48)[1]
    k1r = cumulants_fast(x48, r1)[1]
    k2p = cumulants_fast(y48, p48)[1]
    k2r = cumulants_fast(y48, r1)[1]
    g2 = maxabs((k2p - k2r) - tensor_transport(k1p - k1r, B))
    low_leak = {}
    for name, points, wa, wb in (("layer2", y48, p48, r1),
                                 ("layer3", t48, q["11"], q["01"])):
        ra, rb = raw_fast(points, wa), raw_fast(points, wb)
        low_leak[name] = {f"{a}{b}": ra[a, b] - rb[a, b] for a, b in LOW_EXPS}
    linear_states = {label: make_state(D @ y48, weights) for label, weights in q.items()}
    linear_old = contrasts(linear_states, "k4")["old_given_new"]
    linear_gap = maxabs(linear_old - linear_prediction)
    gates.upper("signed_linear_D_control", linear_gap, IDENTITY_GATE)
    zero_old = {"00": r2, "10": r2, "01": r2 + new2, "11": r2 + new2}
    zero_new = {"00": r2, "10": r2 + old2, "01": r2, "11": r2 + old2}
    zero_errors = {}
    for name, weights_by_label, vanished in (("old", zero_old, "old_given_new"),
                                             ("new", zero_new, "new")):
        control = {label: make_state(t48, weights) for label, weights in weights_by_label.items()}
        error = maxabs(contrasts(control, "k4")[vanished])
        zero_errors[name] = error
        gates.upper(f"zero_{name}_control", error, IDENTITY_GATE)
    k_mut = after
    c = consumers_fast(k_mut)
    i, j = 0, 1
    swapped = (W2[i] * W2[j] * c["d22"][i, j] / 4
               + W3[i] * W1[j] * c["d31"][j, i] / 6
               + W1[i] * W3[j] * c["d31"][j, i] / 6)
    d31_mutant_gap = abs(swapped - c["covariance"][i, j])
    gates.lower("swapped_D31_mutant_detected", d31_mutant_gap, IDENTITY_GATE)
    gates.lower("linear_carry_mutant_detected", g3, 1e-6)
    interaction = contrasts(states["layer3"], "k4")["interaction"]
    interaction_oracle = contrasts(states["layer3"], "oracle_k4")["interaction"]
    mixed_gap = maxabs(interaction)
    mixed_reference_error = maxabs(interaction - interaction_oracle)
    gates.upper("mixed_interaction_oracle", mixed_reference_error, IDENTITY_GATE)
    dense = {name: {label: serialize_state(state) for label, state in layer.items()}
             for name, layer in states.items()}
    checks = gates.checks
    return {
        "fixture": "nonlinear_old_history", "scope": "truth_free_conditional_signed_law",
        "contract_commit": "dca4e7f", "status": "PASS" if all(item["pass"] for item in checks) else "FAIL",
        "checks_passed": sum(item["pass"] for item in checks), "checks_total": len(checks),
        "checks": checks, "frozen": {"rho": RHO, "B": B.tolist(), "D": D.tolist(),
                                  "w1": W1.tolist(), "w2": W2.tolist(),
                                  "w3": W3.tolist(), "w4": W4.tolist(),
                                  "orders": [48, 64], "relative_perturbation": 0.2},
        "bank": {"48": {"node_count": len(p48), "minimum_weight": float(np.min(p48)),
                        "diagnostics": bank48},
                 "64": {"node_count": len(p64), "minimum_weight": float(np.min(p64)),
                        "diagnostics": bank64},
                 "radial": radial_diag, "axis": axis_diag},
        "projections": {"old": projection1, "new": projection2},
        "counterfactual_min_weights": {key: float(np.min(value)) for key, value in q.items()},
        "dense": dense, "contrasts": contrast_output, "numeric": numeric,
        "continuous": {"sector_count": len(sectors), "layer2": contdiag2, "layer3": contdiag3},
        "nonlinear": {"layer2_gap": g2, "layer3_gap": g3,
                      "layer2_conditional_old_k4_max": old_scale,
                      "relative_gap": g3 / old_scale if old_scale else math.inf,
                      "signed_linear_D_gap": linear_gap,
                      "lower_moment_leakage": low_leak,
                      "true_k3_vs_declared_feed_max": max_true_feed_difference,
                      "interaction_max": mixed_gap,
                      "interaction_claimed_nonzero": mixed_gap > IDENTITY_GATE and mixed_reference_error <= IDENTITY_GATE,
                      "mixed_reference_error": mixed_reference_error,
                      "zero_control_errors": zero_errors,
                      "swapped_D31_mutant_gap": d31_mutant_gap},
        "limitations": ["Conditional signed atomic law, not a K4-only closure",
                        "Declared quadratic K4-to-K3 feed differs from full true K3 response",
                        "No wide rank, measured FLOP, server residual, label, or score claim"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = run()
    output["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({"status": output["status"], "checks_passed": output["checks_passed"],
                      "checks_total": output["checks_total"], "output": str(args.output)},
                     sort_keys=True))
    return 0 if output["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
