"""Preregistered full-polar positive-bank repair for the three-layer toy law.

This is a new truth-free fixture under LOG contract ad856c6. The failed
nonlinear_old_history fixture remains immutable. Signed-law helpers retain
its fixed contrasts and gates; the positive bank is constructed separately.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

import test_p2_k4_increment_axis_gauss as old_bank
import test_p2_k4_nonlinear_old_history as old


ORDERS = (32, 48, 64)
RHO = old.RHO
SIGMA = math.sqrt(1.0 - RHO * RHO)
B = old.B
D = old.D
NUMERIC_GATE = 1e-8
IDENTITY_GATE = old.IDENTITY_GATE
MOMENT_GATE = old.MOMENT_GATE

# Fixed analytic roots from the registered B/D rows, not measured errors.
BOUNDARIES = np.array([
    0.0,
    math.atan(11.0 / 28.0),
    math.atan(1.125 / 1.735),
    math.atan(0.94 / 0.83),
    math.atan(8.0 / 3.0),
    math.pi / 2.0,
    math.pi,
    3.0 * math.pi / 2.0,
    2.0 * math.pi,
])


def radial_rule() -> tuple[np.ndarray, np.ndarray, dict]:
    """Positive three-node Rayleigh rule from its six exact moments."""
    mu = np.array([2.0 ** (k / 2.0) * math.gamma(1.0 + k / 2.0)
                   for k in range(6)])
    h0 = np.array([[mu[i + j] for j in range(3)] for i in range(3)])
    h1 = np.array([[mu[i + j + 1] for j in range(3)] for i in range(3)])
    lower = np.linalg.cholesky(h0)
    inverse = np.linalg.inv(lower)
    nodes, eigenvectors = np.linalg.eigh(inverse @ h1 @ inverse.T)
    weights = eigenvectors[0] ** 2
    defects = [abs(float(weights @ nodes ** k) - mu[k]) for k in range(6)]
    return nodes, weights, {"nodes": nodes.tolist(), "weights": weights.tolist(),
                            "exact_moments": mu.tolist(), "moment_defects": defects,
                            "max_moment_defect": max(defects)}


def split_bank(order: int, nodes: np.ndarray, radial_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Unrescaled positive original-Gaussian polar quadrature."""
    abscissa, gl_weights = np.polynomial.legendre.leggauss(order)
    points = []
    weights = []
    sector_masses = []
    normalization = 2.0 * math.pi * SIGMA
    for left, right in zip(BOUNDARIES[:-1], BOUNDARIES[1:]):
        theta = left + (abscissa + 1.0) * (right - left) / 2.0
        wtheta = gl_weights * (right - left) / 2.0
        direction = np.stack((np.cos(theta), np.sin(theta)), axis=0)
        q = (1.0 - RHO * np.sin(2.0 * theta)) / (1.0 - RHO * RHO)
        z = direction[:, :, None] * nodes[None, None, :] / np.sqrt(q)[None, :, None]
        p = (wtheta[:, None] * radial_weights[None, :]
             / (normalization * q[:, None]))
        points.append(z.reshape(2, -1))
        weights.append(p.reshape(-1))
        sector_masses.append(float(np.sum(p)))
    z = np.concatenate(points, axis=1)
    p = np.concatenate(weights)
    x = np.maximum(z, 0.0)
    diagnostics = {"order": order, "node_count": len(p),
                   "sector_masses": sector_masses,
                   "mass": float(np.sum(p)), "mass_error": abs(float(np.sum(p)) - 1.0),
                   "minimum_weight": float(np.min(p)),
                   "minimum_radial_node": float(np.min(nodes)),
                   "maximum_single_array_bytes": max(array.nbytes for array in (z, x, p)),
                   "boundaries": BOUNDARIES.tolist()}
    return x, p, diagnostics


def numeric_errors(raw_a: dict, raw_b: dict, reference: dict) -> dict[str, float]:
    return {f"{a}{b}": abs(raw_a[a, b] - raw_b[a, b]) / max(1.0, abs(reference[a, b]))
            for a, b in old.EXPS}


def keyed_raw(raw: dict) -> dict[str, float]:
    return {f"{a}{b}": value for (a, b), value in raw.items()}


def source_replay(x: np.ndarray, p: np.ndarray, y: np.ndarray,
                  t: np.ndarray, gates: old.Gates) -> dict:
    """Reapply the frozen old/new law and same-bank identity gates."""
    r1, projection1 = old.project_source(x, p, 1)
    v = r1.copy()
    old2 = p - r1
    r2, projection2 = old.project_source(y, v, 2)
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
    gates.upper("q01_equals_v", old.maxabs(q["01"] - v), IDENTITY_GATE)
    gates.upper("q11_equals_p2", old.maxabs(q["11"] - p), IDENTITY_GATE)
    for label, weights in q.items():
        gates.lower(f"q{label}_positive", float(np.min(weights)), 0)
        gates.upper(f"q{label}_mass", abs(float(np.sum(weights)) - 1.0), MOMENT_GATE)

    states = {"layer2": {label: old.make_state(y, weights) for label, weights in q.items()},
              "layer3": {label: old.make_state(t, weights) for label, weights in q.items()}}
    max_identity = 0.0
    max_order_identity = 0.0
    max_true_feed_difference = 0.0
    contrast_output = {}
    for layer_name, layer in states.items():
        layer_max = {"raw": 0.0, "k3": 0.0, "k4": 0.0, "consumers": 0.0}
        for state in layer.values():
            layer_max["raw"] = max(layer_max["raw"], max(abs(state["raw"][exp] - state["oracle_raw"][exp]) for exp in old.EXPS))
            layer_max["k3"] = max(layer_max["k3"], old.maxabs(state["k3"] - state["oracle_k3"]))
            layer_max["k4"] = max(layer_max["k4"], old.maxabs(state["k4"] - state["oracle_k4"]))
            for key in state["consumers"]:
                layer_max["consumers"] = max(layer_max["consumers"],
                                             old.maxabs(state["consumers"][key] - state["oracle_consumers"][key]))
        for key, error in layer_max.items():
            gates.upper(f"{layer_name}_{key}_oracle", error, IDENTITY_GATE)
        fast_raw = old.raw_contrasts(layer, "raw")
        oracle_raw = old.raw_contrasts(layer, "oracle_raw")
        ck3 = old.contrasts(layer, "k3")
        ok3 = old.contrasts(layer, "oracle_k3")
        ck4 = old.contrasts(layer, "k4")
        ok4 = old.contrasts(layer, "oracle_k4")
        max_contrast = 0.0
        max_consumers = 0.0
        per_source = {}
        for name in ck4:
            max_contrast = max(max_contrast, old.maxabs(fast_raw[name] - oracle_raw[name]),
                               old.maxabs(ck3[name] - ok3[name]), old.maxabs(ck4[name] - ok4[name]))
            from_tensor = old.consumers_fast(ck4[name])
            direct = old.consumers_oracle(ok4[name])
            for key in from_tensor:
                max_consumers = max(max_consumers, old.maxabs(from_tensor[key] - direct[key]))
            max_true_feed_difference = max(max_true_feed_difference,
                                           old.maxabs(ck3[name] - from_tensor["feed"]))
            per_source[name] = {"k3": ck3[name].tolist(), "k4": ck4[name].tolist(),
                                "consumers": {key: value.tolist() for key, value in from_tensor.items()}}
        gates.upper(f"{layer_name}_contrast_oracle", max_contrast, IDENTITY_GATE)
        gates.upper(f"{layer_name}_consumer_feed_oracle", max_consumers, IDENTITY_GATE)
        ordering = old.maxabs(ck4["old_given_new"] - ck4["old_given_none"] - ck4["interaction"])
        total = old.maxabs(ck4["total"] - ck4["new"] - ck4["old_given_new"])
        max_order_identity = max(max_order_identity, ordering, total)
        gates.upper(f"{layer_name}_source_assignment", max(ordering, total), IDENTITY_GATE)
        max_identity = max(max_identity, *layer_max.values(), max_contrast, max_consumers)
        contrast_output[layer_name] = per_source
    gates.upper("all_identity_max", max(max_identity, max_order_identity), IDENTITY_GATE)

    before = old.contrasts(states["layer2"], "k4")["old_given_new"]
    after = old.contrasts(states["layer3"], "k4")["old_given_new"]
    linear_prediction = old.tensor_transport(before, D)
    g3 = old.maxabs(after - linear_prediction)
    old_scale = old.maxabs(before)
    gates.lower("nonlinear_gap_absolute", g3, 1e-6)
    gates.lower("nonlinear_gap_relative", g3, 1e-3 * old_scale)
    k1p = old.cumulants_fast(x, p)[1]
    k1r = old.cumulants_fast(x, r1)[1]
    k2p = old.cumulants_fast(y, p)[1]
    k2r = old.cumulants_fast(y, r1)[1]
    g2 = old.maxabs((k2p - k2r) - old.tensor_transport(k1p - k1r, B))
    low_leak = {}
    for name, points, wa, wb in (("layer2", y, p, r1),
                                 ("layer3", t, q["11"], q["01"])):
        ra, rb = old.raw_fast(points, wa), old.raw_fast(points, wb)
        low_leak[name] = {f"{a}{b}": ra[a, b] - rb[a, b] for a, b in old.LOW_EXPS}
    linear_states = {label: old.make_state(D @ y, weights) for label, weights in q.items()}
    linear_old = old.contrasts(linear_states, "k4")["old_given_new"]
    linear_gap = old.maxabs(linear_old - linear_prediction)
    gates.upper("signed_linear_D_control", linear_gap, IDENTITY_GATE)
    zero_old = {"00": r2, "10": r2, "01": r2 + new2, "11": r2 + new2}
    zero_new = {"00": r2, "10": r2 + old2, "01": r2, "11": r2 + old2}
    zero_errors = {}
    for name, weights_by_label, vanished in (("old", zero_old, "old_given_new"),
                                             ("new", zero_new, "new")):
        control = {label: old.make_state(t, weights) for label, weights in weights_by_label.items()}
        error = old.maxabs(old.contrasts(control, "k4")[vanished])
        zero_errors[name] = error
        gates.upper(f"zero_{name}_control", error, IDENTITY_GATE)
    c = old.consumers_fast(after)
    i, j = 0, 1
    swapped = (old.W2[i] * old.W2[j] * c["d22"][i, j] / 4
               + old.W3[i] * old.W1[j] * c["d31"][j, i] / 6
               + old.W1[i] * old.W3[j] * c["d31"][j, i] / 6)
    d31_mutant_gap = abs(swapped - c["covariance"][i, j])
    gates.lower("swapped_D31_mutant_detected", d31_mutant_gap, IDENTITY_GATE)
    gates.lower("linear_carry_mutant_detected", g3, 1e-6)
    interaction = old.contrasts(states["layer3"], "k4")["interaction"]
    interaction_oracle = old.contrasts(states["layer3"], "oracle_k4")["interaction"]
    mixed_gap = old.maxabs(interaction)
    mixed_reference_error = old.maxabs(interaction - interaction_oracle)
    gates.upper("mixed_interaction_oracle", mixed_reference_error, IDENTITY_GATE)
    return {
        "projections": {"old": projection1, "new": projection2},
        "counterfactual_min_weights": {key: float(np.min(value)) for key, value in q.items()},
        "dense": {name: {label: old.serialize_state(state) for label, state in layer.items()}
                  for name, layer in states.items()},
        "contrasts": contrast_output,
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
                      "swapped_D31_mutant_gap": d31_mutant_gap}}


def run() -> dict:
    gates = old.Gates()
    nodes, radial_weights, radial_diag = radial_rule()
    banks = {order: split_bank(order, nodes, radial_weights) for order in ORDERS}
    independent_sectors = old.angular_sectors()
    independent_boundaries = sorted(set([independent_sectors[0][0]]
                                        + [sector[1] for sector in independent_sectors]))
    breakpoint_error = (old.maxabs(np.array(independent_boundaries) - BOUNDARIES)
                        if len(independent_boundaries) == len(BOUNDARIES) else math.inf)
    gates.upper("all_fixed_breakpoints", breakpoint_error, 1e-12)
    gates.upper("boundary_start", abs(float(BOUNDARIES[0])), 1e-12)
    gates.upper("boundary_end", abs(float(BOUNDARIES[-1] - 2.0 * math.pi)), 1e-12)
    gates.lower("minimum_sector_width", float(np.min(np.diff(BOUNDARIES))), 0)
    gates.lower("positive_radial_nodes", float(np.min(nodes)), 0)
    gates.lower("positive_radial_weights", float(np.min(radial_weights)), 0)
    gates.upper("rayleigh_six_moment_defects", radial_diag["max_moment_defect"], 1e-12)
    for order, (x, p, diag) in banks.items():
        gates.upper(f"order{order}_node_count_defect", abs(len(p) - 8 * 3 * order), 0)
        gates.lower(f"order{order}_positive_weight", diag["minimum_weight"], 0)
        gates.upper(f"order{order}_unit_mass", diag["mass_error"], IDENTITY_GATE)
        gates.upper(f"order{order}_single_array_bytes", diag["maximum_single_array_bytes"], 64 * 1024 * 1024)
    direct = {layer: old.continuous_raw(independent_sectors, layer)
              for layer in (2, 3)}
    numeric = {}
    for layer in (2, 3):
        raw = {}
        for order, (x, p, _) in banks.items():
            points = old.push(B, x, relu=True)
            if layer == 3:
                points = old.push(D, points, relu=True)
            raw[order] = old.raw_fast(points, p)
        continuous, contdiag = direct[layer]
        errors = {"32_48": numeric_errors(raw[32], raw[48], continuous),
                  "48_64": numeric_errors(raw[48], raw[64], continuous),
                  "32_64": numeric_errors(raw[32], raw[64], continuous),
                  "48_continuous": numeric_errors(raw[48], continuous, continuous)}
        for pair, values in errors.items():
            gates.upper(f"layer{layer}_{pair}_raw", max(values.values()), NUMERIC_GATE)
        gates.upper(f"continuous_layer{layer}_bound", contdiag["max_reported_or_doubling_bound"], 1e-10,
                    strict=True)
        numeric[f"layer{layer}"] = {
            "raw": {str(order): keyed_raw(value) for order, value in raw.items()},
            "continuous": keyed_raw(continuous), "errors": errors,
            "continuous_error": contdiag}
    # Negative control uses the unchanged, underresolved axis bank at order 48.
    old_radial, old_radial_w, _ = old_bank.radial_rule()
    old_axis, old_axis_w, _ = old_bank.axis_rule()
    unsplit_x, unsplit_p, unsplit_diag = old_bank.bank(
        48, (old_radial, old_radial_w), (old_axis, old_axis_w))
    unsplit_t = old.push(D, old.push(B, unsplit_x, relu=True), relu=True)
    unsplit_raw = old.raw_fast(unsplit_t, unsplit_p)
    unsplit_errors = numeric_errors(unsplit_raw, direct[3][0], direct[3][0])
    unsplit_gap = max(unsplit_errors.values())
    gates.lower("unsplit_negative_control", unsplit_gap, 1e-6)
    positive_stage_pass = all(check["pass"] for check in gates.checks)
    signed = None
    if positive_stage_pass:
        x48, p48, _ = banks[48]
        y48 = old.push(B, x48, relu=True)
        t48 = old.push(D, y48, relu=True)
        signed = source_replay(x48, p48, y48, t48, gates)
    checks = gates.checks
    return {
        "fixture": "nonlinear_split_bank", "contract_commit": "ad856c6",
        "status": "PASS" if all(check["pass"] for check in checks) else "FAIL",
        "checks_passed": sum(check["pass"] for check in checks),
        "checks_total": len(checks), "checks": checks,
        "positive_stage_pass": positive_stage_pass,
        "signed_stage_skipped": not positive_stage_pass,
        "frozen": {"rho": RHO, "B": B.tolist(), "D": D.tolist(),
                   "angular_orders": list(ORDERS), "boundaries": BOUNDARIES.tolist(),
                   "w1": old.W1.tolist(), "w2": old.W2.tolist(),
                   "w3": old.W3.tolist(), "w4": old.W4.tolist(),
                   "relative_perturbation": 0.2},
        "breakpoint_error_vs_independent_root_finder": breakpoint_error,
        "radial_rule": radial_diag,
        "banks": {str(order): diag for order, (_, _, diag) in banks.items()},
        "numeric": numeric,
        "unsplit_negative_control": {"max_material_relative_gap": unsplit_gap,
                                     "errors": unsplit_errors,
                                     "raw": keyed_raw(unsplit_raw),
                                     "bank_diagnostics": unsplit_diag},
        "signed": signed,
        "limitations": ["Future-dependent polar split is a toy numerical diagnostic",
                        "Conditional signed atomic law, not a K4-only nonlinear closure",
                        "No wide-rank, counted-FLOP, server residual, label, or score claim"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    result["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks_passed": result["checks_passed"],
                      "checks_total": result["checks_total"], "output": str(args.output)},
                     sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
