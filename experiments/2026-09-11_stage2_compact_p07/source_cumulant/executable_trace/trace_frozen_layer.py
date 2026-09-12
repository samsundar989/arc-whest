"""Truth-free executable trace of the byte-pinned frozen coherent K4 arm.

Run: uv run python experiments/2026-09-11_stage2_compact_p07/source_cumulant/executable_trace/trace_frozen_layer.py
Only deterministic width-8 and width-2 synthetic inputs are used. The callback
reads frame locals; it never assigns into the frame, source, or flopscope arrays.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import flopscope.numpy as fnp


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent.parent / "frozen_arms_metadata_protocol" / "arm_coherent.py"
EXACT_SOURCE = HERE.parent.parent / "frozen_arms_metadata_protocol" / "arm_exact.py"
OUT = HERE / "trace_result.json"
SOURCE_SHA256 = "d3ec837b72f5dacb4f8f622394d3d635268649c741289fb9754197e993343665"
EXACT_SOURCE_SHA256 = "f704f83b2063894c8dcd13120a8baada762229563f6d08bccee39d3aad2e6efc"
CONTRACT_COMMIT = "2164401"
TOL = 1e-4


def plain(value):
    """Make a read-only copy of a traced array for the external oracle."""
    return np.asarray(value).copy()


def rel_error(actual, expected):
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected)))
                 / max(1e-30, float(np.max(np.abs(expected)))))


def assert_same(a, b):
    """Recursive exact equality of returned state, including birth factors."""
    if isinstance(a, (tuple, list)):
        assert isinstance(b, type(a)) and len(a) == len(b)
        for x, y in zip(a, b):
            assert_same(x, y)
    elif a is None:
        assert b is None
    else:
        assert np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)


def hadamard(n):
    h = np.array([[1.0]], dtype=np.float32)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    assert h.shape == (n, n) and np.array_equal(h @ h.T, n * np.eye(n))
    return h


def capture_layer(module, W, state, n, last, signed_lines=None):
    """Return the ordinary `_layer` result and copied locals from its own frame."""
    observed = {}
    code = module._layer.__code__

    def tracer(frame, event, arg):
        if frame.f_code is not code:
            return None
        if event == "return":
            names = ("C", "next_gcore", "k4v", "k22", "h4", "dG", "D4", "mu")
            observed["return"] = {k: plain(frame.f_locals[k]) for k in names
                                  if k in frame.f_locals and frame.f_locals[k] is not None}
        elif event == "line" and signed_lines is not None:
            if frame.f_lineno == signed_lines[0]:
                observed["k11_before"] = plain(frame.f_locals["K11"])
            elif frame.f_lineno == signed_lines[1]:
                observed["k11_after"] = plain(frame.f_locals["K11"])
                for name in ("D22", "D31", "w1", "w2", "w3"):
                    observed[name] = plain(frame.f_locals[name])
        return tracer

    old_trace = sys.gettrace()
    try:
        sys.settrace(tracer)
        returned = module._layer(W, state, n, n, n, fnp.float32, last)
    finally:
        sys.settrace(old_trace)
    return returned, observed


def k11_assignment_lines(source_bytes):
    tree = ast.parse(source_bytes, filename=str(SOURCE))
    layer = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "_layer")
    nodes = [x for x in ast.walk(layer) if isinstance(x, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "K11" for t in x.targets)]
    targeted = [x for x in nodes
                if any(isinstance(y, ast.Name) and y.id == "K11" for y in ast.walk(x.value))
                and any(isinstance(y, ast.Name) and y.id == "D31" for y in ast.walk(x.value))]
    assert len(targeted) == 1
    assignment = targeted[0]
    return (assignment.lineno, assignment.end_lineno + 1)


def analytic_moments():
    # Raw ReLU(N(0,2)) moments from half-Gaussian integration.
    sigma = math.sqrt(2.0)
    m1 = sigma / math.sqrt(2.0 * math.pi)
    m2 = sigma**2 / 2.0
    m3 = 2.0 * sigma**3 / math.sqrt(2.0 * math.pi)
    m4 = 3.0 * sigma**4 / 2.0
    v = m2 - m1**2
    central4 = m4 - 4*m1*m3 + 6*m1*m1*m2 - 3*m1**4
    cumulant4 = central4 - 3*v*v
    return {"variance": v, "central4": central4, "cumulant4": cumulant4,
            "independent_d4": 4*cumulant4/8}


def load_pinned(source_path, expected_sha, module_name):
    source_bytes = source_path.read_bytes()
    actual_sha = hashlib.sha256(source_bytes).hexdigest()
    assert actual_sha == expected_sha, f"frozen source changed: {source_path}"
    module = types.ModuleType(module_name)
    module.__file__ = str(source_path)
    exec(compile(source_bytes, str(source_path), "exec"), module.__dict__)
    return module, source_bytes, actual_sha


def main():
    module, source_bytes, actual_sha = load_pinned(SOURCE, SOURCE_SHA256,
                                                    "frozen_coherent_trace")
    exact, _, exact_sha = load_pinned(EXACT_SOURCE, EXACT_SOURCE_SHA256,
                                      "frozen_exact_trace")
    signed_lines = k11_assignment_lines(source_bytes)

    n = 8
    W0 = fnp.array((math.sqrt(2.0) * np.eye(n)).astype(np.float32))
    signs = np.array([1, -1, 1, -1, -1, 1, -1, 1], dtype=np.float32)
    W1 = fnp.array((signs[:, None] * hadamard(n) * math.sqrt(2.0/n)).astype(np.float32))
    init = (fnp.zeros(n, dtype=fnp.float32), fnp.eye(n, dtype=fnp.float32),
            fnp.ones(n, dtype=fnp.float32), [], None, None, None)
    normal1 = module._layer(W0, init, n, n, n, fnp.float32, False)
    normal2 = module._layer(W1, normal1, n, n, n, fnp.float32, True)
    traced1, l1 = capture_layer(module, W0, init, n, False)
    traced2, l2 = capture_layer(module, W1, traced1, n, True)
    assert_same(normal1, traced1)
    assert_same(normal2, traced2)
    assert "C" in l1["return"] and "next_gcore" in l1["return"]
    assert "dG" in l2["return"] and "D4" in l2["return"]
    exact1 = exact._layer(W0, init, n, n, n, fnp.float32, False)
    _, exact_l2 = capture_layer(exact, W1, exact1, n, True)
    assert "dG" not in exact_l2["return"] and "D4" not in exact_l2["return"]

    m = analytic_moments()
    C = l1["return"]["C"]
    gcore = l1["return"]["next_gcore"]
    dG = l2["return"]["dG"]
    D4 = l2["return"]["D4"]

    # Signed positive-definite preactivation covariance, with alpha=(-1,+1).
    n2 = 2
    signed_state = (fnp.array(np.array([-1., 1.], dtype=np.float32)),
                    fnp.array(np.array([[1., .2], [.2, 1.]], dtype=np.float32)),
                    fnp.ones(n2, dtype=fnp.float32), [], None, None, None)
    W_signed = fnp.eye(n2, dtype=fnp.float32)
    signed_normal = module._layer(W_signed, signed_state, n2, n2, n2, fnp.float32, False)
    signed_traced, signed = capture_layer(module, W_signed, signed_state, n2, False,
                                          signed_lines=signed_lines)
    assert_same(signed_normal, signed_traced)
    # Subtract the symmetric D22 increment, isolating what the executed K11
    # assignment added from D31. This checks its actual row/column orientation.
    d22_increment = signed["D22"] * np.outer(signed["w2"], signed["w2"]) / 4.0
    d31_increment = signed["k11_after"] - signed["k11_before"] - d22_increment
    phi = math.exp(-.5) / math.sqrt(2*math.pi)
    cdf_negative = (1 + math.erf(-1/math.sqrt(2))) / 2
    cdf_positive = (1 + math.erf(1/math.sqrt(2))) / 2
    expected_signed_01 = (2*.2/3) * phi * cdf_positive / 12
    expected_signed_10 = (2*.2/3) * -phi * cdf_negative / 12

    checks = {
        "source_sha": actual_sha == SOURCE_SHA256,
        "exact_control_source_sha": exact_sha == EXACT_SOURCE_SHA256,
        "exact_control_has_no_k4_terms": "D4" not in exact_l2["return"],
        "trace_preserves_returned_state": True,
        "l1_covariance": rel_error(C, m["variance"] * np.eye(n)) < TOL,
        "l1_central_moment": rel_error(l1["return"]["k4v"], np.full(n, m["central4"])) < TOL,
        "l1_fabricated_mixed_k22": rel_error(l1["return"]["k22"][0, 1], m["central4"]/3) < TOL,
        "l1_gcore": rel_error(gcore, np.full(n, m["central4"]/2)) < TOL,
        "l2_dg": rel_error(dG, np.full(n, m["central4"])) < TOL,
        "l2_d4": rel_error(D4, np.full(n, 2*m["central4"])) < TOL,
        "d4_exceeds_independent_cumulant": float(np.min(D4)) > 8*m["independent_d4"],
        "signed_source_d31_symmetric": abs(float(signed["D31"][0, 1] - signed["D31"][1, 0])) < TOL,
        "signed_k11_increment_asymmetric": abs(float(d31_increment[0, 1] - d31_increment[1, 0])) > 1e-3,
        "signed_k11_increment_matches_analytic_orientation":
            rel_error(d31_increment[0, 1], expected_signed_01) < TOL
            and rel_error(d31_increment[1, 0], expected_signed_10) < TOL,
    }
    result = {
        "schema": "truth_free_frozen_layer_trace_v1",
        "contract_commit": CONTRACT_COMMIT,
        "source_path": SOURCE.relative_to(HERE.parent.parent).as_posix(),
        "source_sha256": actual_sha,
        "exact_control_source_sha256": exact_sha,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fixture": {"width": 8, "depth": 2, "W0": "sqrt(2)*I", "W1": "row-signed Sylvester H8*sqrt(2/8)",
                    "row_signs": signs.tolist(), "signed_k11_alpha": [-1, 1], "signed_k11_C_off": .2},
        "capture": {"method": "sys.settrace return/line events, read-only copied locals",
                    "l1_local_C": C.tolist(), "l1_local_k4v": l1["return"]["k4v"].tolist(),
                    "l1_local_k22_01": float(l1["return"]["k22"][0, 1]),
                    "l1_local_gcore": gcore.tolist(), "l2_local_dG": dG.tolist(),
                    "l2_local_D4": D4.tolist(), "signed_k11_assignment_lines": signed_lines,
                    "signed_d31_increment_01": float(d31_increment[0, 1]),
                    "signed_d31_increment_10": float(d31_increment[1, 0]),
                    "signed_d31_increment_antisymmetry": float(d31_increment[0, 1]-d31_increment[1, 0]),
                    "exact_l2_has_dG": "dG" in exact_l2["return"],
                    "exact_l2_has_D4": "D4" in exact_l2["return"]},
        "analytic": m,
        "signed_analytic": {"d31_increment_01": expected_signed_01,
                            "d31_increment_10": expected_signed_10},
        "relative_errors": {"C": rel_error(C, m["variance"] * np.eye(n)),
                            "gcore": rel_error(gcore, np.full(n, m["central4"]/2)),
                            "dG": rel_error(dG, np.full(n, m["central4"])),
                            "D4": rel_error(D4, np.full(n, 2*m["central4"]))},
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "Deterministic source execution only; no accuracy, score, cost, or residual result.",
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": checks, "output": str(OUT)}, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
