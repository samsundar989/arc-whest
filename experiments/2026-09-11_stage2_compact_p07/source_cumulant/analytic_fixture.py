"""Truth-free analytic discriminator for the frozen coherent K4 source.

This is a small, standalone stdlib calculation. It uses independent
X_i = max(0, Z_i), Z_i ~ N(0, 2), and a row with |W_i| = sqrt(2/n).
Signs of W do not change fourth cumulants or the centered fourth moment.
No MLP, fitted parameter, protected label, or project evaluator is loaded.

The exact row cumulant is computed twice: by additivity of independent
cumulants and by expanding the row's centered fourth moment. The frozen
source prediction is an algebraic reduction of its L1 homogeneous source
and L2 compact map, not an invocation or copy of the implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "frozen_arms" / "arm_coherent.py"
OUT = HERE / "analytic_fixture_result.json"
WIDTHS = (8, 1024)
LAMBDA = 1.0 / 3.0
EXPECTED_SOURCE_SHA256 = "d3ec837b72f5dacb4f8f622394d3d635268649c741289fb9754197e993343665"


def rel_error(observed: float, expected: float) -> float:
    return abs(observed - expected) / max(abs(expected), 1.0e-300)


def relu_gaussian_moments(sigma: float) -> dict[str, float]:
    # Integrating z^k exp(-z^2/(2 sigma^2)) over z >= 0 gives these
    # four raw moments. The atom at zero contributes nothing for k > 0.
    inverse_root_two_pi = 1.0 / math.sqrt(2.0 * math.pi)
    mean = sigma * inverse_root_two_pi
    raw2 = sigma**2 / 2.0
    raw3 = 2.0 * sigma**3 * inverse_root_two_pi
    raw4 = 3.0 * sigma**4 / 2.0
    variance = raw2 - mean**2
    central4 = raw4 - 4.0 * mean * raw3 + 6.0 * mean**2 * raw2 - 3.0 * mean**4
    cumulant4 = central4 - 3.0 * variance**2
    # Independently expand the fourth derivative of log M(t) at t=0.
    cumulant4_raw_polynomial = (raw4 - 4.0 * mean * raw3 - 3.0 * raw2**2
                                + 12.0 * mean**2 * raw2 - 6.0 * mean**4)
    return {
        "mean": mean,
        "raw2": raw2,
        "raw3": raw3,
        "raw4": raw4,
        "variance": variance,
        "central4": central4,
        "cumulant4": cumulant4,
        "cumulant4_raw_polynomial": cumulant4_raw_polynomial,
    }


def row_calculation(width: int, moments: dict[str, float]) -> dict[str, float | int]:
    variance = moments["variance"]
    central4 = moments["central4"]
    cumulant4 = moments["cumulant4"]
    squared_weight = 2.0 / width
    fourth_weight = squared_weight**2
    pairs = width * (width - 1) // 2

    # Independence makes the cumulant of a sum the sum of cumulants.
    exact_additive = width * fourth_weight * cumulant4

    # Independent centered coordinates have E[Y_i^2 Y_j^2] = v^2.
    # Expand E[(sum W_i Y_i)^4] by multiplicity and subtract 3 Var^2.
    row_central4 = (width * fourth_weight * central4
                    + 6.0 * pairs * fourth_weight * variance**2)
    row_variance = width * squared_weight * variance
    exact_by_moments = row_central4 - 3.0 * row_variance**2

    # Frozen L1 uses the central fourth moment as its diagonal "k4v".
    # Its off-diagonal prescription for equal marginals is k4v/3.
    # Thus each h4 is c4 + (n-1)c4/3 = (n+2)c4/3.
    # Reducing the frozen scalar compact transform gives gcore = c4/2.
    # At L2, with independent equal-variance inputs, var_pre = 2v;
    # the lambda covariance subtraction and addition cancel exactly:
    # dG = 2(gcore-lambda*v) + lambda*2v = 2gcore.
    # The frozen consumer sets D4 = 2dG = 2c4, independent of n.
    fabricated_k22 = central4 / 3.0
    source_h4 = (width + 2.0) * central4 / 3.0
    source_gcore = central4 / 2.0
    source_dg = 2.0 * (source_gcore - LAMBDA * variance) + LAMBDA * row_variance
    source_d4 = 2.0 * source_dg

    # A literal dense fourth-tensor contraction exposes separately the
    # wrong diagonal moment and the invented mixed (i,i,j,j) contribution.
    wrong_diagonal_contraction = width * fourth_weight * central4
    invented_mixed_contraction = 6.0 * pairs * fourth_weight * fabricated_k22
    return {
        "width": width,
        "he_weight_squared": squared_weight,
        "row_variance": row_variance,
        "row_central4": row_central4,
        "exact_d4_additivity": exact_additive,
        "exact_d4_moment_expansion": exact_by_moments,
        "frozen_diagonal_k4v": central4,
        "frozen_offdiagonal_k22": fabricated_k22,
        "correct_independent_k22": 0.0,
        "wrong_diagonal_dense_contraction": wrong_diagonal_contraction,
        "invented_mixed_dense_contraction": invented_mixed_contraction,
        "frozen_source_h4": source_h4,
        "frozen_source_gcore": source_gcore,
        "frozen_compact_d4": source_d4,
        "frozen_to_exact_d4_ratio": source_d4 / exact_additive,
        "exact_derivation_relative_error": rel_error(exact_by_moments, exact_additive),
    }


def signed_covariance_example() -> dict[str, object]:
    # Two valid Gaussian preactivation marginals, unit variance and
    # standardized means -1,+1. ReLU derivatives are w1=Phi(alpha),
    # w3=-alpha*phi(alpha); their signs differ. Covariance 0.2 is PSD.
    alpha = (-1.0, 1.0)
    phi = tuple(math.exp(-a * a / 2.0) / math.sqrt(2.0 * math.pi) for a in alpha)
    w1 = tuple((1.0 + math.erf(a / math.sqrt(2.0))) / 2.0 for a in alpha)
    w3 = tuple(-a * p for a, p in zip(alpha, phi))
    offdiagonal_covariance = 0.2

    # Frozen D31=lambda*C_off is symmetric. Multiplying that symmetric
    # factor by w3_i*w1_j only, rather than a symmetric derivative pair,
    # makes its two covariance entries unequal.
    symmetric_d31_sum = 2.0 * LAMBDA * offdiagonal_covariance
    upper = symmetric_d31_sum * w3[0] * w1[1] / 12.0
    lower = symmetric_d31_sum * w3[1] * w1[0] / 12.0
    return {
        "alpha": alpha,
        "w1": w1,
        "w3": w3,
        "input_covariance": [[1.0, offdiagonal_covariance],
                             [offdiagonal_covariance, 1.0]],
        "frozen_d31_sum_offdiagonal": symmetric_d31_sum,
        "frozen_k11_d31_correction_01": upper,
        "frozen_k11_d31_correction_10": lower,
        "antisymmetric_difference_01_minus_10": upper - lower,
    }


def main() -> None:
    source_sha256 = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    script_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    moments = relu_gaussian_moments(math.sqrt(2.0))
    rows = [row_calculation(width, moments) for width in WIDTHS]
    signed = signed_covariance_example()
    variance = moments["variance"]
    central4 = moments["central4"]
    cumulant4 = moments["cumulant4"]
    independent_mixed_centered_22 = (moments["raw2"] - moments["mean"]**2)**2
    independent_mixed_cumulant_22 = independent_mixed_centered_22 - variance**2

    checks = {
        "frozen_source_sha256_matches": source_sha256 == EXPECTED_SOURCE_SHA256,
        "raw_moment_cumulant_derivation_agrees":
            rel_error(moments["cumulant4_raw_polynomial"], cumulant4) < 1.0e-12,
        "central_minus_cumulant_equals_three_variance_squared":
            rel_error(central4 - cumulant4, 3.0 * variance**2) < 1.0e-12,
        "independent_mixed_k22_zero":
            abs(independent_mixed_cumulant_22) < 1.0e-12,
        "frozen_mixed_k22_nonzero":
            central4 / 3.0 > 0.0,
        "width_8_independent_derivations_agree":
            rows[0]["exact_derivation_relative_error"] < 1.0e-10,
        "width_1024_independent_derivations_agree":
            rows[1]["exact_derivation_relative_error"] < 1.0e-8,
        "exact_d4_inverse_width":
            rel_error(rows[1]["exact_d4_additivity"] / rows[0]["exact_d4_additivity"],
                      WIDTHS[0] / WIDTHS[1]) < 1.0e-12,
        "frozen_compact_d4_width_independent":
            rel_error(rows[1]["frozen_compact_d4"], rows[0]["frozen_compact_d4"])
            < 1.0e-12,
        "signed_d31_correction_asymmetric":
            abs(signed["antisymmetric_difference_01_minus_10"]) > 1.0e-6,
    }
    result = {
        "schema": "truth_free_source_cumulant_fixture_v1",
        "contract_commit": "c54eb87",
        "source_path": SOURCE.relative_to(HERE.parent).as_posix(),
        "source_sha256": source_sha256,
        "script_sha256": script_sha256,
        "distribution": "independent max(0,Z_i), Z_i ~ N(0,2)",
        "row": "equal-magnitude He weights, |W_i|=sqrt(2/n)",
        "moments": moments,
        "independent_mixed_centered_22": independent_mixed_centered_22,
        "independent_mixed_cumulant_22": independent_mixed_cumulant_22,
        "true_independent_mixed_k22_derivation":
            "E[(Xi-mu)^2(Xj-mu)^2]-Var(Xi)Var(Xj)-2Cov(Xi,Xj)^2 = v^2-v^2-0",
        "width_rows": rows,
        "signed_covariance_example": signed,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "Algebraic source identity only; no accuracy, score, residual, or deployment result.",
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": checks, "output": str(OUT)},
                     sort_keys=True))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
