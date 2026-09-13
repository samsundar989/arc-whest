"""Exact, label-free discriminator for the oriented D21 term in K11.

This deliberately does not import either estimator or flopscope. A finite signed
moment functional has zero moments through degree two and prescribed mixed
third moments E[X_i**2 X_j] = D21[i,j]. Enumerating it against quadratic
polynomial response functions independently determines the covariance term.
The signed weights are an algebraic finite-difference device, not a law.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction as F
from pathlib import Path


N = 3
ZERO = F(0)
HALF = F(1, 2)
QUARTER = F(1, 4)
FIRST = {F(-1): F(-1), F(1): F(1)}
SECOND = {F(-1): F(1), F(0): F(-2), F(1): F(1)}
W1 = (F(2, 3), F(3, 4), F(5, 6))
W2 = (F(7, 5), F(4, 3), F(2, 5))


def moment_functional(d21):
    """Product finite differences: third mixed moment 1, lower moments 0."""
    origin = (ZERO,) * N
    atoms = {origin: F(1)}
    for i in range(N):
        for j in range(N):
            if i == j or d21[i][j] == 0:
                continue
            for x, wx in SECOND.items():
                for y, wy in FIRST.items():
                    point = list(origin)
                    point[i], point[j] = x, y
                    key = tuple(point)
                    atoms[key] = atoms.get(key, ZERO) + d21[i][j] * wx * wy * QUARTER
    return {point: weight for point, weight in atoms.items() if weight}


def expectation(atoms, fn):
    return sum((weight * fn(point) for point, weight in atoms.items()), ZERO)


def response(x, i):
    return W1[i] * x[i] + HALF * W2[i] * x[i] * x[i]


def masked(matrix):
    return [[ZERO if i == j else matrix[i][j] for j in range(N)] for i in range(N)]


def format_matrix(matrix):
    return [[str(value) for value in row] for row in matrix]


def fixture(name, d21, expect_compact_mismatch):
    atoms = moment_functional(d21)
    assert expectation(atoms, lambda x: F(1)) == 1
    assert all(expectation(atoms, lambda x, i=i: x[i]) == 0 for i in range(N))
    assert all(expectation(atoms, lambda x, i=i, j=j: x[i] * x[j]) == 0
               for i in range(N) for j in range(N))
    assert all(expectation(atoms, lambda x, i=i, j=j: x[i] * x[i] * x[j]) == d21[i][j]
               for i in range(N) for j in range(N) if i != j)

    mean = [expectation(atoms, lambda x, i=i: response(x, i)) for i in range(N)]
    assert mean == [ZERO] * N
    oracle = masked([[expectation(atoms, lambda x, i=i, j=j:
                                   response(x, i) * response(x, j)) - mean[i] * mean[j]
                      for j in range(N)] for i in range(N)])

    # Frozen L2 table: S[(1,2)] = D21.T, scalar 1 on the w1/w2 pair;
    # symmetrizing pK[(1,1)] gives 1/2 of each orientation, K11 = pK11.
    l2 = masked([[HALF * (d21[i][j] * W2[i] * W1[j]
                         + d21[j][i] * W1[i] * W2[j])
                  for j in range(N)] for i in range(N)])
    # Current p2_compact_reference.py K11 expression, isolated to D21.
    compact = masked([[QUARTER * (d21[i][j] * W2[i] * W1[j]
                              + d21[j][i] * W1[i] * W2[j])
                       for j in range(N)] for i in range(N)])
    assert oracle == l2, name
    assert (compact != oracle) == expect_compact_mismatch, name
    assert compact == [[value * HALF for value in row] for row in oracle], name
    assert all(oracle[i][i] == l2[i][i] == compact[i][i] == 0 for i in range(N))
    assert all(oracle[i][j] == oracle[j][i] for i in range(N) for j in range(N))
    return {"fixture": name, "signed_atoms": len(atoms),
            "oracle_K11_D21": format_matrix(oracle),
            "L2_table_is_oracle": True,
            "current_compact_is_half_oracle": True,
            "compact_disagrees_where_expected": True,
            "diagonal_mask_zero": True}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    z = ZERO
    fixtures = [
        ("forward_only", ((z, F(3, 5), z), (z, z, z), (z, z, z)), True),
        ("reverse_only", ((z, z, z), (F(-2, 7), z, z), (z, z, z)), True),
        ("signed_both_orientations", ((F(5, 4), F(3, 5), z),
                                      (F(-2, 7), F(-1, 2), F(4, 9)),
                                      (z, F(-5, 11), F(7, 6))), True),
        ("zero_offdiagonal_and_mask", ((F(5, 4), z, z),
                                       (z, F(-1, 2), z),
                                       (z, z, F(7, 6))), False),
    ]
    results = [fixture(*case) for case in fixtures]
    report = {
        "status": "PASS", "scope": "one-step off-diagonal D21 covariance only",
        "arithmetic": "exact fractions; finite signed polynomial moment enumeration",
        "source_comparison": {
            "frozen_L2": "estimator_k3_strassen_L2.py:843-927, table D21 scalar 1 then symmetrization",
            "compact": "local_sim/p2_compact_reference.py:815-817, D21 scalar 1/4",
        },
        "finding": "frozen L2 contributes 1/2 of each orientation; current compact contributes 1/4",
        "fixtures": results,
        "limits": "No full recurrence, source birth, scalar K4, rank compression, score, or deployment claim.",
    }
    payload = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
