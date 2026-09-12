"""Independent tiny oracle for the proposed D21 shared-right fusion.

No challenge seeds, truth, candidate source, or backend implementation are read.
The exact reference enumerates the unsymmetrized three-leg tensor, rather than
rearranging either matrix expression. NumPy float32 is a rounding diagnostic;
the estimator's actual _mm backend needs its separate trajectory check.

Run after the registered contract and Samuel's predictions exist:
    uv run python local_sim/test_p2_d21_fusion_oracle.py --output <raw-json-path>
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path

import numpy as np


def _fraction_matrix(rows):
    return [[Fraction(x) for x in row] for row in rows]


def _inputs(case):
    d = _fraction_matrix(case["A_D"])
    a = _fraction_matrix(case["A_L"])
    m = _fraction_matrix(case["M"])
    w = [Fraction(x) for x in case["w2b"]]
    n, rank = len(d), len(w)
    assert n > 0 and rank > 0
    assert all(len(x) == n for x in (d, a, m))
    assert all(len(row) == rank for matrix in (d, a, m) for row in matrix)
    c = [[a[i][r] * w[r] for r in range(rank)] for i in range(n)]
    return d, a, m, c


def _tensor_reference(d, a, m, c, *, mask=True):
    """T[p,q,s] = sum_r D[p,r] M[q,r] M[s,r] + 3 L[p,r] M[q,r] C[s,r]."""
    n, rank = len(d), len(d[0])
    t = [[[Fraction(0) for _ in range(n)] for _ in range(n)] for _ in range(n)]
    for p in range(n):
        for q in range(n):
            for s in range(n):
                t[p][q][s] = sum(
                    d[p][r] * m[q][r] * m[s][r]
                    + 3 * a[p][r] * m[q][r] * c[s][r]
                    for r in range(rank)
                )
    # Three distinct placements of the singleton axis. Do not infer symmetry.
    return [[Fraction(0) if mask and i == j else
             (t[i][i][j] + t[i][j][i] + t[j][i][i]) / 3
             for j in range(n)] for i in range(n)]


def _mm(x, y):
    return [[sum(x[i][r] * y[r][j] for r in range(len(y)))
             for j in range(len(y[0]))] for i in range(len(x))]


def _transpose(x):
    return [list(row) for row in zip(*x)]


def _formula(d, a, m, c, fused, *, mask=True):
    """Literal old/fused numerator with the common division and mask at the end."""
    n, rank = len(d), len(d[0])
    dm = [[d[i][r] * m[i][r] for r in range(rank)] for i in range(n)]
    ac = [[a[i][r] * c[i][r] for r in range(rank)] for i in range(n)]
    am = [[a[i][r] * m[i][r] for r in range(rank)] for i in range(n)]
    mm = [[m[i][r] * m[i][r] for r in range(rank)] for i in range(n)]
    if fused:
        left = [[2 * dm[i][r] + 3 * ac[i][r] for r in range(rank)]
                for i in range(n)]
        terms = [_mm(left, _transpose(m)), _mm(mm, _transpose(d)),
                 [[6 * x for x in row] for row in _mm(am, _transpose(c))]]
    else:
        terms = [[[2 * x for x in row] for row in _mm(dm, _transpose(m))],
                 _mm(mm, _transpose(d)),
                 [[6 * x for x in row] for row in _mm(am, _transpose(c))],
                 [[3 * x for x in row] for row in _mm(ac, _transpose(m))]]
    return [[Fraction(0) if mask and i == j else
             sum(term[i][j] for term in terms) / 3
             for j in range(n)] for i in range(n)]


def _float32_diagnostic(case):
    d = np.asarray(case["A_D"], dtype=np.float32)
    a = np.asarray(case["A_L"], dtype=np.float32)
    m = np.asarray(case["M"], dtype=np.float32)
    w = np.asarray(case["w2b"], dtype=np.float32)
    c = a * w
    mask = np.ones((len(d), len(d)), dtype=np.float32) - np.eye(len(d), dtype=np.float32)
    old = 2.0 * ((d * m) @ m.T)
    old = old + ((m * m) @ d.T)
    old = old + 6.0 * ((a * m) @ c.T)
    old = old + 3.0 * ((a * c) @ m.T)
    old = (old / 3.0) * mask
    fused = ((2.0 * (d * m) + 3.0 * (a * c)) @ m.T)
    fused = fused + ((m * m) @ d.T)
    fused = fused + 6.0 * ((a * m) @ c.T)
    fused = (fused / 3.0) * mask
    assert old.dtype == np.float32 and fused.dtype == np.float32
    assert np.isfinite(old).all() and np.isfinite(fused).all()
    assert np.count_nonzero(np.diag(old)) == 0
    assert np.count_nonzero(np.diag(fused)) == 0
    delta = np.abs(old.astype(np.float64) - fused.astype(np.float64))
    return {"max_abs_old_fused": float(delta.max()),
            "n_different_bits": int(np.count_nonzero(old.view(np.uint32) != fused.view(np.uint32))),
            "max_abs_old": float(np.abs(old).max()),
            "backend": "NumPy float32 matmul; diagnostic only"}


CASES = [
    {"name": "oriented_single_column", "A_D": [[1], [2]],
     "A_L": [[0], [0]], "M": [[1], [3]], "w2b": [0]},
    {"name": "signed_non_square_lineage", "A_D": [[1, -2, 3, 0, -1],
     [2, 1, -1, 4, 0], [-3, 2, 0, -1, 1]],
     "A_L": [[-1, 2, 1, -2, 3], [3, -1, 2, 0, -2], [1, 0, -3, 2, 1]],
     "M": [[2, 1, -1, 3, 0], [-1, 2, 3, 1, -2], [3, -2, 1, 0, 2]],
     "w2b": [2, -1, 3, 0, -2]},
    {"name": "column_cancellation", "A_D": [[1, -1, 2, -2],
     [-3, 3, 1, -1], [2, -2, -1, 1]],
     "A_L": [[2, -2, 1, -1], [1, -1, -3, 3], [-1, 1, 2, -2]],
     "M": [[1, 1, 2, 2], [2, 2, -1, -1], [-3, -3, 1, 1]],
     "w2b": [2, 2, -1, -1]},
    {"name": "all_zero", "A_D": [[0, 0, 0], [0, 0, 0],
     [0, 0, 0], [0, 0, 0]], "A_L": [[0, 0, 0], [0, 0, 0],
     [0, 0, 0], [0, 0, 0]], "M": [[0, 0, 0], [0, 0, 0],
     [0, 0, 0], [0, 0, 0]], "w2b": [0, 0, 0]},
    {"name": "wide_lineage_thin_rows", "A_D": [[2, -1, 3, 0, -2],
     [1, 4, -2, 3, -1]], "A_L": [[1, -3, 2, 1, 0],
     [-2, 1, 3, -1, 2]], "M": [[1, 2, -1, 3, 1],
     [2, -1, 4, 0, -2]], "w2b": [-1, 2, 1, -2, 3]},
    {"name": "float32_large_cancellation", "A_D": [[100000000, -100000000, 1],
     [2, -3, 1]], "A_L": [[2, -1, 3], [-4, 2, 1]],
     "M": [[1, 1, 3], [2, 2, -1]], "w2b": [3, -2, 1]},
]


def run():
    rows = []
    for case in CASES:
        d, a, m, c = _inputs(case)
        truth = _tensor_reference(d, a, m, c)
        old = _formula(d, a, m, c, fused=False)
        fused = _formula(d, a, m, c, fused=True)
        assert old == truth, (case["name"], "old does not match tensor")
        assert fused == truth, (case["name"], "fused does not match tensor")
        assert all(truth[i][i] == 0 for i in range(len(d)))
        assert _transpose(fused) == _transpose(truth)
        if case["name"] == "oriented_single_column":
            assert truth == [[0, Fraction(8, 3)], [Fraction(7), 0]]
            assert truth != _transpose(truth)
            unmasked = _tensor_reference(d, a, m, c, mask=False)
            assert any(unmasked[i][i] != 0 for i in range(len(d)))
            assert _formula(d, a, m, c, fused=False, mask=False) == unmasked
            assert _formula(d, a, m, c, fused=True, mask=False) == unmasked
        if case["name"] in ("column_cancellation", "all_zero"):
            assert all(x == 0 for row in truth for x in row)
        rows.append({"name": case["name"], "shape": [len(d), len(d[0])],
                     "exact_old_tensor": True, "exact_fused_tensor": True,
                     "off_diagonal_entries": len(d) * (len(d) - 1),
                     "float32": _float32_diagnostic(case)})
    return {"scope": "tiny exact tensor contraction and NumPy float32 diagnostic",
            "seed_block": None, "candidate_imported": False, "cases": rows,
            "passed": len(rows)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run()
    payload = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
