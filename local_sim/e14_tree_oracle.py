"""Independent tiny oracle for the E14 connected-tree scalar-metric closure.

The reference enumerates labelled trees and dense fourth tensors.  The fast
side uses the four-product formula from the frozen prospective specification.
There are no weights, seeds, truth arrays, or fitted constants in this probe.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/e14_tree_oracle/tree_oracle.json"


def trees():
    """All 4**2 Prufer codes, each interpreted as a labelled tree."""
    for code in itertools.product(range(4), repeat=2):
        degree = [1] * 4
        for x in code:
            degree[x] += 1
        work = degree.copy()
        edges = []
        for x in code:
            leaf = next(i for i in range(4) if work[i] == 1)
            edges.append((leaf, x))
            work[leaf] -= 1
            work[x] -= 1
        u, v = [i for i in range(4) if work[i] == 1]
        edges.append((u, v))
        yield degree, edges


def dense_tree(c, b, a, d):
    n = len(b)
    coeff = {1: b, 2: a, 3: d}
    out = np.zeros((n,) * 4, dtype=np.float64)
    tree_list = list(trees())
    assert len(tree_list) == 16
    for ix in itertools.product(range(n), repeat=4):
        total = 0.0
        for deg, edges in tree_list:
            term = 1.0
            for v in range(4):
                term *= coeff[deg[v]][ix[v]]
            for u, v in edges:
                term *= c[ix[u], ix[v]]
            total += term
        out[ix] = total
    return out


def fast_h(c, b, a, d):
    v = np.diag(c)
    t = (c * c) @ (b * b)
    zstar = (c * (d * b * v)[None, :]) @ c
    rbb = (c * (b * b)[None, :]) @ c
    raa = (c * (a * a * v)[None, :]) @ c
    e = ((c * c) * (a * b)[None, :]) @ c
    r = c @ (a * b * v)
    star_one = c * np.outer(d * t, b)
    hstar = star_one + star_one.T + 2 * np.outer(b, b) * zstar
    path_one = c * np.outer(a * r, b) + (a[:, None] * e) * b[None, :]
    hpath = (2 * np.outer(a, a) * c * rbb
             + 2 * np.outer(b, b) * raa
             + 2 * (path_one + path_one.T))
    return hstar + hpath


def anchored(dense, k4, k22):
    n = len(k4)
    out = dense.copy()
    for ix in itertools.product(range(n), repeat=4):
        counts = sorted([ix.count(v) for v in set(ix)], reverse=True)
        if counts == [4]:
            out[ix] = k4[ix[0]]
        elif counts == [2, 2]:
            i, j = sorted(set(ix))
            out[ix] = k22[i, j]
    return out


def projected(h, cout):
    n = len(h)
    cs = (cout + cout.T) * 0.5
    o = cs - np.diag(np.diag(cs))
    offh = h - np.diag(np.diag(h))
    denom = np.sum(o * o)
    lam = 0.0 if denom == 0.0 else 6.0 / (n + 4) * np.sum(offh * o) / denom
    hh = np.trace(h) / n
    g = 6.0 * (np.diag(h) - hh) / (n + 4)
    gout = np.diag(g) + lam * o
    if not np.isfinite(gout).all() or not np.isfinite(lam):
        raise FloatingPointError("nonfinite projection")
    return gout, g, lam, hh


def scalar_lift(g, metric):
    n = len(g)
    out = np.zeros((n,) * 4)
    for i, j, k, l in itertools.product(range(n), repeat=4):
        out[i, j, k, l] = metric * (
            g[i, j] * (k == l) + g[i, k] * (j == l)
            + g[i, l] * (j == k) + g[j, k] * (i == l)
            + g[j, l] * (i == k) + g[k, l] * (i == j)) / 6.0
    return out


def run_case(n):
    ix = np.arange(n, dtype=np.float64)
    m = np.fromfunction(lambda i, j: (0.11 + 0.07 * i - 0.03 * j)
                        / (1 + abs(i-j)), (n, n), dtype=float)
    c = (m + m.T) * 0.5 + np.diag(0.9 + 0.13 * ix)
    b = 0.33 + 0.08 * ix
    a = -0.17 + 0.09 * ix
    d = 0.06 - 0.03 * ix
    dense = dense_tree(c, b, a, d)
    hslow = np.einsum("abcc->ab", dense)
    hfast = fast_h(c, b, a, d)
    k4 = 0.12 + 0.07 * ix
    k22 = np.fromfunction(lambda i, j: 0.03 + 0.02 * (i+j), (n, n))
    np.fill_diagonal(k22, 0)
    full = anchored(dense, k4, k22)
    hanchor = np.einsum("abcc->ab", full)
    hrepair = hfast.copy()
    np.fill_diagonal(hrepair, k4 + k22.sum(axis=1))
    cout = c + 0.05 * np.outer(a, a)
    gout, g, lam, hh = projected(hanchor, cout)
    w = np.fromfunction(lambda i, j: (0.2 + 0.03 * i - 0.04 * j)
                        / (1 + abs(i-j)), (n, n), dtype=float)
    cpre = w @ cout @ w.T
    dslow = np.diag(w @ gout @ w.T)
    cs = (cout + cout.T) * 0.5
    dfast = (w * w) @ (g - lam * np.diag(cs)) + lam * np.diag(cpre)
    ghat = np.diag(dfast) + lam * (cpre + cpre.T) * 0.5
    np.fill_diagonal(ghat, dfast)
    metric = 2.0
    q = scalar_lift(ghat, metric)
    q4 = metric * np.diag(ghat)
    q22 = metric * (np.diag(ghat)[:, None] + np.diag(ghat)[None, :]) / 6.0
    q31 = metric * ghat / 2.0
    mask = ~np.eye(n, dtype=bool)
    errors = {
        "tree_H": float(np.max(np.abs(hfast-hslow))),
        "anchored_H": float(np.max(np.abs(hrepair-hanchor))),
        "transported_diag": float(np.max(np.abs(dfast-dslow))),
        "q4": float(np.max(np.abs(q4-np.einsum("iiii->i", q)))),
        "q22_off": float(np.max(np.abs(q22[mask]-np.array([q[i,i,j,j] for i,j in zip(*np.where(mask))])))),
        "q31_off": float(np.max(np.abs(q31[mask]-np.array([q[i,i,i,j] for i,j in zip(*np.where(mask))])))),
    }
    assert max(errors.values()) < 2e-12, errors
    wrong_path = hfast - 2 * np.outer(a, a) * c * ((c * (b*b)[None, :]) @ c)
    mutation = float(np.max(np.abs(wrong_path-hslow)))
    assert mutation > 1e-5
    wrong_anchor = float(np.max(np.abs(hfast-hanchor)))
    wrong_diag = float(np.max(np.abs(((w*w) @ g + lam*np.diag(cpre))-dslow)))
    assert wrong_anchor > 1e-5 and wrong_diag > 1e-5
    zero_gout, _, zero_lam, _ = projected(hanchor, np.diag(np.diag(cout)))
    assert zero_lam == 0.0 and np.isfinite(zero_gout).all()
    off = cout - np.diag(np.diag(cout))
    negative_cout = np.diag(np.diag(cout)) - off
    _, _, negative_lam, _ = projected(hanchor, negative_cout)
    assert negative_lam < 0
    asym = cout.copy()
    asym[0, 1] += 0.023
    sym = (asym + asym.T) * 0.5
    asym_projected = projected(hanchor, asym)[0]
    sym_projected = projected(hanchor, sym)[0]
    assert np.array_equal(asym_projected, sym_projected)
    try:
        projected(hanchor, cout * np.nan)
    except FloatingPointError:
        nonfinite_refused = True
    else:
        nonfinite_refused = False
    assert nonfinite_refused
    return {"n": n, "errors": errors, "max_error": max(errors.values()),
            "mutated_path_gap": mutation, "missing_anchor_gap": wrong_anchor,
            "discarded_diag_transport_gap": wrong_diag,
            "zero_denominator_lambda": zero_lam,
            "negative_lambda": float(negative_lam),
            "asymmetry_symmetrized_exactly": True,
            "nonfinite_refused": nonfinite_refused, "lambda": float(lam),
            "trace_g": float(g.sum()), "H": hanchor.tolist(),
            "Gout": gout.tolist(), "Ghatpre": ghat.tolist(),
            "q4": q4.tolist(), "q22": q22.tolist(), "q31": q31.tolist(),
            "dense_tree": dense.tolist(), "anchored_tensor": full.tolist()}


def main():
    result = {"cases": [run_case(3), run_case(4)],
              "scope": "independent dense tree/H/projection/transport/q slices; no estimator or score"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"max_error": max(c["max_error"] for c in result["cases"]),
                      "cases": len(result["cases"])}, indent=2))


if __name__ == "__main__":
    main()
