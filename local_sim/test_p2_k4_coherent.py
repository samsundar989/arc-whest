"""Independent dense-tensor conformance for the Stage-1A K4 arms.

This oracle never calls the generated consumer or birth formulas to construct
its expected tensors. It enumerates tensor indices directly from the defining
symmetric polynomials, then extracts repeated-index slices.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = os.path.join(HERE, "p2_arms")


def load(name):
    path = os.path.join(ARMS, "arm_%s.py" % name)
    spec = importlib.util.spec_from_file_location("oracle_" + name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def lift4_dense(G, metric):
    """Enumerate m/6 Sym(G (x) I) without using slice identities."""
    n = G.shape[0]
    out = np.empty((n, n, n, n), dtype=np.float64)
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    terms = (
                        G[a, b] * (c == d), G[a, c] * (b == d),
                        G[a, d] * (b == c), G[b, c] * (a == d),
                        G[b, d] * (a == c), G[c, d] * (a == b),
                    )
                    out[a, b, c, d] = metric * sum(terms) / 6.0
    return out


def birth3_dense(G, w1, w2, metric):
    """Enumerate the two symmetric cubic polynomials index by index."""
    n = G.shape[0]
    X = w1[:, None] * G * w1[None, :]
    y = (metric / 4.0) * w2
    u = y * np.diag(G)
    v = w1 * w1
    out = np.empty((n, n, n), dtype=np.float64)
    for a in range(n):
        for b in range(n):
            for c in range(n):
                hub = X[a, b] * y[c] + X[a, c] * y[b] + X[b, c] * y[a]
                delta = ((a == b) * v[a] * u[c]
                         + (a == c) * v[a] * u[b]
                         + (b == c) * v[b] * u[a])
                out[a, b, c] = (hub + delta) / 3.0
    return out


def slices4(T):
    n = T.shape[0]
    d4 = np.array([T[i, i, i, i] for i in range(n)])
    d22 = np.array([[T[i, i, j, j] for j in range(n)] for i in range(n)])
    d31 = np.array([[T[i, i, i, j] for j in range(n)] for i in range(n)])
    return d4, d22, d31


def slices3(T):
    n = T.shape[0]
    d3 = np.array([T[i, i, i] for i in range(n)])
    d21 = np.array([[T[i, i, j] for j in range(n)] for i in range(n)])
    return d3, d21


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) /
                 max(np.max(np.abs(a)), np.max(np.abs(b)), 1e-300))


def run():
    rng = np.random.default_rng(20260910)
    rows = []
    modules = {name: load(name) for name in
               ("exact", "coherent", "consumers_only", "births_only")}
    coherent = modules["coherent"]

    expected_switches = {
        "exact": (False, False), "coherent": (True, True),
        "consumers_only": (True, False), "births_only": (False, True),
    }
    for name, expected in expected_switches.items():
        got = (modules[name]._K4_CONSUMERS, modules[name]._K4_BIRTHS)
        rows.append({"check": "switches:" + name, "error": 0.0,
                     "pass": got == expected, "got": list(got)})

    fixtures = []
    n = 5
    A = rng.normal(size=(n, n))
    C = A @ A.T
    fixtures.append(("signed-nonconstant", rng.normal(size=n), 0.37, C))
    fixtures.append(("cancellation", np.array([2., -2., 3., -3., 0.5]), -0.4, C))
    fixtures.append(("zero-core", np.zeros(n), 0.0, np.zeros((n, n))))
    Cz = C.copy()
    Cz[2, :] = 0.0
    Cz[:, 2] = 0.0
    fixtures.append(("zero-variance-coordinate", rng.normal(size=n), 0.21, Cz))

    metric = 2.0
    for label, g, lam, cov in fixtures:
        off = cov - np.diag(np.diag(cov))
        G = np.diag(g) + lam * off
        T4 = lift4_dense(G, metric)
        d4, d22, d31 = slices4(T4)
        gd4 = metric * g
        gd22 = (metric / 6.0) * (g[:, None] + g[None, :])
        gd31 = (metric / 2.0) * G
        mask = 1.0 - np.eye(n)
        for kind, err in (("D4", rel(d4, gd4)),
                          ("D22", rel(d22 * mask, gd22 * mask)),
                          ("D31", rel(d31 * mask, gd31 * mask))):
            rows.append({"check": "%s:%s" % (label, kind), "error": err,
                         "pass": err < 2e-14})

        # The generated O(n^2) transport is checked against a dense G product.
        Wg = rng.normal(size=(n, n))
        next_diag = np.diag(Wg @ cov @ Wg.T)
        exact_diag = np.diag(Wg @ G @ Wg.T)
        cheap_diag = coherent._k4_transport_diag(Wg, g, lam, cov, next_diag)
        err = rel(exact_diag, cheap_diag)
        rows.append({"check": label + ":core-transport", "error": err,
                     "pass": err < 3e-14})

        # Exercise the generated projection, including removal of the scalar
        # r=2 core already carried by the source estimator.
        c4 = -0.23
        total_d4 = d4 + c4 * metric * metric
        total_d22 = d22 * mask + (c4 * metric * metric / 3.0) * mask
        projected = coherent._k4_project_residual(
            total_d4, total_d22, c4, n, metric)
        err = rel(projected, g)
        rows.append({"check": label + ":harmonic-projection", "error": err,
                     "pass": err < 3e-14})

        w1, w2 = rng.normal(size=n), rng.normal(size=n)
        T3 = birth3_dense(G, w1, w2, metric)
        o3, o21 = slices3(T3)
        fs = coherent._k4_birth_factors(g, off, w1, w2, lam, metric,
                                        np.eye(n), np.ones(n))
        tw = coherent._K4BirthTower(np.concatenate([fs[0], fs[3]], axis=1),
                                    np.concatenate([fs[1], fs[4]], axis=1),
                                    np.concatenate([fs[2], fs[5]], axis=1))
        f3 = coherent._k4b_d3(tw)
        f21 = coherent._k4b_d21(tw, mask)
        for kind, err in (("birth-D3", rel(o3, f3)),
                          ("birth-D21", rel(o21 * mask, f21))):
            rows.append({"check": "%s:%s" % (label, kind), "error": err,
                         "pass": err < 3e-14})

        # Transport the full dense birth and the factors independently.
        W = rng.normal(size=(n, n))
        dense_t = np.einsum("ai,bj,ck,ijk->abc", W, W, W, T3)
        tw_t = coherent._k4b_advance(W, tw)
        e3, e21 = slices3(dense_t)
        err = max(rel(e3, coherent._k4b_d3(tw_t)),
                  rel(e21 * mask, coherent._k4b_d21(tw_t, mask)))
        rows.append({"check": label + ":birth-transport", "error": err,
                     "pass": err < 8e-14})

    # Permutation equivariance catches a coordinate-position convention.
    p = rng.permutation(n)
    P = np.eye(n)[p]
    g, lam, cov = fixtures[0][1:]
    G = np.diag(g) + lam * (cov - np.diag(np.diag(cov)))
    base = lift4_dense(G, metric)
    perm = lift4_dense(P @ G @ P.T, metric)
    expected = np.einsum("ai,bj,ck,dl,ijkl->abcd", P, P, P, P, base)
    e = rel(perm, expected)
    rows.append({"check": "permutation-equivariance", "error": e,
                 "pass": e < 2e-14})

    failed = [r for r in rows if not r["pass"]]
    return {"kind": "label-free Stage-1A dense conformance",
            "source_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
            "oracle": "explicit index enumeration of Sym(G tensor I) and cubic births",
            "checks": rows, "n_checks": len(rows), "n_failed": len(failed),
            "pass": not failed}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    result = run()
    for row in result["checks"]:
        print("%-44s %s  error %.3e" %
              (row["check"], "PASS" if row["pass"] else "FAIL", row["error"]))
    print("%d/%d passed" %
          (result["n_checks"] - result["n_failed"], result["n_checks"]))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
            fh.write("\n")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
