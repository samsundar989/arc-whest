"""Generate the four label-free coherent-K4 mechanism arms.

The represented residual fourth cumulant is the symmetric lift

    H(G)_{abcd} = m/6 * sum_{six pairs (p,q)} G[p,q] delta[other pair]

with ``G = diag(g) + lam * C_off`` and ``m = init_scale``. Consequently

    D4[i]     = m G[i,i]
    D22[i,j]  = m (G[i,i] + G[j,j]) / 6                  (i != j)
    D31[i,j]  = m G[i,j] / 2                             (i != j).

``G`` transports as ``W.T @ G @ W``. Its diagonal is available in O(n^2)
from the covariance backbone, while its off diagonal is closed again as
``lam * C_off``. The post-ReLU diagonal is obtained by the r=1 harmonic
projection already used by the upstream construction. We subtract the
isotropic r=2 scalar already carried by ``estimator_k3_full.py``; this makes
the new object a residual and prevents double counting.

The induced K3 birth is also one object, rather than a change to repeated
slices alone. With ``X = diag(w1) G diag(w1)``, ``y = (m/4) w2``,
``u = y * diag(G)`` and ``v = w1**2``, it is

    B_abc = (X_ab y_c + X_ac y_b + X_bc y_a)/3
          + (delta_ab v_a u_c + delta_ac v_a u_b
             + delta_bc v_b u_a)/3.

The generator carries those two symmetric CP families explicitly. Their
repeated slices are subtracted from the generic residual block at birth, so
turning births on changes the representation of the same current K3 tensor,
then changes later layers only through its transported structure.

Arms are exact / coherent / consumers-only / births-only. The exact arm is
required to be bit-identical and counted-cost-identical to its source.

No coefficient is fitted here. ``lam`` is an explicit structural input used by
all consumers and births; the default one-third is a fixture value, not an
empirical estimate.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
ARMS_DIR = os.path.join(_HERE, "p2_arms")
SOURCE = os.path.join(_ROOT, "estimator_k3_full.py")
DEFAULT_LAM = 1.0 / 3.0


_SUPPORT = r'''

class _K4BirthTower:
    """Explicit symmetric CP births, concatenated along their hub columns."""
    __slots__ = ("X", "P", "Y")

    def __init__(self, X, P, Y):
        self.X, self.P, self.Y = X, P, Y


def _k4b_advance(WT, tw):
    return _K4BirthTower(WT @ tw.X, WT @ tw.P, WT @ tw.Y)


def _k4b_wick(col, tw):
    return _K4BirthTower(col * tw.X, col * tw.P, col * tw.Y)


def _k4b_append(tw, X, P, Y):
    if tw is None:
        return _K4BirthTower(X, P, Y)
    return _K4BirthTower(fnp.concatenate([tw.X, X], axis=1),
                         fnp.concatenate([tw.P, P], axis=1),
                         fnp.concatenate([tw.Y, Y], axis=1))


def _k4b_d3(tw):
    return (tw.X * tw.P * tw.Y).sum(axis=1)


def _k4b_d21(tw, off_mask):
    acc = (tw.X * tw.P) @ tw.Y.T
    acc = acc + (tw.X * tw.Y) @ tw.P.T
    acc = acc + (tw.P * tw.Y) @ tw.X.T
    return (acc / 3.0) * off_mask


def _k4_project_residual(K4v, K22, c4_out, n, metric):
    """r=1 harmonic projection minus the scalar r=2 core already carried."""
    h = K4v + K22.sum(axis=1)
    g_total = ((6.0 / (n + 4.0)) * h
               - (3.0 / ((n + 2.0) * (n + 4.0))) * h.sum()) / metric
    return g_total - c4_out * metric


def _k4_transport_diag(W, g, lam, C, next_cov_diag):
    """diag(W [diag(g)+lam*C_off] W.T), using the K2 diagonal."""
    W2 = W * W
    return W2 @ (g - lam * fnp.diagonal(C)) + lam * next_cov_diag


def _k4_birth_factors(dG, C_off, w1, w2, lam, metric, eye_n, ones_n):
    G = fnp.diag(dG) + lam * C_off
    X = (w1[:, None] * G) * w1[None, :]
    y = w2 * (metric / 4.0)
    u = y * dG
    v = w1 * w1
    Y = y[:, None] * ones_n[None, :]
    Y2 = u[:, None] * ones_n[None, :]
    return X, eye_n, Y, fnp.diag(v), eye_n, Y2
'''


def _patches():
    p = []
    p.append(("def _advance(WT, tw):\n", _SUPPORT + "\n\ndef _advance(WT, tw):\n",
              "coherent K4 support", 1))
    p.append(("                 None, 0.0)\n        mus = []",
              "                 None, 0.0, None, None)\n        mus = []",
              "initial residual-core and birth-tower state", 2))
    p.append(("            state = (mu, state[1], state[2], state[3])",
              "            state = (mu, state[1], state[2], state[3], state[4], state[5])",
              "invariant state threading", 2))
    p.append(("    mu, cov, tw, c4 = state",
              "    mu, cov, tw, c4, gcore, k4tw = state",
              "layer state unpack", 1))
    p.append(("    adv = None if tw is None else _advance(WT, tw)\n    tw = None\n\n    mu_pre",
              "    adv = None if tw is None else _advance(WT, tw)\n    tw = None\n"
              "    k4adv = None if k4tw is None else _k4b_advance(WT, k4tw)\n"
              "    k4tw = None\n\n    mu_pre",
              "birth tower transport", 1))
    p.append((
        "    D3 = _dslice3(adv) if adv is not None else fnp.zeros(n, dtype=dtype)\n"
        "    D4 = (c4 * metric * metric) * ones_n",
        "    D3 = _dslice3(adv) if adv is not None else fnp.zeros(n, dtype=dtype)\n"
        "    if k4adv is not None:\n        D3 = D3 + _k4b_d3(k4adv)\n"
        "    D4 = (c4 * metric * metric) * ones_n\n"
        "    dG = None\n"
        "    if _K4_ACTIVE and gcore is not None:\n"
        "        dG = _k4_transport_diag(WT, gcore, _K4_LAM, cov, var)\n"
        "        if _K4_CONSUMERS:\n            D4 = D4 + metric * dG",
        "core transport and D4 consumer", 1))
    p.append((
        "        D21 = (_dslice21(adv, off_mask) if adv is not None\n"
        "               else fnp.zeros((n, n), dtype=dtype))\n"
        "        D22 = (c4 * metric * metric / 3.0) * off_mask\n"
        "        S = {(1,): mu_pre, (2,): var, (1, 1): C11, (2, 1): D21, (1, 2): D21.T,\n"
        "             (3,): D3, (4,): D4, (2, 2): D22}",
        "        D21 = (_dslice21(adv, off_mask) if adv is not None\n"
        "               else fnp.zeros((n, n), dtype=dtype))\n"
        "        if k4adv is not None:\n            D21 = D21 + _k4b_d21(k4adv, off_mask)\n"
        "        D22 = (c4 * metric * metric / 3.0) * off_mask\n"
        "        D31 = 0.0\n"
        "        if _K4_CONSUMERS and dG is not None:\n"
        "            D22 = D22 + (metric / 6.0) * (dG[:, None] + dG[None, :]) * off_mask\n"
        "            D31 = (metric * _K4_LAM / 2.0) * C11\n"
        "        S = {(1,): mu_pre, (2,): var, (1, 1): C11, (2, 1): D21, (1, 2): D21.T,\n"
        "             (3,): D3, (4,): D4, (2, 2): D22, (3, 1): D31}",
        "D22/D31 consumers and carried birth slices", 1))
    p.append((
        "    for ip in list(pK):\n        if len(ip) == 2 and ip[0] == ip[1]:",
        "    # Four AUGMENT terms consuming the (3,1) slice.\n"
        "    if _K4_ACTIVE and _K4_CONSUMERS and not last_layer:\n"
        "        d31 = S[(3, 1)]\n"
        "        pK[(1, 1)] = pK[(1, 1)] + d31 * (w[(1, 1)][:, None] * w[(3, 1)][None, :]) * (1.0 / 3.0)\n"
        "        z21 = d31 * (w[(1, 2)][:, None] * w[(3, 1)][None, :]) * (1.0 / 6.0)\n"
        "        z21 = z21 + d31 * (w[(3, 2)][:, None] * w[(1, 1)][None, :]) * (1.0 / 6.0)\n"
        "        pK[(2, 1)] = pK[(2, 1)] + z21\n"
        "        pK[(2, 2)] = pK[(2, 2)] + d31 * (w[(1, 2)][:, None] * w[(3, 2)][None, :]) * (1.0 / 3.0)\n"
        "    for ip in list(pK):\n        if len(ip) == 2 and ip[0] == ip[1]:",
        "D31 term consumers", 1))
    p.append(("    if last_layer:\n        return (mu_out, None, None, 0.0)",
              "    if last_layer:\n        return (mu_out, None, None, 0.0, None, None)",
              "final schedule", 1))
    p.append((
        "    S3 = K[(3,)] - p111_3\n    S21 = K[(2, 1)] - p111_21",
        "    newborn_k4 = None\n"
        "    if _K4_ACTIVE and _K4_BIRTHS and dG is not None:\n"
        "        fs = _k4_birth_factors(dG, C11, w1, w2, _K4_LAM, metric,\n"
        "                               fnp.eye(n, dtype=dtype), ones_n)\n"
        "        X1, P1, Y1, X2, P2, Y2 = fs\n"
        "        newborn_k4 = _K4BirthTower(fnp.concatenate([X1, X2], axis=1),\n"
        "                                    fnp.concatenate([P1, P2], axis=1),\n"
        "                                    fnp.concatenate([Y1, Y2], axis=1))\n"
        "        birth3 = _k4b_d3(newborn_k4)\n"
        "        birth21 = _k4b_d21(newborn_k4, off_mask)\n"
        "        S3 = K[(3,)] - p111_3 - birth3\n"
        "        S21 = K[(2, 1)] - p111_21 - birth21\n"
        "    else:\n"
        "        S3 = K[(3,)] - p111_3\n"
        "        S21 = K[(2, 1)] - p111_21",
        "all induced K3 birth factors", 1))
    p.append((
        "    if adv is not None:\n        adv = _wick_tower(w[(1, 1)][:, None], adv)\n    # The block born here:",
        "    if adv is not None:\n        adv = _wick_tower(w[(1, 1)][:, None], adv)\n"
        "    if k4adv is not None:\n        k4adv = _k4b_wick(w[(1, 1)][:, None], k4adv)\n"
        "    if newborn_k4 is not None:\n"
        "        k4adv = _k4b_append(k4adv, newborn_k4.X, newborn_k4.P, newborn_k4.Y)\n"
        "    # The block born here:",
        "birth tower wick and append", 1))
    p.append((
        "    return (mu_out, cov_out, out_tw, c4_out)",
        "    gcore_out = (_k4_project_residual(K[(4,)], K[(2, 2)], c4_out,\n"
        "                                      n, metric) if _K4_ACTIVE else None)\n"
        "    return (mu_out, cov_out, out_tw, c4_out, gcore_out, k4adv)",
        "harmonic core projection", 1))
    return p


_HEADER = '''"""Generated Stage-1 coherent-K4 arm: {name}.

{blurb}

Source estimator_k3_full.py sha256 {src_sha}. Generated by
local_sim/p2_build_k4_arms.py; edit the generator, not this file.
No research labels or fitted coefficients were used.
"""

'''


ARMS = (
    ("exact", False, False, "Exact source control; both K4 switches are off."),
    ("coherent", True, True, "Coherent core: consumers and induced K3 births enabled."),
    ("consumers_only", True, False, "Mechanism arm: consumers only."),
    ("births_only", False, True, "Mechanism arm: induced K3 births only."),
)


def _source():
    raw = open(SOURCE, "rb").read()
    return raw.decode("utf-8"), hashlib.sha256(raw).hexdigest()


def build(name, consumers, births, blurb, src, src_sha, lam):
    body = src
    for old, new, why, count in _patches():
        found = body.count(old)
        if found != count:
            raise RuntimeError("%s: anchor %s expected %d, found %d" %
                               (name, why, count, found))
        body = body.replace(old, new, count)
    i = body.find('"""')
    j = body.find('"""', i + 3)
    body = body[j + 3:].lstrip("\n")
    consts = ("_K4_CONSUMERS = %r\n_K4_BIRTHS = %r\n_K4_ACTIVE = %r\n"
              "_K4_LAM = %.17g\n\n" %
              (consumers, births, consumers or births, lam))
    lines = body.splitlines(keepends=True)
    last_import = max(i for i, line in enumerate(lines)
                      if line.startswith(("import ", "from ")))
    lines.insert(last_import + 1, "\n" + consts)
    out = _HEADER.format(name=name, blurb=blurb, src_sha=src_sha) + "".join(lines)
    ast.parse(out, filename="arm_%s.py" % name)
    return out


def emit(lam=DEFAULT_LAM):
    src, sha = _source()
    os.makedirs(ARMS_DIR, exist_ok=True)
    for name, consumers, births, blurb in ARMS:
        out = build(name, consumers, births, blurb, src, sha, lam)
        path = os.path.join(ARMS_DIR, "arm_%s.py" % name)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(out)
        print("wrote %-24s consumers=%-5s births=%-5s bytes=%d" %
              (os.path.basename(path), consumers, births, len(out)))
    return 0


def parity(width=12, depth=3, out=""):
    """Fatal exact-arm output and counted-cost parity, plus arm parse/import."""
    import importlib.util
    sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
    for path in (_HERE, _ROOT):
        if path not in sys.path:
            sys.path.append(path)
    import numpy as np
    import flopscope as flops
    import harness
    from mint_provenance import canonical_mint

    def load(path):
        spec = importlib.util.spec_from_file_location(
            "_k4_" + os.path.basename(path).replace(".py", ""), path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    modules = {name: load(os.path.join(ARMS_DIR, "arm_%s.py" % name))
               for name, *_ in ARMS}
    source = load(SOURCE)
    w, est_seed, _ = canonical_mint(width, depth, 9_000_000_301)
    mlp = harness.MLP(weights=[np.asarray(x) for x in w], width=width,
                      depth=depth, seed=est_seed)
    budget = 2 ** 41

    class Ctx:
        seed = 0
        flop_budget = budget

    def run(mod):
        est = mod.Estimator()
        est.setup(Ctx())
        with flops.BudgetContext(flop_budget=budget, quiet=True) as ctx:
            pred = np.asarray(est.predict(mlp, budget), dtype=np.float64)
        return pred, int(ctx.flops_used)

    ref, ref_cost = run(source)
    exact, exact_cost = run(modules["exact"])
    if not np.array_equal(ref, exact):
        raise AssertionError("exact arm output parity failed: max abs %.3e" %
                             np.max(np.abs(ref - exact)))
    if ref_cost != exact_cost:
        raise AssertionError("exact arm counted-cost parity failed: %d != %d" %
                             (ref_cost, exact_cost))
    rows = {}
    for name, *_ in ARMS:
        pred, cost = run(modules[name])
        rows[name] = {"flops": cost,
                      "relative_shift": float(np.linalg.norm(pred - ref) /
                                              max(np.linalg.norm(ref), 1e-300))}
    if rows["coherent"]["relative_shift"] == 0.0:
        raise AssertionError("coherent arm is a no-op")
    original_active = modules["coherent"]._K4_ACTIVE
    modules["coherent"]._K4_ACTIVE = False
    zero_pred, zero_cost = run(modules["coherent"])
    modules["coherent"]._K4_ACTIVE = original_active
    zero_output_same = np.array_equal(ref, zero_pred)
    zero_cost_same = ref_cost == zero_cost
    if not zero_output_same or not zero_cost_same:
        raise AssertionError("master-off nesting failed: output=%s cost=%d/%d" %
                             (zero_output_same, ref_cost, zero_cost))
    print("exact output and counted-cost parity: PASS (%d FLOPs)" % ref_cost)
    for name, row in rows.items():
        print("  %-20s %14d  shift %.3e" %
              (name, row["flops"], row["relative_shift"]))
    print("master-off coherent nesting: PASS (output and %d FLOPs)" % zero_cost)
    if out:
        payload = {"kind": "label-free generated-arm parity",
                   "width": width, "depth": depth,
                   "source_sha256": _source()[1],
                   "exact_output_bit_identical": True,
                   "exact_counted_cost_identical": True,
                   "source_flops": ref_cost, "arms": rows}
        payload["master_off_nesting"] = {
            "output_bit_identical": zero_output_same,
            "counted_cost_identical": zero_cost_same, "flops": zero_cost}
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--lam", type=float, default=DEFAULT_LAM)
    ap.add_argument("--width", type=int, default=12)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    if a.emit:
        emit(a.lam)
    if a.parity:
        return parity(a.width, a.depth, a.json)
    if not a.emit:
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
