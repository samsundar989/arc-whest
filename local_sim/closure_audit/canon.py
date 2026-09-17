"""Canonical form of the nonlinear moment program a file actually EXECUTES.

Handoff 2026-09-17 step 1. Label-free, no truth, no network: reads source tables.

Canonical term (the same schema `local_sim/closure/generate.py` emits, written
independently):

    {"output": [a] | [a, b], "factors": sorted block names, "derivs": [k_i] | [k_i, k_j],
     "coef": "p/q"}

Block names. Univariate outputs: "D3" = kappa_iii, "D4" = kappa_iiii.
Bivariate outputs, index 0 = i (row), index 1 = j (column):
    C = kappa_ij (off-diagonal)      D21 = kappa_iij      D12 = kappa_ijj
    D22 = kappa_iijj                 D3i, D3j, D4i, D4j   (univariate, row / column)
    Q31 = kappa_iiij, Q13 = kappa_ijjj  (not carried by v32; v17's `wk431` supplies one)

Orientation is explicit: a program that symmetrises `(T + T^T)/2` afterwards is
expanded into both orientations at half weight, so two programs compare term by
term with no convention left implicit.

Executed-path notes, verified here and in `test_canon.py`:
  * v32 (`_layer`): univariate rows take the fast path only when they are exactly
    `{w(0,p), D3 w(3,p)/6, D4 w(4,p)/24}` -- so the table rows ARE the executed
    program; bivariate rows run the generic loop; (1,1) and (2,2) are symmetrised;
    the (1,2) output is the transpose of (2,1) (`_get_pk_block`).
  * v17 (`_term_prog(mode=1)` with regeneration on and `NO_WK431` off, the live
    1024x16 branch): `TERM_SPECS` plus the four `wk431` rows; pk11/pk22
    symmetrised, pk21 not; diagonals zeroed (irrelevant to off-diagonal terms).
    `NO_CORR` defaults on, so no fitted mean rider runs.
"""
from __future__ import annotations

import ast
import base64
import hashlib
import json
import os
import sys
import zlib
from fractions import Fraction

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))

SWAP = {"C": "C", "D21": "D12", "D12": "D21", "D22": "D22", "D3i": "D3j", "D3j": "D3i",
        "D4i": "D4j", "D4j": "D4i", "Q31": "Q13", "Q13": "Q31"}


def frac(x, limit=1_000_000):
    return Fraction(x).limit_denominator(limit)


def sha256_file(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def key(t):
    return (tuple(t["output"]), tuple(sorted(t["factors"])), tuple(t["derivs"]))


def add_term(acc, output, factors, derivs, coef):
    k = (tuple(output), tuple(sorted(factors)), tuple(derivs))
    acc[k] = acc.get(k, Fraction(0)) + coef


def transpose(k):
    output, factors, derivs = k
    return (tuple(reversed(output)), tuple(sorted(SWAP[f] for f in factors)),
            tuple(reversed(derivs)))


def finish(acc, symmetric_outputs):
    """Apply `(T + T^T)/2` to symmetric outputs, drop zero coefficients."""
    out = {}
    for k, c in acc.items():
        if k[0] in symmetric_outputs:
            for kk in (k, transpose(k)):
                out[kk] = out.get(kk, Fraction(0)) + c / 2
        else:
            out[k] = out.get(k, Fraction(0)) + c
    return {k: c for k, c in out.items() if c != 0}


def to_records(terms):
    return [{"output": list(k[0]), "factors": list(k[1]), "derivs": list(k[2]),
             "coef": f"{c.numerator}/{c.denominator}"}
            for k, c in sorted(terms.items())]


# --------------------------------------------------------------------------- #
# ours: the zlib/base64 table embedded in a v3x estimator
# --------------------------------------------------------------------------- #

def _embedded_tables(path):
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "_TABLES_B64" for t in node.targets):
            b64 = ast.literal_eval(node.value)
            return json.loads(zlib.decompress(base64.b64decode(b64)))
    raise ValueError(f"no _TABLES_B64 in {path}")


_BIV_BLOCK = {(1, 1): "C", (2, 1): "D21", (1, 2): "D12", (2, 2): "D22"}
_UNI_BLOCK = {(3,): "D3", (4,): "D4"}


def canon_ours(path):
    T = _embedded_tables(path)
    acc = {}
    for row in T["terms"]:
        ip = tuple(row["int_part"])
        coef = frac(float(row["scalar"]))
        wick = [tuple(int(x) for x in kp) for kp in row["wick"]]
        assert tuple(p for _, p in wick) == ip, (row, "wick p must equal the output power")
        derivs = [k for k, _ in wick]
        if len(ip) == 1:
            factors = [_UNI_BLOCK[tuple(b["vec"])] for b in row["blocks"]]
        else:
            factors = [_BIV_BLOCK[tuple(b["vec"])] for b in row["blocks"]]
        # the derivative order on each axis must equal the blocks' index count
        for ax in range(len(ip)):
            n_ax = sum(b["vec"][ax] for b in row["blocks"])
            assert n_ax == derivs[ax], (row, ax)
        add_term(acc, ip, factors, derivs, coef)
    sym = {(1, 1), (2, 2)}
    return finish(acc, sym), T


# --------------------------------------------------------------------------- #
# public v17 / v29: TERM_SPECS + WICK_PAIRS, parsed without executing the file
# --------------------------------------------------------------------------- #

_V17_BLOCK = {  # name -> (canonical factor or None for ones, axis counts (i, j))
    "ones1": None, "ones2": None, "c_off": "C", "d21": "D21", "d21T": "D12",
    "wk4m": "D22", "d3col": "D3i", "d3row": "D3j", "g4col": "D4i", "g4row": "D4j",
    "d3": "D3", "g4": "D4", "wk431": "Q13",
}


def _literal_assign(tree, name):
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name
                                                for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def canon_public(path, with_wk431=True):
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    pairs = _literal_assign(tree, "WICK_PAIRS")
    specs = _literal_assign(tree, "TERM_SPECS")
    idx = pairs.index
    if with_wk431:
        # appended at import in v17 (lines 246-250); confirm the source spells them
        extra = [((1, 1), ("wk431", "ones2", idx((1, 1)), idx((3, 1)), 1.0 / 3.0)),
                 ((2, 1), ("wk431", "ones2", idx((1, 2)), idx((3, 1)), 1.0 / 6.0)),
                 ((2, 1), ("wk431", "ones2", idx((3, 2)), idx((1, 1)), 1.0 / 6.0)),
                 ((2, 2), ("wk431", "ones2", idx((1, 2)), idx((3, 2)), 1.0 / 3.0))]
        for ip, s in extra:
            if s not in specs[ip]:
                specs[ip].append(s)
    acc = {}
    for ip, rows in specs.items():
        for a, b, wl, *rest in rows:
            coef = frac(rest[-1])
            factors = [_V17_BLOCK[x] for x in (a, b) if _V17_BLOCK[x] is not None]
            if len(ip) == 1:
                kl, pl = pairs[wl]
                assert pl == ip[0]
                derivs = [kl]
            else:
                wr = rest[0]
                (kl, pl), (kr, pr) = pairs[wl], pairs[wr]
                assert (pl, pr) == ip, (ip, a, b, pairs[wl], pairs[wr])
                derivs = [kl, kr]
            if "wk431" in (a, b):
                # `wk431 = (m/2) lam C_off` is SYMMETRIC, so v17 feeds the same
                # matrix as kappa_iiij and kappa_ijjj; the Wick orders say which.
                factors = ["Q31" if derivs[0] == 3 else "Q13" if f == "Q13" else f
                           for f in factors]
            add_term(acc, ip, factors, derivs, coef)
    return finish(acc, {(1, 1), (2, 2)}), {"WICK_PAIRS": pairs}


# --------------------------------------------------------------------------- #
# semantic diff
# --------------------------------------------------------------------------- #

_ORDERS = {"C": (1, 1), "D21": (2, 1), "D12": (1, 2), "D22": (2, 2), "D3i": (3, 0),
           "D3j": (0, 3), "D4i": (4, 0), "D4j": (0, 4), "Q31": (3, 1), "Q13": (1, 3),
           "D3": (3,), "D4": (4,)}


def upstream_cost(factors):
    """Pinned ARC power counting (kprop_harmonic.vec_block_cost): sum(v)-1-[all even]."""
    c = 0
    for f in factors:
        v = _ORDERS[f]
        c += sum(v) - 1 - int(all(x % 2 == 0 for x in v))
    return c


def check_derivs(terms):
    """Derivative orders must equal the factors' index counts on each axis."""
    bad = []
    for (out, factors, derivs) in terms:
        tot = [0] * len(out)
        for f in factors:
            for ax, x in enumerate(_ORDERS[f]):
                tot[ax] += x
        if tuple(tot) != tuple(derivs):
            bad.append((out, factors, derivs))
    return bad


def diff(a, b, name_a, name_b):
    ka, kb = set(a), set(b)
    rows = []
    for k in sorted(ka | kb):
        ca, cb = a.get(k), b.get(k)
        status = ("both_equal" if ca == cb else "coef_differs") if (ca is not None and cb is not None) \
            else (f"only_{name_a}" if cb is None else f"only_{name_b}")
        rows.append({"output": list(k[0]), "factors": list(k[1]), "derivs": list(k[2]),
                     name_a: None if ca is None else str(ca),
                     name_b: None if cb is None else str(cb),
                     "status": status, "upstream_cost": upstream_cost(k[1])})
    summary = {}
    for r in rows:
        summary.setdefault(tuple(r["output"]), {}).setdefault(r["status"], 0)
        summary[tuple(r["output"])][r["status"]] += 1
    return rows, {str(k): v for k, v in sorted(summary.items())}


def main():
    ours_path = os.path.join(ROOT, "estimator_k3_v32.py")
    v17_path = os.path.join(ROOT, "external", "504aldo", "estimators", "estimator_v17.py")
    v29_path = os.path.join(ROOT, "external", "504aldo", "estimators", "estimator_v29.py")
    ours, _ = canon_ours(ours_path)
    v17, _ = canon_public(v17_path)
    out = {"artifact": "closure_audit.canon", "version": 1,
           "sources": {p: sha256_file(p) for p in (ours_path, v17_path)},
           "orientation_convention": "index 0 = row i, index 1 = column j; symmetrised outputs expanded",
           "v32": to_records(ours), "v17_live": to_records(v17)}
    for name, t in (("v32", ours), ("v17_live", v17)):
        bad = check_derivs(t)
        assert not bad, (name, bad[:5])
        out[f"{name}_count_per_output"] = {str(o): sum(1 for k in t if k[0] == o)
                                           for o in sorted({k[0] for k in t})}
        out[f"{name}_max_upstream_cost"] = max(upstream_cost(k[1]) for k in t)
    rows, summary = diff(ours, v17, "v32", "v17_live")
    out["diff_v32_v17"] = rows
    out["diff_summary"] = summary
    if os.path.exists(v29_path):
        try:
            v29, _ = canon_public(v29_path)
            out["sources"][v29_path] = sha256_file(v29_path)
            r29, s29 = diff(v17, v29, "v17_live", "v29")
            out["diff_v17_v29_summary"] = s29
        except Exception as exc:  # v29 spells its tables differently; record, don't guess
            out["diff_v17_v29_summary"] = f"not parsed: {type(exc).__name__}: {exc}"
    dst = os.path.join(ROOT, "experiments", "2026-09-17_closure", "canon_programs.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", dst)
    for name in ("v32", "v17_live"):
        print(name, out[f"{name}_count_per_output"], "max cost", out[f"{name}_max_upstream_cost"])
    print("diff summary", json.dumps(summary, indent=1))
    print("v29 vs v17", out.get("diff_v17_v29_summary"))
    print("\nterms in v17_live absent from v32 (oriented):")
    for r in rows:
        if r["status"] != "both_equal":
            print(" ", r["status"], r["output"], r["factors"], r["derivs"],
                  "v32", r["v32"], "v17", r["v17_live"], "cost", r["upstream_cost"])


if __name__ == "__main__":
    sys.exit(main())
