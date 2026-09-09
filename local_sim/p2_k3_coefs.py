"""Freeze the *n*-independent combinatorial tables the fnp K3 port needs.

Companion to `local_sim/p2_k3_terms.py`, and the same discipline: every table
below depends on **nothing but the partition combinatorics** -- not on `n`, not
on the weights, not on the data -- so it can be enumerated once against upstream
rev `93d091a4`, frozen, and re-checked. That converts "re-derive the Moebius
inversion in a restricted dialect" into "implement a fixed table of products".

Three tables:

1. ``pK_to_K`` -- for each diagonal-slice int_part the port carries, the list of
   ``(vector partition, total coefficient)`` that `cumulants.DS_part_sum` sums,
   with the coefficient already folded to
   ``vec_part_coef(vpart, divide_fac=False) * _pK_to_K_coef(vpart)``.
2. ``lap2_d4`` -- the `_lap_m_dslice(m=2, ., part)` coefficients for the two
   `d = 4` slices the SIMPLE tower holds, already multiplied by
   `int_partition_coef(part)`. Both collapse to a full contraction to a SCALAR,
   which is why `K[4]` is an `HTensor` with `r = 2` and an order-0 core.
3. ``int_partition_coef`` for the slices, used by `DSTensor.to_tensor`.

`proj_coef(n, 4, 2)` is NOT frozen: it is a closed form in `n`
(`harmonic.proj_coef`) and the port evaluates it, so a width change cannot
silently invalidate it.

Runs in **`.venv-gpu`** (upstream imports torch). Nothing here is imported by
`estimator.py`.

    .venv-gpu\Scripts\python.exe local_sim/p2_k3_coefs.py
"""
import json
import math
import os
import sys

sys.path.insert(0, r"C:\Users\samsu\Documents\mlp_cumulant_propagation\src")

from mlp_kprop.cumulants import _pK_to_K_coef
from mlp_kprop.kprop_harmonic import factored_keeps_term, get_all_terms_iso
from mlp_kprop.harmonic import _multigraph_coef, proj_coef
from mlp_kprop.wick import _relu_wick_poly
from mlp_kprop.partitions import (
    check_vec_partition, int_partition_coef, vec_part_coef, vector_partitions,
    weak_compositions,
)

#: The int_parts the SIMPLE factored tower actually holds, by degree.
TOWER = {1: [(1,)], 2: [(2,), (1, 1)], 3: [(3,), (2, 1)], 4: [(4,), (2, 2)]}

_ARGS = None   # set in __main__; see the write guard in main()


def pk_to_k_table() -> dict:
    out = {}
    for d, parts in TOWER.items():
        for int_part in parts:
            rows = []
            for vpart in vector_partitions(int_part):
                if not vpart:
                    continue
                c = vec_part_coef(vpart, divide_fac=False) * _pK_to_K_coef(vpart)
                if c == 0:
                    continue
                rows.append({"blocks": [list(v) for v in vpart], "coef": float(c)})
            out[str(int_part)] = rows
    return out


def term_program(augment: bool = False) -> list:
    """The 25 kept SIMPLE terms as an executable program.

    Each row is everything the fnp side needs to evaluate one term without
    re-deriving anything: which int_part slice it lands in, which cumulant
    diagonal slices multiply together (``blocks``, as the (vec, source degree)
    pairs `eval_part` expands), the scalar it carries
    (``count * vec_part_coef(vpart, divide_fac=True)``) and the per-axis Wick
    indices ``(k_i, p_i)``.
    """
    live = get_all_terms_iso(k_max=3, d_max=4, augment=augment)
    rows = []
    for int_part, vpd in live.items():
        for vec_part, count in vpd.items():
            # Mirrors factor_k3.py 3.1's own filter EXACTLY, including its
            # `(augment or int_part not in [(3,1),(2,1,1)])` clause -- under
            # AUGMENT those two int_parts are KEPT, which is most of the
            # 40 -> 142 term growth.
            keep = (len(int_part) <= 3
                    and (augment or int_part not in [(3, 1), (2, 1, 1)])
                    and int_part != (1, 1, 1)
                    and (int_part, set(vec_part)) != ((2, 1, 1), {(1, 1, 1)})
                    and factored_keeps_term(3, int_part, vec_part))
            if not keep:
                continue
            k_vec = check_vec_partition(vec_part, len(int_part))
            rows.append({
                "int_part": list(int_part),
                "blocks": [{"vec": list(v), "degree": int(sum(v))} for v in vec_part],
                "scalar": float(count * vec_part_coef(vec_part, divide_fac=True)),
                "count": int(count),
                "wick": [[int(k), int(p)] for k, p in zip(k_vec, int_part)],
            })
    rows.sort(key=lambda r: (len(r["int_part"]), r["int_part"], str(r["blocks"])))
    return rows


def wick_table(terms: list) -> dict:
    """The ReLU Wick coefficients the port needs, split by which branch computes them.

    `wick.relu_wick_coef` has three branches. Two are short closed forms in
    `(sigma, alpha)` and the port writes them out. The third (`k < p`) carries
    polynomials `_relu_wick_poly(p, k)` whose coefficients are pure
    combinatorics, so they are frozen here rather than re-derived in `fnp`.
    """
    needed = {(1, 1), (2, 1)}          # the (1, 1, 1) factored block's w(1), w(2)
    for row in terms:
        for k, p in row["wick"]:
            needed.add((int(k), int(p)))
    out = {}
    for k, p in sorted(needed):
        if k < p:
            P1, P2 = _relu_wick_poly(p, k)
            out[f"{k},{p}"] = {"branch": "poly", "k": k, "p": p,
                               "P1": [float(c) for c in P1.coef],
                               "P2": [float(c) for c in P2.coef]}
        elif p > 1:
            out[f"{k},{p}"] = {"branch": "reduce", "k": k, "p": p,
                               "factorial_p": math.factorial(p), "k_reduced": k - p + 1}
        else:
            out[f"{k},{p}"] = {"branch": "base", "k": k, "p": p}
    return out


def lap_d4_table(m: int, parts) -> dict:
    """`_lap_m_dslice(m, dslice, part)` coefficients, generalised over m and part.

    `lap2_d4_table` below is the `m = 2` SIMPLE case, kept verbatim so the frozen
    artifact's sha is unaffected. AUGMENT needs `m = 1` too: its degree-4 state is
    an `r = 1` HTensor, and `DS_harmonic_proj(., r_out=1, geq=True)` sums the
    `r = 1` term (`L^1`, no raise) and the `r = 2` term (`R(L^2)`).

    The `(2,1,1)` row is the one the port turns on: its blocks are sizes
    `2, 1, 1`, so `m = 1` admits exactly ONE cap-graph and `m = 2` admits NONE.
    That is why the dense `n^3` slice collapses to an axis-0 sum, proven end to
    end in `local_sim/test_p2_fused_reduction.py`.
    """
    out = {}
    for part in parts:
        rows = []
        for graph in weak_compositions(len(part), m):
            if any(2 * graph[i] > part[i] for i in range(len(part))):
                continue
            coef = _multigraph_coef(
                [((i, i), graph[i]) for i in range(len(part)) if graph[i] > 0], part)
            l_part = tuple(part[i] - 2 * graph[i] for i in range(len(part)))
            rows.append({"graph": list(graph), "coef": float(coef),
                         "capped": [i for i in range(len(part)) if l_part[i] == 0],
                         "residual_part": [b for b in l_part if b > 0]})
        out[str(part)] = {"terms": rows, "n_terms": len(rows),
                          "int_partition_coef": int(int_partition_coef(part))}
    return out


def lap2_d4_table() -> dict:
    """`_lap_m_dslice(2, dslice, part)` for part in {(4,), (2, 2)}, m = r = 2."""
    out = {}
    for part in [(4,), (2, 2)]:
        rows = []
        for graph in weak_compositions(len(part), 2):
            if any(2 * graph[i] > part[i] for i in range(len(part))):
                continue
            coef = _multigraph_coef(
                [((i, i), graph[i]) for i in range(len(part)) if graph[i] > 0], part)
            l_part = tuple(part[i] - 2 * graph[i] for i in range(len(part)))
            rows.append({"graph": list(graph), "coef": float(coef),
                         "capped": [i for i in range(len(part)) if l_part[i] == 0],
                         "residual_part": [b for b in l_part if b > 0]})
        out[str(part)] = {"terms": rows,
                          "int_partition_coef": int(int_partition_coef(part))}
    return out


def main() -> int:
    global _TERMS
    augment = bool(getattr(_ARGS, "augment", False))
    _TERMS = term_program(augment=augment)
    doc = {
        "kind": "AUGMENT" if augment else "SIMPLE",
        "source": "mlp_kprop.cumulants._pK_to_K_coef / partitions / harmonic",
        "upstream_rev": "93d091a4",
        "note": "depends only on partition combinatorics; independent of n, of the "
                "weights and of the data",
        "tower": {str(k): [list(p) for p in v] for k, v in TOWER.items()},
        "terms": _TERMS,
        "wick": wick_table(_TERMS),
        "pK_to_K": pk_to_k_table(),
        "lap2_d4": lap2_d4_table(),
        "int_partition_coef": {str(p): int(int_partition_coef(p))
                               for ps in TOWER.values() for p in ps},
        "proj_coef_n1024_d4_r2": [float(x) for x in proj_coef(1024, 4, 2)],
        "proj_coef_formula": "harmonic.proj_coef(n, d, r); evaluated by the port, not frozen",
    }
    if augment:
        # AUGMENT keeps degree 4 at r = 1, so the projection sums the r=1 term
        # (L^1, no raise) and the r=2 term (R(L^2)). Both lap tables are needed.
        d4_parts = [(4,), (3, 1), (2, 2), (2, 1, 1)]
        doc["lap1_d4"] = lap_d4_table(1, d4_parts)
        doc["lap2_d4_augment"] = lap_d4_table(2, d4_parts)
        doc["proj_coef_n1024_d4_r1"] = [float(x) for x in proj_coef(1024, 4, 1)]
        doc["degree4_reduction"] = {
            "claim": "the (2,1,1) slice enters the r=1 core ONLY through its "
                     "axis-0 sum, so the dense n^3 array is never needed",
            "why": "(2,1,1) has blocks of sizes 2,1,1, so L^m caps legs within "
                   "blocks and m=2 admits NO valid cap-graph while m=1 admits "
                   "exactly one",
            "lap2_211_n_terms": doc["lap2_d4_augment"]["(2, 1, 1)"]["n_terms"],
            "verified_end_to_end": "local_sim/test_p2_fused_reduction.py -- a 10x "
                                   "sum-preserving perturbation of the n^3 slice "
                                   "moves the chain output by 1.6e-16, while a "
                                   "sum-breaking one moves it by 2.0e-1",
            "three_axis_blocks_are_rank1": {
                "(2,1,1)": "(1/6) * g[i] * A[j,k]",
                "(1,1,2)": "(1/6) * A[i,j] * g[k]",
                "verified": "~1e-16 vs upstream diagslice, 2 widths, 3 metrics",
            },
        }
    here = os.path.dirname(os.path.abspath(__file__))
    committed = os.path.join(
        os.path.dirname(here), "experiments", "2026-09-02_wp14_fnp",
        "k3_coefs_augment.json" if augment else "k3_coefs_simple.json")
    if _ARGS.check:
        # The analogue of `p2_k3_fnp.py --check-terms`: re-enumerate against the
        # LIVE upstream checkout and refuse on drift, without writing anything.
        # `SKEPTIC WP14-FOLLOWUP` found this table frozen with no such check.
        import hashlib
        with open(committed, encoding="utf-8") as fh:
            old_doc = json.load(fh)
        same = all(old_doc.get(k) == doc.get(k) for k in
                   ("terms", "wick", "pK_to_K", "lap2_d4", "int_partition_coef",
                    "tower"))
        sha = hashlib.sha256(open(committed, "rb").read()).hexdigest()
        print(f"  artifact   {os.path.relpath(committed)}")
        print(f"  sha256     {sha}")
        print(f"  upstream re-enumeration identical to the frozen table: {same}")
        if not same:
            for k in ("terms", "wick", "pK_to_K", "lap2_d4", "int_partition_coef"):
                if old_doc.get(k) != doc.get(k):
                    print(f"  DRIFTED: {k}")
        return 0 if same else 1
    out = _ARGS.out or os.path.join(
        os.path.dirname(here), "experiments", "2026-09-02_wp14_fnp",
        "k3_coefs_simple.json")
    # A generator that silently overwrites the artifact its consumer pins by
    # hash is how a frozen table rots. Overwriting the committed path now needs
    # `--force`, and `--out` sends a re-enumeration to scratch instead.
    if os.path.exists(out) and not _ARGS.force:
        raise SystemExit(
            f"REFUSED: {out} exists. Re-enumerating over it would invalidate "
            "p2_k3_fnp.COEFS_SHA256 silently. Use --out <scratch> to compare, "
            "or --force and update COEFS_SHA256 in the same commit.")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    print(f"  term program: {len(doc['terms'])} kept terms")
    for r in doc["terms"]:
        blocks = [tuple(b["vec"]) for b in r["blocks"]]
        print(f"      int_part {str(tuple(r['int_part'])):<8} blocks {str(blocks):<20} "
              f"scalar {r['scalar']:+g}  wick {[tuple(w) for w in r['wick']]}")
    print()
    print(f"  wick coefficients needed: {len(doc['wick'])}")
    for key, spec in doc["wick"].items():
        extra = (f"P1 {spec['P1']} P2 {spec['P2']}" if spec["branch"] == "poly"
                 else (f"{spec['factorial_p']} * w({spec['k_reduced']}, 1)"
                       if spec["branch"] == "reduce" else ""))
        print(f"      w(k={spec['k']}, p={spec['p']:<2}) {spec['branch']:<7} {extra}")
    print()
    for ip, rows in doc["pK_to_K"].items():
        print(f"  pK_to_K {ip:<10} {len(rows)} vector partitions")
        for r in rows:
            print(f"      {str([tuple(b) for b in r['blocks']]):<40} coef {r['coef']:+g}")
    print()
    for p, blk in doc["lap2_d4"].items():
        print(f"  lap2 {p:<8} int_partition_coef {blk['int_partition_coef']}")
        for r in blk["terms"]:
            print(f"      graph {r['graph']} coef {r['coef']:+g} capped {r['capped']} "
                  f"residual {r['residual_part']}")
    print(f"\n  proj_coef(1024, 4, 2) = {doc['proj_coef_n1024_d4_r2']}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    import argparse

    _ap = argparse.ArgumentParser()
    _ap.add_argument("--augment", action="store_true",
                     help="emit the kind=AUGMENT tables (79 kept terms vs "
                          "SIMPLE's 25) instead of the frozen SIMPLE ones")
    _ap.add_argument("--check", action="store_true",
                     help="re-enumerate against the live upstream checkout and "
                          "refuse on drift; writes nothing")
    _ap.add_argument("--out", default="", help="write here instead of the "
                                              "committed artifact path")
    _ap.add_argument("--force", action="store_true",
                     help="overwrite the committed artifact (then update "
                          "p2_k3_fnp.COEFS_SHA256 in the same commit)")
    _ARGS = _ap.parse_args()
    raise SystemExit(main())
