"""Compose ONE release candidate from the slice's independent pieces, in one place.

Handoff 2026-09-17b section 3: *"Combine with the coherent accuracy candidate and
verified P4 algebra; measure the actual combined file."* This is that composition,
written once so the file that gets screened is the file that gets certified and
nobody re-derives the stack by hand.

THE STACK, in the order the generators must run
------------------------------------------------
    frozen estimator_k3_v32.py
      -> s6_build_closure   OUR closure table's extra terms, recurrent   (accuracy)
      -> s2_build_p3        the kappa3 x C distinct-index birth feed      (accuracy)
      -> s7_build_shared    shared old-lineage coordinates, lean          (cost)
      -> s10_query_basis    the consumed-query basis for the join         (cost/accuracy)
      -> s4_build_p4        exact final-var + block-symmetric sandwich    (cost)

Every step is an ASSERTED patch on the previous text: an anchor whose count is
wrong stops the build rather than silently emitting something else. The order is
forced -- s10 rewrites the lean join that s7 introduces, and s4's anchors are
whole-function spans that the earlier steps leave intact.

WHAT THIS IS NOT
----------------
It is not a search. Every knob it exposes is named in the committed contract:
which closure table, the share age and rank, whether the join uses the query
basis, and the P4 depth. Nothing here fits a constant, reads truth or chooses a
configuration; it emits exactly the files a preregistered comparison names.

Run:  uv run python local_sim/s11_build_candidate.py --spec <name>=<table>,<age>,<rank>,<basis>,<depth> --emit
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]

V32 = "estimator_k3_v32.py"


def compose(table, age, rank, basis, depth, *, min_block=32, d3=0, passes=1,
            lean=True, title="", p3=0, p3_comp=True):
    """Return the composed source text and a provenance record."""
    import s2_build_p3 as P3
    import s4_build_p4 as P4
    import s6_build_closure as CL
    import s7_build_shared as SH
    import s10_query_basis as QB

    prov = {"base": V32, "base_sha256": _sha(os.path.join(_ROOT, V32)),
            "closure_table": table, "share_age": age, "share_rank": rank,
            "join_basis": basis, "p4_depth": depth, "p4_min_block": min_block,
            "qb_d3_columns": d3, "share_passes": passes, "lean": lean,
            "steps": []}

    src = open(os.path.join(_ROOT, V32), encoding="utf-8").read()

    # 1. closure -------------------------------------------------------------
    if table:
        prog = CL.program_json(os.path.join(_ROOT, table), "all")
        src = CL.build(src, "all", False, 32, prog,
                       title or "composed candidate: closure %s" % os.path.basename(table))
        prov["closure_table_sha256"] = _sha(os.path.join(_ROOT, table))
        prov["closure_extra_uni"] = len(prog["uni"])
        prov["closure_extra_biv"] = len(prog["biv"])
        prov["steps"].append("s6_build_closure(scope=all, table=%s)" % table)

    # 2. the kappa3 x covariance distinct-index birth feed (P3) --------------
    # AFTER the closure and BEFORE the tower rewrite. Both orderings were tried
    # and both failed loudly, which is what the count-asserted anchors are for:
    #   * P3 first  -> the closure's own cost-model patch anchors on the block P3
    #                  has already rewritten, and stops.
    #   * P3 after s7 -> s7 rewrites `_dslice3`, `_dslice21` and `_append`, which
    #                  are exactly what P3 patches, and stops.
    # The cost model is the one place the two generators collide: they patch the
    # SAME budget-guard block. P3's contribution is extracted as the lines it
    # inserts before `return base + thin` and inserted before the current text's
    # single occurrence of that line, which is the same edit against a block the
    # closure has already extended.
    if p3 and age and rank:
        raise SystemExit(
            "P3 together with the shared-coordinate tower is NOT SUPPORTED yet. "
            "Both generators extend `_Tower.__slots__` and `__init__`, and P3 "
            "anchors on the very consumers (`_dslice3`, `_dslice21`, `_append`) "
            "that s7 rewrites, so there is no order in which all the anchors "
            "survive. The collisions are mechanical and each is a place to get "
            "it wrong, so the integration is DEFERRED until P3 has been shown "
            "to pay on its own -- which is what `estimator_k3_v33.py` is in "
            "batch B for. Refusing loudly beats emitting a file whose patches "
            "half-landed.")
    if p3:
        tail = "    return base + thin"
        for old, new, why, count in P3._patches():
            if why == "cost-model":
                oh, nh = old[:old.rindex(tail)], new[:new.rindex(tail)]
                if not nh.startswith(oh):
                    raise SystemExit("P3's cost-model patch is not an insertion")
                insert = nh[len(oh):]
                if src.count(tail) != 1:
                    raise SystemExit("P3 cost model: %d budget-guard tails, expected 1"
                                     % src.count(tail))
                src = src.replace(tail, insert + tail, 1)
                continue
            found = src.count(old)
            if found != count:
                raise SystemExit("P3 PATCH ANCHOR COUNT WRONG in the composition "
                                 "(%s): expected %d, found %d" % (why, count, found))
            src = src.replace(old, new, count)
        flags = ("#: P3: the kappa3 x C distinct-index birth feed.\n"
                 "_P3_ON = True\n_P3_RANK = %d\n_P3_COMPENSATE = %r\n\n"
                 "_P2_RANK =" % (int(p3), bool(p3_comp)))
        if src.count("_P2_RANK =") < 1:
            raise SystemExit("no _P2_RANK anchor for P3's flags")
        src = src.replace("_P2_RANK =", flags, 1)
        # P3's RUNTIME helpers (`_p3_compress`, `_p3_parts`) come from the
        # generator's own emitted block, not from `_patches()`. Omitting them
        # produced a file whose `_dslice3` called a function that did not exist.
        fa = "def _dslice3(tw):"
        if src.count(fa) != 1:
            raise SystemExit("P3 runtime: %d `_dslice3` definitions, expected 1"
                             % src.count(fa))
        src = src.replace(fa, P3._P3_FUNCS.lstrip(chr(10)) + fa, 1)
        prov["p3_rank"] = int(p3)
        prov["p3_compensate"] = bool(p3_comp)
        prov["steps"].append("s2_build_p3._patches(rank=%d, compensate=%s)"
                             % (int(p3), bool(p3_comp)))

    # 3. shared old-lineage coordinates -------------------------------------
    if age and rank:
        src = SH.build(src, age, rank, passes, False, on=True, lean=lean)
        prov["steps"].append("s7_build_shared(age=%d, rank=%d, passes=%d, lean=%s)"
                             % (age, rank, passes, lean))

        # 4. the join's basis ------------------------------------------------
        if basis in ("qE", "qEd"):
            src = QB.build(src, True, d3 if basis == "qEd" else 0)
            prov["steps"].append("s10_query_basis(query_basis=True, d3=%d)"
                                 % (d3 if basis == "qEd" else 0))
        elif basis == "gram":
            src = QB.build(src, False, 0)
            prov["steps"].append("s10_query_basis(query_basis=False)  # parity spelling")
        elif basis != "none":
            raise SystemExit("unknown join basis %r" % basis)

    # 5. P4 ------------------------------------------------------------------
    if depth:
        src = P4.build(src, final_var=True, blocksym=True, depth=depth,
                       min_block=min_block)
        prov["steps"].append("s4_build_p4(final_var, blocksym, depth=%d, min_block=%d)"
                             % (depth, min_block))
    return src, prov


def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def emit(name, table, age, rank, basis, depth, out_dir=_ROOT, **kw):
    src, prov = compose(table, age, rank, basis, depth, **kw)
    head = ("%s: COMPOSED by `local_sim/s11_build_candidate.py`.\n\n"
            "base %s (sha256 %s)\n%s\n\n"
            "DO NOT EDIT. Edit the generators and re-emit. Not approved for\n"
            "submission until a preregistered gate and Samuel say so.\n\n"
            "--------------------------------------------------------------\n"
            % (name, prov["base"], prov["base_sha256"][:16],
               "\n".join("  -> " + s for s in prov["steps"])))
    # INSERT into the existing module docstring rather than replacing it. An
    # earlier spelling cut from the first triple quote to the second and left an
    # unterminated string, because the inherited header is not the only triple
    # quote near the top of a generated file. Inserting cannot lose a delimiter.
    i = src.find('"""')
    if i < 0:
        raise SystemExit("composed text has no module docstring to annotate")
    src = src[:i + 3] + head + src[i + 3:]
    p = os.path.join(out_dir, name)
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(src)
    prov["path"] = name
    prov["sha256"] = _sha(p)
    print("  wrote %-46s sha256 %s" % (name, prov["sha256"][:16]))
    for s in prov["steps"]:
        print("        %s" % s)
    return p, prov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", action="append", required=True,
                    help="NAME=TABLE,AGE,RANK,BASIS,DEPTH[,P3RANK] ; TABLE may "
                         "be '-' for no closure change, BASIS one of "
                         "gram|qE|qEd|none, P3RANK 0 for no birth feed")
    ap.add_argument("--out-dir", default=_ROOT)
    ap.add_argument("--emit", action="store_true")
    a = ap.parse_args()
    provs = []
    for spec in a.spec:
        name, rest = spec.split("=", 1)
        parts = rest.split(",")
        table, age, rank, basis, depth = parts[:5]
        p3 = int(parts[5]) if len(parts) > 5 else 0
        if a.emit:
            _p, prov = emit(name, None if table == "-" else table,
                            int(age), int(rank), basis, int(depth), a.out_dir,
                            p3=p3)
            provs.append(prov)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
