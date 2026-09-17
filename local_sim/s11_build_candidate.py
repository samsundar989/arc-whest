"""Compose ONE release candidate from the slice's independent pieces, in one place.

Handoff 2026-09-17b section 3: *"Combine with the coherent accuracy candidate and
verified P4 algebra; measure the actual combined file."* This is that composition,
written once so the file that gets screened is the file that gets certified and
nobody re-derives the stack by hand.

THE STACK, in the order the generators must run
------------------------------------------------
    frozen estimator_k3_v32.py
      -> s6_build_closure   OUR closure table's extra terms, recurrent   (accuracy)
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
            lean=True, title=""):
    """Return the composed source text and a provenance record."""
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

    # 2. shared old-lineage coordinates -------------------------------------
    if age and rank:
        src = SH.build(src, age, rank, passes, False, on=True, lean=lean)
        prov["steps"].append("s7_build_shared(age=%d, rank=%d, passes=%d, lean=%s)"
                             % (age, rank, passes, lean))

        # 3. the join's basis ------------------------------------------------
        if basis in ("qE", "qEd"):
            src = QB.build(src, True, d3 if basis == "qEd" else 0)
            prov["steps"].append("s10_query_basis(query_basis=True, d3=%d)"
                                 % (d3 if basis == "qEd" else 0))
        elif basis == "gram":
            src = QB.build(src, False, 0)
            prov["steps"].append("s10_query_basis(query_basis=False)  # parity spelling")
        elif basis != "none":
            raise SystemExit("unknown join basis %r" % basis)

    # 4. P4 ------------------------------------------------------------------
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
                    help="NAME=TABLE,AGE,RANK,BASIS,DEPTH ; TABLE may be '-' for "
                         "no closure change, BASIS one of gram|qE|qEd|none")
    ap.add_argument("--out-dir", default=_ROOT)
    ap.add_argument("--emit", action="store_true")
    a = ap.parse_args()
    provs = []
    for spec in a.spec:
        name, rest = spec.split("=", 1)
        table, age, rank, basis, depth = rest.split(",")
        if a.emit:
            _p, prov = emit(name, None if table == "-" else table,
                            int(age), int(rank), basis, int(depth), a.out_dir)
            provs.append(prov)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
