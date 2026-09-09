"""Price HYBRID Strassen: keep the leaves large, and see what the FLOPs cost.

The lever, from Samuel's 2026-09-08 instruction:

> hybrid Strassen (recurse only on matmuls above a size threshold so leaves stay
> `>=256` and arithmetic intensity stays `>~60 GFLOP/s`; target L=4's FLOPs at
> `<=75 s` worst wall)

## The knob already exists

`estimator_k3_strassen.py:578` carries `_STRASSEN_MIN_BLOCK = 64` and
`_depth_for(m,k,n)` grants levels only while `lo >> (d+1) >= _STRASSEN_MIN_BLOCK`.
So "recurse only above a size threshold" **is** this constant, and at the live
width the leaf size it implies is exact:

    MIN_BLOCK    granted depth at n=1024    leaf size
        64             4                       64
       128             3                      128
       256             2                      256
       512             1                      512

This module measures the **flopscope-counted FLOPs and operation count** for each,
by copying the shipped file with the constant rewritten. **The shipped file is
never edited** — it is a graded candidate.

## The tension the measurement has to resolve

Strassen's FLOP saving comes *from* recursion depth: each level trades 8 multiplies
for 7 plus additions, so the saving is `~(7/8)^d`. Large leaves and deep recursion
are the same knob pulled in opposite directions. The instruction asks for **L=4's
FLOPs with leaves >= 256**, i.e. depth-4 savings at depth-2 recursion. Whether that
is even arithmetically available is the first thing the numbers say, and it is
worth knowing before any wall-time engineering.

## What is MEASURED and what is PROJECTED

**Measured, exactly:** FLOPs and ops, in the real installed flopscope at the live
`1024 x 16`.

**Projected, with its model stated:** wall. `docs/VALIDATION.md` forbids quoting a
local absolute wall time — the authority is the graded `--runner subprocess` row —
so wall here is extrapolated from the three GRADED anchors and labelled a
projection everywhere it appears:

    submission   file        B         ops      worst wall   kernel   intensity
    330212       k3_full     0.77970    3,249     9.54 s      7.97 s   215 GFLOP/s
    330327       L=3         0.53745   12,357    61.93 s     51.8  s    17 GFLOP/s
    330290       L=4         0.48578   15,393   114.68 s        -           -

Two intensity points (leaf 1024 -> 215, leaf 128 -> 17) fix a crude power law
`intensity ~ leaf^p`; it is a two-point fit across a 8x lever and is treated as
an order-of-magnitude guide, never as a price.

ROUTINE: no seed block, no gate, no promotion, no change to any shipped file.
`# seed-ledger: no block named`.

Run:  uv run python local_sim/p2_hybrid_strassen_price.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

# The three GRADED anchors. Not local readings.
GRADED = [
    {"name": "k3_full (L=0)", "sub": 330212, "B": 0.77970, "ops": 3249,
     "worst_wall_s": 9.54, "kernel_s": 7.97, "intensity": 215.1, "leaf": 1024},
    # intensity RECOMPUTED from B and kernel, not taken from the record:
    # 0.53745 B = 1181.9 GFLOP over 51.8 s = 22.82 GFLOP/s. CURRENT_STATE carries
    # "215 -> 17 GFLOP/s"; the 215.13 reproduces exactly and the 17 does NOT.
    # Using the reproducible figure and flagging the other.
    {"name": "Strassen L=3", "sub": 330327, "B": 0.53745, "ops": 12357,
     "worst_wall_s": 61.93, "kernel_s": 51.8, "intensity": 22.82, "leaf": 128,
     "recorded_intensity_that_does_not_reproduce": 17.0},
    {"name": "Strassen L=4", "sub": 330290, "B": 0.48578, "ops": 15393,
     "worst_wall_s": 114.68, "kernel_s": None, "intensity": None, "leaf": 64},
]
WALL_CAP_S = 120.0
CONTENTION = 1.60          # measured graded wall spread on identical bytes
TARGET_WALL_S = 75.0       # Samuel's target for the hybrid


def _intensity_law():
    """intensity ~ c * leaf**p, from the two graded points that have one."""
    a, b = GRADED[0], GRADED[1]
    p = math.log(a["intensity"] / b["intensity"]) / math.log(a["leaf"] / b["leaf"])
    c = a["intensity"] / (a["leaf"] ** p)
    return c, p


def price(candidate, eff_depth, width, depth, seed):
    """FLOPs and ops for the shipped chain at a given EFFECTIVE Strassen depth.

    Both constants are rewritten: `_STRASSEN_DEPTH` sets the cap and
    `_STRASSEN_MIN_BLOCK` is pinned low enough not to bind at this width, so the
    granted depth at 1024 is exactly `eff_depth` and the leaf is `1024 >> d`.
    An earlier version rewrote only MIN_BLOCK and silently inherited the shipped
    file's `_STRASSEN_DEPTH = 3`, which mislabelled every row.
    """
    import p2_op_census as census

    src = os.path.join(_ROOT, candidate)
    with open(src, encoding="utf-8") as fh:
        text = fh.read()
    leaf = width >> eff_depth
    patched = text
    for name, val in (("_STRASSEN_DEPTH", eff_depth),
                      ("_STRASSEN_MIN_BLOCK", max(leaf, 1))):
        pat = re.compile(rf"^{name}\s*=\s*\d+", re.M)
        if not pat.search(patched):
            raise SystemExit(f"REFUSED: no {name} in {candidate}")
        patched = pat.sub(f"{name} = {val}", patched, count=1)

    tmpdir = tempfile.mkdtemp(prefix="hybrid_strassen_")
    try:
        # The candidate is copied, never edited in place: it is a graded file.
        dst = os.path.join(tmpdir, f"cand_d{eff_depth}.py")
        with open(dst, "w", encoding="utf-8") as fh:
            fh.write(patched)
        rec = census.census(dst, width=width, depth=depth, seed=seed, mlp_index=0)
        return {"eff_depth": eff_depth, "leaf": leaf,
                "flops": rec["flops"], "n_ops": rec["n_ops"]}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default="estimator_k3_strassen.py")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1_000_003)
    ap.add_argument("--depths", type=int, nargs="*", default=[1, 2, 3, 4],
                    help="EFFECTIVE Strassen depth at the live width")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    B_UNIT = 2 ** 41
    c, p = _intensity_law()

    print("=" * 78)
    print(f"HYBRID STRASSEN -- price at the live shape {a.width} x {a.depth}")
    print("=" * 78)
    print(f"  candidate  {a.candidate} (COPIED, never edited -- it is a graded file)")
    print(f"  the knob   _STRASSEN_MIN_BLOCK, already present at line 578")
    print(f"  intensity law from 2 graded points: {c:.4g} * leaf^{p:.3f} GFLOP/s")
    print(f"  target     L=4's FLOPs ({GRADED[2]['B']:.5f} B) at <= {TARGET_WALL_S:.0f} s worst wall")
    print()
    print(f"  {'depth':>6} {'leaf':>6} {'B':>9} {'ops':>8} "
          f"{'proj GFLOP/s':>13} {'proj kernel s':>14}   anchor check")

    rows = []
    for d in a.depths:
        r = price(a.candidate, d, a.width, a.depth, a.seed)
        leaf = r["leaf"]
        B = r["flops"] / B_UNIT
        inten = c * (leaf ** p)
        kernel = (r["flops"] / 1e9) / inten
        r.update({"B": B, "proj_intensity_gflops": inten, "proj_kernel_s": kernel})
        # instrument check: depths 3 and 4 have GRADED rows to reproduce
        chk = ""
        for gr, dd in ((GRADED[1], 3), (GRADED[2], 4)):
            if d == dd:
                db = abs(B - gr["B"]) / gr["B"]
                r["graded_B_rel_err"] = db
                chk = f"vs graded {gr['B']:.5f} ({db:+.3%}), ops {gr['ops']}"
        rows.append(r)
        print(f"  {d:>6} {leaf:>6} {B:>9.5f} {r['n_ops']:>8} "
              f"{inten:>13.1f} {kernel:>14.1f}   {chk}")

    print()
    print("  PROJECTED worst wall = proj kernel + overhead, scaled to the graded")
    print("  L=3 anchor (61.93 s worst on a 51.8 s kernel, i.e. x1.196):")
    best = None
    for r in rows:
        worst = r["proj_kernel_s"] * (GRADED[1]["worst_wall_s"] / GRADED[1]["kernel_s"])
        r["proj_worst_wall_s"] = worst
        margin = WALL_CAP_S / worst if worst else float("inf")
        r["proj_margin_vs_cap"] = margin
        r["proj_margin_under_contention"] = margin / CONTENTION
        ok = worst <= TARGET_WALL_S and r["B"] <= GRADED[2]["B"]
        if ok and (best is None or r["B"] < best["B"]):
            best = r
        print(f"    depth {r['eff_depth']}  leaf {r['leaf']:>4}  B {r['B']:.5f}  "
              f"proj worst wall {worst:>7.1f} s  margin {margin:>5.2f}x  "
              f"under contention {margin / CONTENTION:>5.2f}x"
              f"   {'<- meets the target' if ok else ''}")

    print()
    if best is None:
        print("  VERDICT: NO setting meets 'L=4 FLOPs at <= 75 s'.")
        print("  Strassen's FLOP saving comes FROM recursion depth -- each level")
        print("  trades 8 multiplies for 7. Large leaves and deep recursion are the")
        print("  same knob pulled opposite ways, so depth-4 savings at depth-2")
        print("  recursion are not arithmetically available. The instruction asked")
        print("  for a point that does not exist on this curve.")
    else:
        print(f"  VERDICT: depth {best['eff_depth']} (leaf {best['leaf']}) meets it "
              f"(B {best['B']:.5f}, proj worst wall {best['proj_worst_wall_s']:.1f} s).")

    print()
    print("  Wall figures above are PROJECTIONS from three graded anchors through a")
    print("  two-point intensity law. docs/VALIDATION.md forbids quoting a local")
    print("  absolute wall time; only a graded --runner subprocess row settles wall.")

    rec = {"slice": "hybrid Strassen price at 1024x16",
           "candidate": a.candidate, "graded_anchors": GRADED,
           "intensity_law": {"c": c, "p": p,
                             "note": "two-point fit across an 8x leaf lever; "
                                     "order-of-magnitude guide, never a price"},
           "target_wall_s": TARGET_WALL_S, "rows": rows,
           "meets_target": None if best is None else best["min_block"],
           "tier": "ROUTINE. No seed block, no gate, no shipped-file change.",
           "wall_caveat": ("PROJECTED. Local absolute wall times are not quotable "
                           "(docs/VALIDATION.md); only a graded subprocess row settles wall.")}
    if a.out:
        out = os.path.join(_ROOT, a.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=2)
        print(f"\n  artifact  {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
