"""SLICE 4, ITEM 1 -- what does the AUGMENT port actually COST on our chain?

*Strategist handoff `docs/STRATEGIST_RESPONSE_2026-09-09.md` section 3 item 1:
"Meter the fused AUGMENT route: FLOPs, ops, largest array, projected wall at
Strassen depth 2 and 3. Report before transcription. Stop rule: none -- this is
information."*

**The answer is that the port is priced at ~7.5x our chain's FLOPs and cannot
ship.** This module is the measurement behind that sentence.

WHY THIS IS NOT THE `~0.31 B` SANITY ESTIMATE OF 2026-09-08
------------------------------------------------------------
Every previous AUGMENT cost note in this repository priced the **fused `(2,1,1)`
route** -- the `n^3` object that OOMed jobs 399 and 402 and was then proven
reducible to `n^2` end to end. That work is correct and it stands. It priced the
wrong thing. The `(2,1,1)` route is a few per cent of the AUGMENT bill.

What nobody priced is that **`kind=AUGMENT` widens the carried third-cumulant
lineage from `2n` columns per layer to `8n`**, and adds two entirely new
rank-`R` contractions (`pK_111 -> K_211`, `pK_211 -> K_211`) that do not exist in
`SIMPLE` at all. Those items are ~92% of the bill.

TWO INDEPENDENT INSTRUMENTS, AND THEY AGREE TO THREE SIGNIFICANT FIGURES
-------------------------------------------------------------------------
1. **Structural.** Run upstream's own `factor=True` flow and read the factored
   tensor's column count at every layer. `SIMPLE` grows `2n` per layer, `AUGMENT`
   grows `8n` -- measured at `(n, depth) = (24, 6)` and `(32, 5)`, identical.
   Summed over a 16-layer tower that is `885` block-layers against `240`, a
   ratio of **`3.6875`**.
2. **Upstream's own FLOP model.** `fit_flop_polys` at `n = 1024`, `act15`,
   `factor=True`: `contract_W` moves `x3.69` and `get_dslice` moves `x3.68`.

The structural count and the FLOP model were computed from different code paths
and agree on `3.69`. That is why the projection below is quotable rather than an
estimate.

THE METER CAUTION, OBEYED
--------------------------
`docs/CURRENT_STATE.md` records a `5.49x` "affordability target" that turned out
to compare **two different meters** and must never be used as a projection.
Everything upstream-sourced here is a **RATIO measured inside upstream's own
meter with both arms run identically**, exactly as `local_sim/p2_augment.py`
argues. It is applied to our flopscope-anchored total per component. No absolute
upstream FLOP figure is quoted as a cost.

WALL IS PROJECTED AND SAYS SO
------------------------------
`docs/VALIDATION.md` forbids quoting a local absolute wall time. Wall here comes
from the same two-point graded intensity law as
`local_sim/p2_hybrid_strassen_price.py`, which validated **out of sample** at
`+3.1%` on the L=4 graded row. At a FIXED Strassen depth the leaf is fixed, so
intensity is fixed and wall scales linearly in FLOPs -- the least speculative use
that model has been put to.

Run:
    .venv-gpu\\Scripts\\python.exe local_sim/p2_augment_port_price.py --measure
    uv run python local_sim/p2_augment_port_price.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

OUT_DIR = os.path.join(_ROOT, "experiments", "2026-09-09_slice4")
OUT = os.path.join(OUT_DIR, "augment_port_price.json")

WIDTH, DEPTH = 1024, 16
BUDGET = 2 ** 41

# ---------------------------------------------------------------------------
# our chain, ANCHORED. Rows 3 and 4 are reproduced by the grader's own ledger to
# +0.003% (`p2_hybrid_strassen_price.py`); row 0 is the shipped k3_full's graded
# total. `ops` from `p2_op_census.py` on this host.
# ---------------------------------------------------------------------------
OURS = {
    0: {"flops": 1_714_554_351_206, "ops": 3_232, "leaf": 1024},
    1: {"flops": 1_506_154_327_654, "ops": 6_268, "leaf": 512},
    2: {"flops": 1_328_654_462_566, "ops": 9_304, "leaf": 256},
    3: {"flops": 1_181_829_852_774, "ops": 12_340, "leaf": 128},
    4: {"flops": 1_068_211_920_486, "ops": 15_376, "leaf": 64},
}

#: Graded anchors used by the intensity law. Identical to
#: `p2_hybrid_strassen_price.GRADED`; duplicated rather than imported so this
#: module states the two points it extrapolates from.
GRADED_LEAF_HI = {"leaf": 1024, "intensity": 215.13, "kernel_s": 7.97,
                  "worst_wall_s": 9.54}
GRADED_LEAF_LO = {"leaf": 128, "intensity": 22.82, "kernel_s": 51.8,
                  "worst_wall_s": 61.93}
CONTENTION = 1.60
WALL_CAP_S = 120.0
#: The nominee rule adopted in the Sep 9 handoff.
MARGIN_RULE = 2.0

#: Measured by `--measure`; frozen here so `--price` runs without torch.
#: Births are columns of the factored K3 tensor added per layer, in units of n.
RANK_SCHEDULE = {
    "SIMPLE": {"birth_first": 2, "birth_steady": 2},
    "AUGMENT": {"birth_first": 3, "birth_steady": 8},
    "measured_at": [{"n": 24, "depth": 6}, {"n": 32, "depth": 5}],
}

#: Upstream `fit_flop_polys`, `n = 1024`, `act15`, `factor=True`, both arms
#: identical apart from `kind`. RATIOS ONLY -- see the meter caution.
UPSTREAM_1024 = {
    "SIMPLE": {
        "total": 3.253706e12,
        "FactoredTensor3.get_dslice": 1.55598e12,
        "FactoredTensor3.contract_W": 1.54619e12,
        "nonlin_sum 111 factored": 9.77903e10,
        "contract_W_basic": 5.15899e10,
        "pK_to_K": 1.10843e9, "nonlin_sum": 5.83827e8, "eval_part": 2.61218e8,
        "harmonic diagslice": 9.65304e7, "DS_harmonic_proj": 1.67938e7,
    },
    "AUGMENT": {
        "total": 2.660099e13,
        "pK_111 -> K_211": 6.50360e12,
        "pK_211 -> K_211": 5.81245e12,
        "FactoredTensor3.get_dslice": 5.72661e12,
        "FactoredTensor3.contract_W": 5.70157e12,
        "pK_to_K": 7.40497e11, "nonlin_sum": 7.03856e11,
        "nonlin_sum 111 factored": 6.81977e11, "eval_part": 5.64021e11,
        "contract_W_basic": 9.99240e10, "harmonic diagslice": 4.84935e10,
        "DS_harmonic_proj": 1.77000e10,
    },
}

#: The components our FUSED route makes `n^2` and therefore does NOT pay.
#: Upstream evaluates the `(2,1,1)` and `(3,1)` int_parts DENSELY at `n^3`;
#: 2026-09-09 proved end to end that the whole degree-4 route fits `n^2`
#: (a 10x sum-preserving perturbation moves the output by 1.6e-16).
FUSED_AWAY = ("pK_to_K", "nonlin_sum", "eval_part", "harmonic diagslice",
              "DS_harmonic_proj")

#: The tower components -- the ones our W-EFF spelling already owns, and whose
#: ratio therefore transfers to our anchored total.
TOWER = ("FactoredTensor3.get_dslice", "FactoredTensor3.contract_W")

#: Everything AUGMENT adds that has no SIMPLE counterpart and is NOT fused away.
#: Both are rank-R contractions of the same class as `contract_W`, so our
#: spelling has no known structural advantage over upstream's.
NEW_RANK_R = ("pK_111 -> K_211", "pK_211 -> K_211")

#: Share of our anchored total that is tower work. From the shape model in
#: `estimator_k3_strassen._matmuls`: 990 of 1022 matmul-equivalents.
OUR_TOWER_SHARE = 990.0 / 1022.0


def _block_layers(birth_first, birth_steady, depth):
    """Column-blocks (units of n) alive, summed over the layers that advance them.

    Layer `l` enters with the tower built by layers `0..l-1`:
    `birth_first + birth_steady * (l - 1)` blocks.
    """
    return sum(birth_first + birth_steady * (l - 1) for l in range(1, depth))


def _intensity_law():
    a, b = GRADED_LEAF_HI, GRADED_LEAF_LO
    p = math.log(a["intensity"] / b["intensity"]) / math.log(a["leaf"] / b["leaf"])
    c = a["intensity"] / (a["leaf"] ** p)
    return c, p


def project():
    up_s, up_a = UPSTREAM_1024["SIMPLE"], UPSTREAM_1024["AUGMENT"]

    bl_s = _block_layers(RANK_SCHEDULE["SIMPLE"]["birth_first"],
                         RANK_SCHEDULE["SIMPLE"]["birth_steady"], DEPTH)
    bl_a = _block_layers(RANK_SCHEDULE["AUGMENT"]["birth_first"],
                         RANK_SCHEDULE["AUGMENT"]["birth_steady"], DEPTH)
    structural = bl_a / bl_s

    up_tower_s = sum(up_s[k] for k in TOWER)
    up_tower_a = sum(up_a[k] for k in TOWER)
    meter_tower = up_tower_a / up_tower_s

    up_new = sum(up_a[k] for k in NEW_RANK_R)
    new_over_simple_tower = up_new / up_tower_s

    up_fused_away = sum(up_a.get(k, 0.0) for k in FUSED_AWAY)

    rows = []
    for d, row in sorted(OURS.items()):
        base = row["flops"]
        tower = base * OUR_TOWER_SHARE
        rest = base - tower
        aug_tower = tower * meter_tower
        aug_new = tower * new_over_simple_tower
        aug = aug_tower + aug_new + rest
        rows.append({
            "strassen_depth": d, "leaf": row["leaf"], "ops": row["ops"],
            "ours_flops": base, "ours_B": base / BUDGET,
            "augment_flops": aug, "augment_B": aug / BUDGET,
            "ratio": aug / base,
            "component_tower_B": aug_tower / BUDGET,
            "component_new_rank_r_B": aug_new / BUDGET,
            "component_rest_B": rest / BUDGET,
        })

    c, p = _intensity_law()
    kernel_to_worst = GRADED_LEAF_LO["worst_wall_s"] / GRADED_LEAF_LO["kernel_s"]
    for r in rows:
        inten = c * (r["leaf"] ** p)
        r["proj_intensity_gflops"] = inten
        for tag, f in (("ours", r["ours_flops"]), ("augment", r["augment_flops"])):
            kernel = (f / 1e9) / inten
            worst = kernel * kernel_to_worst
            r[tag + "_proj_kernel_s"] = kernel
            r[tag + "_proj_worst_wall_s"] = worst
            r[tag + "_margin"] = WALL_CAP_S / worst
            r[tag + "_margin_contended"] = (WALL_CAP_S / worst) / CONTENTION

    return {
        "structural_ratio": structural,
        "structural_block_layers": {"SIMPLE": bl_s, "AUGMENT": bl_a},
        "upstream_meter_tower_ratio": meter_tower,
        "upstream_meter_total_ratio": up_a["total"] / up_s["total"],
        "new_rank_r_over_simple_tower": new_over_simple_tower,
        "upstream_fused_away_flops": up_fused_away,
        "upstream_fused_away_share_of_augment": up_fused_away / up_a["total"],
        "intensity_law": {"c": c, "p": p},
        "rows": rows,
    }


def cmd_measure(a) -> int:
    """Re-measure the rank schedule and the upstream ratios. NEEDS torch."""
    sys.path.insert(0, a.upstream)
    import torch
    import numpy as np
    from mlp_kprop.kprop_harmonic import SIMPLE, AUGMENT, mlp_kprop
    from mlp_kprop.flop_utils import fit_flop_polys
    import p2_kprop_bridge as BRIDGE

    print("  RANK SCHEDULE -- upstream factor=True, read off the tensor itself\n")
    sched = {}
    for n, depth in ((24, 6), (32, 5)):
        rng = np.random.default_rng(n)
        W = [rng.standard_normal((n, n)) * (2.0 / n) ** 0.5 for _ in range(depth)]
        for kind in (SIMPLE, AUGMENT):
            name = str(kind).split(".")[1]
            mlp = BRIDGE.build_mlp(W, device="cpu", dtype=torch.float64)
            K_in = {1: torch.zeros(n, dtype=torch.float64),
                    2: torch.eye(n, dtype=torch.float64)}
            ranks = []
            for l in range(depth):
                out = mlp_kprop(mlp, K_in, k_max=3, output_all=True, kind=kind,
                                use_avg_metric=True, factor=True,
                                up_to_layer="act%d" % l, output_d_max=4)
                ranks.append(int(out["act%d" % l][3].factors[0].shape[1]) // n)
            births = sorted({ranks[i + 1] - ranks[i]
                             for i in range(len(ranks) - 1)})
            print("    n=%-3d depth=%d  %-8s  ranks/n %s   births/n %s"
                  % (n, depth, name, ranks, births))
            sched.setdefault(name, []).append(
                {"n": n, "depth": depth, "ranks_over_n": ranks,
                 "births_over_n": births})

    print("\n  UPSTREAM FLOP MODEL -- ratios inside ONE meter, both arms identical\n")
    polys, tots, comp = {}, {}, {}
    for name, kind in (("SIMPLE", SIMPLE), ("AUGMENT", AUGMENT)):
        polys[name] = fit_flop_polys({"nonlin": "relu"},
                                     {"k_max": 3, "factor": True, "kind": kind})
    for name in polys:
        o = {k: float(pp(WIDTH, DEPTH - 1)) for k, pp in polys[name].items()}
        comp[name] = o
        tots[name] = o["total"] - o.get("mlp setup", 0.0)
    print("    n=%d act%d   SIMPLE %.6e   AUGMENT %.6e   ratio %.4f"
          % (WIDTH, DEPTH - 1, tots["SIMPLE"], tots["AUGMENT"],
             tots["AUGMENT"] / tots["SIMPLE"]))
    for k in sorted(set(comp["SIMPLE"]) | set(comp["AUGMENT"]),
                    key=lambda k: -abs(comp["AUGMENT"].get(k, 0.0))):
        s, aa = comp["SIMPLE"].get(k, 0.0), comp["AUGMENT"].get(k, 0.0)
        if abs(aa) > 1e8 or abs(s) > 1e8:
            print("      %-36s %10.4e -> %10.4e   %s"
                  % (k, s, aa, ("x%.2f" % (aa / s)) if s > 0 else "NEW"))
    print("\n  Ratio is shape-independent -- check across n:")
    for n_ in (128, 256, 512, 1024, 2048):
        t = {}
        for name in polys:
            o = {k: float(pp(n_, DEPTH - 1)) for k, pp in polys[name].items()}
            t[name] = o["total"] - o.get("mlp setup", 0.0)
        print("      n=%-5d ratio %.4f" % (n_, t["AUGMENT"] / t["SIMPLE"]))

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump({"rank_schedule": sched, "upstream_components": comp,
                       "upstream_totals": tots,
                       "meter": ("upstream torch NamedFlopCounter -- NOT "
                                 "flopscope; ratios only")}, fh, indent=1)
        print("\n  wrote %s" % a.out)
    return 0


def cmd_price(a) -> int:
    d = project()
    print("=" * 78)
    print("  THE AUGMENT PORT'S PRICE ON OUR CHAIN")
    print("=" * 78)
    print("\n  the lineage, MEASURED on upstream's own factored flow:")
    print("      SIMPLE   births 2n columns per layer   %4d block-layers at depth 16"
          % d["structural_block_layers"]["SIMPLE"])
    print("      AUGMENT  births 8n columns per layer   %4d block-layers at depth 16"
          % d["structural_block_layers"]["AUGMENT"])
    print("      structural ratio                       %.4f" % d["structural_ratio"])
    print("      SAME ratio from upstream's FLOP model  %.4f"
          % d["upstream_meter_tower_ratio"])
    print("      -> two independent instruments, agreeing to 3 s.f.")
    print("\n  what AUGMENT adds that SIMPLE has no counterpart for:")
    print("      pK_111 -> K_211 and pK_211 -> K_211, both rank-R contractions,")
    print("      together %.3fx the whole SIMPLE tower."
          % d["new_rank_r_over_simple_tower"])
    print("\n  what our FUSED (2,1,1) route saves that upstream pays:")
    print("      %.3e upstream FLOPs = %.1f%% of upstream's AUGMENT total"
          % (d["upstream_fused_away_flops"],
             100 * d["upstream_fused_away_share_of_augment"]))
    print("      (the dense n^3 int_parts. Real, correct -- and a few per cent.)")

    print("\n  %-6s %-6s %9s %9s %7s   %11s %11s"
          % ("depth", "leaf", "ours B", "AUG B", "ratio", "AUG kernel", "AUG worst"))
    print("  " + "-" * 74)
    for r in d["rows"]:
        print("  %-6d %-6d %9.5f %9.5f %6.2fx   %9.1f s %9.1f s"
              % (r["strassen_depth"], r["leaf"], r["ours_B"], r["augment_B"],
                 r["ratio"], r["augment_proj_kernel_s"],
                 r["augment_proj_worst_wall_s"]))

    print("\n  THE TWO DEPTHS THE HANDOFF ASKED FOR")
    for dd in (2, 3):
        r = next(x for x in d["rows"] if x["strassen_depth"] == dd)
        print("\n    Strassen depth %d (leaf %d)" % (dd, r["leaf"]))
        print("      ours     %8.5f B   proj worst wall %6.1f s   margin %5.2fx "
              "(%.2fx contended)"
              % (r["ours_B"], r["ours_proj_worst_wall_s"], r["ours_margin"],
                 r["ours_margin_contended"]))
        print("      AUGMENT  %8.5f B   proj worst wall %6.1f s   margin %5.2fx "
              "(%.2fx contended)"
              % (r["augment_B"], r["augment_proj_worst_wall_s"],
                 r["augment_margin"], r["augment_margin_contended"]))
        print("        of which  tower %.3f B  |  new rank-R %.3f B  |  rest %.3f B"
              % (r["component_tower_B"], r["component_new_rank_r_B"],
                 r["component_rest_B"]))
        if r["augment_margin"] < 1.0:
            verdict = "EVERY MLP TIMES OUT -- not submittable at any score"
        elif r["augment_margin_contended"] >= MARGIN_RULE:
            verdict = "clears the >=2.0x nominee margin rule"
        else:
            verdict = "under the >=2.0x nominee margin rule"
        print("      verdict: %s" % verdict)

    print("\n  SCORE ARITHMETIC (raw from D1's 24 nets, 1.912e-8, deflated 6% for")
    print("  the graded-100 path; nominee A graded at 2.031e-8 adjusted)")
    raw = 1.912e-8 / 1.06
    for dd in (2, 3):
        r = next(x for x in d["rows"] if x["strassen_depth"] == dd)
        print("      depth %d: %.3e raw x %.3f B = %.3e  vs nominee A 2.031e-08"
              "  (%.2fx WORSE)"
              % (dd, raw, r["augment_B"], raw * r["augment_B"],
                 raw * r["augment_B"] / 2.031e-8))
    print("\n      break-even B against nominee A: %.3f" % (2.031e-8 / raw))
    print("      strategist's expected B for #15-22: 0.59 - 0.86")

    print("\n  LARGEST ARRAY (not the blocker any more, and this is the point)")
    cols = (RANK_SCHEDULE["AUGMENT"]["birth_first"]
            + RANK_SCHEDULE["AUGMENT"]["birth_steady"] * (DEPTH - 2))
    print("      widest factor  (%d, %d) f32 = %.0f MB   x3 factors = %.2f GiB"
          % (WIDTH, cols * WIDTH, WIDTH * cols * WIDTH * 4 / 1e6,
             3 * WIDTH * cols * WIDTH * 4 / 2 ** 30))
    print("      against hard rule 1's 4 GiB single-array cap: LEGAL.")
    print("      the 4.000 GiB n^3 object is gone, exactly as proven. It was")
    print("      never the binding constraint.")

    print("\n  OPS are NOT the constraint either: the tower is issued batched, so")
    print("  op count is flat in rank. Our L=3 is 12,340 against a 24,000 gate.")

    print("\n  CAVEATS")
    print("    - wall is PROJECTED from the two-point graded intensity law, which")
    print("      validated +3.1% out of sample. At a FIXED leaf, intensity is")
    print("      fixed and wall scales linearly in FLOPs -- the mildest use of")
    print("      that model.")
    print("    - the transfer assumes our spelling of the two new rank-R")
    print("      contractions is no better than upstream's. They are the same")
    print("      class of operation as contract_W, so there is no known")
    print("      structural edge; if one is found it moves %.2fx of the %.2fx."
          % (d["rows"][3]["component_new_rank_r_B"] / d["rows"][3]["ours_B"],
             d["rows"][3]["ratio"]))
    print("    - a merge exists (legs 2/3/4 share fac2 and fac3 exactly) that")
    print("      would take births 8n -> 6n. It is NOT applied above. It would")
    print("      move the tower term to %.2fx, which is still far above the"
          % (_block_layers(3, 6, DEPTH) / _block_layers(2, 2, DEPTH)))
    print("      break-even.")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        d["provenance"] = {
            "handoff": "docs/STRATEGIST_RESPONSE_2026-09-09.md section 3 item 1",
            "our_anchor": ("graded ledger, reproduced +0.003% by "
                           "p2_hybrid_strassen_price"),
            "upstream_meter": ("torch NamedFlopCounter via fit_flop_polys -- "
                               "RATIOS ONLY"),
            "wall": ("PROJECTED. docs/VALIDATION.md forbids quoting a local "
                     "absolute wall time."),
            "tier": "no seed block, no gate, no shipped file touched",
        }
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(d, fh, indent=1)
        print("\n  wrote %s" % os.path.relpath(a.out, _ROOT))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--measure", action="store_true",
                    help="re-measure rank schedule + upstream ratios (needs torch)")
    ap.add_argument("--upstream",
                    default=r"C:\Users\samsu\Documents\mlp_cumulant_propagation\src")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.measure:
        a.out = a.out or os.path.join(OUT_DIR, "augment_rank_schedule.json")
        return cmd_measure(a)
    a.out = a.out or OUT
    return cmd_price(a)


if __name__ == "__main__":
    raise SystemExit(main())
