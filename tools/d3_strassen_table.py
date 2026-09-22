"""Per-shape: what does routing a product through `_mm` actually cost?

D3 item 1 says "Strassen for the plain-`@` join products". Applied to every
eligible 2D product in C10 it measures **-2.60% FLOPs and +30.3% op calls**
(23,421 -> 30,528), which pushes the projected server residual to 0.339 s
against a 0.4 s cliff and fails PLAN §5's 24,500-op ceiling. Strassen buys
multiplications with additions, and every addition is an op call.

So the transform has to be applied per shape, not in bulk. This measures both
sides of the trade for one shape at a time under the REAL flopscope meter:

    python tools/d3_strassen_table.py

For each shape it reports plain-`@` FLOPs and ops, `_mm` FLOPs and ops, and the
decisive ratio: **FLOPs saved per extra op call**. An op call is worth about
11 µs of the 0.4 s residual budget, i.e. 2.8e-5 of the budget; a FLOP is worth
1/2^41 of the cost budget. At the margin, spending one op to save F FLOPs is
worth it only when the score gain beats the residual risk, and the table makes
that comparison possible instead of assumed.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "local_sim")):
    if p not in sys.path:
        sys.path.append(p)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import flopscope as flops
import flopscope.numpy as fnp

import lifecycle_eval as le  # noqa: F401  (imported for its real-package guard)

#: The shapes C10 actually issues as plain `@`, from the D3 probe run.
SHAPES = [
    ("J.T @ Om", (2048, 1024), (1024, 256)),
    ("Qn.T @ J", (256, 1024), (1024, 2048)),
    ("(FJ*dJ) @ FJ.T", (256, 2048), (2048, 256)),
    ("Q.T @ Om", (256, 1024), (1024, 256)),
    ("Q @ (Sg@(Q.T@Om))", (1024, 256), (256, 256)),
    ("Sg @ (Q.T@Om)", (256, 256), (256, 256)),
    ("T @ tw.Sg", (256, 256), (256, 256)),
    ("(T@Sg) @ T.T", (256, 256), (256, 256)),
    ("R @ Om  (p2, r=1024)", (1024, 1024), (1024, 1024)),
    ("D @ Om  (p3, r=32)", (1024, 1024), (1024, 32)),
    ("Q.T @ R (p2)", (1024, 1024), (1024, 1024)),
]

BUDGET = 2 ** 55  # diagnostic only: never a candidate budget


def measure(fn, A, B):
    ctx = flops.BudgetContext(flop_budget=BUDGET, quiet=True)
    with ctx:
        fn(A, B)
    try:
        ns = (ctx.summary_dict(by_namespace=True).get("by_namespace") or {})
        ops = sum(float(v.get("calls", 0.0)) for v in ns.values())
    except Exception:
        ops = float("nan")
    return float(ctx.flops_used), ops


def main() -> int:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_c10", ROOT / "estimator_k3_v43_complete_c10.py")
    c10 = importlib.util.module_from_spec(spec)
    sys.modules["_c10"] = c10
    spec.loader.exec_module(c10)

    print(f"_STRASSEN_DEPTH={c10._STRASSEN_DEPTH} "
          f"_STRASSEN_MIN_BLOCK={c10._STRASSEN_MIN_BLOCK}")
    print()
    hdr = (f"{'product':<24} {'depth':>5} {'@ FLOPs':>15} {'@ ops':>6} "
           f"{'_mm FLOPs':>15} {'_mm ops':>7} {'dFLOP':>14} {'dops':>6} "
           f"{'FLOP/op':>11}")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for name, sa, sb in SHAPES:
        A = fnp.array(np.random.default_rng(0).standard_normal(sa).astype(np.float32))
        B = fnp.array(np.random.default_rng(1).standard_normal(sb).astype(np.float32))
        d = c10._depth_for(sa[0], sa[1], sb[1])
        f_plain, o_plain = measure(lambda a, b: a @ b, A, B)
        f_mm, o_mm = measure(c10._mm, A, B)
        df, do = f_mm - f_plain, o_mm - o_plain
        per = (-df / do) if do else float("inf")
        rows.append({"product": name, "shape_a": sa, "shape_b": sb, "depth": d,
                     "plain_flops": f_plain, "plain_ops": o_plain,
                     "mm_flops": f_mm, "mm_ops": o_mm,
                     "d_flops": df, "d_ops": do, "flops_saved_per_extra_op": per})
        print(f"{name:<24} {d:>5} {f_plain:>15,.0f} {o_plain:>6,.0f} "
              f"{f_mm:>15,.0f} {o_mm:>7,.0f} {df:>+14,.0f} {do:>+6,.0f} "
              f"{per:>11,.0f}")

    print()
    print("  Reading this table: dFLOP negative = Strassen saves cost; dops positive")
    print("  = it spends residual. FLOP/op is FLOPs saved per extra op call, so")
    print("  higher is better and anything non-positive is a pure loss.")
    print()
    print("  Scale: the whole cost budget is 2^41 = 2,199,023,255,552 FLOPs, and the")
    print("  whole residual budget is 0.4 s ~ 36,000 op calls at 11.09 us/op. One op")
    print("  call is therefore 2.8e-5 of the residual budget; one FLOP is 4.5e-13 of")
    print("  the cost budget. Trading an op for fewer than ~61,000 FLOPs spends more")
    print("  of the residual budget than it recovers of the cost budget.")

    out = ROOT / "experiments" / "2026-09-22_d3" / "strassen_table.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
