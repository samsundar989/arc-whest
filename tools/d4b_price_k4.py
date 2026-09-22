"""Does K4 (or the AUGMENT term set) pay once cost is counted?

D4 measured the ACCURACY side only and found two large levers at L=8:

    SIMPLE -> AUGMENT (factored)   2.51-2.74x less MSE
    K3 -> K4 (factored)            3.41-6.50x less MSE, GROWING with width

This competition does not score MSE. It scores **MSE x FLOPs**, so a 3x
accuracy gain that costs 4x the FLOPs is a loss. This prices the same arms with
ARC's own FLOP counter (`mlp_kprop.flop_utils.NamedFlopCounter`, a torch
FlopCounterMode) and combines the two sides into the only figure that matters:

    score ratio  =  (MSE_arm / MSE_base) x (FLOPs_arm / FLOPs_base)

< 1 means the arm would win at that shape. FLOP counts are device-independent,
so this runs on CPU and does not contend with anything on the GPU.

**What this can and cannot conclude.** Torch's counter uses its own conventions
(2mnk for a matmul) and counts a different program from the one flopscope bills
for our chain, so an absolute FLOP number here is not comparable to our 0.2187
multiplier. Only the RATIOS between arms measured by the same counter are
meaningful, and only at the shapes measured — both sides scale with n and L, and
the competition sits at (1024, 16), far outside what fits here. This is a
screening result that says whether the lane is worth a real implementation, not
a prediction of our score.

    python tools/d4b_price_k4.py --shapes 128x8 256x8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from mlp_kprop.mlp import MLP
from mlp_kprop.kprop_harmonic import mlp_kprop, SIMPLE, AUGMENT
from mlp_kprop.flop_utils import NamedFlopCounter

ARMS = [
    ("factored SIMPLE K3", SIMPLE, True, 3),
    ("factored AUGMENT K3", AUGMENT, True, 3),
    ("unfactored AUGMENT K3", AUGMENT, False, 3),
    ("factored K4", SIMPLE, True, 4),
]

#: MSE measured by tools/d4_price_diagrams.py, 8 trials, 20M-sample MC truth.
#: experiments/2026-09-22_d4/d4_depth8.json
D4_MSE = {
    (128, 8): {"factored SIMPLE K3": 5.9680e-06, "factored AUGMENT K3": 2.1746e-06,
               "unfactored AUGMENT K3": 1.8333e-06, "factored K4": 6.3844e-07},
    (256, 8): {"factored SIMPLE K3": 6.9433e-07, "factored AUGMENT K3": 2.7677e-07,
               "unfactored AUGMENT K3": 2.5572e-07, "factored K4": 4.2569e-08},
}


def count_flops(n: int, L: int, kind, factor: bool, k_max: int) -> int:
    torch.manual_seed(0)
    mlp = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=L)
    K_in = {1: torch.zeros(n), 2: torch.eye(n)}
    with NamedFlopCounter() as c:
        mlp_kprop(mlp, K_in, k_max=k_max, kind=kind, factor=factor)
        total = c.mode.get_total_flops()
    return int(total)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shapes", nargs="+", default=["128x8", "256x8"])
    ap.add_argument("--base", default="factored AUGMENT K3",
                    help="the arm our chain is closest to")
    ap.add_argument("--json", default="experiments/2026-09-22_d4/d4b_k4_cost.json")
    a = ap.parse_args(argv)
    torch.set_grad_enabled(False)

    out = []
    for s in a.shapes:
        n, L = (int(v) for v in s.lower().split("x"))
        mse = D4_MSE.get((n, L))
        if mse is None:
            print(f"no D4 MSE for {n}x{L}; run tools/d4_price_diagrams.py first")
            continue
        print(f"\n  n={n} L={L}   base arm = {a.base}")
        print(f"    {'arm':<24} {'FLOPs':>16} {'cost x':>9} {'MSE x':>9} {'SCORE x':>9}")
        print("    " + "-" * 70)
        flops = {}
        for name, kind, factor, k_max in ARMS:
            try:
                flops[name] = count_flops(n, L, kind, factor, k_max)
            except Exception as exc:
                print(f"    {name:<24} FAILED {type(exc).__name__}: {str(exc)[:40]}")
        base_f, base_m = flops.get(a.base), mse.get(a.base)
        row = {"n": n, "L": L, "base": a.base, "arms": {}}
        for name, *_ in ARMS:
            if name not in flops:
                continue
            cx = flops[name] / base_f
            mx = mse[name] / base_m
            sx = cx * mx
            row["arms"][name] = {"flops": flops[name], "cost_ratio": cx,
                                 "mse_ratio": mx, "score_ratio": sx}
            flag = "  <-- WINS" if sx < 1.0 else ""
            print(f"    {name:<24} {flops[name]:>16,} {cx:>9.3f} {mx:>9.3f} "
                  f"{sx:>9.3f}{flag}")
        out.append(row)

    print()
    print("  score ratio = cost ratio x MSE ratio, against the arm our chain is")
    print("  closest to. Below 1.0 the arm would win AT THIS SHAPE. Ratios only:")
    print("  torch's counter is not flopscope and these shapes are not (1024, 16).")

    p = Path(a.json)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"\n  wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
