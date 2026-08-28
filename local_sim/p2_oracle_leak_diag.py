"""ORACLE-2 diagnostic: is `oracle_offdiag`'s failure a finding or a defect?

`oracle_offdiag:n` drifts `3.96e-6` -- 4.5x WORSE than `dsproj:n`, which keeps the
same diagonals exactly and throws the remainder away entirely. Adding information
made it worse, which has exactly one plausible mechanism:

    the rank-`n` truncation `Rhat` of the all-distinct remainder `R` does NOT have
    zero diagonal slices, so `FT_diag (+) Rhat` no longer carries the exact
    diagonals -- it carries exact-plus-leakage, and the flow reads only diagonals.

This walks one network layer by layer and measures the leakage directly, against
the true diagonal it is corrupting. No ground truth, no seed block, no gate.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = r"c:\Users\samsu\Documents\arc-whest"
for _p in (os.path.join(_ROOT, "local_sim"), _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

import json  # noqa: E402

import torch  # noqa: E402

import p2_kprop_bridge as bridge  # noqa: E402
import p2_pb_rank as pb  # noqa: E402
from mint_provenance import canonical_mint  # noqa: E402

bridge._src()
from mlp_kprop.factor_k3 import FactoredTensor  # noqa: E402
from mlp_kprop.kprop_harmonic import (  # noqa: E402
    SIMPLE, coerce_input, linear_kprop, metric_needed, nonlin_kprop,
)
from mlp_kprop.wick import WICK_COEF_D  # noqa: E402

WIDTH, DEPTH, SEED = 1024, 16, 1_000_201
ALS = 8


def nrm(x):
    return float(torch.linalg.vector_norm(x))


def main():
    w, _e, _s = canonical_mint(WIDTH, DEPTH, SEED)
    dtype, dev = torch.float32, "cuda"
    mlp = bridge.build_mlp(w, device=dev, dtype=dtype)
    K = coerce_input({1: torch.zeros(WIDTH, device=dev, dtype=dtype),
                      2: torch.eye(WIDTH, device=dev, dtype=dtype)},
                     k_max=3, kind=SIMPLE)
    nonlin_wick = [WICK_COEF_D[nm] for nm in mlp.nonlin_names]
    skip_metric = not metric_needed(3, SIMPLE)
    rows = []

    for l, W_module in enumerate(mlp.Ws):
        if l >= len(mlp.nonlins):
            break
        metric = mlp.init_scale[l]
        K = linear_kprop(K, W_module.weight, k_max=3, set_metric=metric,
                         bias=W_module.bias)
        K = nonlin_kprop(K, nonlin_wick_coef=nonlin_wick[l], k_max=3,
                         kind=SIMPLE, use_pK=True, factor=True)
        FT = K[3]
        r_in = int(FT._factors[0].shape[1])

        # the true diagonal slices the flow actually reads
        t3, t21 = nrm(FT.get_dslice((3,))), nrm(FT.get_dslice((2, 1)))

        # ORACLE-2's two pieces
        diag, rem = pb._split_diagonal(FT)
        wr = pb._column_weight(rem, "frob")
        keep = torch.topk(wr, WIDTH, sorted=False).indices
        warm = tuple(F.index_select(1, keep) for F in rem)
        rhat = pb._cp_als_truncate(rem, WIDTH, iters=ALS, warm=warm)

        fit_rem = pb._cp_fit_residual(rem, rhat)
        rhatFT = FactoredTensor(n=WIDTH, d=3, factors=rhat, device=FT.device,
                                dtype=FT.dtype)
        leak3, leak21 = nrm(rhatFT.get_dslice((3,))), nrm(rhatFT.get_dslice((2, 1)))

        # ORACLE-1's fit, for comparison
        w1 = pb._column_weight(FT._factors, "frob")
        k1 = torch.topk(w1, WIDTH, sorted=False).indices
        warm1 = tuple(F.index_select(1, k1) for F in FT._factors)
        o1 = pb._cp_als_truncate(FT._factors, WIDTH, iters=ALS, warm=warm1)
        fit_o1 = pb._cp_fit_residual(FT._factors, o1)

        row = {"layer": l, "rank_in": r_in,
               "fit_oracle1": fit_o1, "fit_remainder": fit_rem,
               "leak3_over_true3": leak3 / max(t3, 1e-300),
               "leak21_over_true21": leak21 / max(t21, 1e-300)}
        rows.append(row)
        print(f"  L{l:<3} r={r_in:<6} fit(O1)={fit_o1:.4f} fit(rem)={fit_rem:.4f} "
              f"leak(3,)/true={row['leak3_over_true3']:.3f} "
              f"leak(2,1)/true={row['leak21_over_true21']:.3f}", flush=True)

        # continue the flow UNCAPPED so later layers see the true state
        K[3] = FT

    out = os.path.join(_ROOT, "experiments", "2026-08-28_pb_oracle",
                       "oracle2_leakage_diag.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"check": "ORACLE-2 diagonal-leakage diagnostic",
                   "width": WIDTH, "depth": DEPTH, "seed": SEED,
                   "dtype": "f32", "als_iters": ALS, "rows": rows,
                   "note": "uncapped flow; each layer's split measured in place. "
                           "No gate, no seed block, no truth."}, fh, indent=2)
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
