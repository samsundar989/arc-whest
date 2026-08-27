"""P-B instrument: rank-capped factored K3, its REAL cost, and its rank trajectory.

Research only. Never imported by `estimator.py` (upstream needs torch).

## Why this module exists

`experiments/LOG.md` (P-A result, 2026-08-27) records the load-bearing INFERENCE
of the whole P-B plan: that `get_dslice` and `contract_W` -- 95.35% of factored-K3
cost -- scale linearly with the factored-`kappa_3` rank, so capping the rank buys
a proportional saving. That has never been measured in this tree. Neither has the
rank trajectory itself, nor the price of the selector that a cap requires (the
Skeptic's D9: `FactoredTensor.__add__` grows the rank by concatenation, so a cap
needs new machinery whose cost is priced nowhere, and an SVD-based one is
`O(n^2 r)` -- the same order as the saving it claims).

This module measures all three, with **no ground truth and no seed block**: it is
a cost and structure probe, not an accuracy experiment. Accuracy is P-B proper
and has its own contract.

## The upstream checkout is not edited

Capping is implemented by re-driving upstream's own layer loop
(`kprop_harmonic.mlp_kprop`, lines 505-606 at rev 93d091a4) from here, calling
upstream `linear_kprop` / `nonlin_kprop` unmodified and inserting the cap between
layers. `--selftest` asserts this re-drive reproduces `mlp_kprop` bit-for-bit
with capping disabled, which is what licenses reading a capped run as "upstream
plus a cap" rather than "our re-derivation of upstream".

## The two selectors

Both act on a `FactoredTensor` of degree 3 whose three factors are `(n, r)`, and
both are priced by the same runtime counter as everything else -- never modelled.

- `normsort` -- per-column weight `||A_j|| ||B_j|| ||C_j||`, keep the top `r*`.
  `O(n r)`. This is the cheap baseline the Skeptic allowed.
- `sketch` -- the O4 arm. Deterministically keep the top `k = r*/2` columns by
  the same weight (exact, no bias), then **recombine** the remaining `r - k`
  into `r*/2` slots by a seeded random map `S` applied to all three factors.
  A shared map is only unbiased if `E[S] = 0` and `E[S^3] != 0`, because
  `E[sum_m (SA)_m (x) (SB)_m (x) (SC)_m]` picks out exactly the `j = k = l`
  diagonal through the third moment; a Rademacher map has `E[S^3] = E[S] = 0`
  and is therefore identically useless here, which is the trap this arm exists
  to avoid. We use the skewed three-point map below, whose third moment is 1 by
  construction, and divide by `r_slots` so the estimator is unbiased.

The random map is seeded from the network seed, so a capped run is deterministic
and reproducible, as the grader contract requires of anything downstream.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np  # noqa: E402

import p2_kprop_bridge as bridge  # noqa: E402

WIDTH, DEPTH = 1024, 16
BUDGET = 2 ** 41


# --------------------------------------------------------------------------
# the seeded, unbiased rank-mixing map
# --------------------------------------------------------------------------

def _mix_map(r_in: int, r_out: int, seed: int, device, dtype):
    """`S` of shape `(r_in, r_out)` with `E[S_jm] = 0` and `E[S_jm^3] = 1`.

    Why not Rademacher: a shared map on all three legs reproduces the original
    tensor only through `E[S_jm S_km S_lm]`, which is `E[S^3]` on the diagonal
    and zero off it. Rademacher has `S^3 = S`, hence `E[S^3] = 0`, and the whole
    sketch collapses to zero in expectation.

    Three-point law, deliberately ASYMMETRIC (a symmetric one has `E[S^3] = 0`
    for the same reason). Take `S in {c, -c/2}` with `p(c) = 1/3`:
        `E[S]   = c/3 - (2/3)(c/2) = 0`                            (unbiased)
        `E[S^3] = c^3/3 - (2/3)(c^3/8) = 7 c^3 / 24`  ->  `c = (24/7)^(1/3)`.
    """
    import torch

    g = torch.Generator(device="cpu").manual_seed(int(seed) & 0x7FFFFFFF)
    u = torch.rand((r_in, r_out), generator=g, dtype=torch.float64)
    c = (24.0 / 7.0) ** (1.0 / 3.0)
    S = torch.where(u < (1.0 / 3.0), c, -c / 2.0)
    return S.to(device=device, dtype=dtype)


def _column_weight(factors, score: str = "frob"):
    """Per-column relevance. Three scores, and the choice is not cosmetic.

    `frob` -- `||A_j|| ||B_j|| ||C_j||`, the rank-one term's own Frobenius norm.
      The obvious score, and the one the Skeptic's `~3nr` pricing referred to.

    But the Frobenius norm of the term is NOT what the flow consumes. In SIMPLE
    mode `kappa_3` reaches the scored mean only through its DIAGONAL slices --
    `get_dslice((3,))` and `get_dslice((2, 1))` (`factor_k3.py:173-179`, consumed
    by `eval_part` when building `pK_slices[(1,)]`). The full off-diagonal tensor
    is never contracted into the mean. A term whose three factors are individually
    large but mutually unaligned has a big Frobenius norm and contributes almost
    nothing to either diagonal slice, so `frob` systematically keeps the wrong
    columns. Hence:

    `d3`  -- `||A_j o B_j o C_j||`, the column's exact contribution to the `(3,)`
      slice (`o` is elementwise).
    `d21` -- `mean over the 3 pairings of ||A_j o B_j|| ||C_j||`, the column's
      exact Frobenius contribution to the `(2, 1)` slice.

    All three are `O(n r)` and are priced by the same counter as everything else.
    """
    import torch

    A, B, C = factors
    if score == "frob":
        w = None
        for F in factors:
            nrm = torch.linalg.vector_norm(F, dim=0)
            w = nrm if w is None else w * nrm
        return w
    if score == "d3":
        return torch.linalg.vector_norm(A * B * C, dim=0)
    if score == "d21":
        def pair(X, Y, Z):
            return (torch.linalg.vector_norm(X * Y, dim=0)
                    * torch.linalg.vector_norm(Z, dim=0))
        return (pair(A, B, C) + pair(B, C, A) + pair(C, A, B)) / 3.0
    raise ValueError(f"unknown score {score!r}")


def cap_factored(FT, r_star: int, *, mode: str, seed: int, score: str = "frob"):
    """Return a degree-3 `FactoredTensor` of rank <= `r_star`, priced in place."""
    import torch
    from mlp_kprop.factor_k3 import FactoredTensor
    from mlp_kprop.flop_utils import flop_name

    factors = FT._factors
    r = factors[0].shape[1]
    if mode == "none" or r <= r_star:
        return FT

    with flop_name("rankcap"):
        w = _column_weight(factors, score)
        if mode == "normsort":
            keep = torch.topk(w, r_star, sorted=False).indices
            new = tuple(F.index_select(1, keep) for F in factors)
        elif mode == "sketch":
            k = r_star // 2
            slots = r_star - k
            keep = torch.topk(w, k, sorted=False).indices
            drop = torch.ones(r, dtype=torch.bool, device=w.device)
            drop[keep] = False
            drop = drop.nonzero(as_tuple=True)[0]
            S = _mix_map(int(drop.numel()), slots, seed * 1_000_003 + r,
                         factors[0].device, factors[0].dtype)
            scale = float(slots) ** (-1.0 / 3.0)
            new = tuple(
                torch.cat([F.index_select(1, keep),
                           (F.index_select(1, drop) @ S) * scale], dim=1)
                for F in factors
            )
        elif mode == "dsproj":
            # Not a selection at all: re-express kappa_3 by its own DIAGONAL
            # content. `get_repeated()` returns the (3,) and (2, 1) slices --
            # exactly the slices the flow consumes into the scored mean -- and
            # `from_dstensor` rebuilds a FactoredTensor carrying them at rank
            # exactly `n`, discarding only the all-distinct off-diagonal block.
            # So this is EXACT on everything kappa_3 contributes to the mean at
            # the current layer, and approximates only kappa_3's influence on
            # the NEXT layer's kappa_3. Column selection, by contrast, is
            # approximate on both. `r_star` is ignored: the rank is `n` by
            # construction.
            from mlp_kprop.factor_k3 import FactoredTensor as _FT
            return _FT.from_dstensor(FT.get_repeated())
        else:
            raise ValueError(f"unknown cap mode {mode!r}")

    # The repeated cache describes the OLD factor set; it is invalid after a cap
    # and must be rebuilt lazily rather than carried, or every downstream slice
    # silently reports the uncapped tensor.
    return FactoredTensor(n=FT.n, d=FT.d, factors=new,
                          device=FT.device, dtype=FT.dtype)


# --------------------------------------------------------------------------
# the re-driven layer loop
# --------------------------------------------------------------------------

def run_kprop_capped(mlp, K_in, *, k_max=3, factor=True, use_avg_metric=True,
                     up_to_layer=None, output_d_max=1, cap="none", r_star=0,
                     seed=0, score="frob", trace=None):
    """`mlp_kprop(output_all=True)` with a cap applied to `K[3]` after each layer.

    Mirrors `kprop_harmonic.mlp_kprop` rev 93d091a4 exactly; `--selftest` proves it.
    """
    from mlp_kprop.kprop_harmonic import (
        SIMPLE, clone_tower, coerce_input, linear_kprop, metric_needed,
        nonlin_kprop,
    )
    from mlp_kprop.wick import WICK_COEF_D

    kind = SIMPLE
    nonlin_wick = [WICK_COEF_D[nm] for nm in mlp.nonlin_names]
    init_scale = mlp.init_scale
    skip_metric = not metric_needed(k_max, kind)

    K = coerce_input(K_in, k_max=k_max, kind=kind)
    out = {}
    for l, W_module in enumerate(mlp.Ws):
        W = W_module.weight
        if up_to_layer == f"pre{l}" or (l == len(mlp.nonlins) and up_to_layer is None):
            K = linear_kprop(K, W, k_max=k_max, d_max=output_d_max,
                             set_metric=1.0 if skip_metric else None,
                             bias=W_module.bias)
            out[f"pre{l}"] = clone_tower(K, d_max=output_d_max)
            break
        if l < len(mlp.nonlins):
            metric = init_scale[l] if use_avg_metric else (1.0 if skip_metric else None)
            K = linear_kprop(K, W, k_max=k_max, set_metric=metric, bias=W_module.bias)
            out[f"pre{l}"] = clone_tower(K, d_max=output_d_max)
            K = nonlin_kprop(K, nonlin_wick_coef=nonlin_wick[l], k_max=k_max,
                             kind=kind, use_pK=True, factor=factor)
            if trace is not None and 3 in K and hasattr(K[3], "_factors"):
                trace.append({"layer": l, "rank_out": int(K[3]._factors[0].shape[1])})
            if cap != "none" and 3 in K and hasattr(K[3], "_factors"):
                K[3] = cap_factored(K[3], r_star, mode=cap, seed=seed + l,
                                    score=score)
                if trace is not None:
                    trace[-1]["rank_capped"] = int(K[3]._factors[0].shape[1])
            out[f"act{l}"] = clone_tower(K, d_max=output_d_max)
            if up_to_layer == f"act{l}":
                break
    return out


def predict(weights, *, cap="none", r_star=0, seed=0, score="frob", dtype=None,
            device="cuda", count_flops=False, trace=None):
    """`(depth, width)` predicted post-ReLU means, plus cost if asked."""
    import torch
    from mlp_kprop.flop_utils import NamedFlopCounter

    dtype = dtype or torch.float32
    depth, width = len(weights), weights[0].shape[0]
    mlp = bridge.build_mlp(weights, device=device, dtype=dtype)
    K_in = {1: torch.zeros(width, device=device, dtype=dtype),
            2: torch.eye(width, device=device, dtype=dtype)}
    kw = dict(k_max=3, factor=True, use_avg_metric=True,
              up_to_layer=f"act{depth - 1}", output_d_max=1,
              cap=cap, r_star=r_star, seed=seed, score=score, trace=trace)

    flops = None
    if count_flops:
        with NamedFlopCounter() as ctr:
            out = run_kprop_capped(mlp, K_in, **kw)
        flops = ctr.flop_dict()
    else:
        out = run_kprop_capped(mlp, K_in, **kw)
    means = np.stack([bridge._degree1(out[f"act{l}"][1]) for l in range(depth)], axis=0)
    return means, flops


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_selftest(a) -> int:
    """The re-driven loop IS `mlp_kprop`, with capping off. Bit-for-bit."""
    import torch
    from mlp_kprop.kprop_harmonic import SIMPLE, mlp_kprop

    n, depth = a.width, a.depth
    print(f"  self-test: re-driven loop vs upstream mlp_kprop at {n} x {depth}, "
          f"float64, cap=none")
    from mint_provenance import canonical_mint
    w, _e, _s = canonical_mint(n, depth, a.seeds[0])
    dtype = torch.float64
    mlp = bridge.build_mlp(w, device=a.device, dtype=dtype)
    K_in = {1: torch.zeros(n, device=a.device, dtype=dtype),
            2: torch.eye(n, device=a.device, dtype=dtype)}

    ref = mlp_kprop(mlp, K_in, k_max=3, output_all=True, kind=SIMPLE,
                    use_avg_metric=True, factor=True,
                    up_to_layer=f"act{depth - 1}", output_d_max=1)
    ours = run_kprop_capped(mlp, K_in, up_to_layer=f"act{depth - 1}", cap="none")

    worst = 0.0
    for l in range(depth):
        x = bridge._degree1(ref[f"act{l}"][1])
        y = bridge._degree1(ours[f"act{l}"][1])
        worst = max(worst, float(np.abs(x - y).max()))
    exact = worst == 0.0
    print(f"    worst |ref - ours| over {depth} layers: {worst:.3e}   "
          f"{'BIT-IDENTICAL' if exact else 'DIFFERS'}")
    if a.out:
        bridge._dump(a.out, {"check": "re-driven loop vs mlp_kprop", "width": n,
                             "depth": depth, "worst_abs": worst,
                             "bit_identical": exact})
    return 0 if exact else 1


def cmd_trace(a) -> int:
    """The rank trajectory, uncapped. Answers 'where does 15n come from?'"""
    import torch
    from mint_provenance import canonical_mint

    print(f"  uncapped rank trajectory at {a.width} x {a.depth}, "
          f"dtype={a.dtype}, device={a.device}")
    w, _e, _s = canonical_mint(a.width, a.depth, a.seeds[0])
    dtype = torch.float64 if a.dtype == "f64" else torch.float32
    trace: list = []
    t0 = time.perf_counter()
    _means, _f = predict(w, cap="none", dtype=dtype, device=a.device, trace=trace)
    el = time.perf_counter() - t0
    n = a.width
    print(f"    layer   rank_out   rank/n")
    for row in trace:
        print(f"    {row['layer']:>5}   {row['rank_out']:>8}   "
              f"{row['rank_out'] / n:>6.2f}")
    print(f"\n    final rank {trace[-1]['rank_out']} = "
          f"{trace[-1]['rank_out'] / n:.2f} n     elapsed {el:.1f}s")
    if a.out:
        bridge._dump(a.out, {"check": "uncapped rank trajectory", "width": a.width,
                             "depth": a.depth, "dtype": a.dtype, "trace": trace,
                             "seconds": el})
    return 0


def cmd_cost(a) -> int:
    """Priced cells: the real counted cost of each (selector, r*) pair."""
    import torch
    from mint_provenance import canonical_mint

    dtype = torch.float64 if a.dtype == "f64" else torch.float32
    w, _e, _s = canonical_mint(a.width, a.depth, a.seeds[0])
    cells = []
    for spec in a.cells:
        if spec == "none":
            mode, r_star = "none", 0
        else:
            parts = spec.split(":")
            mode, rs = parts[0], parts[1]
            score = parts[2] if len(parts) > 2 else "frob"
            r_star = int(eval(rs, {"n": a.width}))  # noqa: S307 -- "n//8" style specs
        trace: list = []
        torch.cuda.reset_peak_memory_stats() if a.device == "cuda" else None
        t0 = time.perf_counter()
        means, flops = predict(w, cap=mode, r_star=r_star, seed=a.seeds[0],
                               score=score, dtype=dtype, device=a.device,
                               count_flops=True, trace=trace)
        el = time.perf_counter() - t0
        peak = (torch.cuda.max_memory_allocated() / 2 ** 30) if a.device == "cuda" else float("nan")
        total = sum(v for k, v in flops.items())
        row = {"cell": spec, "mode": mode, "r_star": r_star, "score": score,
               "total_flops": total, "x_budget": total / BUDGET,
               "flops": {k: int(v) for k, v in sorted(flops.items())},
               "peak_gib": peak, "seconds": el,
               "final_rank": trace[-1].get("rank_capped", trace[-1]["rank_out"]),
               "mean_of_final_layer": float(means[-1].mean())}
        cells.append(row)
        print(f"    {spec:<16} r*={r_star:<6} total {total:>12.4e} = "
              f"{total / BUDGET:>7.4f} B   rankcap {flops.get('rankcap', 0):>10.3e} "
              f"({100 * flops.get('rankcap', 0) / max(total, 1):>5.2f}%)   "
              f"peak {peak:>5.2f} GiB   {el:>6.1f}s")
    if a.out:
        bridge._dump(a.out, {"check": "priced rank-cap cells", "width": a.width,
                             "depth": a.depth, "dtype": a.dtype,
                             "budget": BUDGET, "seed": a.seeds[0], "cells": cells,
                             "caveat": "upstream's torch-side runtime FLOP counter, "
                                       "NOT flopscope 0.12 pricing. Any operating "
                                       "point must be re-priced by flopscope."})
    return 0


def cmd_drift(a) -> int:
    """How far does a capped prediction move from the UNCAPPED one, and what
    does that alone bound the capped MSE to?

    No ground truth, no seed block: this is a relative diagnostic, and it is
    reported as a BRACKET rather than an estimate because drift is not an error
    increment. Writing `e = p - y`, the capped error is `e_full + delta` with
    `delta = p_cap - p_full`, so by the triangle inequality in L2

        | sqrt(MSE_full) - sqrt(drift) |  <=  sqrt(MSE_cap)  <=  sqrt(MSE_full) + sqrt(drift)

    and NOTHING narrower is available without measuring the cross term against a
    real label, which is P-B's job. A small drift therefore proves the cap is
    survivable; it does not prove where in the bracket the answer lands, and the
    bracket is quoted so nobody reads the optimistic end as a result.
    """
    import torch
    from mint_provenance import canonical_mint

    dtype = torch.float64 if a.dtype == "f64" else torch.float32
    mse_full = a.mse_full
    print(f"  drift of capped vs UNCAPPED K3, {a.width} x {a.depth}, dtype={a.dtype}")
    print(f"  anchor MSE_full = {mse_full:.4e} (P-A / P-A-CONFIRM; an ASSUMPTION "
          f"here, measured there)\n")
    rows = []
    for seed in a.seeds:
        w, _e, _s = canonical_mint(a.width, a.depth, seed)
        base, _f = predict(w, cap="none", dtype=dtype, device=a.device)
        print(f"    seed {seed}")
        print(f"      {'cell':<16}{'drift':>12}{'/MSE_full':>11}"
              f"{'MSE_cap lo':>13}{'MSE_cap hi':>13}")
        for spec in a.cells:
            if spec == "none":
                continue
            parts = spec.split(":")
            mode, rs = parts[0], parts[1]
            score = parts[2] if len(parts) > 2 else "frob"
            r_star = int(eval(rs, {"n": a.width}))  # noqa: S307
            p, _f = predict(w, cap=mode, r_star=r_star, seed=seed, score=score,
                            dtype=dtype, device=a.device)
            drift = float(((p[-1] - base[-1]) ** 2).mean())
            lo = max(abs(mse_full ** 0.5 - drift ** 0.5), 0.0) ** 2
            hi = (mse_full ** 0.5 + drift ** 0.5) ** 2
            rows.append({"seed": seed, "cell": spec, "r_star": r_star,
                         "score": score,
                         "drift_final_layer": drift, "ratio_to_mse_full": drift / mse_full,
                         "mse_cap_lo": lo, "mse_cap_hi": hi})
            print(f"      {spec:<16}{drift:>12.3e}{drift / mse_full:>11.3f}"
                  f"{lo:>13.3e}{hi:>13.3e}")
        print()
    if a.out:
        bridge._dump(a.out, {"check": "capped-vs-uncapped drift bracket",
                             "width": a.width, "depth": a.depth, "dtype": a.dtype,
                             "mse_full_anchor": mse_full, "rows": rows,
                             "caveat": "drift is not an error increment; the bracket "
                                       "is the only truth-free statement available."})
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--cost", action="store_true")
    ap.add_argument("--drift", action="store_true")
    ap.add_argument("--mse-full", type=float, default=4.0047e-08,
                    help="the P-A anchor for uncapped K3's true final-layer MSE")
    ap.add_argument("--cells", nargs="*", default=["none"],
                    help="cell specs: 'none', 'normsort:n//8', 'sketch:n//16', ...")
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--depth", type=int, default=DEPTH)
    ap.add_argument("--dtype", choices=["f32", "f64"], default="f32")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", type=int, nargs="*", default=[1_000_201],
                    help="UNREGISTERED probe seeds. This command measures no "
                         "truth and makes no decision.")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    src = bridge._src()
    print("=" * 78)
    print("P-B RANK INSTRUMENT  --  rank-capped factored K3, priced")
    print("=" * 78)
    v = bridge.verify_upstream(src)
    print(f"  upstream rev    {v['rev'][:12]}  "
          f"({'as pinned' if v['rev'] == bridge.UPSTREAM_REV else 'MISMATCH'})")
    print(f"  local patch     {v['changed_lines']} changed lines vs recorded "
          f"{v['recorded_lines']}  -> "
          f"{'IDENTICAL' if v['changed_md5'] == v['recorded_md5'] else 'DIFFERENT'}")
    if not v["pass"]:
        raise SystemExit(
            "REFUSED: the upstream checkout is not rev 93d091a4 plus exactly the "
            "recorded patch. Every number from it would be unattributable.")
    print()

    rc = 0
    if a.selftest:
        rc |= cmd_selftest(a)
    if a.trace:
        rc |= cmd_trace(a)
    if a.cost:
        rc |= cmd_cost(a)
    if a.drift:
        rc |= cmd_drift(a)
    if not (a.selftest or a.trace or a.cost or a.drift):
        print("  nothing to do; pass --selftest, --trace, --cost and/or --drift")
    print("=" * 78)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
