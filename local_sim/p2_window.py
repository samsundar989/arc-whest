"""P2-WINDOW: late-window / layer-age exact K3, screened against full K3 WITHOUT LABELS.

**Research only. Never imported by `estimator.py`** -- upstream needs torch, and
the grader sandbox has no torch.

## What this is, and what it is not

This is an **imitation diagnostic**, not an accuracy experiment. It reads no
ground truth, claims no seed-ledger block, and produces no MSE. What it measures
is **drift from the full factorized-K3 prediction on identical networks**:

    D = sqrt( mean_i ( pred_arm[i] - pred_full[i] )^2 )   over the final layer

Drift is **not** an error increment. The only sound thing to do with it is the
**reverse** triangle bracket on a same-population candidate RMSE,

    |RMSE_full - D|  <=  RMSE_cand  <=  RMSE_full + D

carried with the full-K3 anchor's own interval and an explicit
population-transfer caveat. Every score column this module prints is a
**bracket**, and a bracket that straddles the control decides nothing.

**Corrected 2026-08-31 (WP0B), and this docstring carried the refuted form for
three days after the code was fixed.** It said `max(0, RMSE_full - D)`, which
throws the lower bound away whenever `D > RMSE_full` -- true of every arm in
this module's own panel. `bracket()` below was repaired to `|RMSE_full - D|`
after an independent review found it, and that repair is what made GATE-W's
kill *provable* rather than merely unendorsed. Only the prose was stale; no
number in `experiments/LOG.md` was computed from the wrong form.

## The arms

`--selftest` proves the re-driven loop reproduces `kprop_harmonic.mlp_kprop`
bit-for-bit when the arm is `full`. That is what licenses reading every other arm
as "upstream with one intervention" rather than "our re-derivation of upstream".

| arm | intervention |
|---|---|
| `full` | none -- the control, `k_max=3, factor=True, kind=SIMPLE` |
| `late:m` | `k_max=2` for layers `0..L-m-1`; at the checkpoint inject a **zero** `K[3]` (rank 0) and a **zero** `K[4]` scalar; `k_max=3, factor=True` for the last `m` |
| `age:m` | `k_max=3` throughout, but after each nonlin drop every factor column born more than `m` layers ago |
| `normsort:r` | `k_max=3` throughout with upstream's rank cap at `r` -- the **matched-retained-rank** control the window arms must beat |

### The checkpoint initializers, honestly

`GAME_PLAN.md` Track 2 names three: zero K3 state; a faithful augmented-K2
trace/harmonic state; and selected K3 diagonal queries estimated by a structured
checkpoint. **Only the first is supportable today**, and pretending otherwise
would be the exact failure this repository keeps catching. The second needs
`kind=AUGMENT`, whose factored path is *not* equivalent to the unfactored one
(upstream drops the diagrams it cannot factor, `factor_k3.py:290-299`), and it
has no evidence path -- it is Track 4. The third needs the carrier, which is
Track 3. Both are recorded as unimplemented with the reason, not silently
dropped.

## Cost: two meters, never conflated

- **Upstream's torch-side counter** (`NamedFlopCounter`) bills every arm as it
  actually runs. That is what the P-A/P-B ladders used.
- **flopscope 0.12** cannot bill a torch program. The flopscope price of a
  windowed path comes from the measured per-layer ledger of **P2-WEFF**
  (`experiments/2026-08-28_weff/weff_price_f32_1000301.json`), whose contract and
  result are in `experiments/LOG.md`. That table is a *measurement*, not a model,
  but it prices the **W-EFF spelling of the window**, not this torch arm.

Residual and memory for a windowed candidate are **unmeasured**. The `0.4 s`
cliff is not smooth and an overrun zeroes the row and forces multiplier `1.0`,
so that is a blocking deployment item for any candidate built on this, not a
footnote.

Run (GPU venv):

    .venv-gpu\\Scripts\\python.exe local_sim/p2_window.py --selftest
    .venv-gpu\\Scripts\\python.exe local_sim/p2_window.py --drift --out experiments/<dir>
"""

from __future__ import annotations

import argparse
import json
import math
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

#: Full factorized K3 on 100 organiser networks, `v2-phase2@mini`
#: (`experiments/2026-08-27_pa_confirm/pa_confirm.json`). The GT-noise floor is
#: `7.478e-11`. The CI is the anchor's own and must be propagated, not dropped.
MSE_FULL = 3.9249e-08
MSE_FULL_CI = (3.821e-08, 4.029e-08)
GT_NOISE_FLOOR = 7.478e-11

#: Shipped K2p1, official-package local anchor, and graded v2.7.
K2P1_FLOPS = 137_600_326_656
V27_ADJUSTED = 1.373807110667258e-07

#: MEASURED flopscope-0.12 W-EFF cost of layer `l` of the exact K3 tower, f32,
#: width 1024 -- `experiments/2026-08-28_weff/weff_price_f32_1000301-1000302-1000303.json`,
#: the RE-BINNED run that followed the independent review's C5 correction.
#: A final-`m`-layer window started from a zero K3 checkpoint costs exactly the
#: first `m` entries, because its rank trajectory inside the window is
#: `0, 2n, 4n, ...` by construction.
WEFF_LAYER_FLOPS_F32 = [
    4_310_176_256, 21_508_397_056, 40_851_479_552, 60_194_562_048,
    79_537_644_544, 98_880_727_040, 118_223_809_536, 137_566_892_032,
    156_909_974_528, 176_253_057_024, 195_596_139_520, 214_939_222_016,
    234_282_304_512, 253_625_387_008, 272_968_469_504, 292_311_552_000
]


def weff_window_cost(m: int) -> dict:
    """flopscope-0.12 cost of a final-`m`-layer exact-K3 window, W-EFF spelling.

    Measured per layer, summed here. The K2 prefix column adds the shipped
    K2p1's WHOLE-tower cost while the per-layer figures already carry their own
    `contract_W_basic`, so the total is an OVER-estimate by at most
    `m * 1.95e-3 B`. Conservative in the safe direction, and stated rather than
    quietly absorbed.
    """
    k3 = sum(WEFF_LAYER_FLOPS_F32[:m])
    total = k3 + K2P1_FLOPS
    return {"m": m, "k3_window_flops": k3, "k3_window_budgets": k3 / BUDGET,
            "total_flops": total, "total_budgets": total / BUDGET,
            "multiplier": max(0.1, total / BUDGET),
            "floor_pinned": (total / BUDGET) <= 0.1,
            "overestimate_budgets": m * 1.95e-3,
            "meter": "flopscope 0.12, W-EFF spelling, measured in P2-WEFF",
            "lower_bound": True,
            "note": "P2-WEFF's figure is a lower bound: its un-respelled "
                    "remainder can only add."}



# ==========================================================================
# the preregistered gate -- executable, and derived from frozen constants only
# ==========================================================================


#: two-sided 95% studentised-t critical values, small n. The panel is n = 6.
_T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365,
        9: 2.306, 10: 2.262, 12: 2.201, 16: 2.131, 20: 2.093}


def _t_crit(n: int) -> float:
    if n < 2:
        return float("nan")
    if n in _T95:
        return _T95[n]
    return 1.96 if n > 30 else _T95[min(_T95, key=lambda k: abs(k - n))]

#: The same-day engineering bar: at least `2x` the graded v2.7 adjusted score.
#: Samuel answered "I don't know" when asked to adopt it, so it stands as the
#: handoff's stated DEFAULT and is recorded as not personally committed to.
SHIP_BAR_ADJUSTED = V27_ADJUSTED / 2.0


def d_max_for_m(m: int, bar: float = SHIP_BAR_ADJUSTED) -> float:
    """The largest drift `D` at which a final-`m` window could still clear the bar.

    Derived from three already-frozen numbers and nothing else:

      * `MSE_FULL = 3.9249e-08`, the organiser-confirmed full-K3 anchor;
      * the MEASURED flopscope multiplier of the `m`-window (`weff_window_cost`);
      * the ship bar.

    The triangle inequality gives `RMSE_cand <= RMSE_full + D`, so the best
    score the candidate could possibly have is `(RMSE_full + D)^2 * mult(m)`.
    Requiring that to clear the bar and solving for `D`:

        D_max(m) = sqrt(bar / mult(m)) - sqrt(MSE_FULL)

    **The direction, stated correctly.** `D_hi95 <= D_max` tests the bracket's
    UPPER end, so passing it would prove the candidate clears the bar; failing
    it proves only **undetermined**. The contract's prose asserted the reverse
    ("`D > D_max` says it cannot"), and **that sentence was wrong** -- an
    independent review caught it. The decisive-negative test is the LOWER
    bracket end, and it lives in `provable_score_floor`. Drift is not an error
    increment either way.
    """
    mult = weff_window_cost(m)["multiplier"]
    return math.sqrt(bar / mult) - math.sqrt(MSE_FULL)


def gate_table(ms=(2, 4, 6, 8)) -> list:
    rows = []
    for m in ms:
        c = weff_window_cost(m)
        rows.append({"m": m, "multiplier": c["multiplier"],
                     "total_budgets": c["total_budgets"],
                     "floor_pinned": c["floor_pinned"],
                     "D_max": d_max_for_m(m),
                     "bar_adjusted": SHIP_BAR_ADJUSTED})
    return rows


def paired_diff(a_vals, b_vals) -> dict:
    """Paired difference `b - a` with a studentised-t 95% interval.

    Common random numbers by construction: every arm runs on the same networks.
    Percentile bootstrap and BCa are refused by `VALIDATION.md` and are not
    computed here.
    """
    a = np.asarray(a_vals, dtype=np.float64)
    b = np.asarray(b_vals, dtype=np.float64)
    d = b - a
    n = len(d)
    if n < 2:
        return {"n": n, "mean": float(d.mean()), "se": None,
                "ci95": None, "excludes_zero": None}
    se = float(d.std(ddof=1) / math.sqrt(n))
    # studentised-t, two-sided 95%
    tcrit = _t_crit(n)
    lo, hi = float(d.mean() - tcrit * se), float(d.mean() + tcrit * se)
    return {"n": n, "mean": float(d.mean()), "se": se, "t_crit": tcrit,
            "ci95": [lo, hi], "excludes_zero": bool(lo > 0 or hi < 0),
            "improves": bool(hi < 0)}


# ==========================================================================
# the re-driven loop -- upstream, with exactly one intervention
# ==========================================================================

def _zero_k3_checkpoint(n, device, dtype):
    """The zero third/fourth-cumulant checkpoint, in upstream's own types.

    A `k_max=2` tower carries only `K[1]` and `K[2]` (verified at runtime by
    `--selftest`); a `k_max=3, kind=SIMPLE` step additionally consumes a
    degree-3 `FactoredTensor` and a degree-4 `HTensor` whose core is a scalar
    with `r = 2` (`get_r_x(4, 3, SIMPLE) == 2`). Both are set to zero here,
    which IS the checkpoint: the window starts with no third-cumulant memory.
    """
    import torch
    from mlp_kprop.factor_k3 import FactoredTensor
    from mlp_kprop.harmonic import HTensor

    return (FactoredTensor(n=n, d=3, device=device, dtype=dtype),
            HTensor(core=torch.zeros((), device=device, dtype=dtype), r=2, n=n))


def _drop_old_columns(FT, keep_from: int, births: list[int]):
    """`age:m` -- keep only factor columns born at or after layer `keep_from`.

    Columns are appended in birth order and never reordered by `contract_W`,
    `contract_wick_` or `__add__` (all three are column-preserving maps), so a
    column's birth layer is a function of its index alone. `births` records the
    (index -> birth layer) boundaries as the run goes, so nothing is inferred.
    """
    import torch
    from mlp_kprop.factor_k3 import FactoredTensor
    from mlp_kprop.flop_utils import flop_name

    r = FT._factors[0].shape[1]
    assert len(births) == r, f"birth ledger {len(births)} != rank {r}"
    with flop_name("agecap"):
        keep = torch.as_tensor([i for i, b in enumerate(births) if b >= keep_from],
                               device=FT._factors[0].device, dtype=torch.long)
        if keep.numel() == r:
            return FT, births
        new = FactoredTensor(
            n=FT.n, d=FT.d,
            factors=tuple(f.index_select(1, keep) for f in FT._factors),
            device=FT.device, dtype=FT.dtype)
    return new, [births[int(i)] for i in keep.tolist()]


def run_arm(mlp, K_in, *, arm: str, m: int = 0, r_star: int = 0, seed: int = 0,
            up_to_layer=None, output_d_max=1, trace=None, slice_trace=None):
    """`mlp_kprop(output_all=True)` with exactly one intervention.

    Mirrors `kprop_harmonic.mlp_kprop` rev 93d091a4; `--selftest` proves the
    `full` arm is bit-for-bit identical to it.
    """
    from mlp_kprop.kprop_harmonic import (
        SIMPLE, clone_tower, coerce_input, linear_kprop, metric_needed,
        nonlin_kprop,
    )
    from mlp_kprop.wick import WICK_COEF_D

    kind = SIMPLE
    nonlin_wick = [WICK_COEF_D[nm] for nm in mlp.nonlin_names]
    init_scale = mlp.init_scale
    L = len(mlp.nonlins)
    n = K_in[1].shape[0]
    device, dtype = K_in[1].device, K_in[1].dtype

    if arm.startswith("late:") or arm.startswith("k3pre:"):
        start = L - m
    else:
        start = 0

    # k_max for the PREFIX; the window always runs k_max=3.
    #
    # `late:m`  -- k_max=2 prefix. Changes TWO things at once: it discards the
    #              kappa_3 memory AND hands the window a K2-quality K[1]/K[2].
    # `k3pre:m` -- k_max=3 prefix, same zero K3/K4 checkpoint. Changes ONLY the
    #              kappa_3 memory, so the two effects can be separated. It is a
    #              DIAGNOSTIC, never a candidate: its prefix is the full-tower
    #              W-EFF already killed on price at 1.0723 B.
    prefix_k = 2 if arm.startswith("late:") else 3
    K = coerce_input(K_in, k_max=(prefix_k if start > 0 else 3), kind=kind)

    births: list[int] = []
    out = {}
    for l, W_module in enumerate(mlp.Ws):
        W = W_module.weight
        in_window = (l >= start)
        k_max = 3 if in_window else prefix_k
        factor = bool(in_window)
        skip_metric = not metric_needed(k_max, kind)

        if l == start and start > 0:
            # the checkpoint. Both arms inject the SAME zero K3/K4 state; they
            # differ only in the quality of the K[1]/K[2] that reaches it.
            if arm.startswith("late:"):
                assert sorted(K) == [1, 2], f"prefix tower carried {sorted(K)}"
            else:
                assert sorted(K) == [1, 2, 3, 4], f"prefix tower carried {sorted(K)}"
            K3, K4 = _zero_k3_checkpoint(n, device, dtype)
            K = dict(K)
            K[3], K[4] = K3, K4
            births = []

        if up_to_layer == f"pre{l}" or (l == L and up_to_layer is None):
            K = linear_kprop(K, W, k_max=k_max, d_max=output_d_max,
                             set_metric=1.0 if skip_metric else None,
                             bias=W_module.bias)
            out[f"pre{l}"] = clone_tower(K, d_max=output_d_max)
            break
        if l < L:
            metric = init_scale[l] if True else (1.0 if skip_metric else None)
            K = linear_kprop(K, W, k_max=k_max, set_metric=metric,
                             bias=W_module.bias)
            out[f"pre{l}"] = clone_tower(K, d_max=output_d_max)
            K = nonlin_kprop(K, nonlin_wick_coef=nonlin_wick[l], k_max=k_max,
                             kind=kind, use_pK=True, factor=factor)

            rank_out = None
            if 3 in K and hasattr(K[3], "_factors"):
                rank_out = int(K[3]._factors[0].shape[1])
                births = births + [l] * (rank_out - len(births))

            rank_after = rank_out
            if arm.startswith("age:") and rank_out is not None:
                K[3], births = _drop_old_columns(K[3], l - m + 1, births)
                rank_after = int(K[3]._factors[0].shape[1])
            elif arm.startswith("normsort:") and rank_out is not None:
                from p2_pb_rank import cap_factored
                K[3] = cap_factored(K[3], r_star, mode="normsort", seed=seed + l)
                rank_after = int(K[3]._factors[0].shape[1])
                births = births[:rank_after] if len(births) > rank_after else births
                births = births + [l] * (rank_after - len(births))

            if trace is not None:
                trace.append({"layer": l, "k_max": k_max, "in_window": in_window,
                              "rank_out": rank_out, "rank_retained": rank_after})
            if slice_trace is not None and 3 in K and hasattr(K[3], "get_dslice"):
                slice_trace[l] = {
                    "d3": K[3].get_dslice((3,)).detach().to("cpu").numpy().copy(),
                    "d21": K[3].get_dslice((2, 1)).detach().to("cpu").numpy().copy(),
                }

            out[f"act{l}"] = clone_tower(K, d_max=output_d_max)
            if up_to_layer == f"act{l}":
                break
    return out


def predict(weights, *, arm="full", m=0, r_star=0, seed=0, dtype=None,
            device="cuda", count_flops=False, trace=None, slice_trace=None):
    """`(depth, width)` predicted post-ReLU means, plus this arm's own cost."""
    import torch
    from mlp_kprop.flop_utils import NamedFlopCounter

    dtype = dtype or torch.float32
    depth, width = len(weights), weights[0].shape[0]
    mlp = bridge.build_mlp(weights, device=device, dtype=dtype)
    K_in = {1: torch.zeros(width, device=device, dtype=dtype),
            2: torch.eye(width, device=device, dtype=dtype)}
    kw = dict(arm=arm, m=m, r_star=r_star, seed=seed,
              up_to_layer=f"act{depth - 1}", output_d_max=1,
              trace=trace, slice_trace=slice_trace)

    # `build_mlp` wraps weights in `nn.Linear`, so without this the whole flow
    # builds an autograd graph in a module that differentiates nothing -- the
    # 2026-08-28 instrument defect, 75.21 GiB live on a 31.84 GiB card.
    flops = None
    with torch.no_grad():
        if count_flops:
            with NamedFlopCounter() as ctr:
                out = run_arm(mlp, K_in, **kw)
            flops = ctr.flop_dict()
        else:
            out = run_arm(mlp, K_in, **kw)
    means = np.stack([bridge._degree1(out[f"act{l}"][1]) for l in range(depth)],
                     axis=0)
    return means, flops


# ==========================================================================
# selftest: the `full` arm IS `mlp_kprop`
# ==========================================================================

def cmd_selftest(a) -> int:
    import torch
    from mlp_kprop.kprop_harmonic import SIMPLE, mlp_kprop

    n, depth = a.width, a.depth
    rng = np.random.default_rng(20260828)
    Ws = [np.asarray(rng.standard_normal((n, n)) * math.sqrt(2.0 / n),
                     dtype=np.float64) for _ in range(depth)]
    mlp = bridge.build_mlp(Ws, device=a.device, dtype=torch.float64)
    K_in = {1: torch.zeros(n, device=a.device, dtype=torch.float64),
            2: torch.eye(n, device=a.device, dtype=torch.float64)}

    with torch.no_grad():
        ref = mlp_kprop(mlp, K_in, k_max=3, output_all=True, kind=SIMPLE,
                        use_avg_metric=True, factor=True,
                        up_to_layer=f"act{depth - 1}", output_d_max=1)
        got = run_arm(mlp, K_in, arm="full", up_to_layer=f"act{depth - 1}",
                      output_d_max=1)

    worst = 0.0
    identical = True
    for l in range(depth):
        r = bridge._degree1(ref[f"act{l}"][1])
        g = bridge._degree1(got[f"act{l}"][1])
        identical &= bool(np.array_equal(r, g))
        den = max(float(np.max(np.abs(r))), 1e-300)
        worst = max(worst, float(np.max(np.abs(r - g)) / den))

    # `late:depth` degenerates to the control (start == 0, so the checkpoint
    # branch never fires). It is kept ONLY as a labelled non-check: it is
    # satisfied by construction and is NOT evidence that the injection works.
    with torch.no_grad():
        allw = run_arm(mlp, K_in, arm=f"late:{depth}", m=depth,
                       up_to_layer=f"act{depth - 1}", output_d_max=1)
    worst_full_window = 0.0
    for l in range(depth):
        r = bridge._degree1(ref[f"act{l}"][1])
        g = bridge._degree1(allw[f"act{l}"][1])
        den = max(float(np.max(np.abs(r))), 1e-300)
        worst_full_window = max(worst_full_window,
                                float(np.max(np.abs(r - g)) / den))

    # ------------------------------------------------------------------
    # THE check that actually exercises the intervention: an INDEPENDENT
    # composition of two unmodified upstream calls must reproduce one
    # `late:m` run of the re-driven loop.
    #
    #   mlp_kprop(k_max=2) over layers 0..L-m-1
    #     -> inject the zero K3/K4 checkpoint
    #     -> mlp_kprop(k_max=3, factor=True) over a sub-MLP of the last m
    #
    # If these agree, the k_max switch, the checkpoint injection and the
    # window are right; if they disagree, `late:m` is our re-derivation of
    # upstream rather than upstream with one intervention.
    # ------------------------------------------------------------------
    comp = {}
    for mm in [x for x in (2, 4, 8) if 0 < x < depth]:
        with torch.no_grad():
            # Leg 1: an entirely unmodified k_max=2 run over the prefix.
            pre = mlp_kprop(mlp, K_in, k_max=2, output_all=True, kind=SIMPLE,
                            use_avg_metric=True, factor=False,
                            up_to_layer=f"act{depth - mm - 1}")
            tower = pre[f"act{depth - mm - 1}"]
            assert sorted(tower) == [1, 2], f"k_max=2 tower carried {sorted(tower)}"
            assert all(t.r == 0 for t in tower.values()), "prefix HTensors are not r=0"

            # Leg 2: an entirely unmodified k_max=3 run over a sub-MLP of the
            # last mm layers, handed the prefix's RAW degree-1/2 cumulants.
            # `coerce_input` then builds the k_max=3 tower with K[3] and K[4]
            # ABSENT -- which is numerically the zero checkpoint, because
            # `factored_nonlin_kprop_k3` substitutes a rank-0 FactoredTensor for
            # a missing K[3] (factor_k3.py:379-384) and `4 in WK` is consumed
            # only under `augment` in SIMPLE mode. That equivalence is not
            # assumed: it is exactly what this check tests, since `run_arm`
            # injects explicit zeros instead.
            sub = bridge.build_mlp(Ws[depth - mm:], device=a.device,
                                   dtype=torch.float64)
            post = mlp_kprop(sub, {1: tower[1].core, 2: tower[2].core},
                             k_max=3, output_all=True, kind=SIMPLE,
                             use_avg_metric=True, factor=True,
                             up_to_layer=f"act{mm - 1}", output_d_max=1)

            mine = run_arm(mlp, K_in, arm=f"late:{mm}", m=mm,
                           up_to_layer=f"act{depth - 1}", output_d_max=1)

        w = 0.0
        bit = True
        for j2 in range(mm):
            r = bridge._degree1(post[f"act{j2}"][1])
            g = bridge._degree1(mine[f"act{depth - mm + j2}"][1])
            bit &= bool(np.array_equal(r, g))
            den = max(float(np.max(np.abs(r))), 1e-300)
            w = max(w, float(np.max(np.abs(r - g)) / den))

        # ...and it must DIFFER from the control, or the arm is not an
        # intervention at all and the agreement above would be vacuous.
        d_ctrl = _rms(bridge._degree1(mine[f"act{depth - 1}"][1])
                      - bridge._degree1(ref[f"act{depth - 1}"][1]))
        comp[f"late:{mm}"] = {
            "vs_two_unmodified_upstream_calls_worst_rel": w,
            "bit_identical": bit,
            "drift_from_control": d_ctrl,
            "differs_from_control": bool(d_ctrl > 0.0),
            "pass": bool(w <= 1e-12 and d_ctrl > 0.0)}

    res = {"check": "re-driven loop vs upstream mlp_kprop",
           "width": n, "depth": depth, "dtype": "f64",
           "bit_identical": identical, "worst_rel": worst,
           "late_full_window_worst_rel": worst_full_window,
           "late_full_window_is_satisfied_by_construction": True,
           "late_full_window_note":
               "start == 0 so the checkpoint branch never fires; this is a "
               "labelled non-check, not evidence",
           "composition_check": comp,
           "composition_check_note":
               "each late:m arm is reproduced by TWO unmodified upstream "
               "mlp_kprop calls glued at the checkpoint, and each is required "
               "to DIFFER from the control -- an agreement that could not fail "
               "would be no check at all",
           "upstream": bridge.verify_upstream(bridge._src()),
           "pass": bool(identical and worst == 0.0
                        and all(v["pass"] for v in comp.values()))}
    print(json.dumps(res, indent=1, default=float))
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "window_selftest.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(res, fh, indent=1, default=float)
    return 0 if res["pass"] else 1


# ==========================================================================
# the drift panel
# ==========================================================================


def _free(device: str) -> None:
    """Release cached GPU blocks between arms.

    Not cosmetic: the `k3pre` arms carry a full-K3 prefix, so a panel that runs
    the control plus three of them accumulates past a 31.84 GiB card. Measured,
    not guessed -- the first attempt OOM'd inside `DS_harmonic_proj`.
    """
    import gc

    gc.collect()
    if str(device).startswith("cuda"):
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:                                         # noqa: BLE001
            pass

def _rms(x) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def bracket(D: float, mse_full: float = MSE_FULL) -> dict:
    """The ONLY sound use of a drift number, and it is a two-sided bracket.

        |RMSE_full - D|  <=  RMSE_cand  <=  RMSE_full + D

    **The first version of this function wrote `max(0, rf - D)` and threw the
    lower bound away whenever `D > RMSE_full`** -- which is every arm in the
    P2-WINDOW panel. That is not the triangle inequality; the reverse triangle
    inequality gives `|rf - D|`, and when the candidate drifts FURTHER than the
    control's own error the lower bound is `D - rf`, which is large and
    informative. `local_sim/p2_pb_rank.py:839` in this same repository already
    had it right (`max(abs(mse_full**0.5 - drift**0.5), 0.0)**2`); this file
    did not, and an independent review caught it.

    The consequence is not cosmetic and it runs AGAINST the arm: a large `D`
    proves a large *floor* on the candidate's error, so a drift panel can
    decisively kill an arm rather than merely fail to endorse it.
    """
    rf = math.sqrt(mse_full)
    lo, hi = abs(rf - D), rf + D
    lo_ci = abs(math.sqrt(MSE_FULL_CI[1]) - D)     # widest anchor -> smallest floor
    hi_ci = math.sqrt(MSE_FULL_CI[1]) + D
    return {"D": D, "rmse_full": rf,
            "mse_lo": lo ** 2, "mse_hi": hi ** 2,
            "mse_lo_anchor_ci": lo_ci ** 2, "mse_hi_anchor_ci": hi_ci ** 2,
            "D_over_rmse_full": D / rf}


def provable_score_floor(D_lo: float, m: int, mse_full: float = MSE_FULL) -> dict:
    """The DECISIVE test a drift panel can actually run.

    `G1` as written tests the bracket's UPPER end: passing it would prove the
    candidate clears the bar, failing it proves only "undetermined". The
    contract's prose asserted the opposite direction, and that sentence was
    wrong. The decisive-negative test is the LOWER end:

        score_floor(m) = (|RMSE_full - D_lo95|)^2 * mult(m)

    If that exceeds the bar, the candidate **provably cannot** clear it on this
    population. Uses the conservative (low) end of `D`'s own interval and the
    widest anchor CI, so the floor is the smallest one the data supports.
    """
    rf = math.sqrt(MSE_FULL_CI[1])
    mult = weff_window_cost(m)["multiplier"] if m else 0.1
    floor = (abs(rf - D_lo)) ** 2 * mult
    return {"m": m, "D_lo": D_lo, "multiplier": mult,
            "mse_floor": (abs(rf - D_lo)) ** 2, "score_floor": floor,
            "bar": SHIP_BAR_ADJUSTED,
            "provably_cannot_clear_bar": bool(floor > SHIP_BAR_ADJUSTED),
            "x_worse_than_v27": floor / V27_ADJUSTED}


def cmd_drift(a) -> int:
    import torch

    seeds = [int(s) for s in a.seeds.split(",")]
    arms = []
    for m in [int(x) for x in a.windows.split(",")]:
        arms.append(("late:%d" % m, {"arm": "late:%d" % m, "m": m}))
    for m in [int(x) for x in (a.k3pre or "").split(",") if x]:
        arms.append(("k3pre:%d" % m, {"arm": "k3pre:%d" % m, "m": m}))
    for m in [int(x) for x in (a.age or "").split(",") if x]:
        arms.append(("age:%d" % m, {"arm": "age:%d" % m, "m": m}))
    for r in [int(x) for x in (a.normsort or "").split(",") if x]:
        arms.append(("normsort:%d" % r, {"arm": "normsort:%d" % r, "r_star": r}))

    dtype = torch.float32 if a.dtype == "f32" else torch.float64
    report = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "live_shape": (a.width, a.depth) == (WIDTH, DEPTH),
              "device": a.device,
              "width": a.width, "depth": a.depth, "dtype": a.dtype,
              "seeds": seeds, "arms": [k for k, _ in arms],
              "mse_full_anchor": MSE_FULL, "mse_full_anchor_ci": MSE_FULL_CI,
              "gt_noise_floor": GT_NOISE_FLOOR,
              "upstream": bridge.verify_upstream(bridge._src()),
              "labels_read": False, "seed_block_claimed": False,
              "per_seed": {}}

    for seed in seeds:
        # `bridge._mint` is pinned to the live 1024x16 shape. A reduced-shape
        # run mints at the requested shape and MUST declare the deviation.
        if (a.width, a.depth) == (WIDTH, DEPTH):
            Ws = bridge._mint(seed)
        else:
            from mint_provenance import canonical_mint
            Ws, _e, _s = canonical_mint(a.width, a.depth, seed)
        t0 = time.perf_counter()
        ctrl_trace, ctrl_slices = [], {}
        ctrl, ctrl_flops = predict(Ws, arm="full", dtype=dtype, device=a.device,
                                   count_flops=True, trace=ctrl_trace,
                                   slice_trace=ctrl_slices)
        _free(a.device)
        row = {"control": {"flops": ctrl_flops,
                           "total_flops": float(sum(ctrl_flops.values())),
                           "trace": ctrl_trace,
                           "seconds": time.perf_counter() - t0}}
        for name, kw in arms:
            t1 = time.perf_counter()
            tr, sl = [], {}
            # `k3pre:m` runs FULL K3 over its prefix, so it is nearly as heavy as
            # the control; three of them on one card OOM'd at 31.84 GiB. Release
            # the previous arm's allocations before starting the next.
            _free(a.device)
            got, fl = predict(Ws, dtype=dtype, device=a.device,
                              count_flops=True, trace=tr, slice_trace=sl, **kw)
            D_final = _rms(got[-1] - ctrl[-1])
            D_all = _rms(got - ctrl)
            slice_drift = {}
            for l in sorted(set(sl) & set(ctrl_slices)):
                for key in ("d3", "d21"):
                    den = max(float(np.linalg.norm(ctrl_slices[l][key])), 1e-300)
                    slice_drift.setdefault(key, {})[l] = float(
                        np.linalg.norm(sl[l][key] - ctrl_slices[l][key]) / den)
            row[name] = {
                "D_final_layer": D_final, "D_all_layers": D_all,
                "bracket": bracket(D_final),
                "flops_upstream_counter": fl,
                "total_flops_upstream_counter": float(sum(fl.values())),
                "trace": tr, "slice_drift_relative": slice_drift,
                "seconds": time.perf_counter() - t1,
            }
            if kw.get("arm", "").startswith("late:"):
                row[name]["weff_flopscope_cost"] = weff_window_cost(kw["m"])
            del got, fl, sl, tr
            _free(a.device)
            print(f"  seed {seed} {name:14s} D_final={D_final:.4e} "
                  f"D/RMSE_full={row[name]['bracket']['D_over_rmse_full']:.3f} "
                  f"upstream_flops={row[name]['total_flops_upstream_counter']:.4e}",
                  flush=True)
        report["per_seed"][str(seed)] = row

    # paired summary across networks -- the Skeptic's open item (a) from
    # 2026-08-28: no drift figure had ever carried an interval.
    summ = {}
    for name, _ in arms:
        vals = np.array([report["per_seed"][str(s)][name]["D_final_layer"]
                         for s in seeds], dtype=np.float64)
        se = float(vals.std(ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else float("nan")
        # studentised-t, as the contract specifies. The first version put a
        # z-critical 1.96 into fields named `t95_*`, which is a mislabelled
        # interval ~24% too narrow; the gate block itself always used 2.571,
        # so nothing decided moved, but the reported CI was wrong.
        tcrit = _t_crit(len(vals))
        summ[name] = {"n": len(vals), "mean_D": float(vals.mean()), "se_D": se,
                      "t_crit": tcrit,
                      "min_D": float(vals.min()), "max_D": float(vals.max()),
                      "t95_lo": float(vals.mean() - tcrit * se) if len(vals) > 1 else None,
                      "t95_hi": float(vals.mean() + tcrit * se) if len(vals) > 1 else None,
                      "bracket_at_mean_D": bracket(float(vals.mean())),
                      "bracket_at_D_lo95": bracket(
                          float(vals.mean() - tcrit * se)) if len(vals) > 1 else None}
    for name, v in summ.items():
        if v.get("t95_lo") is not None:
            mm = 0
            for nm, kw in arms:
                if nm == name:
                    mm = kw.get("m", 0)
            v["provable_score_floor"] = provable_score_floor(v["t95_lo"], mm)
    report["summary"] = summ
    report["gate_table"] = gate_table()
    report["ship_bar_adjusted"] = SHIP_BAR_ADJUSTED

    # ---- GATE-W, evaluated exactly as preregistered ----------------------
    def Ds(name):
        return [report["per_seed"][str(s_)][name]["D_final_layer"] for s_ in seeds]

    gate = {"clauses": {}, "note":
            "G1/G2/G3 are conjunctive; m is the SMALLEST window passing all "
            "three. There is deliberately no separate cost clause: cost "
            "already enters through D_max(m), and a redundant clause that "
            "cannot fail is not a gate."}
    base_name = "late:0"
    have_base = base_name in dict(arms) or base_name in report["arms"]
    for name, kw in arms:
        if not kw.get("arm", "").startswith("late:") or kw.get("m", 0) == 0:
            continue  # k3pre is a diagnostic, never gated as a candidate
        m = kw["m"]
        vals = np.asarray(Ds(name), dtype=np.float64)
        se = float(vals.std(ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else float("nan")
        d_hi = float(vals.mean() + 2.571 * se)
        dmax = d_max_for_m(m)
        g1 = bool(d_hi <= dmax)
        g2 = paired_diff(Ds(base_name), Ds(name)) if have_base else None
        ctrl_name = "normsort:%d" % (2 * a.width * m)
        g3 = paired_diff(Ds(ctrl_name), Ds(name)) if ctrl_name in report["arms"] else None
        gate["clauses"][name] = {
            "m": m, "mean_D": float(vals.mean()), "se_D": se, "D_hi95": d_hi,
            "D_max": dmax, "G1_could_clear_the_bar": g1,
            "G2_beats_pure_K2": (g2 or {}).get("improves"),
            "G2_detail": g2,
            "G3_beats_matched_rank_normsort": (g3 or {}).get("improves"),
            "G3_detail": g3, "G3_control": ctrl_name,
            "passes_all": bool(g1 and (g2 or {}).get("improves")
                               and (g3 or {}).get("improves")),
        }
    passing = [v["m"] for v in gate["clauses"].values() if v["passes_all"]]
    gate["selected_m"] = min(passing) if passing else None
    gate["verdict"] = "PASS" if passing else "FAIL"
    report["gate"] = gate
    print("\n  GATE-W:", gate["verdict"],
          "selected m =", gate["selected_m"])
    for k, v in gate["clauses"].items():
        print(f"    {k:10s} mean_D={v['mean_D']:.4e} hi95={v['D_hi95']:.4e} "
              f"D_max={v['D_max']:.4e}  G1={v['G1_could_clear_the_bar']} "
              f"G2={v['G2_beats_pure_K2']} G3={v['G3_beats_matched_rank_normsort']}")

    if a.out:
        os.makedirs(a.out, exist_ok=True)
        # Run-specific name. The committed panel must not be overwritable by a
        # later invocation -- the 2026-08-21 "an evidence file a documented
        # invocation can shrink" trap.
        tag = ("_" + a.tag) if a.tag else ""
        path = os.path.join(a.out, f"window_drift_{a.dtype}{tag}_"
                                   f"{'-'.join(str(s) for s in seeds)}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1, default=float)
        print("wrote", path)
    return 0


def cmd_cost(a) -> int:
    """Print the measured flopscope window table. No networks, no data."""
    print(f"{'m':>3} {'K3 window B':>12} {'+K2 prefix B':>13} {'multiplier':>11} "
          f"{'vs floor':>9}")
    for m in range(1, DEPTH + 1):
        c = weff_window_cost(m)
        print(f"{m:>3} {c['k3_window_budgets']:12.4f} {c['total_budgets']:13.4f} "
              f"{c['multiplier']:11.4f} {c['multiplier']/0.1:8.2f}x"
              + ("   FLOOR-PINNED" if c["floor_pinned"] else ""))
    print(f"\nGATE-W screen, ship bar {SHIP_BAR_ADJUSTED:.6e} adjusted "
          f"(2x graded v2.7), RMSE_full {math.sqrt(MSE_FULL):.6e}:")
    print(f"{'m':>3} {'multiplier':>11} {'D_max':>13}")
    for r in gate_table():
        print(f"{r['m']:>3} {r['multiplier']:11.6f} {r['D_max']:13.6e}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--drift", action="store_true")
    ap.add_argument("--cost", action="store_true")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=DEPTH)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="f32")
    ap.add_argument("--seeds", default="1000401,1000402,1000403,1000404,1000405,1000406")
    ap.add_argument("--windows", default="2,4,6,8")
    ap.add_argument("--k3pre", default="")
    ap.add_argument("--age", default="")
    ap.add_argument("--normsort", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    bridge._src()
    if a.selftest:
        return cmd_selftest(a)
    if a.cost:
        return cmd_cost(a)
    if a.drift:
        return cmd_drift(a)
    ap.error("choose one of --selftest --drift --cost")


if __name__ == "__main__":
    raise SystemExit(main())
