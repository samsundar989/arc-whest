"""WP21: does batched Strassen-Winograd cost the chain ACCURACY?

Contract (Rule 1): `experiments/LOG.md`, 2026-09-07, "CONTRACT (Rule 1): WP21",
committed BEFORE this module produced a number. The gate is frozen there:

    PROMOTE the deepest L whose PAIRED ARITHMETIC ADJUSTED-SCORE difference
    against the exact chain has a 95% studentised-t interval strictly
    excluding 0, on common random numbers over the 24 P01 nets.

Tier: **P01 DESIGN**, the same 24 networks Slice 1 scored, so the control's raw
MSE is directly comparable to the recorded `4.004541e-08` -- and the harness
REFUSES if the control does not reproduce it, because a control that has drifted
makes every paired difference meaningless.

`L` is SELECTED here, which is a DESIGN role, and P01 is a DESIGN block where
selection is permitted. No DECIDE block is drawn. The independent CONFIRM is the
graded 100.

Research only. Never imported by `estimator.py`.

Run:  uv run python local_sim/p2_wp21_score.py --levels 0,1,2,3,4
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# APPEND, never insert: `local_sim/` holds numpy shims named `flopscope` and
# `whestbench`, and prepending it would meter the shim.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]

import numpy as np  # noqa: E402

WIDTH, DEPTH = 1024, 16
BUDGET = 2 ** 41
B_FLOPS = float(2 ** 41)
#: The HIGH-RESOLUTION truth Slice 1 scored against: 24 P01 nets, `2**26`
#: samples per half, two independent halves A and B.
#: `experiments/2026-08-21_p2_cp1/truth_P01.npz` is the WRONG file for this
#: comparison -- it is `2**22` per replica with a cross-replica floor of
#: `9.0e-09`, twenty times coarser, and the control reads `4.79e-08` on it
#: instead of `4.00e-08`. The harness refuses rather than quietly using it.
TRUTH_A = os.path.join(_ROOT, "experiments", "2026-08-26_p2_gt", "gt_halfA.npz")
TRUTH_B = os.path.join(_ROOT, "experiments", "2026-08-26_p2_gt", "gt_halfB.npz")

#: Slice 1's recorded control, `experiments/2026-09-05_slice1/k3_full_score.json`.
CONTROL_RAW = 4.004541351994352e-08
#: The grader-convention total. This harness meters `predict` only, so it counts
#: 32,768 FLOPs and one operation LESS -- whestbench's own in-window
#: `materialise_predictions` on the (16,1024) f32 output, at 2 FLOP/element.
CONTROL_FLOPS = 1_714_587_938_406
MATERIALISE_FLOPS = 32_768


def _load(path):
    name = "wp21_" + os.path.basename(path).replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class _Ctx:
    def __init__(self, seed):
        self.seed = seed
        self.scratch_dir = _ROOT
        self.submission_dir = _ROOT


def run_arm(path, depth_L, seeds, weights_mode="f64_readonly"):
    """Predictions and cost for one arm, over `seeds`, in the grader's convention.

    Returns `(preds (n_nets, depth, width) float64, flops, ops)`.
    """
    import flopscope
    import flopscope.numpy as fnp
    import whestbench

    from lifecycle_eval import build_weights
    from mint_provenance import canonical_mint

    mod = _load(path)
    if hasattr(mod, "_STRASSEN_DEPTH"):
        mod._STRASSEN_DEPTH = int(depth_L)
    elif depth_L:
        raise RuntimeError(f"{path} has no _STRASSEN_DEPTH; cannot set L={depth_L}")

    est = mod.Estimator()
    est.setup(_Ctx(int(seeds[0])))          # OUTSIDE the meter: its own 5 s window

    out, flops, ops = [], None, None
    for s in seeds:
        w_np, _e, _s = canonical_mint(WIDTH, DEPTH, int(s))
        w = build_weights(w_np, weights_mode, fnp)
        mlp = whestbench.MLP(weights=w, width=WIDTH, depth=DEPTH, seed=int(s))
        with flopscope.BudgetContext(BUDGET) as ctx:
            pred = est.predict(mlp, BUDGET)
            flops = int(ctx.flops_used)
            ops = len(ctx.op_log)
        out.append(np.asarray(pred, dtype=np.float64))
    return np.stack(out), flops, ops


def per_net_mse(preds, truth_final):
    """Final-layer MSE per net, against one replica."""
    d = preds[:, -1, :] - truth_final
    return (d * d).mean(axis=1)


def studentised_t(diffs, iters=20000, seed=20260907):
    """95% studentised-t bootstrap interval for the mean of `diffs`.

    The binding interval of `VALIDATION.md` section 2b-i. Percentile and BCa are
    refused there on measured false-promotion rates.
    """
    x = np.asarray(diffs, dtype=np.float64)
    n = x.size
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(n)
    if se == 0:
        return m, m, m
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(iters, n))
    bs = x[idx]
    bm = bs.mean(axis=1)
    bse = bs.std(axis=1, ddof=1) / math.sqrt(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(bse > 0, (bm - m) / bse, 0.0)
    lo_t, hi_t = np.percentile(t, [2.5, 97.5])
    return m, m - hi_t * se, m - lo_t * se


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--candidate", default="estimator_k3_strassen.py")
    ap.add_argument("--control", default="estimator_k3_full.py")
    ap.add_argument("--levels", default="0,1,2,3,4")
    ap.add_argument("--n-nets", type=int, default=24)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    levels = [int(x) for x in a.levels.split(",")]

    import official_config as oc

    ta, tb = np.load(TRUTH_A), np.load(TRUTH_B)
    if not np.array_equal(ta["seeds"], tb["seeds"]):
        print("REFUSED: the two truth halves are not the same networks.")
        return 1
    seeds = ta["seeds"][:a.n_nets]
    m1 = ta["means"][:a.n_nets].astype(np.float64)
    m2 = tb["means"][:a.n_nets].astype(np.float64)
    truth = 0.5 * (m1 + m2)
    n_samp = int(ta["n_samples"])
    # Cross-replica MSE on the final layer, and the noise on the AVERAGED truth
    # each arm is actually scored against. `VALIDATION.md` 3.5 -- every MSE is
    # quoted with its noise floor.
    d = m1[:, -1, :] - m2[:, -1, :]
    cross = (d * d).mean(axis=1)
    gt_noise = cross / 4.0          # var of the mean of two independent halves

    print("=" * 100)
    print("WP21 -- batched Strassen: does it cost accuracy?")
    print("=" * 100)
    print(f"  tier              P01 DESIGN, {len(seeds)} nets, seeds "
          f"{int(seeds[0])}..{int(seeds[-1])}")
    print(f"  truth             gt_halfA/B.npz, {n_samp:,} samples per half")
    print(f"  GT noise floor    {gt_noise.mean():.4e} on the averaged truth"
          f"   (cross-replica final-layer MSE {cross.mean():.4e})")
    print(f"  op gate           {oc.PHASE2_OP_COUNT_GATE:,}"
          f"   at {oc.GRADED_RESIDUAL_S_PER_OP * 1e6:.1f} us/op")

    # ---- the control, and the refusal that protects every paired difference --
    print("\n  CONTROL — the shipped bytes of submission 330212")
    cp, cf, co = run_arm(a.control, 0, seeds)
    c_mse = per_net_mse(cp, truth[:, -1, :])
    c_raw = float(c_mse.mean())
    print(f"    raw final-layer MSE  {c_raw:.6e}"
          f"   recorded {CONTROL_RAW:.6e}"
          f"   delta {abs(c_raw / CONTROL_RAW - 1) * 100:.3f}%")
    print(f"    counted FLOPs        {cf + MATERIALISE_FLOPS:,}"
          f"   recorded {CONTROL_FLOPS:,}"
          f"   (metered {cf:,} + {MATERIALISE_FLOPS:,} materialise)")
    print(f"    operations           {co + 1:,} in the grader's convention")
    if cf + MATERIALISE_FLOPS != CONTROL_FLOPS:
        print("\n  REFUSED: the control does not reproduce the recorded FLOP total.")
        print("  A drifted control makes every paired difference below meaningless.")
        return 1
    if abs(c_raw / CONTROL_RAW - 1) > 0.02:
        print("\n  REFUSED: the control's raw MSE is more than 2% from the record.")
        print("  Either the net subset or the truth file is not Slice 1's.")
        return 1
    print("    -> control reproduces Slice 1. Paired differences are meaningful.")

    c_adj = c_mse * max(0.1, (cf + MATERIALISE_FLOPS) / B_FLOPS)

    print(f"\n  ARMS  (paired, common random numbers, same process, same order)")
    print(f"  {'L':>3}{'raw MSE':>14}{'vs ctrl':>9}{'B':>9}{'ops':>8}{'gate':>6}"
          f"{'adjusted':>13}{'vs ctrl':>9}{'95% t interval on the diff':>30}{'':>6}")
    rows = []
    for L in levels:
        if L == 0:
            preds, f, o = cp, cf, co
        else:
            preds, f, o = run_arm(a.candidate, L, seeds)
        mse = per_net_mse(preds, truth[:, -1, :])
        raw = float(mse.mean())
        mult = max(0.1, (f + MATERIALISE_FLOPS) / B_FLOPS)
        adj = mse * mult
        # exact - strassen : POSITIVE favours the Strassen arm
        diff = c_adj - adj
        mean, lo, hi = studentised_t(diff)
        excl = (lo > 0) or (hi < 0)
        verdict = ("PROMOTE" if lo > 0 else ("REJECT" if hi < 0 else "no call"))
        if L == 0:
            verdict = "control"
        print(f"  {L:>3}{raw:>14.4e}{raw / c_raw:>9.4f}"
              f"{(f + MATERIALISE_FLOPS) / B_FLOPS:>9.5f}"
              f"{o + 1:>8,}{'ok' if o + 1 <= oc.PHASE2_OP_COUNT_GATE else 'OVER':>6}"
              f"{float(adj.mean()):>13.4e}{float(adj.mean()) / float(c_adj.mean()):>9.4f}"
              f"   [{lo:+.3e}, {hi:+.3e}]{verdict:>10}")
        rows.append({"L": L, "raw_mse": raw, "raw_ratio": raw / c_raw,
                     "flops": f + MATERIALISE_FLOPS,
                     "B": (f + MATERIALISE_FLOPS) / B_FLOPS, "ops": o + 1,
                     "adjusted": float(adj.mean()),
                     "adj_ratio": float(adj.mean()) / float(c_adj.mean()),
                     "paired_diff_mean": float(mean),
                     "ci_lo": float(lo), "ci_hi": float(hi),
                     "ci_excludes_zero": bool(excl), "verdict": verdict,
                     "predicted_residual_s": (o + 1) * oc.GRADED_RESIDUAL_S_PER_OP,
                     "residual_headroom": 0.4 / ((o + 1) * oc.GRADED_RESIDUAL_S_PER_OP)})

    print("\n  The interval is on the PAIRED ARITHMETIC ADJUSTED-SCORE difference")
    print("  (control - arm), so POSITIVE and excluding zero means the arm wins.")
    print("  Studentised-t bootstrap, 20,000 resamples. Percentile and BCa are")
    print("  REFUSED by VALIDATION.md 2b-i on measured false-promotion rates.")
    print(f"\n  Every raw MSE above is quoted against a GT noise floor of")
    print(f"  {gt_noise.mean():.4e} (cross-replica {cross.mean():.4e}).")

    winners = [r for r in rows if r["verdict"] == "PROMOTE"]
    if winners:
        best = max(winners, key=lambda r: r["L"])
        print(f"\n  GATE: the deepest L clearing the frozen gate is L={best['L']}"
              f"  ->  {best['B']:.5f} B, {best['ops']:,} ops,"
              f" adjusted {best['adjusted']:.4e}")
    else:
        print("\n  GATE: NO arm clears the frozen gate. Per the contract's")
        print("  falsifier, the lever CLOSES.")

    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump({"tier": "P01 DESIGN", "n_nets": int(len(seeds)),
                       "seeds": [int(x) for x in seeds],
                       "n_samples_per_half": n_samp,
                       "gt_noise_mean": float(gt_noise.mean()),
                       "cross_replica_mse_mean": float(cross.mean()),
                       "control_raw": c_raw,
                       "control_flops": cf + MATERIALISE_FLOPS,
                       "op_gate": oc.PHASE2_OP_COUNT_GATE,
                       "graded_s_per_op": oc.GRADED_RESIDUAL_S_PER_OP,
                       "arms": rows}, fh, indent=2)
        print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
