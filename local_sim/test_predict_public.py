"""The public-score projector does the arithmetic it claims, and refuses the rest.

Slice 6 contract, `docs/CLAUDE_CODE_HANDOFF_2026-09-14.md` §0.5.

The projector's whole job is to answer "did that help?" without spending a
submission slot, so the ways it can be WRONG are the ways a slot gets spent on
a candidate that did not help:

  * pairing two runs that did not see the same networks;
  * projecting from an anchor that does not reproduce;
  * scoring an arm whose rows the grader would have failed;
  * quoting one multiplier when the server picks between two;
  * pricing residual in local seconds, which the Slice 6 contract removed from
    every decision path on 2026-09-14.

Each gets a test with a measured negative control. Synthetic reports
throughout: no truth is read, no network is minted, no block is claimed.

Run: uv run python local_sim/test_predict_public.py
"""

import copy
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np  # noqa: E402

import competition_score as cs  # noqa: E402
import predict_public as pp  # noqa: E402

RESULTS = []
B = 2.0 ** 41


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"  {name:<66} {'ok ' if ok else 'FAILED'}  {detail}")
    return ok


def head(t):
    print(f"\n{t}\n{'-' * len(t)}")


def make_run(n=40, mse=4.0e-8, cb=0.42, seed=1, ops_per_mlp=9000.0,
             wall=30.0, fail_idx=(), names=None):
    """A synthetic `whest run --json` report with everything the tool reads."""
    rs = np.random.default_rng(seed)
    per = []
    for i in range(n):
        m = float(mse * rs.uniform(0.85, 1.15))
        e = {
            "mlp_index": i,
            "mlp_name": (names[i] if names else "net-%03d" % i),
            "final_layer_mse": m,
            "flops_used": int(cb * B),
            "wall_time_s": float(wall * rs.uniform(0.9, 1.1)),
            "residual_wall_time_s": 0.12,
            "budget_exhausted": False, "time_exhausted": False,
            "residual_wall_time_exhausted": i in fail_idx,
            "combined_budget_exhausted": False, "traceback": None,
        }
        per.append(e)
    return {
        "results": {
            "per_mlp": per,
            "n_failed_mlps": len(fail_idx),
            "breakdowns": {"estimator": {"by_namespace": {
                "estimator.estimator-client": {"calls": int(ops_per_mlp * n)}}}},
        },
        "run_config": {"flop_budget": int(B), "width": 1024, "depth": 16,
                       "n_mlps": n,
                       "dataset": {"path": "hf://…@v2-phase2"}},
    }


def fake_graded(adjusted=1.5957e-8, n=50):
    """A graded report whose replay lands on a known public score."""
    mult = 0.422214
    mse = adjusted / mult
    pm, ps = [], []
    for i in range(n):
        pm.append({"mlp_index": i, "mlp_name": "graded-%03d" % i,
                   "telemetry": {"flops_used": mult * B,
                                 "residual_wall_time_s": 0.1}})
        ps.append({"mlp_index": i,
                   "scores": {"final_layer_mse": mse}})
    return {"evaluation": {"config_snapshot": {"flop_budget": B},
                           "submission_id": 330950,
                           "results": {"per_mlp": pm, "public_scores": ps}}}


ANCHOR = pp.graded_anchor(fake_graded())


def t_anchor():
    head("1. the anchor is REPLAYED, not read off a summary field")
    check("the synthetic anchor replays to its intended score",
          abs(ANCHOR["adjusted"] - 1.5957e-8) / 1.5957e-8 < 1e-9,
          f"{ANCHOR['adjusted']:.6e}")
    check("...over all 50 public rows, none failed",
          ANCHOR["n"] == 50 and ANCHOR["n_failed"] == 0, "")
    # The REAL control, since the projection actually hangs on this number.
    real = os.path.join(_ROOT, "experiments", "run_reports",
                        "f21_diagnostic_330950_graded_report.json")
    if os.path.exists(real):
        a = pp.graded_anchor(pp.load_run(real))
        check("the REAL F21 #330950 report replays to 1.5957e-8",
              abs(a["adjusted"] - 1.595740e-8) < 1e-13,
              f"{a['adjusted']:.6e}  raw {a['raw_mse']:.4e}  "
              f"mult {a['multiplier']:.6f}")
    else:
        check("the real F21 graded report is present", False, real)


def t_projection_arithmetic():
    head("2. the projection is anchor x paired ratio, and nothing else")
    ctrl = make_run(seed=1, mse=4.0e-8)
    # A candidate better by exactly 20% on every network, same cost.
    cand = copy.deepcopy(ctrl)
    for e in cand["results"]["per_mlp"]:
        e["final_layer_mse"] *= 0.8
    res = pp.project(pp.run_rows(cand), pp.run_rows(ctrl), ANCHOR)
    check("a uniform 20% MSE improvement gives ratio 0.8",
          abs(res["ratio"] - 0.8) < 1e-12, f"{res['ratio']:.9f}")
    check("...and arithmetic gain 1.25x", abs(res["arithmetic_gain"] - 1.25) < 1e-9,
          f"{res['arithmetic_gain']:.6f}")
    want = ANCHOR["adjusted"] * 0.8
    check("the projected public score is anchor x ratio",
          abs(res["projected_public_adjusted"] - want) / want < 1e-12,
          f"{res['projected_public_adjusted']:.6e} vs {want:.6e}")
    check("...which is BELOW the control's graded score, as it must be",
          res["projected_public_adjusted"] < ANCHOR["adjusted"], "")
    # A uniform RATIO improvement does NOT give a degenerate interval: the
    # per-network DIFFERENCES still vary, because the networks have different
    # MSEs. The first draft of this test asserted the opposite and was wrong
    # about the arithmetic, not about the code.
    check("a uniform ratio still leaves a real interval, and it brackets 0.8",
          res["ratio_ci"][0] <= 0.8 <= res["ratio_ci"][1],
          f"[{res['ratio_ci'][0]:.6f}, {res['ratio_ci'][1]:.6f}] "
          f"-- the differences vary even when the ratio does not")
    # identical arms
    same = pp.project(pp.run_rows(ctrl), pp.run_rows(ctrl), ANCHOR)
    check("identical arms project the anchor back, exactly",
          same["projected_public_adjusted"] == ANCHOR["adjusted"],
          f"{same['projected_public_adjusted']:.6e}")

    # A NOISY improvement must produce a real interval that brackets the truth.
    rs = np.random.default_rng(7)
    noisy = copy.deepcopy(ctrl)
    for e in noisy["results"]["per_mlp"]:
        e["final_layer_mse"] *= float(rs.uniform(0.6, 1.0))
    r2 = pp.project(pp.run_rows(noisy), pp.run_rows(ctrl), ANCHOR)
    check("a noisy improvement produces a NON-degenerate band",
          r2["ratio_ci"][1] - r2["ratio_ci"][0] > 1e-3,
          f"[{r2['ratio_ci'][0]:.4f}, {r2['ratio_ci'][1]:.4f}]")
    check("...and the band brackets the point estimate",
          r2["ratio_ci"][0] <= r2["ratio"] <= r2["ratio_ci"][1], "")


def t_cost_enters_the_score():
    head("3. cost is half the objective, and the projector knows it")
    ctrl = make_run(seed=3, mse=4.0e-8, cb=0.42)
    # Same accuracy, HALF the cost. The score must halve.
    cheap = make_run(seed=3, mse=4.0e-8, cb=0.21)
    res = pp.project(pp.run_rows(cheap), pp.run_rows(ctrl), ANCHOR)
    check("halving C/B at equal MSE halves the projected score",
          abs(res["ratio"] - 0.5) < 1e-9, f"ratio {res['ratio']:.9f}")
    # ...but not below the floor.
    floored = make_run(seed=3, mse=4.0e-8, cb=0.02)
    r2 = pp.project(pp.run_rows(floored), pp.run_rows(ctrl), ANCHOR)
    check("...and the 0.1 multiplier FLOOR stops it going further",
          abs(r2["ratio"] - (0.1 / 0.42)) < 1e-9,
          f"ratio {r2['ratio']:.6f} = floor/0.42, not 0.02/0.42")


def t_refusals():
    head("4. what it refuses, each with a control that passes")
    ctrl = make_run(seed=5)
    cand = make_run(seed=6)
    ok_res = pp.project(pp.run_rows(cand), pp.run_rows(ctrl), ANCHOR)
    check("the positive control projects", math.isfinite(ok_res["ratio"]),
          f"ratio {ok_res['ratio']:.4f}")

    other = make_run(seed=6, names=["other-%03d" % i for i in range(40)])
    try:
        pp.project(pp.run_rows(other), pp.run_rows(ctrl), ANCHOR)
        ok = False
    except cs.PairingError as e:
        ok = "not the same MLPs" in str(e)
    check("two runs over DIFFERENT networks are refused", ok,
          "equal length is not common random numbers")

    perm = copy.deepcopy(cand)
    perm["results"]["per_mlp"] = list(reversed(perm["results"]["per_mlp"]))
    res_p = pp.project(pp.run_rows(perm), pp.run_rows(ctrl), ANCHOR)
    check("a PERMUTED arm is realigned, not mispaired",
          abs(res_p["ratio"] - ok_res["ratio"]) < 1e-12
          and abs(res_p["t_ci_difference"][0]
                  - ok_res["t_ci_difference"][0]) < 1e-24,
          "same ratio AND the same interval")
    check("...and the result says it reordered",
          any("REORDER" in n.upper() for n in res_p["notes"]), "")

    failed = make_run(seed=6, fail_idx=(3, 11))
    try:
        pp.project(pp.run_rows(failed), pp.run_rows(ctrl), ANCHOR)
        ok = False
    except pp.ProjectionRefused as e:
        ok = "failed MLPs" in str(e)
    check("an arm with FAILED MLPs is refused by default", ok,
          "a failed row scores as zero-prediction MSE")
    res_f = pp.project(pp.run_rows(failed), pp.run_rows(ctrl), ANCHOR,
                       allow_failures=True)
    check("...but --allow-failures proceeds and REPORTS the count",
          res_f["n_failed_candidate"] == 2,
          f"{res_f['n_failed_candidate']} failed, and the ratio is "
          f"{res_f['ratio']:.3f} -- a failure is not a small effect")

    nob = make_run(seed=6)
    nob["run_config"]["flop_budget"] = 0
    try:
        pp.run_rows(nob)
        ok = False
    except pp.ProjectionRefused:
        ok = True
    check("a report with no budget is refused, not defaulted", ok, "")

    unnamed = copy.deepcopy(cand)
    for e in unnamed["results"]["per_mlp"]:
        e.pop("mlp_name")
        e.pop("mlp_index")
    try:
        pp.project(pp.run_rows(unnamed), pp.run_rows(ctrl), ANCHOR)
        ok = False
    except cs.PairingError:
        ok = True
    check("rows with no identity at all are refused (require_ids)", ok,
          "the projector always demands identities")


def t_multiplier_band():
    head("5. the server picks ONE offset; both ends are reported")
    band = pp.multiplier_band(0.25 * B, B)
    check("two ends are returned", len(band) == 2,
          f"offsets {[b['offset'] for b in band]}")
    check("the first is the bare counted figure", band[0]["offset"] == 0, "")
    check("the second adds exactly 2**24", band[1]["offset"] == 2 ** 24,
          f"{band[1]['offset']:,}")
    # The offset is ABSOLUTE, so its relative effect depends on F. The repo's
    # documented "+0.0122%" is at F = 137,600,326,656 -- the shipped anchor --
    # not at an arbitrary C/B, and the first draft of this test quoted it at
    # C/B 0.25 where the true figure is 0.00305%.
    check("the offset is exactly 2**24 counted FLOPs, whatever F is",
          band[1]["flops"] - band[0]["flops"] == float(2 ** 24),
          f"{band[1]['flops'] - band[0]['flops']:,.0f}")
    anchor_band = pp.multiplier_band(137_600_326_656.0, B)
    rel = (anchor_band[1]["c_over_b"] / anchor_band[0]["c_over_b"] - 1) * 100
    check("...which is the documented +0.0122% at the shipped anchor F",
          abs(rel - 0.0121927) < 1e-5, f"{rel:.6f}% at F = 137,600,326,656")
    low = pp.multiplier_band(0.02 * B, B)
    check("both ends respect the 0.1 multiplier floor",
          low[0]["multiplier"] == 0.1 and low[1]["multiplier"] == 0.1,
          "a cheap candidate is pinned, and the offset cannot unpin it")


def t_residual_is_operations_not_seconds():
    head("6. residual is priced on OPERATIONS — section 0.1 of the contract")
    import official_config as oc
    r = pp.residual_forecast(9000.0)
    check("the forecast uses the calibrated per-operation model",
          abs(r["predicted_s_point"] - 9000.0 * oc.GRADED_RESIDUAL_S_PER_OP) < 1e-15,
          f"{r['predicted_s_point']:.4f} s at "
          f"{oc.GRADED_RESIDUAL_S_PER_OP * 1e6:.1f} us/op")
    check("...and carries the model's RANGE, not just its point",
          r["predicted_s_range"][0] < r["predicted_s_point"] <= r["predicted_s_range"][1],
          f"[{r['predicted_s_range'][0]:.4f}, {r['predicted_s_range'][1]:.4f}]")
    check("9,000 operations passes the 22,000 gate", r["gate_passed"], "")
    over = pp.residual_forecast(25_000.0)
    check("25,000 operations FAILS it", not over["gate_passed"],
          f"predicted {over['predicted_s_point']:.3f} s against a "
          f"{over['cap_s']} s cliff")
    check("...and the gate is the published constant, not a local reading",
          r["gate_ops"] == oc.PHASE2_OP_COUNT_GATE,
          f"{oc.PHASE2_OP_COUNT_GATE:,}")
    # The whole point of section 0.1: a run report's LOCAL seconds must not be
    # what decides. Two runs with identical operation counts and wildly
    # different local residuals must forecast identically.
    a = pp.residual_forecast(9000.0)
    b = pp.residual_forecast(9000.0)
    check("two runs with equal ops forecast equally, whatever the host did",
          a["predicted_s_point"] == b["predicted_s_point"],
          "local seconds are advisory and enter no decision")


def t_ops_extraction():
    head("7. operations per MLP, read from the report the runner writes")
    run = make_run(n=40, ops_per_mlp=9000.0)
    check("operations per MLP are recovered from by_namespace calls",
          abs(pp.run_ops_per_mlp(run) - 9000.0) < 1e-9,
          f"{pp.run_ops_per_mlp(run):,.1f}")
    bare = make_run(n=40)
    bare["results"]["breakdowns"] = {}
    check("a report with no call counts returns None, not a guess",
          pp.run_ops_per_mlp(bare) is None,
          "no operation count means no residual forecast, and it says so")
    real = os.path.join(_ROOT, "experiments", "2026-08-25_trackA",
                        "mini_100_subprocess.json")
    if os.path.exists(real):
        got = pp.run_ops_per_mlp(pp.load_run(real))
        check("a REAL mini-100 report yields its operation count",
              got is not None and got > 0, f"{got:,.1f} operations per MLP")


def t_wall():
    head("8. the wall projection states what its factor is")
    import official_config as oc
    w = pp.wall_forecast(40.0)
    check("local worst wall is multiplied by the measured contention factor",
          abs(w["projected_worst_s"] - 40.0 * oc.GRADED_WALL_CONTENTION_FACTOR) < 1e-9,
          f"{w['projected_worst_s']:.2f} s at "
          f"{oc.GRADED_WALL_CONTENTION_FACTOR:.2f}x")
    check("...against the published 120 s cap", w["cap_s"] == 120.0,
          f"margin {w['margin']:.2f}x")
    check("a report with no wall times says so rather than assuming",
          pp.wall_forecast(None)["available"] is False, "")


def main():
    print("=" * 100)
    print("THE PUBLIC-SCORE PROJECTOR — anchor x paired ratio, and three labels")
    print("=" * 100)
    for fn in (t_anchor, t_projection_arithmetic, t_cost_enters_the_score,
               t_refusals, t_multiplier_band,
               t_residual_is_operations_not_seconds, t_ops_extraction, t_wall):
        fn()
    n_ok = sum(1 for r in RESULTS if r)
    print(f"\n{n_ok}/{len(RESULTS)} checks passed")
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
