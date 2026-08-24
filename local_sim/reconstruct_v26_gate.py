"""
RECONSTRUCT THE v2.6 GATE — the recorded decision, re-derived by the canonical scorer.

Stage 0, item 4 of the implementation handoff. Read-only: it consumes two durable
CONFIRM artifacts and mints nothing, claims no seed block, and touches no vault.

WHAT IS BEING RE-DECIDED
------------------------
`experiments/LOG.md` 2026-08-19 (evening) records the v2.6 build with a paired
100-MLP CONFIRM comparison and three residual models, and reports the deciding
statistic as a **paired geometric mean**. `docs/CURRENT_STATE.md` carries the
result forward as 0.9521x local / 1.068-1.093x projected.

The P0.1 slice established that the competition's estimand is the **arithmetic
mean of per-MLP scores**, and that the two estimands select opposite winners on
this project's own graded data (v2.4 -> v2.4.1). So the v2.6 numbers have to be
re-derived under the right estimand before anything is decided on them. That is
all this script does. **It does not gate v2.6** — the gate needs a decision block,
Samuel's prediction, and a Phase-2 shape.

THE THREE COST MODELS, unchanged from the frozen rule in docs/VALIDATION.md 2b
-----------------------------------------------------------------------------
  local     C = counted + lambda * (local residual, as measured)
            This machine bills the same file ~4.3x what the grader billed it, so
            this model is reported and never decides.
  per-op    C = counted + lambda * ops * (grader us/op)
            The grader's price, re-derived from the most recent graded report:
            submission 327743 ran 1,416 ops in 9.845 ms => 6.953 us/op. The
            LOG's standing number is 7.0 us/op.
  transfer  C = counted + lambda * (local residual / local-to-graded ratio)
            The local residuals are MEDIANS here; experiments/LOG.md 2026-08-19
            (evening) used MEANS, giving 4.367x where this gives 4.264x. Same
            model, 0.1% apart on the gain.
            Gives the candidate no credit for being more residual-efficient than
            the control; the conservative one, and the one the LOG quotes.

Run: uv run python local_sim/reconstruct_v26_gate.py
"""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np  # noqa: E402

from competition_score import (PHASE1_LAMBDA_FLOPS_PER_SECOND,  # noqa: E402
                              PHASE1_FLOP_BUDGET, MlpRow, format_paired,
                              paired_compare, score_suite)

PROBES = os.path.join(_ROOT, "experiments", "2026-08-19_carrier_probes")
CTRL = os.path.join(PROBES, "v2_4_1_mini_control.json")
CAND = os.path.join(PROBES, "v2_6_mini_confirm.json")
OUT = os.path.join(_ROOT, "experiments", "2026-08-19_coverage", "v26_gate_reconstruction.json")

#: Graded anchor, submission 327743 (Phase 1, 2026-08-10): 1,416 ops in 9.845 ms.
GRADED_OPS = 1416.0
GRADED_RESIDUAL_S = 0.009845
GRADED_US_PER_OP = GRADED_RESIDUAL_S / GRADED_OPS * 1e6


def _rows(path):
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    r = d.get("results", d)
    cfg = d.get("run_config") or {}
    out = []
    for e in r["per_mlp"]:
        bd = ((e.get("breakdowns") or {}).get("estimator") or {})
        ns = (bd.get("by_namespace") or {})
        ops = sum(float(v.get("calls", 0.0)) for v in ns.values())
        out.append({
            "name": e.get("mlp_name", ""),
            "mse": float(e["final_layer_mse"]),
            "flops": float(e["flops_used"]),
            "resid": float(e.get("residual_wall_time_s", 0.0) or 0.0),
            "ops": ops,
            "failed": bool(e.get("error_code"))
            or any(bool(e.get(k)) for k in
                   ("budget_exhausted", "time_exhausted",
                    "residual_wall_time_exhausted", "combined_budget_exhausted")),
            "reported": float(e["adjusted_final_layer_score"]),
        })
    return out, r, cfg


def _score(rows, resid_s, budget):
    # PHASE 1, named explicitly. This tool reconstructs a Phase-1 gate from
    # Phase-1 graded reports, and it is the one replay path in this repository
    # that PRICES residual (`C = F + lambda*R`). Since P2-P0 the module default
    # is the LIVE round, where lambda is 0 -- inheriting it here would silently
    # delete the entire residual half of every residual model this file exists to
    # compare, and the three models would collapse onto one another.
    return score_suite(
        [MlpRow(mse=r["mse"], flops_used=r["flops"], residual_wall_time_s=rs,
                budget=budget, failed=r["failed"], name=r["name"])
         for r, rs in zip(rows, resid_s)],
        lam=PHASE1_LAMBDA_FLOPS_PER_SECOND,
        on_over_budget="fail",
    )


def main():
    a, ra, cfg_a = _rows(CTRL)
    b, rb, cfg_b = _rows(CAND)
    budget = float(cfg_b.get("flop_budget") or PHASE1_FLOP_BUDGET)

    print("=" * 100)
    print("v2.6 GATE, RECONSTRUCTED UNDER THE CANONICAL SCORER")
    print("=" * 100)
    print(f"  control   v2.4.1  {os.path.basename(CTRL)}   {len(a)} MLPs")
    print(f"  candidate v2.6    {os.path.basename(CAND)}   {len(b)} MLPs")
    print(f"  budget    {budget:,.0f}   lambda {PHASE1_LAMBDA_FLOPS_PER_SECOND:.3g}"
          f"   (round v1-phase1, named explicitly)")

    paired = [x["name"] for x in a] == [x["name"] for x in b] and all(x["name"] for x in a)
    print(f"  paired    same MLPs in the same order: {paired}")
    if not paired:
        raise SystemExit("arms are not paired; a paired statistic would be invalid")

    # Identity check first: our scorer must reproduce each artifact's own numbers.
    for label, rows, res in (("v2.4.1", a, ra), ("v2.6", b, rb)):
        s = _score(rows, [r["resid"] for r in rows], budget)
        worst = max(abs(s.scores[i] - rows[i]["reported"]) /
                    max(abs(rows[i]["reported"]), 1e-300) for i in range(len(rows)))
        agg_ok = abs(s.objective - float(res["adjusted_final_layer_score"])) / \
            float(res["adjusted_final_layer_score"])
        print(f"  replay {label:<7} per-MLP worst rel {worst:.3e}   "
              f"aggregate rel {agg_ok:.3e}   failed {s.n_failed}")

    ops_a = float(np.mean([x["ops"] for x in a]))
    ops_b = float(np.mean([x["ops"] for x in b]))
    res_a = float(np.median([x["resid"] for x in a]))
    res_b = float(np.median([x["resid"] for x in b]))
    print(f"\n  ops/MLP        v2.4.1 {ops_a:,.0f}   v2.6 {ops_b:,.0f}   "
          f"ratio {ops_b / ops_a:.3f}x")
    print(f"  counted/MLP    v2.4.1 {np.mean([x['flops'] for x in a]):.4e}   "
          f"v2.6 {np.mean([x['flops'] for x in b]):.4e}")
    print(f"  local residual (MEDIAN) v2.4.1 {res_a * 1e3:.2f} ms   v2.6 {res_b * 1e3:.2f} ms   "
          f"({res_a / GRADED_RESIDUAL_S:.2f}x the graded 9.845 ms)")
    print(f"  graded price   {GRADED_US_PER_OP:.3f} us/op  (327743: "
          f"{GRADED_OPS:,.0f} ops in {GRADED_RESIDUAL_S * 1e3:.3f} ms)")

    transfer = res_a / GRADED_RESIDUAL_S      # local-to-graded inflation on the CONTROL
    models = {
        "local (as measured here) — NEVER decides": (
            [x["resid"] for x in a], [x["resid"] for x in b]),
        f"per-op at the graded {GRADED_US_PER_OP:.2f} us/op": (
            [x["ops"] * GRADED_US_PER_OP * 1e-6 for x in a],
            [x["ops"] * GRADED_US_PER_OP * 1e-6 for x in b]),
        f"transfer: local residual / {transfer:.2f} (conservative)": (
            [x["resid"] / transfer for x in a], [x["resid"] / transfer for x in b]),
        "counted FLOPs only (residual excluded)": (
            [0.0] * len(a), [0.0] * len(b)),
    }

    out = {"budget": budget, "n": len(a), "graded_us_per_op": GRADED_US_PER_OP,
           "transfer_ratio": transfer, "models": {}}
    for name, (rsa, rsb) in models.items():
        sa, sb = _score(a, rsa, budget), _score(b, rsb, budget)
        r = paired_compare(sa.scores, sb.scores)
        out["models"][name] = {
            "mean_control": r.mean_control, "mean_candidate": r.mean_candidate,
            "arithmetic_gain": r.arithmetic_gain,
            "relative_improvement": r.relative_improvement,
            "boot_ci_relative": list(r.boot_ci_relative),
            # The studentised-t interval is the GATE-ELIGIBLE one since 2026-08-20
            # (docs/VALIDATION.md 2b-i). BCa and the percentile bootstrap are
            # refused by `competition_score.decide()` -- measured false-promotion
            # rates 9.1-11.5% and 6.8-8.2% against a nominal 5% at n ~ 50. They are
            # recorded here only because the earlier artifact quoted them, and they
            # must not be the sole uncertainty statement on a lane D19 will gate.
            "t_ci_difference": list(r.t_ci_difference),
            "bca_ci_difference_NOT_GATE_ELIGIBLE": list(r.bca_ci_difference),
            "geometric_gain": r.geometric_gain,
            "geometric_ci": list(r.geometric_ci),
            "n_candidate_better": r.n_candidate_better,
            "estimands_disagree": r.estimands_disagree,
        }
        print("\n--- " + name + " " + "-" * max(0, 74 - len(name)))
        print(format_paired(r, control="v2.4.1", candidate="v2.6"))

    print("\n" + "=" * 100)
    print(f"{'residual model':<46}{'ARITH gain':>12}{'95% CI (rel)':>24}{'geo gain':>11}")
    print("-" * 100)
    for name, v in out["models"].items():
        lo, hi = v["boot_ci_relative"]
        print(f"{name:<46}{v['arithmetic_gain']:>11.4f}x"
              f"{f'[{lo*100:+.2f}%, {hi*100:+.2f}%]':>24}{v['geometric_gain']:>10.4f}x")
    print("\nNOTE. This reconstructs the recorded comparison under the right estimand.")
    print("It does NOT gate v2.6: no decision block is spent, the networks are the")
    print("public split (CONFIRM tier, read-only, never an objective), and the")
    print("deciding quantity is still the grader's residual price, classified E1.")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nsaved {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
