"""1024 x 16 shape/fallback smoke, on the organisers' OWN published Phase-2 rows.

P2-CP0 deliverable. Nothing in this repository had ever run at the Phase-2 shape.
`estimator.py` (v2.6.1) is shape-generic and deliberately takes the analytic-only
fallback for `width > 256`, so it is shape-*valid* but has never been exercised at
1024 x 16, under `B = 2^41`, through the grader's lifecycle.

Why the public split rather than fresh mints: a smoke **selects nothing**, so it
is a legitimate CONFIRM-tier read (`docs/VALIDATION.md` section 0), and it costs
**no seed block and no new truth set** -- the 16 rows of one 1.1 GB shard carry
1e9-sample truth and a published `avg_variance`, so the noise floor is exact
rather than estimated. `local_sim/phase2_mint_parity.py` proved
`canonical_mint(1024, 16, mlp_seed)` reproduces those rows' weights bit for bit,
so feeding the published `mlp_seed` values through the canonical lifecycle
reproduces the grader's own networks exactly.

**This measures nothing that may choose anything.** No constant, rank,
calibration or arm is selected here. The MSE printed is a compatibility readout
on a Phase-1-tuned estimator at a shape it was never built for; it is telemetry,
not a baseline, and it must never be quoted as one.

What it asserts:

  S1  every MLP completes -- no exception, no budget/time exhaustion
  S2  the prediction is (depth, width) float-finite, i.e. all-layer shape
  S3  counted cost is under B, and the multiplier it implies is reported
  S4  layer 1 matches the exact closed form ||W_1[:, j]|| / sqrt(2 pi)
  S5  determinism -- two independent runs give identical evidence-row hashes
  S6  setup() fits its window at the new shape (a carrier that builds a point
      set in setup() is where this bites)

Run:
    uv run python local_sim/p2_shape_smoke.py
    uv run python local_sim/p2_shape_smoke.py --candidate estimator_v2_4_1.py
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

import numpy as np  # noqa: E402

import competition_score as cs  # noqa: E402
import lifecycle_eval as le  # noqa: E402
from phase2_manifest import load as load_manifest  # noqa: E402
from phase2_mint_parity import _shard  # noqa: E402

OUT_DIR = os.path.join(_ROOT, "experiments", "2026-08-21_p2_shape_smoke")

_PASS = 0
_FAIL = 0


def _candidate_layer1_error(candidate: str, input_seed: int, spec) -> float:
    """Max relative error of the candidate's OWN layer-1 row vs the closed form.

    Layer 1 is exact for x ~ N(0, I): E[ReLU(h_1)_j] = ||W_1[:, j]|| / sqrt(2 pi).
    No sampling, no closure, a known answer -- so the bar is float64 epsilon and
    not a tolerance chosen to pass.
    """
    import flopscope as flops

    import harness
    from mint_provenance import canonical_mint

    mod, _ = le.load_candidate(candidate)
    est = mod.Estimator()
    est.setup(le.SetupContext(spec))
    w_np, est_seed, _ = canonical_mint(spec.width, spec.depth, int(input_seed))
    mlp = harness.MLP(weights=[np.asarray(x) for x in w_np], width=spec.width,
                      depth=spec.depth, seed=est_seed)
    with flops.BudgetContext(flop_budget=spec.flop_budget, quiet=True):
        pred = np.asarray(est.predict(mlp, spec.flop_budget), dtype=np.float64)
    exact = np.linalg.norm(np.asarray(w_np[0], dtype=np.float64), axis=0)
    exact = exact / math.sqrt(2 * math.pi)
    return float(np.abs(pred[0] - exact).max() / np.abs(exact).max())


def check(name: str, ok: bool, detail: str = "") -> bool:
    global _PASS, _FAIL
    if ok:
        _PASS += 1
        print(f"  PASS  {name}" + (f"   {detail}" if detail else ""))
    else:
        _FAIL += 1
        print(f"  FAIL  {name}" + (f"   {detail}" if detail else ""))
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default="estimator.py")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n", type=int, default=8, help="rows to run (of 16 in a shard)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    man = load_manifest()
    width = man["shape"]["width"]
    depth = man["shape"]["depth"]
    budget = man["budget"]["flop_budget_per_mlp"]
    floor = man["budget"]["multiplier_floor"]
    n_truth = man["truth"]["n_samples_per_mlp"]
    parity = man["package_parity"]

    import pyarrow.parquet as pq

    tbl = pq.read_table(
        _shard(a.shard, man["dataset"]["commit"]),
        columns=["mlp_seed", "final_means", "weights", "avg_variance"],
    )
    n = min(a.n, tbl.num_rows)
    seeds = [int(x) for x in tbl.column("mlp_seed").to_pylist()[:n]]
    truth = np.asarray(
        [tbl.column("final_means")[i].as_py() for i in range(n)], dtype=np.float32
    )
    gt_noise = [float(v) / n_truth for v in tbl.column("avg_variance").to_pylist()[:n]]
    w1 = np.asarray(
        [np.asarray(tbl.column("weights")[i].as_py(), dtype=np.float32)
         .reshape(depth, width, width)[0] for i in range(n)]
    )

    spec = le.LifecycleSpec(width=width, depth=depth, flop_budget=budget, floor=floor)

    print("=" * 78)
    print(f"PHASE-2 SHAPE SMOKE  --  {a.candidate}  at  {width} x {depth}")
    print("=" * 78)
    print(f"  dataset      v2-phase2 @ {man['dataset']['commit'][:12]}  shard {a.shard}")
    print(f"  networks     {n} published MLPs (CONFIRM tier, read-only, selects nothing)")
    print(f"  budget       B = {budget:,}   floor {floor}")
    print(f"  stack        flopscope {parity['installed_here']['flopscope']}"
          f" / whestbench {parity['installed_here']['whestbench']}"
          f"   PARITY {parity['status']}")
    if parity["limitation_stamp"]:
        print(f"  STAMP        {parity['limitation_stamp'][:70]}...")
    print()

    # ---- the run, through the CANONICAL lifecycle: one Estimator, setup once
    rows, prov = le.evaluate(
        a.candidate, seeds, truth, spec=spec, gt_noise=gt_noise, progress_every=2
    )
    suite = le.score(rows, spec=spec)
    print()
    print(le.format_suite(rows, suite, prov))
    print()

    # ---- a second, independent run for determinism -------------------------
    rows2, prov2 = le.evaluate(a.candidate, seeds, truth, spec=spec, gt_noise=gt_noise)

    print("ASSERTIONS")
    check("S1  every MLP completes, no failure path taken",
          suite.n_failed == 0, f"{suite.n_failed}/{suite.n} failed")
    check("S1b no non-finite prediction",
          not any(r.nonfinite for r in rows))
    check("S1c no error_code on any row",
          all(not r.error_code for r in rows),
          "; ".join(sorted({r.error_code for r in rows if r.error_code}) or ["none"]))

    # S2: the evaluator already validated (depth, width) per row; assert the
    # contract explicitly so a future relaxation there cannot pass silently.
    check("S2  all-layer (depth, width) shape enforced",
          all(r.width == width and r.depth == depth for r in rows),
          f"{depth} x {width}")

    # COUNTED and EFFECTIVE are different quantities and are never mixed
    # (docs/VALIDATION.md section 2b). The multiplier the grader applies is a
    # function of the EFFECTIVE compute C = F + lambda*R, and it is computed by
    # `competition_score`, which is the ONE place the score identity is allowed
    # to live. An earlier build of this file recomputed `max(floor, F/B)` here
    # -- a second implementation that disagreed with the canonical one (0.0626
    # vs 0.0684) and was hidden only because both were under the floor. An
    # independent Skeptic caught it; at a candidate near C/B 0.09 it would have
    # written a false multiplier into the artifact.
    cb_counted = [r.flops_used / budget for r in rows]
    cb_eff = [cs.effective_compute(r.flops_used, r.residual_wall_time_s, spec.lam)
              / budget for r in rows]
    mult = [cs.score_multiplier(
                cs.effective_compute(r.flops_used, r.residual_wall_time_s, spec.lam),
                budget, failed=r.failed, floor=floor)
            for r in rows]
    check("S3  COUNTED cost strictly under B on every MLP",
          all(c < 1.0 for c in cb_counted),
          f"F/B in [{min(cb_counted):.6f}, {max(cb_counted):.6f}]")
    check("S3b EFFECTIVE cost (F + lambda*R) also under B, via competition_score",
          all(c < 1.0 for c in cb_eff),
          f"(F+lam*R)/B in [{min(cb_eff):.6f}, {max(cb_eff):.6f}]"
          f"  -> multiplier {min(mult):.4f}"
          f"   [lambda = {spec.lam:.3g}, a PHASE-1 value; residual is a"
          f" --runner local figure and does NOT transfer]")
    # How much residual it would take to lift this off the floor -- reported so
    # the floor claim carries its own margin instead of asserting itself.
    r_breach = (floor * budget - max(r.flops_used for r in rows)) / spec.lam
    print(f"        floor margin: residual would have to exceed "
          f"{r_breach:.3f} s/MLP at lambda = {spec.lam:.3g} to lift the "
          f"multiplier off {floor}")

    # S4: the CANDIDATE's own layer-1 row against the exact closed form.
    #
    # The first build of this check passed the literal `True` as its condition
    # and computed a closed form it never compared to anything -- a check that
    # cannot fail, which is the failure class docs/GOTCHAS.md names five times
    # and an independent Skeptic found here for the sixth. It now runs the
    # candidate's `predict` and compares row 0.
    #
    # `axis=0` is the COLUMN norm ||W1[:, j]||, correct under the row-vector
    # convention h @ W. The column convention would use axis=1, and that is
    # exactly what this assertion exists to catch.
    l1_max_rel = _candidate_layer1_error(a.candidate, seeds[0], spec)
    check("S4  the CANDIDATE's layer-1 row equals || W1[:, j] || / sqrt(2 pi)",
          l1_max_rel < 1e-12,
          f"max relative error {l1_max_rel:.3e}  (exact arithmetic, not a "
          f"sampling comparison -- the bar is float64 epsilon, not a tolerance)")

    h1 = le.rows_hash(rows)
    h2 = le.rows_hash(rows2)
    check("S5  determinism: two runs give identical evidence-row hashes",
          h1 == h2, f"{h1[:24]}...")

    setup_s = prov["setup_seconds"]
    check("S6  setup() fits its window at the new shape",
          setup_s + le.HOST_COLD_START_S < spec.setup_timeout_s,
          f"{setup_s * 1e3:.1f} ms + {le.HOST_COLD_START_S * 1e3:.0f} ms host "
          f"cold start < {spec.setup_timeout_s * 1e3:.0f} ms (PACKAGE DEFAULT, "
          f"not observed grader config)")

    print()
    print(f"  {_PASS} passed, {_FAIL} failed")
    print()
    print("  READ THIS BEFORE QUOTING THE MSE ABOVE. It is a Phase-1-tuned")
    print("  estimator taking its analytic-only fallback at a shape it was never")
    print("  built for, scored on PUBLIC data. It is a compatibility readout.")
    print("  It is NOT a Phase-2 baseline and it selects nothing.")

    tag = a.tag or os.path.basename(a.candidate).replace(".py", "")
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"smoke_{tag}_shard{a.shard}_n{n}.json")
    prov = dict(prov)
    prov.update({
        "purpose": "P2-CP0 shape/fallback smoke -- selects nothing",
        "tier": "CONFIRM (read-only public mini shard)",
        "dataset_commit": man["dataset"]["commit"],
        "shard": a.shard,
        "package_parity": parity["status"],
        "package_parity_blockers": parity["blockers"],
        "limitation_stamp": parity["limitation_stamp"],
        "assertions": {"passed": _PASS, "failed": _FAIL},
        "determinism_rows_hash": h1,
        "counted_cb_min": min(cb_counted), "counted_cb_max": max(cb_counted),
        "effective_cb_min": min(cb_eff), "effective_cb_max": max(cb_eff),
        "multiplier_min": min(mult), "multiplier_max": max(mult),
        "multiplier_source": "competition_score.score_multiplier on the "
                             "EFFECTIVE compute F + lambda*R",
        "lambda_used": spec.lam,
        "lambda_provenance": "PHASE-1 value 1e11. Phase-2 lambda is UNPUBLISHED.",
        "residual_is_graded": False,
        "residual_note": "residual_wall_time_s is a --runner local figure. It "
                         "does not transfer to the grader (2-3% of C there, "
                         "10-13% here). Ratios inside one path only.",
        "floor_breach_residual_s": r_breach,
        "candidate_layer1_max_rel": l1_max_rel,
    })
    # NO os.remove() bypass. `write_evidence` refuses to overwrite on purpose,
    # and an earlier build of this file deleted the file first to get past it --
    # the same defect that silently shrank the mint-parity evidence from 16 rows
    # to 4. A rerun with different parameters writes a different file; a rerun
    # with the same parameters is a no-op.
    if os.path.exists(path):
        print(f"\n  artifact already exists, not rewritten -> "
              f"{os.path.relpath(path, _ROOT)}")
    else:
        le.write_evidence(path, rows, prov, suite)
        print(f"\n  artifact -> {os.path.relpath(path, _ROOT)}")
    print("=" * 78)
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
