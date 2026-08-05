# Experiment log

## 2026-08-04 — v0: covariance propagation + offline per-layer calibration

**Setup.** Cloud sim (local_sim/): self-generated He-init 256×32 MLPs (identical distribution to the challenge generator); MC truth = 200k samples/MLP. Empirical MC-vs-MC final-layer MSE at 200k: **4.97e-7** ⇒ each measured MSE carries ≈2.5e-7 of measurement noise (small vs everything below).

**Calibration fit** (`fit_calibration.py`, 5 MLPs per depth at depths 8/16/32): per-layer least-squares scale factors for the raw covariance-propagation estimator. Results: `c_final(8)=0.99471`, `c_final(16)=0.99353`, `c_final(32)=0.99314`. The public Phase-1 write-up (forum thread 18063) independently reported ~0.992 at depth 32 — agreement. **Prefix property verified**: depth-8/16 fits match the first 8/16 entries of the depth-32 table within noise, and layer 1's factor = 1.0000 (exact math, as expected). Factors baked into `estimator.py::_CAL_LAYER_SCALE`; provenance in `calibration.json`.

**Held-out validation** (`validate_estimator.py`, 6 fresh-seed MLPs, disjoint from fit):

| | final-layer MSE | approx adjusted (×0.1 floor) |
|---|---|---|
| raw covariance propagation | 7.98e-5 | ~8.0e-6 |
| + per-layer calibration | **2.67e-5** | **~2.7e-6** |

Improvement **2.99×**. Per-MLP spread was wide (1.1e-5 … 5.0e-5) — expect grader-side luck-of-the-draw on 10-MLP suites.

**Context.** Published cov-prop baseline: 8.4e-5 → adjusted 8.4e-6. Plain Monte Carlo at full budget: ~2.8e-6 adjusted. So v0 roughly *ties full-budget brute force while spending ~1% of the budget*, and beats the strongest shipped baseline ~3×. Distance to Phase-1 leader (1.24e-8): ~200×. That's Phases A–C's job.

**Caveats.** Fitted at (width 256); estimator falls back to uncalibrated for other widths. If Phase 2 changes shape, refit is ~10 min here (or seconds on the GPU box once torch harness exists).

## 2026-08-04 — v0 SUBMITTED (submission 323872) — first graded result

Pipeline brought online on the Windows box: `git init` → `uv sync` → `whest login` →
`validate` → mini-split runs → submit. **No estimator changes were needed**:
`whest validate` passed first try against real flopscope 0.10.0, so every API name we
guessed is correct (`fnp.diagonal`, `fnp.diag`, `fnp.outer`, `fnp.einsum`,
`flops.stats.norm.cdf/pdf`, and `flops.as_symmetric(data, *, symmetry=(0,1))` — the
latter also accepts our width-256 covariance without a `SymmetryError`). `estimator.py`
is byte-identical to the cloud-authored v0.

### Results

| run | adjusted_final_layer_score | final_layer_mse | mean_compute_utilization | n_failed_mlps |
|---|---|---|---|---|
| local_sim held-out (6 MLPs) | ~2.7e-6 | 2.67e-5 | — | — |
| `whest run --runner local` (mini, 100 MLPs) | 2.678e-6 | 2.678e-5 | 1.349% | 0 |
| `whest run --runner subprocess` (mini, 100) | 2.678e-6 | 2.678e-5 | 1.329% | 0 |
| **grader (submission 323872, 100 MLPs)** | **2.4927e-6** | **2.4927e-5** | **1.4499%** | **0** |

Local↔subprocess were bit-identical (determinism confirmed). Grader came in **7% better**
than the local mini split — different MLP draw, well inside the per-MLP spread. Nothing
to investigate; the local harness predicts the grader to within a few percent.

Grader detail: `all_layers_mse` 1.7965e-5, best MLP 9.875e-7, worst 6.077e-6,
`mean_score_multiplier` 0.1 (floor), `mean_effective_compute` 3.9437e9,
all `failure_breakdown` categories 0, smoke test passed (4/4 workers).
Report: `run_reports/2026-08-04_v0_submission_323872_graded_report.json`.

### ⚠️ Phase status — we are still in PHASE 1, not Phase 2

The challenge page's round objects (authoritative):

| round id | name | window (UTC) | state |
|---|---|---|---|
| 1425 | warm-up | — | closed (leaderboard top 3.91e-8) |
| **1428** | **Phase 1** | **2026-06-18 → 2026-08-07 23:59** | **LIVE** |
| 1429 | Phase 2 | 2026-08-08 → 2026-09-19 23:59 | UPCOMING (empty leaderboard) |

Our docs assumed Phase 2 opened Aug 1. It did not — submission 323872 was graded into
round 1428 ("Phase 1"), which closes **Aug 7 23:59 UTC**. Phase 2 opens Aug 8.
Consistent with this, still **no `v1-phase2` HF dataset tag** (tags: `v1-phase1`,
`v1-warmup`). **So the config below is Phase 1's, freshly confirmed — Phase 2's is
still unknown and cannot be checked before Aug 8.**

### Config confirmed from the graded report (round 1428 = Phase 1)

`admin_config` is **not exposed to participants** — the submission page's renderer gates
it behind an internal-only flag and the field is absent from our payload. What the report
does give (`evaluation.config_snapshot`) plus what is exactly derivable from per-MLP
telemetry:

| quantity | value | source |
|---|---|---|
| dataset shape | **width 256 × depth 32, n_mlps 100** | `config_snapshot` (direct) |
| `flop_budget` B | **2.72e11** | `config_snapshot` (direct) |
| residual λ | **1.000e11 FLOP-equiv/s** | derived: `(effective_compute − flops_used)/residual_wall_time_s` = 1e11 on **all 100** MLPs, min=median=max |
| multiplier floor | **0.1** | derived: `mean_score_multiplier` = 0.1 while C/B = 1.45% ⇒ floor binding at 0.1 (not 0.5) |
| `wall_time_limit_s` | **NOT exposed** | only a lower bound: max observed per-MLP wall time 0.592 s with `time_exhausted` = 0 |
| grader runtime | whestbench 0.14.0, flopscope 0.10.0, numpy 2.2.6, Python 3.10.20, 4 workers | `runtime_environment` |

Everything matches the Phase-1 assumptions we had been carrying (256×32, B=2.72e11,
floor 0.1, λ=1e11). Nothing to refit.

### Compute accounting (per MLP, from grader telemetry)

- flopscope FLOPs: **3.269109e9**, identical on all 100 MLPs (shape-only, deterministic).
  Our pre-submission analytic estimate was ~3.3e9 — accurate.
- Dominant op: `einsum` (W^T Cov W) 3.219e9 = **98.5%** of counted FLOPs.
  `as_symmetric` costs 2.94e7 (0.9%), `stats.norm.cdf` 7.9e5, `outer` 2.1e6 — all noise.
- Residual (uninstrumented) wall time: **6.74 ms/MLP** → charged 6.74e8 FLOP-equiv,
  i.e. **17% of our total C** comes from wall-time billing, not arithmetic.
- Total C = 3.943e9 = **1.45% of budget** ⇒ deep inside the 0.1 floor. We have ~7×
  headroom in FLOPs before the multiplier starts rising, and the wall-time channel is
  cheap at our current ~0.48 s/MLP grader runtime.

### Standing

Live Phase 1 leaderboard (round 1428): #1 = 4.87e-8, #10 = 9.85e-8, rank 200 = 6.40e-7.
At 2.49e-6 v0 sits **below the displayed top 200** — the published field is far stronger
than the Jul 28 snapshot in RESEARCH_NOTES §2 suggested (that snapshot's "Phase 1 #1 =
1.24e-8" does not match the live board; treat these fresh numbers as authoritative).
v0's job was to prove the pipeline, and it did: local harness ≈ grader, zero failures,
config confirmed.

## Next planned experiments

1. ~~Submit v0 → parse run report~~ ✅ done (above). Remaining open item: `wall_time_limit_s`
   is not in the participant report — infer it from a deliberately slow submission later,
   or ask on the forum.
2. Phase A: RQMC sampler (Sobol + antithetic + exact layer 1) in local_sim; error-vs-samples
   curve → pick C/B operating point. Budget headroom measured above: 1.45% used of 10%-free.
3. Exact pair-covariance ReLU update (Wright et al. 2024 formulas) as drop-in upgrade to the
   gain approximation — measure whether residual bias shrinks enough to matter before Phase B.
4. Decide Phase 1 endgame: round 1428 closes Aug 7 23:59 UTC. Anything we want on the Phase 1
   board must land before then; Phase 2 (round 1429) opens Aug 8 and its config is unverified
   until the first graded run in that round.
