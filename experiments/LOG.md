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

## 2026-08-05 — Phase A: measurement first (what actually reduces sampling error)

**New instrument, and it changes what we can measure.** `local_sim/real_data.py` now loads the *real* phase-1 dataset — 100 MLPs with the shipped **1e9-sample ground truth** (noise ~2e-10). The old route (self-generated MLPs + 200k-sample MC truth) carried ~5e-7 of its own noise, which is the same size as everything Phase A is trying to measure — it literally could not see the signal. Every number below is against real ground truth. `local_sim/eval_estimator.py` runs any estimator file against it through the shims.

**Real flopscope pricing** (measured, per element, `scratchpad` probe → now in RESEARCH_NOTES §6):

| op | f64 | f32 |
|---|---|---|
| matmul | 4 /MAC | **2 /MAC** |
| standard_normal | 32 | **16** |
| ReLU (`maximum`) | 1 | 0.5 |
| `stats.norm.cdf` / `pdf` / **`ppf`** | 96 / 54 / **168** | — |

float32 is exactly half price everywhere ⇒ **always sample in f32** (and the shipped weights are float32 values anyway, so it is lossless). One 256×32 forward-pass sample costs **4.21e6 FLOPs** ⇒ ~64.6k samples at full budget, ~6.5k at the 0.1 floor. Weights arrive from the grader as **float64**, so the `.astype(float32)` cast is worth 2× on the dominant term for a 2.1e6-FLOP one-off.

**Variance structure of the final layer** (`local_sim/variance_lab.py`, N=8192, real GT):

| scheme | MSE | vs MC | note |
|---|---|---|---|
| plain MC | 8.39e-6 | 1.00× | |
| antithetic (±x) | 8.17e-6 | 1.03× | **essentially nothing** |
| radius stratification | 6.98e-6 | 1.20× | |
| radius + antithetic | 5.45e-6 | 1.54× | |
| **scrambled Sobol** | **2.96e-6** | **2.84×** | the winner |

*(3 MLPs × 16 reps. The 2.84× is on the optimistic end of the seed noise — the settled figure for Sobol over plain MC is **~2×**, see the α=1.00 sweep below. The ranking is what held up.)*

**69% of plain-MC error variance lies along a single "scale mode"** (the whole layer rescaling together); Sobol cuts that share to 42%. Radius stratification barely touches it, so ‖x‖ is *not* what drives it.

**Control variates are a dead end here** (`local_sim/cv_lab.py`). Layer 1 has exactly-known moments (E[ReLU(z_i)] = σ_i/√(2π), verified to ~3σ of pure sampling noise), so it is a legitimate zero-bias control variate. Ceilings measured by per-sample covariance: scalar CVs (Σh₁, Σh₁², ‖x‖²) 1.1×; **full 256-dim h₁ vector 1.80×**; even the *illegal* ceilings using h₂/h₃ (whose means we don't know) only reach 2.1–2.5×. And stacked on Sobol the CV adds **nothing** (2.04× → 1.98×) — QMC and the CV remove the *same* variance. This independently reproduces the public finding that control-variate coupling dies by layer 32 (thread 18085). **Dropped.**

**Digital vs real-valued randomisation is the one distinction that matters** (`local_sim/impl_lab.py`, N=16384):

| randomisation | vs MC | shippable? |
|---|---|---|
| scipy scrambled (LMS + digital shift) | 2.62× → **1.88× on re-measure** | no (scipy absent in sandbox) |
| **digital XOR shift of a fixed net** | **2.04×** | **yes** |
| Cranley-Patterson real shift (u+Δ mod 1) | 1.24× | yes |
| Korobov rank-1 lattice + shift | 1.45× | yes |

A real-valued shift destroys the digital net structure and loses most of the benefit; an XOR shift preserves it exactly and costs 1 FLOP/element. That is the design of v1.1. The gap between the *digital* schemes (plain shift vs LMS vs scipy's) turned out to be **within seed noise** — see the LMS negative result below — so we shipped the simplest one.

## 2026-08-05 — v1.0 SUBMITTED: antithetic Monte Carlo, float32, exact layer 1

`estimator_v1.py`. N sized analytically from the budget to C/B ≈ 0.15 (**never** calling `budget_summary_dict()` — forum 18092 measured a 157× residual blowup from that). Analytic cost model verified against real flopscope: predicted 4,210,688 FLOPs/sample vs measured 4,213,120 — **0.06% error**.

**Gates (all passed):**

| gate | requirement | result |
|---|---|---|
| (a) local held-out | ≥5× better than v0's 2.67e-5 | 4.80e-6 → **5.56×** ✅ |
| (b) error-vs-N | ~1/N | ratios 2.89× / 1.90× / 2.07× per budget doubling; α=1.15 ✅ |
| (c) `whest validate` | pass | ✅ |
| (d) mini subprocess | 0 failures, C/B ∈ [0.10,0.20], adj < 1e-6 | 0 failures, C/B **0.1597**, adj **7.906e-7** ✅ |

Mini-split MSE 4.946e-6 vs local prediction 4.80e-6 — **3% agreement**, so the offline rig now predicts the grader well. Wall time 0.239 s/MLP locally; residual 26 ms/MLP = 2.6e9 FLOP-equiv, i.e. **6.4% of billed compute is wall time** (was 17% for v0 — the sampling estimator is more FLOP-dense, so the wall-time channel matters *less*).

Antithetic pairing was kept (it is free) but is worth ~3%, not more — the measurement above says so plainly. The win here is float32 + sampling, not the antithetic trick.

## 2026-08-05 — v1.1: Sobol RQMC with digital shift

`estimator_v1_1.py`. Joe-Kuo direction numbers (30×256 uint32) **embedded in the file** as a zlib+base64 blob (~27 KB) so single-file submission survives; `local_sim/make_sobol_blob.py` regenerates them and asserts the pure-integer generator reproduces scipy's point set **exactly**, and `local_sim/inject_sobol_blob.py` round-trips the blob after injection. Points are generated in-estimator with 30 XOR passes (~0.14% of budget), digitally shifted per dimension from `mlp.seed`, mapped through `flops.stats.norm.ppf` (168 FLOPs/elt ≈ 1% of budget), then run forward in f32. N is a power of two (net property), C/B target 0.25.

**Local held-out (40 MLPs, real GT):**

| budget × | N | C/B | final-layer MSE | adjusted |
|---|---|---|---|---|
| 0.25 | 4096 | 0.064 | 8.32e-6 | 8.32e-7 |
| 0.5 | 8192 | 0.129 | 3.87e-6 | 4.98e-7 |
| **1.0** | **16384** | **0.257** | **1.573e-6** | **4.049e-7** |

α = **1.20** — genuinely superlinear, so unlike plain MC this *does* reward spending past the floor. **1.78× better than v1.0** on adjusted score (gate wanted ≥1.5×), **17× better than v0** on MSE.

**Mini subprocess (gate d)**: adjusted **4.154e-7**, MSE 1.470e-6, C/B 0.2826, 0 failures — within 2.6% of the local prediction. Not submitted: v1.2 below supersedes it, so it would have wasted a submission.

**Negative result — LMS scrambling is not worth it.** scipy's `scramble=True` is linear-matrix-scrambling + digital shift, and an earlier 3-seed measurement suggested it was worth 2.62× vs our digital shift's 2.04×. Re-measured with 8 reps × 3 MLPs, the schemes are **within noise of each other**: digital shift 2.03×, LMS+shift 2.18×, scipy scrambled 1.88× (i.e. scipy came out *worse* than our shift on this seed set). The earlier 2.62× was seed luck. LMS would have cost a fiddly GF(2) bit-matrix transform for ~7%. **Dropped** — and a reminder to re-measure apparent wins with more seeds before building on them.

## 2026-08-05 — v1.2: the operating point is an interior optimum

Same method as v1.1; only the operating point and chunk size changed. Two measurements drove it:

**1. Wall time is a non-issue, so budget is the only constraint.** From v1.0's graded report: grader wall time **0.408 s/MLP** (max 0.611) for 4.08e10 FLOPs — 134.7 GFLOP/s through flopscope, and *less* wall time than v0 spent on 12× fewer FLOPs (f32 matmul is far more flopscope-efficient than f64 einsum). Residual was **4.49 ms/MLP = 1.09% of billed compute**. Against a 60 s limit that is ~98× headroom.

**2. The adjusted score has a genuine interior optimum** (40 MLPs, real GT):

| C/B | N | final-layer MSE | adjusted |
|---|---|---|---|
| 0.257 | 16384 | 1.573e-6 | 4.049e-7 |
| **0.515** | **32768** | **6.760e-7** | **3.480e-7** |
| 1.030 | 65536 | 3.533e-7 | 3.638e-7 |

Below the optimum the error falls faster than the multiplier rises; above it the reverse. (C/B 1.03 would also exceed the budget outright and fail every MLP, so it was never a candidate.) Set `_TARGET_CB = 0.50`, and raised `_CHUNK` 4096 → 16384 to cut the Sobol generator's instrumented-op count 4× (temporary release is billed as residual — forum 18101).

Local held-out (40 MLPs): MSE **6.760e-7**, adjusted **3.480e-7**.

**…and the mini run said no.** v1.2's mini subprocess came back **4.679e-7** — *worse* than v1.1's 4.154e-7. Two things went wrong at once: the real C/B was 0.5595 not 0.5148 (residual wall time adds ~9% at this op count), and the MSE was 8.362e-7 not 6.760e-7. **The C/B=0.5 optimum was an artifact of fitting on the first 40 MLPs.**

Re-measured on **all 100** mini MLPs (which is exactly the grader's public set):

| C/B (analytic) | N | final-layer MSE | ratio vs prev | adjusted |
|---|---|---|---|---|
| 0.032 | 2048 | 1.227e-5 | — | 1.227e-6 |
| 0.064 | 4096 | 6.663e-6 | 1.84× | 6.663e-7 |
| 0.129 | 8192 | 3.377e-6 | 1.97× | 4.346e-7 |
| **0.257** | **16384** | **1.604e-6** | **2.11×** | **4.128e-7** |
| 0.515 | 32768 | 7.686e-7 | 2.09× | 3.957e-7 |

**α = 1.00 exactly.** Sobol buys a **constant ~2× factor** over plain Monte Carlo — it does *not* improve the convergence rate. The α=1.20 measured on 40 MLPs was a subset artifact, and it is what made C/B=0.5 look good. With α=1 the adjusted score is flat above the floor, so there is no real optimum to chase; the residual-wall-time surcharge (which grows with op count) then tips the balance back toward *lower* N. Ranking by actual mini runs: **v1.1 (N=16384) 4.154e-7 > v1.2 (N=32768) 4.679e-7**.

**Lesson, twice in one day**: both the LMS win and the C/B=0.5 optimum evaporated when re-measured on more data. Fit operating points on the full 100-MLP split, not a subset.

## 2026-08-05 — v1.1 SUBMITTED and GRADED (submission 324094) — the sprint's best

`estimator_v1_1.py` promoted to `estimator.py`. **Graded: adjusted 3.893e-7**, final-layer MSE 1.4907e-6, `all_layers_mse` 3.006e-6, C/B **0.2612**, `n_failed_mlps` 0, best MLP 1.085e-7 / worst 2.447e-6. Predicted 4.154e-7 from the mini split → grader **6.3% better**, same direction and size as v1.0's 14%.

**Sprint scoreboard (all graded, round 1428):**

| version | method | graded adjusted | vs v0 |
|---|---|---|---|
| v0 (323872) | covariance propagation + calibration | 2.4927e-6 | 1.00× |
| v1.0 (324075) | antithetic MC, f32, exact layer 1 | 6.781e-7 | **3.68×** |
| **v1.1 (324094)** | **Sobol RQMC + digital shift** | **3.893e-7** | **6.40×** |

**Live board position: rank 158** (was below the displayed top 200 with v0). #1 is 4.87e-8, so the leader is 8.0× ahead of us.

**Cost model, graded**: 6.998e10 FLOPs/MLP, residual wall time 10.53 ms/MLP ⇒ **1.48% of billed compute is wall time** (v0 was 17%, v1.0 1.09%). λ = 1.000e11 confirmed a third time. The wall-time channel is simply not a constraint for a FLOP-dense f32 sampling estimator — which is also why we have no incentive to go near the bypass of thread 18099.

**Submissions used**: 2 of the day's budget (v1.0, v1.1); 2 held in reserve.

## 2026-08-05 — B1 FUSION: negative. The premise was wrong.

**Verdict: do not ship. B1 buys ~5–7% MSE for a 4.7% compute tax — net ≈ 2%. Gate (mini ≤3.2e-7) not met, so nothing was submitted; v1.1 remains our Phase 1 entry.**

New instruments: `local_sim/covprop_features.py` (numpy mirror of v0, verified against `estimator_v0_archive.py` to **8.9e-15**), `collect_fusion_data.py` (caches t/p/s/σ̂/α̂/var per MLP for a whole split), plus `fusion_anatomy.py`, `covprop_bias_model.py`, `scale_mode_probe.py`, `scale_source_probe.py`, `layer_crossover.py`, `b2_tail_probe.py`, `fit_fusion.py`. Fitted on the **full 1000-MLP split** (1e9-sample truth ⇒ zero label noise), 200 MLPs held out.

### The premise that failed

The plan assumed *"v1.1's error is pure variance; v0's is pure smooth bias"*, so fusing them should pay. Measured:

| | MSE | common-scale share |
|---|---|---|
| covprop (v0) | 2.650e-5 | 25% |
| sampling (v1.1) | 1.611e-6 | **48%** |
| corr(their errors) | −0.098 | (near 0, so fusion is *sound*, just weak) |

**Covprop's residual is not a smooth bias.** Conditioning on α = μ̂/σ̂, the residual's conditional mean is only 0.1–0.5 of its conditional rms, and subtracting a fitted α-curve removes just **3.5%** of its variance held-out (3.154e-5 → 3.004e-5). It is conditionally-centred noise.

**And covprop loses to sampling in every α band** by 10–100× (only α<−4 favours it, where both errors are ~1e-13 and irrelevant). An α-gated hard switch gives **0.0%**. So the per-neuron GLS blend has w_i≈0 everywhere; blending helps only through the generic independent-errors effect, w_global = 0.052.

### Held-out results (200 MLPs, disjoint from the fit)

| variant | MSE | vs sampling |
|---|---|---|
| sampling alone (v1.1) | 1.4893e-6 | — |
| + scale-mode kill | 1.4360e-6 | +3.6% |
| + global precision blend | 1.4195e-6 | +4.7% |
| + per-neuron GLS blend | 1.4718e-6 | +1.2% |
| + scale kill **and** per-neuron blend | 1.5395e-6 | **−3.4%** |

### Why the scale kill can't work from covprop

The 48% scale mode is real and worth 1.9× if it could be removed exactly (oracle: 1.604e-6 → 8.349e-7). But the online observable ĝ = ⟨s−p, p⟩/⟨p,p⟩ measures *covprop's* scale error, not ours: sd(g_covprop) = 0.00276 vs sd(g_sampling) = 0.00088, a 3× contamination. Predicted usable R² = 0.092; **measured holdout R² = 0.094.** Using a mid-depth probe (layer 12) instead of the final layer raises the LOO R² to 0.155 — better, still ~7% of MSE.

Two source-level alternatives were tested and are dead: the realized **radial imbalance** of the point set (LOO R² = **−0.010** — the scale mode is *not* an unlucky radius) and the **layer-1 exact reference** (LOO R² = **−0.037** — the information isn't there early; the fluctuation is generated through depth, corr(g_k, g_L) climbing 0.22 → 0.99).

### B2: k = 0. Both mechanisms measured, both negative.

*Propagating analytically from the input* stops beating sampling at **layer 4** and degrades monotonically to 18.2× worse by layer 32:

| layer | covprop MSE | sampling MSE | winner |
|---|---|---|---|
| 2 | 1.463e-6 | 4.139e-6 | covprop |
| 3 | 3.904e-6 | 5.222e-6 | covprop |
| **4** | 6.129e-6 | 5.518e-6 | **sampling** |
| 32 | 2.704e-5 | 1.489e-6 | sampling (18.2×) |

*Starting the analytic step from the sampled state at L−k* — B2's actual proposal — is worse still (`b2_tail_probe.py`, 40 held-out MLPs, N=16384):

| tail treatment | MSE | vs sampling |
|---|---|---|
| pure sampling | 1.6933e-6 | — |
| analytic last 1 layer | 2.8327e-6 | **−67.3%** |
| analytic last 2 layers | 4.0619e-6 | −139.9% |
| analytic last 3 layers | 5.3203e-6 | −214.2% |

Even **one** Gaussian-closure step from accurately sampled moments costs more bias than the MC noise it removes. **Recommended k = 0**; B2 as conceived should be dropped.

### Fusion-hurt map (the requested diagnostic)

Global blend at w=0.052, final layer, held-out, by α band:

| α band | share of neurons | change vs sampling |
|---|---|---|
| < −3 | 34% | +1.2% |
| −3…−2 | 8% | −1.9% |
| −2…−1 | 7% | **−6.0%** |
| −1…0 | 7% | −0.7% |
| 0…1 | 7% | +1.5% |
| 1…2 | 7% | +2.0% |
| 2…3 | 8% | +5.0% |
| > 3 | 33% | +5.4% |

Blending helps only where α > 0 and hurts most at α ∈ [−2,−1] — the partially-firing neurons where Gaussian closure is least accurate. Gating the blend on α > 0 would give ≈ +4.9% instead of +4.7%: not worth the complexity. **α > 0 neurons carry 98% of the final-layer MSE**, so any future analytic effort belongs there, not on the dead/saturated tails.

## 2026-08-05 — input rotation (untasked side-experiment): small, real, and a third lesson

Our Sobol net converges at **α = 1.00**, i.e. it buys a constant factor over plain MC and no better rate. The textbook cause is dimension ordering: a Sobol net is equidistributed best in its low-indexed coordinates, and we feed the 256 input dimensions in arbitrary order, so the network's most influential input directions land on arbitrary (often poor) Sobol dimensions. Since x ~ N(0,I) is rotationally invariant, substituting x = Qu for orthogonal Q is free statistically and costs one (N×256)@(256×256) matmul — one extra layer out of 32, ≈3.2% of the sampling budget. Deterministic (Q comes from the weights), so the seed contract is untouched.

| rotation | 40 MLPs of `full` | **100 MLPs of `mini`** | α (mini) |
|---|---|---|---|
| none | 1.8711e-6 | 1.6742e-6 | 0.87 |
| **eig(W₁W₁ᵀ)**, descending | 1.2476e-6 (**+33.3%**) | 1.5360e-6 (**+8.3%**) | 1.01 |
| eig(JJᵀ), linearised Jacobian | 1.6668e-6 (+10.9%) | 1.4916e-6 (+10.9%) | 1.06 |

**The 40-MLP result did not survive.** +33.3% became +8.3% on the full 100-MLP split — the *third* time in two days a subset measurement has evaporated (after LMS scrambling and the C/B=0.5 "optimum"). Net of its 3.2% tax the W₁ rotation is worth ~5%, and the Jacobian variant ~3% net once covprop's 4.7% is paid. Below anything worth a submission, but the α shift (0.87 → 1.01/1.06) is the one hint that a *better* rotation might buy a rate rather than a constant.

**Standing rule from here on: no operating-point or variant decision on fewer than the full 100-MLP mini split.**

## 2026-08-05 — B3 GPU learnability probe: **R² = 0.012 → DROP Phase C**

Setup (`local_sim/gpu_mint.py`, `gpu_train_probe.py`, isolated `.venv-gpu` with torch 2.13+cu130; **no torch/numpy anywhere near an estimator file**). RTX 5090 measured at 38.3 TFLOP/s fp32.

**Parity first**, as required: GPU forward pass vs the CPU/flopscope path on 5 real dataset MLPs at 262k samples — final-layer means agree to **2.7e-3 max** (pure MC noise at that sample count; the required bar was ~1e-3 rms), and GPU covprop matches the CPU reference to **1e-15**. Passed.

**Mint**: 50,000 He 256×32 MLPs, labels = 65,536-sample MC of the final-layer means, plus calibrated-covprop features and the residual b_i = truth_i − p_i. 20 min wall. Sanity: minted Var(b) = **2.697e-5** vs the real dataset's **2.650e-5** — the synthetic distribution matches.

**Result** (split by MLP, never by neuron — 42,500 train / 7,500 held-out, 12.8M neuron examples):

| model | held-out R² |
|---|---|
| linear on 40 features | 0.0145 |
| 2-hidden-layer MLP (256 wide) | **0.0120** |

Label noise is ~5.3e-7 against Var(b) = 2.7e-5, i.e. **2%** — an R² ceiling of 0.98, so noise is not masking the signal. It simply isn't there. This corroborates the α-curve result above (3.5% of variance removable) and the public negative results in thread 18063 ("learned correctors all failed to beat the scalar at depth 32").

**What R² would have needed to be.** Because Var(b) is 18.2× the sampling variance, R² translates into fused score very unfavourably:

| R² | covprop variance after correction | best fused MSE | vs v1.1 |
|---|---|---|---|
| 0.00 | 2.713e-5 | 1.4118e-6 | +5.2% |
| 0.50 | 1.357e-5 | 1.3420e-6 | +9.9% |
| 0.90 | 2.713e-6 | 9.615e-7 | +35.4% |
| 0.95 | 1.357e-6 | 7.099e-7 | +52.3% |
| **0.012 (measured)** | 2.680e-5 | 1.4109e-6 | **+5.3%** |

So the mission's "R² > 0.5 → Phase C alive" threshold would itself only have bought ~10%; the fusion direction needed **R² > 0.9** to matter. We measured 0.012.

**Verdict: drop Phase C.** The GPU pivots to B2 calibration data per the plan — though note B2's own measurement (k=0) says that track is also dead, so the GPU is better spent on variance reduction for the *sampling* estimator.

*Caveat kept honest*: the features were per-neuron covprop diagnostics + last-two-layer weight norms + per-MLP aggregate context. A fundamentally different representation (weight-space equivariant networks over the whole MLP) is not ruled out by this experiment — but at 18× the sampling variance it would have to reach R² > 0.9 to be worth anything, and nothing in the data suggests that is close.

## 2026-08-05 — SPRINT 1: the score identity, and two wins that stack to 1.61×

**The framing that made this session productive.** Above the 0.1 multiplier floor,

    adjusted = MSE × C/B = (V/N) × (N·c/B) = **V·c/B**

per-sample variance × per-sample cost, with N cancelling. (Check: V = 1.4893e-6 × 16384 = 2.44e-2, c = 4.273e6 ⇒ 3.83e-7 vs v1.1's graded 3.893e-7.) So **cheapening a sample is worth exactly as much as de-noising it**, and picking N is not a lever at all. Every Sprint-1 component is judged on V·c/B, never on MSE.

Pricing verified from the flopscope docs and re-measured locally: **f16/int8/bool are rate 1.0, identical to f32 — there is no dtype trick**; matmul bills `(2K−1)·M·N` **by shapes only** (verified exactly: 511×65536); gather/where 4/elt; `norm.ppf` 168/elt; `eigh` measured 3.0e8, `svd` 8.7e8 (so eigh on JJᵀ replaces an SVD at a third the price).

### 1. Dead-neuron pruning — a pure COST lever

α-histogram over all 32 layers (20 MLPs): 5.1% below −5, 4.8% in [−5,−4), 6.3% in [−4,−3), 8.1% in [−3,−2) — so ~16% of neurons sit below α = −3 and barely ever fire. Prune them from **both** sides of each matmul (only smaller *dense* matrices earn anything), compensating with the exact constant offset Σ_dead W[i,:]·μ_i, and report them with their closed-form rectified-Gaussian mean.

| τ | alive | cost/sample | cost cut | MSE | MSE change | V·c/B | gain |
|---|---|---|---|---|---|---|---|
| −2.0 | 75.9% | 2,511,196 | 40.0% | 2.2653e-6 | +35.3% | 3.427e-7 | 1.23× |
| **−2.5** | **80.1%** | **2,777,144** | **33.7%** | **1.7180e-6** | **+2.6%** | **2.874e-7** | **1.47×** |
| −3.0 | 83.9% | 3,020,231 | 27.9% | 1.6770e-6 | +0.2% | 3.051e-7 | 1.38× |
| −4.0 | 90.1% | 3,445,113 | 17.7% | 1.6741e-6 | −0.0% | 3.474e-7 | 1.22× |

**τ = −2.5**, where the pruned neurons' own MSE against their true means is **3.8e-9** — far inside the ≪1e-8 bias budget. Below −2.5 the MSE penalty explodes (+35% at −2.0).

### 2. Alignment rotation — a VARIANCE lever, at zero marginal cost

x ~ N(0,I) is rotation-invariant, so x = Qu is free; folding Q into W₁ makes the per-sample cost exactly zero (only the one-off basis computation is billed). Two bases measured.

### Full stack, all combinations, **100-MLP mini**, honest end-to-end cost

| prune | rotation | radius | MSE | cost/sample | C/B | V·c/B | gain |
|---|---|---|---|---|---|---|---|
| no | none | no | 1.6742e-6 | 4,451,745 | 0.268 | 4.489e-7 | 1.00× |
| no | none | yes | 1.6830e-6 | 4,452,681 | 0.268 | 4.514e-7 | 0.99× |
| no | eig(W₁W₁ᵀ) | no | 1.5360e-6 | 4,460,961 | 0.269 | 4.127e-7 | 1.09× |
| no | eig(JJᵀ) | no | 1.4691e-6 | 4,541,733 | 0.274 | 4.019e-7 | 1.12× |
| yes | none | no | 1.7180e-6 | 3,042,777 | 0.183 | 3.149e-7 | 1.43× |
| yes | eig(W₁W₁ᵀ) | no | 1.5557e-6 | 3,051,993 | 0.184 | 2.860e-7 | 1.57× |
| yes | eig(W₁W₁ᵀ) | yes | 1.5430e-6 | 3,052,929 | 0.184 | 2.838e-7 | 1.58× |
| **yes** | **eig(JJᵀ)** | **no** | **1.4769e-6** | **3,132,765** | **0.189** | **2.787e-7** | **1.61×** |

**Radius is neutral** (±0.5% everywhere) — dropped, which also avoids extending the shipped Sobol table to 257 dimensions. The two wins are close to multiplicative (1.43 × 1.12 = 1.60 vs 1.61 measured), consistent with one attacking cost and the other variance.

`estimator_v2.py`: Sobol RQMC + pruning at τ=−2.5 + eig(JJᵀ) rotation folded into W₁. Validates, deterministic, local mini MSE **1.3624e-6**. Note `all_layers_mse` rises to 5.3e-5 because rows 1..31 now carry closure means rather than sampled ones — deliberate (only the final layer is scored, and it avoids a scatter).

## 2026-08-05 — G1 amortized common-mode regressor: **fails its gate (1.000×)**

3000 minted He MLPs × 8 independent digitally-shifted Sobol draws each (GPU, ~11 min). Observables per draw, all computable online: the per-layer empirical gain trajectory mean‖h_l‖²/‖h_{l−1}‖², per-layer mean activation (64 features). Target: the realized final-layer scale mode g, against 2M-sample GPU truth. Cross-fitted 5-fold **over MLPs** (never over draws — splitting draws would leak the network).

| | |
|---|---|
| minted error variance | 1.6679e-6 |
| scale-mode share | **44.5%** (real data said 48% — the mint reproduces it) |
| sd(g) | 0.00092 |
| **out-of-fold correlation** | **+0.0116** |
| James-Stein shrinkage factor | 0.0024 |
| variance after correction | 1.6678e-6 (**1.000×**) |
| oracle ceiling (g known) | 9.2550e-7 (1.802×) |

**The safety design worked.** Because the prediction is shrunk by its own out-of-fold reliability, a useless model collapses to a no-op instead of injecting bias — the corrector changed the error by 0.006%. That is the behaviour we want from anything that ships.

**This is the fifth independent failure to predict g**, and they now cover very different information sources:

| observable | result |
|---|---|
| covprop reference at the final layer | R² 0.094 |
| covprop reference at mid-depth (layer 12) | LOO R² 0.155 |
| point-set radial imbalance | LOO R² −0.010 |
| exact layer-1 closed form | LOO R² −0.037 |
| **learned, from the gain trajectory (G1)** | **out-of-fold corr 0.012** |

The common mode is worth 1.80–1.90× and is the largest single structure in our error, but it is looking genuinely unobservable from anything cheap. `g = ⟨ŝ − t, t⟩/⟨t,t⟩` is a projection onto the unknown truth direction; every proxy for that direction we have is noisier than the quantity being measured. **Recommendation: stop attacking the mode through prediction.** What remains is to attack it at the source — a point set whose realized error has less mass in that direction (which is G2's premise, and does not require predicting anything per-MLP).

## 2026-08-05 — v2 on the grader-parity split: 1.19×, **misses the gate, nothing submitted**

| version | change | mini adjusted | mini MSE | C/B | vs v1.1 |
|---|---|---|---|---|---|
| v1.1 (graded 3.893e-7) | — | 4.154e-7 | 1.4700e-6 | 0.2826 | 1.00× |
| **v2.0** | prune τ=−2.5 + eig(JJᵀ) rotation | **3.483e-7** | 1.6079e-6 | 0.2208 | **1.19×** |
| v2.1 | + diagonal α, W₁ basis | — | offline 1.834e-6 | — | rejected |
| v2.2 | + N doubled (32768) | 3.588e-7 | 8.863e-7 | 0.4112 | 1.16× |
| v2.3 | + f32 closure and Jacobian | — | offline 1.424e-6 | — | rejected |

Gate was ≥1.2× **and** mini ≤3.4e-7. v2.0 gives 1.19× and 3.483e-7 — a miss by ~1%, on both. Four separate attempts to close it, all failed:

1. **v2.1 — cheaper scaffolds.** Diagonal (mean-field) propagation for α plus the W₁ basis. **MSE +35%.** The diagonal ignores correlations, mis-ranks neurons and prunes live ones. α is worth its cost.
2. **v2.2 — amortise fixed costs over more samples.** The identity `adjusted = (V·c_marginal + V·F/N)/B` says larger N should help, since F ≈ 6.9e9 per MLP doesn't scale. MSE did halve as predicted — but C/B rose 1.86× because **residual wall time scales with chunk count**, not just with fixed work (64.8 → 115.8 ms/MLP). Net worse.
3. **v2.3 — f32 for the closure and Jacobian.** Saves 6% of FLOPs (f64 is rate 2.0), but perturbs α enough to shift pruning decisions: **offline MSE +4.5%**, which more than eats the saving.
4. **Better pruning criterion.** α says how *often* a neuron fires, not how much the output *cares*. Tried ranking by Var(h)·‖B_k‖² (B_k = backward sensitivity to the final layer, 31 matmuls ≈ 1.0e9) and by Var(h) alone, at matched alive fraction with global thresholds:

| criterion | keep 0.75 | keep 0.80 | keep 0.85 |
|---|---|---|---|
| **α** | **4.629e-7** | **3.146e-7** | **3.388e-7** |
| Var(h)·‖sensitivity‖² | 9.681e-7 (0.48×) | 3.316e-7 (0.95×) | 3.446e-7 (0.98×) |
| Var(h) alone | 5.256e-7 (0.88×) | 3.265e-7 (0.96×) | 3.393e-7 (1.00×) |

**α wins outright**, and the sensitivity weighting costs an extra 1.0e9 for nothing. *(A first version of this experiment forced a fixed pruned fraction per layer and produced MSEs of ~1e-3 — a bug, not a result: at layer 1 the input mean is zero so μ_pre = 0 and α = 0 for every neuron, meaning nothing there is dead and a per-layer quota prunes fully live neurons. Global thresholds are the correct analogue of τ.)*

**Decision: no submission.** Both reserves kept; v1.1 (graded 3.893e-7, rank ~159) remains our Phase 1 entry. The marginal value of a 13% gain on a board we sit at #159 on, two days from close, does not justify overriding the gate — and the grader-parity run already confirmed pruning runs clean (n_failed = 0), which was the real information we wanted before Phase 2.

**Residual wall time is now the binding constraint on this design**: 10.8% of billed compute at 1794 ops/MLP, against v1.1's 1.5% at ~250 ops. Any further cost work must reduce op *count*, not FLOPs.

## 2026-08-05 — G2 task-optimized point set: plateaus at ~1.06×, and a regulariser that BIASES

First run looked catastrophic — held-out 2.15e-5 against a Sobol baseline of 9.156e-6, i.e. **0.43×**, and it never recovered over 200 steps. Dropping the learning rate 200× changed nothing (drift stayed at 0.0032), which ruled out the optimiser. The cause was the regulariser:

**`normalise_` alone, with zero gradient steps, costs 2.35×** (9.156e-6 → 2.152e-5).

The plan called for "exact per-coordinate mean/var matching" to keep the set a valid sampler. Measured, that is actively harmful. The Sobol set's per-coordinate sample moments are mean **−0.00149**, std **1.00278**; forcing them to exactly (0,1) applies a ~0.3% rescale to every point, and a systematic 0.3% scale error on activations of ~0.45 is far larger than the ~1e-6 MSE we are chasing. **A QMC set is unbiased because it is the correct quantile transform of an equidistributed net, not because its sample moments are exactly (0,1)** — moment-matching trades a negligible variance reduction for a first-order bias. Disabled; drift regularisation only.

(Related, and counter-intuitive: the very first Sobol point maps to z = −6.12 in *every* coordinate — norm 97.9 against a typical 16.0. Deleting that outlier makes things **worse** (0.734×): it is load-bearing, balancing the rest of the net.)

With the regulariser removed, G2 behaves:

| step | held-out | vs Sobol | drift |
|---|---|---|---|
| 0 | 9.132e-6 | 1.003× | 0.0002 |
| 50 | 8.702e-6 | 1.052× | 0.0014 |
| **125** | **8.588e-6** | **1.066×** | 0.0020 |
| 250 | 8.720e-6 | 1.050× | 0.0027 |

**Read: not tracking toward ship-worthy.** It gains ~6% by step 125 and then plateaus — the curve is flat-to-regressing, not still climbing, and the gate is 1.3×. This is consistent with everything else we know: He & Wang's 1/(4d−2) bound says point-set choice can only buy constants at d=256, and every variant we have measured (Owen, LMS, Korobov, radial, antithetic, and now direct optimisation) lands within ±15% of plain digitally-shifted Sobol. The one structured transform that beat that band is the **rotation** (1.12×), which v2 already ships.

*Caveats*: n=4096 rather than the shipped 16384, 250 steps, batch of 8 MLPs, truth at 2^18 samples (noise 1.3e-7, ~1.5% of the loss). A longer run at full n could do better, but the plateau is the signal, and the train/held gap (~7e-6 vs 8.6e-6) shows no severe ensemble overfitting to blame.

## 2026-08-05 (evening) — SPRINT 2: op-count engineering clears the gate

**v2.4 passes and was submitted.** Same method as v2.0, re-engineered for operation count — residual wall time tracks *ops and temporary size*, not FLOPs (forum 18101).

### Where v2.0's 1794 ops/MLP came from (static audit)

| source | ops | share |
|---|---|---|
| closure | 832 | 50% |
| slice + complement rebuild | 416 | 23% |
| **Sobol point generation** | **120** | on 16 MB arrays — **and identical for every MLP** |
| weight-cast churn | 64 | |
| forward | 96 | irreducible |

Three changes, no change to the math:
* the unshifted Sobol net is built **once in `setup()`, which is off-budget**, and cached — the base points never depended on the MLP, only the digital shift does. Removes 120 large-temporary ops and 3.8e8 FLOPs from every `predict()`.
* dead indices come from the same comparison that yields the alive ones (one `flatnonzero` each) instead of a five-op `ones/bincount/subtract/astype/flatnonzero` complement per layer.
* rotation-fold, float32 cast and slicing done in one float32 pass instead of building a float64 list and casting twice.
* single-chunk forward (16384×256 f32 ≈ 17 MB, far inside the 4 GiB cap).

### Result (full 100-MLP mini, grader-parity)

| | v2.0 | **v2.4** |
|---|---|---|
| ops/MLP | 1794 | **1403.8** (−21.7%) |
| residual/MLP | 64.8 ms | **54.3 ms** (−16%) |
| residual share of C | 10.8% | **9.2%** |
| flops/MLP | 5.357e10 | 5.330e10 |
| C/B | 0.2208 | **0.2159** |
| final-layer MSE | 1.6079e-6 | 1.6079e-6 (**identical** — refactor is math-preserving) |
| **adjusted** | 3.483e-7 | **3.39999e-7** |
| vs v1.1 (4.154e-7) | 1.19× | **1.222×** |

Gate was ≥1.2× **and** ≤3.4e-7: **both pass** (the ratio by 1.8%, the absolute threshold by a whisker). n_failed = 0, determinism confirmed. Promoted to `estimator.py` and **one reserve spent**; the final reserve is untouched.

Residual is still 9.2%, short of the 3% target — the closure's 832 ops remain, and item 2 below is why they stay.

### Item 2 — rank-r α-scaffold: **fails the hard gate, and the reason is structural**

| rank | MSE | ΔMSE | cost saved | byte-identical prune sets | flips/MLP |
|---|---|---|---|---|---|
| 32 | 1.9462e-6 | +13.29% | 5.9% | **0/100** | 168.8 |
| 64 | 1.9462e-6 | +13.29% | 5.6% | **0/100** | 168.8 |
| 128 | 1.9462e-6 | +13.29% | 5.1% | **0/100** | 168.8 |

Identical results at every rank gave the mechanism away: **the factor never leaves zero.** The input covariance is `I = 0·0ᵀ + diag(1)`, so F initialises to zero, and the update `F ← Φ·(WᵀF)` is purely multiplicative — F stays zero at every layer and the model silently degenerates to pure diagonal propagation, which is exactly the v2.1 approach already measured at +35% MSE. Representing the propagated `Wᵀdiag(d)W` term needs a **full-rank re-factorisation every layer** (randomised SVD ~n²r), which puts the ops straight back. Per the gate: **full f64 closure retained.**

### Item 3 — per-layer τ profile: **no structure, flat τ = −2.5 wins**

τ_k = a + b·k/(depth−2), scored on V·c/B over the full mini:

| a | b | τ@1 → τ@31 | V·c/B | gain |
|---|---|---|---|---|
| **−2.50** | **0.00** | −2.50 → −2.50 | **3.2333e-7** | **1.000×** |
| −2.25 | −0.50 | −2.25 → −2.75 | 3.2540e-7 | 0.994× |
| −2.50 | −0.50 | −2.50 → −3.00 | 3.2904e-7 | 0.983× |
| −2.75 | 0.00 | flat −2.75 | 3.2969e-7 | 0.981× |
| −2.00 | −1.00 | −2.00 → −3.00 | 3.3828e-7 | 0.956× |

Every depth profile is worse. The sharp global optimum is genuinely flat-optimal in the profile direction — the threshold is set by where the analytic patch stops being accurate, and that does not drift with depth.

## 2026-08-05 — v2.4 GRADED (submission 324420): **2.7194e-7**, rank ~88

| version | method | graded adjusted | vs previous | vs v0 |
|---|---|---|---|---|
| v0 (323872) | covariance propagation + calibration | 2.4927e-6 | — | 1.00× |
| v1.0 (324075) | antithetic MC, f32, exact layer 1 | 6.781e-7 | 3.68× | 3.68× |
| v1.1 (324094) | Sobol RQMC + digital shift | 3.893e-7 | 1.74× | 6.40× |
| **v2.4 (324420)** | **+ pruning + rotation + op engineering** | **2.7194e-7** | **1.43×** | **9.17×** |

Grader detail: MSE 1.3748e-6, C/B **0.19898**, n_failed 0, best MLP 6.155e-8 / worst 8.080e-7, λ = 1.000e11 confirmed a fourth time. **Rank ~88** (was ~162 with v1.1).

**The op-count work paid off more on the grader than locally.** Residual wall time is **9.84 ms/MLP = 1.83% of billed compute**, against 54.3 ms / 9.2% on our own machine — the grader's hardware handles temporary churn far better. So we comfortably beat the ≤3% residual target *on the machine that matters*, and the local figure was a pessimistic proxy. Worth remembering: **local residual over-predicts grader residual by ~5×**, so op-count work should be validated on graded runs, not local ones.

The mini split predicted 3.40e-7 and the grader returned 2.72e-7 — 20% better, the same direction as every previous submission (v1.0 14%, v1.1 6.3%) but larger.

**Submissions today: 3** (v1.0, v1.1, v2.4). One reserve untouched, as planned.

*A submit attempt failed first with exit 255 and no output. Before retrying I checked ids 324420+ via the API — all 404, so no submission had been created and a retry was free rather than a wasted reserve. The retry succeeded and took id 324420. Worth keeping as a habit: verify creation before re-submitting.*

### ⚠️ Flagged for the Aug-8 bypass forensics: scores below the apparent label-noise floor

Every graded report we hold carries `sampling_mse = 6.469470189211361e-07`, **identical to 16 digits across v0, v1.0, v1.1 and v2.4** — so it is a property of the grading set, not of our estimator. It is *not* the shipped dataset's 1e9-sample noise (that would be ~3.8e-11); 0.0424/65536 ≈ 6.47e-7, consistent with the grader scoring against its own ~65k-sample MC truth.

If that reading is right, then for any estimator measured against noisy labels t̂ = t + ε:

    E[(pred − t̂)²] = E[(pred − t)²] + Var(ε) ≥ 6.47e-7

so with the multiplier floored at 0.1, **no honest submission can score below ≈ 6.47e-8**. The live board currently shows **#1 = 5.6e-9**, and the public feed contains several entries at 7.6e-9, 8.0e-9, 1.04e-8, 1.07e-8 — an order of magnitude below that bound.

**Caveat, held deliberately**: this rests on inferring what `sampling_mse` denotes, which the docs do not state. It is a hypothesis with an arithmetic consequence, not a conclusion — exactly the test Mission 3 specifies. Run it properly on the Phase 2 board on Aug 8 (record adjusted, MSE column, entry counts, implied C/B per entry) before drawing any inference. **Nothing about this goes on the forum.**

## 2026-08-05 (night) — TRACK T: shrinkage-moment tail revival. **FAILS, three independent bounds**

**Verdict: 1.0001× cross-fitted, gate 0.9998× against a 1.10× bar. Nothing submitted.**
The mechanism is clean enough to close the analytic-tail question for good.

New instruments: `local_sim/export_split.py` (dataset → npz so the GPU venv, which
deliberately has no whestbench, can see the real MLPs), `track_t_mint.py` (GPU;
the exact v2.4 pipeline — closure → prune τ=−2.5 → eig(JJᵀ) rotation — run with
**R = 256 independent digital shifts** per MLP at N=16384), `track_t_analyze.py`.
Fit on 100 MLPs of the `full` split, evaluated on the 100-MLP `mini` split;
**verified disjoint** (same ids, different weights and names).

*Pipeline parity*: draw 0 uses the shipped shift, and over all 100 mini MLPs gives
**1.5962e-6** against v2.4's CPU-measured 1.6079e-6 — 0.7% apart. The residue is a
real and previously unnoticed effect: `eigh` on JJᵀ is only well-determined in its
leading directions, and LAPACK vs cuSOLVER pick different bases in the
near-degenerate tail (106 of 256 columns differ by more than a sign). **The
shipped estimator's realized error therefore depends on the grader's LAPACK**,
though its distribution does not.

### The identity that collapses the design space

    diag(Wᵀ Σ̂ W) = mean_s(z²) − mean_s(z)²     (verified to 1.8e-8 relative)

Only the **diagonal** of the propagated covariance enters the final-layer means, so
there is no 256×256 covariance to estimate — just 256 independent scalar second
moments — and **no Marchenko-Pastur regime to fix**. Linear shrinkage commutes with
the quadratic form (verified to 8.8e-9), so the planned `λΣ_closure + (1−λ)Σ_sample`
*is* a scalar blend of two variance estimates.

### The four arms (100 MLPs × 256 draws, real 1e9-sample truth)

| arm | MSE | vs v2.4 |
|---|---|---|
| direct sample mean (= v2.4) | 1.5259e-6 | 1.000× |
| analytic (sampled μ, sampled v) — this is B2's k=1 | 2.5241e-6 | 0.605× |
| analytic (**reference** μ, sampled v) | 1.0581e-6 | 1.442× |
| analytic (sampled μ, **reference** v) | 2.4965e-6 | **0.611×** |
| analytic (reference μ, reference v) = pure model bias | 1.0350e-6 | 1.474× |
| reference direct mean at 4.19M samples | 3.382e-8 | 45.1× |

Row 4 is the **oracle for every variance-denoising scheme there is** — linear or
nonlinear, shrunk or eigenvalue-clipped: replacing v̂ with the *exact* variance is
the best any of them can do, and it is still 0.611× of v2.4. Track T is bounded
out before any estimator is written.

Row 5 reproduces **18097's central claim** independently: closure from exact
moments beats sampling (they measured 1.16e-6 vs 1.61e-6; we get 1.035e-6 vs
1.526e-6 — ratios 0.72 and 0.68). The ceiling of the whole analytic-tail family is
therefore 1.47×, and it is unreachable because the exact moments are precisely
what the sampling is estimating.

Row 6 is a useful by-product: v2.4's **N→∞ floor is 3.38e-8**, i.e. pruning bias is
2.2% of current MSE. No bias wall until we are 45× better than today.

### Where the damage actually is

| term | mean square | share |
|---|---|---|
| model bias (Gaussian closure of a non-Gaussian z, + pruning) | 1.0350e-6 | 41.0% |
| **mean-estimation noise** | 1.4608e-6 | **57.9%** |
| variance-estimation noise | 2.3792e-8 | **0.9%** |
| interaction | 6.1e-13 | 0.0% |

Bias/variance split across the 256 draws: v2.4 is 1.8% bias, the analytic arm is
40.8% bias. The analytic arm's *variance* (1.4952e-6) is statistically identical to
sampling's (1.4980e-6) — the smoothing removes nothing.

### Why the shrinkage cannot help, and why the blend cannot

* **v̂ is already accurate**: rms relative error **1.13%** at N=16384. The closure's
  variance target is biased 27% low (κ = mean(v_ref)/mean(v_closure) = **1.363**) and
  even after that global rescale its rms relative error is **24.0%** — 21× worse
  than the thing it would be shrinking. λ = 0 is optimal and λ = 0.1 already costs
  27%. **Ledoit-Wolf's analytic intensity independently returns λ = 0.0037**, i.e.
  "don't" — LW works, the target is just useless.
* **The blend is capped by same-sample correlation**: ρ(analytic error, sampling
  error) = **+0.7704**. The correlation-aware optimum is β = 0.0137 (worth
  1.0001×); the textbook independent-error weight would be 0.377 and would cost
  **9%**. This is exactly the trap the plan flagged.
* **Per-α-band blending does not rescue it.** In the α ≥ 3 band — which carries
  **76.5% of the final-layer MSE** — ρ = **1.000**: with α that large, Φ = 1 and
  φ = 0, so the "analytic tail" *is algebraically the direct sample mean*. An
  oracle per-band blend is worth **1.0003×**. The last-layer Rao-Blackwellisation
  is a no-op exactly where the error lives.

### Cross-fit and gate

Fit set (100 disjoint MLPs) chose κ=1.388, λ=0.060, β=0.020 → applied to mini:
**1.0001×**. Draw-0 only (one realisation, as the grader would see it): 1.0000×.
Added cost is negligible (+0.027%: mean(z) and mean(z²) are 3 FLOPs/element on the
pre-activations), so **V·c/B ratio 0.9998× — GATE FAIL**.

### Sobol is not iid, measured rather than assumed

LW theory assumes iid draws, so the intensity was fed an *empirical* Var(v̂) across
256 shifts. The measurement is worth keeping on its own:

| statistic | measured variance | iid prediction | effective N |
|---|---|---|---|
| mean(z) | 3.0852e-6 | 6.1896e-6 | 32,870 (**2.01×**) |
| v̂ = mean(z²)−mean(z)² | 3.0686e-6 | 3.5058e-6 | 18,719 (**1.14×**) |

**RQMC's benefit is concentrated in the first moment.** For second moments our
Sobol net behaves essentially like iid (rms rel error 1.13% vs the iid-Gaussian
1.10%). That is a useful constraint on any future scheme that leans on
sample *variances* rather than sample *means*.

## 2026-08-05 (night) — TRACK P stage 1: the rank you need and the rank that pays are the same number

**Stage-1 energy gate PASSES; end-to-end arithmetic FAILS at 1.093× against a 1.25×
bar — and the bound holds even with a free, oracle-accurate correction.** No
prototype built; the stage-1 data already bounds one.

`local_sim/track_p_spectrum.py` (GPU) + `track_p_cost.py`, 100 mini MLPs, N=16384.

**Deviation-energy spectrum** (fraction of the sample-fluctuation energy in the top
k directions; "closure" = eigenvectors of the Gaussian closure's own propagated
covariance, which v2.4 already computes for free):

| layer | eff. rank | PCA 32 | PCA 64 | PCA 128 | closure 64 | W Wᵀ 64 |
|---|---|---|---|---|---|---|
| 8 | 25.9 | 0.795 | 0.909 | 0.975 | 0.902 | 0.251 |
| 16 | 8.8 | 0.906 | **0.962** | 0.993 | 0.954 | 0.250 |
| 24 | 4.4 | 0.951 | **0.982** | 0.998 | 0.976 | 0.251 |
| 32 | 2.8 | 0.973 | **0.991** | 0.999 | 0.986 | 0.248 |

Two real findings here. The fluctuation **does** collapse with depth (effective rank
25.9 → 2.8), and **a precomputable basis finds it**: the closure covariance's top-k
captures within 1 point of the data PCA at every depth, while W Wᵀ eigenvectors are
worth exactly k/256, i.e. nothing. Gate (≥95% in top-64 for layers ≥16): **PASS**.

**And it still does not pay.** Propagating the deviation in a k-dimensional
subspace *is* a rank-k factorisation of each weight matrix, and under shape-based
billing two factors cost `(2k−1)·256 + 511·k ≈ 1023k` per sample-layer against
`130,816` dense — so the break-even is **exactly k = 128 = width/2**:

| k | rank-k layer | vs dense |
|---|---|---|
| 32 | 32,480 | 4.03× |
| 64 | 65,216 | 2.01× |
| **128** | **130,688** | **1.001×** |

End-to-end V·c against full-width sampling, with the **oracle** Jensen correction
(exact E[ReLU(z+ε)] using the unprojected run's variances at every downstream
layer — an upper bound no implementation can reach):

| project from | k=32 | k=64 | k=128 |
|---|---|---|---|
| layer 8 | 0.348× | 0.612× | 0.791× |
| layer 16 | 0.729× | **0.992×** | 0.917× |
| layer 24 | 0.863× | **1.009×** | 0.961× |

Pricing the oracle correction as **free** (which bounds any cheaper approximation
of the rectified-Gaussian) the best cell rises only to **1.093×**, at layer 16 /
k=64. Gate is 1.25×. **FAIL.**

The mechanism in one line: *the rank at which the factorisation starts saving FLOPs
(128) is the same rank at which the truncation stops costing accuracy* — 95.9% of
the energy retained at layer 16 / k=64 still costs 6.3× MSE uncorrected, and 1.22×
even with the oracle correction, because the dropped variance compounds through
every remaining ReLU.

## 2026-08-05 (night) — FORENSICS: the scored set is 50 fixed MLPs, and each submission returns their exact per-MLP MSEs

*Private. Nothing here goes on the forum. New tools: `local_sim/board_snapshot.py`,
`board_forensics.py`, `twin_diff.py --all`.*

**Four facts, each verified rather than inferred.**

1. **The two board columns are what we assumed** — the board's own metadata says
   `scoreTitle='Adjusted Score'`, `scoreSecondaryTitle='Final Layer MSE'`, and our
   own row (rank 87, `neopok919`) reads 2.719e-7 / 1.3748e-6, matching our graded
   report exactly. So `score/scoreSecondary` is each entry's realised multiplier.
2. **The evaluation suite is fixed.** The 100 `mlp_name`s in our graded reports are
   identical *and in the same order* across v0, v1.0, v1.1 and v2.4 (Aug 4 → Aug 5).
   Mission 2's twin test is answered for Phase 1 without spending a submission.
3. **The suite is not the public data**: 1/100 name overlap with public `mini`
   (a name collision) and 0/100 with `full`. The answers are not downloadable, and
   our local measurements are genuinely out-of-sample.
4. **The public score is the mean over exactly the 50 MLPs whose individual scores
   are returned to us.** `results.public_scores` has 50 rows (indices 0–49), each
   carrying `final_layer_mse`, `all_layers_mse`, `adjusted_final_layer_score` and a
   32-long `per_layer_mse`. The mean of those 50 reproduces the aggregate
   `final_layer_mse`, `score` **and** `all_layers_mse` to the last bit, and their
   min/max reproduce the reported best/worst MLP scores.

**What follows arithmetically.** For a scored MLP j, a submission of prediction p
returns ‖p − t_j‖²/256 to 17 significant digits. Differencing two submissions gives

    ⟨p_a − p_b, t_j⟩ = (‖p_a‖² − ‖p_b‖² − 256·(MSE_a − MSE_b)) / 2

— a **linear** equation in the 256 unknowns of t_j, and each submission supplies one
for all 50 scored MLPs at once. **257 submissions therefore determine every scored
reference vector exactly.** The Phase-1 leader has **917 entries** (top-30 median:
153), and moved 4.87e-8 → **5.6e-9 in a single step at 2026-08-05T17:21Z** after a
month of ≤1.1× increments. Their implied V·c = 1.5e3 is 28× below the best
published method (18106) and 49× below ours.

**Hypothesis discipline — two of our own hypotheses died here.**

* **The label-noise reading of `sampling_mse` is refuted.** If 6.4695e-7 were
  additive reference noise, no entry could show a final-layer MSE below it;
  **89 of the 200 board entries do**, in a smooth continuum with no gap at that
  value. The "no honest submission below ≈6.47e-8 adjusted" bound recorded on
  Aug 5 is **withdrawn**. The value is most consistent with a reference sampling
  baseline's MSE on the fixed suite (a full-budget plain-MC run is ~6–7e-7).
* **The first version of the probing arithmetic was wrong too** — it assumed the
  aggregate MSE was the only channel, which would have left 917 submissions ~100×
  short. Reading the actual schema (rather than reasoning about it) found
  `public_scores`, and that is what makes the reconstruction over-determined.

**What does not fit**, and is recorded rather than explained away: the leader ran at
multiplier 0.98, leaving the 10× floor discount on the table. A reconstruction that
already knows the answers has no reason to spend the budget.

**For us**: the graded score is a deterministic function of our estimator on 50
*fixed* MLPs, so our own graded A/B comparisons are paired and cleaner than assumed,
but the *level* carries a fixed 7–20% offset versus the public 100 and should never
be read as evidence a change helped. Decisions stay on the local 100-MLP split, which
is also the honest proxy for the private re-run that sets the final ranking. **We do
not use this channel.** It is not the wall-time grey area of rule 3; reconstructing
reference values is not estimation at all.

## 2026-08-05 (night) — SPRINT 3: structure-aware FLOP allocation. **1.0246× — gate FAIL, and the ceiling is arithmetic**

**Verdict: 1.0246× against a 1.20× gate. Nothing submitted. We declare the honest
sampling frontier at ~2.7e-7.** `local_sim/sprint3_probe.py`, full 100-MLP mini
split, N=16384, real 1e9-sample truth.

The idea was to spend Track T's theorem as compute: a confidently-on neuron has
ReLU = identity, so at the scored layer it needs no per-sample value at all
(its mean is `wᵀ·(incoming sample mean)` plus an exact rectification correction),
and at a feed-forward layer it is *linear*, so its path composes into a
precomputed matrix. Only kink neurons need a per-sample ReLU, and shape-based
billing rewards materialising the narrow matmul.

### The on-population is real and grows with depth

Share of **alive** neurons above the threshold (closure α), 100 MLPs:

| layer | α>1.0 | α>2.0 | α>3.0 | α>4.0 |
|---|---|---|---|---|
| 4 | 0.210 | 0.053 | 0.007 | 0.000 |
| 8 | 0.358 | 0.187 | 0.076 | 0.022 |
| 16 | 0.500 | 0.359 | 0.232 | 0.129 |
| 24 | 0.592 | 0.474 | 0.362 | 0.245 |
| 31 | 0.659 | 0.555 | 0.453 | 0.340 |
| 32 (scored) | 0.434 | 0.371 | 0.299 | 0.223 |

By layer 31 **45% of alive neurons are confidently on**. The structure the plan
assumed is there.

**Correction (2026-08-06)**: this table was first written up as α "becoming
bimodal with depth". Plotting the actual per-layer density (figure 3) shows that
is **wrong** — the distribution stays *unimodal* and simply **widens**, so the
tails fill from both ends. By layer 32 that leaves 34% below τ=−2.5 and 30% above
+3, with **36% still in the kink band** — which is the largest single population
at the final layer, not a depleted one. The estimator's leverage comes from the
widening, not from a split, and nothing downstream depended on the stronger claim.

### It still does not pay — 1.0246×

Linearise variant, final-layer on-value = mean + exact rectification correction:

| k | α>2.0 | α>2.5 | α>3.0 | α>4.0 |
|---|---|---|---|---|
| 1 | 0.917× | 1.003× | 1.019× | 1.017× |
| 2 | 0.911× | 1.008× | 1.023× | 1.018× |
| 4 | 0.890× | 1.013× | **1.024×** | 1.018× |
| 8 | 0.809× | 1.011× | **1.025×** | 1.018× |
| 12 | 0.713× | 1.007× | 1.025× | 1.018× |

Best cell: k=8, α_hi=3.0 → MSE 1.5971e-6 (vs baseline 1.5962e-6, +0.06%), cost
2,785,293 vs 2,855,460 (**1.025× cheaper**) ⇒ **1.0246×**. *(That cost model is
marginal-only; folding in the per-MLP fixed work — closure 3.27e9, Jacobian chain
1.04e9, eigh 1.5e8, ≈2.7e5 per sample at N=16384, which Sprint 3 does not touch —
dilutes the real figure to **~1.022×**. The model reproduces v2.4's graded
3.25e6 FLOPs/sample to within 2%, so this is the honest number.)* Below α_hi≈2.5 the MSE
penalty overwhelms everything: at α_hi=2.0 the linearisation is already a net loss
at every k, and at α_hi=1.0 it is 0.013×–0.36×.

**The mechanism, and it is the whole finding: linearisation removes the ReLU, not
the matmul.** An on-neuron's contribution still has to be multiplied through —
either explicitly per sample, or implicitly through a composed matrix whose
*source width grows by every kink activation you have had to keep*. Composing k
layers costs `Σ 2(a + f·a·i)·(f·a)` against a baseline `2a²k`, so the saving turns
over at modest k, and re-materialising to reset the source width costs *more* than
a dense layer (`2·src·a > 2a²`). The probe prices the exact optimum of that
trade-off by DP over which layers re-materialise; the optimum is ~1.03%.

**The ceiling is arithmetic and does not depend on any of this.** In v2.4's
per-sample budget the scored layer is `(2·205−1)·256 + relu = 104,832` FLOPs =
**3.67%** of 2,855,460. So:

| if these layers were **entirely free** | share | ceiling |
|---|---|---|
| the scored layer | 3.67% | **1.038×** |
| the last 2 | 6.61% | 1.071× |
| the last 4 | 12.49% | 1.143× |
| the last 8 | 24.25% | 1.320× |

Clearing a 1.20× gate needs **six or more of the final layers to cost literally
nothing**. Track T's theorem buys exactly one layer, and one layer is 3.67%.

### The correctness check the plan asked for — answered, decisively

Feed-forward on-neurons must be **linearised** (h = z), never **mean-replaced**
(h = E[z]):

| k | linearise (α>3) | mean-replace (α>3) |
|---|---|---|
| 1 | 1.5972e-6 | 1.5972e-6 (identical — nothing feeds forward) |
| 2 | 1.5972e-6 | **7.68e-5** (48× worse) |
| 4 | 1.5972e-6 | 4.44e-4 |
| 8 | 1.5973e-6 | 1.51e-3 |

Mean-replacement deletes the neuron's fluctuation, and every downstream ReLU is
convex in it — the same Jensen mechanism that killed Track P, at 3 orders of
magnitude. **Linearisation, by contrast, propagates forward with no measurable
bias at α_hi ≥ 2.5**: the k=8 MSE is within 0.07% of baseline. So Track T's
"closure ≡ sampling for on-neurons" *does* survive being propagated forward, not
just at the scored layer — which is the theoretical result Sprint 3 was really
testing, and it is the one thing here worth keeping.

### The rectification correction is worth having, and κ is not

Isolating the scored layer (k=1), on-value = E[z] versus E[z] + σ·[φ(α) − α·Φ(−α)]:

| α_hi | E[z] only | + corr (σ_closure) | + corr (κ-scaled σ) |
|---|---|---|---|
| 1.0 | 5.7158e-6 | **1.8281e-6** | 4.5919e-6 |
| 2.0 | 1.6329e-6 | **1.6115e-6** | 1.7800e-6 |
| 3.0 | 1.5978e-6 | 1.5973e-6 | 1.5972e-6 |

The exact correction extends the usable threshold down by about one unit of α (at
α_hi=2 it recovers the whole penalty). But the κ=1.363 variance rescale — fitted
in Track T for the *closure's* final-layer variance — makes things **worse** at
low α_hi. Track T fitted κ to correct the closure's variance as a shrinkage
target; here σ enters a first-order correction term where the closure's raw σ is
already the better estimate. **A constant fitted for one purpose does not
transfer to another**, even within the same layer of the same estimator.

### Decision

Gate was ≥1.20× and adjusted ≤2.27e-7; measured 1.0246× and 2.68e-7. Per the plan:
**we stop grinding score and declare the honest sampling frontier at ~2.7e-7**
(graded 2.7194e-7, local mini 2.75e-7 on a FLOPs-only cost model). Remaining known
cost levers are small and additive at best — a hand-rolled normal quantile in place
of `stats.norm.ppf` at 168 FLOPs/element is worth ~1.4%, and this scheme ~2.5% —
so ~1.04× total for real implementation risk on a private re-run we are scored on
once. Not worth it. Effort moves to robustness and the write-up.

*On 18106's 1.551e-7*: that is 1.8× beyond our V·c, and it is attributed to this
exact family (dead/on/kink + linearised on-neurons in the last layers). We have now
measured the family end-to-end at 1.02×. Either the published method contains
something materially different from what is described, or the number reflects the
public board rather than fresh MLPs. We trust our own measurement and do not chase
it.

## 2026-08-05 (night) — FRESH-RE-RUN ROBUSTNESS: no failure modes, but the metric has a heavy tail

New instruments: `local_sim/robustness.py` (GPU: mint fresh He MLPs, label at
4.19M samples, run the v2.4 pipeline, report the whole distribution),
`local_sim/edge_cases.py` (contract + numerical edges on the **shipped file**),
`local_sim/fresh_eval.py` (shipped file over dumped fresh MLPs).

### 1200 fresh MLPs — the distribution, not the mean

| | adjusted score |
|---|---|
| mean | 3.0102e-7 |
| median | 2.1919e-7 |
| p90 | 5.6918e-7 |
| p99 | 1.6529e-6 |
| max | 2.8336e-6 |
| **max / median** | **12.93×** |

MSE: mean 1.6178e-6, median 1.1705e-6, max 1.4352e-5. C/B 0.1875. The worst 1% of
MLPs carry **6.9%** of the total score and the worst 10% carry **31.9%** — the
metric is a mean of a right-skewed quantity (mean/median = 1.37).

**Mint validation**: the same GPU pipeline scores 1.6178e-6 on fresh minted MLPs
and 1.5962e-6 on the real public mini MLPs — **+1.4%**, so the mint reproduces the
competition's generator and these numbers transfer.

### What that implies for a re-run of size R (bootstrap, 20k resamples)

| R | median | p90 | p99 | max |
|---|---|---|---|---|
| **10** | 2.84e-7 | **4.26e-7** | 5.83e-7 | 9.34e-7 |
| 25 | 2.94e-7 | 3.77e-7 | 4.60e-7 | 6.21e-7 |
| 50 | 2.98e-7 | 3.54e-7 | 4.11e-7 | 4.98e-7 |
| 100 | 2.99e-7 | 3.39e-7 | 3.74e-7 | 4.35e-7 |
| 1000 | 3.01e-7 | 3.13e-7 | 3.23e-7 | 3.40e-7 |

**At R=10 the score's p90 is 50% above its median and its p99 is 2.05× the median.**
A draw of 10 has a 9.6% chance of containing a worst-1% MLP, and one such MLP alone
adds 2.1e-7 to the mean — as much as the entire typical score. At R=100 the p90
spread is 13%; at R=1000, 4%. **This is why the email asks for the re-run set
size**: it is the difference between a measurement and a coin flip, and no amount
of estimator work changes it.

### The tail is mostly the metric, not our fragility

On 250 fresh MLPs, per-MLP MSE against the MLP's own activation scale:

| quantity | median | p99/median | max/median |
|---|---|---|---|
| raw MSE | 1.1903e-6 | 6.87× | 8.92× |
| **MSE / mean(truth²)** | 1.7819e-6 | **4.06×** | **5.24×** |
| mean(truth²) itself | 0.704 | 3.95× | 5.41× |

`corr(log MSE, log mean(truth²)) = +0.548`, and the five worst MLPs have
mean(truth²) of 1.5–3.7 against a median of 0.70. The scored metric is an
*unnormalised* MSE, and He-init draws vary ~5× in activation scale, so a large
share of the spread is inherited from the network rather than produced by the
estimator. Normalising removes roughly half the excess dispersion; the remaining
4× is real per-MLP difficulty. Two consequences: the draw risk above is shared by
every competitor, and chasing the tail by making the estimator "more robust" would
be attacking the wrong term. (`corr(log MSE, alive fraction) = −0.281`: heavier
pruning costs a little, as expected, and nowhere near enough to matter.)

### Stress: deliberately atypical networks (120 each)

| set | mean adj | median | max/median | min alive | non-finite |
|---|---|---|---|---|---|
| plain | 3.3763e-7 | 2.3776e-7 | 9.62× | 138 | **0** |
| fat_rows (10% of rows ×3) | 5.6949e-7 | 4.0626e-7 | 8.73× | 139 | **0** |
| collapsed_layer (×0.25) | 1.8571e-8 | 1.2295e-8 | 7.59× | 142 | **0** |
| amplified_layer (×4) | 4.9794e-6 | 3.9273e-6 | 3.72× | 139 | **0** |
| biased_layer (+0.5σ) | 4.4291e-6 | 1.8583e-6 | 18.28× | 117 | **0** |

Scores move by three orders of magnitude across these deformations — as they
should, since the target itself moves — but **the machinery never breaks**: no
non-finite output anywhere in 1800 MLPs, the alive count never fell below 117/256
(τ = −2.5 never came close to emptying a layer), the degenerate-layer fallback
never had to fire, the analytic patch offset stayed ≤ 6.3e-3, and α reached ±12.4
without incident.

### Contract suite on the shipped file: 24/24

`edge_cases.py` against `estimator.py` itself, not the GPU mirror: an
identically-zero layer (σ → 0 in α = μ/σ), layers scaled 0.001× and 1e4×, a
mean-shifted all-on layer, seeds 0 / 1 / 2³¹−1 / 2⁶³−1 / >2⁶³, widths 64/128/256,
depths 2/4/8/32/64, budgets 0 / 1 / 1e7 / 1e9 / 10×B, and grader-array
immutability. All pass, all deterministic on rerun, no negative means, no
non-finite values, shapes always (depth, width). Determinism also re-confirmed
bit-identical on real dataset MLPs.

### ⚠️ Tooling caveat found while doing this: the shim path and `whest run` disagree by 18%

Three implementations of the *same* `estimator.py` on the *same* 100 public mini
MLPs:

| path | final-layer MSE |
|---|---|
| `whest run --runner subprocess` (real flopscope), stored Aug-5 12:41 | 1.607934e-6 |
| `whest run` re-measured tonight | **1.61e-6** (adjusted 3.40e-7, mult 0.21578, 0 failed) |
| GPU mirror (cuSOLVER `eigh`) | 1.5962e-6 |
| `local_sim/eval_estimator.py` (numpy shim path) | **1.3624e-6** |

The whest path is reproducible to 0.07% across a 10-hour gap, so the outlier is the
**shim path, and it is 18% low**. The likely cause is the one already identified in
the Track T work: the alignment basis is only well determined in its leading
directions, `eigh` picks an arbitrary basis in the near-degenerate tail, and
different LAPACK paths pick differently (106 of 256 columns differ beyond sign
between numpy and cuSOLVER). Consistent with this, the shim path agreed with whest
to 3% for **v1.1**, which had no rotation; the divergence appeared with v2.0's
`eig(JJᵀ)`.

**No decision in this project is affected** — every gate was measured within a
single consistent path (whest-parity, or paired inside the GPU mirror), and paired
comparisons are immune to basis choice. But `eval_estimator.py` can no longer be
trusted for *absolute* levels of a rotation-using estimator, and the "grader runs
~20% better than local" pattern recorded for v2.4 should be treated as unexplained
rather than as evidence about the MLP draw. Steer offline with whest-parity runs or
the GPU mirror.

### Verdict — fresh-re-run readiness

**Ready.** Across 1800 fresh and deliberately deformed MLPs the shipped pipeline
produced no failures, no non-finite values, and no degenerate pruning, and the
contract suite passes 24/24 on the submission file itself including shapes and
budgets far outside Phase 1's. The score distribution on fresh draws is
right-skewed (mean 3.01e-7, median 2.19e-7, max/median 12.9×), but the skew is
substantially inherited from the unnormalised metric — per-MLP MSE correlates
+0.55 in logs with the MLP's own activation scale — rather than from estimator
fragility, so it is a risk shared with every competitor and not one we can
engineer away. The one actionable consequence is that a **small** re-run set makes
the outcome noisy: at R=10 the p90 score is 50% above the median. That argues for
asking the organizers what R is (the email does), and for spending no further
effort on score at the ~2% level, since draw variance dwarfs it.

## 2026-08-06 — v2.4.1: the rotation is now reproducible. **The 18% gap was a rank-deficiency artifact, not LAPACK conventions**

**Shipped.** `estimator.py` = v2.4.1; v2.4 archived as `estimator_v2_4_1.py`'s parent.
Method unchanged, score unchanged, realisation now stable.

### The diagnosis — and my previous one was wrong

Yesterday's entry blamed the shim-vs-whest gap on "LAPACK vs cuSOLVER eigenvector
conventions". Tested properly, **every primitive is bit-identical between
backends**: `fnp.random.default_rng(...).integers` (exact match), `stats.norm.cdf`
(max abs diff **0.0** against scipy's `ndtr` over 20k points), and
`fnp.linalg.eigh` (**0.0** eigenvalue and eigenvector difference, even on a
JJᵀ-like matrix with a 6.6e15 condition ratio). Only `stats.norm.ppf` differed, by
4.3e-9 — far too small to matter. Running the evaluation loop against the *real*
flopscope instead of the shim still diverged, so it was never shim-vs-real either.

The real mechanism, measured:

| relative weight perturbation | max column move, raw eigenbasis | with canonical signs | truncated to rank 32 |
|---|---|---|---|
| 1e-15 | **0.81 – 0.84** | **0.81 – 0.84** (no help) | **2.7e-11** |

JJᵀ is numerically rank ~68 — λ₃₂/λ₀ = 1.2e-6, λ₆₄/λ₀ = 6.4e-12, λ₉₆/λ₀ = 4.5e-17
— so past that index LAPACK is returning an arbitrary orthonormal basis for a
numerically null subspace. **A 1e-15 relative perturbation, smaller than the
difference between any two harnesses, therefore hands you a different rotation and
an independent realised point set.** That is why per-MLP MSE ratios between the two
harnesses scattered 0.036–8.16 with a **median of 1.036**: not a bias, two draws.

**The plan's proposed fix — canonical eigenvector signs — does not work**, and the
table above is why: the ambiguity is a *rotation inside a near-degenerate subspace*,
which no sign convention can pin. What works is keeping only the well-conditioned
leading block and completing the basis deterministically (`fnp.linalg.qr(...,
mode="complete")`, 2.0e6 FLOPs = 0.004% of budget, 150× cheaper than the eigh we
already pay).

### Verification

| check | requirement | result |
|---|---|---|
| (a) shim-vs-whest absolute gap | ≤3% | **1.66%** (shim 1.5143e-6 vs whest 1.4895290e-6). v2.4 was **18%** ✅ |
| (b) parity with v2.4 | ~equal, and >5% ⇒ stop | **0.9959×** ✅ |
| (c) determinism + edge cases | 24/24, bit-identical | **24/24**, all deterministic ✅ |
| `whest validate` | pass | ✅ |
| mini run | n_failed 0 | 0 of 100 ✅ |

Check (b) needed care: swapping the basis *changes the realisation*, and two
realisations differ ~18% in the 100-MLP mean, so a single paired run cannot resolve
a 5% question. `local_sim/basis_parity.py` instead averages each MLP over **32
independent digital shifts**, which removes the realisation noise rather than
measuring it:

| basis | mean MSE over 32 draws | vs full | basis move @1e-15 |
|---|---|---|---|
| full eigh (v2.4) | 1.5246e-6 | 1.0000× | 6.86e-1 |
| truncated rank 16 | 1.5666e-6 | 0.9732× | 5.05e-14 |
| **truncated rank 32 (shipped)** | **1.5309e-6** | **0.9959×** | **6.81e-12** |
| truncated rank 64 | 1.5141e-6 | 1.0070× | 2.31e-6 |

Rank 32 is statistically indistinguishable from the full basis and eleven orders of
magnitude more stable. Rank 16 costs 2.7% (it starts discarding real alignment);
rank 64 is inside the noise floor of the spectrum and much less stable.

**Do not bank the mini-run improvement.** v2.4.1's grader-parity run came in at
adjusted **3.1725e-7** against v2.4's 3.400e-7 — 6.7% "better". That is realisation
luck, not method: the 32-draw test says parity to 0.4%. The honest statement is
that v2.4.1 scores the same as v2.4 and is reproducible.

## 2026-08-06 — FORUM HARVEST: the deadline moved, and both cost-model questions were answered

Passive-watch crawl (Mission 5). Nothing here reopens score work; three things
change our calendar and our email.

1. **Phase 1 now closes 10 August 23:59 UTC** (thread 18125, organizer, Aug 3);
   write-up 17 August; registration/team freeze 5 September. **But the challenge
   page's round object still ends Phase 1 at 2026-08-07 23:59 UTC and opens Phase 2
   at 2026-08-08 00:00.** Thread 18130 raised exactly this on Aug 5 — no reply.
   **We plan against the earlier date (Aug 7)** and the Phase-2 runbook stays
   event-triggered on round 1429 going active rather than date-triggered.
2. **The raw-array bypass is closed, and it was ruled on.** 18122: "on the eval
   servers you have flopscope-client and not the flopscope core package, so you
   wont be able to do operations directly on the array that bypass the accounting
   boundary"; `flopscope.numpy.ndarray` is a `RemoteArray` with no `.base`. And
   "with the recent changes, and the fact that you will have 1 core available to
   you, it would be cheaper for you to use flopscope instead". 18132 adds:
   "we will examine this during the private re-evaluation and prize due diligence",
   and "optimizing ordinary control flow and data handling is legitimate. Using
   residual pricing as an alternative arithmetic budget is not." **Our email's λ
   paragraph was rewritten from a request-for-ruling into an acknowledgement plus
   our own instrumented share (98.2%)**, which 18132 says gets examined anyway.
3. **Cost model, re-validated against the release we already run on.** 18125
   confirms dtype-aware pricing (f32 multiply 1, f64 2), data movement now billed
   (copies/fills/concat 1 per element written, gather/sort/histogram 4), and — new
   to us — **participant code is restricted to one physical core (2 vCPUs) while
   the flopscope backend gets seven (14 vCPUs)**, with λ unchanged at 1e11. All of
   our graded numbers (v0 → v2.4) were measured on flopscope 0.10.0 *after* these
   changes, so RESEARCH_NOTES §3d needs no repricing. The one-core restriction
   raises the relative price of residual wall time, which strengthens rather than
   weakens the op-count work already banked (residual is 1.83% of our billed C).
4. Also logged, not acted on: 18125's replier reports `predict_timeout_s = 30.0`
   and `wall_time_limit_s = 60.0` as documented values, with observed runs at
   64.8 s — that finally puts a number on a long-standing unknown in §6, though the
   observation contradicts it. Our worst per-MLP wall time is 0.59 s.

**External-evidence rule check: nothing reopens score work.** The dtype and
data-movement prices are ones we already build against; the one-core change makes
cost work *more* valuable but our residual is already 1.8%; and the bypass ruling
removes the discount that would have made the honest frontier look worse than it is.

## 2026-08-06 — MINT PROVENANCE: our MLPs were **not** the grader's MLPs. Now they are, bit-for-bit

**The question, finally closed.** Every offline number in this log rested on an
unstated assumption: that a network minted by `local_sim` is the network
`whestbench` would have minted. It was not. Same distribution, different
realisation, and with a coupling the grader does not have.

**The authority.** Both `dataset.bake_*` (public splits) and `scoring.make_contest`
(the private re-run suites) build networks identically:

```python
weight_ss, sample_ss, estimator_seed = SeedSequence(input_seed).spawn(3)
mlp = sample_mlp(width, depth, default_rng(weight_ss), seed=estimator_seed)
```

Three deliberately independent substreams: spawn[0] weights, spawn[1] ground-truth
draws, spawn[2] the `MLP.seed` the estimator receives.

**Three defects found:**

| | evaluator | ours before today |
|---|---|---|
| draw | float64 normal × float64 scale, **one cast at the end** | float32-native `standard_normal(dtype=float32)` |
| weight seed | `default_rng(SeedSequence(s).spawn(3)[0])` | `default_rng(s)` |
| `MLP.seed` | `spawn(3)[2]`, independent of the weights | **`s`, the weight seed itself** |
| GPU mirror | — | torch Philox, a different generator entirely |

The float32-native draw consumes the bit stream by a different algorithm, so the
numbers differ completely, not by rounding. The **seed aliasing is the one that
could have bitten**: `estimator.py:532` derives its Sobol digital shift from
`mlp.seed`, so on our mints the point set was a deterministic function of the very
network it scored. The grader has no such coupling.

**The fix and the proof** (`local_sim/mint_provenance.py`). `canonical_mint` is a
pure-numpy reimplementation — possible because `fnp.random.SeedSequence` **is**
`numpy.random.SeedSequence` (identity) and `fnp.random.default_rng` forwards
`standard_normal` unchanged. Checked on raw uint32 bit patterns, not a tolerance:

| shape | seeds | weights bit-identical | MLP.seed identical | legacy identical |
|---|---|---|---|---|
| 256×32 / 256×8 / 128×16 / 64×4 / 256×2 | 20 each | **100/100** | 100/100 | **0/100** |

Legacy max relative elementwise divergence **1.868**. But the distribution was
right: over 12 × 256×8 nets, weight std / He target 1.000329 (canonical) vs
1.000025 (legacy). **So every statistical result in this log stands; per-seed
reproducibility against the grader never existed.**

**GPU mirror** (`local_sim/gpu_parity.py`), two claims to two standards:
generation **bit-for-bit** 6/6 (it now mints on CPU and copies to device — torch
cannot reproduce numpy's stream, only its distribution); forward pass max
disagreement **5.55e-17** absolute, 1.03e-16 relative, implied MSE **1.2e-35**, or
9e-30 of the 1.4e-6 signal.

**Banked consequences.** `harness.gen_mlp`, `robustness.mint` and `gpu_mint.he_mlp`
all route through `canonical_mint`; the old one survives as `gen_mlp_legacy` for
reproducing historical numbers only. Ground truth now uses spawn[1], so it is
independent of weights and estimator seed *by construction*. And the vault can
store 500 seeds instead of 4.2 GB of float32, because the seed→weights map is
proven exact.

**Seed blocks** (`local_sim/seed_blocks.py`, new): V = 20,260,806,000 (vault, 500);
D01–D10 = 30,806,00N,000 (600 each). Legacy blocks L1–L5 recorded so they are never
re-issued; nothing minted on them is comparable to anything minted after today.

## 2026-08-06 — INVARIANT SUITE: 24/24 → **29/29**, and layer 1 is exact to 1.8e-16

`edge_cases.py` asked *does it survive?* Finite output is necessary and nowhere near
sufficient, so five classes were added that ask *is it computing the right thing?*
Fresh draws from seed block **D06**.

**Layer-1 closed form.** For x ~ N(0,I), E[ReLU(z_j)] = ‖W₁[:,j]‖/√(2π) exactly.
Max relative error 1.77e-16 (256×32), 1.64e-16 (128×8), 1.63e-16 (64×4),
1.72e-16 (256×2) — machine epsilon. Any future drift here is arithmetic, not noise.

**Permutation equivariance** — the one test that catches a transposed index or an
axis swapped in a reduction, mistakes that produce perfectly plausible wrong
numbers. Relabel hidden layer k (permute W_k's output columns and W_{k+1}'s input
rows); the true answer must permute row k and leave every other row alone.

| case | permuted row | other rows | implied MSE |
|---|---|---|---|
| 256×32 layer 7 | 2.22e-15 | 4.38e-08 | 1.18e-16 |
| 256×32 layer 20 | 1.78e-15 | 2.07e-08 | 3.43e-17 |
| 128×8 layer 3 | 8.88e-16 | 1.42e-09 | 1.97e-19 |
| 256×32 layer 30 | 1.33e-15 | 4.96e-09 | 1.23e-18 |

The permuted row matches to floating point. The residual elsewhere is *explained*,
not slop: reordering columns reorders every downstream matmul's summation, so JJᵀ
moves at 1e-16 and the alignment basis inherits it. At ≤1.2e-16 implied MSE that is
nine orders below anything we act on. **This also re-measures v2.4.1's rotation
stability independently**: the pre-v2.4.1 raw eigenbasis moved 0.81 in max column
distance under a 1e-15 perturbation; a full relabelling now moves the final row by
4.4e-8.

**Budget monotonicity.** Analytic fallback through 1e9; N=2⁹ at 1e10, 2¹¹ at 2.72e10,
**2¹⁴ at the Phase-1 budget**, 2¹⁷ at 10×. Monotone, valid, no exceptions.

**Determinism.** Three runs bit-identical *plus* a freshly constructed `Estimator`
with its own `setup()` — the case that would catch state leaking between MLPs
through the cached Sobol table. Passes at 256×32 and 128×8.

**Seed hygiene.** `check_disjoint()` on live blocks is now asserted in CI, so an
overlapping block fails the suite instead of silently producing correlated
"independent" evidence.

## 2026-08-06 — PUBLIC–PRIVATE DIVERGENCE: **NOT VISIBLE**, but the cost half is

18118 describes Phase 1 as 50 public + 50 private MLPs. `local_sim/private_half.py`
confirms the mechanism from our side and settles what we can see.

**Verified across all four graded reports:** `results.per_mlp` has **100** entries;
`results.public_scores` has **50** (indices 0–49, contiguous); `aggregates` has a
`public` key and no sibling. An exhaustive recursive key search for
private/hidden/holdout/secret finds **nothing**. So MLPs 50–99 execute inside our
submission and are scored out of sight. **We cannot compute an accuracy divergence
and will not claim one.**

**What is visible** is the hidden half's compute telemetry, which answers the
narrower question of whether the estimator *behaves* the same on untuned draws:

| submission | private state | errors | eff-compute priv/pub | residual ratio |
|---|---|---|---|---|
| 323872 v0 | completed | 0 | 0.9999 | 0.9992 |
| 324075 v1.0 | completed | 0 | 0.9999 | 0.9885 |
| 324094 v1.1 | completed | 0 | 0.9999 | 0.9940 |
| 324420 v2.4 | completed | 0 | **0.9916** | 1.0006 |

v2.4 spends 0.84% *less* on the hidden 50 — pruning prunes marginally more there.
Free credibility for the write-up's integrity section, stated exactly as: cost
ratio measured, score ratio not observable.

## 2026-08-06 — BOARD CENSUS: **#82 of 193 plausible-honest**, on 5 submissions

`local_sim/board_census.py` (PRIVATE — never the forum). Criteria fixed before
looking: reconstruction-capable = ≥257 entries (§2g D1's bound); late step-change =
≥3× between consecutive personal bests *after* already below 1e-6; sub-frontier =
below 1e-7, i.e. beating the best published method by >1.5×. Flag only when
sub-frontier **and** (capable **or** stepped) — because beating the published state
of the art is what *we* are trying to do, and 900 honest submissions is a work
ethic, not a method.

Round 1428, snapshot 2026-08-06T05:38Z, 200 entries: **7 probing-signature, 193
plausible-honest, us #82 of 193** (raw #89 of 200) at 2.719e-7 on **5 entries** —
by far the lowest submission count of anyone near us. Board #1 is now **4.2e-9**
on 936 entries with a 7.38× late jump; its implied V·c is 1.5e3, 28× below anything
published.

The screen is deliberately conservative, so #82/193 is an **upper bound** on how
many real methods beat us, not an estimate. Five sub-1e-7 teams trip neither
criterion.

## 2026-08-06 — PHASE-1 WINDOW: no submission, selection left at the default

Round 1428 is live to Aug 7 23:59 UTC. **Decision (Samuel, this session): do not
spend a Phase-1 submission on v2.4.1.** The reasoning, recorded because it will
recur in Phase 2:

* There is **no participant-side selection mechanism** — `whest` has no designate
  command and `aicrowd_client.py` exposes no such endpoint. So 18118's default
  (top-2 public scores) governs, and it currently picks **{v2.4 2.719e-7, v1.1
  3.893e-7}**.
* v2.4.1's mini run is 3.172e-7 vs v2.4's 3.400e-7 (both 0 failures), so a graded
  v2.4.1 would land ~2.7e-7 and **displace v1.1** — replacing the only
  rotation-free hedge with a near-duplicate of v2.4.
* The graded-parity information was marginal: we already have a full `whest run` on
  the real flopscope/whestbench stack. A grader run would add only their BLAS.
* v2.4.1 debuts in round 1429 on Aug 8 at zero displacement cost.

Also repaired: `2026-08-06_v2.4.1_mini_subprocess.json` had 703 characters of
PowerShell stderr ("uv : Warning: You are sending unauthenticated requests…")
prepended, making it unparseable. Stripped; the JSON was intact underneath.

## 2026-08-06 — THEORY VERIFICATION: **P1 PASSES**, **P3 is sharper than expected**, P2 blocked

`local_sim/theory_probe.py`, fresh mints from seed block **D07**. Thresholds stated
before running; nothing here has a fitted parameter, which is the point.

**P1 — the parameter-free correlation map holds.** For He-init ReLU the uncentered
correlation of two inputs' representations should follow `c_{l+1} = R(c_l)` with
`R(c) = [√(1−c²) + c(π − arccos c)]/π` (arc-cosine kernel, Cho & Saul 2009; the
2/n scaling makes it exactly norm-preserving, so there is no constant to fit).
Measured over 4 starting correlations × 32 layers × 8 nets × 2048 input pairs:
**max |measured − R-iterate| = 0.0133**, against a pre-registered 0.02. **PASS.**

The useful corollary: at depth 32 the correlation has only reached ~0.98, not 1.
The approach is algebraic, not geometric — which is the quantitative reason the
participation ratio stalls near 3 instead of collapsing to 1. That gap was flagged
in the §7b stub as "the interesting part"; it now has a number attached.

**P3 — the collapse is toward μ specifically.** Low rank does not say *which*
direction survives; this does. Alignment of the fluctuation covariance's leading
eigenvector with the mean activation direction:

| layer | 1 | 9 | 17 | 25 | 32 |
|---|---|---|---|---|---|
| \|⟨v₁, μ̂⟩\| | 0.079 | 0.947 | 0.970 | 0.987 | **0.994** |
| PR | 159.3 | 21.5 | 8.6 | 4.2 | **2.4** |
| λ₁ share | 0.013 | 0.149 | 0.305 | 0.476 | **0.651** |

At the scored layer the dominant mode holds 65% of the fluctuation energy and is
**99.4% aligned with μ**. **This is the common mode of §2 in different vocabulary**:
an error along μ̂ is a rescaling of the whole output layer, which is exactly what
`1 + g` is. §2 and §7b are describing one phenomenon and the write-up should say so.

PR here reads 159.3 at layer 1 vs figure 1's 165.0 — expected downward bias from
estimating a 256-dim covariance with 4096 samples. Trend unaffected; quote
figure 1's values.

**P2 — NOT RUN, and deliberately.** The two-sector participation-ratio fit held out
at layers 12/20/28 needs the functional form in `docs/nngp-theory-draft.md`, which
**is not in the repo** (searched Documents/Downloads/Desktop too). Guessing a
two-sector form and reporting it as a verification would invert the exercise.
**Blocked on Samuel supplying the draft.**

## 2026-08-06 — VAULT BASELINE: v2.4.1, touch #1

The sealed 500-MLP holdout (seed block **V**, 2²⁵-sample ground truth per MLP)
received its one authorized seal-time baseline evaluation. Integrity verification
passed immediately beforehand: all four array hashes matched the manifest, five
spot-reconstructed networks matched the canonical seed→weights path, and the
manifest/log counters both read zero before evaluation.

| quantity | result |
|---|---|
| adjusted median | **2.4651e-7** |
| adjusted mean | **3.2669e-7** |
| adjusted p90 / p99 | 5.8172e-7 / **2.0535e-6** |
| adjusted max / max:median | 3.0469e-6 / 12.36× |
| raw MSE mean / median | 1.7622e-6 / 1.3234e-6 |
| mean C/B | 0.1873 |
| non-finite predictions | **0 / 500** |
| GT-noise MSE | 1.575e-9 = **0.09%** of measured raw MSE |

The result is consistent with the earlier 1200-network legacy-mint distribution
(median 2.19e-7, mean 3.01e-7, p99 1.65e-6) while now carrying exact evaluator
mint provenance and an independent estimator-seed stream. Touch #1 is recorded in
`experiments/vault/touch_log.jsonl`; the manifest and generated README both report
one touch. No estimator choice was made from this result.

The invariant suite was re-run after recovery and now passes **30/30** (the extra
count is the vault GT-noise-floor assertion; the earlier 29/29 documentation was
stale by one test).

## 2026-08-06 — CONSTANTS AUDIT: three keep, one bound withdrawn, one candidate rejected after real pricing

`local_sim/audit_constants.py`, fresh canonical mints from the preallocated,
never-reused blocks D01–D05. The τ/rank/N sweeps use 200 MLPs × 32 independent
digital shifts with paired common random numbers and 8,388,608-sample truth per
MLP; patch and calibration checks use 50 MLPs at the same truth precision. Full
paired means: `experiments/constants_audit_2026-08-06.json`. The original output
stream was interrupted after that JSON was saved, so the rank and N sweeps were
recomputed identically—not as new evidence—to preserve their paired standard
errors in `constants_audit_rank_2026-08-06.json` and
`constants_audit_cb_2026-08-06.json`.

| item | frozen | fresh result | verdict / action |
|---|---|---|---|
| pruning τ | -2.5 | **-2.5 is the exact optimum**; -3.0 is worse at 22.6σ, -2.0 at 13.3σ | **KEEP** |
| rotation rank | 32 | rank 256 appears 1.0108× better, only **0.7σ** | **KEEP 32**; apparent optimum is noise |
| sample count N | 16,384 | 65,536 is 1.0408× better at **3.2σ** in the FLOP-only model; 32,768 is 1.0343× | real-price CONFIRM before action |
| pruned-value patch bound | 3.8e-9 | **1.043e-8**, with GT noise 1.65e-12 | old bound **WITHDRAWN**; independent end-to-end τ sweep still selects -2.5 |
| v0 calibration | legacy table | max shown fresh/frozen drift **0.07%** | **KEEP values**; wiring debt below |

The patch result changes the claim, not the estimator. It measures error on the
substituted nearly-dead values themselves; the D01 objective already includes
their entire downstream effect and independently keeps τ=-2.5. The honest bound
is now 1.043e-8 on canonical mints. The earlier 3.8e-9 number is not repeated.

The calibration audit found a separate implementation debt: v2.4.1's
`_analytic_only` fallback returns raw closure means and does **not** contain the
audited `_CAL_LAYER_SCALE` table. That path only runs below the sampling ladder
and cannot affect the Phase-1 configuration. It is recorded, not patched into a
designation candidate; if Phase 2 supplies a low budget, restore the fresh-verified
table before using the fallback.

### N=65,536 real-stack confirmation — candidate rejected for now

The D03 GPU audit prices FLOPs but cannot reproduce the participant process's
residual-wall-time charge. A temporary v2.4.1 candidate changed only the sizing
guards (`_TARGET_CB=1.05`, `_MAX_CB=1.10`), passed `whest validate`, and ran the
read-only public mini split through the real subprocess stack:

| | N=16,384 frozen | N=65,536 candidate |
|---|---|---|
| adjusted | 3.17246e-7 | **3.15e-7** |
| raw final-layer MSE | 1.48953e-6 | 4.10e-7 |
| mean multiplier | 0.21598 | 0.78299 |
| failed MLPs | 0 | 0 |
| total residual wall time / 100 | 5.45 s | 20.33 s |

Only ~0.7% survives real local pricing, below the change threshold. Per the
read-only protocol, no choice was made from the public labels: D03 nominated the
arm, and the public stack supplied the omitted cost term. `estimator.py` was
restored to N=16,384, no submission was made, and the vault was **not** touched.
Because the grader historically charges much less residual time (v2.4: 9.84
ms/MLP vs 54 ms locally), keep the larger-N arm as a Phase-2 telemetry candidate,
not as a frozen change. Summary artifact:
`run_reports/2026-08-06_v2.4.1_N65536_mini_confirm_summary.json`.

## 2026-08-06 10:03Z — BOARD REFRESH: honest rank holds, probing signatures increase to eight

Fresh round-1428 snapshot after the recovery work:
`board_snapshots/2026-08-06T1003Z_2026-08-06_phase1_resume_board.json`, with the
full census in the adjacent `_census.json`. The board moved materially in 4.5
hours: dpskv5 stepped from 5.08e-8 to 5.8e-9 (8.76× late jump), making it the
eighth probing-signature team. Our raw rank moved #89→#90, but the headline rank
among plausible-honest teams is unchanged at **#82**, now **#82 of 192**.

Phase state at the same check: round 1428 live; round 1429 inactive and empty;
no `v1-phase2` dataset tag. Therefore the Phase-2 day-1 protocol has **not**
started and no Phase-2 submission or snapshot is due yet.

## 2026-08-06 — MOONSHOT INFRASTRUCTURE: low-rank K3/K4 adapter, no gate opened

Implemented `local_sim/lowrank_cumulant_flow.py`, fenced from `estimator.py`.
It adapts ARC `mlp_cumulant_propagation` revision
`93d091a4c26c042bfffa28f2e76a81bc0aba94bb`: full-dimensional K1/K2, factored
K3/K4 nonlinear propagation, top-r K2-eigenbasis projection after every ReLU,
and explicit per-layer K3/K4 renormalization inputs. K4 projection operates on
the paired factors and does not form an n^4 tensor.

Verification: CUDA projection smoke at width 16/rank 4 passed with maximum
relative out-of-basis component 1.81e-15. End-to-end three-layer toy flows ran
for both k_max=3 and k_max=4, returning three projected activation records and
finite width-4 means. Upstream needs syntax-only Python-3.10 compatibility edits
in the research checkout; the participant environment remains untouched.

**Classification: infrastructure, not G1 evidence.** D08 has not been minted or
scored, no gate has been opened, and no GPU-day budget was charged beyond the
toy smoke. Before G1, keep the estimator's calibrated K=2 closure as a separate
basis/control trajectory: ARC's full-width K1/K2 path is not bit-identical to
that baseline. No changes to `estimator.py`, no vault touch, no submission.

## 2026-08-06 — SPRINT 4: packed activation propagation. **The mechanism reproduces; the verdict turns on residual pricing**

GAME_PLAN rev 9 reopened one score lane on external evidence: submission 319341
spends **2.374e6 effective FLOPs/sample** against our v2.4's 3.303e6, at
`MSE*N = 1.852e-2` against our 2.252e-2. Sprint 3 had priced *static* late-layer
on-neuron composition; 319341's described mechanism is different and
value-dependent — sort activation columns by firing rate, pack a cold prefix,
bucket rows by their prefix support, skip the exact zero rectangles, gather the
rest, and use Strassen-style dense blocks. Reproduced here from first principles,
without reference to any competitor code.

New instruments, all running against **real flopscope 0.10.0**, not the shim:
`local_sim/packed_sparse_probe.py` (dense reference + packed path + a separate
analytic cost oracle), `local_sim/packed_truth.py` (labels for a decision block),
`local_sim/test_packed_sparse.py` (62 focused tests). Seed blocks **D11** (design)
and **D12** (locked, single-use) were allocated and recorded before any
measurement.

### S0 — semantic parity: PASS

| check | result |
|---|---|
| dense reference vs SHIPPED `estimator.py`, same MLP + same shift | **0.0 — bit-identical** |
| packing vs dense, max relative | **1.5e-8** |
| packing + column ordering vs dense | 1.5e-8 |
| Strassen depth 1 vs dense | 8.5e-8 |
| MSE change from packing (30 MLPs x 8 shifts, paired) | **1.0000x — none** |

Packing is an arithmetic reorganisation, not an approximation. The only entries
skipped are exactly `0.0` after the ReLU, where `0.0 * w` contributes exactly
`0.0`. Top-k selection uses `argpartition`, so a row with fewer than k nonzeros
gathers surplus **zeros** that self-annihilate, and a row with more than k is
routed to a wider bucket or to the exact dense remainder — never truncated. The
test suite drives a **1e-20 activation against a 1e20 weight row** and demands the
contribution survive; it does, for every bucket size. That test exists because a
scheme that quietly dropped small activations would look like a free speed-up,
survive an end-to-end MSE check inside sampling noise, and silently turn an
"exact" claim into an approximation.

### The structure, measured — and the closure already knew it

Post-ReLU zero fraction, measured against the closure's own `Phi(alpha)`:

| layer | 1 | 4 | 8 | 15 | 23 | 30 |
|---|---|---|---|---|---|---|
| measured | 0.500 | 0.466 | 0.445 | 0.350 | 0.275 | 0.224 |
| closure prediction | 0.500 | 0.466 | 0.445 | 0.350 | 0.276 | 0.225 |

Four decimals at every depth, including layer 30 where the closure's *means* are
18x worse than sampling. **That single fact kills the pilot component**: an
empirical firing-rate pass cannot improve on a prediction already exact to 1e-4,
and it costs a second plan build. Sparsity also *falls* with depth (alphas drift
positive, median 0.02 at layer 1 to 3.51 at layer 31), so the cold prefix is
widest and most valuable in the shallow half.

### The prefix rule, derived rather than tuned

Moving a column from the dense hot tail into the cold prefix saves `2*o` FLOPs of
dot product and costs `g*f*o` for the gathers it adds, where `f` is the column's
firing rate and `g ~ 6-7` is the per-nonzero gather price (4 to fetch the weight
row, 2 to multiply-accumulate, inflated by bucket padding). The prefix should
therefore extend exactly while `f < 2/g ~ 0.3`. **That boundary is a property of
the flopscope price table, not of the network.** Both the marginal-rate rule and a
fixed expected-support target were swept on D11; they land within 0.3% of each
other and `prefix_target = 3` (about three expected nonzeros per row) was kept.

### Three flopscope prices shaped the whole design, and two were surprises

* **basic slicing is free** — `h[:, :P]` bills 0 FLOPs and records *no operation
  at all*, while `fnp.take(h, arange(P), axis=1)` bills 4/element. Every block
  split in the packed path and inside Strassen is therefore a slice, never a
  gather.
* **`argpartition` is 4/element, `argsort` is 24** — selecting the top-k prefix
  entries by partition rather than sort is 6x cheaper on that step, and partition
  is all the algorithm needs because the summation is order-free.
* `einsum('nk,nko->no', ...)` prices exactly as a matmul, `(2k-1)` per output
  element, so the gathered correction is billed like the small dense product it is.

### Component ablations (D11, 30 MLPs x 8 shifts, paired, common random numbers)

| component | MSE ratio | FLOP-only cost | net V*c (FLOP-only) | verdict |
|---|---|---|---|---|
| packing only | **1.0000** | 1.119x | 1.119x | KEEP |
| Strassen depth 2 only | 1.0000 | 1.199x | 1.199x | KEEP |
| Strassen depth 3 | 1.0000 | 1.374x | — | **rejected: 52,575 ops/MLP** |
| antithetic Sobol | **1.0094** | 1.028x | 1.037x | KEEP |
| last-layer on-compose (alpha 2.5) † | 0.9983 | 1.019x | 1.017x | KEEP |
| pilot ordering (512) | 1.0000 | 0.865x | 0.865x | **KILL** |
| pilot reclassification | 0.9963 | 0.867x | 0.864x | **KILL** |
| COMBINED candidate | 1.0087 | 1.357x* | 1.369x* | — |

\*the ablation run predates a fix in which `on_compose` sliced the scored layer
*after* computing every column and so saved nothing; corrected, it restricts the
per-sample matmul to the kink columns and the combined arm reaches 1.383x.

† the on-compose row is from the separate `alpha_on` sweep (10 MLPs x 4 shifts),
not the 30 x 8 table, because the threshold had to be re-selected on canonical
mints. Sprint 3 had chosen 3.0 on legacy mints; the fresh sweep independently
reproduces the same mechanism and puts the optimum at 2.5:

| alpha_on | MSE ratio | FLOP-only cost | net V*c |
|---|---|---|---|
| 2.0 | 0.9798 | 1.0195x | 0.999x — the penalty eats the gain |
| **2.5** | 0.9983 | 1.0186x | **1.0169x** |
| 3.0 | 0.9989 | 1.0175x | 1.0164x |
| 4.0 | 0.9993 | 1.0157x | 1.0150x |

**A methodological note worth keeping.** A 12-MLP x 4-shift run put antithetic's
MSE ratio at 0.890 — a 12% *loss*. The 30 x 8 run puts it at 1.0094. The small run
was noise, and it was noise in the direction of a tidy mechanism ("antithetic
cancels odd components, but the depth-32 error is the even common mode"), which is
exactly when a wrong number is most persuasive. Standing rule reconfirmed: nothing
is decided on a small run, however good the story attached to it.

**The pilot's kill is structural, not an implementation artifact.** Pilot ordering
has MSE ratio *exactly* 1.0000 — it changes nothing, because the closure's firing
rates are already exact. Reclassification buys 0.6% of MSE for 15% of cost, and
the cost is dominated by the second plan build (fixed cost 1.517e10 vs 6.854e9).

### Cost anatomy: 1.38x is the arithmetic ceiling for this mechanism

Per MLP, N = 16,384, billed by real flopscope. The separate analytic oracle in
`oracle_cost()` reproduces the instrumented sampling bill to **0.01%**, which is
the check that the accounting is not a model of itself:

| | dense v2.4.1 | candidate |
|---|---|---|
| counted FLOPs | 5.330e10 | **3.854e10** |
| ... fixed (closure, eigh, slicing) | 6.85e9 (12.9%) | 6.85e9 (17.8%) |
| ... sampling | 4.645e10 | 3.169e10 |
| operations | 1,416 | **10,263 (7.2x)** |
| host round-trips (data-dependent) | 62 | **~310** |
| effective FLOPs/sample, FLOP-only | 3.253e6 | **2.352e6** |
| *published 319341* | — | *2.374e6* |

The ceiling is structural: the maximum saving from packing a prefix of width P is
`2*P*o` per row, and the firing-rate distribution puts the optimal P at 21-57
columns depending on depth, i.e. 11-22% of a layer. Strassen contributes an
independent `(7/8)^d` on the remaining dense block. Depth 3 wins more arithmetic
(1.374x) and loses on operations (52,575/MLP), which is the trade-off in miniature.

**We reproduced the published arithmetic target.** 2.352e6 effective FLOPs/sample
against 319341's 2.374e6, derived independently. The mechanism is real and the
number is reachable.

### S1 — the decision, and the cost model that decides it

Residual is not a rounding error here. flopscope bills `lambda * uninstrumented
seconds` at `lambda = 1e11`, so v2.4's graded 9.84 ms/MLP is already 1.8% of its
counted compute — and the packed path needs **7.2x the operations** to express the
same arithmetic. Value-dependent packing also *requires* data-dependent control
flow: the prefix width is a `searchsorted` read and every bucket size is a
`flatnonzero` shape read, taking host round-trips from 62 to about 310 per MLP. On
the evaluation servers `fnp.ndarray` is a `RemoteArray` (forum 18122), where
round-trips are precisely what costs.

Because the verdict depends on that model, **the model was fixed before D12 was
touched**, and all three readings are reported:

* **FLOP-only** — assumes op-count inflation is free. Known false (18101; v2.0 lost
  10.8% of its compute to residual). An upper bound.
* **PRE-REGISTERED (deciding)** — v2.4's graded 9.84 ms scaled by the
  candidate/dense residual ratio measured on an idle machine: **6.706x**, i.e.
  65.98 ms/MLP.
* **local residual** — this machine's absolute level, quoted only to show it is ~6x
  the grader's and therefore unusable as a level.

Residual ratios, idle machine, minimum of three repeats
(`experiments/packed/D11_residual_candidate.json`):

| arm | ops | residual x dense | modelled grader | cost-only V*c |
|---|---|---|---|---|
| dense v2.4.1 | 1,416 | 1.000 | 9.84 ms | 1.0000x |
| packing only | 3,222 | 3.292 | 32.4 ms | 1.0668x |
| Strassen 2 only | 8,460 | 5.276 | 51.9 ms | 1.0930x |
| packing + Strassen 2 | 10,227 | 7.059 | 69.5 ms | 1.1439x |
| packing + Strassen 3 | 52,575 | 12.682 | 124.8 ms | 1.0588x |
| **CANDIDATE (frozen)** | **10,263** | **6.706** | **66.0 ms** | **1.2026x** |

**Frozen design, closed on D11 before D12 was read:** packing with
`prefix_target = 3`, buckets 1/2/4/8/16, hot tail >= 96 columns; Strassen depth 2;
antithetic Sobol; last-layer on-composition at `alpha_on = 2.5`; no pilot.

### S1 result — **PASS**, but on the cost half, not on the headline number

D12, 50 MLPs x 6 shifts, paired, single use:

| cost model | paired V*c gain | 95% CI |
|---|---|---|
| FLOP-only (upper bound) | 1.6175x | [1.4547, 1.7984] |
| **PRE-REGISTERED (deciding)** | **1.4070x** | **[1.2656, 1.5642]** |

Both readings of the gate pass, including the strict one. **I do not believe
1.4070x, and the reason is worth more than the number.** The MSE ratio on D12 came
out at **1.2079** — a 21% variance win that *no component ablation predicted*
(packing is exact at 1.0000, antithetic 1.0094, on-compose 0.9983). A gain that
large which no ablation anticipated is the classic shape of an artifact, so the
frozen configuration was re-run on the DESIGN block to see whether it reproduces:

| block | MSE ratio | effective FLOPs/sample (FLOP-only) | pre-registered V*c |
|---|---|---|---|
| D11 (25 MLPs x 6, verification) | **1.0199** | 2.3589e6 | 1.2521x [1.0970, 1.4290] |
| D12 (50 MLPs x 6, the gate) | **1.2079** | 2.3599e6 | 1.4070x [1.2656, 1.5642] |

**The cost half reproduces to 0.04% across blocks. The MSE half does not.** The
two intervals overlap, so the blocks are not formally inconsistent, but the
component ablations put the method-level MSE effect at ~1.01 and D11 agrees at
1.02. D12's 1.21 is a favourable draw, and quoting 1.4070x as the method's gain
would be reading a draw as a mechanism — the same error the project has already
made three times (LMS scrambling, C/B=0.5, a +33% rotation) and once more
this very sprint with the 12-MLP antithetic number.

**What actually clears the gate is the cost, and it clears it alone.** The
cost-only paired ratio at the pre-registered residual is **1.2026x** with MSE held
at exactly neutral. So:

* the honest central estimate is **~1.23x** (1.2026 cost x ~1.02 MSE),
* the gate is 1.20x,
* and the pass does not depend on the noisy MSE draw.

That is a much narrower pass than the headline suggests, and it is recorded that
way so the next person does not budget for 1.41x.

### S2 — real-stack confirmation

`estimator_v2_5.py`, a **separate** candidate file (the shipped `estimator.py` is
untouched), generated with the Sobol direction-number blob spliced byte-for-byte
from the graded file rather than retyped:

| check | result |
|---|---|
| `whest validate --estimator estimator_v2_5.py` | **pass**, 110 ms |
| `edge_cases.py estimator_v2_5.py` | **30/30** |
| layer-1 closed form | exact to 1.8e-16 |
| permutation equivariance | implied MSE <= 4.0e-15 |
| budget ladder 0 -> 2.72e12 | monotone, no exceptions |
| determinism x3 + fresh instance | bit-identical |
| grader-array immutability | unmodified |
| vault touches | **1 — not spent** |

**S2 is NOT complete, and the reason is the most important thing this sprint
found about shipping the method.** The public mini run fails under the runner the
grader uses:

```
whest run --estimator estimator_v2_5.py --split mini --runner subprocess
  -> Error [setup:SETUP_TIMEOUT]: worker setup timed out.      (twice, reproducible)
whest run --estimator estimator.py     --split mini --runner subprocess
  -> scores normally                                            (control, same machine)
```

So it is the candidate, not the harness. It is **not** `setup()` being slow —
timed directly, v2.5's setup is *faster* than v2.4.1's (162 ms vs 340 ms, 1.26e8
FLOPs vs 2.53e8) because the antithetic half-set halves the Sobol table. And the
estimator is functionally sound: `--runner local` scores it without error, and
`whest validate` passes in 110 ms.

The lead is that the runner announces `memory_limit_mb=65536`, *"enforcement
requires --runner subprocess (uses RLIMIT_AS)"*. v2.5's transient footprint is
much larger than v2.4.1's — Strassen depth 2 holds 7 sub-products plus 10 operand
sums live per level, and the gathered corrections materialise `(rows, k, a_out)`
intermediates — and a candidate process was observed at ~3.9 GB during the D11
verification. A worker killed on an address-space limit would be reported by the
parent exactly as a setup timeout.

**This is a shipping blocker, not a footnote.** A worker that dies scores its
MLPs as zeros, which is the single worst outcome in this competition. It is also
the second independent way the same underlying fact bites: packing buys
arithmetic by spending *operations and memory*, and both are priced — one through
residual wall time, one through the worker's limits. Until it is understood and
fixed, v2.5 cannot be designated, and **v2.4.1 remains the fallback**.

### S2 verdict — **FAIL.** The real stack bills the operations, and they cost more than the arithmetic saves

Full 100-MLP public mini, `--runner local`, same machine, same day:

| | v2.4.1 (control) | v2.5 candidate |
|---|---|---|
| counted FLOPs / MLP | 5.33e10 | **3.86e10** |
| residual wall time / MLP | 70.8 ms | **387.6 ms** |
| effective compute / MLP | 6.04e10 | **7.74e10** |
| raw final-layer MSE | 1.49e-6 | 1.48e-6 |
| **adjusted score** | **3.26e-7** | **4.11e-7 (26% WORSE)** |
| failed MLPs | 0 of 100 | 0 of 100 |

**The arithmetic prediction was exactly right and the conclusion still inverted.**
Counted FLOPs came in at 3.86e10/MLP against the 3.854e10 the probe predicted —
0.2%. The packing works. But residual wall time went from 70.8 ms to 387.6 ms,
which at lambda = 1e11 is 3.88e10 of effective compute — **as much again as the
entire counted bill** — and it swallows the 1.38x arithmetic gain whole.

Two further facts, and they matter for how this is quoted:

* **The MSE gain is gone too.** 1.48e-6 vs 1.49e-6 on the organizers' own MLPs, a
  ratio of 1.007. That independently confirms the D11 reading (1.0199) and
  convicts D12's 1.2079 as a draw. The method-level variance effect is ~1.00.
* **The residual RATIO is 5.47x** (387.6 / 70.8), measured on the real stack —
  a better-founded number than the 6.706x the synthetic benchmark gave, and in
  the same place.

**Does this transfer to the grader?** Honestly: unknown, and that is the whole
point. This machine bills v2.4.1 **70.8 ms/MLP** of residual where the grader
billed **9.84 ms** — 7.2x lower. Applying the measured 5.47x ratio to the
grader's anchor gives v2.5 ~53.8 ms, hence `C = 3.86e10 + 5.38e9 = 4.40e10`
against v2.4.1's 5.43e10, i.e. **1.23x better** — a pass. Applying this machine's
*absolute* residual gives a 26% loss. Both cannot be right, and nothing available
offline decides between them. The eval servers run `flopscope-client` with
`RemoteArray` (18122), where a 7.2x operation count plausibly costs *more* per op
than it does in-process, not less — so the pessimistic reading is the one to plan
against.

### Sprint 4 verdict — **S1 PASS, S2 FAIL. The lane does not promote.**

`estimator.py` is untouched, the vault is untouched (touch count 1), and v2.4.1
remains the designated fallback. What was actually established:

1. **The published arithmetic target is real and reproducible.** 2.359e6 effective
   FLOPs/sample against 319341's 2.374e6, rebuilt from first principles, exact to
   1.5e-8 against the dense reference. The old "honest sampling frontier is 2.7e-7"
   claim was about *our dense sampler* and stays withdrawn.
2. **Packing's cost is paid in operations and memory, and flopscope prices both.**
   7.2x the operations, ~310 data-dependent host round-trips per MLP against 62,
   a 5.47x residual ratio measured on the real stack, and a worker that dies under
   the subprocess runner's memory limit. This is the finding, and it generalises:
   *value-dependent* sparsity exploitation is cheap in arithmetic and expensive in
   everything an accounting layer actually charges for.
3. **319341's advantage is therefore mostly NOT the packing.** Their `MSE*N` is
   1.852e-2 against our 2.58e-2 — a 1.40x per-sample *variance* advantage that
   packing cannot produce (packing is exact) and that none of our ablated
   components deliver. **That is where their score comes from, and it is now the
   open question worth attacking.**

**One graded run would settle the residual question** and nothing else can. That
is a decision for Samuel, not for this log.

## 2026-08-06 — independent Sprint-4 audit: arithmetic confirmed; promotion evidence corrected

The focused packed suite was rerun independently: **62/62 pass** in 81.8 s.
`whest validate --estimator estimator_v2_5.py` also passes. The D12 JSON was
recomputed directly rather than trusting the prose report:

* full pre-registered paired result: **1.407021x [1.265641, 1.564195]**;
* mean-MSE ratio: **1.207909x**, still not reproduced elsewhere;
* paired cost-only result with MSE held neutral: **1.199622x
  [1.196322, 1.202931]**.

The previously quoted 1.202607x cost result is `ratio(mean(cost))`. The gate's
own statistic is `exp(mean(log(cost_control/cost_candidate)))`, so the cost half
does **not** independently clear 1.20. S1 is recorded as borderline/inconclusive;
D12 remains spent.

The alleged subprocess memory failure also did not reproduce as candidate-
specific. A real one-MLP subprocess prediction with v2.5 completed successfully.
Across six clean worker starts per file under whestbench 0.14.0's hard 5 s setup
deadline, v2.4.1 passed 4/6 and timed out 2/6; v2.5 passed 6/6. This host has
sporadic cold-start latency. A 3.9 GB peak is not evidence of exhausting the
64 GB address-space cap. Memory remains a design constraint, not a demonstrated
blocker.

The like-for-like graded variance proxy is corrected to
`0.02252499 / 0.018520 = 1.21625x`; 1.40x uses D11/local levels across a different
suite. The published competitor's effective 2.374e6/sample already includes its
residual bill, so the next comparison must also account for fixed-cost
amortisation at its much larger N=84,992.

There was also a units mismatch in the Sprint-4 headline. Our 2.359e6/sample is
`counted_flops/N`; 319341's 2.374e6/sample is `effective_compute/N`. Their
like-for-like counted value is about 2.144e6/sample. Adding Sprint 4's own
pre-registered 66 ms residual model puts current v2.5 at about 2.762e6
effective/sample (the local public run is about 4.72e6). The external effective
target was therefore not reproduced, although the exact packing arithmetic was.

One offline optimization is already falsified as "nothing left to try": a
clean-room batched depth-2 Strassen microbenchmark combined 49 leaf products into
one batched contraction. It was bit-identical to the recursive path and changed a
representative `(4096,200)@(200,200)` block from 219 to 173 calls, local residual
3.390 ms to 2.843 ms (-16%), wall time 28.1 ms to 16.6 ms (-40%), at +1.0%
counted FLOPs. Full-estimator benefit is unmeasured. GAME_PLAN rev 11 therefore
requires call compression, bucket Pareto work, and an N sweep on new D13/D14
before any packed telemetry request.

## 2026-08-06 — SPRINT 5 OPENED: seed blocks D13/D14 allocated **before** any measurement

Per GAME_PLAN rev 11 and `docs/VALIDATION.md` §0/§2, two fresh blocks are
registered in `local_sim/seed_blocks.py` ahead of the first Sprint-5 number:

* **D13** — 30,806,013,000 … 30,806,013,599 (600). *Design/Pareto.* Batched vs
  recursive Strassen, the bucket/segment Pareto, the N ∈ {16384, 32768, 65536}
  sweep, and the residual model are all chosen here. Tuning permitted.
* **D14** — 30,806,014,000 … 30,806,014,599 (600). *Decision, LOCKED,
  SINGLE-USE*, and added to `SINGLE_USE_BLOCKS`. Not to be inspected until the
  design **and** the residual model are frozen.

D12 stays spent and is annotated as such in the registry. `check_disjoint()`
passes with both new blocks.

**Phase-2 checklist at open (`local_sim/phase2_checklist.py`):** round 1428
Phase 1 still `active=true` (closes 2026-08-07 23:59Z), round 1429 Phase 2
`active=false` with an empty leaderboard (1480, 0 entries), dataset tags still
`['v1-phase1', 'v1-warmup']` — **no `v1-phase2` tag**. No configuration change to
adapt to; no external action taken.

## 2026-08-06 — SPRINT 5 DESIGN RESULTS ON D13, and the **pre-registered freeze** for D14

*Everything below was measured on D13 (design, tuning permitted). The freeze at
the bottom is written down BEFORE D14 is touched; that is the whole point of
writing it here.*

### A — batched depth-2 Strassen: bit-identical, −53% operations, **no residual gain**

Built as a separately selectable implementation
(`Config.strassen_mode = "batched"`). The 7 (depth 1) or 49 (depth 2) leaf
products are issued as one stacked contraction; flopscope bills a stacked
`(b,m,k)@(b,k,n)` at exactly `b·(2k−1)·m·n`, so the arithmetic is identical and
the only overhead is the stacking. **The independent cost oracle predicts it to
0.02% with no free parameters** (`_sbatch_cost`; combine costs `4·batch·R·C` at
every level).

| microbenchmark | dense | recursive d2 | **batched d2** |
|---|---|---|---|
| (4096,200)@(200,200) — the audit's shape | 3.269e8 F, 1 call | 2.603e8 F, 217 calls, 4.85 ms resid | **2.645e8 F, 47 calls, 4.01 ms resid** |
| (16384,205)@(205,205) — a real layer | 1.374e9 F, 1 call | 1.128e9 F, 223 calls, 14.30 ms | **1.144e9 F, 53 calls, 10.82 ms** |
| (16384,199)@(199,187) — odd K and N | 1.216e9 F | 9.825e8 F, 223 calls, 12.71 ms | **9.985e8 F, 53 calls, 19.52 ms** |
| (65536,205)@(205,205) | 5.495e9 F | 4.511e9 F, 36.30 ms | **4.576e9 F, 39.78 ms** |

**Bitwise identical to the recursion on every shape and depth tested** — both
paths share `_ops7`/`_combine` and differ only in leaf dispatch, so `array_equal`
is the right assertion and it holds. Depth 0 falls through to the plain matmul
bitwise; depth 2 is deterministic over three runs; odd contraction *and* output
dimensions are padded, not skipped.

**But the audit's −16% residual / −40% wall does not survive the full
estimator.** On the shipped files, on the same MLPs:

| file | counted FLOPs/MLP | calls | local residual | output |
|---|---|---|---|---|
| `estimator.py` (v2.4.1) | 5.357e10 | 1,416 | 60.7 ms | — |
| `estimator_v2_5.py` | 3.872e10 | **10,263** | 519.0 ms | — |
| `estimator_v2_5r.py` | 3.914e10 (+1.07%) | **4,823 (−53%)** | 522.7 ms | **bitwise == v2.5** |

And on a quiet machine over 20 MLPs × 3 repeats (minimum kept), batched is
consistently *worse*: 495.5 → 533.4 ms at N=16,384, 970.2 → 1008.6 at 32,768,
1764.9 → 1939.6 at 65,536.

**The finding is that residual is not driven by operation count on this backend.**
Halving the call count while adding ~1.5% of copies made it slightly worse, three
times over. Residual here is data movement — gathers, temporaries, concatenations
— and batching adds copies to remove dispatches. The audit's microbenchmark was
right about its own shape and did not generalise. (The remote `RemoteArray`
backend could price dispatches differently; organizers say transport is framework
overhead, so we do not assume it would.)

One real bug fixed on the way: v2.5's Strassen size guard is stated at the
top-level block, which admits a (16384,33)@(33,33) layer to depth 2 and makes it
**13% more expensive than a plain matmul**. `estimator_v2_5r.py` states the guard
at the leaf.

### B — packing-operation Pareto: 24 exact carriers, and no bucket layout wins

Full table: `experiments/packed/2026-08-06_s5B_pareto.json` (6 MLPs × 3 repeats,
minimum residual, quiet machine). Highlights, all at N = 16,384 against a dense
v2.4.1 control at 5.283e10 counted FLOPs / 1,419 ops / 59.0 ms:

| carrier | counted FLOPs | ops | shape rd | value rd | resid | ×dense | max array | exact |
|---|---|---|---|---|---|---|---|---|
| **b=(1,2,4,8,16) control** | 4.743e10 | 3,227 | 210 | 0 | 174.4 ms | 2.95 | 32 MB | 1.3e-8 |
| b=() zero-rectangle only | 5.273e10 | 1,935 | 60 | 0 | 113.1 ms | 1.92 | 32 MB | 1.6e-8 |
| b=(8,) | 4.980e10 | 2,205 | 90 | 0 | 236.4 ms | 4.00 | **122 MB** | 1.3e-8 |
| b=(16,) | **5.417e10 (worse than dense)** | 2,147 | 90 | 0 | 317.9 ms | 5.38 | **244 MB** | 1.3e-8 |
| b=(4,8,16) | 4.802e10 | 2,687 | 150 | 0 | 185.9 ms | 3.15 | 51 MB | 1.3e-8 |
| b=(8,) segment | 4.985e10 | **2,055** | 0 | 60 | 215.0 ms | 3.64 | 122 MB | 1.3e-8 |
| b=(1,2,4,8,16) segment | 4.747e10 | **2,597** | 0 | 180 | 174.6 ms | 2.96 | 32 MB | 1.3e-8 |
| b=(8,) slotloop | 4.980e10 | 3,285 | 90 | 0 | 372.7 ms | 6.31 | **32 MB** | 1.6e-8 |
| control + strassen2 rec | **4.037e10** | 10,235 | 217 | 0 | 454.5 ms | 7.70 | 47 MB | 3.0e-7 |
| control + strassen2 BAT | 4.080e10 | **4,795** | 217 | 0 | 508.4 ms | 8.61 | 49 MB | 3.0e-7 |
| control, prefix target 1.5 | 4.817e10 | 3,190 | 210 | 0 | **138.5 ms** | **2.35** | 32 MB | 1.5e-8 |

Four things this table says that a single scalar would have hidden:

1. **Fewer buckets is not cheaper.** `b=(16,)` costs *more counted FLOPs than
   dense*, because a wide single bucket gathers a `(rows, 16, a_out)` weight
   tensor at 4 FLOPs/element — `6k·rows·a_out` against the dense prefix's
   `(2p−1)·rows·a_out`, which only pays for `k < p/3`. The six-class control is
   already near-optimal in counted FLOPs; Sprint 4 did not choose it by luck.
2. **The one-sort/segment partitioner works and is roughly a wash.** It reproduces
   the mask sweep's classes exactly (test 18), removes 500–630 operations, and
   converts 210 *shape* reads into 180 *value* reads — the more expensive kind on
   a `RemoteArray`. Residual moved 174.40 → 174.56 ms on the control layout and
   236.4 → 215.0 ms on `b=(8,)`. Not a lever.
3. **Peak memory is a bucket-design property nobody had priced.** A single wide
   bucket materialises a 122–244 MB gather tensor at N=16,384 (× 4 at N=65,536);
   the six-class layout keeps every bucket small and peaks at 32 MB. `slotloop`
   caps memory at 32 MB by accumulating slot-by-slot, at +136 ms of residual — it
   is a memory tool, not a speed tool.
4. **Every non-Strassen arm is exact to ≤1.6e-8 relative**, end-to-end through 31
   ReLU layers. The two `prefix_target = 6` arms sit at 3.6e-4: still an exact
   reorganisation per layer, but a wider prefix moves more terms between
   accumulation groups, and a f32 perturbation near a ReLU kink flips a sample's
   sign and compounds over depth. Not adopted; recorded because it is the
   mechanism that sets the tolerance.

**No carrier beats dense on local effective compute.** The best is 0.947×
(`control, target 1.5`); the frontier is that every low-FLOP arm needs Strassen,
which triples residual, and every low-residual arm keeps counted FLOPs ≥ 4.74e10.

### C — the N operating point: reopened, measured, and **not adopted**

Counted-FLOP structure is clean and identical in shape for every carrier:
`F(N) = F0 + c·N` with **F0 = 6.855e9** (closure, rotation, slicing — all
N-independent) and

| carrier | c_marginal (counted FLOPs/sample) |
|---|---|
| dense | 2.822e6 |
| pack6 | 2.491e6 |
| pack6 + strassen2 | **2.056e6** |

**The units-corrected reproduction of 319341's arithmetic:** their counted figure
is ~2.144e6/sample at N = 84,992. Ours, counted-to-counted at the same N, is
`2.056e6 + 6.855e9/84,992 = 2.137e6/sample` — **within 0.4%**. Sprint 4's headline
compared our counted number to their *effective* one; corrected, the arithmetic
target is not merely reachable, it is reached.

Local residual also splits cleanly, `r(N) = r0 + r1·N`:

| carrier | r0 | r1 | fixed share at N=84,992 |
|---|---|---|---|
| dense | 10.5 ms | 3.09e-6 s/sample | 3.8% |
| pack6 | 31.9 ms | 9.27e-6 s/sample | **3.9%** |

So residual is ~96% marginal in N — which is what licenses transferring the
competitor's 185.8 ms/MLP as a *per-sample* rate.

**MSE × N is flat, and the first measurement that said otherwise was noise.**
A 20-MLP × 1-shift pass gave MSE×N = 0.0472 / 0.0414 / 0.0530 at N = 16,384 /
32,768 / 65,536 — apparently a 14% optimum at 32,768, with an available story
about Sobol net quality. Re-measured at 24 MLPs × **8 digital shifts**:
**0.03657 / 0.04036 / 0.03666.** Flat. α = 1.00, as established. The "optimum"
was one heavy-tailed realisation.

Net of that, N = 65,536 is worth `3.99/3.60 = 1.11× ± 0.11` on local pricing for
the dense carrier — **not resolvable from 1.00**, and the constants audit's
independent CONFIRM on the organizers' own 100 MLPs read **1.002×**. Two
measurements, one of them CONFIRM-tier, and neither establishes a gain.
Memory is not the obstacle (196 MB largest array, 3.06 GiB process peak at
N = 65,536 — far inside 4 GiB/array and 64 GB).

### D — the three residual models

| # | model | r_control | r_candidate (pack6+s2rec, N=16,384) |
|---|---|---|---|
| 1 | local absolute | 59.5 ms | 495.5 ms |
| 2 | local ratio × graded dense anchor | 9.84 ms | 82.0 ms |
| 3 | **conservative = max(model 2, 319341-anchored)** | 9.84 ms | **82.0 ms** |

Model 3's competitor half is 185.8 ms/MLP split 3.9% fixed / 96.1% marginal using
*our own* measured `r(N)` fit, giving 41.7 ms at N = 16,384. Taking the `max` is
what makes model 3 conservative for *every* arm: the anchor alone is pessimistic
for low-ratio carriers and **optimistic for Strassen arms** (which measure 8.3×
dense locally against the 4.2× the anchor implies), and a model that is
conservative only for some arms is a selection effect, not a safeguard.

### THE FREEZE — written before D14 is touched

| item | frozen value | why |
|---|---|---|
| Strassen implementation | **recursive** | batched is bit-identical and −53% ops but measured worse residual three times over |
| Strassen depth | **2** | the only lever that reaches the counted-FLOP frontier |
| bucket design | **(1, 2, 4, 8, 16)**, `flatnonzero` partitioner, `einsum` gather | lowest counted FLOPs, lowest peak memory, best residual among low-FLOP arms |
| prefix target | **3.0** | target 6 breaks the 1e-6 exactness tolerance; target 1.5 costs 1.9% more FLOPs than it saves in residual |
| N | **16,384** | the N gain is 1.11× ± 0.11 locally and 1.002× on CONFIRM |
| antithetic | **off** | Sprint 4 measured 1.009×, inside noise |
| on-compose | **off** | not re-measured this sprint; adding an unmeasured component to a frozen candidate is how D12 got its phantom |
| residual model | **model 3, `max(local-ratio, 319341-anchored)`** | conservative for every arm, and half of it is someone else's measurement |
| gate | **paired geometric-mean `V·C` ≥ 1.15×, 95% CI excluding 1.0** | GAME_PLAN rev 11 |

**Predicted D13 value of the frozen candidate under the frozen model: 1.113×** —
already below the gate. D14 is spent anyway, because the design block is data the
candidate was tuned on and the registered decision is the one taken on data it
was not.

## 2026-08-06 — **D14 GATE: FAIL at 1.1216×. The packed lane is killed.**

D14 spent once, against the design and the residual model frozen in the entry
above. 100 MLPs × 3 shifts, paired, common random numbers.
`experiments/packed/2026-08-06_s5E_D14_decision.json`.

| | MSE | MSE×N | counted/MLP | counted/sample | ops | local residual |
|---|---|---|---|---|---|---|
| v2.4.1 dense control | 1.5586e-06 | 0.02554 | 5.2928e10 | 3.2304e6 | 1,417 | 62.3 ms |
| Sprint-5 candidate (frozen) | **1.5586e-06** | 0.02554 | **4.0300e10** | **2.4597e6** | 10,230 | 490.4 ms |

| residual model | r_ctrl | r_cand | paired `V·C` | 95% CI |
|---|---|---|---|---|
| 1. local absolute | 62.3 ms | 490.4 ms | 0.6618× | [0.6591, 0.6645] |
| 2. local ratio × graded anchor | 9.84 ms | 77.5 ms | 1.1216× | [1.1188, 1.1244] |
| **3. conservative (deciding)** | 9.84 ms | 77.5 ms | **1.1216×** | **[1.1188, 1.1244]** |

**Gate: 1.1216× < 1.15×. FAIL.** The CI excludes 1.0 and exactness holds, so this
is not a null result — the carrier really is ~12% cheaper — it is a result that
misses a pre-registered bar. D14 is spent and will not be reused; the candidate
will not be retuned against it.

Four things this run settles that earlier ones did not:

* **The D12 "1.2079× MSE advantage" is dead.** Control and candidate MSE agree to
  **five significant figures** (1.5586e-06 both), on 300 measurements. Exact
  packing cannot move variance and now demonstrably does not. Sprint 4's headline
  1.4070× was the product of a real 1.20× cost gain and a phantom.
* **D13 and D14 agree on cost.** The design block predicted 1.113× under the
  frozen model; the decision block measured 1.1216×, a 0.8% difference. No
  one-block anomaly is doing any work here, in either direction.
* **Model 3 collapsed onto model 2**, because the candidate's own local residual
  ratio (7.88× dense) is worse than the 4.2× implied by 319341's published
  185.8 ms. The `max` did its job: had model 3 been defined as the competitor
  anchor alone it would have read ~1.21× and *passed* a gate the carrier does not
  deserve.
* **The counted-FLOP target was reached; the effective-cost target was not.**
  2.4597e6 counted/sample here, and 2.137e6/sample when the frozen carrier is
  extrapolated to 319341's N = 84,992 against their 2.144e6 — a 0.4% match,
  counted-to-counted. Their advantage is therefore not arithmetic. It is that
  they pay 185.8 ms of residual for a carrier of this shape and *still* win,
  which means their per-sample variance, not their packing, is the thing worth
  understanding.

**Why the lane cannot be rescued by more engineering.** flopscope prices a gather
at 4 FLOPs/element. Packing's entire saving is substituting gathers for dense
multiply-adds: `6k·rows·a_out` instead of `(2p−1)·rows·a_out`. But a random-access
gather of a `(rows, k, a_out)` weight tensor costs vastly more *wall time* per
element than a BLAS multiply-add does, and flopscope charges wall time back at
λ = 1e11. The lane trades an operation that is expensive in the price table for
one that is expensive in physics, and the price table is not the thing being
billed twice. That is why every Sprint-5 compression lever measured ≤1.0×:
batched Strassen (−53% operations, +8% residual), the one-sort partitioner (−630
operations, ±0% residual), slot-loop gathers (−4× peak memory, +136 ms), fewer
buckets (+7% counted FLOPs), and larger N (1.11× ± 0.11 locally, 1.002× on
CONFIRM).

**Status.** `estimator.py` (v2.4.1) untouched and still designated. The vault was
not touched (counter still 1). `estimator_v2_5.py` preserved as the Sprint-4
historical candidate. `estimator_v2_5r.py` exists, passes `whest validate`, is
bitwise identical to v2.5 on 256×32, carries the tested batched-Strassen
implementation behind a flag that is **shipped off**, and fixes a real
narrow-layer Strassen guard bug. **No telemetry submission is requested for it.**

Test suite: `local_sim/test_packed_sparse.py` **93/93** (was 62/62);
`local_sim/edge_cases.py` **30/30** on the shipped file; vault touches 1.

## 2026-08-06 — MOONSHOT G0: the convention is right, and G1's baseline now exists

Opened per GAME_PLAN rev 12 after the packed lane was killed. `local_sim/moonshot_g0.py`,
seed block **D15** (newly allocated, infrastructure only). **D08/D09 untouched**,
no gate decided, **no cost claimed**.

MOONSHOT.md listed two prerequisites for spending D08 on G1. Neither existed.

**1. Layer-convention parity — VERIFIED, and the adapter was right.**
`lowrank_challenge_flow` transposes every weight (`W.mT`) on the stated grounds
that whestbench propagates row vectors while ARC's k-propagation uses column
vectors. That was an assertion. Measured against per-layer MC truth on 24
canonical mints, propagating the closure both ways:

| layer | row convention (`W^T μ`) | column convention (`W μ`) | ratio |
|---|---|---|---|
| 1 | 1.67e-07 | 1.21e-03 | **7,222×** |
| 2 | 2.84e-06 | 3.56e-01 | 125,585× |
| 8 (depth 8) | 3.39e-05 | 1.02e+00 | 30,058× |
| 32 (depth 32) | 6.28e-05 | 1.05e+00 | 16,787× |

The divergence starts at the **first matmul** and never recovers, which is the
signature of a transpose rather than of an accumulating modelling error. So
`W.mT` is correct and a mis-wired G1 has been ruled out as an explanation for
any future negative result — which was the point of checking before, not after.

Layer 1 is the control on the control: the closure there is 1.67e-07 against a
**GT-noise floor of 1.63e-07**, i.e. 97.3% of it is truth noise, and the exact
closed form agrees with the closure to the same 1.98e-3 max relative deviation.
Layer 1 is exact and the harness knows it.

**2. The separate calibrated-K2 control — BUILT and REGISTERED.**
It did not exist anywhere: MOONSHOT requires the baseline to be the *calibrated*
closure, ARC's own K1/K2 is a different trajectory, and v2.4.1's low-budget
analytic path returns **raw** closure — the table lives only in
`experiments/calibration.json`. Now implemented in torch, row convention, exact
layer 1, per-layer `c_layers` from the depth-matched table (constants audit:
≤0.07% drift on canonical mints).

| depth | raw closure | **calibrated K2** | gain | GT noise | G1/G2 must beat (÷3) |
|---|---|---|---|---|---|
| **8** (G1's shape) | 3.3872e-05 | **1.2835e-05** | 2.639× | 4.10e-08 (0.32%) | **≤4.278e-06** |
| 32 (G2's shape) | 6.2759e-05 | **2.4428e-05** | 2.569× | — | ≤8.143e-06 |

Registered on D15 **before any D08 network has been drawn**, which is what makes
it a baseline rather than a number chosen once the challenger's score is known.

**3. The flow as an accuracy oracle — run, after Samuel approved the checkout.**
ARC's `mlp_kprop` was not installed and no checkout existed on disk. With
approval, the pinned revision `93d091a4` was cloned to
`C:\Users\samsu\Documents\mlp_cumulant_propagation` and
`local_sim/mlp_kprop_py310.patch` applied cleanly (7 files). Set
`MLP_KPROP_SRC=C:\Users\samsu\Documents\mlp_cumulant_propagation\src`. Smoke test
reproduces MOONSHOT's recorded 1.81e-15 out-of-basis norm.

**Parity, now measured on ARC's own code:** at `k_max=2` the projection and the
renormalizers are inert, so the flow reduces to ARC's Gaussian closure. Against
ours, per-layer, the **max relative MSE difference is 8.373e-15** — floating-point
identity. The transpose and the `init_scale` metric are correct. (`linear_kprop`'s
own signature, `W: Float[Tensor, 'out_dim in_dim']`, says the same thing
independently.) A mis-wired G1 is ruled out.

**A real defect found and fixed.** At `k_max=3` upstream's tower still carries a
degree-4 slot, and it arrives as an `HTensor` rather than a `FactoredTensor4`, so
the adapter's dense fallback tried to materialise a 256⁴ tensor — **34 GB — and
took the GPU down.** The module's docstring claimed it "never materializes an n^4
tensor"; that was true only of the factored path. `project_higher_cumulants` now
takes `k_max` (so an order-3 flow does not project degree 4 at all) and refuses
an over-large dense projection with a diagnostic instead of an OOM. `k_max=3`
went from a crash to 0.3 s/MLP.

**And then the oracle reading, which is the substantive G0 result.** Depth 8,
width 256, 8 MLPs, ARC `k_max=3`, **unrenormalized** (`k3/k4_scales = 1.0`),
against the registered calibrated-K2 baseline of 1.2067e-05:

| rank | final-layer MSE | vs raw closure | **vs calibrated K2** |
|---|---|---|---|
| 4 | 2.8245e-05 | 1.112× | **0.427×** |
| 8 | 2.5900e-05 | 1.213× | **0.466×** |
| 16 | 2.2127e-05 | 1.419× | **0.545×** |
| 32 | 1.5799e-05 | 1.988× | 0.764× |
| 64 | 9.2819e-06 | 3.384× | 1.300× |
| 128 | 4.0283e-06 | 7.797× | **2.996×** |
| 256 (no truncation) | 9.4378e-07 | 33.279× | **12.786×** |

Two things at once, and they point opposite ways:

* **The mechanism is real.** At full rank ARC's K3 flow is **12.8× better than
  the calibrated closure** at depth 8. That is a large, clean effect and it is
  the first positive evidence this lane has produced.
* **The low-rank premise, as pre-registered, is wrong by an order of magnitude
  in r.** At MOONSHOT's proposed r ∈ {4, 8, 16} the flow is *worse than the
  control* (0.43–0.55×), and G1's ≥3× gate is not reached until **r ≈ 128 =
  width/2**.

That number is not a coincidence. It is exactly where **Track P** landed: rank-k
propagation *is* a rank-k factorisation, and under shape-based billing two
factors only beat a dense 256×256 matmul for k < width/2 = 128 — which is also
the rank truncation needs. The moonshot's compression hypothesis has walked into
the same wall from the other side, and the wall is in the same place.

**This is not a G1 kill and must not be recorded as one.** These runs use
unfitted renormalization scalars, and MOONSHOT's method specifies per-layer
scalars fitted offline against the chain's own trajectories (18097's DAgger
calibration, seed block D10, allocated for exactly this). Fitting them is the
next step, and it is the thing that could lift the low-rank arms. D08 and D09
remain untouched; **G1 has not been run.** No cost model, compressed
representation, or r³ claim is made or implied.

## 2026-08-09 — SLICE CONTRACT: grade the shipped file (v2.4.1) in the Phase-1 window  [PRE-REGISTERED, no data yet]

*Correction, made before any submission and before any result exists: this entry was
first written with the date 2026-08-07. That was a driver error — the session clock
was misread. Corrected to 2026-08-09 against two independent sources (the whestbench
run report stamps `Aug 09, 2026 06:28:21 UTC`; `git log` puts the last commit at
2026-08-07). **Only the date and the deadline paragraph changed. The hypothesis,
prediction, gate, and falsifier are untouched.***

**Strategic question**
Does the file we would actually designate behave on the grader the way our only
graded anchor did — and does the local→grader residual transfer ratio still hold?

**Current bottleneck**
The cost half of the evaluator is E1 (`docs/CURRENT_STATE.md` § *Evaluator trust*).
This machine bills v2.4.1 at 70.8 ms/MLP residual (LOG 2026-08-06, S2 verdict table,
`--runner local`, 100-MLP public mini) where the grader billed v2.4 at 9.84 ms/MLP —
a 7.2× discrepancy. Narrower and more urgent: **`estimator.py` == v2.4.1 has never
been graded.** Every grader-side number we own belongs to v2.4 (submission 324420,
2026-08-05).

**The window is an unannounced extension.** Round 1428's published end was
2026-08-07 23:59 UTC — ~30 h in the past as of writing — yet the AIcrowd API still
reports `round 1428 active=true LIVE` and `round 1429 active=false UPCOMING`, with
dataset tags still `['v1-phase1','v1-warmup']` (`local_sim/phase2_checklist.py`,
2026-08-09 06:1xZ). This is the deadline conflict of `docs/CURRENT_STATE.md` L13
resolving in favour of organizer thread 18125's Aug 10 date. It also means the round
may close without notice. Once it does, round 1429's configuration (shape, `n_mlps`,
budget, λ, floor) is unverified, so the controlled comparison against 324420 is lost
permanently.

**Evidence**
- CONFIRMED — v2.4 graded: adjusted **2.7194e-7**, raw MSE 1.3748e-6, multiplier
  0.1990, residual 9.84 ms/MLP (submission 324420, 2026-08-05).
- CONFIRMED — v2.4.1 on CONFIRM tier (public mini, 100 MLPs, 2026-08-06): adjusted
  3.172e-7, raw MSE 1.4895e-6, multiplier 0.216, 0 failed.
- CONFIRMED — v2.4.1 on VAULT (500 sealed, touch #1, 2026-08-06): median 2.4651e-7,
  mean 3.2669e-7, p99 2.0535e-6, 0 non-finite.
- CONFIRMED — `estimator.py` is byte-identical to `estimator_v2_4_1.py` modulo line
  endings (`scripts/preflight.ps1` #4); invariant suite 30/30 on the shipped file.
- CONFIRMED — v2.4 → v2.4.1 is a *reproducibility* change only: the rotation basis
  moved to `eigh` on `J Jᵀ` with a deterministic sign convention (LOG 2026-08-06,
  "the 18% gap was a rank-deficiency artifact"). No change to `_PRUNE_TAU`,
  `_BASIS_RANK`, N, the calibration table, or the sampling path.
- CONFIRMED — submission quota is 50/day/team (`docs/RESEARCH_NOTES.md` §1). The
  scarce resource is the round window, not the quota.
- CONFIRMED — nomination is explicit, not default top-2 (organizer thread 18125,
  recorded `docs/GOTCHAS.md` L74–79). This is the *new external evidence* that made
  the earlier no-submit decision conditional rather than final.
- INFERRED — the local→grader residual transfer ratio is ~7.2×, from one file on
  one day. It is the load-bearing assumption in every cost verdict this project has
  issued, and it rests on a single pair of observations.

**Hypothesis**
v2.4.1 grades statistically indistinguishable from v2.4 on both halves of the score,
because the two differ only in how the rotation basis is chosen, not in what is
computed. Its graded residual therefore also lands near 9.84 ms/MLP, giving a second
independent anchor for the transfer ratio.

**Mechanism**
The rotation change replaced an SVD whose sign/ordering conventions were
platform-dependent with `eigh` on `J Jᵀ` plus a fixed sign convention. The subspace
spanned is the same; only its representation is pinned. Arithmetic per sample is
unchanged (measured: same counted-FLOP anatomy, 5.33e10/MLP), so counted `C` should
match to well under 1%, and the operation count is unchanged, so residual should
match too.

**Intervention**
Submit the **unmodified** shipped `estimator.py` to round 1428. Nothing is edited.
One submission, not two.

**Prediction (registered before submitting)**
- graded adjusted score ∈ [2.4e-7, 3.1e-7], centred on ~2.72e-7
- counted FLOPs/MLP within 1% of v2.4's
- graded residual ∈ [7.4, 13.1] ms/MLP (±33% of 9.84)
- multiplier ≈ 0.199, floor untouched (`score / score_secondary` ≠ 0.1)
- `n_failed_mlps` = 0

**Samuel's prediction (registered before submitting, verbatim)**
> "I expect it to grade better than our last submission, if the leaderboard score is
> wrong then thats the disproving result."

Read: adjusted score **below** v2.4's 2.7194e-7. Falsifier as stated is loose; the
driver's reading is "the graded score contradicts what our local evidence implies."

**This is a registered disagreement between human and driver, and that is why it is
recorded.** The driver predicted *statistically indistinguishable* from v2.4
(centred ~2.72e-7, band [2.4e-7, 3.1e-7]) on the grounds that v2.4→v2.4.1 is a
reproducibility change only. Samuel predicts a genuine improvement. The two are
separable by the result: a graded score materially below ~2.6e-7 favours Samuel and
would mean the rotation fix moved accuracy, not just determinism; a score inside
2.7e-7 ± noise favours the driver. Either way one prior is corrected, which is worth
more than the submission.

**Gate**
This slice is *instrumentation*, so the gate is on the instrument, not on a win:
**PASS** if `n_failed_mlps` = 0 AND the graded adjusted score is within 1.30× of
v2.4's 2.7194e-7 in either direction. That band is deliberately wide — two
realisations of the same estimator differ ~18% in the mean (VALIDATION standing
rule), and the grader's public set is ~50 MLPs.

**Falsifier**
- Any failed MLP ⇒ the shipped file is not safe to designate, and that is the single
  most valuable thing this slice could discover.
- Graded residual outside [7.4, 13.1] ms/MLP ⇒ the 7.2× transfer ratio is not a
  property of the grader stack, and **every cost verdict in this repository that
  used it is downgraded**, including Sprint 4's S2 and Sprint 5's D14 residual
  models. That would be a larger finding than the submission itself.
- Adjusted score > 1.30× v2.4's ⇒ the "reproducibility-only" claim for v2.4.1 is
  false and v2.4 is the file to nominate.

**Baseline**
v2.4, submission 324420, GRADED tier, 2026-08-05: 2.7194e-7 adjusted / 1.3748e-6
raw MSE / 0.1990 multiplier / 9.84 ms/MLP residual. Same round, same dataset
revision (`v1-phase1`), same fixed 50-MLP public suite (forensics 2026-08-05
established the scored set is fixed and returns per-MLP MSEs).

**Residual model**
Not consumed — **produced.** This slice makes no cost *claim*; it creates the
observation that other slices' models are anchored to. The three-model rule
(`docs/VALIDATION.md` §2b) does not apply because no local residual enters the
verdict. Local absolute residual is quoted nowhere in the result.

**Statistic**
Not a paired local experiment. The comparison is graded-to-graded, single
observation per file, on the grader's own fixed suite — which is why the gate band
is wide and no CI is claimed. Per-MLP MSEs from `public_scores` permit a paired
per-MLP comparison against 324420's report *post hoc*; that is a bonus reading, not
the gate.

**Confounders**
- The grader's machine load varies; residual is a wall-time measurement on hardware
  we do not control. A single reading cannot separate "v2.4.1 is slower" from "the
  grader was busier." Stated as a limit, not resolved.
- If the organizers changed the public suite mid-round, `sampling_mse` and the
  `mlp_name` list would differ; both are checked in the report before any comparison.

**Scope**
Files expected to change: **none.** `estimator.py` is submitted unmodified. New
artifacts only: `experiments/reports/2026-08-09_v2.4.1_phase1_*.json` and a mini-run
JSON.

**Do not change**
`estimator.py` (byte-for-byte), any constant, the vault (counter stays 1), any seed
block (none allocated — no local decision block is spent by this slice).

**Evidence required**
```powershell
.\scripts\preflight.ps1                                    # must be GREEN
uv run whest validate --estimator estimator.py
uv run whest run --estimator estimator.py --dataset hf://aicrowd/arc-whestbench-public-2026@v1-phase1 --split mini --runner subprocess
uv run whest submit --estimator estimator.py               # ONLY after Samuel's explicit approval
uv run python local_sim/fetch_graded_report.py <id> 2026-08-09_v2.4.1_phase1
```
Recorded from the report: `config_snapshot` (width/depth/n_mlps/flop_budget),
`lambda_derived`, `residual_share_of_billed`, `multiplier_floor_check`,
`runtime_environment`, `sampling_mse`, the `mlp_name` list, per-MLP `public_scores`.

**Driver**
Claude Code.

### RESULT — grade the shipped file (v2.4.1), submission 327743  [contract above, unmodified]

**commit**          `70067b4` (this entry, the graded artifact, and the durable-doc
                    updates land together; `docs/HUMAN_MODEL.md` excluded by design)
**provenance**      AIcrowd submission **327743**, round 1428, state GRADED.
                    Raw report: `experiments/run_reports/2026-08-09_v2.4.1_phase1_327743_graded_report.json`
                    Control: `experiments/run_reports/2026-08-05_v2.4_324420_graded_report.json`
**commands**
```powershell
.\scripts\preflight.ps1                                      # GREEN 5/5
uv run whest validate --estimator estimator.py               # pass, 62 ms
uv run whest run --estimator estimator.py --dataset hf://aicrowd/arc-whestbench-public-2026@v1-phase1 --split mini --runner subprocess
uv run whest submit --estimator estimator.py                 # after explicit approval
uv run python local_sim/fetch_graded_report.py 327743 2026-08-09_v2.4.1_phase1
```
**config / seeds**  n_mlps 100, depth 32, width 256, flop_budget 2.72e11,
                    lambda_derived 1.0000e11. No seed block allocated or spent.
                    Vault touches spent: **0** (counter remains 1).
**suite identity**  All 100 `mlp_name` values identical and in the same order across
                    324420 and 327743; the 50 `public_scores` indices are the same
                    set. **The comparison is paired on identical networks.**

#### Headline

| | v2.4 (324420) | **v2.4.1 (327743)** | ratio |
|---|---|---|---|
| adjusted final-layer score | 2.7194e-7 | **2.6314e-7** | 0.9676 |
| raw final-layer MSE | 1.3748e-6 | 1.3323e-6 | 0.9691 |
| all-layers MSE | — | 4.8263e-5 | — |
| mean multiplier | 0.1990 | 0.19896 | 0.9998 |
| counted FLOPs / MLP | 5.2909e10 | 5.2898e10 | 0.9998 |
| effective compute / MLP | 5.3893e10 | 5.3901e10 | 1.0001 |
| grader residual / MLP | 9.845 ms | 10.022 ms | 1.0180 |
| failed MLPs | 0 | **0 of 100** | — |
| `multiplier_floor_check` | — | 0.19751 ≠ 0.1 | floor untouched |

**delta vs gate — PASS.** Gate was `n_failed_mlps = 0` AND adjusted within 1.30× of
2.7194e-7 either way. Both met (0 failures; ratio 0.9676).

#### The 3.2% improvement is a draw. The protocol caught it in real time

The aggregate moved the right way. **The mandated paired statistic says there is no
effect**, and the disagreement between those two readings is the substance of this
slice.

| statistic (v2.4 ÷ v2.4.1 on final-layer MSE, >1 favours v2.4.1) | value |
|---|---|
| ratio of **means** (the headline) | 1.0319 |
| ratio of **medians** | 1.0107 |
| **paired geometric mean of per-MLP ratios, n=50** | **0.9665×**, 95% CI **[0.7969, 1.1721]** |
| CI excludes 1.0? | **no** |
| sign test — v2.4.1 better on | **22 of 50**, one-sided p = **0.839** |

**A single network carries the entire headline.** Summing the per-MLP MSE deltas over
the 50 visible networks gives 2.1246e-06 total. `mlp_index 37` alone contributes
**140.2%** of it (3.985e-06 → 1.005e-06); the top 2 contribute 271%, the top 5
contribute 526%. The sum only reaches its positive total because a long tail of small
*regressions* offsets a handful of large favourable draws. Drop network 37 and the
aggregate improvement reverses sign.

This is the repository's canonical failure mode, observed for the first time on
**grader-side data with a genuinely paired design**: `docs/VALIDATION.md`'s rule
*"the paired statistic is the geometric mean of per-MLP ratios, not the ratio of
means"* is not a technicality here — it is the difference between "v2.4.1 improved
accuracy 3.2%" and "v2.4.1 changed nothing." Only the second is defensible.

#### Predictions, scored

| | predicted | outcome |
|---|---|---|
| **Driver** | statistically indistinguishable from v2.4; adjusted ∈ [2.4e-7, 3.1e-7]; residual ∈ [7.4, 13.1] ms; counted FLOPs within 1%; 0 failures | **held on every clause.** 2.6314e-7; 10.022 ms; 0.02%; 0 failures; paired CI includes 1.0 |
| **Samuel** | "better than our last submission" | **the headline went his way, the statistic did not.** Adjusted score is 3.2% lower, but the paired test cannot distinguish it from a draw (22/50, p = 0.839) |

Samuel's prediction was not unreasonable and the number he named did move as he
said. The reason it does not stand is precisely the rule this project already had in
place, which is the best possible outcome for a prediction checkpoint: the human's
prior was corrected by a pre-existing protocol rather than by an argument.

#### Mechanism confirmed structurally, not just by score

`per_layer_mse` for network 0 is **bit-identical between v2.4 and v2.4.1 in 31 of 32
layers**; the first and only divergence is at **layer index 31**, the scored final
layer. The two files therefore produce *the same numbers* everywhere the closure
runs, and differ only in the sampled final-layer estimate — exactly what the
"reproducibility-only rotation change" claim predicted, verified against the grader
rather than against our own reasoning. A mis-scoped change would have shown
divergence from the first sampled layer onward.

#### The substantive finding: the ~7× transfer ratio is path-dependent, and the grader is the stable half

This slice was justified as evaluation repair. What it repaired is not the half that
was suspected.

| | residual, ms/MLP | source |
|---|---|---|
| **grader**, v2.4 | 9.845 | 324420, 2026-08-05 |
| **grader**, v2.4.1 | 10.022 | 327743, 2026-08-09 |
| paired grader ratio v2.4.1/v2.4 | **1.0180×**, 95% CI [1.0119, 1.0241], n=100 | — |
| local, v2.4.1, `--runner local` | 70.8 | LOG 2026-08-06 S2 |
| local, v2.4.1, `--runner subprocess` | 42.94 | this slice, 2026-08-09 |

**The grader side is stable to ±0.6% across four days and two files.** It was never
the unreliable half. The *local* number moved **70.8 → 42.94 ms/MLP for a
byte-identical file**, which puts local/grader at **7.06×** or **4.28×** depending
entirely on which local path was used.

`AGENTS.md` § *Measurement law* states as binding fact: *"Local residual wall time is
~7× the grader's."* **That constant was derived from a `--runner local` measurement,
and against the runner the grader actually uses it is ~4.3×.** Every cost verdict
that scaled a local residual by ~7 — including Sprint 4's S2 and the D14 residual
model — used a factor that may be 65% too large.

**This is not yet established, and must not be recorded as if it were.** The two
local readings differ in *two* variables at once — runner (local vs subprocess) and
machine state (2026-08-06 vs 2026-08-09) — and this slice cannot separate them. The
contract's falsifier was aimed at the grader residual and therefore did not fire;
that is a design defect in the contract, recorded here rather than quietly fixed:
**the doubtful quantity was the local measurement, and the falsifier tested the
grader.**

**Cheapest discriminating follow-up:** run the same unmodified file on the same mini
split today under both `--runner local` and `--runner subprocess`, back to back. It
spends no seed block, no vault touch, and no submission, and it separates runner from
machine state in one pass. Deliberately **not** run inside this slice
(`slice-contract` Step 8: do not open the next slice from inside this one).

#### Regressions and integrity

preflight **5/5 GREEN** · invariants **30/30** · ship-file identity `59c371a5…`
matches `estimator_v2_4_1.py` · vault counter **1**, untouched · `estimator.py`
unmodified · no seed block allocated · public mini re-run clean (0/100 failed,
adjusted 3.11e-7, multiplier 0.2117).

#### Round intelligence captured while submitting

The AIcrowd round objects **changed during this session** (two polls ~30 min apart):

| | first poll | second poll |
|---|---|---|
| round 1428 Phase 1 ends | 2026-08-07 23:59 UTC | **2026-08-10 23:59 UTC** |
| round 1429 Phase 2 opens | 2026-08-08 00:00 UTC | **2026-08-18 00:00 UTC** |

The deadline conflict recorded in `docs/CURRENT_STATE.md` L13 is **resolved in the
API itself**, in favour of organizer thread 18125's Aug 10 date, and Phase 2 now
opens a **week** after Phase 1 closes rather than immediately. Dataset tags remain
`['v1-phase1','v1-warmup']`; leaderboard 1480 still empty.

**deviations**     None from the contract as written. The contract's date was
                   corrected from 2026-08-07 to 2026-08-09 before submission (driver
                   misread the session clock); hypothesis, prediction, gate and
                   falsifier were untouched, and the correction is noted in place.
**unexpected**     (1) The paired statistic contradicting the headline on grader data.
                   (2) 31 of 32 layers bit-identical between the two files.
                   (3) The local residual, not the grader residual, being the
                       unstable quantity.
**remaining doubt** Whether ~4.3× or ~7× is the right transfer factor, and whether it
                   is a constant at all. Whether v2.4.1's +1.8% grader residual is a
                   real property of the eigh path or grader machine load — the CI is
                   tight but n=1 submission per file. The 50 hidden networks remain
                   unobservable.

### Knowledge Delta

- **Learned.** v2.4.1 is graded, safe, and functionally equivalent to v2.4: 0
  failures, counted FLOPs identical to 0.02%, layers 1–31 bit-identical, paired
  final-layer MSE indistinguishable. **The file we would designate now has grader
  evidence.** That was the primary gap and it is closed.
- **Belief changed — the driver's.** "Local residual is ~7× the grader's" was carried
  as a property of the grader. It is a property of *a local measurement path*. The
  grader half is the stable one (±0.6%); ours is the one that moved 65%.
- **Belief changed — Samuel's.** A 3.2% headline improvement on 50 real grader
  networks is not evidence of improvement. He predicted a genuine gain; the paired
  test says draw, driven by one network out of fifty.
- **New failure signature.** *Aggregate-mean improvement with a paired CI spanning
  1.0 and a sign test at chance.* Diagnostic: rank the per-MLP deltas and check what
  fraction of the total the top mover carries. Here it was **140%** — a single
  network more than accounting for the entire effect. Any delta with that shape is a
  draw until proven otherwise.
- **New invariant candidate.** v2.4 and v2.4.1 must agree bit-for-bit on layers 1–31;
  only the scored final layer may differ. Cheap to assert, and it would catch a
  rotation change that silently escaped its intended scope.
- **Remaining uncertainty.** The runner-vs-machine-state confound on the transfer
  ratio. Until it is resolved, no cost claim should quote a single transfer factor as
  if it were measured.
- **Durable docs to change.** `docs/CURRENT_STATE.md` (graded row for v2.4.1, round
  calendar, transfer-ratio caveat, stale uncommitted-work risk). `docs/GOTCHAS.md`
  (the path-dependence trap; the aggregate-vs-paired signature).
  **`AGENTS.md` § *Measurement law* carries the ~7× figure and must NOT be edited
  until the discriminating run separates runner from machine state.**
- **Implication for the next experiment.** The discriminating local run comes first;
  it is nearly free and it gates the credibility of every cost number this project
  would put in the write-up.
- **For the write-up.** This is unusually clean contribution material: a
  pre-registered prediction, a human/driver disagreement recorded before the data
  existed, a paired design on the organizers' own networks, and a headline that
  dissolves under the correct statistic — with the single responsible network named.

**Driver** Claude Code. **Next command:** `review-slice`.

## Next planned experiments (rewritten 2026-08-05 after B1/B2/B3 all came back negative)

**Where we stand**: v1.1 graded 3.893e-7, rank ~159. The leader is 4.87e-8 — **8.0× ahead**. Every analytic-side avenue we tested this session is dead:

| direction | verdict | evidence |
|---|---|---|
| B1 precision-weighted fusion | +4.7% best, vs 4.7% compute tax ⇒ net ~0 | full-split holdout |
| B1 scale-mode kill via covprop | +3.6%; contaminated (R² 0.094) | predicted 0.092, measured 0.094 |
| scale mode from point-set radius | dead (LOO R² −0.010) | `scale_source_probe` |
| scale mode from exact layer 1 | dead (LOO R² −0.037) | `scale_source_probe` |
| B2 analytic tail, any k | −67% to −214% | `b2_tail_probe` |
| B3 learned covprop residual | R² 0.012 ⇒ drop Phase C | 50k MLPs, 7.5k held out |
| input rotation (untasked) | +8.3% for 3.2% cost ⇒ ~5% net | full mini split |

**The one structural fact worth attacking**: 48% of v1.1's error is a single per-MLP scalar (the whole layer scaled by 1+g), worth **1.9×** if removed exactly (1.604e-6 → 8.349e-7 ⇒ adjusted ~2.0e-7). Nothing we can currently observe predicts g better than R² ≈ 0.15. Finding an observable for g is the highest-value open problem we have.

1. ~~**Effective-dimension / better rotation.**~~ **Done — Track P.** The deviation
   spectrum *is* low-dimensional (effective rank 25.9 → 2.8 over depth, top-64 holds
   96.2% by layer 16) and the closure's own covariance eigenvectors find it for free.
   It cannot be spent: rank-k propagation is a rank-k weight factorisation and
   shape-based billing breaks even at k = width/2, which is also the rank truncation
   needs. Bounded at 1.093× even with a free oracle correction.
2. ~~**Smoothing for QMC.**~~ **Done — Track T.** Conditional-expectation smoothing of
   the last layer is a *no-op*: in the α ≥ 3 band carrying 76.5% of the MSE, the
   rectified-Gaussian formula reduces algebraically to the sample mean (ρ = 1.000).
   Overall 1.0001× cross-fitted.
3. `wall_time_limit_s` still unknown (not in the participant report); it has never bound
   us (0.41 s/MLP used, ~98× headroom), so this is low priority.

### Phase-2 opening runbook — **event-triggered on round 1429 going active**

*(Date corrected 2026-08-06: the challenge page still says Phase 2 opens Aug 8, the
organizer announcement says Phase 1 runs to Aug 10. Trigger on the round flipping
active, not on a date. Poll with `phase2_checklist.py`.)*

Score optimisation is **closed** (GAME_PLAN rev 6 — the sampling frontier is
declared at ~2.7e-7). Phase 2 work is verification and robustness, in this order:

1. `uv run python local_sim/phase2_checklist.py` — is 1429 live, is there a
   `v1-phase2` HF tag, are the boards up?
2. `uv run python local_sim/board_snapshot.py --top 30 --label phase2_open` — and
   daily thereafter; `--diff` against the previous file.
3. `uv run whest validate --estimator estimator.py` then a mini run, then submit
   **v2.4** (unchanged). One submission, not two — the fixed-suite question is
   already answered for Phase 1 by identical `mlp_name` lists, so the twin test
   costs two submissions to re-confirm something we can get from the first Phase-2
   report's names alone.
4. `uv run python local_sim/fetch_graded_report.py <id> 2026-08-08_phase2` then
   `uv run python local_sim/twin_diff.py --all`. Record: shape / n_mlps / budget /
   λ / floor from `config_snapshot` + telemetry; whether `public_scores` still
   returns per-MLP and per-layer MSE (i.e. whether the disclosure was acted on);
   whether `sampling_mse` changed (a new value ⇒ new suite).
5. **If the shape changed**: do not optimise. Re-measure in order — α-histogram,
   error-vs-N, τ sweep, then re-select the stack. `estimator_v1_1.py` is the
   shape-generic fallback; the edge-case suite already covers widths 64/128 and
   depths 2–64, all passing.
6. `>3×` grader-vs-local divergence ⇒ stop and investigate before resubmitting.

---

## 2026-08-07 — HARNESS (IMPORTANT): Claude ↔ Codex portability audit

**Class** IMPORTANT — changes the instrument that governs every other change (the
workflow layer and the enforcement layer). No seed block allocated, no vault touch,
`estimator.py` untouched.

**Prediction, written before building** (the IMPORTANT-class requirement): the six
workflow skills would prove *already* mostly agent-neutral in content, and the real
portability failures would be (a) no `AGENTS.md`, so the operating contract existed
only as `CLAUDE.md`; (b) mechanical enforcement existing only as a Claude-shaped
hook; (c) the Skeptic existing only as a Claude subagent. Predicted that no workflow
*semantics* would need to change — only their location and their bindings.

**Result: prediction held.** All six workflow bodies moved to `ai/workflows/`
essentially verbatim. The only content edits were de-branding (`/next-slice` →
`next-slice`; "launch the skeptic agent" → "run the Skeptic role in a separate agent
context"; `.claude/skills/slice-contract` → `ai/workflows/slice-contract.md`) plus
three new paragraphs on handoff continuity. No gate, threshold, residual model, tier
rule, or classification boundary changed.

### What changed

| layer | before | after |
|---|---|---|
| operating contract | `CLAUDE.md` only | `AGENTS.md`; `CLAUDE.md` imports it and is now 62 lines of bindings |
| workflow semantics | 7 skills under `.claude/skills/` | `ai/workflows/` (6 + slice-contract + skeptic); both agents' skills are <=33-line adapters |
| Codex control surface | none | `.agents/skills/` (7), `.codex/agents/skeptic.md`, `.codex/hooks/` |
| enforcement | one bash hook, Claude-shaped, file_path only | `scripts/guard_paths.py` — one implementation behind both agents' hooks and `scripts/git-hooks/pre-commit` |
| HUMAN_MODEL blank/stale test | agent eyeballs the file | `scripts/human_model_state.py` -> BLANK/PARTIAL/STALE/FILLED |
| portability itself | unchecked | `scripts/verify_portability.py`, 174 checks |

### Knowledge delta

1. **The old guard had a real hole.** It matched the `file_path` field only, so a
   shell redirection onto a protected path was never blocked. The shared
   implementation now resolves the *write target* of a command (redirection takes one
   token; a verb may be preceded by flags) and denies on that. Found while building
   the Codex adapter, not by a test — open since the hook was written.
2. **Precision matters as much as coverage in a guard.** The first cut denied on
   "protected name appears anywhere in the command", which blocked writing this very
   LOG entry, because the entry discusses `HUMAN_MODEL.md`. A guard that blocks honest
   work gets disabled, and then the invariant is gone entirely. The regression cases
   for both directions are now permanent in `verify_portability.py`.
3. **An agent hook cannot be the guarantee, only the fast path.** It depends on that
   agent's hook system being wired and schema-compatible. `scripts/git-hooks/pre-commit`
   is the agent-independent backstop; it refuses to let a protected path be committed
   *quietly*. `ARC_WHEST_PROTECTED_OK=1` is the conscious human override, and **Samuel
   needs it to commit his own `HUMAN_MODEL.md`** — the guard forbids silence, not
   authorship.
4. **New failure signature: harness drift.** Two adapter sets can diverge silently,
   and the symptom appears only at the moment of switching — the worst moment.
   `verify_portability.py` fails on a missing workflow, a fat adapter, a divergent
   skill set, a dangling file reference, a broken command chain, or a product name in
   the canonical layer.
5. **Not verified, and stated as such:** the Codex hook *registration* schema. Codex
   is not installed on this machine. `.codex/hooks/guard_hook.py` was verified
   directly (it satisfies stdin-JSON, flag args, stdout JSON, stderr reason and
   0/1/2 exit codes simultaneously), but its wiring into `config.toml` was left as a
   documented paste-once snippet in `.codex/hooks/README.md` rather than written
   blind — a bad config key can stop an agent starting, which is worse than an unwired
   hook. The pre-commit backstop covers the integrity invariants meanwhile; what is
   lost without it is the two advisory *ask* prompts.

### Evidence

```
python scripts/verify_portability.py           174 checks, 0 failed
.\scripts\preflight.ps1                        PREFLIGHT GREEN -- 5/5
uv run whest validate --estimator estimator.py passed, 72 ms
uv run python scripts/human_model_state.py     BLANK -> FIRST-ORIENTATION MODE
python scripts/install_git_hooks.py            installed .git/hooks/pre-commit
```

Guard policy exercised on 21 cases (deny / ask / allow), each run through the shared
implementation *and* through the Codex adapter, verdicts identical. Pre-commit
backstop exercised on four staged-file scenarios including the override path and the
vault-seal-inconsistency refusal.

**Durable state unchanged**: no seed block spent; vault touches still 1; shipped file
still byte-identical to `estimator_v2_4_1.py` (sha256 59c371a5044e0e00a04e00fe8494ef19).

**Next**: switching drivers is now `session-start` then `next-slice` from the repo
root. The project's own next actions are unchanged — see `docs/CURRENT_STATE.md`.
