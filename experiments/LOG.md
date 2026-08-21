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

## 2026-08-10 — SLICE CONTRACT: grade the shipped file (v2.4.1) in the Phase-1 window  [PRE-REGISTERED, no data yet]

*Correction, made before any submission and before any result exists: this entry was
first written with the date 2026-08-07. That was a driver error — the session clock
was misread. Corrected to 2026-08-10 against two independent sources (the whestbench
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
2026-08-10 06:1xZ). This is the deadline conflict of `docs/CURRENT_STATE.md` L13
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
artifacts only: `experiments/reports/2026-08-10_v2.4.1_phase1_*.json` and a mini-run
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
uv run python local_sim/fetch_graded_report.py <id> 2026-08-10_v2.4.1_phase1
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
                    Raw report: `experiments/run_reports/2026-08-10_v2.4.1_phase1_327743_graded_report.json`
                    Control: `experiments/run_reports/2026-08-05_v2.4_324420_graded_report.json`
**commands**
```powershell
.\scripts\preflight.ps1                                      # GREEN 5/5
uv run whest validate --estimator estimator.py               # pass, 62 ms
uv run whest run --estimator estimator.py --dataset hf://aicrowd/arc-whestbench-public-2026@v1-phase1 --split mini --runner subprocess
uv run whest submit --estimator estimator.py                 # after explicit approval
uv run python local_sim/fetch_graded_report.py 327743 2026-08-10_v2.4.1_phase1
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
| **grader**, v2.4.1 | 10.022 | 327743, 2026-08-10 |
| paired grader ratio v2.4.1/v2.4 | **1.0180×**, 95% CI [1.0119, 1.0241], n=100 | — |
| local, v2.4.1, `--runner local` | 70.8 | LOG 2026-08-06 S2 |
| local, v2.4.1, `--runner subprocess` | 42.94 | this slice, 2026-08-10 |

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
machine state (2026-08-06 vs 2026-08-10) — and this slice cannot separate them. The
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
                   corrected from 2026-08-07 to 2026-08-10 before submission (driver
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
- *(Unrelated to this entry: see the 2026-08-11 Skeptic review of the control-variate
  kill, which withdraws the "≥58% bias" reading and establishes that our final-layer
  residual is ~99.9% variance.)*
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

### REVIEW — independent Skeptic on 327743. **Verdict: PROVISIONAL**

Run 2026-08-10 in a separate agent context that never saw the builder's reasoning,
briefed with the contract, the commits, the protocol and the raw artifact paths only.
Verdict relayed without softening. The builder then re-verified each finding against
the artifacts before accepting; all four checked findings **confirmed against the
builder**.

**What the Skeptic upheld.** It recomputed the headline from the two JSONs
independently and reproduced every quoted digit: suite identity 100/100 `mlp_name`
in order with identical `sampling_mse`; paired geometric mean **0.966508**, CI
[0.7969, 1.1721]; ratio of means 1.031894; medians 1.010707; 22/50 with exact
one-sided p = 0.8389; counted FLOPs 0.99981, effective 1.00015, residual 1.01800
[1.01189, 1.02414]. Integrity: `estimator.py` in **neither** commit, byte-identical
to `estimator_v2_4_1.py` (sha256 `59c371a5…`), vault counter 1 against exactly 1
touch-log line, no seed block spent, `AGENTS.md` "~7×" unedited as required,
`docs/writeup-draft.md` untouched so no overreach leaked into the deliverable.
**The primary gap is genuinely closed: v2.4.1 is graded, 0 of 100 failed, safe to
designate.** It also proved claim 3 *stronger* than the builder had: layer-31-only
divergence holds on **all 50** networks (max relative difference across layers 0–30
exactly 0.0 over 50 × 31 comparisons), not the single network the packet cited.

**Defects found — five, all accepted.**

1. **The date was wrong by one day, twice over, and the "correction notice" was
   itself unverifiable.** The graded report carries `createdAt 2026-08-10T19:00:37Z`
   and both commits are stamped 2026-08-10 19:12Z. The session was **2026-08-10**,
   not Aug 9 (nor the Aug 7 first written). The contract's correction cited "git log
   puts the last commit at 2026-08-07" — **a lower bound on the present, which cannot
   date anything** — and a run-report stamp that exists in no file on disk.
   Consequence beyond pedantry: **327743 went in ~5 hours before round 1428 closed**,
   not with the day-and-a-half margin the contract's deadline paragraph implied.
   *Fixed:* all occurrences corrected across LOG, `CURRENT_STATE.md`, `GOTCHAS.md`,
   and the artifact renamed to `2026-08-10_v2.4.1_phase1_327743_graded_report.json`.
2. **"Purely a function of which local path was used" is false, and the falsifying
   artifact was already on disk and already cited in the contract's own evidence
   list.** A `--runner subprocess` reading of the same byte-identical file exists from
   2026-08-06 at **54.47 ms/MLP** — recovered from
   `2026-08-06_v2.4.1_mini_subprocess.json` and confirmed independently from its
   recorded multiplier 0.216 with counted 5.330e10. Runner effect (same day)
   **1.301×**; machine-state effect (runner fixed) **1.269×**; product 1.651 = the
   observed 70.85/42.94. The confound the packet called unseparable was **half
   separated by data in hand**, and "~65% too large" is ~30% on the same-day
   comparison.
3. **"Sprint 4 S2 and the D14 residual model both scaled by ~7" is false — two
   different sevens were conflated.** D14's deciding model is *local **ratio** ×
   graded anchor*: `9.84 × (490.4/62.3) = 9.84 × 7.87`. That 7.87 is a within-run
   candidate/control ratio, invariant to uniform rescaling of this machine; the
   local→grader level never enters, and model 1 (local absolute) is explicitly flagged
   "never decides". **Withdrawn.** The Skeptic identified the *real* exposure in its
   place, which was recorded nowhere: **the candidate/control ratio itself reads 5.47×
   in one local harness and 7.87× in another — a 44% spread on the quantity that
   actually decides D14.**
4. **"One network contributes 140% of the delta" mis-attributes the effect.** 140.2%
   recomputes, but three networks each individually flip the sign of the total
   (idx 37, 14, 1), and the 2.1246e-6 total is the residue of +1.9102e-5 in gains
   against −1.6978e-5 in regressions. Correct description: *a small residue of
   near-cancelling sums*. The promoted diagnostic also has a measured false-positive
   rate — simulating a true 5% improvement at the observed scatter (sd log ratio
   0.696) fires it in ~10% of 2000 draws — so it is a screen, not a proof.
5. **"The paired statistic says no effect" overstates, and a better-powered answer
   already existed.** At n = 50 the design can only exclude |effect| > ~20%;
   `docs/GOTCHAS.md` already recorded **0.9959× over 32 draws/MLP** for exactly this
   A/B, and already warned that v2.4.1's mini run "looks 6.7% better than v2.4 — that
   is luck, not method." The grader run **confirms** the repo's existing finding on
   grader data; it did not discover it. The GT-noise floor was also not quoted in the
   result section, as `docs/VALIDATION.md` §3.5 requires.

**Also flagged:** the 42.94 ms/MLP figure has **no saved artifact** (the contract's
Scope promised a mini-run JSON; none was written) — internally consistent but not
reproducible by the next session. The +1.8% grader residual's n=100 CI is
pseudoreplication for a between-submission question (effective n = 1), and
`residual_s` is a subtraction remainder whose components moved ~78 ms while the
remainder moved 0.177 ms, so it must not be attributed to the `eigh` path.
`docs/BENCHMARK.md` still lists 324420 as the graded row and was missing from the
packet's docs-to-change list.

**The Skeptic rejected the packet's proposed follow-up**, correctly: ranking the
"both runners, same day" run as highest-value rested on the false premise that no
same-day runner pair existed. **Cheapest discriminating follow-up, adopted:** three
`--runner subprocess` repeats plus one `--runner local` on the unmodified file, same
day, each saved with `--format json`. If the three repeats span >10%, the conclusion
is not "4.3× not 7×" but *"this machine has no quotable absolute residual level"*.

**What this review changes about the slice.** The **primary** result stands and is
promoted: v2.4.1 is graded, safe, equivalent to v2.4, and the designation gap is
closed. The **secondary** "substantive finding" is downgraded from a discovery to a
partially-confounded observation, and three of its claims are withdrawn from the
durable docs. The builder's own falsifier never fired because it was pointed at the
grader residual while the doubtful quantity was the local one — and the contract said
so, which is the only reason it reads as a recorded defect rather than a concealed
one.

**Meta-lesson, and it is the most transferable thing here.** Every one of the five
defects was findable from artifacts already in the repository at the time the claims
were written; none needed a new measurement. The builder's error was not bad
arithmetic — every number it computed reproduced exactly — it was **reaching for a
narrative ("the grader is stable, we are not") before checking whether the repository
already contained the disconfirming case.** The 54.47 ms/MLP figure was sitting in
the contract's own evidence list, quoted as "multiplier 0.216".

## 2026-08-10 — BOARD DECOMPOSITION: the honest frontier is **2–3.6× ahead of us on ACCURACY**, and it is not an operating-point difference

ROUTINE diagnostic, no seed block, no submission, no vault touch. Snapshot
`experiments/board_snapshots/2026-08-10T2048Z_phase1_close_full_board.json` (200
entries, round 1428, 20:48Z).

**The method.** The board publishes `scoreSecondary` alongside `score`, and
`score = MSE × max(0.1, C/B)`, so `score / scoreSecondary` **recovers every team's
C/B**. Verified on ourselves: 2.6314e-7 / 1.3323e-6 = 0.1975 against the graded
0.19896. The board therefore reports each team's **accuracy and compute separately**
— it is not a single opaque number, and we had never decomposed it before.

**Where we actually stand.** Us: MSE **1.3323e-6** at C/B **0.198**.

| rank | team | MSE | C/B | score | MSE vs us | entries |
|---|---|---|---|---|---|---|
| 4 | J2W | 2.784e-7 | **0.194** | 5.39e-8 | **4.8×** | 1093 |
| 5 | dstepanov | 2.199e-7 | 0.283 | 6.22e-8 | 6.1× | 611 |
| 7 | yanggan_gu | 2.422e-7 | 0.299 | 7.24e-8 | 5.5× | **31** |
| 8 | thylinao | 1.724e-7 | 0.439 | 7.56e-8 | 7.7× | 170 |
| 19 | sweaty_dog | 1.296e-7 | 0.874 | 1.133e-7 | 10.3× | 102 |
| 22 | andrew_epstein | 2.508e-7 | 0.462 | 1.158e-7 | 5.3× | **52** |
| 25 | mohanty | 2.037e-7 | 0.607 | 1.236e-7 | 6.5× | **18** |
| — | **us (327743)** | **1.3323e-6** | **0.198** | **2.6314e-7** | 1.0× | 5 |

**The decisive row is J2W.** They run at **C/B 0.194 against our 0.198** — the same
compute to within 2% — and get **4.8× lower MSE**. There is no operating-point
confound to explain away: at our own budget, someone is 4.8× more accurate.
`yanggan_gu` (31 entries), `andrew_epstein` (52) and `mohanty` (18) reach 2–3.6×
better *scores* on entry counts too low for meaningful public-set probing.

**More compute does not close it — measured, not assumed.** Our own N=65,536 arm,
projected onto the **grader's** cost structure rather than this machine's
(which over-bills residual 4–7×): counted 1.910e11, C/B 0.717, MSE 3.667e-7 (using
the CONFIRM-measured 3.63× MSE ratio) → projected score **2.630e-7 = 0.999× of
current**. Even at *zero* residual it is 0.979×, inside the ~18% run-to-run noise.
The `N` lane stays closed, now confirmed against graded cost anatomy.

**Reconstruction screen, on physics rather than entry counts.** The metric's
GT-noise floor is ~2e-10 and the best *published* honest method is 1.551e-7 MSE:

| team | MSE | × GT floor | vs best published honest | entries | read |
|---|---|---|---|---|---|
| alexander_luz | 1.0e-9 | 5.0× | **155× better** | 259 | not estimation |
| rayan53 | 1.35e-8 | 67× | 11.5× better | 50 | implausible |
| ednacob | 3.63e-8 | 182× | 4.3× better | 169 | implausible |
| J2W and below | ≥2.78e-7 | ≥1392× | *worse* than published | — | plainly legitimate |

Only the top three are implausible. **From rank 4 down the board is real**, which
means our true standing among honest teams is roughly **#102 of 197**, not the #82
of 192 the 2026-08-06 census estimated from entry-count heuristics. The earlier
screen was too generous because it never looked at accuracy.

**Knowledge Delta — this reopens the score question as a lane.** `GAME_PLAN` rev 6
declared "the honest sampling frontier is ~2.7e-7" and `docs/CURRENT_STATE.md`
records score optimisation CLOSED on that basis. **That was a statement about our
own method's frontier, and the board now shows it is not the field's.** Honest teams
at our exact compute are 4.8× more accurate. Per the reopening rule this is
*new external evidence* — competitor telemetry — and it is the strongest such
evidence the project has seen. The open question "319341's variance advantage,
1.216×" was measuring a corner of a much larger gap.

**What it does NOT tell us.** The mechanism. MSE alone cannot distinguish a better
variance reduction from a better analytic core from a fundamentally different
estimator. That is the strategist's question.

## 2026-08-10 — SLICE CONTRACT: layer-local control variate, DESIGN on D16 → DECISION on D17  [PRE-REGISTERED, no data yet]

**Origin.** Strategist Packet rev 2 (issued 2026-08-10 after the board decomposition
reopened the accuracy lane). The returned contract was re-verified against
`estimator.py` before allocation, per `ai/workflows/next-slice.md`; four corrections
were made and are recorded below rather than silently applied.

**Strategic question**
Does removing the *transported layer-1 sampling error* at every layer recover a
material share of the 4.8× accuracy gap to the honest field, without changing the
sampled forward path?

**Current bottleneck**
Accuracy, not cost. Board decomposition (2026-08-10) shows honest teams at our own
compute (J2W, C/B 0.194 vs our 0.198) reach **4.8× lower MSE**. Our graded raw MSE
is 1.3323e-6 at C/B 0.198.

**Evidence**
- CONFIRMED — layer 1 has an exact closed-form mean, error 1.8e-16
  (`estimator.py:682`, invariant suite check 8).
- CONFIRMED — the sampled path is a **single trajectory through all 32 layers**
  (`estimator.py:542-546`): `h` is overwritten by `max(h @ sliced[li], 0)` each
  layer. Sampling error therefore compounds with depth. *Verified by reading the
  code, not assumed.*
- CONFIRMED — the α ≥ 3 band carries **76.5%** of final-layer MSE (LOG 2026-08-05,
  Track T).
- CONFIRMED — propagating on-neurons linearly for eight layers changes MSE by
  **0.07%** (LOG 2026-08-05), which is what licenses a first-order transport model.
- CONFIRMED — deterministic dead-neuron offsets (`estimator.py:544-545`) are
  identical across samples and so inject **no sampling error**; the contract's
  "deterministic offsets cancel" is correct.
- CONFIRMED — `d/dμ E[ReLU(μ + σZ)] = Φ(μ/σ)` exactly, so the closure's `Φ` is the
  correct first-order transport gain, not a fitted coefficient.
- CONFIRMED — historical whole-layer scale error is worth ~1.9× if removed exactly,
  but every end-of-network predictor for it failed (best R² ≈ 0.15).
- **NOT SUBSTANTIATED** — the strategist's claim that "closure `Φ(α)` predicts firing
  probabilities to four decimals" appears nowhere in the repo. Nearest evidence is
  indirect (pilot reordering was an exact no-op "because the closure's firing rates
  are already exact", LOG 2026-08-06). **Φ's accuracy is therefore a D16 precondition,
  not an input.**
- SPECULATIVE — the recursion removes 2–4× of final MSE (strategist). See *Ceiling*.

**Hypothesis**
With `s_l` the sampled post-ReLU mean at layer `l` and `mtilde_1` the exact
first-layer mean,

`mtilde_l = s_l − (((s_{l-1} − mtilde_{l-1}) @ W_l) * Φ_l)`

using the existing sliced weights, alive indices and closure `Φ`, reduces
final-layer MSE at effectively unchanged cost.

**Mechanism**
Layer 1 is the one place we know the truth, so `s_1 − mtilde_1` is our sampling
error *observed exactly*. To first order a mean perturbation `δ` entering layer `l`
moves that layer's post-ReLU mean by `(δ @ W_l) · Φ_l`. Subtracting it at each layer
stops a measurable error from being amplified through the remaining depth.

**Ceiling — the correction is narrower than the strategist stated, and this is the
main risk.** Telescoping the recursion, `e_l = s_l − mtilde_l` unwinds to the
layer-1 error transported forward by `∏(W·Φ)`. Errors **originating** at layers 2–31
are never corrected, because they are unobservable. The method's ceiling is therefore
*the share of final-layer MSE whose origin is layer-1 sampling error*, and **nothing
in the repository bounds that share.** D16 exists to bound it before D17 is spent.

**Intervention**
A fenced lab probe implementing exactly the recursion above. Sampled activations,
point set, shifts, pruning, rotation and N are all unchanged; only the reported
running mean changes. `estimator.py` is NOT edited in this slice.

**Corrections made to the returned contract** (all before any data):
1. **Split into DESIGN (D16) and DECISION (D17).** The returned contract gated on the
   same block it would implement and debug on. Debugging is tuning; that is the
   failure mode that cost Sprint 4 two results (D11→D12, D13→D14 exist for this).
2. **`s_l` does not currently exist.** `estimator.py:548` takes `h.mean()` once, at
   the end; rows 1..30 are filled from the *closure* (`mus`), which is why
   `all_layers_mse` 4.83e-5 is 36× worse than `final_layer_mse` 1.33e-6. The probe
   must add 32 reductions — cheap (~0.25% counted) but genuinely new code, not a
   reinterpretation of existing values.
3. **Two-tier gate** (below). The returned single gate recorded a project-best result
   as a failure: 1.25× accuracy at ~0.3% cost is 2.6314e-7 → ~2.1e-7, the largest
   single gain in this project's history.
4. **Φ accuracy demoted from evidence to precondition** (see above).

**Prediction — driver**
Mechanism is correct but narrow. Central estimate **G ≈ 1.1–1.6×**, with material
probability of ~1.0× if the layer-1-originated share is small. Counted overhead
**<0.5%** (estimated 0.25%).

**Prediction — Samuel (registered before any data, verbatim)**
> 1. "I don't know" (what part of the system it changes — taught, then recorded)
> 2. "A slight improvement"
> 3. "A major reduction in score/performance" would disprove it

Read: Samuel predicts a positive but modest effect, i.e. closer to the driver's
1.1–1.6× than to the strategist's 2–4×. **Three-way registered disagreement:
strategist 2–4×, driver 1.1–1.6×, Samuel "slight".** Samuel's falsifier is a
*regression*, which is weaker than the pre-registered KILL condition (CI including
1.0); the gate below is the binding one.

**Gate — two-tier, fixed NOW, before D16 or D17 is touched**
On 200 D17 MLPs × 8 identical digital shifts,
`G = exp(mean_j(log(MSE_control,j / MSE_candidate,j)))`:

| tier | threshold | consequence |
|---|---|---|
| **EXPLAINS THE GAP** | `G ≥ 2.0` **and** 95% CI lower bound ≥ 1.5 | strategist's claim validated |
| **PROMOTE** | `G ≥ 1.25` **and** 95% CI excludes 1.0 | real and bankable; proceed to a cost + estimator slice |
| **KILL** | 95% CI includes 1.0 | layer-1-originated error is not a material share; lane closes |

All outputs must be finite and GT noise must be <10% of candidate MSE in every tier.
Arithmetic means and worst-decile behaviour are reported but do not replace the
paired gate.

**Baseline**
Unmodified v2.4.1 pruned/rotated Sobol computation at N=16,384, on identical samples
and shifts (common random numbers).

**Residual model**
**Not consumed — the gate is accuracy-only.** Counted overhead is recorded. **No
adjusted-score claim, no submission, and no `estimator.py` edit is authorised by a
passing D17 result**; those require a subsequent slice reporting the three residual
models and a clean grader-style mini run. This is deliberate: the local residual is
E1 and currently reads 4.28×–7.07× depending on path.

**Statistic**
Paired, common random numbers. Per-MLP MSE averaged over 8 shifts; decision
statistic is the geometric mean of per-MLP ratios with a 95% log-ratio CI. Never a
ratio of means. Per the 2026-08-10 review, also report the fraction of the aggregate
delta carried by the single largest mover.

**Confounders**
Φ is Gaussian-optimal but the true transported gain is not exactly Φ · existing
pruning bias (the dead-neuron offset uses the *closure* mean, which the recursion
does not correct) · high correlation between successive recursive errors · GT noise ·
heavy-tailed MLP difficulty (max/median 12.93×).

**Scope**
`local_sim/seed_blocks.py` (D16, D17, SINGLE_USE_BLOCKS) · a fenced lab probe ·
raw JSON under `experiments/` · this LOG entry. **`estimator.py`: unchanged.**

**Do not change**
N, Sobol point set, digital shifts, `_PRUNE_TAU`, rotation, `_BASIS_RANK`, closure
equations, `beta = Φ`, block size after measurement, or the gate.

**Evidence required**
Preflight GREEN before D17 is touched · exact commands and commit sha · raw per-MLP
per-shift control and candidate MSE · paired ratios with CI · GT-noise floor as a
fraction of candidate MSE · counted-operation estimate · **zero vault touches**.

**Driver** Claude Code.

### RESULT — **KILLED BEFORE MEASUREMENT. D16 and D17 were allocated but never touched.**

No data was taken. No seed block was spent. No vault touch. `estimator.py` unchanged.
The lane was closed by evidence already in this repository, found during the
pre-build verification the workflow requires.

**The finding.** `experiments/LOG.md` line 135 (2026-08-05, `local_sim/cv_lab.py`),
restated independently in `docs/GAME_PLAN.md` line 41:

> Control variates are a dead end here. Layer 1 has exactly-known moments, so it is
> a legitimate zero-bias control variate. Ceilings measured by per-sample covariance:
> scalar CVs 1.1×; **full 256-dim h₁ vector 1.80×**; even the *illegal* ceilings using
> h₂/h₃ only reach 2.1–2.5×. **And stacked on Sobol the CV adds nothing
> (2.04× → 1.98×)** — QMC and the CV remove the same variance. Dropped.

**Why it applies to this exact proposal, checked rather than assumed.** Telescoping
the recursion: `e₁ = ε₁`, `e₂ = (ε₁ @ W₂)·Φ₂`, `e₃ = ((ε₁ @ W₂)·Φ₂ @ W₃)·Φ₃`, so
every `e_l` is a **fixed linear map of ε₁** and the final correction is `M(ε₁)` with
`M = ∏(W_l·Φ_l)`. That is precisely a control variate with control `mean(h₁)` and
known mean `E[h₁]` — the 256-dim layer-1 vector CV — carrying an *analytic*
coefficient where `cv_lab` measured the *optimal fitted* one. A fixed coefficient
cannot beat the optimum, so:

| bound | value | source |
|---|---|---|
| ceiling of this mechanism class, standalone | **1.80×** | cv_lab, per-sample covariance |
| ceiling including *illegal* h₂/h₃ information | 2.1–2.5× | cv_lab |
| **the same CV stacked on Sobol — our configuration** | **~1.00×** (2.04→1.98) | cv_lab |
| share of residual that is bias, not variance | **≥58%** | forum 18085 |

The strategist predicted **2–4×**, which exceeds the measured ceiling of the whole
class even before the Sobol result is applied. Our estimator uses Sobol
(`estimator.py:534-541`), which is the configuration measured at ~1.00×.

**The "per-layer, not end-patch" novelty does not survive either.**
`docs/RESEARCH_NOTES.md` line 282 correctly identifies per-layer correction as the
untested angle, but that would only be new if it corrected error *originating* at
layers 2–31. This recursion cannot: layer 1 is the only layer whose truth we know, so
every term traces back to ε₁. It is the end-patch in recursive form.

**What was nonetheless verified and is worth keeping.** The strategist's mechanism is
mathematically correct and its architectural claims held up against the code:
`d/dμ E[ReLU(μ+σZ)] = Φ(μ/σ)` exactly; the sampled path really is one trajectory
through all 32 layers (`:542-546`); deterministic dead-neuron offsets inject no
sampling error (`:544-545`); the row convention matches. **The idea was sound and the
arithmetic was right — it was already measured.**

**Root cause, and it is the driver's.** The Strategist Packet's "experiments already
tried and rejected" section listed nine killed lanes and **omitted the control-variate
result**, the one directly on point. The packet's own text claims that section
"prevents the recipient re-proposing a mechanism already measured and killed." The
structural cause is that `docs/CURRENT_STATE.md`'s closed-lane table has eight rows
and does not include control variates — the kill lives only in a LOG line and a
GAME_PLAN Phase-A paragraph, neither of which a packet author reads by default.
**Fixed: the closed-lane table now carries the control-variate row.**

**Blocks D16 and D17 are UNSPENT and remain available.** No experiment has drawn a
single seed from either. They are recorded in the registry as allocated-then-released
so the audit trail shows why, and they may be reused by a future experiment.

### Knowledge Delta

- **Learned.** The layer-1 exact-mean control variate has a measured ceiling of 1.80×
  standalone and ~1.00× on top of QMC, and any correction that is a linear function
  of the layer-1 sample-mean error — including a recursive per-layer one — lies
  inside that class.
- **Belief changed.** "Correct per-layer rather than patch the end" was carried as an
  *untested* angle (RESEARCH_NOTES 282). It is untested only for errors originating
  below layer 1, which are unobservable. As stated, it is not a distinct lane.
- **New failure signature.** *A proposal whose correction telescopes to a fixed linear
  map of a quantity we already built a control variate from.* Diagnostic: unroll the
  recursion and ask what it is ultimately a function of. If the answer is a statistic
  whose exact mean we know, it is a control variate and its ceiling is already
  measured.
- **Process defect, now fixed.** A Strategist Packet is only as good as its
  rejected-lanes section, and that section was assembled from `CURRENT_STATE.md`'s
  table rather than from the LOG. Any future packet must be checked against the LOG's
  full kill record, not the summary table.
- **Cost of the error.** Zero blocks spent, zero GPU hours, ~40 minutes of driver
  time. Caught by the mandatory pre-build verification, which is the second time
  today that step has prevented a wrong claim.
- **What remains open.** The 4.8× accuracy gap is untouched and remains the
  bottleneck. Variance reduction of the layer-1-anchored *linear* kind is bounded out.
- ~~"Combined with 18085's ≥58%-bias finding, the gap is probably mostly bias, which
  points at the closure/calibration core rather than at the sampler."~~
  **WITHDRAWN 2026-08-11 by the independent Skeptic — this was backwards, and it had
  already been written into `docs/CURRENT_STATE.md`'s #1 open question.** The ≥58%
  figure describes *radiant-allomancer's own* estimator (RQMC + shrinkage + Edgeworth
  + ridge corrector, 4.47e-7), not ours (1.3323e-6), and our own data contradicts it:
  `MSE×N` = **0.03657 / 0.04036 / 0.03666** at N = 16,384 / 32,768 / 65,536
  (24 MLPs × 8 shifts, α = 1.00 — LOG 2026-08-06). Solving `MSE(N) = V/N + B²` on the
  4× endpoints gives **B² ≈ 1.8e-9 ≈ 0.08% of MSE**. **Our final-layer residual is
  essentially all variance.** The correct implication is the opposite of what was
  written: the 4.8× gap is a **variance** gap, and the next Strategist Packet must be
  aimed at variance mechanisms, not at the closure/calibration core.

### REVIEW — independent Skeptic on the control-variate kill. **Verdict: ACCEPT**

Run 2026-08-11 in a fresh context briefed with artifacts only. Relayed without
softening; the builder verified each finding against the artifacts before accepting.

**The Skeptic did not argue the kill — it measured it.** Finding that the decisive
"~1.00× stacked on Sobol (2.04 → 1.98)" number had **no reproducible artifact**
(`local_sim/cv_lab.py` contains no Sobol at all — `collect()` draws
`rng.standard_normal` at line 43; the only committed CV-on-QMC arm,
`local_sim/impl_lab.py`, uses a Cranley–Patterson *real* shift scored at 1.24× in
LOG line 143, and the "2.04×" in that table is a **point-set comparison**, not a CV
stack), it rebuilt the experiment in the shipped configuration: unscrambled Sobol +
30-bit digital XOR shift exactly as `estimator.py:540`, N = 16,384, real public MLPs
against 1e9-sample truth, one shared forward pass (perfect common random numbers),
14 MLPs × 6 shifts, paired geometric mean.

| point set | correction | paired geo-mean gain |
|---|---|---|
| plain MC | fitted-optimal 256-dim h₁ CV | 1.584× |
| plain MC | **the strategist's fixed analytic `M = ∏(W_l·Φ_l)`** | 1.383× |
| Sobol + digital shift (**ours**) | fitted-optimal CV | 0.992× |
| Sobol + digital shift (**ours**) | **the strategist's exact `M`** | **1.010×, 95% CI [0.979, 1.042]** |

**The CI includes 1.0 — the contract's own pre-registered KILL condition** — and
excludes both the 1.25× PROMOTE tier and the strategist's 2–4×. The kill stands, now
on an artifact rather than by inheritance.

**What the Skeptic UPHELD in the builder's argument.** The telescoping algebra is
correct: `e_l = (e_{l-1} @ W_l)·Φ_l` with `e_1 = ε_1` gives
`e_L = ε_1 @ ∏(W_l diag(Φ_l))`, a fixed linear map of `ε_1` alone. It independently
verified the load-bearing premise that **`Φ_l` never sees a sample** (`_closure`,
`estimator.py:574-595`, is a function of `w64` only). It checked every candidate
escape hatch and found no missed term: pruning restricts the map to alive columns
(still linear, and smaller); dead-neuron offsets are deterministic and cancel in the
difference; the folded rotation is orthogonal and pre-layer-1, leaving `x ~ N(0,I)`
and `ε_1` exactly observable. It also measured *how tight* the containment is — the
analytic `M` reaches 87% of the fitted optimum on plain MC, so the fixed coefficient
is a **good** coefficient; it wins nothing because QMC has already removed the
variance it targets. `mc/cv_fitted` (1.3894e-6) and `sobol_digital/plain` (1.4077e-6)
coincide, making the "QMC and the CV remove the same variance" mechanism directly
visible and predicting its *magnitude*, not just its sign. Hygiene clean: `estimator.py`
byte-identical and untouched, vault 1/1, `check_disjoint()` passes, and the strings
`30_806_016`/`30_806_017` appear **only** in `seed_blocks.py` — D16/D17 genuinely
never drawn from.

**Defects found — four, all accepted.**

1. **"Optimal fitted β ≥ any fixed analytic β" is FALSE in the configuration that
   decides.** `cv_lab.py:54-74` fits β to minimise per-sample covariance, which is the
   variance-minimising coefficient **under plain MC only**; under an RQMC point set it
   optimises the wrong objective. Measured: on digital-shift Sobol the fitted
   coefficient scores 0.992× while the analytic `M` scores 1.010× — **the bound is
   beaten by the thing it was supposed to bound.** It happens not to change the
   verdict (both are 1.0), but **this argument must never be reused.**
2. **The "≥58% bias" cap is inapplicable, and the conclusion drawn from it was
   backwards.** See the withdrawal above. This is the most consequential finding of
   the review, because the wrong conclusion had already reached
   `docs/CURRENT_STATE.md`'s #1 open question.
3. **The 1.80× standalone ceiling is weaker than the durable docs present it.**
   `cv_lab.py:135-138` aggregates as `mean(v0)/mean(v)` — **a ratio of means, the
   exact statistic `docs/VALIDATION.md` §2b forbids** — over **3 MLPs** of the public
   mini split, on the dense unpruned/unrotated network. The Skeptic's paired geometric
   mean over 14 MLPs reads **1.584×**, not 1.80×. Both `docs/CURRENT_STATE.md` and
   this LOG quoted 1.80× as settled without any of those qualifiers.
4. **Tier violation, disclosed rather than hidden.** The kill was decided entirely on
   CONFIRM-tier public-split data, which `docs/VALIDATION.md` §0 says may not be used
   to choose anything. The Skeptic's confirmation shares the defect and says so. The
   effect size (1.01× against a 1.25× gate) is far outside any plausible
   DECIDE/CONFIRM difference (documented at +1.4%), so the verdict survives — but the
   provenance does not meet the protocol.

**Scope correction to the closed-lane row.** "Any linear form" is the honest scope. A
correction using an **empirical** per-layer gain (the free `mean(z_l > 0)` from the
forward pass) instead of the analytic `Φ_l` is a *nonlinear* function of the sample
and formally escapes the linear-CV class. The same overlap mechanism suggests it is
also worthless, but **no measurement covers it** and the kill must not be cited
against it.

**Also unmeasured:** the CV has never been run on the actual v2.4.1 path (pruned +
rank-32 rotated). The Skeptic argues both restrictions shrink the CV's headroom
further — the rotation deliberately aligns the best-equidistributed Sobol dimensions
with the dominant directions of the linearised map, which are exactly the directions
the layer-1 CV exploits — but flags this as a directional argument, not a measurement.

**Recommended follow-up, adopted:** re-measure `MSE×N` for the shipped file at
N ∈ {16,384, 65,536} on **fresh canonical mints**, 200 MLPs × 8 shifts, and report the
implied bias floor `B²`. It decides whether the next Strategist Packet targets
variance or calibration — the single highest-leverage question in the project — and
**D16 is clean, already allocated to this lane, and sufficient.** No single-use block,
no vault touch, no submission. The Skeptic also recommends promoting its CV probe into
`local_sim/` so the kill owns a committed artifact.

## 2026-08-11 — VARIANCE-OR-BIAS: the shipped file's bias floor on fresh canonical mints (D16)

**Class: IMPORTANT** (a new probe's math + a design block). Not STRATEGIC: D16 is a
*design* block, not a decision block; `estimator.py` is untouched; no gate is being
decided, a physical quantity is being measured. Written prediction below is recorded
**before the probe was run**.

**Why this is the highest-value move.** The 2026-08-11 Skeptic review withdrew the
"the gap is mostly bias" conclusion and replaced it with "the residual is ~99.9%
variance", derived from `MSE×N` = 0.03657 / 0.04036 / 0.03666 (LOG 2026-08-06,
24 MLPs × 8 shifts, **local mints, one configuration**). That single inference now
steers open question #1 — whether the next Strategist Packet attacks **variance
mechanisms** or the **closure/calibration core**. It is worth one clean measurement
before Phase 2 opens on 18 Aug.

**A defect in the previous inference, found while designing this and frozen into the
model here.** Ground-truth labels are themselves Monte Carlo averages, so

`MSE_measured(N) = E[(pred_N − truth − η)²] = MSE_true(N) + σ_gt²`

with `η` the GT-sampling noise, independent of the estimator. Therefore

`MSE_measured(N) = V/N + B² + σ_gt²`

and **σ_gt² is perfectly confounded with the bias floor B²: GT noise masquerades as
bias, exactly.** Varying `N` alone cannot separate them. The vault's GT noise at
2²⁵ samples is **1.575e-9**, and the B² the Skeptic derived was **≈1.8e-9** — the
same number. **The previously inferred bias floor is plausibly nothing but the GT
noise of its own labels.** Any measurement that does not subtract σ_gt² explicitly
cannot answer this question.

**Frozen model and estimator (before any D16 seed is drawn):**
- `MSE_measured(N) = V/N + B² + σ_gt²`, fitted on the two endpoints
  N ∈ {16,384, 65,536}.
- `σ_gt²` is **computed, not fitted**: per MLP, `mean_i Var_i(final-layer ReLU) / n_gt`,
  accumulated in the same GPU pass that builds the truth (the method `vault.py:114-120`
  already uses, giving the vault's audited 1.575e-9).
- `B² = (MSE(N₂)·N₂ − MSE(N₁)·N₁)/(N₂ − N₁) − σ_gt²`.
- Reported with a paired bootstrap CI over MLPs. Paired common random numbers: the
  same MLPs and the same 8 digital shifts at both N.

**Prediction — driver, registered before running**
1. `B²` will be **consistent with zero** — 95% CI including 0, and its upper bound
   below **5.6e-8** (10% of MSE at N=65,536).
2. Once `σ_gt²` is subtracted, the apparent bias floor will **drop by roughly the
   vault's 1.575e-9**, i.e. most or all of the previously inferred 1.8e-9 will prove
   to be label noise rather than estimator bias.
3. `α` (the exponent in `MSE ∝ N^−α`) will come out at **1.00 ± 0.03**, reproducing
   the local-mint result on canonical mints.
4. The variance-dominance conclusion will **strengthen, not weaken**.

**Falsifier.** `B²` significantly greater than zero after subtracting σ_gt², or
`α` materially below 1.0, would mean a real bias floor exists, the Skeptic's
correction was itself wrong, and the next packet must target the closure/calibration
core after all.

**Scope.** New fenced probe `local_sim/bias_floor_lab.py`; raw JSON under
`experiments/`; this entry. **`estimator.py` unchanged. No vault touch. D17 untouched.**

**Driver** Claude Code.

### RESULT — **the residual is 99.4% VARIANCE. All four registered predictions held.**

**commands**
```powershell
.venv-gpu\Scripts\python.exe local_sim/bias_floor_lab.py --mlps 200 --shifts 8 --n-gt 16777216
.\scripts\preflight.ps1        # GREEN 5/5, invariants 30/30
```
**config** block **D16** (design), 200 fresh canonical mints × 8 digital shifts,
paired common random numbers, N ∈ {16,384, 65,536}, GT 2²⁴ = 16,777,216 samples/MLP,
RTX 5090, 4,203 s. **Vault touches: 0. `estimator.py` unchanged. D17 untouched.**
**raw artifact** `experiments/bias_floor_D16.json` (per-MLP MSE at both N, per-MLP
σ_gt², seeds).

| quantity | value |
|---|---|
| MSE(N = 16,384) | 1.635761e-6 |
| MSE(N = 65,536) | 4.130391e-7 |
| **MSE × N at N₁ / N₂** | **0.02680 / 0.02707 — flat** |
| **α** (MSE ∝ N^−α) | **0.9928**, 95% CI **[0.9586, 1.0274]** — includes 1.00 |
| σ_gt² (label noise, computed not fitted) | 3.0925e-9 |
| B² **before** subtracting label noise | 5.4652e-9 |
| **B² after subtracting label noise** | **2.3727e-9**, 95% CI **[−2.460e-8, 2.840e-8]** — **includes 0** |
| **B² as a share of MSE(N₂)** | **0.574%** |

**The design defect was real and it accounted for the majority of the effect.**
Subtracting the computed label noise moved the apparent bias floor from 5.4652e-9 to
2.3727e-9 — **57% of what looked like estimator bias was the noise in its own
ground-truth labels.** The measured σ_gt² = 3.0925e-9 at 2²⁴ samples is the expected
2× of the vault's audited 1.575e-9 at 2²⁵ (half the samples, double the noise),
which is an independent check that the subtraction is calibrated and not a fudge.

**Predictions, scored against what was written before the block was drawn:**

| # | registered | outcome |
|---|---|---|
| 1 | B² consistent with zero, upper bound < 5.6e-8 | **HELD** — CI [−2.46e-8, **2.84e-8**] includes 0 |
| 2 | apparent floor drops by ≈ the vault's 1.575e-9 once σ_gt² comes out | **HELD** — dropped 3.09e-9, and 3.09e-9 ≈ 2 × 1.575e-9 exactly as the sample count implies |
| 3 | α = 1.00 ± 0.03 on canonical mints | **HELD** — 0.9928 [0.9586, 1.0274] |
| 4 | variance-dominance strengthens | **HELD** — 99.4% variance |

**Level note, quoted honestly.** `MSE × N` here is 0.0268 on canonical D16 mints
against 0.0366 on the older *local* mints (LOG 2026-08-06) and 0.022525 on the
**graded** suite. The canonical-mint level sits much closer to graded than the old
local level did, which is what the 2026-08-06 provenance fix was supposed to buy. The
*shape* — flatness — is what this slice measures, and it reproduces on both.

**deviations** None. **unexpected** That label noise was the majority of the previously
inferred floor, not a minor correction.
**remaining doubt** α = 0.9928 is marginally below 1.0; the CI includes 1.00 but also
0.96. A genuine α slightly under 1 would mean QMC is doing *slightly worse* than
plain-MC scaling at this N, which is worth knowing but does not change the
variance/bias verdict. Two endpoints cannot distinguish a small bias floor from a
small α deficit; a three-point fit would.

### Knowledge Delta

- **Learned, and this is the load-bearing fact for the next phase.** The shipped
  estimator's final-layer residual is **99.4% variance**, measured at DECIDE tier on
  fresh canonical mints with label noise removed. There is no bias floor worth
  attacking.
- **Belief changed — the driver's, twice, in the same direction it was wrong before.**
  On 2026-08-10 the LOG asserted "the gap is probably mostly bias → attack the
  closure/calibration core". The Skeptic withdrew it on 2026-08-11 using local-mint
  data. This slice confirms the withdrawal at DECIDE tier and closes the question.
- **New instrument, and a trap it removes.** `local_sim/bias_floor_lab.py`. **Any
  measurement that fits a bias floor from MSE-versus-N without subtracting σ_gt² is
  measuring its own label noise.** The two terms are perfectly confounded —
  `E[(pred − truth − η)²] = E[(pred − truth)²] + σ_gt²` — so GT noise enters exactly
  as a constant added at every N, indistinguishable from B². This trap had already
  produced one wrong number in this repository.
- **New invariant candidate.** Every future MSE-versus-N or bias-floor claim must
  quote σ_gt² alongside it, the way `docs/VALIDATION.md` §3.5 already requires for
  plain MSE.
- **Strategic consequence.** J2W's 4.8× lower MSE at our own compute (C/B 0.194 vs
  0.198) is **4.8× lower variance per sample**. The next Strategist Packet targets
  **variance mechanisms**, and it must carry the full kill list — including the
  control-variate lane, whose omission caused the 2026-08-10 round trip.
- **What is now bounded out of the variance search.** Layer-1-anchored *linear*
  control variates (1.010× on our Sobol path, CI includes 1.0). Still open and
  explicitly NOT covered: corrections using an *empirical* per-layer gain
  (`mean(z>0)`), which are nonlinear.
- **Docs changed.** `docs/CURRENT_STATE.md` — open question #1 now states the
  variance verdict as measured at DECIDE tier rather than inferred from local mints.

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

## 2026-08-19 — CARRIER AUDIT: the input-space variance lane is closed, and Strassen is not

**Class: IMPORTANT** (four new probes' math; no seed block spent as a *decision*
block; `estimator.py` untouched; no gate decided). **These are DESIGN-TIER
ORIENTATION PROBES, not pre-registered evidence.** They were run to validate a
proposed pivot, not to promote a candidate. Nothing here may be cited as a gate
result. Seeds: the unused tail of D16 (30806016200+, 30806016300+), a design block.

**Why this was run.** A strategy review on 2026-08-17/18 proposed pivoting the
project's whole next phase onto *input-space variance reduction* — full 256x256
whitening of the point set, the tent/baker fold, ReLU-homogeneity sphere
projection, and a structured design carrier — on the strength of a competitor's
published ladder in which whitening reads **2.46x**. Before spending 31 days of
Phase 2 on that, the pivot was measured.

### Probe 1 — the point set's own moments (no MLPs, no truth, no block)

`local_sim/probe_pointset_moments.py`. Our shipped construction (Sobol integers
-> 30-bit digital XOR shift -> (u+0.5)/2^30 -> `norm.ppf` -> float32), N=16,384,
d=256, 8 realisations, against iid:

| | rms error | vs iid |
|---|---|---|
| first moments | 1.613e-4 | 48.6x better |
| second moments, **diagonal** | 1.041e-4 | **107x better** |
| second moments, **off-diagonal** | **1.315e-3** | **only 5.9x better** |

The off-diagonal block is **12.6x worse than the diagonal**. This is a physical
property of the construction and needs no protocol tier.

**It also explains two of our own past results.** The 2026-08-05 "moment
standardisation" kill forced per-coordinate (mean, std) -> (0,1) — the *diagonal*,
already 107x better than iid. The "point-set radial imbalance" predictor of the
common mode (LOO R2 = -0.010) is tr(S)/d — also diagonal. **Both probed the
part of the point set that was already exact.**

### Probe 2 — does fixing the off-diagonal help? `local_sim/probe_whitening.py`

24 fresh canonical mints x 24 digital shifts, paired CRN, N=16,384, GT 2^23
samples/MLP, **sigma_gt^2 = 5.4393e-9 = 0.38% of MSE**. Artifact:
`experiments/2026-08-19_carrier_probes/white_probe.json`.

| arm | mean MSE | paired geo-mean vs plain | 95% CI |
|---|---|---|---|
| plain (our Sobol) | 1.4486e-6 | 1.0000x | — |
| `whitediag` (the 2026-08-05 kill) | 1.4487e-6 | **1.0000x** | [0.999, 1.001] |
| `sphere` (ReLU homogeneity) | 1.4497e-6 | 1.0010x | [0.998, 1.004] |
| **`white` (full 256x256 ZCA)** | 1.4504e-6 | **0.9972x** | **[0.986, 1.009]** |
| `sphere+white` | 2.4039e-6 | 0.6824x | [0.632, 0.733] |
| `tent` (baker fold) | 1.5164e-6 | 0.9810x | [0.928, 1.037] |
| **`tent+white`** | 1.3307e-6 | **1.0931x** | **[1.044, 1.148]** |
| `iid` | 2.9160e-6 | 0.6085x | [0.566, 0.657] |
| **`iid+white`** | 2.2087e-6 | 0.7478x | [0.695, 0.804] |

**Whitening on our carrier is worth nothing** (0.9972x, CI includes 1.0).
**Whitening on iid is worth 1.23x** (0.7478/0.6085). The device works; our Sobol
already collects that variance. Only `tent+white` clears 1.0, at **+9.3%**.

### Probe 3 — a structured design carrier. `local_sim/probe_design_carrier.py`

Cheap unbiased proxy for the honest frontier's spherical 5-design: a bank of
Haar-rotated orthonormal frames (+/- Q e_i, scaled to radius E||x|| = 15.984383),
exact for every polynomial of degree <= 3 on the sphere. Same 24 MLPs.

| carrier | vs our Sobol | 95% CI |
|---|---|---|
| sobol + antithetic | 1.0158x | [0.959, 1.078] |
| sobol + antithetic + sphere | 1.0153x | [0.958, 1.077] |
| **Haar frames, on the sphere** | **1.1012x** | **[1.040, 1.173]** |
| Haar frames, chi radius | 1.0604x | [1.001, 1.131] |

Homogeneity, worth nothing on Sobol, is worth **+4%** *inside* a design carrier.

### Probe 4 — does the design survive our stack? `local_sim/probe_carrier_transfer.py`

The transfer risk: the rank-32 J J^T rotation is tuned to put Sobol's
best-equidistributed coordinates on the network's most sensitive directions.
20 MLPs x 16 realisations, two sample counts, frames / sobol:

| stack | N = 16,384 | N = 65,536 |
|---|---|---|
| bare | **1.1817x [1.092, 1.290]** | **1.1060x [1.021, 1.199]** |
| + rank-32 rotation | 1.0529x [0.984, 1.126] | 1.0401x [0.950, 1.135] |
| + rotation + pruning (**shipped**) | **1.0470x [0.979, 1.120]** | **1.0341x [0.951, 1.122]** |

**Our rotation already collects what a degree-3 design provides.**

### Probe 5 — Strassen-Winograd, measured on the real flopscope 0.10.0

`local_sim/probe_strassen_flops.py`, `probe_cost_stack.py`, `probe_strassen_ops.py`.
Not modelled — measured against the installed grader cost model. Dense
(16384,256)@(256,256) bills **2,143,289,344**, exactly m*n*(2k-1).

| depth | FLOPs/layer | vs dense |
|---|---|---|
| 0 | 2,143,289,344 | 1.000x |
| 1 | 1,893,810,176 | 1.132x |
| 2 | 1,692,102,656 | **1.267x** |
| 3 | 1,544,635,392 | 1.388x |
| 4 | 1,466,398,464 | 1.462x |

Strassen is exact, so **raw MSE is unchanged**. Interaction with pruning: padding
the pruned 205-wide matrices *up* to 256 to enable splitting is a **loss**
(0.812x vs shipped) — pruning already shrinks both dimensions quadratically. At a
Strassen-friendly alive width the two compose: pruned-192 + depth-2 reads
**1.430x vs shipped**, i.e. Strassen contributes 1.254x on top of pruning.

**Then the residual, which is what killed this lane before.** Our shipped file
runs 1,416 ops at 9.845 ms graded residual -> **7.0 us/op = 695,268
FLOP-equivalents per operation**. Naive recursive implementation, width 192:

| depth | counted saving | extra ops | break-even us/op | net at our 7.0 us/op |
|---|---|---|---|---|
| 1 | 4.377e9 | 864 | 50.7 | **1.108x** |
| 2 | 7.810e9 | 6,912 | 11.3 | 1.083x |
| 3 | 10.118e9 | 49,248 | 2.1 | **0.614x (loss)** |

An interior optimum at shallow depth, independently reproducing forum 18184's
depth-2 finding. Hand-unrolling (weight-side combinations precomputed once per
MLP, outputs kept as blocks instead of concatenated) should reach ~1.12-1.16x.

**Our 2026-08-06 Strassen negative result is superseded, not contradicted.** It
measured *batched depth-2 Strassen on the packed v2.5 carrier* — tiny irregular
bucketed blocks where the add overhead dominates and counted FLOPs actually rose
1.07%. On the dense chain the arithmetic is entirely different.

### Knowledge Delta

- **The input-space variance lane is CLOSED for this estimator, by our own
  measurement in our own configuration.** Whitening 0.997x, sphere 1.001x, tent
  0.981x, antithetic 1.016x, degree-3 design 1.047x — every CI includes or sits
  at 1.0. Only `tent+white` (1.093x) clears, and it is small.
- **The governing mechanism is SUBSTITUTION, measured three separate ways today**
  and corroborated by four independent external teams: whitening vs Sobol
  (0.997x on ours, 1.23x on iid); design vs rotation (1.18x bare -> 1.047x
  shipped); and previously CV vs QMC (our own 2.04->1.98, and forum 18179's
  1.42x->1.04x). **Two devices that annihilate the same low-order structure do not
  compose; whichever is added first takes the credit.**
- **The 2.46x that motivated the proposed pivot is a rung ratio against a
  degraded baseline.** Its own author's four-arm ablation reads 0.97x for
  "lattice on whitening" and 0.994x for "whitening on lattice"; the technique
  census records whitening at **1.05x**; and the same document's harmonic budget
  caps it at **1.22x**. Any ladder table must be read for what its baseline arm
  contains, not for its largest rung.
- **A cost lane we had closed is open.** Strassen-Winograd is worth 1.25-1.27x on
  counted FLOPs, MSE-exact, and ~1.10-1.16x net after residual at our measured
  7.0 us/op. This is currently the largest bankable lever in the project.
- **New standing number: 7.0 us/op = 695,268 FLOP-equivalents per operation** on
  the grader, calibrated from 327743 (1,416 ops, 9.845 ms). Any candidate that
  trades counted FLOPs for operation count can now be priced before it is built.
- **A durable doc states a mechanism that is wrong.** `docs/writeup-draft.md` and
  `docs/RESEARCH_NOTES.md` attribute the 2.35x moment-standardisation
  regression to "a first-order bias". Moment matching's bias is **O(1/N)**
  (Glasserman 4.5.1 p.245), i.e. second order, and both this probe and forum
  18151 measure diagonal forcing at **exactly 1.000x**. The 2.35x may be real;
  the stated mechanism is not, and the write-up must not ship it.
- **Contract correction.** `AGENTS.md` says the prize is decided by a re-run of
  "ONE designated submission". Forum 18143 (organiser, 2026-08-10) says **up to
  TWO per team, defaulting to the two highest-ranked public entries**. The Rules
  tab still says one; the discrepancy was raised on the forum and not resolved.

### External bounds gathered alongside (not our measurements — cite the source)

- Mean dimension of the layer-32 integrand, Gaussian-Poincare bound: **nu ~ 20**;
  share of variance in spherical-harmonic degrees <= 5 falls **0.99 (layer 1) ->
  ~0.36-0.39 (layer 32)**. A degree-5 design's ceiling at the scored layer is
  therefore **~1.46x**, and at N=2^16 in 256 dimensions a point set resolves at
  most ~16 coordinates jointly against interaction order ~20.
- Bakhvalov/Novak information-based bound for the Lipschitz class at d=256: the
  best possible *sampling* method beats plain MC by **~4.4% RMSE / ~9% MSE**.
  This is a query-model minimax bound and does **not** bind an algorithm that
  reads the weights — which is precisely ARC's mechanistic thesis.
- ARC's own cumulant propagation (arXiv:2605.05179) breaks even only when
  L^K < n, i.e. L < 16 for K=2 and L < 6.3 for K=3. At L=32 every variant is
  either over budget or worse than MC. Their own experiments stop at L=12.

### What this does NOT close

1. **A true degree-4/5 design.** The proxy above is degree-3, which the rotation
   absorbs. A hypercube strength-4 orthogonal array (BCH dual, 2^16 runs) or
   Kuperberg's 2^16+512-point degree-5 cubature is a different object. Ceiling
   from the harmonic spectrum: **~1.46x**.
2. **The mid-layer control variate.** Not measured here. Its deployment condition
   is |delta_c| < SE_c and the predicted gain is G = 1/[(1-rho^2) + rho^2 r^2],
   r = delta_c/SE_c — one cheap probe decides it.
3. **Phase 2's shape**, which is unpublished (organiser, forum 18186, 2026-08-18:
   "hope to launch mostly on 19th ... the deadline will also be shifted").

**Driver** Claude Code. **Next command:** `review-slice`.

## 2026-08-19 (later) — STAGE 0: the mid-layer CV dies by identity; the Strassen carrier clears

**Class: IMPORTANT.** Two discriminating probes, both pre-registered in the
plan above before they were run. DESIGN TIER — no gate promoted, no decision
block spent, `estimator.py` untouched, vault untouched. Seeds: D16 tail
(30806016400+, 30806016500+).

### P0.1 — mid-layer control variate: **CLOSED BY IDENTITY**

`local_sim/probe_midlayer_cv.py`, artifact
`experiments/2026-08-19_carrier_probes/midlayer_cv.json`.
20 fresh mints x 16 shifts, N=16,384, GT 2^23, **shipped rank-32 rotation on**.

The lane reduces to one number. With `delta_c` the closure's error on the layer-c
mean, `SE_c` OUR sampling error on that same mean, and `r_c = delta_c / SE_c`:

    G_c = 1 / [ (1 - rho^2) + rho^2 * r_c^2 ]      deploys only if r_c < 1

Rather than fit anything, the probe DEPLOYS the correction that would ship —
`mu_hat = mean(a_D) - (mean(a_c) - m_c) @ M_{c->D}` with `M` the closure's own
linearised transport `prod W_l diag(Phi_l)`, which `_closure` already computes —
and measures the paired MSE ratio.

| layer c | delta_c | SE_c | **r_c** | deployed G_c | 95% CI | wins |
|---|---|---|---|---|---|---|
| 1 | 1.652e-3 | 2.045e-3 | **0.81** | **0.8354x** | [0.762, 0.911] | 108/320 |
| 2 | 2.833e-3 | 2.279e-3 | 1.24 | 0.4998x | [0.414, 0.601] | 62/320 |
| 4 | 4.457e-3 | 2.332e-3 | 1.91 | 0.1519x | [0.114, 0.199] | 9/320 |
| **6** | 5.506e-3 | 2.212e-3 | **2.49** | **0.0863x** | [0.065, 0.116] | 2/320 |
| 8 | 6.187e-3 | 2.036e-3 | 3.04 | 0.0586x | [0.043, 0.079] | 1/320 |
| 12 | 7.036e-3 | 1.774e-3 | 3.97 | 0.0443x | [0.031, 0.066] | 1/320 |
| 16 | 7.820e-3 | 1.589e-3 | 4.92 | 0.0334x | [0.023, 0.049] | 0/320 |
| 24 | 8.189e-3 | 1.315e-3 | 6.23 | 0.0226x | [0.017, 0.029] | 0/320 |

**G_c < 1 at every depth**, and 0.0863x at layer 6 — the exact operating point
where forum 18154 reports **2.39x**. `r_c` crosses 1.0 between layers 1 and 2 and
rises monotonically to 6.23.

**The mechanism, and it explains the competitor rather than contradicting them.**
A control variate helps only when its analytic centre is more accurate than the
sampling estimate it corrects. Ours is **1.2x to 6.2x LESS accurate**. Forum
18154 clears the same bar from both sides at once: a weaker carrier (LHS, raw
1.4x, so larger `SE_c`) and a much better closure (skew-augmented Gram-Charlier
with a Mehler co-skewness term plus an offline-trained per-layer mean correction,
so smaller `delta_c`). Their deployed entry grades **2.9045e-7**, worse than our
2.6314e-7 — the 2.39x is a within-pipeline factor over their own plain MC, not a
frontier.

**The reopening condition is now quantified, which is the useful part.** This
lane reopens iff a closure delivers a mid-layer mean **3-6x more accurate than
our present one** (specifically `r_c < 1` at some c where rho^2 is large). That
is a concrete target for the Moonshot / cumulant lane, and the first time it has
had one.

### P0.2 — Strassen carrier: **CLEARS, with an 8x margin on the deciding risk**

`local_sim/probe_strassen_carrier.py`. Real flopscope 0.10.0, 6 canonical mints.

Design as chosen: tau = -2.5 kept; the least-alive neurons trimmed until the
width divides by 4 (they move to `dead`, where the existing offset machinery
carries their closed-form mean exactly); **block-resident** — the four output
quadrants of layer l ARE the four input quadrants of layer l+1, so the 32-layer
chain never concatenates; the seven weight-side combinations formed once per MLP.

| quantity | value |
|---|---|
| mean alive width, tau = -2.5 | 205.88 |
| mean alive width, trimmed to a multiple of 4 | 204.46 |
| counted FLOPs, dense (trimmed pruning) | 44,764,200,960 |
| counted FLOPs, Strassen depth 1 | 39,451,077,607 |
| ops, dense chain / Strassen chain | 94 / 1,051 |
| **counted-FLOP ratio** | **1.1500x** |
| **effective ratio at 7.0 us/op** | **1.1155x** |
| **break-even residual price** | **55.5 us/op** |
| max relative prediction deviation | **2.186e-7** |

**Two things make this the most certain lever the project has had.**

1. **The deciding risk has an 8x margin.** Strassen trades counted FLOPs for
   operation count, and operations are billed as residual. Break-even is 55.5
   us/op; we measure 7.0. `docs/GOTCHAS.md` records that the candidate/control
   residual ratio reads 5.47x in one local harness and 7.87x in another — a 44%
   spread on the deciding quantity. **An 8x margin swallows that entirely**, so
   for the first time a cost decision here does not depend on the E1 half of the
   evaluator being trustworthy.
2. **The accuracy question is arithmetic, not statistical.** Strassen is exact in
   exact arithmetic; in float32 it reorders summation and moves the prediction by
   **2.19e-7 relative**, against a relative RMS error of ~0.23% at the scored
   layer. That is ~10,000x below our own error. No MSE experiment is needed and
   no decision block is required to establish it.

**Projected score.** Forward chain 4.537e10 -> 3.945e10 counted; `F0` unchanged
at ~0.80e10; ops 1,416 -> ~2,373 so residual 9.845 -> ~16.6 ms. C: 5.43e10 ->
~4.91e10, a **1.107x** effective reduction. Expected **2.6314e-7 -> ~2.38e-7**,
the largest single gain since v2.4's 1.222x.

**Depth 2 is not worth it.** Counted improves to 1.267x, but block-resident depth
2 costs ~7x the operations (~7,400), and the two effects cancel almost exactly at
our residual price. Depth 1 is simpler, more numerically stable, and equal.

### Knowledge Delta

- **The mid-layer control-variate lane is closed for this estimator, by identity,
  at every depth 1..24.** The census's "most actionable open direction, priced at
  3-5x on score" is worth **< 1x** for us. The deciding quantity is
  `r_c = delta_c / SE_c` and it is 0.81 at c=1 rising to 6.23 at c=24.
- **A control variate cannot be evaluated without measuring its centre's accuracy
  against your own sampling error at your own N.** Every external number for this
  family is carrier-relative and does not transfer. A better carrier makes a
  biased control HARDER to deploy, twice over: it shrinks the variance available
  to remove and it shrinks the bias budget.
- **Strassen-Winograd on the dense chain is worth 1.1155x effective**, MSE-neutral
  to 2.2e-7 relative, with a 55.5 us/op break-even against our measured 7.0.
- **New standing trap, and it invalidates any probe that hits it.** Python places
  a script's own directory at `sys.path[0]`, and `local_sim/` contains a numpy
  SHIM named `flopscope`. **Any probe living in `local_sim/` that imports
  flopscope gets the shim, not the grader's cost model** — and the shim's
  `BudgetContext` exposes none of the real attributes, so it fails loudly if you
  read `flops_used` and silently if you do not. Probes needing the real cost
  model must evict their own directory from `sys.path` before importing (see the
  header of `probe_strassen_carrier.py`), exactly as `mint_provenance.py`
  already does for `whestbench`.

### Next

Stage 1: promote the Strassen carrier to **v2.6** under a pre-registered gate —
predictions within 1e-6 relative of v2.4.1 on >=100 mints, 30/30 invariants,
paired `V.c` >= 1.08x with CI excluding 1.0 on the three-model residual rule.
Decision block **D17**. One vault touch at promotion. **Hold for the Phase 2
configuration**, which is unpublished (organiser 18186, 2026-08-18) and which
re-fits `_PRUNE_TAU` and `_BASIS_RANK` if the shape changes.

**Driver** Claude Code. **Next command:** `review-slice`.

## 2026-08-19 (evening) — STAGE 1 BUILD: v2.6, the Strassen carrier. BUILT, GREEN, NOT PROMOTED

**Class: STRATEGIC** (`estimator.py` changed). **This entry records a BUILD, not a
promotion.** No decision block spent, no vault touch, no submission. The gate is
deliberately deferred until the Phase 2 configuration is published, because two
shipped constants (`_PRUNE_TAU`, `_BASIS_RANK`) are fitted to 256x32 and D17 is
single-use.

### What changed

`estimator.py` 691 -> 784 lines; `estimator_v2_6.py` is the byte-identical
archive; `scripts/preflight.ps1` `$Archive` promoted to it. Three edits:

1. **`_STRASSEN = True`** and a new `_strassen_chain`. One Strassen-Winograd
   level per layer, replacing eight (m/2,k/2)@(k/2,n/2) products with seven.
   flopscope bills a matmul as `(2K-1)*M*N` from shapes alone, so a fast matrix
   multiplication reduces the counted bill exactly.
2. **The chain is BLOCK-RESIDENT.** Layer l's four output quadrants ARE layer
   l+1's four input quadrants, so the 32-layer chain never concatenates once
   (concatenate bills 1/element and costs ~2.7x that again in residual). Only
   the final column mean is reassembled. The seven right-hand sides depend only
   on the weights and are formed once per MLP per layer.
3. **Every prunable width is made even** by raising the threshold just past the
   weakest survivor, which sheds exactly the neuron closest to dead. It moves
   into `dead`, where the existing offset carries its closed-form rectified-
   Gaussian mean exactly. **Depth 1 needs only EVEN widths, not multiples of 4** —
   the trim is <=1 neuron per layer, not <=3. Any layer that cannot be made even
   (a deformed or tiny network) falls back to the dense chain, i.e. v2.4.1.

### Gate status

`scripts/preflight.ps1` **GREEN 5/5** — mint provenance, invariants **30/30**
(including 3x bit-identity determinism and a fresh-instance rebuild), `whest
validate`, ship-file identity, vault seal (touches still 1).

### Accuracy — measured, and the trim is the whole cost

`trim_mse` probe, 24 mints x 16 shifts, GT 2^23, rotation on, paired CRN:

| arm | mean MSE | vs shipped | 95% CI |
|---|---|---|---|
| A shipped pruning, dense | 1.5474e-6 | 1.0000x | — |
| B trimmed-to-even, dense | 1.5513e-6 | **0.9949x** | [0.9912, 0.9981] |
| C trimmed + Strassen | 1.5514e-6 | **0.9949x** | [0.9914, 0.9984] |

**B and C agree to four digits: Strassen's float32 reordering contributes
nothing.** The entire accuracy cost is the one trimmed neuron per odd layer, and
it is **0.51%**. Confirmed on the public split: final-layer MSE paired geo-mean
**0.9942x [0.9870, 1.0013]**, 41/100 wins.

### Cost — and the honest disagreement between two harnesses

CONFIRM tier, read-only, 100-MLP mini, subprocess runner, **both arms run within
the same hour on the same machine**. Artifacts:
`experiments/2026-08-19_carrier_probes/{v2_6_mini_confirm,v2_4_1_mini_control}.json`.

| paired, n=100 | v2.4.1 | v2.6 | paired geo-mean | 95% CI |
|---|---|---|---|---|
| adjusted score | 3.10757e-7 | 3.26105e-7 | **0.9521x** | [0.9453, 0.9590] |
| final-layer MSE | 1.48953e-6 | 1.49264e-6 | 0.9942x | [0.9870, 1.0013] |
| effective compute | 5.75999e10 | 6.01423e10 | **0.9577x** | [0.9548, 0.9605] |
| failed MLPs | 0 | 0 | | |

**On this machine v2.6 LOSES 5%, 0/100 wins on compute — while its counted FLOPs
are 10.5% LOWER.** The whole sign flip is the residual price:

| | counted/MLP | residual/MLP | ops | us/op |
|---|---|---|---|---|
| v2.4.1, local | 5.3301e10 | 42.99 ms | 1,417 | 30.35 |
| v2.6, local | 4.7697e10 | 124.46 ms | 2,434 | 51.12 |
| v2.4.1, **graded 327743** | 5.2898e10 | **9.845 ms** | **1,416** | **7.0** |

**The op count transfers exactly — 1,417 measured locally against 1,416 graded.
Only the price per operation differs, and it differs by the documented amount:**
local residual is **4.37x** the grader's, local us/op is **4.34x** the grader's.
Those two agree to 1%, which is what licenses the transfer.

Break-even is **55.5 us/op**. The grader sits at 7.0; this machine at 30-51.

Three models, all anchored on the graded 9.845 ms:

| model | cost ratio | net V.c after the 0.9942x MSE cost |
|---|---|---|
| A — grader per-op price 7.0 us/op | 1.0990x | **1.0926x** |
| B — transfer the measured local residual ratio (2.89x) | 1.0740x | **1.0677x** |
| local machine as measured | 0.9577x | 0.9522x |

**Model B is the one to quote.** It already absorbs the fact that v2.6's per-op
price rises (30.3 -> 51.1 us/op locally, a 1.68x increase): even if the same
relative rise happens on the grader, 7.0 x 1.68 = 11.8 us/op is still 4.7x under
break-even. Projected: **2.6314e-7 -> ~2.41-2.46e-7**.

### Knowledge Delta

- **v2.6 is built, green, and NOT promoted.** The deciding quantity is the
  grader's residual price, which `docs/CURRENT_STATE.md` classifies **E1 —
  directional only**. A local run cannot settle it: the same byte-identical
  comparison reads **0.952x here and 1.068-1.093x projected on the grader**, and
  the sign flips. **Only a graded run can decide this, and that requires Phase 2
  to be open and Samuel's explicit approval.**
- **The op count is the transferable quantity, not the residual.** 1,417 local
  vs 1,416 graded on the same file. Any future cost candidate should be priced
  as `counted + ops x (grader us/op)`, with the grader's us/op re-derived from
  the most recent graded report — never from a local absolute.
- **An independent corroboration of the Skeptic's 2026-08-10 correction.** The
  local subprocess residual measures **4.37x** the grader's today, against the
  "~7x" that `AGENTS.md` still carries from the `--runner local` path. That is
  now two independent readings in the 4.3x band. Per the Skeptic's own
  instruction the text stays unedited until three same-day repeats exist; this
  is the second.
- **A structural reason to like this candidate anyway:** it is the first change
  in the project whose accuracy claim is *arithmetic* rather than statistical.
  Strassen is exact; the only MSE cost is the 0.51% trim, and it is measured
  with a CI. There is no scenario in which v2.6's MSE surprises us.

### Next

1. **Hold.** Phase 2 opens 2026-08-20 00:00 UTC (round 1429; slipped from the
   18th, then the 19th). Width, depth, budget, lambda and floor are unpublished;
   no `v1-phase2` dataset tag exists.
2. On the announcement: re-fit `_PRUNE_TAU` and `_BASIS_RANK` if the shape moved,
   re-measure the cost balance, then gate on **D17** with the three-model rule and
   one vault touch.
3. **The first Phase 2 submission is the instrument that decides v2.6**, because
   it returns the grader's own residual telemetry for the new carrier. Until then
   the honest statement is "1.07-1.09x projected, 0.95x measured locally, and the
   difference is a quantity we cannot measure on this machine."

**Driver** Claude Code. **Next command:** `review-slice` (a Skeptic has not seen
this build, and the local/graded disagreement is exactly what one is for).

---

## 2026-08-19 (night) — STAGE 0 / P0.1 CONTRACT: the canonical arithmetic competition scorer  [PRE-REGISTERED, no data yet]

**Classification: IMPORTANT** (it changes *the instrument* — how a promotion number
is computed — under `AGENTS.md` § *Work classification*). It is deliberately **not**
STRATEGIC, because this slice **adds** a primitive and **changes no gate**:
`docs/VALIDATION.md` §2b's geometric-mean rule stays binding and unedited until a
Skeptic has seen this and Samuel has approved the protocol amendment. Nothing here
decides anything.

**Driver** Claude Code. **Source** `docs/CLAUDE_CODE_IMPLEMENTATION_HANDOFF.md`
→ `docs/FRONTIER_ARCHITECTURE_PROPOSAL.md` § *Stage 0 validation blockers*, P0.1.

### Scope, and what is explicitly out of it

**In.** One module, `local_sim/competition_score.py`, that reproduces the installed
`whestbench` 0.14.0 scoring identity exactly, plus its test suite
`local_sim/test_competition_score.py`.

**Out, and untouched by this slice:** `estimator.py`; any seed block (D16, D17 or
otherwise — this slice consumes **zero** networks and needs none, its inputs are
synthetic rows and already-durable graded reports); the vault (no evaluate, no
`--dry-run`, no read of truth); any submission; `docs/HUMAN_MODEL.md`; the twelve
probes that carry an inline `max(0.1, C/B)` — they are **not** rewritten here; and
`docs/VALIDATION.md` §2b, whose geometric-mean statistic remains the binding gate.

### The identity being reproduced (read out of the installed package, not assumed)

`.venv/Lib/site-packages/whestbench/budget.py` and `scoring.py`, whestbench 0.14.0:

```text
C_i        = F_i + lambda * R_i                      # budget.effective_compute
mult_i     = 1.0                     if failed_i     # budget.score_multiplier
           = 1.0                     if B_i <= 0
           = max(0.1, C_i / B_i)     otherwise       # floored, UNCAPPED above
s_i        = mse_i * mult_i                          # scoring._compute_budget_adjusted_score
objective  = arithmetic_mean(s_i)                    # scoring.evaluate_estimator
failed_i   = bool(error_code) or budget_exhausted or time_exhausted
             or residual_wall_time_exhausted or combined_budget_exhausted
```

with `lambda = 1e11` and the floor `0.1` **hardcoded** in 0.14.0. On the exception
path the grader also records `C_i = 0.0`, `F_i = 0`, and `mse_i` = the
zero-prediction MSE; on the exhaustion paths it keeps the real `F_i`/`R_i` but still
forces `mult_i = 1.0`. Both are worse than the floor, not better.

### Written prediction — falsifiable, recorded before the code exists

| # | prediction | how it is falsified |
|---|---|---|
| **P1** | The reimplementation matches `whestbench.budget.score_multiplier` and `effective_compute` **exactly** (`==`, not `isclose`) on >=10,000 randomised rows spanning failures, `C/B` on both sides of the floor, `B<=0`, zero residual, and huge residual | any single row differs |
| **P2** | One failed MLP in 100 at our operating point (`mult ~ 0.199`) costs **more than 5x** that MLP's normal contribution to the objective, because the failure multiplier is `1.0` and not the floor `0.1` | measured ratio <= 5 |
| **P3** | Arithmetic and geometric aggregation **select opposite winners** on a fixture built from the real 50 graded public rows of 327743. Pre-registered mechanism: a candidate that improves every row by a uniform `g` and regresses only the single worst row wins geometrically and loses arithmetically. Pre-registered flip threshold `X* = g * n * mean / worst`; at `g = 3%`, `n = 50` I predict **`X*` lands in [25%, 60%]** | no flip exists at any `X`, or `X*` falls outside [25%, 60%] |
| **P4** | Read-only replay: for all 50 public rows of **both** 327743 (v2.4.1) and 324420 (v2.4), `mse_i * max(0.1, C_i/B)` reproduces the report's own `adjusted_final_layer_score` to relative **1e-12**, with `B = 2.72e11` | any row exceeds 1e-12 |
| **P5** | The grader's own `effective_compute` equals `flops_used + 1e11 * residual_wall_time_s` to relative **1e-9** on all 100 rows of both reports — an independent confirmation of `lambda = 1e11` from graded telemetry rather than from the package default | any row exceeds 1e-9 |
| **P6** | `mean_i(mse_i) * mean_i(mult_i)` differs from `mean_i(mse_i * mult_i)` by **more than the 1e-12 identity tolerance** on the real graded rows — i.e. the MSE/cost correlation is nonzero and the factorised form used inline by several probes is biased | the two agree to 1e-12 |
| **P7, open** | Whether the report's `aggregates.public` block is the mean over the **50** public rows or over all **100** is not documented anywhere in our record. I predict it is over **100** (the field is a sibling of `n_failed_mlps: 0`, which counts 100). Discriminated by `mean_effective_compute = 5.4118e10` against the telemetry of the first 50 vs all 100 | the 50-row mean matches instead |

### Verification plan

1. `uv run python local_sim/test_competition_score.py` — the suite above, which
   imports the installed `whestbench` and compares against it directly.
2. `.\scripts\preflight.ps1` — must stay **GREEN 5/5**; this slice touches no
   shipped file, so a red preflight means collateral damage.
3. `uv run python scripts/verify_portability.py` — unchanged count; no harness
   files are edited.
4. Independent **Skeptic** over the diff, the raw artifacts and this contract.

### What would make this slice a failure

Any of P1, P4, P5 failing means the reimplementation is not the grader's identity
and must not be used. P3 failing means the arithmetic/geometric divergence that
motivates P0.1 is not demonstrable on our own data, and the proposed protocol
amendment should be withdrawn rather than argued.

### Phase-2 configuration status at contract time

Round 1429 is **UPCOMING, not open** (`phase2_checklist`, this session):
`2026-08-20T23:59:00Z -> 2026-09-19T23:59:00Z`, leaderboard 1480 exists with 0
entries, HF tags are still `['v1-phase1', 'v1-warmup']`. **width, depth, biases,
input distribution, n_mlps, B, lambda, the multiplier floor and the number of
designatable submissions are all UNKNOWN for round 1429.** The scorer is therefore
built with `B`, `lambda` and the floor as **per-call parameters**, defaulting to the
installed 0.14.0 values, and with `B` allowed to vary per MLP even though 0.14.0
uses one `spec.flop_budget` for all of them. No Phase-1 constant is baked in.


---

## 2026-08-19 (night) — STAGE 0 / P0.1 EVIDENCE PACKET: the canonical scorer is built and green. **Two of my seven predictions were falsified, and both falsifications are the findings.**

Contract: this file, the entry immediately above (line ~3883). **Nothing in this
slice decides anything** — `docs/VALIDATION.md` §2b's geometric-mean gate is still
the binding rule, unedited, pending Skeptic and Samuel.

### Files (both new; no existing file was modified)

| file | what |
|---|---|
| `local_sim/competition_score.py` | the primitive. numpy + stdlib only — imports **nothing** from `whestbench`, verified by import-blocking, and runs unchanged in `.venv-gpu` |
| `local_sim/test_competition_score.py` | the proof, 64 checks |
| `experiments/2026-08-19_p0_1_scorer/test_competition_score.out` | raw output |
| `experiments/2026-08-19_p0_1_scorer/preflight.out` | raw output |

### Commands and outputs

```text
uv run python local_sim/test_competition_score.py     -> 64/64 checks passed, exit 0 (twice)
.\scripts\preflight.ps1                               -> PREFLIGHT GREEN 5/5, exit 0
uv run python scripts/verify_portability.py           -> 174 checks, 0 failed
uv run python local_sim/phase2_checklist.py           -> round 1429 UPCOMING, 0 entries, no v1-phase2 tag
environment: whestbench 0.14.0  flopscope 0.10.0  numpy 2.2.6
```

Vault touch counter unchanged at **1**. No seed block allocated or consumed —
this slice generated no networks from any registered block; its synthetic
contests use ad-hoc `ContestSpec.seed` values (515, 99, 20260819) in the
`whestbench` namespace, which is disjoint from every `seed_blocks.py` range and
carries no labels. D16 and D17 untouched.

### Verdicts on the seven pre-registered predictions

| # | verdict | measured |
|---|---|---|
| **P1** | **CONFIRMED** | `effective_compute`, `score_multiplier`, `is_combined_budget_exhausted` all **0 mismatches under `==`** on 12,000 randomised rows (failures, `B<=0`, `B<0`, zero and large residual, `C/B` on both sides of the floor) |
| **P2** | **FALSIFIED AS STATED** | predicted "> 5x"; measured **4.680x** on row 0. See below — the prediction was arithmetic sloppiness, and correcting it produced a better fact |
| **P3** | **CONFIRMED** | pre-registered `X* = g*n*mean/worst` = **41.8%**, inside the pre-registered [25%, 60%]; the observed first flip is at **39.0%**, within 15% of the formula |
| **P4** | **CONFIRMED, and stronger than predicted** | tolerance was 1e-12; the measured worst relative deviation is **0.000e+00 — bitwise exact** on all 50 rows of *both* submissions |
| **P5** | **CONFIRMED, bitwise** | `C == F + 1e11*R` to **0.000e+00** on all 100 rows of both reports. Lambda = 1e11 is now confirmed from the grader's own telemetry, not from a package default |
| **P6** | **CONFIRMED** | factorised bias **+0.7367%**, `corr(mse, mult) = -0.3391`. The correlation is **negative**, so the factorised form **overstates** here; the synthetic positive-correlation case understates by 27.6% |
| **P7** | **FALSIFIED** | predicted the report aggregate covers 100 MLPs. It covers the **50 public rows**, exactly |

### The two falsifications, which are the load-bearing results

**P2 — the failure penalty is exactly `1/mult_i`, and I predicted it off the wrong
multiplier.** The multiplier half of a failure costs exactly `1 / mult_i`, because
the grader replaces `max(0.1, C_i/B_i)` with `1.0`. I derived "> 5x" from the
*suite-mean* multiplier 0.19896 (-> 5.03x); row 0's own multiplier is 0.2137
(-> 4.680x). Across the real 50-row suite the penalty ranges **[4.59x, 5.41x]**.
The test now asserts the exact law rather than my wrong threshold. The number that
matters is unchanged in spirit and now exact: **a failure costs ~5x on the
multiplier alone, before the much larger cost of the zeroed prediction.**

**P7 — the published score is the arithmetic mean over the FIFTY public MLPs, and
it is now reproducible offline, bitwise.** The report's `aggregates.public` block
is *not* an aggregate over all 100 scored networks. Discriminated on
`mean_effective_compute`:

| submission | reported | 50-row mean | 100-row mean |
|---|---|---|---|
| 324420 (v2.4) | 5.412154e10 | **5.412154e10** | 5.389333e10 |
| 327743 (v2.4.1) | 5.411771e10 | **5.411771e10** | 5.390057e10 |

And therefore, from `mse_i` and `C_i` alone:

| submission | our 50-row objective | published leaderboard score |
|---|---|---|
| 324420 | 2.719427529595e-07 | 2.719427529595e-07 |
| 327743 | 2.631432177242e-07 | 2.631432177242e-07 |

**Bitwise equal.** Every graded number this project has is now re-derivable from
the durable artifact without the grader.

### The estimand disagreement is not hypothetical — it is in our own graded data

Scored by the canonical scorer, the real paired v2.4 -> v2.4.1 comparison on the
same 50 MLPs in the same order:

| statistic | value | says |
|---|---|---|
| arithmetic mean difference | **+8.799535e-09** (`+3.236%`, gain 1.0334x) | **v2.4.1 better** |
| paired bootstrap 95% CI (20,000 resamples, seed 12345) | [-4.854e-08, +6.610e-08] | **not resolvable** |
| geometric gain (the current binding gate) | **0.9666x**, CI [0.7969, 1.1723] | **v2.4 better** |
| candidate better on | 22/50, sign-test p = 0.480 | — |

The two estimands **select opposite winners on real graded data**. `CURRENT_STATE`
already records the geometric half (0.9665x, 22/50) and the aggregate half
(2.7194e-7 -> 2.6314e-7) — what was missing is that these are the *same* 50 rows
and therefore a genuine contradiction, not two different measurements.

The controlled fixture pins the mechanism: a candidate that improves 49 of 50 MLPs
by 3% and regresses the single worst one by 49% reads **geo 1.0221x (a win, CI
excluding 1.0, 49/50 rows better, sign-test p < 1e-4)** and **arithmetic 0.9927x
(a 0.73% loss)**. The geometric mean weights each MLP's log equally; the
competition sums scores, and that one MLP is 7.2% of the objective.

### Knowledge Delta

- **The competition objective is an arithmetic mean over the 50 public MLPs, and
  the exact identity is now in one place.** `local_sim/competition_score.py` is
  bitwise-equal to `whestbench` 0.14.0 on the scalar primitives, on the aggregate,
  and on the real scoring loop including both failure paths.
- **A failure is priced at multiplier 1.0, not at the 0.1 floor** — ~5x the
  normal multiplier at our operating point, *on top of* the zeroed prediction's
  MSE. Confirmed on the real loop for the exception path (`C` recorded as 0) and
  the exhaustion path (`C` recorded truthfully, multiplier still forced to 1.0).
- **Lambda = 1e11 is confirmed from graded telemetry**, bitwise, on 200 rows.
- **`mean(mse) * mean(mult)` is biased and the sign of the bias is the sign of
  the MSE/cost correlation**, which is **-0.34** on our real graded rows.
  Twelve probes in `local_sim/` carry an inline `max(0.1, C/B)` and several use
  the factorised form. **They are not rewritten in this slice** and remain a
  Stage-0 debt.
- **The heavy tail is measured, not assumed.** On 327743's 50 rows the worst MLP
  is **3.59x the mean** and carries **7.2%** of the objective; the top five carry
  **24.0%**. On 324420, 2.97x and 5.9% / 27.1%. This is why the paired bootstrap,
  not the t interval, is the honest interval for an arithmetic difference.
- ~~**Minor correction owed to `docs/CURRENT_STATE.md`**: it quotes "Ratio of
  means 1.0319" ... the correct value ... is **1.0334**.~~ **WITHDRAWN by the
  Skeptic — see finding D1 in the addendum below. The number is in `GOTCHAS.md`,
  not `CURRENT_STATE.md`, and 1.0319 is CORRECT in the raw-MSE frame that
  paragraph is written in. No doc correction is owed.**
- **Toolchain intel, gathered read-only and NOT installed.** whestbench **0.15.0**
  shipped 2026-08-17 and requires flopscope >=0.11.0,<0.12.0. Its `scoring.py`,
  `budget.py`, `domain.py` and `generation.py` are **byte-identical to the
  installed 0.14.0**; only `cli.py`, `runner.py`, `simulation.py` and
  `dataset_torch.py` differ, and the visible changes are flopscope-0.11 compat.
  **So the score identity, the 0.1 floor, lambda = 1e11, and He-init/no-bias MLP
  generation are unchanged in the newest published evaluator.** That is not the
  same as knowing the Phase-2 *contest configuration*, which is a runtime spec.

### What this slice deliberately did NOT do

`estimator.py` untouched. No seed block. No vault read, evaluate, or `--dry-run`.
No submission. `docs/HUMAN_MODEL.md` untouched. `docs/VALIDATION.md` §2b unedited —
**the geometric mean is still the binding gate until a Skeptic and Samuel say
otherwise.** The twelve probes with inline multiplier arithmetic are unchanged.

### Proposal for review (NOT enacted)

Amend `docs/VALIDATION.md` §2b so the **primary** promotion statistic is the paired
arithmetic score difference with its paired-bootstrap interval over MLPs, and the
geometric mean of per-MLP ratios is retained as a **secondary** stability
diagnostic. Rationale: the competition's estimand is the arithmetic mean, the two
disagree on our own graded data, and the private re-run is a small-`R` draw from a
tail-heavy distribution where the arithmetic mean is what is actually sampled.
This is a **gate change and therefore STRATEGIC** — it needs a Slice Contract of
its own, Samuel's prediction, and the Skeptic's verdict first.

### Next

`review-slice` — Skeptic on this build. Then propose, in order and one at a time,
the next Stage-0 slice from the handoff's ordered list (odd-width/fallback fix ->
seed ledger -> vault -> v2.6 gate reconstruction).

### SKEPTIC VERDICT: **PROVISIONAL** — and it found two real errors of mine

Independent falsifier, launched with the contract, both new files, the evaluation
protocol and the raw artifact paths; it re-ran the gate itself (64/64 at the time,
preflight GREEN 5/5, portability 174/174) and recomputed every load-bearing number
from the raw JSON rather than from this packet.

**What it confirmed, having reproduced it independently:** P4 and P5 bitwise on
both submissions; P7 with *four* independent bitwise matches (it also found that
the 50-row objective equals the report's **top-level `score` field**, i.e. the
leaderboard number itself, which this packet had not checked); the estimand
disagreement, and that it is robust to the frame — **both** frames flip; that the
parity test is **not circular** (`_rows_from_inprocess` recomputes `C` from
`flops_used`/`residual_wall_time_s` rather than reading the grader's
`effective_compute`, and compares against a field produced by a different code
path); that the P2 rewrite is a **legitimate** correction and not goalpost-moving,
because the replacement asserts an identity derivable from `budget.py` without
data and the falsification is pinned as a *passing* test; every scope claim
(vault manifest `touches: 1` with 1 log line, both mtime 2026-08-06;
`HUMAN_MODEL.md` dirty but mtime 2026-08-09, ten days before this slice;
`estimator.py`, `local_sim/`, `docs/VALIDATION.md`, `scripts/`, `ai/` diffs all
empty; ContestSpec seeds 515 / 99 / 20260819 disjoint from every registered block
*and* a different protocol from input seeds, so no DECIDE-tier contamination path
exists even in principle); and claim 8, by SHA-256 over every file in the 0.15.0
wheel, slightly strengthened — `simulation.py` and `dataset_torch.py` differ by
**comments only**, and `runner.py` only by a predict-response timeout grace.

**D1 — WITHDRAWN. My "correction owed to CURRENT_STATE.md" was wrong twice.**
The string `1.0319` is not in `CURRENT_STATE.md` at all; it is at
`docs/GOTCHAS.md:30`. And it is **not** a wrong digit — it is the *raw-MSE frame*,
which is the frame that whole paragraph is written in. Recomputed on the 50 rows:

| frame | ratio of means | ratio of medians | geo | geo 95% CI | sd(log) |
|---|---|---|---|---|---|
| raw final-layer MSE | **1.031894** | **1.010707** | 0.966508 | **[0.7969, 1.1721]** | **0.6959** |
| adjusted score | 1.033440 | 1.011137 | 0.966555 | [0.7969, 1.1723] | 0.6962 |

`GOTCHAS.md:29-32` quotes 1.0319, 1.0107, [0.7969, 1.1721] and sd 0.696 — **all
four are the MSE frame to every printed digit.** Enacting my "correction" would
have dropped a score-frame number into an MSE-frame paragraph: precisely the
frame-mixing that `VALIDATION.md` §2b's "counted compares to counted; effective
compares to effective" exists to prevent, committed by the slice whose whole
purpose is to make the frame canonical. **No doc correction is owed.** What is
true and worth recording is only this: the two frames give 1.0319 and 1.0334, and
the disagreement with the geometric statistic holds in both.

**D2 — my pre-registered flip formula was algebraically wrong, and the tolerance
hid it.** Solving `mean(ctrl) - mean(cand) = 0` for the fixture gives
`X_flip = g * (n*mean/worst - 1)` = **38.80%**; my `X* = g*n*mean/worst` = 41.80%
**drops the `-g` term**. The scan grid is 0.5%, so the "observed 39.0%, within 15%
of the formula" was not a near-miss confirmation — it is the exact analytic answer
at grid resolution, and my ±15% tolerance passed a wrong formula. Worse, the
[25%, 60%] band was equivalent to predicting `worst/mean ∈ [2.5, 6.0]` when
`worst_mlp_adjusted_final_layer_score / adjusted_final_layer_score` = **3.5884**
was already in the JSON I had open. **P3 is downgraded from CONFIRMED to an
arithmetic identity: it tested nothing.** The mechanism and the fixture survive
untouched — those were never in doubt and are reproduced by the Skeptic. The test
now records the exact law, states that the contract's formula was wrong, and
checks the flip against the exact value at grid resolution.

**D3, D4, D5 — three defects in the module, all fixed in this slice.**

- **D3.** `_t975`'s comment claimed the df>30 branch is "within 1e-3"; measured
  against scipy it is 3.02e-3 at df=31, 1.80e-3 at df=40, **1.20e-3 at df=49 —
  the case actually used here** — and uniformly *below* the true quantile, so the
  t interval runs systematically narrow. "Peizer-Pratt" was a misattribution
  (that is a binomial/beta approximation; this is Cornish-Fisher). Comment
  corrected to the measured numbers and the direction of the error.
- **D4, the most operationally dangerous.** `is_combined_budget_exhausted` was
  defined and **never called**: nothing derived `failed`, so a hand-built row with
  `C > B` would have been scored at multiplier `C/B` instead of the grader's
  forced 1.0 *and* with an MSE the grader would have discarded. This is exactly
  the trap that made the first draft of the evaluator-parity test score zero clean
  rows. `score_suite` now takes `on_over_budget` and **raises by default**;
  `over_budget_rows()` names the offenders; `"fail"` scores them as failures while
  documenting that it is still optimistic (this module cannot zero a prediction it
  never saw); `"ignore"` keeps the raw identity for deliberate probing. Six new
  checks cover it.
- **D5.** The module told the reader the studentised CI "undercovers on a heavy
  tail; read the bootstrap" — **asserted with no evidence**, and on this data the
  two intervals agree to ~3%. The percentile bootstrap is itself the weakest
  member of its family under skew. The claim is **withdrawn** from both the
  docstring and the printed output and replaced with an explicit statement that
  neither interval's coverage has been measured. This mattered because the packet
  proposed making that interval the *primary promotion statistic*.

**D6, D7, D10 — recorded, not fixed here.**

- **D6.** "Arithmetic mean over the 50 public MLPs" is a **leaderboard** fact.
  The prize is a private re-run of unpublished size, and no report contains a
  private score field of any kind (`local_sim/private_half.py` already established
  this). The extrapolation to the prize is an assumption, and is now labelled one.
- **D7.** whestbench 0.15.0 requires **flopscope >=0.11.0,<0.12.0** and we run
  0.10.0. flopscope is the *pricing* engine — the cost half, this project's
  weakest half. "The score identity is unchanged" says nothing about whether the
  **price table** moved. Unknown, and it belongs in the Phase-2 unknowns list.
- **D10, out of scope but live.** `AGENTS.md` § *Layout* still says `estimator.py`
  is "Currently v2.4.1, byte-identical to `estimator_v2_4_1.py`". It is not:
  preflight and an independent hash both say `estimator.py == estimator_v2_6.py`
  (`f2e70641b71d0dd0` vs v2.4.1's `59c371a5044e0e00`). The shipped file is the
  v2.6 carrier that the LOG says was deliberately **not promoted**. **Flagged for
  Samuel; not edited, because `AGENTS.md` is the operating contract and editing it
  is its own IMPORTANT slice.**

**Also owed, and stated here because the standing rule requires it:** no
ground-truth-noise floor was quoted anywhere in this slice. The public split's
truth is drawn at 1e9 samples, implying a floor of order **5e-11** against the
8.8e-9 arithmetic difference measured above — not binding, but it should have been
said rather than inferred.

### Gate after corrections

```text
uv run python local_sim/test_competition_score.py   -> 71/71 checks passed, exit 0
.\scripts\preflight.ps1                             -> PREFLIGHT GREEN 5/5, exit 0
uv run python scripts/verify_portability.py         -> 174 checks, 0 failed
```

Vault touches still **1**. No seed block consumed.

### The Skeptic's discriminating follow-up, adopted as the next Stage-0 item

**An interval-coverage study for the proposed primary statistic**, before the
STRATEGIC gate-change contract rather than after. Inject a known uniform effect,
subsample n = 50 repeatedly against a large-sample ground truth, and measure the
*actual* coverage of (i) the 20,000-resample percentile bootstrap, (ii) the
studentised t interval, (iii) BCa. It settles D3 and D5 together, and it is the
only thing standing between the current gate and the proposed one. The Skeptic
notes **D16 is free** — `local_sim/seed_blocks.py:72-77` records it as "RELEASED
UNSPENT … NO SEED WAS EVER DRAWN. Reusable." No decision block, no vault touch.
**That claim must be re-verified against `experiments/bias_floor_D16.json` before
D16 is drawn on** — P0.4 of the frontier proposal says the registry and the
committed artifacts disagree about D16, and this slice did not check it.

**Committing note from the Skeptic:** `docs/HUMAN_MODEL.md` is dirty with Samuel's
own edits. Commit with an explicit pathspec, never `git add -A`, and never with a
blind `ARC_WHEST_PROTECTED_OK=1`.

---

## 2026-08-19 (night) — STAGE 0 / P0.4 CONTRACT: the seed ledger. **D16 is spent, and two probes drew the same 24 seeds.**  [PRE-REGISTERED]

**Classification: IMPORTANT** — it changes the instrument (`local_sim/seed_blocks.py`,
`scripts/preflight.ps1`). No gate moves, no estimator changes, no block is spent.

**Why this comes before the interval-coverage study.** The coverage study was the
accepted next slice, with the stated precondition that D16's "reusable" claim be
re-verified first. It is false. `seed_blocks.py:72-77` says of D16: *"NO SEED WAS
EVER DRAWN. Reusable."* Read out of the source that actually drew them:

| range | n | consumer | artifact |
|---|---|---|---|
| 30,806,016,000 – …199 | 200 | `bias_floor_lab.py` (`BLOCK = "D16"`) | `experiments/bias_floor_D16.json` (200 seeds, contiguous from the block start) |
| 30,806,016,200 – …223 | 24 | `probe_whitening.py` (`SEED0`, `N_MLPS=24`) | `2026-08-19_carrier_probes/white_probe.json` |
| **30,806,016,200 – …223** | **24** | **`probe_design_carrier.py` (`SEED0`, `N_MLPS=24`)** | `2026-08-19_carrier_probes/design_probe.json` |
| 30,806,016,300 – …319 | 20 | `probe_carrier_transfer.py` (`SEED0`, `N_MLPS=20`) | — none written |
| 30,806,016,400 – …419 | 20 | `probe_midlayer_cv.py` (`SEED0`, `N_MLPS=20`) | `2026-08-19_carrier_probes/midlayer_cv.json` |
| 30,806,016,500 – …505 | 6 | `probe_strassen_carrier.py` (`SEED0`, `range(6)`) | — none written |

**270 distinct seeds consumed, and rows 2 and 3 are the same 24 networks.** The
whitening kill (0.9972x) and the degree-3 design-carrier kill (1.047x) — recorded
side by side in `docs/CURRENT_STATE.md` § *the carrier audit* under "the governing
mechanism, measured three ways today: SUBSTITUTION" — were measured on **the same
24 networks**. Both are DESIGN tier and no gate was decided on either, so nothing
is invalidated; but they are **correlated evidence, not two independent readings**,
and the audit's framing implies otherwise.

A second, quieter defect: **not one of those five artifacts records which seeds it
consumed.** The consumption is recoverable only by reading `SEED0` out of the
source, which is why the registry and reality diverged for nine days without
anything failing.

### What gets built

1. `experiments/seed_ledger.jsonl` — **append-only**, one record per consumption:
   block, start, n, end, consumer, artifact path, artifact sha256, git commit,
   tier, backfilled flag, note.
2. `local_sim/seed_blocks.py` gains, without removing anything:
   - a **tier** on every block (`VAULT` / `DECISION` / `DESIGN` / `INFRA` / `LEGACY`);
   - `claim(block, n, consumer, ...)` and `claim_range(...)` — the sanctioned way
     to obtain seeds. They **hard-fail** on an unregistered block, on a range that
     leaves the block, on any overlap inside a `DECISION` block, and on a second
     claim of a `SINGLE_USE` block;
   - `consumed_ranges()`, `free_ranges()`, `ledger()`;
   - `check_ledger()` — every claimed range inside its block; no overlap in any
     DECISION/SINGLE_USE block; DESIGN overlaps allowed but **reported**;
   - `scan_sources()` — a static scan of `local_sim/*.py` for integer literals that
     fall inside a registered block but have no ledger record. This is what catches
     the next `SEED0 = 30_806_0xx_xxx` typed straight into a probe.
3. The six historical consumptions above are **backfilled** into the ledger with
   `backfilled: true`, and D16's registry entry is rewritten from "Reusable" to
   what it is.
4. **D17 is LOCKED single-use** — verified never drawn (no source file anywhere
   references `30_806_017`), so locking is correct and allocating a replacement
   would waste 600 seeds.
5. **D18 is allocated** as the DESIGN block for the interval-coverage study.
6. `scripts/preflight.ps1` gains **step 6**: `check_ledger()` + `scan_sources()`.
   The invariant suite stays at 30/30 — a documented count that many docs quote,
   and this belongs in the change gate, not in the estimator's invariants.

### Written prediction

| # | prediction | falsified by |
|---|---|---|
| **A1** | Exactly **one** duplicate-consumption collision exists across every live block: `probe_whitening` vs `probe_design_carrier` on D16 …200–…223. Zero collisions in D01–D15, D17, or V | a second collision, or none |
| **A2** | D17 consumption is **zero**: no file under `local_sim/`, `scripts/` or `experiments/` references any seed in 30,806,017,000–…599 | any reference found |
| **A3** | `scan_sources()` run against the tree as it stands finds **exactly the six** consumptions tabled above and **no seventh** — i.e. no probe is drawing on a block nobody noticed | a seventh appears, or one of the six is missed |
| **A4** | The backfilled ledger passes `check_ledger()` **only** because DESIGN-tier overlap is permitted-and-reported; forcing DECISION strictness onto D16 would fail it | it passes under DECISION strictness too |
| **A5** | `scripts/preflight.ps1` goes **GREEN 6/6**, and `verify_portability.py` stays **174/174** | either regresses |
| **A6** | The distinct-seed count consumed from D16 is **270**, leaving **330** free in four gaps (…224–…299, …320–…399, …420–…499, …506–…599) | any other count |

### Out of scope, deliberately

The five probes are **not** rewritten to call `claim()`. They are finished
experiments whose artifacts are durable; editing them now would change code that
produced recorded numbers for no epistemic gain. The ledger records what they did
and `scan_sources()` keeps them honest. New work uses `claim()`.

No block is spent by this slice. `estimator.py`, the vault, and
`docs/VALIDATION.md` are untouched.

---

## 2026-08-19 (night) — STAGE 0 / P0.4 EVIDENCE PACKET: the seed ledger. **The registry was wrong about more than D16.**

Contract: the entry immediately above. **Prediction A3 was falsified by the tool
built to test it**, which is the result of the slice.

### Files

| file | what |
|---|---|
| `local_sim/seed_ledger.py` | the ledger: `claim()`, `check_ledger()`, `scan_sources()`, tiers |
| `local_sim/test_seed_ledger.py` | **39 negative tests** — every check drives the module into the failure it exists to prevent and demands a refusal |
| `experiments/seed_ledger.jsonl` | the ledger itself, 21 records |
| `experiments/2026-08-19_seed_ledger_backfill.py` | the durable, auditable record of how the ledger was seeded |
| `local_sim/seed_blocks.py` | D16 corrected, D17 locked, D18/D19 allocated, tiers referenced |
| `local_sim/moonshot_g0.py` | one comment: a `seed-ledger: excludes` pragma on the guard that refuses D08/D09/D10 |
| `scripts/preflight.ps1` | **step 6** |

### Commands and outputs

```text
uv run python experiments/2026-08-19_seed_ledger_backfill.py --write  -> 21 records, exit 0
uv run python local_sim/seed_ledger.py --check --scan                 -> CONSISTENT, no unrecorded use, exit 0
uv run python local_sim/test_seed_ledger.py                           -> 39/39, exit 0
uv run python local_sim/test_competition_score.py                     -> 71/71, exit 0 (P0.1 unregressed)
.\scripts\preflight.ps1                                               -> PREFLIGHT GREEN 6/6, exit 0
uv run python scripts/verify_portability.py                           -> 174 checks, 0 failed
```

Vault touches still **1**. **No block was spent by this slice** — the ledger
records history; the negative tests write only to a temporary file, and
`t_real_ledger_readonly` asserts the real ledger's byte size is unchanged by the
test run.

### Verdicts

| # | verdict | measured |
|---|---|---|
| **A1** | **CONFIRMED** | exactly **one** accidental collision in the entire record: `probe_whitening` vs `probe_design_carrier`, D16 …200–…223. Asserted mechanically, not by eye |
| **A2** | **CONFIRMED** | D17 consumption **zero** — by the literal scan, by the new `seeds()`-helper scan, and by the ledger. It is now **LOCKED SINGLE-USE** |
| **A3** | **FALSIFIED** | I predicted "exactly the six consumptions, no seventh". The scan found **eleven more**, across **D01–D07, D11, D12, D13, D14, D15** |
| **A4** | **CONFIRMED** | the ledger passes only because DESIGN-tier overlap is permitted-and-reported; the same two records under DECISION strictness produce a hard error (test 6 constructs exactly that state and checks it fails) |
| **A5** | **CONFIRMED** | preflight **GREEN 6/6**, portability **174/174**, invariants still **30/30** |
| **A6** | **CONFIRMED** | D16: **270** distinct seeds consumed, **330** free, in exactly the four predicted gaps (…224–…299, …320–…399, …420–…499, …506–…599) |

### A3, the falsification — the registry was wrong about eleven more consumptions

The six I tabled from reading five probes were the ones I already suspected. The
static scan found eleven more that **nobody had written down anywhere**:

| block | consumer | recorded |
|---|---|---|
| D01–D05 | `audit_constants.py` (`BLOCKS` dict, `--mlps` default 200) | 200 each |
| D06 | `edge_cases.py` — twelve individual literals | 200 (upper bound) |
| D07 | `theory_probe.py` — `P1_STARTS*mlps + mlps` | 320 |
| D11 | `packed_sparse_probe.py` (two offsets) + `test_packed_sparse.py` (7 literals) + `packed_truth.py` (the truth for the same nets) | 500 + 7 + 200 |
| **D12** | `packed_sparse_probe.py --mode decide4` — **SPENT** | 200 |
| D13 | `packed_sparse_probe.py` + one test literal | 200 + 1 |
| **D14** | `packed_sparse_probe.py --mode decide5` — **SPENT** | 200 |
| D15 | `moonshot_g0.py` | 200 |

Where a count comes from a CLI argument, the **documented run size is recorded as
an UPPER BOUND**. Over-recording consumption is the safe direction: it can only
stop a seed being re-drawn, never license one. Every record cites the source line
that fixes its start.

**D12 and D14 were spent on 2026-08-06 and the registry's prose said so — but
nothing machine-readable did**, and neither could be reached by the literal scan
at all, because both go through `_bs_seeds(args.block)`. Which leads to the second
finding.

### The scan had a blind spot, and finding it is most of the value

The first version scanned for integer literals. It found 28 and missed every
consumer that goes through `seed_blocks.seeds(block)` — `packed_sparse_probe`
(D12, D13, D14), `packed_truth` (D11), `moonshot_g0` (D15). **A literal scan
cannot see a block consumed by name.** The scan now has a second pass: any file
that imports the `seeds` helper and names a block must have a ledger record for
it. That pass is what surfaced D12, D14 and D15.

A file may also name a block precisely in order to **refuse** it —
`moonshot_g0.py` names D08/D09/D10 in the guard that keeps G1/G2's data unseen.
Naming a block to protect it is the opposite of consuming it, so a file declares
that with a `# seed-ledger: excludes D08, D09, D10` pragma, and the scan honours
it. The pragma is auditable in the diff and cannot silently hide consumption,
because the ledger's own consumption counters are asserted separately.

### The epistemic finding, which outlives the tooling

**The whitening kill and the degree-3 design-carrier kill were measured on the
same 24 networks.** `docs/CURRENT_STATE.md` § *Added 2026-08-19 — the carrier
audit* presents them under **"The governing mechanism, measured three ways today:
SUBSTITUTION"**, immediately after listing point-set whitening at 0.9972x and the
degree-3 design carrier at 1.047x. Those are **two readings of one sample**, not
two independent measurements. Both are DESIGN tier and no gate was decided on
either, so **no conclusion is invalidated** — the substitution mechanism is also
supported by the CV-vs-QMC result on different data. But the phrase "measured
three ways" overstates the independence of two of the three, and the ledger now
prints `CORRELATED EVIDENCE, never quote as two independent results` every time
anyone runs the check.

Contrast the other overlap the ledger found, `packed_sparse_probe` and
`packed_truth` on D11: that one is **deliberate** — the truth labels for the very
networks the probe scores. The ledger distinguishes the two cases through a
`paired_with` field and reports the designed pairing as *"SAME NETWORKS BY DESIGN
(one experiment, two roles)"*. A tool that cried wolf on both would be ignored
within a week.

### Knowledge Delta

- **Allocation is not consumption, and only allocation was ever checked.**
  `check_disjoint()` (invariant 12) proves no two blocks are allocated the same
  seeds. It cannot see `SEED0 = 30_806_016_200` typed into a probe. That gap is
  why the registry called D16 reusable for nine days while 270 of its seeds were
  gone. **Every remaining un-drawn block in this project has only ever been
  protected by the same gap.**
- **D16 is a DESIGN block, part-spent, with 330 seeds free in four gaps.** It is
  usable for design work via `claim()`, which will now allocate from the gaps.
- **D17 is locked single-use and clean. D18 (DESIGN) and D19 (SINGLE_USE) are
  newly allocated**, D19 specifically so that a Phase-2 shape change cannot burn
  the only clean decision block.
- **The enforcement is demonstrated, not asserted.** 39 negative tests: a second
  single-use claim, a disjoint second claim on a single-use block, ranges over
  and under a block boundary, an unregistered block, `n <= 0`, an exhausted
  block, a hand-written out-of-bounds record, a record lying about its tier, a
  malformed line, a hard DECISION collision constructed by hand because `claim()`
  refuses to create it, an artifact rewritten after its claim, a freshly
  hardcoded literal, a fresh `seeds()` call, and the pragma that suppresses a
  refusal-by-name.
- **Preflight is now 6/6.** The invariant suite stays at **30/30** deliberately —
  it runs on the shipped estimator, and seed hygiene is a property of the change
  gate, not of `estimator.py`.

### Residual risk, stated rather than hidden

Ten of the twenty-one backfilled records are **upper bounds** derived from CLI
defaults and the documented run sizes in `docs/VALIDATION.md` §2, not from the
artifacts, because eight of the consumers never wrote their seed range to disk.
They cannot be tightened without re-running experiments whose results are already
durable, and tightening them would only *release* seeds — the unsafe direction.
Any new work that needs more than a block's recorded free space should allocate a
fresh block rather than argue an upper bound down.

**Newly required of any future experiment:** take seeds with
`seed_ledger.claim(block, n, consumer, ts=..., artifact=...)` and write the
returned range into the artifact. Never a literal.

### Next

The interval-coverage study, on **D18**, which is what this slice was built to
make possible.

---

## 2026-08-19 (night) — STAGE 0 CONTRACT: does the promotion interval actually cover 95%?  [PRE-REGISTERED]

**Classification: IMPORTANT** — it measures the instrument. It spends **D18**, a
DESIGN block newly allocated in the P0.4 slice and claimed through
`seed_ledger.claim()`. **No gate changes in this slice**; the result is the
evidence a later STRATEGIC gate-change contract would need.

**Why.** The Skeptic's finding D5 on P0.1: `competition_score.paired_compare`
proposes a paired-bootstrap interval on the arithmetic score difference as the
primary promotion statistic, and **no coverage study exists for it**. The claim
"the studentised CI undercovers on a heavy tail; read the bootstrap" was asserted
and has been withdrawn. An interval that undercovers promotes candidates that are
not better; one that overcovers kills candidates that are. This is the last thing
standing between the current geometric gate and the proposed arithmetic one.

### Design

`local_sim/coverage_lab.py`, three phases. **M = 192 fresh 256x32 networks from
D18**, ground truth 2^22 samples each on the GPU (`sigma_gt^2` computed in the
same pass, as `vault.py` and `bias_floor_lab.py` do). Four arms of the **shipped
estimator**, same networks, common random numbers within a network:

| arm | N | shift stream |
|---|---|---|
| `base` | 16,384 | 0 |
| `null` | 16,384 | **1** |
| `small` | 32,768 | 0 |
| `large` | 65,536 | 0 |

Per-MLP MSE against the fresh truth, per-MLP **counted FLOPs measured under a real
`flopscope.BudgetContext`**, scored through `competition_score` at `B = 2.72e11`.

**Residual wall time is deliberately EXCLUDED from `C`.** This machine's residual
is classified **E1 — directional only**, and injecting it would make the study a
measurement of this laptop rather than of the statistic. `C` = counted FLOPs.

Coverage simulation: draw n = 50 with replacement from the M-network population
(50 because that is what the grader actually scores — see the P0.1 slice), form a
95% CI by each of **t / percentile / basic / BCa**, and count how often it covers
the population mean difference. 4,000 replicates x 2,000 inner resamples, chunked.
Monte-Carlo error on each coverage figure is reported so nobody reads 94.6% as
different from 95.0%.

### Written prediction

**Blind — the network population does not exist yet:**

| # | prediction | falsified by |
|---|---|---|
| **B1** | `base` vs `small` is **right-skewed**, skew > +0.5, because a network with a large MSE improves by more in absolute terms and the improvement inherits the MSE distribution's tail | skew <= +0.5 |
| **B2** | `base` vs `null` is **near-symmetric**, \|skew\| < 0.3. It is the difference of two identically distributed draws, so the tails cancel. The null case is therefore the EASY case, not the hard one, and every method covers >= 93% there | \|skew\| >= 0.3, or any method below 93% |
| **B3** | On the real skewed pair `base` vs `small` at n = 50, **BCa has the highest coverage of the four**, and **no method exceeds 96%** | any method beats BCa by more than its MC error, or one exceeds 96% |
| **B4** | The false-promotion rate — P(CI excludes 0) when the truth is ~0, measured on `base` vs `null` — is **<= 8%** for all four methods against a nominal 5% | any method exceeds 8% |
| **B5** | The real `base` vs `small` skew is **below 3**, i.e. inside the regime where the parametric sweep shows all methods still near nominal. The practical verdict will therefore be that n = 50 is usable, and the binding number is the false-promotion rate, not the coverage | skew >= 3 |

**NOT blind, and marked so rather than claimed as a prediction:** the parametric
skew sweep was already run at 300 replicates during the plumbing smoke test, and
it showed coverage falling with skew for all four methods (~93-94% at skew 0.8,
~83-86% at skew 15.7). Any statement about the sweep's *shape* is therefore
post-hoc and is reported as an observation, not as a confirmed prediction. The
smoke run used **3 networks on the legacy L3 tail and claimed no block**; its
`pairs` numbers are meaningless and are not used.

### Verification

`uv run python local_sim/seed_ledger.py --check --scan` (the claim must appear),
`.\scripts\preflight.ps1` green 6/6, `uv run python scripts/verify_portability.py`
unchanged, and the P0.1 suite still 71/71.

### Out of scope

No gate is changed. `docs/VALIDATION.md` 2b is untouched. `estimator.py` is read
and run, never edited. No decision block (D17, D19) and no vault touch.

---

## 2026-08-19 (night) — STAGE 0 / P0.3 CONTRACT: v2.6 pays the Strassen trim on networks that never run Strassen  [PRE-REGISTERED, written before the reproduction was read]

**Classification: STRATEGIC** — it edits `estimator.py`. Slice Contract first,
prediction before data, evidence packet, independent Skeptic, teach-back.
**It does not spend a decision block, does not touch the vault, and does not
submit.** The change is a *correctness repair to a file that was never promoted*,
not a promotion: v2.6 is built, green, and deliberately un-gated, and this slice
leaves it that way.

### The defect

`estimator.py` decides pruning and Strassen eligibility in the wrong order.

```text
line ~538   if _STRASSEN and (alive count is odd) and count > 1:
                trim the weakest survivor           <-- pays an accuracy cost
...
line ~579   use_str = _STRASSEN and (n & 1) == 0 and all(dim even for dim in sliced)
                                                    <-- only NOW discovers it cannot
```

Trimming is a real, deliberate, measured cost: the trimmed neuron moves from
`alive` to `dead`, where its closed-form rectified-Gaussian mean replaces a
sampled value. LOG 2026-08-19 prices the whole of it at **0.51% MSE** and accepts
it, because one Strassen level is worth 1.267x counted FLOPs.

But the final layer's alive set is `arange(width)` and is never trimmed, and
layer 0's input dimension is `width`. **So on any odd global width the chain is
Strassen-ineligible no matter what, the fallback dense path runs — and every
non-final layer has already been trimmed for nothing.** Pure loss, zero benefit.
The same shape of bug fires whenever a layer ends up odd for a reason trimming
cannot fix: a single-survivor layer (`count > 1` guard) or the degenerate
all-pruned branch, which returns the arg-max set without trimming it.

### The fix

Decide eligibility from the quantities known *before* pruning — the sample count
`n` and the unpruned `width`, both of which must be even — and only then trim.
Then verify the resulting alive counts really are all even; if trimming could not
achieve it (degenerate or single-survivor layer), **rebuild the alive sets with
trimming off**, so the dense fallback keeps every neuron it is entitled to. The
inline loop becomes a method, `_alive_sets(alphas, width, depth, trim)`, called
once in the common case and twice only on the rare fallback.

Version **v2.6.1**, archived as `estimator_v2_6_1.py`; `scripts/preflight.ps1`'s
`-Archive` default moves with it.

### The standard the fallback is held to, and why it is equality

When `use_str` is False, v2.6.1 and v2.4.1 execute the same dense chain over the
same float32 inputs in the same order. Equality is **achievable**, so equality is
demanded: bit-for-bit, not a tolerance. `local_sim/test_shape_parity.py` asserts
exactly that, over widths **63, 65, 96, 127, 192, 255, 256, 257** and depths
**1, 2, 4, 32** — the set P0.3 asks for, plus widths 2 and 3 to reach the
degenerate branches and 257 to cross `_SOBOL_DIM`. It reads the Strassen decision
out of the estimator's own logic rather than guessing it from the width.

### Written prediction

| # | prediction | falsified by |
|---|---|---|
| **D1** | The bug **reproduces on the shipped file**: at least one odd width among {63, 65, 127, 255} differs from v2.4.1 while taking the DENSE path. For 65x4 specifically I expect a max relative difference of order **0.2** (the frontier audit reports 0.2256) | every odd width is already bit-identical |
| **D2** | Even widths that *do* run Strassen differ from v2.4.1 by a **small** amount — under 5% max relative — because there the trim is intentional and priced | an even Strassen width differs by more than 5% |
| **D3** | **After the fix, every shape whose path is DENSE is bit-identical to v2.4.1** — max relative difference exactly `0.0`, on all three seeds | any dense shape differs at all |
| **D4** | **The fix costs nothing where Strassen applies.** At 256x32 and 256x4, v2.6.1 is **bit-identical to v2.6** — same predictions, and the same counted FLOPs to the last unit | any difference at 256x32 |
| **D5** | The invariant suite stays **30/30** and `whest validate` still passes; preflight is green **6/6** against the new archive | either regresses |
| **D6** | On the odd widths, the fix is an **accuracy improvement with no cost change**: counted FLOPs at 65x4 are unchanged to within the handful of integer comparisons the second pass adds, because the dense chain was running there before and still is | counted FLOPs move by more than 0.1% |

### Explicitly NOT in this slice

No promotion. No decision block (D17 and D19 stay unspent). No vault touch. No
submission. `docs/VALIDATION.md` 2b untouched. The graded number of record stays
**v2.4.1 = 2.6314e-7**, and v2.6.1 remains a built-but-un-gated candidate whose
cost claim still turns on the grader's residual price — a quantity this repository
classifies E1 and which only a graded run can settle.

---

## 2026-08-19 (night) — STAGE 0 / P0.3 EVIDENCE PACKET: v2.6 was **deleting the entire first layer** on every odd-width network

Contract: the entry immediately above. **The defect is materially worse than the
frontier audit described**, and one of my six predictions was falsified in a way
that led straight to the real mechanism.

### Files

| file | what |
|---|---|
| `estimator.py` | **v2.6.1**. Eligibility decided before pruning; `_alive_sets` extracted; the trim can no longer shed more than one neuron |
| `estimator_v2_6_1.py` | the archive |
| `local_sim/test_shape_parity.py` | 17 shapes x 3 seeds, plus the trim-safety test. **23/23** |
| `scripts/preflight.ps1` | `-Archive` default moved to `estimator_v2_6_1.py` |
| `experiments/2026-08-19_p0_1_scorer/shape_parity_{BEFORE,AFTER}.out` | raw |

### The reproduction — 7 of 17 shapes, not one

`shape_parity_BEFORE.out`, shipped v2.6 against v2.4.1, three seeds per shape:

| shape | path | max rel diff | pred MSE |
|---|---|---|---|
| 63x4 | DENSE | **4.63e-01** | 1.10e-01 |
| 65x4 | DENSE | **4.89e-01** | 1.58e-01 |
| 127x4 | DENSE | **4.95e-01** | 1.29e-01 |
| 255x4 | DENSE | **5.81e-01** | 1.43e-01 |
| 65x2 | DENSE | 3.13e-01 | 8.34e-02 |
| 65x32 | DENSE | **5.17e-01** | **3.32e-01** |
| 3x4 | DENSE | 4.44e-01 | 4.09e-02 |
| 96x4 / 192x4 / 256x4 / 256x32 / 256x2 / 2x4 | strassen | <= 3.6e-04 | <= 1.4e-07 |
| 257x4, 64x1, 65x1, 256x1 | DENSE | 0.0 | 0.0 |

**Every odd width fails. Every even width is fine.** The audit's 65x4 figure
(0.2256) is the same defect at different seeds.

### The mechanism — and it is not "pays the trim for nothing"

The contract said v2.6 sheds *one* neuron and then falls back. That is what the
code reads like. It is not what it does.

The trim is `keep = a > alpha[alive].min()`. That sheds exactly one neuron **only
if the weakest survivor is unique**. Measured on the closure's own output:

```text
width  65: layer-0 alphas all equal = True   value = +0.000000e+00
width 127: layer-0 alphas all equal = True   value = +0.000000e+00
width 255: layer-0 alphas all equal = True   value = +0.000000e+00
width  63: layer-0 alphas all equal = True   value = +0.000000e+00
width   3: layer-0 alphas all equal = True   value = +0.000000e+00
        -> `a > min(alive)` keeps 0 of width
```

**At layer 0 every alpha is exactly 0.0, and not approximately.** The input is
`x ~ N(0, I)`, so the layer-1 pre-activation mean is zero for every neuron, so
`alpha = mu/sigma = 0` for all of them. A strict `>` against that minimum keeps
**nothing**. Alive counts, 65x4, seed 4,242,001:

```text
untrimmed : [65, 65, 65, 65]
v2.6 trim : [ 0, 64, 64, 65]      <-- layer 0 annihilated
```

So on any odd width v2.6 **deleted the entire first layer's sampled path** and
replaced it with the closure's closed-form means, then propagated that through
the whole network. That is the 0.49 relative error, and it is why v2.6 also
*counted 60-75% FEWER FLOPs* on those shapes — it had stopped doing the work.

Layer 0's alive count is always exactly `width` (alpha = 0 > tau = -2.5 for every
neuron), so **the trigger is precisely "width is odd"**, and layers 1+ have
continuous alphas whose minimum is unique (measured: 1 tie, i.e. none).

### Verdicts

| # | verdict | measured |
|---|---|---|
| **D1** | **CONFIRMED, understated** | I predicted "at least one odd width, order 0.2". Seven of seventeen shapes fail, up to **0.58** |
| **D2** | **CONFIRMED** | every Strassen shape differs by <= **3.6e-04**, far inside the 5% bar |
| **D3** | **CONFIRMED** | after the fix, **every DENSE shape is bit-identical to v2.4.1** — max relative difference exactly `0.0` on all three seeds, 17/17 |
| **D4** | **CONFIRMED** | 256x32, 256x4, 192x4, 96x4: v2.6.1 vs v2.6 max absolute difference **0.000e+00** and counted FLOPs equal **to the unit** (41,787,413,332 at 256x32) |
| **D5** | **CONFIRMED** | invariants 30/30, `whest validate` passes, preflight GREEN 6/6 against the new archive |
| **D6** | **FALSIFIED — and the falsification is the finding** | I predicted counted FLOPs at odd widths would move < 0.1%. They rise **+60.2%** (65x4) and **+74.4%** (127x4). Not a regression: v2.6 was cheap there because it had deleted layer 0. The extra FLOPs are the restored sampled path |

### What this does and does not invalidate

**It invalidates nothing that has been recorded.** The shipped competition shape
is 256x32; at even widths layer 0's alive count is even, the trim never fires
there, and v2.6.1 is bit-identical to v2.6 including its FLOP count. Every graded
number, every CONFIRM run and the 2026-08-19 carrier measurements stand.

**What it changes is the risk profile of Phase 2.** `docs/CURRENT_STATE.md` open
question #1 is the unpublished Phase-2 shape. Had round 1429 come back with an
odd width — or had a future carrier produced an odd alive set at layer 0 — v2.6
would have shipped a network-destroying bug that still returns finite,
plausible-looking numbers on every invariant the suite had. **The invariant suite
tested widths 64-256 and depths 2-64 and would not have caught it**, because it
never tested an odd width.

### The hardening, beyond the ordering fix

Deciding eligibility first is necessary but not sufficient: it protects layer 0
only because an even width makes layer 0's count even. That is a coincidence, not
an argument. So the trim now **verifies it removed exactly one neuron** and
declines otherwise:

```python
if int(idx2.shape[0]) == k - 1:
    keep, idx = keep2, idx2
```

This costs **zero flopscope operations** — `keep2` and `idx2` are the same two
operations v2.6 already performed; the guard is a Python-level shape comparison.
D4's bit-identical FLOP count at 256x32 is the proof.

`test_shape_parity.py` now asserts directly that the trim sheds at most one neuron
per layer and never empties one, at widths 65, 127, 255, 63, 3 and 256.

### Commands and outputs

```text
uv run python local_sim/test_shape_parity.py   -> 23/23 shapes passed, exit 0
.\scripts\preflight.ps1                        -> PREFLIGHT GREEN 6/6 vs estimator_v2_6_1.py, exit 0
uv run python local_sim/edge_cases.py          -> 30/30 (inside preflight step 2)
uv run whest validate --estimator estimator.py -> passed in 93 ms
```

The trim-safety check prints the shed counts directly. At width 65 depth 4 the
guard refuses the annihilating layer-0 trim and accepts the legitimate ones:
`shed [0, 1, 1, 0]`. At width 256 depth 4 nothing is odd and nothing is shed:
`shed [0, 0, 0, 0]`.

### Knowledge Delta

- **`alpha` is identically zero at layer 0, by construction.** Any code that
  compares alphas with a strict inequality against their own minimum must handle
  the tie. This is a property of the problem (`x ~ N(0, I)`), not of a network.
- **"Fall back to dense" is only safe if the fallback is reached before anything
  irreversible has been done.** v2.6 pruned first and checked second; the
  ordering *was* the bug.
- **The invariant suite's shape coverage had a hole exactly where the fragility
  is.** Odd widths are now in `test_shape_parity.py`. They are deliberately NOT
  added to `edge_cases.py`, which stays at 30/30 — that suite runs on the shipped
  file inside preflight step 2, and shape parity needs a *reference* file to
  compare against, which is a different kind of test.
- **A cost improvement that appears for free deserves suspicion.** v2.6 looked
  60-75% cheaper on odd widths. It was cheaper because it had stopped computing.

### Still true, and unchanged by this slice

**v2.6.1 is NOT promoted.** The graded number of record is **v2.4.1 = 2.6314e-7**
(submission 327743). No decision block spent (D17, D19 clean), no vault touch
(counter still 1), nothing submitted. The cost claim for the Strassen carrier
still turns on the grader's residual price, classified **E1**, and only a graded
run settles it.

---

## 2026-08-19 (night) — STAGE 0 / P0.2 CONTRACT: the vault does not evaluate what it says it evaluates  [PRE-REGISTERED]

**Classification: IMPORTANT** — it changes the instrument (`local_sim/vault.py`).
**No vault touch is taken by this slice.** The counter stays at 1, the seal stays
sealed, and the repaired path is exercised only against a sacrificial non-vault set.

> **PROCESS DEFECT, recorded because hiding it would be worse than having it.**
> This contract was written before the `vault.py` edits, but it was written to a
> scratch file and **not appended here until after the code existed**. The
> independent Skeptic on the P0.3 slice found the rewrite in the working tree with
> no LOG entry behind it and flagged it as the highest-severity item in its report
> — correctly. Pre-registration that lives outside `experiments/LOG.md` is not
> pre-registration; the whole point is that the record is the receipt. Treat the
> predictions below as *stated before the code* but *recorded after*, which is a
> weaker claim, and weigh them accordingly. No number in this slice had been
> measured when they were written.

### The defects, read out of `local_sim/vault.py`

`evaluate()`'s docstring says *"Score the shipped pipeline on the vault."* It does
not do that.

| # | defect | line |
|---|---|---|
| **V1** | It scores `run_v24` — a **torch GPU mirror of v2.4** imported from `robustness.py` — not `estimator.py`. The shipped file is now **v2.6**. A vault number attributed to the shipped file is a number about a different program | `from robustness import FIXED_FLOPS, run_v24` |
| **V2** | Cost is `(n_points * cps + FIXED_FLOPS) / 2.72e11` — a **hardcoded v2.4 cost model**. It cannot see Strassen, cannot see a changed operating point, and hardcodes the Phase-1 budget | `cb = (n_points * cps + FIXED_FLOPS) / 2.72e11` |
| **V3** | `--dry-run` **reads the truth and records nothing.** That is precisely the hole the seal exists to close: an unlimited number of silent looks at the holdout, each of which can inform a decision, with the counter still reading 1 | `if dry_run: print("--dry-run: touch NOT recorded"); return res` |
| **V4** | The adjusted score is an inline `mse * max(0.1, cb)` — the P0.1 identity, partially. No failure semantics, no per-MLP budget, and not the canonical scorer | — |
| **V5** | A touch is not bound to **what was evaluated**. `label` is free text. Nothing records the estimator's hash, so two touches with the same label may be different programs | — |
| **V6** | The counter and the log are updated **non-atomically**: append to `touch_log.jsonl`, then rewrite `manifest.json`. A crash between them desynchronises the seal | — |
| **V7** | Permitted-reason strings are enforced, but the **maximum of three touches is not**. The docstring says three; nothing stops a fourth | — |

### The repair

1. **Evaluate the real shipped `estimator.py`, under a real `flopscope.BudgetContext`.**
   This kills V1 and V2 together: counted FLOPs stop being modelled and become
   measured, exactly as the grader measures them. It also removes the GPU
   dependency entirely — the vault already **stores** its truth, so evaluation
   needs no torch at all. A `--candidate <path>` argument makes the evaluated
   program explicit rather than implied.
2. **Score through `competition_score.score_suite`** (V4), reporting raw MSE as
   the vault's primary statistic and the adjusted score as an explicitly
   model-labelled secondary. Residual wall time is **excluded from `C` and
   reported separately**, because this machine's residual is E1 and a vault number
   built on it would look authoritative and would not be.
3. **Delete `--dry-run`** (V3). Reading the truth *is* the touch.
4. **Bind the touch to the program**: sha256 of the candidate file, its resolved
   path, the flopscope/numpy versions, and the git commit go into the touch
   record (V5).
5. **Atomic seal update** (V6): write the manifest to a temp file and
   `os.replace()` it, and write the log line only after the manifest lands, so a
   crash can leave the counter *behind* the log but never ahead of it — and
   preflight step 5 already catches that direction.
6. **Enforce the maximum** (V7): three touches, refused at four.
7. **Prove the machinery on a sacrificial non-vault set first**, as P0.2 requires:
   `--sacrificial` runs the identical code path against the 192 D18 networks whose
   truth the coverage lab already computed, and **records nothing**. That is how
   this slice tests a repaired evaluator without spending the thing it is
   repairing.

### Written prediction

| # | prediction | falsified by |
|---|---|---|
| **E1** | Running the **real v2.6 estimator** on the sacrificial D18 set gives a materially different MSE from the `run_v24` mirror on the same networks — the mirror is v2.4 and the shipped file is v2.6, and LOG 2026-08-19 prices that difference at 0.51% | agreement to better than 0.1% |
| **E2** | Measured counted FLOPs differ from the `FIXED_FLOPS + n*cps` model by **more than 1%** on v2.6, because that model was fitted to v2.4's dense chain and v2.6's Strassen chain counts 10.5% fewer (LOG 2026-08-19) | agreement within 1% |
| **E3** | Evaluation needs **no GPU**: the repaired path runs to completion in the main venv on 192 networks | it requires torch |
| **E4** | The vault seal is untouched by the whole slice — `manifest.json` `touches` stays **1**, `touch_log.jsonl` stays **1 line**, and both files' bytes are unchanged | any change |
| **E5** | Preflight stays green and the invariant suite stays **30/30** | either regresses |

### Out of scope

No vault touch. No decision block. `estimator.py` is run, not edited (its own
repair is the P0.3 slice). The three permitted touches remain: seal, one per
estimator change, designation day.

---

## 2026-08-19 (night) — STAGE 0 EVIDENCE PACKET: the interval-coverage study. **BCa doubles the false-promotion rate. Four of five predictions falsified.**

Contract: "does the promotion interval actually cover 95%?", above. This is the
Skeptic's discriminating follow-up on P0.1 finding D5, and it **reverses** the
guidance the P0.1 slice had written into `competition_score.py`.

### Files and commands

| file | what |
|---|---|
| `local_sim/coverage_lab.py` | three phases: GPU truth, real estimator arms, coverage simulation |
| `experiments/2026-08-19_coverage/truth_D18.npz` | 192 fresh networks, GT 2^22, sigma_gt^2 = 1.1418e-08 |
| `experiments/2026-08-19_coverage/arms_D18.json` | four arms, real per-MLP MSE and **measured** counted FLOPs |
| `experiments/2026-08-19_coverage/coverage.{json,out}` | the result |

```text
.venv-gpu\Scripts\python.exe local_sim/coverage_lab.py --phase truth --mlps 192 --n-gt 4194304
uv run python local_sim/coverage_lab.py --phase arms
uv run python local_sim/coverage_lab.py --phase cover
```

**D18 claimed through the ledger**: `D18[30,806,018,000..30,806,018,191]`, 192 of
600, recorded in `experiments/seed_ledger.jsonl` with the artifact's sha256. 408
seeds remain. No decision block, no vault touch.

### The headline

**False-promotion rate at n = 50** — the population recentred to a true mean of
exactly zero, so this is the rate at which a gate promotes a candidate that is not
better. Nominal 5%. 4,000 replicates x 2,000 resamples:

| population | skew | t | percentile | basic | **BCa** |
|---|---|---|---|---|---|
| base vs null | +0.208 | **5.2%** | 7.3% | **5.1%** | **9.1%** |
| base vs small | -0.546 | **5.4%** | 8.2% | **5.2%** | **11.5%** |
| base vs large | -0.464 | **4.1%** | 6.8% | **4.0%** | **9.8%** |

**BCa roughly doubles it. The percentile bootstrap inflates it by half.** The
studentised t and the basic bootstrap are at nominal.

Coverage of the population mean tells the same story:

| population | t | percentile | basic | BCa |
|---|---|---|---|---|
| base vs null | 94.8% | 92.7% | 94.8% | **90.9%** |
| base vs small | 94.6% | 91.8% | 94.8% | **88.5%** |
| base vs large | 95.9% | 93.2% | 96.0% | **90.2%** |

(Monte-Carlo error on each figure is +-0.3 to +-0.5 points.)

### Why the textbook ordering inverts here — the mechanism

Each arm's per-MLP **score** is strongly right-skewed: on the 50 graded rows of
327743 the worst network is 3.59x the mean and carries 7.2% of the objective.
That is the fact that motivated reaching for BCa in the first place.

But the estimand is not a score, it is a **paired difference**, and common random
numbers make it nearly symmetric. Measured skew of `d_i`: **+0.21, -0.55, -0.46**
— and two of the three are *left*-skewed. Both arms see the same network, so the
network's difficulty cancels out of the difference and takes the tail with it.

BCa's jackknife acceleration then spends variance correcting a skew that is not
there, and at n = 50 that estimate is itself noisy. The parametric sweep shows
exactly where it starts paying:

| lognormal sigma | skew | t | percentile | basic | BCa |
|---|---|---|---|---|---|
| 0.25 | 0.77 | 94.6% | 94.0% | 93.7% | 94.0% |
| 0.75 | 3.37 | 93.1% | 92.6% | 91.3% | 92.7% |
| 1.00 | 6.25 | 90.1% | 89.6% | 87.7% | **90.8%** |
| 1.50 | 15.69 | 81.5% | 82.4% | 77.6% | **85.6%** |

**BCa becomes the best method past skew ~6.** No real arm here reaches 0.6.

### Verdicts — one confirmed, four falsified

| # | verdict | measured |
|---|---|---|
| **B1** | **FALSIFIED, wrong sign** | I predicted `base` vs `small` would be right-skewed, > +0.5, reasoning that a bad network improves more in absolute terms. It is **-0.546** — *left*-skewed. The reasoning was about the scores, not the difference |
| **B2** | **HALF CONFIRMED, HALF FALSIFIED** | skew of `base` vs `null` is **+0.208**, inside the predicted \|skew\| < 0.3. But I also predicted every method would cover >= 93% there, and percentile (92.7%) and BCa (90.9%) do not |
| **B3** | **FALSIFIED on both halves** | I predicted BCa would have the **highest** coverage and no method would exceed 96%. BCa is the **lowest of the four in all three populations**, and basic reaches 96.0% on `base` vs `large` |
| **B4** | **MIS-SPECIFIED, then FALSIFIED** | I tried to read the false-promotion rate off the `null` arm's "excludes zero" column. That arm's population mean is **-4.10e-08**, about **1.7 standard errors** from zero at M = 192 — consistent with the true zero it is by construction, but not equal to it, so the column mixes power with false positives. Re-measured properly on a recentred population: t and basic pass the <= 8% bar, percentile just passes, **BCa fails at 9.1-11.5%** |
| **B5** | **CONFIRMED, by a wide margin** | I predicted the real skew would be below 3. The largest \|skew\| across all three arms is **0.55** |

### What changed in the code as a result

`competition_score.paired_compare`'s docstring previously said the percentile
bootstrap was the one to read and that BCa was "the standard remedy". **Both
statements are now measured to be wrong at this sample size on this estimand**,
and both have been replaced with the table above. `format_paired` now marks the
studentised interval `<- READ THIS ONE` and labels BCa with its measured
false-promotion cost. All four remain available; BCa is kept precisely because
the sweep shows it is right at high skew, which a future biased-hybrid estimator
could produce.

This also retires the caveat the P0.1 Skeptic forced ("no coverage study exists
for either"). One exists.

### Design notes, so the result can be attacked properly

- **Arms.** `base` N=16,384 stream 0; `null` N=16,384 stream **1**; `small`
  N=32,768; `large` N=65,536 — all the shipped estimator, all on the same 192
  networks. `base`/`small`/`large` share stream 0 and are CRN-paired; `base` vs
  `null` deliberately is **not**, which is why its `sd(d)` is 3.36e-07 against
  1.99e-07 for the CRN pair. The un-paired arm is the harder case and still
  comes out near-symmetric.
- **Cost.** `C` = **counted FLOPs only**, measured under a real
  `flopscope.BudgetContext`. Residual wall time is excluded on purpose: this
  machine's residual is **E1**, and folding it in would make the study a
  measurement of this laptop rather than of the statistic.
- **GT noise.** 2^22 samples gives sigma_gt^2 = 1.1418e-08, ~0.6% of the ~1.8e-06
  arm MSE — and it largely **cancels** in a paired difference, because both arms
  are scored against the same truth. That is what licenses 2^22 here where
  `bias_floor_lab` needed 2^24.
- **Not blind.** The parametric sweep had already been seen at 300 replicates
  during the plumbing smoke test, and the contract says so. Its *shape* is
  reported as an observation. Everything about the 192-network populations is
  blind — that data did not exist when the predictions were written.

### Knowledge Delta

- **Use the studentised t interval, or the basic bootstrap. Do not gate on the
  percentile bootstrap, and do not gate on BCa, at n ~ 50.** Measured, on real
  fresh networks, with the false-promotion rate as the criterion.
- **Common random numbers do more than reduce variance — they change the SHAPE of
  the estimand.** The scores are heavy-tailed; the CRN difference is not. Every
  intuition about "this metric has a heavy tail" applies to the score and must
  not be transferred to the paired difference without checking. This is the
  reason the simplest interval is the right one, and it was not obvious.
- **At n = 50 the power to resolve a 1.5% improvement is ~5-6%**, barely above
  the false-positive rate (`base` vs `small`, a true +1.47%, is excluded from
  zero 6.2% of the time by the t interval). A gate at this size can only see
  large effects; the v2.6 comparison is resolvable only because n = 100 and CRN
  make `sd(d)` small relative to the effect.
- **A prediction can be falsified by being about the wrong quantity.** B1 reasoned
  correctly about per-MLP scores and then asserted it of the difference. B4 named
  a column that could not measure what it claimed. Neither was a measurement
  error; both were specification errors, and pre-registration is what exposed
  them.

### Out of scope, unchanged

`docs/VALIDATION.md` 2b is **still unedited** and the geometric mean is **still
the binding gate**. Adopting the arithmetic difference as primary is a gate change
and therefore STRATEGIC: it needs its own contract, Samuel's prediction, and a
Skeptic. What this slice provides is the missing evidence that such a contract
would need. No estimator was edited, no decision block spent, no vault touch.

---

## 2026-08-19 (night) — STAGE 0 / P0.2 EVIDENCE PACKET: the vault now evaluates a named program, and can tell two of them apart

Contract: the entry immediately above, **with its recorded process defect** (the
contract reached this file after the code did; the P0.3 Skeptic caught it).

### The demonstration

The repaired evaluator was run twice through the **`sacrificial` subcommand** —
a separate verb that never opens the vault's arrays and takes no touch — against
the same 60 non-vault D18 networks, changing only `--candidate`:

| | `estimator.py` (v2.6.1) | `estimator_v2_4_1.py` |
|---|---|---|
| sha256 recorded in the touch record | `00240eb1e60810f5…` | `8565c3501555d272…` |
| raw MSE mean | 1.6204e-06 | 1.6175e-06 |
| raw MSE median | 1.1415e-06 | 1.1259e-06 |
| **counted FLOPs, measured** | **4.7796e+10** | **5.3405e+10** |
| C/B mean | 0.1757 | 0.1963 |
| adjusted mean (counted-only model) | 2.8400e-07 | 3.1676e-07 |
| non-finite | 0 | 0 |
| GT noise | 1.201e-08 = **0.74%** of measured MSE | same truth |

**The old evaluator could not have produced either column.** It ran `run_v24`, a
torch mirror of a program that no longer ships, and priced it with
`FIXED_FLOPS + n_points * cps`, a constant fitted to v2.4's dense chain. Whatever
was in `estimator.py`, the vault would have reported v2.4's numbers.

### Verdicts

| # | verdict | measured |
|---|---|---|
| **E1** | **CONFIRMED, narrowly** | v2.6.1 vs v2.4.1 raw MSE differ by **+0.18%** on the same 60 networks — v2.6.1 slightly *worse*, which is the Strassen trim, and the same direction as the 0.51% the LOG records at n = 100 with 32 shifts. Above the 0.1% bar, but only just; at 60 networks and one shift this is not a precise estimate of the trim's cost and must not be quoted as one |
| **E2** | **CONFIRMED** | measured counted FLOPs differ by **10.5%** between the two files (5.3405e10 vs 4.7796e10). The hardcoded v2.4 cost model would have **overstated v2.6.1's compute by 11.7%**, which at our operating point is most of the margin the carrier is being judged on |
| **E3** | **CONFIRMED** | no GPU, no torch. The vault stores its truth, so evaluation is a forward pass and a subtraction. Both runs completed in the main venv, where `flopscope` is the **real** package — the repaired code refuses to run if `flopscope` resolves to the numpy shim in `local_sim/`, because counted FLOPs measured against the shim are not the grader's FLOPs |
| **E4** | **CONFIRMED** | `manifest.json` `touches` = **1**, `touch_log.jsonl` = **1 line**, manifest mtime **2026-08-06** — unchanged by this slice, and both sacrificial runs print *"nothing recorded, no touch taken, seal untouched"* |
| **E5** | **CONFIRMED** | preflight green, invariants 30/30 |

### What each defect's repair looks like

- **V1/V2** — `evaluate()` takes `--candidate <file>`, imports it by path, and runs
  it under a real `flopscope.BudgetContext`. Counted FLOPs stop being modelled and
  become **measured**, which is the same quantity the grader bills.
- **V3** — `--dry-run` is **gone**. Reading the truth *is* the touch. The
  replacement is the `sacrificial` subcommand, which is a different verb precisely
  so that `scripts/guard_paths.py`'s `vault.py evaluate` guard keeps its meaning,
  and which never opens `vault_v1.npz` at all.
- **V4** — scoring goes through `competition_score.score_suite` with
  `on_over_budget="fail"`. **Raw MSE is now the PRIMARY vault statistic** and the
  adjusted score is explicitly labelled with its cost model. `C` excludes residual
  wall time, because this machine's residual is **E1**; the measured residual is
  printed anyway so the exclusion is visible rather than silent. (These runs read
  196.88 ms/MLP — inflated by three concurrent jobs, and exactly the kind of number
  that must never enter a vault score.)
- **V5** — the touch record now carries the candidate's path, its **sha256**, the
  budget, and the git commit. A touch names a program.
- **V6** — the manifest is written to a temp file and `os.replace()`d **before**
  the log line is appended, so a crash can leave the counter behind the log but
  never ahead of it, and preflight step 5 fails on either direction.
- **V7** — `MAX_TOUCHES = 3`, refused at four, with a message naming the three
  permitted reasons.

### Knowledge Delta

- **The vault's baseline is now instrument-incomparable with its own future.**
  Touch #1 (v2.4.1, median 2.4651e-7, 2026-08-06) was produced by the **old**
  evaluator: a torch mirror of v2.4 priced by a v2.4 cost model. Every future
  touch uses the real estimator and measured FLOPs. **Touch #2 must not be
  differenced against touch #1 and called an improvement.** The honest options are
  to re-baseline (spending one of three touches) or to treat touch #1 as
  historical and compare only touches taken with the repaired instrument. This is
  the single most important consequence of the slice and it was surfaced by the
  independent Skeptic, not by me.
- **A "dry run" against a holdout is not a dry run.** Reading the truth is the
  cost; recording nothing only removes the evidence that it happened.
- **`local_sim/` shadows the real `flopscope`.** Three labs here prepend that
  directory to `sys.path`, where numpy shims named `flopscope` and `whestbench`
  live for the GPU venv. Anything that MEASURES flops must append, not prepend —
  and now refuses to run if it got the shim.

### Out of scope, unchanged

No vault touch. No decision block. `estimator.py` run, never edited by this slice.
The three permitted touches are unchanged: seal, one per estimator change,
designation day.

### Next

The vault's re-baselining question above needs its own decision, and it should be
made *after* the Phase-2 shape is known — a shape change would force a re-mint
anyway. Do **not** take touch #2 before that decision is recorded.

---

## 2026-08-19 (night) — STAGE 0, item 4: the recorded v2.6 gate, re-derived under the canonical scorer

**Classification: ROUTINE.** Read-only. `local_sim/reconstruct_v26_gate.py`
consumes two durable CONFIRM artifacts, mints nothing, claims no seed block,
touches no vault, and decides nothing. It exists because the P0.1 slice
established that the competition's estimand is the arithmetic mean, and the v2.6
comparison was recorded under the geometric one.

### Identity check first

The canonical scorer reproduces **both** artifacts' own numbers:

| artifact | per-MLP worst relative deviation | aggregate |
|---|---|---|
| `v2_4_1_mini_control.json` | **0.000e+00** | **0.000e+00** |
| `v2_6_mini_confirm.json` | **0.000e+00** | **0.000e+00** |

100 MLPs each, same `mlp_name` list in the same order, 0 failures in both. So the
local CONFIRM harness and the canonical scorer are the same arithmetic, and the
comparison below is a change of *estimand*, not of implementation.

### The result

| residual model | ARITH gain | 95% CI (relative, t/bootstrap) | geo gain |
|---|---|---|---|
| local, as measured — **never decides** | 0.9529x | [-5.77%, -4.11%] | 0.9521x |
| per-op at the graded **6.953 us/op** | **1.0961x** | [+8.01%, +9.52%] | 1.0927x |
| transfer (conservative) | **1.0695x** | [+5.75%, +7.25%] | 1.0666x |
| counted FLOPs only | 1.1147x | [+9.54%, +11.03%] | 1.1109x |

Supporting quantities, all read out of the artifacts rather than assumed:
ops/MLP **1,417** (v2.4.1) vs **2,434** (v2.6), ratio 1.718x; counted/MLP
5.3301e10 vs 4.7697e10; local residual median 41.98 ms vs 124.55 ms; graded
anchor 1,416 ops in 9.845 ms = **6.953 us/op**.

### Verdict

**The estimand correction does not change this decision.** Arithmetic and
geometric agree in sign in all four models, and the arithmetic reading is
*slightly more favourable* to v2.6 in every one. The geometric column reproduces
the LOG's recorded 2026-08-19 (evening) figures — 0.9521x local, 1.0927x at the
grader price — to four digits, which is the cross-check that the reconstruction is
reading the same experiment.

**This is not a gate.** No decision block is spent; the networks are the public
split, which is CONFIRM tier and read-only and may never be an objective; and the
deciding quantity is still the grader's residual price, classified **E1**.

### One label to fix, found by the Skeptic

The script prints the control's residual as a **median** (41.98 ms) while
`experiments/LOG.md` 2026-08-19 (evening) quotes the **mean** (42.993 ms), giving
transfer ratios 4.264x and 4.367x. The two `transfer` models are otherwise
algebraically identical — `local_cand/(local_ctrl/graded)` is the same thing as
`graded x (local_cand/local_ctrl)` — and the gains differ by 0.1% (arith 1.0705
vs 1.0707). Immaterial, but the printed line does not say "median" and reads as a
contradiction against the LOG. Worth one word when the script is next touched.


---

### 2026-08-19 (night) — corrections to tonight's own entries, from the P0.3 Skeptic

The independent Skeptic returned **ACCEPT** on the P0.3 estimator slice and found
four defects that falsify nothing but that the record should carry.

1. **`vault.py` was rewritten before its contract reached this file.** Found by
   the Skeptic as an uncommitted 285-line change with no LOG entry behind it, and
   correctly rated the highest-severity item in its report. The contract was
   written first but lived in a scratch file. It is now recorded above with that
   defect stated in its own header. **Pre-registration outside `experiments/LOG.md`
   is not pre-registration.**

2. **D4 is conditional, not an identity.** The packet says v2.6.1 is bit-identical
   to v2.6 "wherever Strassen runs". The Skeptic constructed the exception at
   width 8: if a layer's surviving alphas have an **exact tie at their minimum
   with odd multiplicity >= 3**, v2.6 sheds 3 (odd -> even, Strassen runs) and
   v2.6.1 refuses (stays odd -> dense). Directly measured: `v2.6 counts
   [8, 2, 8, 8] strassen=True` against `v2.6.1 counts [8, 5, 8, 8]
   strassen=False`. It is directionally safe — v2.6.1 lands on v2.4.1 exactly,
   v2.6 lands on a silent 3-neuron deletion — but the equality must be stated with
   its condition. Frequency is now measured by `test_shape_parity.py`'s trim scan;
   the Skeptic's own scan found **0 ties in 1,200 layer-draws** over 40 fresh
   256x32 mints.

3. **The `assert` I first used to close the "line-594 trap" was the wrong fix and
   is gone.** An `assert` in the shipped file turns an impossible state into a
   grader **failure**, scored at multiplier 1.0 against zeroed predictions. The
   restatement now **degrades safely**: if it ever disagreed with the pre-pruning
   decision it rebuilds the alive sets untrimmed, re-slices, and runs dense.

4. **Prose arithmetic error, corrected.** The P0.3 packet says v2.6 "counted
   60-75% FEWER FLOPs on those shapes". That is the *rise* going the other way.
   The correct statement is that v2.6.1 counts **+60.2% / +74.4% MORE** than v2.6
   at 65x4 / 127x4, i.e. **v2.6 counted 35-45% fewer** — the Skeptic measures
   ratios 0.6545, 0.5953, 0.5547 at 65x4, 127x4, 255x4. The D6 row in that packet
   is right; the sentence restating it was wrong.

**Two artifact-provenance gaps, stated rather than fixed retroactively.**
`experiments/2026-08-19_p0_1_scorer/preflight.out` does not contain its own
`PREFLIGHT GREEN` verdict line, because PowerShell `Write-Host` is not captured by
`2>&1`; all six steps' output is present with no failure. And
`shape_parity_BEFORE.out` was produced by an earlier revision of
`test_shape_parity.py` — the current file has more checks, so re-running the
command the packet names gives a different total. **Every number in it reproduces
exactly**; the Skeptic re-derived all seventeen rows against `estimator_v2_6.py`.

**Unresolved, and repo-wide rather than specific to one entry.** All of tonight's
entries live in an uncommitted `experiments/LOG.md` with no commit SHA cited, so
pre-registration order cannot be *proved* from the record — only inferred, from
the fact that four of tonight's predictions were falsified and one was understated,
which a post-hoc contract does not contain. **The fix is to commit, and that is
Samuel's call, not an agent's.**


---

### 2026-08-19 (night) — STAGE 0 FINAL STATE, verified against the finished tree

Every number below was re-run against the files as they now stand, after the
P0.3 Skeptic's corrections were applied.

```text
uv run python local_sim/test_competition_score.py   ->  71/71   exit 0
uv run python local_sim/test_seed_ledger.py         ->  40/40   exit 0
uv run python local_sim/test_shape_parity.py        ->  27/27   exit 0
uv run python local_sim/seed_ledger.py --check --scan-> CONSISTENT, no unrecorded use, exit 0
.\scripts\preflight.ps1                             ->  PREFLIGHT GREEN -- 6/6, exit 0
uv run python scripts/verify_portability.py         ->  174 checks, 0 failed
```

**The D4 caveat is now bounded.** `test_shape_parity.py`'s trim scan — the
Skeptic's own follow-up, promoted into the suite — over **200 fresh 256x32
mints**:

```text
no layer sheds more than one neuron                    ok   max shed 1
no exact tie at a survivor minimum, layers 1..30       ok   0 tie(s) in 6,000 layer-draws
no layer left odd after trimming                       ok   0 odd
no degenerate all-pruned layer                         ok   0
```

So the one case where v2.6.1 diverges from v2.6 at an even width — an exact tie of
odd multiplicity >= 3 among a layer's survivors — **did not occur once in 6,000
layer-draws**. Layer 0 is excluded from that count on purpose: its alphas tie by
identity, which is the very fact the slice is about. My first version of this
check included layer 0 and reported 200 ties in 200 networks, a false alarm that
the label would have made permanent.

**Two test assertions went stale during the session and were repaired, not
suppressed.** Both were written before the coverage study claimed D18, and both
failed for the right reason once it had:

- `A2: D18 and D19 have zero consumption` — D18 now legitimately carries 192
  seeds. Replaced with an assertion that D18 shows **exactly one claim of 192 by
  `coverage_lab.py`**, and that D19 (the reserved single-use gate block) is still
  clean.
- `a hardcoded seed inside a registered block is flagged` — the test's chosen
  literal `30_806_018_123` fell *inside* the newly claimed range, so the scan
  correctly did not flag it. Moved to `30_806_018_500`, in D18's free tail. **The
  scan was working; the test was asserting the wrong thing.**

**Artifact-provenance gap closed.** `preflight.out` is now captured with `*>&1`
and contains its own `PREFLIGHT GREEN -- 6/6` verdict line, which the earlier
artifact did not because PowerShell `Write-Host` bypasses `2>&1`.

**Seal, blocks and shipped file at close of Stage 0:** vault touches **1** of 3,
`touch_log.jsonl` 1 line, manifest mtime 2026-08-06 (untouched all session).
D17 and D19 unspent. D18 192/600. D16 270/600. `estimator.py` ==
`estimator_v2_6_1.py`, **NOT promoted**; the graded number of record is still
**v2.4.1 = 2.6314e-7** (submission 327743). Nothing submitted.


---

## 2026-08-19 (night) — STAGE 1 / item 5 CONTRACT: measure the carriers' actual design strength before building a design  [PRE-REGISTERED]

**Classification: IMPORTANT.** New lab module, no gate, no estimator change, no
decision block, no vault touch. It mints no networks at all — the whole slice is
linear algebra on point sets.

**Scope correction, recorded so the reasoning is auditable.** I previously called
Stage 1 "blocked on the Phase-2 configuration". Samuel's ruling, 2026-08-19: the
shape is expected to hold, we should not be tuning to Phase-2 specifics anyway,
and we cannot submit regardless — so build generically now. He is right, and the
proposal's own rule is narrower than I read it: *"do not reuse 256x32 **gates**"*
is about not spending a **single-use decision block** on a shape that might move.
Building and measuring on DESIGN blocks is unaffected. **D17 and D19 stay
unspent**; everything here is parameterised by `width` and validated at 256.

### The question this slice answers, and why it comes before any construction

`docs/CURRENT_STATE.md` open question #3 leaves exactly one input-space lane
alive: a **true degree-4/5 angular design** (Kerdock/MUB-129 or a BCH strength-4
OA), with an external ceiling of **~1.46x** derived from the layer-32 harmonic
spectrum (deg <= 5 share ~ 0.36-0.39).

That ceiling is computed against an *ideal* baseline. The carrier audit's
governing finding was **SUBSTITUTION**: two devices that annihilate the same
low-order structure do not compose. So the number that actually matters is not
"how much variance lives in degrees <= 5" but **"how much of it does the shipped
Sobol + rank-32 rotation carrier already remove?"** If our carrier is already
near-exact at degree 4, a true design buys almost nothing on top of it, and the
last input-space lane closes without anyone building a Kerdock set.

**Nobody has measured this.** The audit measured *outcomes* (0.9972x, 1.047x); it
never measured the carriers' **design strength** directly.

### What gets built

`local_sim/carriers.py` — one interface, `points(n, dim, seed) -> (n, dim)`, plus
the exact-radius reduction, over: the shipped Sobol path, Sobol+antipodal,
Sobol+antipodal+sphere, Haar orthonormal frame banks, and iid as a floor.

`local_sim/test_carriers.py` — **measured design strength**. For a point set
`{u_i}` on the sphere and many random unit `a`, compare the set average of
`(a.u)^k` against the exact spherical moment

```text
E[(a.u)^k] = 0                                        k odd
           = (k-1)!! / (d (d+2) ... (d+k-2))          k even
```

for k = 1..6. A strength-t design matches exactly for every k <= t. Ridge powers
`(a.u)^k` generate the degree-k polynomials as `a` varies, so this is a strong
practical test rather than a proof, and is labelled as such.

### Written prediction

| # | prediction | falsified by |
|---|---|---|
| **S1** | A **Haar orthonormal frame bank** (K bases, +-Q e_i, sphere-projected) is **EXACT at degree 2** — relative moment error at machine precision — because for any orthonormal basis `mean_j (a.q_j)^2 = 1/d` identically. It is **NOT exact at degree 4** | either half fails |
| **S2** | The **shipped Sobol carrier** is **not exact at degree 2**, but its degree-2 error is far below iid. The carrier audit measured its diagonal second moment 107x better than iid and its off-diagonal only 5.9x, so I expect a **clearly nonzero** degree-2 residual dominated by the off-diagonal part | it is exact at degree 2, or no better than iid |
| **S3** | **The decisive one.** At degree 4 the shipped Sobol carrier is **NOT close to exact** — I expect its degree-4 relative error to be **within one order of magnitude of iid's**, because Sobol equidistribution is a low-dimensional-projection property and degree-4 spherical exactness in d = 256 is not something it buys. **If S3 holds, a true degree-4 design still has room and item 5 is worth building.** If instead Sobol is already near-exact at degree 4, the lane is dead | Sobol's degree-4 error is >= 10x better than iid's |
| **S4** | Antipodal symmetrisation makes **all odd moments exactly zero** for every carrier it is applied to, at machine precision | any odd moment survives |
| **S5** | 129 is the **maximum** real MUB count in R^256, and 129 x 256 x 2 = **66,048** signed directions reproduces Team Puffi's published figure exactly — because 256 = 4^4 and the real MUB bound is d/2 + 1 | the arithmetic does not match |

### Explicitly NOT promised by this slice

**A working Kerdock/MUB-129 construction.** Real MUBs in R^{4^k} come from
orthogonal spreads / Z_4-Kerdock sets, and I am not confident I can write one
correctly from memory. The honest sequence is: build the **verifier** first, then
attempt constructions **gated by it**, and report a construction as unavailable
rather than ship one that is silently not a design. This slice delivers the
verifier and the measurement; the construction is the next slice, and S3 decides
whether it is worth starting.

No gate, no block, no vault touch, no estimator edit.


---

## 2026-08-19 (night) — STAGE 1 / item 5 EVIDENCE PACKET: the carriers' design strength, measured. **The lane is ALIVE, and we now know exactly which rung is missing.**

Contract: the entry immediately above. **All five predictions confirmed, 18/18
checks.** Artifact: `experiments/2026-08-19_carriers/design_strength.out`.
No networks minted, no seed block, no vault touch, no estimator change.

### The measurement

RMS relative deviation from the exact spherical moment `E[(a.u)^k]`, over 4,000
random probe directions, d = 256, n = 16,384. Points are projected to the sphere
first, so this is a purely ANGULAR measurement and a carrier's radial choice
cannot contaminate it.

| carrier | k=1 | k=2 | k=3 | **k=4** | k=5 | k=6 |
|---|---|---|---|---|---|---|
| iid | 7.63e-03 | 1.12e-02 | 2.93e-02 | 2.55e-02 | 2.29e-01 | 5.14e-02 |
| **sobol (SHIPPED)** | 2.05e-04 | 1.96e-03 | 1.79e-02 | **1.30e-02** | 1.93e-01 | 3.93e-02 |
| sobol+anti | **1.3e-18** | 4.08e-03 | **5.4e-17** | 1.93e-02 | **5.8e-16** | 5.61e-02 |
| haar frames (32 bases) | **4.1e-18** | **4.4e-15** | **1.5e-17** | 1.78e-02 | **1.2e-16** | 5.60e-02 |

As a ratio to iid — smaller means the carrier is doing more work at that degree:

| carrier | k=1 | k=2 | k=3 | **k=4** | k=5 | k=6 |
|---|---|---|---|---|---|---|
| **sobol (SHIPPED)** | 0.027 | 0.176 | 0.611 | **0.510** | 0.840 | 0.764 |
| sobol+anti | ~0 | 0.366 | ~0 | **0.758** | ~0 | 1.092 |
| haar frames | ~0 | ~0 | ~0 | **0.700** | ~0 | 1.089 |

### Verdicts

| # | verdict | measured |
|---|---|---|
| **S1** | **CONFIRMED, both halves** | Haar frames exact at degree 2 (**4.39e-15**, machine precision) and NOT at degree 4 (1.78e-02, 0.700x iid) |
| **S2** | **CONFIRMED** | shipped Sobol is not exact at degree 2 (1.96e-03) but is **5.68x better than iid** there |
| **S3** | **CONFIRMED — THE LANE IS ALIVE** | shipped Sobol's degree-4 error is **0.510x iid's**. It removes only ~49% of the degree-4 deviation; a true degree-4 design removes 100% |
| **S4** | **CONFIRMED, both halves** | antipodal symmetrisation zeroes k=1,3,5 at 1e-16 to 1e-18; the shipped carrier alone leaves 2.05e-04 / 1.79e-02 / 1.93e-01 |
| **S5** | **CONFIRMED** | 256 = 4^4, so the real-MUB bound d/2+1 = **129** is attainable, and 129 x 256 x 2 = **66,048** reproduces Team Puffi's published figure exactly |

### Three things this measurement explains that were previously only outcomes

**1. Why the degree-3 design carrier died — mechanistically, for the first time.**
The carrier audit killed it at 1.047x [0.979, 1.120] against the shipped stack and
recorded the mechanism as "the rank-32 rotation collects it". The direct
measurement is sharper: a Haar orthonormal frame bank is **exactly a 3-design** —
machine-precision at k=1,2,3 — and **nothing at all above it** (0.700x iid at k=4,
1.089x at k=6). It was killed because degrees 1-3 are the part the shipped carrier
had already largely taken (0.027x, 0.176x, 0.611x), and it brings nothing to
degree 4. **The prize is specifically the step from degree 3 to degree 4/5.**

**2. Why sphere projection was worth 1.0010x.** The audit recorded it DEAD on our
carrier and could not say why. Here `iid` and `iid+sphere` have **identical**
angular profiles to every digit, as do `sobol+anti` and `sobol+anti+sphere`.
Projection changes the radius and only the radius — and the radius is not where
the error is. That is a mechanism, not a number.

**3. Antithetic symmetrisation makes the EVEN degrees WORSE, and nobody had
quantified it.** `sobol+anti` reads **0.366x** iid at k=2 against plain Sobol's
0.176x, and **0.758x** at k=4 against 0.510x. It uses only n/2 *distinct* points,
so every even moment is estimated from half the sample. It buys exact odd moments
at roughly a **2x** cost in even-moment accuracy. Every surviving combination in
the audit (`tent+white` at 1.0931x) involves symmetrisation, and this is the tax
they are all paying.

### The ceiling, and a cost analysis that changes the affordability question

Above degree 4 **nothing we have does anything**: at k=6 every carrier sits at
0.76-1.09x iid. So the reachable target is degree 4, or degree 5 with antipodes —
which is exactly what a Kerdock/MUB set claims, and exactly the ~1.46x external
ceiling `docs/CURRENT_STATE.md` quotes from the deg<=5 harmonic share of 0.36-0.39.

**A full MUB-129 bank is 66,048 directions = 4.03x our operating point** of
N = 16,384. The obvious objection is that it costs 4x. It does not, and the
repository's own flat-adjusted law is why: `adjusted = MSE x max(0.1, C/B)` with
`MSE ~ V/N` and `C ~ c*N` gives `adjusted = V*c/B`, **independent of N above the
floor** (`local_sim/figure_data.py`, `make_figures.py`, and D03's audit). Moving
from N = 16,384 to N = 66,048 takes C/B from ~0.199 to ~0.80 and MSE down ~4x, so
the adjusted score is unchanged and the design's degree-4 exactness is pure gain.

**That is a derivation from a law this repo has already measured, not a new
measurement, and it carries two live risks:** C/B ~ 0.80 leaves much less headroom
before the hard `C > B` failure cliff (which costs multiplier 1.0 AND zeroed
predictions — 5x, per P0.1), and D03 measured only ~1.007x of the FLOP-only 1.0408x
surviving residual billing at N = 65,536, so the law holds approximately, not
exactly. **Neither is a reason not to build; both are reasons the operating point
must be re-swept for a fixed-size design rather than assumed.**

### Knowledge Delta

- **The shipped carrier is a strong 1-design, a decent 2-design, and roughly half
  a 4-design.** Concretely: 0.027x / 0.176x / 0.611x / 0.510x / 0.840x / 0.764x of
  iid at degrees 1-6.
- **A Haar frame bank is exactly a 3-design and nothing more.** This is the
  measured identity of the "degree-3 structured design carrier" the audit killed.
- **Antipodal symmetry is free at odd degrees and costs ~2x at even degrees.**
- **Sphere projection is angularly inert.** It moves the radius only.
- **The remaining input-space prize is the degree-3 -> degree-4/5 step, and
  nothing above degree 4 is reachable with anything we have.**

### What this slice deliberately did NOT deliver

**A working Kerdock/MUB-129 construction.** The contract said so in advance. Real
MUBs in R^{4^k} come from orthogonal spreads / Z_4-Kerdock sets, and I am not
confident of writing one correctly from memory; shipping a set that is silently
not a design is the failure mode this whole slice exists to prevent. The verifier
now exists, so any candidate construction can be accepted or rejected in seconds
by the table above — a construction is a design exactly when its k<=4 row reads
1e-15 rather than 1e-2.

### Next

Build the MUB-129 / BCH candidates against this verifier. The pass condition is
now concrete and cheap: **degree 4 at machine precision.** Only after that does a
paired end-to-end comparison on a DESIGN block make sense, and only after that
does the operating-point re-sweep the cost analysis above calls for.


---

## 2026-08-19 (night) — STAGE 1 / item 5, part 2: **a verified strength-5 orthogonal array, 17x better than the shipped carrier at degree 4**

Same slice, same contract. The contract said a working construction was **not**
promised. One landed anyway, and the verifier built in part 1 is what makes it
safe to believe. 22/22.

### The construction

The extended dual-BCH orthogonal array, `local_sim/carriers.py:bch_oa`.

Let `B` be the 2-error-correcting BCH code of length 255 (designed distance 5).
The codewords of a linear code `C` form an orthogonal array of strength
`d(C_perp) - 1`, so `C = B_perp` gives strength 4. `B_perp` is the classical
`{Tr(a x + b x^3) : a, b in F_256}` code of dimension 16, and taking `x` over all
of `F_256` (not just `F_256*`) plus the all-ones coset gives length **256**,
`2^17 = 131,072` rows, closed under negation.

**Verified combinatorially, not assumed** — every random subset of `t` columns
must see all `2^t` sign patterns equally often:

```text
strength 2: 200/200   strength 3: 200/200   strength 4: 200/200   strength 5: 120/120
```

**Strength 5**, one better than the forum's reported strength-4 OA, because the
extension coset closes the array under negation and buys the odd degrees free.

### What it does to the moment profile

RMS relative deviation from the exact spherical moment, and the ratio to iid:

| carrier | n | k=1 | k=2 | k=3 | **k=4** | k=5 | **k=6** |
|---|---|---|---|---|---|---|---|
| **sobol (SHIPPED)** | 16,384 | 0.027 | 0.176 | 0.611 | **0.510** | 0.840 | **0.764** |
| haar frames | 16,384 | ~0 | ~0 | ~0 | 0.700 | ~0 | 1.089 |
| **bch OA** | 131,072 | ~0 | ~0 | ~0 | **0.0305** | ~0 | **0.138** |

Absolute: the OA reads **8.2e-15 / 9.1e-16 / 3.9e-15** at k=2/3/5 — machine
precision — and **7.77e-04** at k=4 against the shipped carrier's 1.30e-02.

**That is a 17x reduction in degree-4 moment error against the shipped carrier,
and 33x against iid.** At degree 6, where *every* other carrier we have sits at
0.76-1.09x iid, the OA reads **0.138x**. This is precisely the degree-3 ->
degree-4/5 step that part 1 identified as the only remaining input-space prize.

### It is a cube design, not a sphere design — and the distinction is measurable

Every coordinate of every OA point is `+-1/sqrt(d)`, so probed **along a
coordinate axis** the 4th moment is identically `1/d^2 = 1.526e-05` against the
spherical `3/(d(d+2)) = 4.542e-05` — **0.336x, wrong by a factor of three**.
Probed along **random** directions the same points read 7.55e-04 relative. A
generic direction spreads mass across all coordinates and the array's balance
takes over; a coordinate direction is exactly where it does not. So the OA is a
very good *approximate* spherical 4-design in generic position and provably not an
exact one, which is why it must be **randomly rotated in deployment**.

**A check of mine was mis-specified and is recorded as such.** I asserted the
unrotated array would measure worse at degree 4 than the rotated one. It does not
— 7.550e-04 against 7.772e-04, identical. Rotating the point set is the same as
rotating the probe directions, and the probes are already isotropic, so **an
isotropic moment test is rotation-invariant by construction and cannot see axis
alignment**. The rotation matters for the real integrand, not for this
measurement. Same class of error as prediction B4 in the coverage slice: naming a
statistic that cannot measure the thing claimed.

### The binding constraint is now COST, and it has a number

Under the repo's flat-adjusted law (`adjusted = V*c/B`, independent of `N` above
the floor) a bigger point set is free — until it hits the hard `C > B` cliff,
which is scored as a **failure**: multiplier 1.0 *and* zeroed predictions, ~5x,
per P0.1. At the shipped `C/B = 0.199` for `N = 16,384`:

| point set | n | multiple of N | projected C/B | |
|---|---|---|---|---|
| full extended OA (strength 5) | 131,072 | 8.00x | **1.592** | **FAILS the C>B cliff** |
| one row per antipodal pair | 65,536 | 4.00x | 0.796 | affordable |
| Rao lower bound for strength 4 | 32,897 | 2.01x | 0.400 | affordable |

**The full array is not affordable and must not be shipped as-is.** Taking one row
per antipodal pair (65,536) is affordable at `C/B ~ 0.80` and keeps strength 4,
losing only the free odd-degree exactness — which antithetic symmetrisation
restores cheaply, at the ~2x even-degree tax part 1 measured. The Rao bound says a
strength-4 OA on 256 factors needs at least **32,897** runs, so our 131,072 is 4x
the information-theoretic minimum and a tighter construction may exist.

**None of this is an end-to-end result.** Moment exactness is a property of the
point set; what it is worth is a property of the network response, and that needs
a paired comparison on a DESIGN block. The projected `C/B` figures are derived
from the flat-adjusted law and the shipped operating point, not measured.

### Knowledge Delta

- **The missing rung exists and is buildable from standard coding theory.** No
  Kerdock/Z_4 machinery is needed: a dual-BCH OA reaches strength 5
  combinatorially and 17x the shipped carrier's degree-4 accuracy.
- **Verify the array combinatorially before trusting the moment profile.** Column
  balance is a cheap, exact, decisive check, and it caught nothing here only
  because the construction was right.
- **The lane's binding constraint has moved from "can we build a design" to "can
  we afford one".** 131,072 points is 8x the operating point and fails the budget
  cliff; 65,536 is affordable. The operating point must be re-swept for a
  fixed-size carrier rather than inherited.
- **Rotation-invariant tests cannot detect rotation-dependent defects.** Obvious
  in hindsight; it cost a prediction.

### Next

1. Cut the array to 65,536 (one per antipodal pair) plus antithetic, re-verify
   strength, and confirm `C/B ~ 0.80` against a real flopscope count rather than
   the projection above.
2. Paired end-to-end comparison against the shipped carrier on a **DESIGN** block,
   scored with `competition_score` and the studentised interval the coverage study
   selected. **D17 and D19 stay unspent** — this is design work, not a gate.
3. Only then, an operating-point re-sweep.


---

## 2026-08-19 (night) — STAGE 1 / item 5 part 3 CONTRACT: what is the OA actually worth, end to end?  [PRE-REGISTERED]

**Classification: IMPORTANT.** Spends a fresh range of the **D18 DESIGN block**
(408 free). No gate, no single-use block, no vault touch, no estimator change.
**D17 and D19 stay unspent.**

Part 2 measured the dual-BCH OA's *moment* exactness — 17x the shipped carrier at
degree 4. That is a property of the POINT SET. What it is worth is a property of
the NETWORK RESPONSE.

### Design

192 fresh 256x32 networks from D18's free tail, ground truth **2^24** (chosen so
GT noise is ~0.8% of the smallest arm's MSE rather than ~3% at 2^22, because the
OA is expected to push MSE down and GT noise is an additive floor). Three arms,
same networks, common random numbers, **everything except the point set is the
shipped pipeline called through the shipped file's own methods**:

| arm | carrier | N | isolates |
|---|---|---|---|
| A | sobol | 16,384 | the incumbent, shipped operating point |
| **B** | **sobol** | **65,536** | **"more points" — the control the audit demands** |
| C | bch OA | 65,536 | "the design" |

`C vs B` is the design's own contribution; `C vs A` is the deployable comparison;
`A vs B` is a check on the flat-adjusted law. The rotation is folded into the
first-layer weight (`h @ (R^T S) == (h R^T) @ S`), which costs one 256x256 matmul
(~3.3e7 FLOPs) instead of rotating 65,536 x 256 points (~8.6e9, 16% of the
shipped spend).

### Written prediction

| # | prediction | falsified by |
|---|---|---|
| **T1** | At **matched point count** (C vs B) the OA improves raw final-layer MSE by a paired factor in **[1.10x, 1.60x]**. Strictly below the naive 1/(1 - 0.38) = 1.61 ceiling from the deg<=5 harmonic share, because the shipped Sobol carrier already collects part of the low-degree content (part 1: 0.510x iid at k=4) | outside that range |
| **T2** | The **flat-adjusted law holds**: A vs B differ in adjusted score by **less than 5%**, because MSE ~ 1/N and counted cost ~ N cancel. `bias_floor_D16` already measured MSE x N flat (0.0268 vs 0.0271 at N = 16k and 64k) | they differ by more than 5% |
| **T3** | Arm C's measured `C/B` lands in **[0.75, 0.85]**, confirming part 2's projection of 0.796 from the flat law rather than a guess | outside that range |
| **T4** | Folding the rotation into W1 keeps arm C's counted FLOPs within **1%** of arm B's — the design is free at matched N, and the entire cost of the lane is the 4x point count | the arms differ by more than 1% |
| **T5** | The OA is **unbiased**: subtracting the per-network GT-noise floor moves every gain by **less than 3%**, and no arm's GT-subtracted MSE goes negative | a gain moves by more than 3% |

### What would make this a kill

T1 failing low (gain < 1.10x) closes the last input-space lane: it would mean the
network response has little exploitable degree-<=5 content *beyond what Sobol
already takes*, which is the SUBSTITUTION mechanism applying one final time.
That is a real possible outcome and the reason B is in the design.


---

## 2026-08-19 (night) — CONTRACT: the grader's residual, calibrated on FIVE graded submissions instead of one  [PRE-REGISTERED]

**Classification: IMPORTANT** — it changes the instrument, specifically the half
`docs/CURRENT_STATE.md` classifies **E1 — directional only** and about which it
says *"improving the cost evaluator outranks any further score optimization."*
Read-only: no networks decided on, no seed block, no vault touch, no submission.

### The question Samuel asked, and why the answer is not "we have no infrastructure"

We have a complete local replica of the competition: `whest run` **is** the real
`whestbench.scoring.evaluate_estimator` on the organisers' real public split with
1e9-sample truth; `local_sim/competition_score.py` is bitwise-equal to its
identity; `mint_provenance` proves our networks are the grader's bit-for-bit.
Local and graded agree well:

| | local mini (100) | graded 327743 | gap |
|---|---|---|---|
| multiplier `C/B` | 0.216 | 0.19896 | **8.6%** |
| counted FLOPs | 5.330e10 | 5.2898e10 | **0.8%** |

**The only soft term is residual wall time**, and at our operating point it is a
*small* part of `C`: 9.845 ms x 1e11 = 9.85e8 against 5.29e10 counted, i.e.
**1.8% of C** on the grader. That is why the shipped file's local score is only
~8% off.

**It is v2.6 specifically that breaks this**, because it trades counted FLOPs for
operations: 1.7x the ops, so locally residual becomes **20.7% of C**. A quantity
that is 2% of the answer for v2.4.1 is 20% of it for v2.6, which is exactly why
that one decision is unresolved while the shipped file's score is not.

### The unused data

Five graded submissions, each with per-MLP residual telemetry, and all five
estimator files archived:

| submission | file | counted/MLP | graded residual |
|---|---|---|---|
| 323872 v0 | `experiments/estimator_v0_archive.py` | 3.269e9 | 6.743 ms |
| 324075 v1.0 | `estimator_v1.py` | 4.082e10 | 4.491 ms |
| 324094 v1.1 | `estimator_v1_1.py` | 6.998e10 | 10.527 ms |
| 324420 v2.4 | `estimator_v2_4.py` | 5.291e10 | 9.845 ms |
| 327743 v2.4.1 | `estimator_v2_4_1.py` | 5.290e10 | 10.022 ms |

**We have been using ONE of these as the calibration point** — "7.0 us/op from
327743" — and treating it as a single ratio through the origin. Five points at
different operation counts fit a **line**, which separates fixed overhead from
marginal per-operation cost and is a genuinely predictive model rather than a
ratio.

`local_sim/residual_model.py` runs each archived version locally under a real
`flopscope.BudgetContext`, recording counted FLOPs **and operation count**, then
regresses the *graded* residual on the *local* operation count.

**The validation that makes this trustworthy**: counted FLOPs are deterministic,
so local counted must reproduce graded counted for all five. If it does, we are
provably running the same programs the grader ran, and the operation counts are
trustworthy by the same argument.

### Written prediction

| # | prediction | falsified by |
|---|---|---|
| **R1** | Local counted FLOPs reproduce graded counted FLOPs for **all five** versions to within **2%** (allowing for pruning varying across network draws) | any version off by more than 2% |
| **R2** | Graded residual is **better explained by operation count than by counted FLOPs** — higher R^2 — because forum 18101 and this repo's own evidence say residual tracks the number of operations and the size of their temporaries, not FLOPs. v1.0 is the discriminating case: it has **12x** v0's counted FLOPs but **less** residual | counted FLOPs explain it at least as well |
| **R3** | The fitted line has a **positive intercept**: a fixed per-MLP overhead independent of the estimator, from process/IPC/telemetry. I expect it to be **1-5 ms**, i.e. a large share of the 4.5-10.5 ms observed range | the intercept is negative or above 5 ms |
| **R4** | The marginal cost per operation is therefore **below** the 7.0 us/op standing number, because that number is a ratio through the origin and absorbs the intercept. I predict **1-5 us/op** | outside that range |
| **R5** | Applying the fitted model to v2.6.1's measured local op count gives a projected graded residual, and a projected `V.c` gain against v2.4.1, that lands **inside** the 1.068-1.096x band the three-model rule already brackets — i.e. the calibration **tightens** the estimate rather than moving it outside the existing bracket | it lands outside that band |

**If R5 fails the calibration is the more trustworthy number**, not the bracket —
five graded observations beat one ratio — and that would be a finding, not an error.

### Standing limits, unchanged by any of this

n = 5 is a small fit and every point is a **different program**, so the model
confounds "more operations" with "different operations". It cannot become E3. The
honest ceiling here is a better-supported E2 with a stated interval, which is
still a large upgrade on a single ratio through the origin.


---

## 2026-08-19 (night) — EVIDENCE PACKET: the grader's residual is a LINE, and the cost half is no longer E1

Contract: the entry immediately above. **6/7, one prediction falsified.**
Artifacts: `experiments/2026-08-19_carriers/residual_model.{out,json}`.
Read-only: no networks decided on, no seed block, no vault touch, no submission.

### R1 — we are provably running the graded programs

Local counted FLOPs against graded counted FLOPs, 24 throwaway networks per
version:

| ver | local counted | graded counted | gap | ops | graded residual | n sigma |
|---|---|---|---|---|---|---|
| v0 | 3.2670e9 | 3.2691e9 | -0.06% | 841 | 6.743 ms | 0.0 |
| v1.0 | 4.0817e10 | 4.0821e10 | -0.01% | 329 | 4.491 ms | 0.0 |
| v1.1 | 6.9975e10 | 6.9979e10 | -0.01% | 1134 | 10.527 ms | 0.0 |
| v2.4 | 5.2954e10 | 5.2909e10 | +0.09% | 1339 | 9.845 ms | 0.1 |
| v2.4.1 | 5.2956e10 | 5.2898e10 | +0.11% | 1352 | 10.022 ms | 0.2 |

**Confirmed at <= 0.2 sigma** — but only after a mis-specification of my own.
The first run used **6** seeds and a flat 2% tolerance, and read 2.51% / 2.53%
for v2.4 and v2.4.1, i.e. FAILED. The gap is not a program mismatch: v2.4+
**prunes**, and the alive fraction is network-dependent, so counted FLOPs have a
**per-network sd of 3.22%** (measured over 40 networks) against **exactly 0.00%**
for v1.1, which does not prune. At n = 6 the standard error is 1.3% and a 2.5%
gap is 2 SE — noise. **A tolerance that ignores the estimator's own sampling
error is not a test**, and this is the third time tonight that class of error has
bitten (prediction B4, the rotation check, this).

### R2 — residual tracks OPERATIONS, not FLOPs, and now we have shown it ourselves

| model | intercept | slope | R^2 | leave-one-out MAE |
|---|---|---|---|---|
| `residual ~ a + b * OPS` | **+2.551 ms** | **5.780 us/op** | **0.898** | **0.829 ms** |
| `residual ~ a + b * COUNTED` | +5.471 ms | 0.065 ps/FLOP | 0.387 | 3.179 ms |

**The discriminating case is v1.0: 12.5x v0's counted FLOPs and LESS residual**
(4.491 ms against 6.743 ms). No monotone function of FLOPs can produce that; a
function of operation count can, because v1.0 runs 329 operations against v0's
841. Forum 18101 said this and `docs/GOTCHAS.md` records it; **this is the first
time it has been demonstrated on our own graded telemetry.**

### R3 — a fixed per-MLP overhead of 2.55 ms, which nothing we write can avoid

The intercept is **+2.551 ms**, inside the predicted 1-5 ms. That is **25% of
v2.4.1's entire 10.02 ms residual**, and at lambda = 1e11 it is **2.55e8
FLOP-equivalents charged to every MLP before our first operation**. A ratio
through the origin cannot see it and therefore mis-prices every candidate whose
operation count differs from the one it was fitted on.

### R4 — FALSIFIED

I predicted the marginal cost would fall in 1-5 us/op once the intercept absorbed
the fixed part. It is **5.780 us/op** — below the standing 7.0, but by 17%, not
the 30-80% I implied. The intercept is real but smaller relative to the slope than
I guessed. **The standing "7.0 us/op" is not badly wrong; it is a ratio through
the origin that over-charges low-op candidates and under-charges high-op ones**,
and at v2.4.1's 1,352 ops the two models agree to 6% (9.40 ms vs 10.37 ms fitted).

### R5 — and the answer for v2.6.1

| candidate | model | C/B control | C/B candidate | cost ratio | net V.c (MSE 0.9942x) |
|---|---|---|---|---|---|
| v2.6.1 | **fitted line, 5 points** | 0.1983 | **0.1802** | **1.1004x** | **1.0940x** |
| v2.6.1 | ratio through origin | 0.1979 | 0.1803 | 1.0979x | 1.0915x |

Projected graded residual for v2.6.1: **16.251 ms** (fitted) against v2.4.1's
observed 10.022 ms. Counted FLOPs fall 10.5%, residual rises 62%, and **because
residual is only ~2-3% of C on the grader the counted saving wins**.

**CONFIRMED, inside the pre-registered 1.068-1.096x bracket, near its top.**

### Why this changes the classification of the cost half

`docs/CURRENT_STATE.md` calls the cost half **E1 — directional only**, because
"uninstrumented wall time is not measurable on this machine". True, and
irrelevant: **we never needed to measure it here.** The grader measured it for us
five times and the telemetry has been sitting in `experiments/run_reports/`.

Residual's share of `C` **on the grader**:

| | counted | lambda x residual | residual share of C |
|---|---|---|---|
| v2.4.1, observed | 5.2956e10 | 1.002e9 | **1.9%** |
| v2.6.1, projected | 4.7391e10 | 1.625e9 | **3.3%** |

The fit's leave-one-out error is **0.829 ms**. Even allowing double that for
extrapolation, +-1.7 ms on v2.6.1's C of 4.90e10 is **+-0.35% on C**. So the cost
ratio 1.1004x is known to well under 1%, against the old three-model bracket of
**[0.9529, 1.1147]** — a 17-point spread whose sign flipped.

**The local residual measurement was never the right instrument.** It reads 124.55
ms/MLP for v2.6 where the grader is projected at 16.25 ms — 7.7x — and that ratio
is the entire reason the sign flipped. Deleting the local number and using graded
telemetry removes the problem rather than modelling it.

### The honest limits — the new load-bearing assumption

- **n = 5, and every point is a different program.** The model confounds "more
  operations" with "different operations". It cannot be E3.
- **v2.6.1 sits at 2,370 ops; the fit is calibrated on 329-1,352. That is
  EXTRAPOLATION by 1.75x**, and it is now the load-bearing assumption. It is much
  weaker than the old one — which was that a *local* machine's residual ratio
  transfers to different hardware — but it is an assumption, and the script prints
  it every run rather than burying it.
- **Residual also tracks the SIZE of temporaries, not only their count** (forum
  18101). v2.6's Strassen blocks are half-size, so its true per-op price may be
  *below* the fitted line, which would make 1.1004x conservative.
- v1.1 is the worst-fit point at **-1.42 ms**; the model is not exact.

### Knowledge Delta

- **`residual_graded ~ 2.551 ms + 5.780 us/op`**, R^2 = 0.898, LOO 0.829 ms,
  calibrated on five graded submissions. Replaces "7.0 us/op" as a ratio through
  the origin.
- **2.551 ms of every MLP's residual is fixed overhead** — 25% of v2.4.1's, and
  2.55e8 FLOP-equivalents charged before any of our code runs.
- **Residual is ~2-3% of C on the grader**, so the cost half is *far* better
  determined than the local measurement suggested. The shipped file's local score
  is only ~8% off graded, and almost all of that gap is the residual term.
- **v2.6.1 is a projected 1.0940x on V.c**, i.e. **2.6314e-7 -> ~2.406e-7**,
  known to well under 1% on the cost side rather than to a 17-point bracket.
- **We had the data for a month and used one fifth of it.** The single-ratio
  anchor was chosen because it came from the most recent submission, not because
  the others were unusable.

### Proposed reclassification, NOT enacted

`docs/CURRENT_STATE.md` § *Evaluator trust* should move the cost half from **E1 —
directional only** to **E2 — quantitative with a stated interval**, on the
grounds above, with the extrapolation limit named. That is a change to a durable
classification that gates autonomy rules ("no autonomous loop while the evaluator
is E1"), so it is **STRATEGIC and needs Samuel's decision**, not an agent's.


---

### 2026-08-19 (night) — STAGE 1 item 5: **the OA only works WHOLE**, and that is the lane's real constraint

Recorded before the end-to-end arms finish, because it is a finding in its own
right and it changes the deployable question.

Part 2 projected that the array could be halved to 65,536 rows — one per
antipodal pair — for an affordable `C/B ~ 0.80`. **That is wrong three times
over**, and each way was caught by a different cheap check.

**1. The obvious halving is not an array at all.** One row per antipodal pair
returns the linear code `{Tr(ax + bx^3)}` itself, and that code has a **constant
column**: coordinate `x = 0` carries `Tr(a*0 + b*0^3) = 0` for every `(a, b)`.
Centroid norm **1.000000** — the point set sits off-centre by a full unit vector
and explores a 255-dimensional slice. Run end to end it measured **41x worse than
Sobol** (arm C 6.53e-05 against arm A 1.58e-06 at 8 networks) before the run was
killed and the cause found.

**2. Puncturing to `F_256*` and appending overall parity does not fix it.** Built
and measured: still one constant column, still centroid norm 1.0. The reason is
arithmetic — the parity is `Tr(a * sum_{x in F*} x) + Tr(b * sum_{x in F*} x^3)`,
and **both sums vanish**: `sum x = 0` for m >= 2, and the cubes form the index-3
subgroup, whose elements are the roots of `X^85 - 1` and therefore sum to zero.
So parity is identically 0.

**3. A RANDOM half is no design at all.** Measured moment profiles at matched
n = 65,536, projected to the sphere:

| point set | centroid | k=2 | k=4 | k=6 |
|---|---|---|---|---|
| iid | 3.86e-03 | 5.52e-03 | 1.29e-02 | 2.62e-02 |
| **sobol** | 6.19e-05 | **6.70e-04** | **6.78e-03** | 2.08e-02 |
| **OA, random half** | 2.71e-03 | **4.06e-03** | **9.31e-03** | 1.97e-02 |
| **OA, full 131,072** | **0.00e+00** | **8.54e-15** | **7.55e-04** | 7.15e-03 |

**A random half of the array is WORSE than Sobol at both k=2 and k=4**, and
barely better than iid. The design's entire value is its exact balance, and
subsampling destroys it completely. There is no partial credit.

### What this does to the lane

**The array is worth a great deal and costs 8x the operating point, and those are
not separable.** 131,072 points at the shipped per-point cost projects
`C/B ~ 1.59`, which **exceeds the budget** — scored as a failure, multiplier 1.0
*and* zeroed predictions. So the deployable question is no longer "is the design
better" but **"can 131,072 points be made to cost what 16,384 do now"**.

Three routes, in the order they look affordable:

1. **A smaller genuine strength-4 OA.** The Rao bound for strength 4 on 256
   two-level factors is **32,897 runs**, so ours is 4x the information-theoretic
   minimum. A tighter construction would land at `C/B ~ 0.40`.
2. **An FWHT first layer.** The points are `+-1` vectors, so `h @ W1` has
   Hadamard structure and need not be a dense matmul. **Both frontier write-ups
   name FWHT** (Team Puffi and the BCH/OA author) — and this is the first
   explanation this project has for *why they need it*: not as an optimisation,
   but because the carrier is otherwise unaffordable.
3. More aggressive pruning, which trades against accuracy.

### Knowledge Delta

- **Check a carrier's CENTROID before anything else.** One line, catches a
  degenerate point set instantly, and would have saved the 41x run. Now the first
  thing `test_carriers.py` reports.
- **Orthogonal arrays do not degrade gracefully.** Unlike Sobol, where any prefix
  is still low-discrepancy, an OA subsampled is an OA destroyed.
- **The frontier teams' FWHT is not an optimisation, it is a precondition.** The
  carrier that buys degree-4/5 exactness is 8x the point count, and a dense first
  layer cannot pay for it.


---

## 2026-08-19 (night) — STAGE 1 item 5 EVIDENCE PACKET: **the design is worth 1.48x, and we cannot afford it**

Contract T1-T5, above. **T1, T4, T5 confirmed; T2 and T3 falsified.** 48 fresh D18
networks, GT 2^24, sigma_gt^2 = 3.485e-09. Artifacts:
`experiments/2026-08-19_carriers/oa_{arms,score}.{json,out}`.

### The three arms

| arm | raw MSE | GT-subtracted | counted/MLP | C/B | adjusted |
|---|---|---|---|---|---|
| A sobol N=16,384 | 2.14948e-06 | 2.14599e-06 | 5.1923e10 | **0.1909** | 4.06009e-07 |
| B sobol N=131,072 | 2.53683e-07 | 2.50198e-07 | 3.6741e11 | **1.3508** | 3.38893e-07 |
| C bchOA N=131,072 | 1.71103e-07 | 1.67618e-07 | 3.6744e11 | **1.3509** | 2.29366e-07 |

### T1 — the design's own contribution, at matched point count AND matched cost

`B -> C`, paired on the same 48 networks, costs identical to 4 significant
figures (3.6741e11 vs 3.6744e11):

```text
raw-MSE gain                1.4826x          GT-subtracted 1.4927x
adjusted gain               1.4775x
arithmetic difference       +1.095273e-07   (+32.319%)
studentised 95% CI          [+3.7953e-08, +1.8110e-07]     <- excludes zero
relative 95% CI             [+14.396%, +45.923%]
geometric (secondary)       1.4321x  [1.1714, 1.7507]
better on                   33/48    sign-test p = 0.0133
```

**CONFIRMED, inside the pre-registered [1.10x, 1.60x].** And it lands within 1.5%
of the **~1.46x external ceiling** `docs/CURRENT_STATE.md` derived independently
from the layer-32 harmonic spectrum — two unrelated routes to the same number.

**This is the largest measured effect in the project.** Every other input-space
device died between 0.98x and 1.09x.

### T4 — the rotation fold is free, as designed

Counted FLOPs differ by **0.008%** between arms B and C (3.6741e11 vs 3.6744e11).
Folding the random rotation into the first-layer weight instead of rotating
131,072 x 256 points costs one 256x256 matmul. **The design is free at matched N;
the entire cost of the lane is the point count.**

### T5 — unbiased, as the homogeneity argument requires

Subtracting the per-network GT-noise floor moves the gain from 1.4826x to
1.4927x — **0.68%**, inside the predicted 3%. No arm's GT-subtracted MSE went
negative.

### T2 — FALSIFIED, and the decomposition is worth keeping

I predicted `A -> B` would differ by less than 5% under the flat-adjusted law
(`adjusted = V*c/B`, independent of N). Measured **1.1980x** — a 20% violation.
It decomposes exactly:

| factor | value | why |
|---|---|---|
| `MSE x N` ratio | **1.0591** | QMC beats the 1/N law by 5.9% at 8x the points |
| counted-FLOPs-per-point ratio | **1.1306** | the fixed closure cost amortises over more points |
| product | **1.1974** | vs the measured **1.1980** |

**The flat law is an approximation that breaks in the candidate's favour**, and
both terms push the same way. This also confirms D03's finding (N = 65,536 read
1.0408x better) with a mechanism rather than a number.

### T3 — FALSIFIED, because it was written for a point set that does not exist

I predicted arm C's `C/B` would land in [0.75, 0.85]. That was written for the
**halved** 65,536-row array, which turned out to be degenerate (constant column,
centroid norm 1.0). The whole array measures **C/B = 1.3509**.

### The blocker, stated precisely

**48 of 48 MLPs are over budget in both arms B and C.** On the real grader
`C > B` is a **failure**: multiplier forced to 1.0 *and* predictions zeroed. So
neither arm is deployable, and the 1.4775x is the value of a carrier we cannot
currently run.

**And the obvious fix does not work.** I speculated earlier today that the
frontier teams' FWHT first layer is what makes this affordable. Measured: the
first layer is `131,072 x 256 x 256 x 2 = 1.718e10` of a total **3.674e11** —
**4.7%**. Even a free first layer leaves `C/B = 1.29`. **The cost is the 32-layer
chain, not the first layer.** That speculation is withdrawn.

The cost is `N x depth x width^2`, so the only levers are:

1. **A smaller genuine strength-4 array.** The Rao bound for strength 4 on 256
   two-level factors is **32,897 runs**; ours is 131,072, i.e. **4x** the
   information-theoretic minimum. At 32,768 points `C/B ~ 0.34` and the lane is
   comfortably affordable. **This is the highest-value open construction problem
   in the project** — and it is pure combinatorics, needing no networks, no seed
   block and no grader.
2. Cheaper propagation across all 32 layers — Strassen (v2.6.1 already, 10.5%),
   more aggressive pruning (trades accuracy), or structured weights.
3. A shallower control layer, which is item 6 of the plan.

### What it would be worth if solved

At matched cost the design is **1.4775x**. Applied to the graded number of record:

    2.6314e-07 / 1.4775  =  1.781e-07

against the best *published and explained* honest score of 9.10e-8. That would
close roughly **half** the remaining gap to the published frontier in one step.
**Conditional on a construction that does not exist yet**, and not a projection of
anything currently runnable.

### Knowledge Delta

- **A true degree-4/5 design is worth 1.4775x [+14.4%, +45.9%] against the shipped
  stack at matched cost** — measured, paired, on fresh networks, with the interval
  the coverage study selected. Within 1.5% of the independently derived ~1.46x
  ceiling.
- **The design is free; the point count is the cost.** Matched-N arms differ by
  0.008% in counted FLOPs.
- **FWHT is not the answer to affordability** — the first layer is 4.7% of the
  chain. Withdrawn.
- **The flat-adjusted law breaks by 20% in the candidate's favour at 8x N**, and
  splits cleanly into QMC superconvergence (1.059) and closure amortisation
  (1.131).
- **The open problem is now purely combinatorial**: a strength-4 orthogonal array
  on 256 two-level factors with ~32,768 runs. Rao says 32,897 is the floor; we are
  at 4x it.

### Out of scope, unchanged

No gate. No single-use block — **D17 and D19 remain unspent**; this slice used
D18[30,806,018,192..383], a DESIGN block. No vault touch. `estimator.py` was run,
never edited. Nothing submitted.


---

## 2026-08-19 (night) — STAGE 1 item 5, part 4: **131,072 is the linear floor at width 256, and the 256th coordinate is what costs us the factor of two**

**Classification: ROUTINE.** Pure combinatorics on 16-bit vectors. No networks,
no seed block, no vault, no estimator change. It exists to stop anyone spending
days hunting for a smaller array.

### The question

The measured design is worth **1.4775x** and costs `C/B = 1.3509` — over the
budget cliff. Part 3 named the obvious escape: our array is **131,072 rows**
against a Rao bound of **32,897**, so it is ~4x the information-theoretic
minimum. Is there a smaller one?

### The answer, for linear constructions: no, and here is why

An orthogonal array from a linear code `C` has strength `d(C_perp) - 1`. Strength
4 needs `d(C_perp) >= 5`, which in terms of the generator matrix `G` says: **no
set of <= 4 columns of `G` may sum to zero.** To halve to 65,536 rows we need
`dim C = 16`, i.e. `G` is `16 x 256` — 256 vectors in `F_2^16` with that property.

The classical construction gives exactly 255 of them: the BCH parity-check
columns `h_x = (x, x^3)` for `x` in `F_256*`. Verified directly:

```text
BCH parity-check columns: 255   distinct: 255
pairwise XORs: 32385, distinct 32385  -> Sidon: True
  any pairwise XOR equal to a column (a 3-subset summing to 0)? False
```

So `{h_x}` is a **Sidon set** in `F_2^16` — all 32,385 pairwise sums distinct,
which is precisely "no 4 columns sum to zero" — and no 3 sum to zero either.
**At length 255 a strength-4 array with 65,536 rows exists and is affordable.**

The 256th column is the problem. Appending `v` creates a weight-4 dual codeword
exactly when `v = h_a + h_b + h_c` for some triple. Enumerated:

```text
distinct 3-subset sums: 65535 of 65535 nonzero vectors in F_2^16 (100.000%)
candidate 256th columns that create NO weight-4 dual word: 0
```

**Every single nonzero vector in `F_2^16` is a 3-subset sum.** There are
`C(255,3) = 2,731,135` triples landing in 65,535 values — an average multiplicity
of 42 — so the covering is not close, it is total. **No 256th column exists.**

Consistent with the Hamming bound, which caps a length-256 distance-5 code at
dimension 240 (redundancy >= 16), and with the extended BCH `[256, 239, 6]` being
the best known — one dimension short of what a 65,536-row array would need.

### The uncomfortable corollary

**At width 255 this carrier would be affordable.** 65,536 rows gives
`C/B ~ 0.68`, comfortably inside budget, with the same strength 4. The
competition's width being `2^8` rather than `2^8 - 1` costs us **exactly a factor
of two** on this lane. That is not a fixable engineering problem; it is a property
of the numbers.

*(Splitting the difference does not work: the design must integrate all 256
directions, and a 255-coordinate array embedded in `R^256` with a zero or a
Gaussian in the last slot leaves the point set off the sphere and breaks the
positive-homogeneity identity the whole carrier rests on.)*

### What is left, priced

| floor | value |
|---|---|
| Rao bound, 2-level strength-4 OA on 256 factors | **32,897** runs |
| Delsarte-Goethals-Seidel, spherical 4-design in `R^256` | **33,152** points |
| our linear array | **131,072** — **3.95x** the floor |

A **non-linear** array or a genuine spherical 4-design near 33k points would give
`C/B ~ 0.34` and settle the lane outright. Both are open construction problems;
tight designs at these parameters are rare and this is a research question, not an
engineering one.

The cost levers this repository already owns, applied to `C/B = 1.3509`:

| lever | factor | resulting C/B |
|---|---|---|
| depth-2 Strassen (LOG 2026-08-19: 1.267x counted) | 1.267 | 1.0662 |
| FWHT first layer (measured 4.7% of the chain) | 1.047 | 1.2903 |
| **both together** | 1.327 | **1.0184 — still over** |

**Both known levers together still leave it over budget**, and "just over" is the
worst place to be: `C > B` is scored as a *failure*, multiplier 1.0 **and** zeroed
predictions. A carrier that lands at 1.02 and needs to land at 0.95 is not a
carrier, it is a coin flip on the grader's hardware.

### A correction to my own reasoning, made before it reached a document

While working this out I claimed a shallow analytic centre (item 6) would make the
carrier affordable by cutting the sampled chain from 32 layers to 23. **That is
wrong.** A control variate reduces variance at unchanged cost — the final layer
still requires propagating every sample through all 32 layers. Starting the chain
*at* layer `c` would need the full layer-`c` distribution, not its mean, and that
is the Moonshot's hard problem, not a cheap win. **Item 6 does not pay for item
5.**

### Knowledge Delta

- **131,072 is the linear-code floor for a strength-4 OA on 256 binary factors.**
  Proven by exhaustive enumeration of 3-subset sums, not argued. Stop looking for
  a smaller linear one.
- **The BCH Sidon set is complete at 255 and cannot be extended.** All 65,535
  nonzero vectors of `F_2^16` are covered as 3-subset sums.
- **Width 256 vs 255 is worth exactly 2x on this lane.**
- **Both known cost levers together leave `C/B = 1.0184`** — over the cliff, and
  too close to it to ship even if it were under.
- **A control variate does not reduce cost.** Recorded because I nearly wrote the
  opposite.

### Where that leaves the lane

**The design is measured, large (1.4775x), and blocked on a construction that is a
genuine open problem.** It is not blocked on engineering, on the Phase-2 shape, or
on a decision. The three routes, honestly ranked:

1. A **non-linear strength-4 OA** or a spherical 4-design near 33k points. Would
   settle it outright. Research.
2. A **weighted subspace cubature**: the estimator already rotates so that `h`'s
   first 32 coordinates are the top-32 eigendirections of `J J^T`
   (`_alignment_basis` returns them in descending eigenvalue order, and the
   constants audit found rank 256 only 1.0108x better at 0.7 sigma). A degree-5
   Gaussian cubature in **32** dimensions needs `O(d^2) ~ 2,049` nodes, not 131,072.
   The obstruction is that such rules carry **unequal weights**, so the estimator's
   plain `h.mean(axis=0)` becomes a weighted mean — a small change, and
   flopscope-cheap. **This is the most promising affordable route and nothing in
   the repository has tried it.**
3. Accept the carrier only if a future Phase-2 budget is larger relative to the
   shape.


---

## 2026-08-19 (night) — STAGE 1 item 5, route 2 CONTRACT: the weighted subspace cubature  [PRE-REGISTERED]

**Classification: IMPORTANT.** Measurement only — no networks, no seed block, no
vault, no estimator change. It tests whether the design can be confined to the
32 directions that matter, which is the one route to affordability that is
engineering rather than research.

### The idea, and the arithmetic that may already have killed it

`_alignment_basis` returns the top-32 eigenvectors of `J J^T` in descending order,
so `h`'s first 32 coordinates are the important ones. A degree-5 cubature in **32**
dimensions needs `O(d^2) ~ 2,049` nodes rather than 131,072 — the lane becomes
free. That was the plan.

**Step 0 killed the cheap version of it.** The 32-factor array has the identical
obstruction as the 256-factor one, measured:

```text
m=5  F_2^10  columns=31  Sidon=True  3-subset sums cover 1023/1023 (100.00%)  -> free columns: 0
m=8  F_2^16  columns=255 Sidon=True  3-subset sums cover 65535/65535 (100.00%) -> free columns: 0
```

So the minimal linear strength-4 array on 32 factors is 2,048 rows, not 1,024 —
still tiny, so this is not the blocker. The blocker may be deeper.

### The arithmetic that predicts failure

Our carrier's points are `+-1/16` vectors. For a strength-4 array all cross terms
vanish and, for a unit direction `a`,

    E_array[(a.u)^4]  =  (3 - 2S) / 65536,      S = sum_i a_i^4

against the spherical `3/(256 * 258)`. The ratio is `(3 - 2S) * 258 / (3 * 256)`,
which equals **exactly 1 when `S = 3/258`** — and `3/258` is precisely the expected
`S` for a uniformly random unit vector **in R^256**. *That* is why the full array
measures 7.55e-04: it is exact in expectation over spread directions, and the
residual is only the fluctuation of `S`.

Confine `a` to a 32-dimensional subspace and `S` rises to `~3/34 = 0.0882`:

    ratio = (3 - 0.1765) * 0.3359375 = 0.94847      ->  5.2% LOW

**Sobol's measured degree-4 error is 1.30e-02 = 1.3%.** So a cube-based design
probed along confined directions should be **four times worse than the carrier it
is meant to replace**.

### Written prediction

| # | prediction | falsified by |
|---|---|---|
| **U1** | A hybrid array — strength-4 on 32 columns, random `+-1` on the other 224 — probed along directions confined to those 32 columns reads a degree-4 relative error near **5%**, i.e. **worse than Sobol's 1.3%** | it beats Sobol |
| **U2** | **THE DECISIVE ONE.** The **full 256-column strength-5 array**, probed along the *same confined* directions, reads **the same ~5%** — because the error is a property of the CONFINEMENT, not of the array. If this holds, no array of any size fixes it, and the subspace route dies for a general reason rather than a constructional one | the full array stays near 7.55e-04 on confined directions |
| **U3** | The full array probed along **spread** directions still reads ~7.55e-04, reproducing part 2 | it does not reproduce |
| **U4** | Consequently the design's value comes from the rotation spreading the response across **all 256** columns, and strength-4 on all 256 is therefore required — **the array cannot be shrunk by exploiting the rank-32 structure** | U2 fails |

**If U2 holds, route 2 is dead** and the lane's only remaining routes are a
non-linear/spherical design near the 33k floor, or a larger Phase-2 budget.
**If U2 fails**, the confinement is harmless and a 2,048-row array in the top-32
block delivers the 1.4775x at the shipped operating point.


---

## 2026-08-19 (night) — ROUTE 2 EVIDENCE PACKET: **the subspace cubature is dead, and the reason closes the lane**

Contract U1-U4, above. **All four confirmed.** 27/27 in `test_carriers.py`.
Artifact: `experiments/2026-08-19_carriers/design_strength.out`. No networks, no
seed block, no vault, no estimator change.

### The measurement

Degree-4 relative moment error, probed along unit directions confined to the
first `k` coordinates **of the array's own frame**:

| point set | conf 256 | conf 128 | conf 64 | **conf 32** |
|---|---|---|---|---|
| sobol (SHIPPED) | 1.283e-02 | 1.292e-02 | 1.206e-02 | **1.164e-02** |
| **bch OA, UNROTATED** | **7.553e-04** | 8.038e-03 | 2.321e-02 | **5.378e-02** |
| **bch OA, rotated** | 7.919e-04 | 7.630e-04 | 7.383e-04 | **7.676e-04** |

against the analytic prediction `(3 - 2S) * 258 / (3*256)` with `S = 3/(k+2)`:

| k | predicted low by | measured (unrotated) |
|---|---|---|
| 256 | 0.00% | 7.553e-04 |
| 128 | 0.77% | 8.038e-03 |
| 64 | 2.27% | 2.321e-02 |
| **32** | **5.15%** | **5.378e-02** |

**The prediction and the measurement agree at every rung.**

### What it means, and why it kills the route

The subspace route proposed aligning a small array's balanced columns with the
response's rank-32 subspace, buying the same exactness at 1/64th the point count.
The measurement says that trade is not available:

- **Unrotated, on directions confined to 32 dimensions, the array reads
  5.378e-02 — 4.6x WORSE than Sobol's 1.164e-02.** So an array aligned with the
  response is worse than the carrier it would replace.
- **Rotated, the same array reads 7.676e-04 on the same confined directions** —
  flat across every confinement level, 7.6e-04 to 7.9e-04. **A random rotation
  turns any confined direction into a spread one.**

So the rotation is not an implementation detail. **The rotation IS the mechanism**,
and it requires balance on all 256 columns, because a rotated direction touches
all 256 and every 4-subset of columns must cancel. **You cannot have the cheap
array and the rotation at the same time.**

The arithmetic behind it, which is the durable part:

    E_array[(a.u)^4] = (3 - 2S)/65536,   S = sum_i a_i^4

against the spherical `3/(256*258)`. The ratio is exactly 1 when `S = 3/258` —
**precisely the expected `S` for a uniformly random unit vector in R^256.** The
`+-1` cube array is a degree-4 design *in expectation over spread directions* and
nothing else; confinement raises `S` and the exactness degrades as measured.

### Verdicts

| # | verdict | measured |
|---|---|---|
| **U1** | **CONFIRMED** | the hybrid (strength-4 on 32 columns) reads 5.381e-02 confined — worse than Sobol |
| **U2** | **CONFIRMED — the decisive one** | the **full 131,072-row array**, unrotated, reads **the same 5.378e-02** on confined directions. The error is a property of the CONFINEMENT, not the array. **No array of any size fixes it** |
| **U3** | **CONFIRMED** | 7.553e-04 on spread directions, reproducing part 2's 7.55e-04 |
| **U4** | **CONFIRMED** | strength-4 on all 256 columns is required; the array cannot be shrunk by exploiting the rank-32 structure |

### Knowledge Delta

- **Route 2 is DEAD, for a general reason rather than a constructional one.**
  Aligning a design with the response's subspace costs more than the alignment
  saves. This is not a failure to find the right array; no array works.
- **The random rotation is the mechanism, not a detail.** It converts every
  response direction into a spread one, which is precisely the regime where a
  `+-1` cube array is exact. Remove the rotation and the design is worse than
  Sobol; keep it and you need all 256 columns balanced.
- **A `+-1` cube array is a degree-4 design only in expectation over spread
  directions.** `S = sum a_i^4` is the whole story, and `S = 3/(d+2)` is the
  break-even.
- The 32-factor array has the identical Sidon obstruction as the 256-factor one
  (3-subset sums cover 1023/1023 of `F_2^10`), so even the small array costs 2,048
  rows rather than 1,024. Moot now, but recorded so nobody re-derives it.

### Where the lane stands, complete

The design is **worth 1.4775x** [+14.4%, +45.9%], measured end to end. It needs
**131,072 points**, `C/B = 1.3509`, over the failure cliff. And now:

| route | status |
|---|---|
| smaller **linear** array | **CLOSED** — 131,072 is the floor, proven by exhaustive 3-subset enumeration |
| **subspace** cubature | **CLOSED** — this slice; the rotation is the mechanism and it needs all 256 columns |
| depth-2 Strassen + FWHT together | **INSUFFICIENT** — leaves `C/B = 1.0184`, still over |
| non-linear array / spherical 4-design near the 33,152 floor | **OPEN — research** |
| a Phase-2 budget larger relative to the shape | **OPEN — not ours to choose** |

**Two of the four routes are now closed with proofs rather than opinions**, which
is the useful outcome: nobody should spend time on either again. What remains is
a genuine open problem in design theory, and it is the only thing standing between
this project and a measured 1.48x.


---

# 2026-08-19 (night) — SESSION CLOSE

**Everything is committed and pushed.** Working tree clean at `33c0dc9`. Seven
commits. `docs/CURRENT_STATE.md` § *Next actions* is rewritten for a cold start
and is the only thing the next session needs to read after `session-start`.

## What this session did

| slice | class | outcome |
|---|---|---|
| **P0.1** canonical scorer | IMPORTANT | `local_sim/competition_score.py`, bitwise-equal to whestbench 0.14.0. **71/71**. Skeptic **PROVISIONAL** |
| **P0.4** seed ledger | IMPORTANT | `local_sim/seed_ledger.py` + `experiments/seed_ledger.jsonl`, preflight step 6. **40/40** negative tests |
| interval coverage | IMPORTANT | `local_sim/coverage_lab.py`, 192 fresh networks (D18) |
| **P0.3** odd-width repair | **STRATEGIC** | `estimator.py` → **v2.6.1**. **27/27**. Skeptic **ACCEPT** |
| **P0.2** vault repair | IMPORTANT | candidate-aware, `--dry-run` deleted, touch bound to a sha256 |
| item 4 gate reconstruction | ROUTINE | reproduces both CONFIRM artifacts bitwise |
| residual calibration | IMPORTANT | five graded submissions, not one |
| **Stage 1 item 5** carrier | IMPORTANT | design **measured at 1.4775×**, and **unaffordable** |
| route 1 / route 2 closure | ROUTINE / IMPORTANT | two affordability routes **closed with proofs** |

**Fourteen pre-registered predictions were falsified.** Five of them were the
findings; the rest were mis-specified tolerances or statistics that could not
measure what they claimed. The recurring error — bitten **four** times — is *a
tolerance that ignores the estimator's own sampling error*. Worth remembering.

## The seven durable facts

1. **The competition objective is the arithmetic mean over the FIFTY public MLPs**,
   reproducible offline **bitwise** from a graded report. A failure is priced at
   multiplier **1.0**, not the 0.1 floor — exactly `1/mult_i`, 4.6–5.4× here.
2. **Arithmetic and geometric estimands select OPPOSITE winners on our own graded
   data** (v2.4 → v2.4.1: 1.0334× vs 0.9666×, same 50 rows).
3. **Use the studentised t or basic bootstrap. Never BCa at n ≈ 50** — measured
   false-promotion rate 9.1–11.5% against a nominal 5%, because CRN makes the
   *paired difference* nearly symmetric even though the scores are heavy-tailed.
4. **The grader's residual is a LINE**: `2.551 ms + 5.780 µs/op`, R² = 0.898,
   fitted on five graded submissions. **2.551 ms is fixed overhead.** Residual is
   only 1.9–3.3% of `C` on the grader, so the cost half was never as uncertain as
   the local measurement suggested.
5. **Layer-0 alphas are EXACTLY 0.0** — never compare them with a strict `>`
   against their own minimum. v2.6 did, and deleted the whole first layer on every
   odd width.
6. **Allocating a seed block is not consuming it.** D16 read "reusable" for nine
   days while five probes had drawn 270 of its seeds, two of them the same 24.
7. **A ±1 cube array is a degree-4 design only in expectation over SPREAD
   directions**, and the random rotation is what makes that true. That is why the
   design works and why it cannot be shrunk.

## Where the shipped file stands

`estimator.py` == `estimator_v2_6_1.py`, **NOT promoted**. Graded number of record
is still **v2.4.1 = 2.6314e-7** (submission 327743). Projected under the
calibrated residual line: **net V·c 1.0940× → ~2.405e-7**, known to well under 1%
on the cost side, against a bracket that used to span 17 points and flip sign.

## State of the protected things

- **Vault: 1 of 3 touches.** Untouched all session; manifest mtime still
  2026-08-06.
- **D17 and D19: SINGLE_USE, both unspent.** D16 has 330 free, D18 has 216.
- **Nothing submitted.** `docs/HUMAN_MODEL.md` never written by an agent.
- **preflight GREEN 6/6**, portability **174/174**, invariants **30/30**.

## Three decisions waiting, all Samuel's

1. The **gate change** (arithmetic primary, t/basic interval) — evidence complete.
2. The **E1 → E2** reclassification of the cost half — it gates the autonomy rule.
3. **Vault re-baselining** — touch #1 came from the old instrument.

## The one open research problem

A **non-linear orthogonal array or spherical 4-design near 33,152 points** on 256
dimensions. Pure design theory: no networks, no seed block, no grader. It is the
only thing between this project and a measured **1.4775×** — worth
`2.6314e-7 / 1.4775 = 1.781e-7`, roughly half the remaining gap to the best
published honest score (9.10e-8).


**Addendum at close — a fourth instance of the recurring error.** The seed-ledger
test's D18 assertion went stale twice in one session: first "D18 has zero
consumption" (the coverage lab claimed 192), then "exactly one claim of 192 by
coverage_lab" (the OA probe claimed 192 more). **A test of a LIVE, growing ledger
must assert invariants, not a snapshot.** Replaced with: the accounting closes
(`consumed + free == block size`) and every claim names a real consumer inside the
block. 41/41.

That is the same failure mode as predictions B4 and R1 and the rotation check —
**an assertion written against a moment rather than against a property.** Four
instances in one session is a pattern worth naming, and it is now in
`docs/GOTCHAS.md`.



---

# 2026-08-20 — CP0 RECONCILIATION + SLICE CP0.1: the preflight gate does not gate what it claims

*Session opens against `docs/FRONTIER_ARCHITECTURE_PROPOSAL.md` rev 2, treated as a
non-binding strategist packet. `docs/CURRENT_STATE.md` and the registered contracts
remain authoritative. Read-only reconciliation first; no seed block drawn, no vault
touch, no estimator edit, nothing submitted.*

## Entry state, verified not assumed

`preflight` **GREEN 6/6**. `vault.py verify` clean, counter **1 of 3**, log 1 entry.
`verify_portability.py` **174/174**. `estimator.py` == `estimator_v2_6_1.py`
(sha256(lf) `1caa3752be56e8625ddb8b163ff8ffa8`). Working tree carries only doc edits.

Phase 2 re-polled 2026-08-20: round 1429 slipped a **fourth** time, Aug 20 ->
**2026-08-21T23:59Z**. Leaderboard 1480 exists with **0 entries**; HF tags still
`['v1-phase1', 'v1-warmup']`. Width, depth, `n_mlps`, `B`, lambda and the multiplier
floor remain unpublished. **CP0's phase/config manifest deliverable is BLOCKED on the
organisers** and no amount of local repair changes that.

## The ten proposed validation blockers, classified against the code

| # | blocker | status | evidence |
|---|---|---|---|
| 1 | gate bound to the geometric mean | **open, deliberate** | `docs/VALIDATION.md` 2b unchanged; `test_competition_score.py` 71/71 carries the graded fixture where the estimands disagree (arith 1.0334x, geo 0.9666x). STRATEGIC, Samuel's |
| 2 | `paired_compare()` exposes a percentile bootstrap | **partially fixed** | `competition_score.py:514` makes arithmetic primary, `:656` says read `t_ci_difference`. But percentile owns the generic name `boot_ci_difference` and prints first (`:650`); `PairedResult` has no `basic` field although `paired_diff_ci:453` implements it; no single `decide()` a gate is forced through |
| 3 | preflight regression coverage | **open** | `scripts/preflight.ps1` is 6 steps; step 5 compares counter to log-line count and never re-hashes the vault arrays. Measured 2026-08-20, all green, none gated: scorer 71/71 (9.8 s), shape parity 27/27 (479.3 s), ledger negatives 41/41 (0.6 s), carriers 27/27 (104.4 s), `vault.py verify` (0.4 s) |
| 4 | ledger scan misses reuse after a claim | **open** | `seed_ledger.py:460` tests `val not in known`, so a literal already covered by ANY record passes silently; `:448` matches block not range; `:466` needs a literal block name, so computed access is invisible. Both D18 rows in `experiments/seed_ledger.jsonl` carry an EMPTY `artifact_sha256` while their artifacts exist |
| 5 | evaluators do not reproduce the grader lifecycle | **open, measured** | see below |
| 6 | vault touch is not transactional | **open** | `vault.py:367` opens the truth, `:384` runs 500 MLPs, seal written only at `:429`. An interrupt in between reads the truth and logs nothing. `candidate_preds.npz` is overwritten each touch |
| 7 | residual model | **open** | `residual_model.py:118` fits five points of per-submission AGGREGATE means; cannot score per MLP, and per-network MSE and cost are correlated. 1.75x op extrapolation stands |
| 8 | public CONFIRM still adaptive | **partially fixed** | declared read-only in VALIDATION 0; nothing mechanically prevents a "it survives" result from changing selection |
| 9 | truth seeds masked to 31 bits | **open, low impact** | `& 0x7FFFFFFF` at `vault.py:105`, `coverage_lab.py:96`, `audit_constants.py:252/334/403`, `bias_floor_lab.py:57`, `gpu_parity.py:110`, `robustness.py:199`; whestbench `seeds.py:26` uses the full 32-bit value. Unbiased, not bit-reproducible |
| 10 | one truth replica | **open** | no replica support anywhere; no cross-product estimator available |

### Blocker 5, measured rather than asserted

Cost-only probe, 6 networks, infra seeds `5_150_000 + i` (outside every registered
block, the `residual_model.py` convention), real flopscope, no truth read.
Arm G = grader lifecycle (one `Estimator`, `setup()` once OFF-budget, per-MLP
`BudgetContext`), matching `whestbench/runner.py:150` and
`subprocess_worker.py:197`. Arm V = the vault/coverage lifecycle
(`vault.py:282`, `coverage_lab.py:226`): fresh `Estimator()` per MLP, `setup()`
never called.

| i | seed | G flops | G ops | V flops | V ops | delta |
|---|---|---|---|---|---|---|
| 0 | 5,150,000 | 4.56510e10 | 2371 | 4.59038e10 | 2526 | **+2.5273e08** |
| 1 | 5,150,001 | 4.72473e10 | 2383 | 4.72473e10 | 2383 | +0.0000e00 |
| 2-5 | ... | ... | ... | ... | ... | +0.0000e00 |

**The defect is exactly one MLP wide.** MLP #0 is over-charged +2.5273e8 counted
FLOPs (+0.554% of its own `C`, +155 operations); every other MLP is
**bit-identical**, because `_BASE_POINTS` (`estimator.py:497`) is a MODULE-level
dict, so the Sobol table is built inside MLP #0's budget and is free thereafter.
**The recorded vault baseline and the coverage study are NOT invalidated** — the
aggregate effect is 1/500 of 0.55%. It becomes load-bearing only for a candidate
with expensive `setup()`, which is precisely a K129 carrier: 66,048 directions built
off-budget on the grader, billed to MLP #0 here — or, if held in instance state
rather than a module global, billed to all 500.
Artifact: scratchpad `lifecycle_gap.json`; `residual_model.py:128` already calls
`setup()` and is the one existing path that gets this right.

## SLICE CP0.1 — wire the ungated regressions into the gate. IMPORTANT.

Route EVALUATION_REPAIR, move type instrumentation. Touches `scripts/` only;
no estimator edit, no seed block, no vault touch, no submission.

**The problem in one line:** four suites and the vault's own hash verification pass
and are not gated, so "preflight GREEN" is a weaker claim than every later slice
will take it to be.

**The design decision, stated before implementing.** Two tiers, and the split is by
*cost*, not by importance, with the skip made loud:

- **default** adds the three cheap checks — canonical scorer (9.8 s), seed-ledger
  negative tests (0.6 s), full `vault.py verify` hash + seed-reconstruction +
  counter/log agreement (0.4 s). Target: 9 steps, still ~2-3 minutes.
- **`-Full`** adds shape parity (479 s) and carriers (104 s). Required before any
  promotion, submission or vault touch, and the default run must PRINT which checks
  it skipped rather than implying full coverage.

The rejected alternative is a sha-keyed cache that would auto-run shape parity only
when `estimator.py` moves. It is more precise and it adds silent-skip state to a
gate; `docs/GOTCHAS.md` records four separate cases this session's predecessor lost
to cleverness in a check. Rejected on that ground, not on effort.

### Written predictions, before any of it is run

- **G1.** Default preflight ends **GREEN 9/9** on the current tree.
- **G2.** Default preflight wall time **<= 200 s**.
- **G3.** `-Full` ends **GREEN 11/11**, wall time **<= 800 s**.
- **G4.** `vault.py verify` costs **no touch**: counter reads **1** before and after,
  and `touch_log.jsonl` still has exactly 1 line. (Partially pre-verified — a manual
  `verify` at 0.4 s left the counter at 1.)
- **G5.** The vault step must degrade, not fail, when `experiments/vault/vault_v1.npz`
  is absent — it is gitignored, so a fresh clone has no arrays. Predict that a naive
  `vault.py verify` call exits **non-zero** in that case and therefore needs an
  explicit missing-arrays guard. This is the one prediction I expect to be the
  failure that costs time.
- **G6.** `verify_portability.py` stays at **174/174** after the edit.
- **G7.** No new step touches `docs/HUMAN_MODEL.md`, the vault seal, or any seed
  block; the guard hooks stay silent.

Recorded before implementation. Results follow.

### CP0.1 RESULTS — all seven predictions held, and G5 was the one that cost time

Implemented in `scripts/preflight.ps1`: a `-Full` switch, three new default-tier
steps, two `-Full`-only steps, and a red skip-notice in the default trailer.

| # | prediction | result | verdict |
|---|---|---|---|
| G1 | default ends GREEN 9/9 | **GREEN 9/9** | HELD |
| G2 | default <= 200 s | **103.9 s** | HELD |
| G3 | `-Full` GREEN 11/11, <= 800 s | **GREEN 11/11, 697.6 s** | HELD |
| G4 | `vault.py verify` costs no touch | counter **1 -> 1**, log **1 -> 1** across both runs | HELD |
| G5 | a naive `vault.py verify` exits NON-ZERO with the arrays absent, so a guard is needed | **exit 1**, `vault.py:169` `raise SystemExit("vault not minted yet")`. Tested by renaming `vault_v1.npz` inside a `try/finally`: guarded step 9 printed SKIP and the run stayed **GREEN 9/9**; arrays restored and all four sha256s re-verified OK | HELD |
| G6 | portability stays 174/174 | **174/174** | HELD |
| G7 | no guard fires; no protected path touched | no denial; `git status` shows only `preflight.ps1`, `LOG.md`, `VALIDATION.md` and the two pre-existing doc edits | HELD |

**The design decision, as built.** Default tier = 9 steps, 103.9 s. `-Full` = 11
steps, 697.6 s, and steps 10-11 are 583 s of that. The default trailer prints the
skipped checks **in red** with the exact command to run them and the sentence
"required before promoting, submitting, or spending a vault touch". Step 8 is not a
duplicate of step 6: step 6 proves the ledger is CONSISTENT, step 8 proves its
REFUSALS still fire.

### Knowledge Delta

1. **"preflight GREEN" now covers 165 more checks than it did this morning** —
   scorer 71, ledger refusals 41, shape parity 27, carriers 27 — plus the vault's
   four array hashes and the seed->weights reconstruction. Every prior LOG entry
   that cites "preflight GREEN 6/6" was making a weaker claim than its wording
   suggests. Nothing recorded is wrong; the coverage was.
2. **`vault.py verify` costs no touch, and that is now load-bearing rather than
   incidental.** It is inside the gate, so it runs on every change. Measured: the
   counter and the log both stay at 1. The touch semantics are about scoring
   predictions against the truth, not about hashing it.
3. **The vault arrays are gitignored, so the honest default on a fresh clone is
   SKIP, not FAIL.** A gate that goes red for a legitimately absent 33 MB artifact
   teaches people to ignore red. Step 5 still covers the seal there.
4. **A sha-keyed cache to auto-run shape parity only when `estimator.py` moves was
   designed and rejected.** It is more precise and it adds silent-skip state to a
   gate. `docs/GOTCHAS.md` records four separate checks lost to cleverness in one
   prior session; this is the same shape and was declined on that ground.

Durable: `docs/VALIDATION.md` section 7 now carries the two-tier table and the
measured wall times.

## Also established this session, read-only — the K129 arithmetic closes exactly

Not a slice; a reconciliation finding. From our own graded record and the published
K129 aggregates in `docs/FRONTIER_ARCHITECTURE_PROPOSAL.md`:

```
adjusted-score gap, ours/theirs         2.3636x
  = variance efficiency (MSE*N ratio)   1.4976x
  x per-sample cost                     1.5782x        (product reproduces the gap exactly)
```

**Our own independently measured degree-4 design-carrier gain is 1.4775x** (D18, 48
networks, studentised CI excluding zero). The variance leg implied by their
published aggregates is **1.4976x** — a **1.4%** discrepancy between two
completely independent estimates of the same quantity.

And the affordability blocker reads differently at half the point count. Modelling
our own per-sample counted cost (3.3030e6 FLOPs/sample at the graded operating
point) at K129's **66,048** directions:

| arm | raw MSE | C/B | score | vs graded |
|---|---|---|---|---|
| our Sobol carrier at N = 66,048 | 3.3049e-07 | 0.8021 | 2.6507e-07 | **1.000x** — reproduces "N is not a lever" exactly, as it must |
| + the measured 1.4775x design gain | 2.2368e-07 | 0.8021 | 1.7941e-07 | **1.478x** |
| + their per-sample cost engineering | 2.2368e-07 | 0.5082 | 1.1368e-07 | 2.332x |

`C/B = 0.8021` is **under the failure cliff**. The 131,072-row BCH array died at
1.3509 — over it. **Halving the point count is what makes the design lane
affordable**, and `docs/CURRENT_STATE.md`'s proof that 131,072 is the floor is a
proof about **linear codes**; a MUB frame bank is not one, so it is not covered.

**INFERRED, not measured.** It assumes the 1.4775x gain measured on a strength-5
array at 131,072 points survives on a MUB bank at 66,048. `local_sim/carriers.py`
has `sobol`, `haar_frames` and `bch_oa` and **no Kerdock/MUB construction**;
`test_carriers.py` S5 checks only the arithmetic `129 x 256 x 2 = 66,048` and never
builds one. The existing degree-1..6 moment harness would price it for free.

**The cheapest discriminating experiment in the project right now needs no
networks, no seed block and no grader**: build the MUB-129 carrier and read its
degree-4 moment error against `bch_oa`'s 0.0305x iid and `haar_frames`' 0.700x. If
it lands near the array, CP1 is alive and affordable. If it lands near the Haar
bank, it is only a 3-design, the published K129 advantage is not coming from
degree-4 strength, and the plan's premise needs rewriting before anything is built.

---

## 2026-08-20 — CP0.1 REVIEW: independent Skeptic **PROVISIONAL**, and it was right twice

Skeptic run in a separate agent context that never saw the builder's reasoning,
handed the contract, the diffs, the evaluation protocol and the raw artifact paths
— never a prose summary. Verdict relayed unsoftened.

> **Skeptic verdict: PROVISIONAL**
>
> *"The gate change (CP0.1) survives everything I threw at it, with two real
> defects in what it claims and one new coupling hazard it created. The K129
> arithmetic in the same entry does not survive: it prices a K129 comparison with a
> different team's multiplier, mislabels an effective cost as counted, and calls a
> tautology a confirmation."*

It replicated all seven predictions G1–G7 independently, and confirmed by a route
the builder had not used that step 9 costs no touch: `manifest.json` and
`touch_log.jsonl` **mtimes unmoved from 2026-08-06** across two of its own runs,
plus `vault.py:182 verify()` → `:168 load()` having no write path at all.

### What it falsified, and what was done about it

| # | finding | verified? | action |
|---|---|---|---|
| 1 | **Step 9's SKIP printed a green PASS and was counted in "9/9"**, and was never added to `$skipped` — so `docs/VALIDATION.md`'s new sentence "prints in red exactly which checks it skipped" was **false as built**, in exactly the fresh-clone case it was written for | **yes** | **FIXED.** `Step` now honours `$script:StepSkipped`; a skip prints yellow SKIP, is counted separately, and is named in the red block. **The denominator is now always 11.** A default run reports `GREEN 9/11, 2 SKIPPED`, never `9/9`; arrays absent reports `8/11, 3 SKIPPED` |
| 1b | *(found while fixing 1, by the builder)* the first fix denominated against the tier that ran, so it printed `6/9, 3 SKIPPED` — wrong in the other direction | **yes** | **FIXED** in the same edit |
| 2 | **`test_seed_ledger.py` asserted D17 and D19 have ZERO consumption, and D16 exactly 270 seeds with four exact gaps.** D19 exists to be spent once, on the v2.6 promotion gate. Since this file now runs inside the mandatory gate, **the first legitimate claim on D19 would turn preflight RED at the exact moment `-Full` is mandatory**, and the only way forward would be to edit the gate | **yes** | **FIXED.** Converted to invariants, following the pattern already used for D18 three lines below: a SINGLE_USE block has **at most one** recorded claim, and any claim names a real consumer inside the block; D16's accounting **closes** and its free ranges are disjoint, ascending and in-block. **41/41 → 44/44.** Proved on a synthetic ledger copy: the new assertion holds unspent **and** after a legitimate spend, where the old one flips to False; a second claim is still refused with `PermissionError` |
| 3 | Step 8 writes `local_sim/_ledger_scan_tmp/sneaky_probe.py` containing a real D18 literal into the tracked tree; a hard kill mid-step leaves it, and **step 6's scan runs BEFORE step 8**, so it would fail the gate on every later run | **yes** | **FIXED**, minimally: directory added to `.gitignore`, and the test clears any corpse before it starts. Deliberately did NOT add a skip rule to `scan_sources` — a scanner that ignores a directory can hide a real finding, and the scan's message names the file, so the failure is self-diagnosing |
| 4 | **The 103.9 s default-tier wall time did not replicate** — the Skeptic measured 113.36 s and 113.61 s | **yes** | **CORRECTED.** Three replications: **111.3 s mean, range [109.7, 112.8], spread 2.8%.** `docs/VALIDATION.md` now quotes n and range. 103.9 s was a single unreplicated first run |
| 5 | Stale gate references across durable docs: `ai/workflows/review-slice.md:32,151`, `routine.md:31`, `session-close.md:92`, `slice-contract.md:136` all said "preflight 5/5"; `docs/CURRENT_STATE.md:101` said "GREEN 6/6" and "shape parity 23/23" | **yes** | **FIXED** in all six places. `review-slice.md`'s promotion rule now demands **`-Full` 11/11** explicitly and states that a default `9/11` run is not sufficient |
| 6 | Pre-registration ordering is **not reconstructible from the repository** — no commit, and G1–G7 plus the RESULTS live in the same uncommitted file | **yes** | **ACCEPTED, not fixed.** Labelled unverified rather than asserted. IMPORTANT work does not require single-use pre-registration so it is not disqualifying, but the Skeptic is right that "recorded before implementation" cannot be checked from disk. **This is an argument for committing each slice, and that decision is Samuel's** |

Attacks that **failed** — the Skeptic tried and could not break these: the SKIP
guard cannot mask a genuine `vault.py verify` failure (mutually exclusive branches
of one `if`; a non-zero exit from the `else` side still FAILs the step); step 8 does
not mutate the live ledger (refusal tests run in `tempfile.TemporaryDirectory()`,
bracketed by a size assertion on the real path); step 8 is genuinely not a duplicate
of step 6; the rejected sha-keyed cache was the right call on the stated ground; the
`$total` count is honest on failure; and G1, G4, G5, G6, G7 all replicated exactly.

### CORRECTION — the K129 factorization, withdrawn and rebuilt

**Three errors, all confirmed by recomputation:**

1. **`0.5082` is not K129's multiplier.** It belongs to the **board #1 entry
   (ednacob)** — a different team, different method, unknown point count
   (`docs/FRONTIER_ARCHITECTURE_PROPOSAL.md:90-92`; 1.845e-8/3.633e-8 = 0.50784).
   K129's own is **0.51728** (140.7e9/B) or **0.51808** (adjusted/raw). The tell was
   sitting in the builder's own output — the recomputed score read 1.1215e-7 against
   a published 1.1433e-7 and was not chased. At 0.51808 it reproduces exactly.
2. **`3.3030e6 FLOPs/sample` was labelled COUNTED and is EFFECTIVE** — it is
   `0.19896 × B / 16384`, residual included. `docs/VALIDATION.md` §2b exists to stop
   precisely this. Our counted per-sample is 3.1691e6 (D18 arm A).
3. **"the product reproduces the gap exactly" is an algebraic identity, not a
   check.** The `MSE × (C/B)` gap is *by definition* (MSE·N ratio) × (per-sample-cost
   ratio). It cannot fail, and it was presented as corroboration. **Withdrawn.**

**And a fourth, which the Skeptic found and which matters most: the "N is not a
lever" row was circular.** It read 1.000× "as it must" — but only because the row
*assumed* MSE·N flat. This repository's own T2 result from the previous night
(`experiments/LOG.md`, 2026-08-19 night, **FALSIFIED**) measured the flat law
breaking by **1.1980×** at 8× the points on 48 networks, decomposing into MSE·N
**1.0591** and counted-FLOPs-per-point **1.1306**. Part of K129's apparent variance
advantage is therefore "QMC beats 1/N at 4× the points", not carrier strength.

**Rebuilt from our own measured D18 arms rather than from a borrowed multiplier.**
Fitting `C(N) = c0 + c1·N` on arms A and B (same 48 networks, same carrier, COUNTED
FLOPs): `c0 = 6.8534e9`, `c1 = 2.7508e6`.

| quantity | as first written | corrected | source |
|---|---|---|---|
| K129 `C/B` | 0.5082 | **0.51728** | its own 140.7e9 effective / B |
| our `C/B` at N = 66,048 | 0.8021 (flat per-sample) | **0.6932** counted | the fitted line on our own arms — the first figure was **pessimistic** |
| variance leg | 1.4976× | **1.414× – 1.498×**, point estimate **1.441×** | corrected for the measured N-effect; the bracket spans applying none of the 8× factor to all of it |
| cost leg | 1.5782× | **NOT COMPUTABLE** | K129's *counted* figure is unpublished; only "140.7B effective" exists, and counted compares to counted |
| "product reproduces the gap" | presented as a check | **withdrawn** | algebraic identity |

**What survives, and it is the load-bearing part.** Our measured design gain is
**1.4775×** adjusted / 1.4826× raw (bchOA vs sobol at matched N *and* matched cost,
48 D18 networks, studentised CI excluding zero). The corrected variance leg implied
by K129's published aggregates is **1.414× – 1.498×**, and 1.4775 lies **inside**
that bracket. So the two are **consistent** — but the claimed "1.4% agreement" is
withdrawn: the honest statement is *consistent within a bracket of about ±3%, whose
width is set by a modelling choice about how the N-effect interpolates from 8× to
4×.*

**And the affordability conclusion is STRONGER than first claimed, not weaker.**
`C/B = 0.6932` counted at 66,048 points, against the failure cliff at 1.0 and the
131,072-row array's 1.3509. Halving the point count clears the cliff with more
margin than the first estimate showed.

**The caveat the first version never stated, and which the Skeptic named:**
1.4775× is bchOA-vs-sobol under matched conditions; the K129 leg is us-vs-them
across a different first layer (FWHT), different pruning, and a different pipeline.
Attributing all of it to the carrier requires K129 to draw **zero** variance benefit
from anything else. That is an assumption, it is unstated in their write-up, and
their per-component ablation is not published.

### Unresolved, by both of us

(a) whether G1–G7 were written before the edit — no commit, no separating artifact;
(b) whether K129's variance advantage is carrier strength or FWHT/structure/pruning;
(c) what point count the ednacob 0.5082 entry uses; (d) whether a MUB-129 bank
reaches degree-4 strength — `local_sim/carriers.py` still has no Kerdock/MUB
construction.

### The ten review questions

1. **Prediction supported?** All seven, G1–G7, replicated by the Skeptic. G2's
   *value* was wrong (103.9 s vs 111.3 s mean) though its *gate* (≤ 200 s) held.
2. **Mechanism or only sign?** Mechanism. Step 9's no-touch property is established
   by reading the code path (`verify()` → `load()`, no writes) and confirmed by two
   independent observables — counter/log, and file mtimes.
3. **Strongest competing explanation?** That the added steps are theatre because
   they already passed. Refuted by finding #2: converting D17/D19 from snapshot to
   invariant removed a defect that would have fired at promotion time — the gate
   found a real bug in itself within one review cycle.
4. **Regressions or tradeoffs?** The default gate now costs 111 s and `-Full` ~700 s.
   Real friction, deliberately taken, with the skip made loud. No estimator change,
   no seed drawn, no vault touch; the tree is byte-identical after runs.
5. **New information?** The gate had three defects of its own; a durable claim in the
   same entry had four arithmetic errors; the affordable-carrier `C/B` is 0.693, not
   0.802.
6. **Belief strengthened?** That the design-carrier lane is affordable at 66,048
   points — on a *better* number than before, computed from our own arms.
7. **Belief weakened?** That the published K129 advantage is cleanly attributable to
   the carrier. The cost leg is not computable at all, and the variance leg is a
   bracket, not a point.
8. **Uncertain?** (a)–(d) above.
9. **Durable invariant or failure mode?** Yes — the **fifth** instance of "assert
   properties, not snapshots", and the first one wired into a mandatory gate.
   Recorded in `docs/GOTCHAS.md`.
10. **Next?** Terminal state below.

### Terminal state: **FOLLOW_UP_EXPERIMENT**

Skeptic PROVISIONAL. Its named cheapest discriminating follow-up — recompute the
K129 factorization before anything is built on it — **has been executed above, and
its stated decision rule is met**: the corrected variance leg (1.441×, bracket
[1.414, 1.498]) lands within 5% of the measured 1.4775×, so the carrier attribution
survives and CP1 may proceed. The follow-up it gates is the MUB-129 moment probe,
which needs no networks, no seed block and no grader.

The three gate defects are fixed and re-verified. Two items are accepted and not
fixed, both needing Samuel: the `-Full` requirement is unenforced prose rather than
an executable precondition in `scripts/guard_paths.py`, and the pre-registration
ordering is not reconstructible without per-slice commits.

---

# 2026-08-20 — SLICE CP0.2: the canonical grader-lifecycle evaluator. IMPORTANT.

Route EVALUATION_REPAIR, move type instrumentation. Blocker 5 of the CP0 audit,
plus the per-MLP evidence-row requirement the frontier proposal calls "the required
target architecture for evaluation". Touches `local_sim/` only: no estimator edit,
no seed block, no vault touch, no submission.

**Phase-2 assumption in force** (Samuel, 2026-08-20, recorded in
`docs/CURRENT_STATE.md` § *The clock*): build at 256×32, no biases, He init,
`B = 2.72e11`, λ = 1e11, floor 0.1 — Phase 1 with different MLPs, possibly a
little harder. Everything below is shape-generic regardless.

## The problem, measured

`local_sim/vault.py:282` and `local_sim/coverage_lab.py:226` construct a fresh
`Estimator()` for every MLP and **never call `setup()`**. The grader constructs one
instance and calls `setup(context)` **once, outside any `BudgetContext`**
(`whestbench/runner.py:150`, `subprocess_worker.py:197`). Measured cost of the
mismatch today: MLP #0 over-charged **+2.5273e8** counted FLOPs (+0.554% of its own
`C`, +155 ops), every other MLP bit-identical, because `_BASE_POINTS`
(`estimator.py:497`) is a module global.

It is small now and it is structural. A K129 carrier builds 66,048 signed
directions in `setup()` — free on the grader, billed to MLP #0 here, or billed to
**all 500** if that carrier is held in instance state instead of a module global.
CP1 cannot price a carrier candidate on an evaluator that gets this wrong.

## What is being built

`local_sim/lifecycle_eval.py` — one canonical evaluator, used by every future
comparison, that reproduces the grader's lifecycle and emits immutable per-MLP
evidence rows. It scores through `local_sim/competition_score.py` (already 71/71
bitwise against the installed whestbench) rather than re-deriving the identity.

Grader semantics to be mirrored exactly, read from `whestbench/scoring.py`:

- one `Estimator`, `setup(SetupContext)` once, off-budget, **timed against
  `spec.setup_timeout_s = 5.0`**;
- per MLP, its own `flopscope.BudgetContext(flop_budget, wall_time_limit_s)`;
- six failure paths, all zeroing the prediction: `BudgetExhaustedError`,
  `TimeExhaustedError`, any other `Exception`, post-predict `wall_time_s` over
  limit, post-predict `residual_wall_time_s` over limit, and the post-hoc combined
  check `C = F + λR > B`;
- `failed` is the **disjunction** of budget / time / residual / combined;
- failure is priced at multiplier **1.0**, not the 0.1 floor, against **zeroed**
  predictions, so the MSE becomes `mean(target²)`;
- MSE computed in **float32**, as `scoring.py:846` does.

## Written predictions, before any of it is run

- **L1.** Under the correct lifecycle the per-MLP counted FLOPs equal the arm-G
  numbers already measured, and **MLP #0 no longer carries the +2.5273e8
  inflation**: max |Δ counted FLOPs| == **exactly 0** across the 6 probe networks.
- **L2.** The evaluator's aggregate equals a hand-computed
  `mean_i mse_i · max(0.1, C_i/B)` at **0.0 relative deviation**, and equals
  `competition_score.score_suite` on the same rows bitwise.
- **L3.** `setup()` on the shipped v2.6.1 completes in **< 1.0 s**, well inside the
  grader's `setup_timeout_s = 5.0`. **This 5-second ceiling is recorded nowhere in
  this repository and is load-bearing for CP1** — a 66,048-direction carrier must
  build inside it or fail every MLP.
- **L4.** All three provokable failure paths reproduce exactly: a tiny budget →
  `BudgetExhaustedError`; a raising `predict` → the exception path; a row with
  `C > B` → the combined check. Each gives `failed=True`, zeroed predictions,
  multiplier **exactly 1.0**, and `mse == mean(target²)`.
- **L5.** Re-running the vault's *current* lifecycle beside the grader lifecycle on
  the same 6 networks reproduces the CP0 finding: **exactly one** row differs and
  it is **index 0**.
- **L6.** Evidence rows are content-addressed and the hash is **stable across two
  runs**, with timing fields excluded from the hash because they are not
  deterministic. *This is the prediction I expect to fail first* — the risk is
  failing to exclude every nondeterministic field.
- **L7.** `preflight -Full` stays **GREEN 11/11**; no seed block drawn; vault
  counter stays **1** and the touch log stays at 1 line.

Recorded before implementation. Results follow.

### CP0.2 RESULTS — 26/26, and L4's first spelling was falsified before it was fixed

Built `local_sim/lifecycle_eval.py` and wired its self-test into the gate as
**step 10**. The gate is now **10 of 12** by default, **12 of 12** under `-Full`.

| # | prediction | result | verdict |
|---|---|---|---|
| L1 | grader-lifecycle counted FLOPs equal arm-G; MLP #0 loses the +2.5273e8 inflation; max abs delta on rows 1.. is exactly 0 | rows 1–5 **0.0000e+00**, all five; MLP #0 delta **+2.5273e8**, +155 ops, and the old path only ever over-charges | **HELD** |
| L2 | objective equals a hand-computed `mean_i mse_i · max(0.1, C_i/B)` at 0.0 relative | **0.000e+00** relative deviation | **HELD** |
| L3 | `setup()` under 1.0 s, inside the grader's 5.0 s ceiling | **176.9 ms** of 5000 ms allowed, no error | **HELD** |
| L4 | all provokable failure paths reproduce exactly | **HELD, after the first spelling was FALSIFIED** — see below | **HELD (corrected)** |
| L5 | exactly one row differs between lifecycles, and it is index 0 | **1 row, index 0** | **HELD** |
| L6 | `rows_hash` stable across two runs, timing fields excluded | stable; **and the two lifecycles hash differently**, so the address can discriminate; timings *did* move between runs and the hash did not | **HELD** |
| L7 | `preflight -Full` GREEN, no seed, vault counter 1 | **GREEN 12/12**, 755.0 s; ledger scan clean; touches **1 → 1**, log **1 → 1** | **HELD** |

**L4's first spelling was falsified, and the finding is worth more than the
prediction.** The check "a failed MLP's MSE equals `mean(target²)`" failed at a
1e-12 relative tolerance while printing two apparently identical numbers
(`8.458488e-02` vs `8.458488e-02`). The cause: the evaluator computed
`zero_prediction_mse` in **float64** while the grader computes MSE in **float32**
(`whestbench/scoring.py:846`). Measured divergence **1.96e-9 relative** — far below
anything that would change a decision, and far above an equality claim.

That is precisely the *two-implementations-of-one-identity* error this module's own
docstring warns against, committed in the same file that warns against it. The fix
was to call `competition_score.zero_prediction_mse`, which already existed and
already matched the grader at **0.000e+00**. Both assertions were then **tightened
from a 1e-12 tolerance to exact equality**, which is the standard the quantity
actually supports.

```
float32 path (grader)  8.45848843455314636e-02
float64 path (mine)    8.45848841797775808e-02     <- 1.96e-9 relative
canonical helper       8.45848843455314636e-02     <- 0.000e+00
```

### What the evaluator pins

Six failure paths, each provoked and asserted, not assumed:

| path | how it was provoked | verdict |
|---|---|---|
| `BudgetExhaustedError` | `flop_budget = 1e6` | 2/2 exhausted, failed, zeroed, multiplier **exactly 1.0** |
| generic `Exception` | a candidate whose `predict` raises | `error_code = RuntimeError`, failed, exactly zeroed |
| validation failure | a candidate returning a **transposed** `(width, depth)` array | caught as `ValueError` — scored as a failure, never scored as a prediction |
| combined `C = F + λR > B` | a hand-built row at C = 3.000e11 vs B = 2.72e11 | flagged by `competition_score.over_budget_rows`, not by a second implementation |
| `TimeExhaustedError`, post-predict wall/residual limits | wired and read from `ContestSpec`; not provokable cheaply | code path present, constants asserted against the installed package |

And step 0 of the self-test asserts the four transcribed grader constants still
match the installed whestbench, so a version bump that moves `setup_timeout_s`,
`wall_time_limit_s`, `residual_wall_time_limit_s` or λ fails the gate instead of
silently changing what "the grader's lifecycle" means.

### Knowledge Delta

1. **The grader allows `setup()` exactly 5.0 seconds** (`ContestSpec.setup_timeout_s`),
   and that ceiling was recorded **nowhere in this repository** before today. It is
   directly load-bearing for CP1: a K129/MUB carrier builds 66,048 signed directions
   in `setup()`, and if that build exceeds 5 s the submission fails **every** MLP.
   The shipped v2.6.1 uses **176.9 ms of the 5,000 ms**, so there is real headroom —
   but it is finite and now measured on every preflight run.
2. **The lifecycle defect is exactly one MLP wide on the shipped file, and it is
   only that narrow by accident.** `_BASE_POINTS` is a module-level dict. Any
   candidate that holds its carrier in **instance** state would be billed the full
   build on **every** MLP under the old evaluator. The narrowness is a property of
   v2.6.1, not of the old evaluator.
3. **`mean(target²)` is a float32 quantity, not a float64 one**, and the two differ
   by ~2e-9 relative. Any future "the failure path is exactly X" claim must go
   through `competition_score.zero_prediction_mse`.
4. **Evidence rows are now content-addressed**, with wall-clock fields excluded from
   the hash. Two runs of the same candidate on the same networks produce the same
   `rows_hash`; the two lifecycles produce different ones. `write_evidence` **refuses
   to overwrite** an existing bundle — the failure mode that lost per-touch
   prediction evidence before (`vault.py` overwrites `candidate_preds.npz` on every
   touch).

### Deliberately NOT done in this slice

`local_sim/vault.py` and `local_sim/coverage_lab.py` were **not** rewired to use the
new evaluator. That is a separate slice: both have recorded results attached, and
`vault.py` is guarded, so changing them is its own IMPORTANT change with its own
prediction. The new evaluator is additive and nothing currently depends on it except
the gate.

Blockers 4, 6, 7, 9 and 10 of the CP0 audit remain open and untouched.

---

## 2026-08-20 — CP0.2 REVIEW: independent Skeptic **PROVISIONAL**, and its lead finding invalidated the slice's own licence

Second Skeptic, fresh context, handed the contract, the diffs, the evaluation
protocol and the raw artifact paths. Verdict relayed unsoftened.

> **Skeptic verdict: PROVISIONAL**
>
> *"The per-MLP lifecycle is faithful and I could not break it. The licence the
> slice grants — 'can therefore be trusted to price a candidate, specifically a
> future K129/MUB carrier whose `setup()` is expensive' — is not supported, because
> the setup half of the lifecycle is the one part that is recorded but never
> enforced, measured on a narrower scope than either grader path, and calibrated
> against an unverified constant. A K129 carrier that blows the setup budget
> produces a clean, plausible objective from this evaluator while the grader aborts
> the entire submission."*

That is exactly right, it is the one thing the slice existed to make safe, and it
was wrong in the direction that would have cost the most.

### What it falsified, and what was done about it

| # | finding | verified? | action |
|---|---|---|---|
| 1 | **`setup_timed_out` / `setup_error` were computed into a provenance blob and read by nothing.** The grader calls `runner.start()` OUTSIDE the try wrapping `evaluate_estimator` (`cli.py:2357`) and `runner.py:155-172` raises `SETUP_ERROR`/`SETUP_TIMEOUT` out of it — **the whole submission dies, every MLP included**. Our evaluator returned a clean objective. A sixth instance of "reports a state instead of testing a property" | **yes**, read at `cli.py:2340-2372` and `runner.py:140-172` | **FIXED.** `evaluate()` now raises `SetupFailure` by default; `on_setup_failure="score_as_failure"` is the explicit opt-in. **Tested**, not asserted: a 6 s `setup()` and a raising `setup()` both refuse |
| 2 | **The setup timer started in the wrong place.** `runner.py:144` starts its clock BEFORE `load_estimator_from_path`; ours started after. The subprocess path (`runner.py:281`) additionally covers the worker spawn and its imports | **yes** | **FIXED.** `t0` moved before `load_candidate`. Host cold start measured here, 3 repeats: spawn + `import flopscope, flopscope.numpy` = **0.222 / 0.245 / 0.238 s**, recorded as `HOST_COLD_START_S` and added to the margin |
| 3 | **L3's headroom was measured on 13% of the real consumption.** Reported "176.9 ms of 5000 ms"; the corrected figure is **433 ms** (193 ms module-load + setup, plus 240 ms host cold start) | **yes** | **CORRECTED.** Also chased the Skeptic's claim that the apparent ~1 s was `import whestbench` pulling HF `datasets` — **confirmed**: first load 1052.5 ms, second 0.3 ms, third 0.2 ms, raw `compile()` of the 58,641-byte source only 1.9 ms. So estimator.py's own module body is ~0.3 ms and the second is a cold package import the grader's worker has warm |
| 4 | **"The grader allows `setup()` exactly 5.0 seconds" is a package default asserted as a grader fact.** All five durable graded reports publish `config_snapshot` with exactly `n_mlps`, `depth`, `width`, `flop_budget`. **None** of the five timing/memory constants appears in any of them; only λ = 1e11 is independently corroborated | **yes** | **CORRECTED** in the module, in Knowledge Delta #1 below, and in `docs/VALIDATION.md` §7. Step 0 now prints "these are PACKAGE DEFAULTS" every run |
| 5 | **`predict_timeout_s = 30.0` and `memory_limit_mb = 65,536` were not transcribed at all** — and the memory limit (RLIMIT_AS) is a shipping blocker that **already killed v2.5** at 3.9 GB peak | **yes** | **FIXED.** Both added to `LifecycleSpec` and to step 0's equality check against the installed package |
| 6 | **Every `residual_wall_time_s` is a `--runner local` figure, unlabelled**, so the printed `objective` is not a graded objective. λ·R is ~10–13% of `C` here against 2–3% on the grader | **yes** | **FIXED.** Module docstring, `docs/VALIDATION.md` §7, and the evidence bundle itself (`residual_is_graded: false`, `objective_is_graded: false`, `residual_measurement_path`) |
| 7 | **Two of the six failure paths were untested and the stated reason — "not provokable cheaply" — was false.** The Skeptic provoked both in under a minute | **yes** | **FIXED.** Both now tested: `residual_wall_time_limit_s = 1e-9` trips the residual path; a candidate sleeping 50 ms at `B = 1e9` gives `C = 6.043e9 > B` and trips the combined path. Both zeroed at multiplier exactly 1.0 |
| 8 | **A duplicate implementation of `F + λR > B` sat at `:404-408`** — a second implementation of the identity, inside the module whose docstring warns against second implementations | **yes** | **FIXED.** Deleted in favour of `competition_score.is_combined_budget_exhausted` |
| 9 | Divergences from `evaluate_estimator`: the exception path recorded real `F`/`R` where the grader records `flops_used: 0, effective_compute: 0.0` (`scoring.py:743-744`), and post-predict checks ran on exception rows where the grader `continue`s at `:760` | **yes** | **FIXED.** Exception rows now zero `F` and `R` and skip every post-predict check |
| 10 | `# seed-ledger: excludes V` was a **no-op that disabled a real check** — `seed_ledger.py:461` reads it as `bid not in excluded`, permanently suppressing the vault-seed-literal scan in this file | **yes** | **REMOVED**, with the reason written where the pragma was |
| 11 | `.gitignore` covered `_ledger_scan_tmp/` but not `local_sim/_lifecycle_broken_tmp.py`, which step 10 writes into the real tree on every gate run | **yes** | **FIXED** — `local_sim/_lifecycle_*_tmp.py` |
| 12 | `preflight.ps1`'s accounting comment was a garbled duplicated sentence; step 8's comment said "41 negative tests" where the step name says 44/44; `VALIDATION.md` quoted "~134 s" as a single reading three paragraphs above its own warning against single readings, and the Skeptic measured **156.5 s** | **yes** | **ALL FIXED.** The timing row is now a range with an n, and the 111 s figure is explicitly marked as describing the superseded nine-step tier |

**Self-test: 26/26 → 37/37.**

Attacks that **failed** — tried and could not break: `VOLATILE` is genuinely not a
dataclass field (the Skeptic enumerated `dataclasses.fields`); `rows_hash` is
design rather than luck, because `n_ops` and `flops_used` are deterministic *given
a network* (GOTCHAS' 3.22% is a per-network spread, not run-to-run) — and it
reproduced across two separate **processes**; L6 is not self-fulfilling, since the
check fails if timings did not move; `_is_frozen` returns True only via
`FrozenInstanceError`; `_lifecycle="per_mlp_no_setup"` is detectable in a written
bundle via `provenance.lifecycle` and `setup_called`; the seed literal is outside
every registered block; L7 re-verified independently, including
`manifest.json`/`touch_log.jsonl` mtimes unmoved at 2026-08-06 04:17.

And it **independently reproduced the CP0 baseline** rather than reading it:
MLP #0 over-charged **+2.5273e8** counted FLOPs (+0.5536% of its own
`C = 4.56510e10`), **+155 ops**, rows 1–5 delta **exactly 0.0000e+00**. It also
anchored the absolute level, which the slice had not: the 6-row mean counted is
**4.798e10** against v2.4.1's graded **5.330e10** = **0.900×**, matching the
recorded "counted FLOPs fall 10.5%" for v2.6.1.

It also found the premise **understated**: `vault.py:282` and `coverage_lab.py:227`
not only skip `setup()`, they compute MSE in **float64** where the grader uses
float32.

### Knowledge Delta

1. **A setup failure is a RUN-level event, not a per-MLP one.** `cli.py:2357` puts
   `runner.start()` outside the try that wraps scoring. A candidate whose `setup()`
   raises or overruns scores **nothing** — not one bad MLP, all of them. Any
   evaluator that reports a per-MLP objective in that case is lying about the most
   expensive failure mode available to a carrier candidate.
2. **The grader's timing and memory limits are UNOBSERVED.** `setup_timeout_s`,
   `wall_time_limit_s`, `residual_wall_time_limit_s`, `predict_timeout_s` and
   `memory_limit_mb` appear in **no** graded `config_snapshot` we hold. They are
   `ContestSpec` defaults and nothing more. Only λ = 1e11 is corroborated, from
   per-MLP telemetry. Quote them as a prior; never as "the grader allows".
3. **`memory_limit_mb = 65,536` is RLIMIT_AS under `--runner subprocess`, and it
   already killed v2.5 at a 3.9 GB peak.** A 66,048-point carrier is a
   66,048 × 256 float64 array = **135 MB** for the point set alone, before any
   propagation temporaries. This is now in the spec and checked on every gate run.
4. **The setup window is not the duration of `setup()`.** The local path starts its
   clock before the module is even loaded; the subprocess path also pays the worker
   spawn and its imports. Measured here: 0.24 s of spawn+flopscope, and a cold
   `import whestbench` worth ~1.05 s that the grader's worker has warm. The shipped
   v2.6.1 uses **433 ms of a nominal 5,000 ms** — comfortable, but 2.5× what a
   naive timer reports.
5. **This evaluator's `objective` is a local-runner objective.** Counted FLOPs
   transfer exactly; residual seconds do not, and they enter `C`. Ratios inside one
   path only. The bundle now says so in machine-readable form.

### The ten review questions

1. **Prediction supported?** L1, L2, L5, L6, L7 held as written. **L3 held on its
   gate and was wrong on its number** — 176.9 ms was 40% of the real 433 ms. **L4
   held only after its first spelling was falsified twice**: once by the float32
   catch, once by the Skeptic proving "not provokable cheaply" false.
2. **Mechanism or only sign?** Mechanism, and the mechanism is now enforced rather
   than observed — which was the entire finding.
3. **Strongest competing explanation?** That the lifecycle repair is cosmetic
   because the defect is one MLP wide. Refuted twice over: the narrowness is an
   accident of `_BASE_POINTS` being a module global, and the setup-abort path the
   Skeptic found is not narrow at all — it is total.
4. **Regressions or tradeoffs?** Default gate 111 s → ~135–160 s; `-Full` ~755 s.
   No estimator change, no seed drawn, no vault touch.
5. **New information?** Five items above, of which #2 and #3 are the ones that
   change what CP1 may claim.
6. **Belief strengthened?** That the per-MLP lifecycle is now faithful — an
   independent context tried to break it and could not.
7. **Belief weakened?** That we know the grader's operating limits. We know four
   configuration fields and λ. Everything else is a package default.
8. **Uncertain?** Whether the real `setup_timeout_s` / `wall_time_limit_s` /
   `residual_wall_time_limit_s` are the defaults — not settleable without a
   submission that deliberately trips one. And the standing E1 residual-transfer
   problem, which this slice does not touch.
9. **Durable invariant or failure mode?** Yes, three, now in `docs/GOTCHAS.md`.
10. **Next?** Terminal state below.

### Terminal state: **PROMOTE_AND_CONTINUE**

Every Skeptic finding is fixed and re-verified in-tree, the self-test went 26/26 →
37/37 with the new checks provoking rather than asserting, and nothing was promoted
into `estimator.py`, no block drawn, no vault touch taken. The evaluator is
additive: only the gate depends on it.

The Skeptic's cheapest discriminating follow-up was a four-item edit list to this
same file. **All four were executed**, plus the eight housekeeping items. The one
thing it asked for that cannot be done from here — verifying the grader's actual
timeout constants — is recorded as unobserved rather than closed.

---

### Provenance for the 2026-08-20 CP0 entries

Code, gate and durable docs committed as **`44634f9`** — *"CP0: repair the
decision layer — the gate, and the grader-lifecycle evaluator"*, 13 files, +1981
/ −323. This LOG entry follows in its own commit so the record cites the change
rather than being part of it.

**Both Skeptics recorded, correctly, that the pre-registration ordering for CP0.1
and CP0.2 was NOT reconstructible from disk**: the predictions and the results
lived in the same uncommitted file, so nothing separated them. That gap is closed
from here on — Samuel authorised per-slice commits on 2026-08-20, and the
standing rule is now: **commit the slice when the predictions are written, before
the implementation exists.** The two entries above remain marked as unverified on
that point, because they are.

*(Also verified while committing: the "uncommitted harness layer" standing risk in
`docs/CURRENT_STATE.md` § Standing risks is STALE. `AGENTS.md`, `ai/`, `.agents/`,
`.codex/`, `scripts/guard_paths.py`, `scripts/verify_portability.py` and the six
adapters are all tracked — they show as modified, not untracked. The risk was real
on 2026-08-10 and was resolved by a later commit without the note being updated.)*

---

# 2026-08-20 — SLICE CONTRACT CP0.3: adopt the ARITHMETIC gate  [PRE-REGISTERED, no data yet]

**STRATEGIC.** Changes the binding promotion statistic — `AGENTS.md` § *Work
classification*. Committed before the implementation exists, and before any
re-scoring is run.

**No seed block is drawn and none is needed.** This slice re-scores artifacts that
already exist on disk; it mints nothing and touches no network. D17 and D19 stay
unspent. Vault counter stays 1.

**Samuel's decision, 2026-08-20:** *"adopt arithmetic."* Given after being shown
that the two estimands select opposite winners on our own graded data.

**Deviation from Step 4, recorded rather than hidden:** Samuel's own numeric
prediction was NOT taken. He directed continuous execution ("proceed until complete
or hard blocked / something requiring my decision"), and a prediction is not a
decision. The protocol's purpose — that no number is chosen after the answer is
visible — is served by the builder's predictions below being committed to git
before the re-scoring runs. The calibration record for this slice is therefore
missing, and that is a real, stated loss.

**Strategic question**
Does binding promotion to the arithmetic mean of per-MLP scores — the competition's
own identity — change any verdict this project has already recorded?

**Current bottleneck**
`docs/VALIDATION.md` §2b binds the geometric mean of per-MLP ratios. The
competition scores `mean_i (mse_i × max(0.1, C_i/B))` over the 50 public MLPs, and
`local_sim/competition_score.py` reproduces that **bitwise** from the graded reports
(2.719427529595e-07 and 2.631432177242e-07, exactly). The gate and the objective
are different quantities, and on our own graded data they pick opposite winners:
arithmetic 1.0334× (v2.4.1 better) against geometric 0.9666× (v2.4 better), same 50
rows, 22/50 wins.

**Evidence**
- CONFIRMED — the two estimands disagree on real graded data (`test_competition_score.py`, 71/71).
- CONFIRMED — interval coverage measured on 192 fresh networks (D18): false-promotion rate against a true zero, nominal 5%: t **4.1–5.4%**, basic **4.0–5.2%**, percentile 6.8–8.2%, BCa **9.1–11.5%**.
- CONFIRMED — CRN makes the paired *difference* nearly symmetric (skew +0.21, −0.55, −0.46) even though each arm is strongly right-skewed. That is why BCa loses here.
- INFERRED — arithmetic has read HIGHER than geometric in all three recorded comparisons where both exist: 1.0334/0.9666 = **1.069**, 1.4775/1.4321 = **1.032**, 1.0961/1.0927 = **1.003**.

**Hypothesis**
The change is correct and low-risk: it aligns the gate with the objective, and it
does not resurrect any lane that was killed on a measurement.

**Mechanism**
Per-MLP scores are strongly right-skewed. The geometric mean of ratios discounts the
heavy networks; the arithmetic mean does not, and neither does the competition. So
the geometric gate answers a question with no prize attached. Where a candidate
helps most in *absolute* terms on the heavy networks, arithmetic will read higher.

**Intervention**
1. Bind arithmetic + studentised-t (or basic bootstrap) in `docs/VALIDATION.md` §2b,
   `AGENTS.md` § *Measurement law*, `ai/workflows/review-slice.md` and
   `slice-contract.md`. Demote geometric to a **stability diagnostic**.
2. Make the gate **executable, not prose**: one `decide()` in
   `local_sim/competition_score.py` that a caller cannot get wrong, returning the
   arithmetic verdict, the t interval, and the geometric number as a diagnostic.
3. **Re-score every recorded verdict that has per-MLP data on disk**, under both
   statistics, and report whether any flips.

**Predictions — builder's, committed before the numbers exist**

- **A1.** The re-scoring pipeline reproduces the D18 carrier's published arithmetic
  gain **1.4775×** at **0.0** relative deviation (it was already computed
  arithmetically), and reproduces the graded aggregate 2.631432177242e-07 bitwise.
  *If A1 fails, the pipeline is wrong and no other number in this slice counts.*
- **A2.** Arithmetic ≥ geometric in **at least 3 of 4** re-scored comparisons,
  by the mechanism above.
- **A3 — the consequential one.** **D14** (Sprint 5 packed lane, SPENT single-use
  block, killed at geometric **1.1216×** against a pre-registered **1.15×** gate)
  re-scores arithmetically into **[1.12, 1.19]**. Applying the median observed
  arith/geo gap of 1.03 to 1.1216 gives **1.155** — *above the gate*. **I therefore
  predict roughly even odds that D14's arithmetic reading crosses 1.15.**
- **A4.** D12 (v2.5, retired at 1.2079×) and the v2.6 gate reconstruction both move
  by less than 5% and neither changes sign.
- **A5.** `preflight -Full` stays **12/12**; portability **174/174**; vault counter
  **1**; no seed block consumed.

**Gate / decision rule, fixed NOW — this is the part that must not move**

Adoption of arithmetic is **not contingent** on the re-scoring; Samuel has decided
it and it is right on first principles. The re-scoring decides something else:
**what we must say about the lanes it touches.**

- If **no** verdict flips → adopt, and record that the change is verdict-neutral.
- If **D14 crosses 1.15** → **the packed lane does NOT automatically reopen.**
  `AGENTS.md` requires **new external evidence** to reopen a closed lane, and
  re-scoring old data under a new statistic is not external evidence. Three further
  reasons the kill stands independently of the statistic: (i) the lane's kill mechanism
  is priced, not statistical — flopscope charges a gather 4 FLOPs/element while a
  random-access gather costs far more wall time than a BLAS multiply-add, and λ
  bills that back; (ii) D13 predicted 1.113% and D14 measured 1.1216×, so cost
  behaviour agreed across two blocks; (iii) D14's verdict also rests on a frozen
  residual model, unchanged here.
  **What MUST change is the wording:** the kill becomes *"failed its gate on the
  binding statistic of the time, and is within noise of that gate under the
  statistic we now use"* — materially weaker, and it goes in `CURRENT_STATE.md`
  § *Closed lanes* in those words.
- If any **other** verdict flips → same treatment, stated per lane.

**Falsifier**
A1 failing falsifies the pipeline. A verdict flipping in the *direction of killing
something we currently believe* would be the serious case — e.g. the D18 carrier's
1.4775× failing to reproduce.

**Baseline**
Each lane's own recorded verdict and tier, as written in `docs/CURRENT_STATE.md`
§ *Closed lanes* and the corresponding LOG entries.

**Residual model**
UNCHANGED, and deliberately so. Each re-scoring uses the residual figures already
stored in that artifact. This slice changes the *statistic*, not the cost model;
changing both at once would make neither attributable.

**Statistic**
Both, side by side, on every comparison: arithmetic difference with the
**studentised-t** interval (primary), geometric mean of per-MLP ratios (diagnostic),
sign test, and the biggest-mover share.

**Confounders**
- Re-scoring uses stored per-MLP arrays; if an artifact's arrays are not the ones
  its headline was computed from, the "reproduction" is vacuous. A1 exists to catch
  exactly that.
- Artifacts differ in what they store (some have `resid`, some `ops`), so a uniform
  cost model cannot be imposed. Report per artifact what was available.

**Scope**
`docs/VALIDATION.md`, `AGENTS.md`, `ai/workflows/review-slice.md`,
`ai/workflows/slice-contract.md`, `local_sim/competition_score.py` (add `decide()`),
`local_sim/test_competition_score.py`, `docs/CURRENT_STATE.md`, `docs/GOTCHAS.md`,
a new re-scoring artifact under `experiments/`.

**Do not change**
Any residual model. Any seed block. `estimator.py`. The three-tier protocol. The
vault. The 0.1 floor, λ, or the failure semantics.

**Evidence required**
`uv run python local_sim/rescore_verdicts.py` (new), its JSON artifact,
`preflight -Full`, `verify_portability.py`.

**Driver**
Claude Opus 5 (1M context), repo-local Builder, session 2026-08-20.

### RESULT — CP0.3, the arithmetic gate  [contract above, unmodified, committed 0eb1279]

```
contract commit   0eb1279  (contract ONLY — no implementation, no numbers)
commands          uv run python local_sim/rescore_verdicts.py
                  uv run python local_sim/test_competition_score.py
                  .\scripts\preflight.ps1 -Full
config / seeds    NO seed block drawn. No network minted. No vault touch.
                  Every number below is a re-aggregation of arrays already on disk.
raw artifact      experiments/2026-08-20_rescore/rescore_verdicts.json
```

#### A1 — reproduction, 5/5. Nothing below counts if these fail

| check | result |
|---|---|
| both graded aggregates reproduce **bitwise** | `2.719427529595e-07` / `2.631432177242e-07` |
| the graded pair reproduces the documented disagreement | arith **1.0334×**, geo **0.9666×** |
| D14's geometric headline reproduces from the stored arrays | **1.1215684126** vs published **1.1215684126** (rel < 1e-9) |
| the D18 carrier reproduces its published arithmetic gain | **1.477523** vs published **1.4775** |

#### The re-scoring, every lane with per-MLP data on disk

| lane | n | **arithmetic** | geometric | arith/geo | wins | top mover |
|---|---|---|---|---|---|---|
| GRADED v2.4 → v2.4.1 | 50 | **1.0334×** | 0.9666× | 1.0692 | 22/50 | **−158.0%** |
| **D14 packed (SPENT, gate 1.15×)** | 100 | **1.1177×** | 1.1216× | **0.9966** | **100/100** | 2.6% |
| D12 packed (SPENT) | 50 | **1.4138×** | 1.3787× | 1.0254 | 39/50 | 9.7% |
| D18 design carrier B→C | 48 | **1.4775×** | 1.4321× | 1.0317 | 33/48 | 14.7% |
| D18 A→B, the N lever | 48 | **1.1980×** | 1.0158× | 1.1794 | 20/48 | **59.9%** |

#### Verdict against the pre-registered decision rule

**NO GATED VERDICT FLIPS. The change is verdict-neutral.** The only lane carrying a
recorded numeric gate is D14, and it fails under both estimands. The contract's
"if D14 crosses 1.15" branch is **not** triggered and no closed-lane wording needs
weakening.

#### A3 — FALSIFIED, and in the direction that matters

I predicted D14 would re-score into **[1.12, 1.19]** and gave *roughly even odds of
crossing the 1.15 gate*, reasoning from a median observed arith/geo gap of 1.03
applied to 1.1216.

It reads **1.1177×** — **below** my predicted range, and **below** the geometric
1.1216×. `arith/geo = 0.9966`. **The packed lane's kill is STRENGTHENED by the new
statistic, not weakened.**

The reasoning error is worth keeping: I generalised an arith/geo *gap* from three
comparisons where the candidate helped most on the heavy networks, to a lane where
it does not. Predicting it from three prior readings was extrapolation dressed as a
mechanism.

**And the real rule is an exact identity, not a tendency.** The arithmetic gain
`mean(ctrl)/mean(cand)` is algebraically the **candidate-score-weighted mean of the
per-MLP ratio**; the geometric mean is the unweighted one. So:

```
sum(cand_i * ratio_i) / sum(cand_i)  =  1.117743  ==  the arithmetic gain, exactly
unweighted geometric mean of ratio_i =  1.121568
corr(cand_i, ratio_i)                = -0.5286    <- the sign that decides which is larger
```

~~**Arithmetic reads above geometric iff the candidate's relative advantage is
POSITIVELY correlated with the score it is being weighted by.**~~ **WITHDRAWN the
same day — the "iff" is FALSE, and the identity above is a tautology.** See the
CP0.3 review below: the correlation is negative on *every* lane and arithmetic
still exceeds geometric on most, because the AM–GM gap of the ratios usually
dominates the covariance term. Positive correlation is **sufficient, never
necessary**. D14 is simply the one lane whose ratios are tight enough (AM–GM gap
1.0001) for the covariance to decide — and it is the lane the rule was
generalised from, which is the same error A3 confesses to two paragraphs above.
What *is* true of D14: the candidate wins on 100/100 networks but wins *least*
where the score is largest — cheapest third mean ratio **1.1292**, priciest third
**1.1133**, `corr = −0.53`.

#### The other predictions

- **A2 HELD** — arithmetic ≥ geometric in **4 of 5** (1.0692, 0.9966, 1.0254, 1.0317, 1.1794).
- **A4 HELD ON HALF ITS CONTENT, and that is a defect.** A4 predicted D12 *and*
  the v2.6 gate reconstruction. Only D12 was checked. D12 moves 2.5% and does not
  change sign — but its artifact carries **no `gate`, `verdicts`, `deciding_model`
  or `passed` key**, so the re-score picks the residual model itself, reproduces
  none of D12's published numbers, gets **no A1 anchor**, and still counted in A2.
  Treat the D12 row as **unanchored**. The v2.6 half is fine on pre-existing P0.1
  evidence (per-op arithmetic 1.0961 vs geometric 1.0927) but CP0.3 did not
  check it.
- **A5** — see the regressions line below.

#### Two diagnostics that earn their place

- **The graded pair's biggest mover carries −158% of the delta.** The screen fires
  exactly as designed: ≥100% means treat as a draw. That is independent confirmation
  of the recorded "no *resolvable* effect" reading, arrived at from a different
  direction.
- **D18 A→B — the N lever — reads arithmetic 1.1980× with a top-mover share of
  59.9%.** The 1.1980× reproduces T2 (LOG 2026-08-19) exactly, so T2 was already
  arithmetic. But **one network out of 48 carries 60% of that delta**, which the
  original entry does not say. ~~The T2 decomposition is unaffected... the bracket
  rests on the MSE·N leg, so it stands.~~ **THAT CONCLUSION IS BACKWARDS AND IS
  WITHDRAWN** — the Skeptic decomposed the legs and the MSE·N divisor is the
  *fragile* one: 174.2% top mover, estimands disagreeing 1.0591 vs 0.8983, t
  interval spanning zero, and a leave-one-out range of [0.9551, 1.1080] that
  crosses 1.0. See the CP0.3 review below. **The N-effect is not resolvable on 48
  networks.** The counted-FLOPs-per-point leg (48/48 wins, p = 0.000, 2.1% mover)
  is solid, so `C/B = 0.6932` at 66,048 points is untouched.

#### What was built

1. **`local_sim/competition_score.py::decide()`** — the gate, executable. It is
   the only function in this repository that returns a promotion boolean. It
   **requires** `gate=` (no default), **raises** on `interval="percentile"` and
   `"bca"`, returns a **frozen** `Verdict`, and reports the geometric gain, the
   estimand-disagreement flag and the biggest-mover share as diagnostics that
   cannot change the outcome.
2. **`local_sim/rescore_verdicts.py`** — the re-scoring, reproducible.
3. **Eleven new gate tests** in `test_competition_score.py`: **71/71 → 82/82**,
   including that a true null does not promote, that the verdict follows arithmetic
   when the estimands disagree, that the gate has no default, and that `Verdict` is
   frozen.
4. The binding change in **seven** governing documents: `docs/VALIDATION.md` §2b-i
   (new, authoritative), `AGENTS.md` § *Measurement law*,
   `ai/workflows/review-slice.md` (promotion rule), `slice-contract.md` (template),
   `next-slice.md` (packet format), `ai/workflows/skeptic.md` (checklist item 1
   rewritten to catch a geometric/percentile/BCa gate), `docs/CURRENT_STATE.md`.

#### Deviations from the contract

One, and it was in the builder's favour to notice: `lane_graded()` was first
written with a hand-rolled report parser that looked for `final_layer_mse` in
`per_mlp`, where it actually lives in `results.public_scores[].scores`. It read
`mse = 0`, produced `log(0)`, and reported an **infinite** arithmetic gain. The
canonical adapter `rows_from_graded_report` existed the whole time. Caught by A1 on
the first run, because A1 was written to check reproduction *before* any headline —
**the third instance in two days of re-implementing an identity that already had a
canonical implementation**, and the first one caught by the slice's own
pre-registered check rather than by a Skeptic.

#### Regressions

`preflight -Full` **GREEN 12/12**. Portability **174/174**. Invariants 30/30.
Scorer **82/82**. Vault counter **1**, log 1 line. D17 and D19 still unspent.
`estimator.py` untouched.

#### Remaining doubt

- The re-scoring covers every lane that **stored per-MLP arrays**. Lanes killed
  before that practice began — Track T, Track P, Sprint 3, the tent/sphere/whitening
  probes — cannot be re-scored, only re-run. None carried a numeric gate close
  enough to its result for the estimand to plausibly matter (0.9998×, 1.093× vs a
  1.25× gate, 1.0246×), but that is an argument, not a measurement.
- Adopting arithmetic makes the gate **more** sensitive to a single network. That is
  correct — the competition is too — but it raises the value of the biggest-mover
  screen, which is now surfaced in every `Verdict` rather than left to the reader.

---

## 2026-08-20 — CP0.3 REVIEW: Skeptic **PROVISIONAL**. The headline held; two of my three supporting claims did not

Third Skeptic, fresh context. It **verified the pre-registration from git** — this is
the first slice in this record where that was possible:

> `git show --stat 0eb1279` = `experiments/LOG.md | 145 +++++` and **nothing else**
> — no implementation, no results. Every number the contract cites is prior
> published evidence. `f8f22d9`, 21 minutes later, carries the code, the artifact
> and the result. **The gate did not move**: contract says 1.15× for D14, result
> says 1.15×.

> **Skeptic verdict: PROVISIONAL**
>
> *"The headline — 'no recorded verdict crosses its gate' — survives everything I
> threw at it. Two of the slice's other three load-bearing claims do not.
> `decide()` **can** be talked into passing a candidate that is worse on every
> single network; and the one downstream recommendation the slice makes (the K129
> N-effect caution) is **backwards on the slice's own diagnostics**."*

### What it falsified

| # | finding | verified? | action |
|---|---|---|---|
| 1 | **`decide()` passes a strictly worse candidate.** `excl = ci[0] > 0 or ci[1] < 0` is two-sided, so a *significant regression* counted as evidence FOR promotion; with `gate < 1.0` nothing else caught it. A candidate 30% worse on all 60 networks returned `passed=True` with an **empty** `reasons` list | **yes**, reproduced | **FIXED, two independent locks.** The interval test is now **one-sided** (`ci[0] > 0`), and `gate` must be **finite and >= 1.0** or `decide()` raises. Verified: gates 0.0, 0.5, −1.0, ±inf, NaN all refuse; at gate 1.0 the 30%-worse candidate fails with three stated reasons including "the candidate is WORSE" |
| 2 | **The "iff" rule I wrote into the LOG is false.** I claimed arithmetic exceeds geometric *iff* `corr(cand, ratio) > 0`. Recomputed on every lane: the correlation is **negative on all of them**, yet arithmetic exceeds geometric on **most** | **yes** — corr −0.5286 / −0.4241 / −0.1852, arith > geo on 2 of those 3 | **WITHDRAWN and replaced** with the correct decomposition, below |
| 3 | **The K129 caution is backwards.** I said the bracket "rests on the MSE·N leg, so it stands". The MSE·N leg is the **more** fragile of the two | **yes**, and it is worse than the Skeptic's summary | **CORRECTED**, below. This is the finding that changes what a durable doc says |
| 4 | **"the only function that returns a promotion boolean" was false three times over**: `packed_sparse_probe.py:2227` (`--mode decide5`, a live CLI path, still gating on the geometric mean with a 1.96·SE-on-log interval); `rescore_verdicts.py:139` (inline `arith >= gate` with **no interval condition at all** — a fourth re-implementation, in the slice that adopted the canonical home); `reconstruct_v26_gate.py:165` (BCa as the **sole** interval on the lane D19 is reserved to gate) | **yes** | **ALL THREE FIXED.** `decide5` now **refuses to run** without `ARC_WHEST_ALLOW_LEGACY_GEOMETRIC_GATE=1`, which exists only to reproduce D14's historical decision and says so; `rescore_verdicts` routes every gated lane through `decide()`; `reconstruct_v26_gate` now records `t_ci_difference` and renames the other to `bca_ci_difference_NOT_GATE_ELIGIBLE` |
| 5 | **A tautological test.** `check("a diagnostic can never flip the verdict", decide(...).passed == (vd.gain >= 1.0 and vd.ci_excludes_zero))` recomputes the implementation's own expression from the Verdict's own fields and **cannot fail for any implementation** | **yes** | **FIXED** — and fixing it exposed that my *replacement* expectation was also wrong (see below) |
| 6 | A4 was reported HELD on half its content (D12 checked, the v2.6 reconstruction not); the D12 row is unanchored — its artifact has no `gate`/`verdicts`/`passed` key, so it gets no A1 check yet counts in A2 | **yes** | **BOTH LABELLED** in the table below rather than quietly carried |

**Tests: 82/82 → 88/88.** The new ones are the leak itself (a 30%-worse candidate at
every legal gate), the gate-floor and non-finite refusals, and an explicit assertion
that the interval condition is one-sided.

### The correct decomposition — replacing the withdrawn "iff"

```
arith = weighted-AM(ratio, weights = candidate score)
      = uAM(ratio) + cov(cand, ratio) / mean(cand)
geo   = uGM(ratio)  <=  uAM(ratio)      by AM-GM, ALWAYS
```

So **positive correlation is SUFFICIENT for `arith > geo`, and never necessary** —
arithmetic also wins whenever the AM–GM gap of the ratios exceeds the (negative)
covariance term. Measured AM–GM gap: **1.0001 on D14** against 1.28 / 1.27 / 1.35 on
the lanes where arithmetic wins. D14 is the *one* lane whose per-MLP ratios are tight
enough for the covariance term to dominate — and it is the lane I generalised the
rule from. **That is structurally the same extrapolation A3 already confessed to,
committed again two paragraphs later.**

The underlying identity is exact (`|wAM − arith|/arith <= 2e-16`) — and the Skeptic
is right that it is exact *because it reduces to `sum(ctrl)/sum(cand)` after
cancellation*. It is a tautology and corroborates nothing. That is the CP0.1 trap —
*could the two sides ever have disagreed?* — for the second time in one day.

### CORRECTION — the K129 N-effect, and it goes the other way

I wrote: *"the bracket rests on the MSE·N leg, not on the combined figure, so it
stands — but it should not be tightened."* **The MSE·N leg is the fragile one.**
Decomposed under the newly binding statistic:

| leg of D18 A→B | arith | geo | t CI on the paired difference | wins | sign p | top mover | leave-one-out gain |
|---|---|---|---|---|---|---|---|
| **MSE·N — the divisor the bracket uses** | **1.0591** | **0.8983** | **SPANS ZERO** | 19/48 | 0.193 | **174.2%** | **[0.9551, 1.1080] — crosses 1.0** |
| counted FLOPs/point | 1.1306 | 1.1308 | excludes zero | **48/48** | 0.000 | 2.1% | [1.1304, 1.1308] |
| combined `V·c` | 1.1980 | 1.0158 | SPANS ZERO | 20/48 | 0.312 | 59.9% | [1.0812, 1.2503] |

The divisor's estimands **disagree** (1.0591 vs 0.8983, undeclared anywhere before
now), its interval **spans zero**, **one network of 48 carries 174% of its delta** —
firing the very ≥100% screen this slice surfaces in every `Verdict` — and **deleting
that one network flips its sign**.

**So the honest statement is not "it stands, don't tighten it". It is: the N-effect
is NOT RESOLVABLE on 48 networks.** The bracket `[1.414, 1.498]` is not a
measurement interval; its low end is exactly `1.4976 / 1.0591`, i.e. the span
between *no correction* and *a correction that is itself a draw*. The corroboration
recorded earlier today — that our measured design gain 1.4775× lies inside that
bracket — therefore **carries less weight than it was given**. It is still true and
still the right direction; it is just not the ±3% agreement it was described as.

**What does NOT change:** the counted-FLOPs-per-point leg is rock solid (48/48 wins,
p = 0.000, 2.1% mover), so **`C/B = 0.6932` at 66,048 points stands unqualified**,
and the affordability conclusion — the single most decision-relevant number in the
K129 lane — is untouched.

### Where the Skeptic and I still cannot settle it

The graded pair's own t interval on the difference **spans zero** (sign p = 0.48).
So the fixture that motivated adopting arithmetic is *itself a draw under the
statistic it motivated*. **That does not weaken the case** — the competition
computes an arithmetic mean and that settles the estimand on first principles, not
on this fixture — but `docs/VALIDATION.md` §2b-i must not read as though a real
disagreement was adjudicated. Amended accordingly.

Resolving the 1.0591 N-effect needs **fresh networks**, and a **DESIGN block will
do** — D16 has 330 free seeds, D18 is already drawn on. **D17 and D19 must not be
touched for it**; neither the arithmetic adoption nor the K129 bracket is worth a
single-use block.

### Terminal state: **PROMOTE_AND_CONTINUE**

The gate change itself is correct, verdict-neutral, and now enforced by a function
that survived a deliberate attempt to talk it into a wrong answer. Every Skeptic
finding is fixed in-tree and re-verified: tests 82/82 → 88/88, `decide()` leak closed
on two independent locks, three stray verdict paths closed, two durable claims
corrected. Nothing was promoted into `estimator.py`, no seed block drawn, D17 and D19
unspent, vault counter 1 of 3.

---

# 2026-08-20 — SLICE CONTRACT CP1.1: does a Kerdock/MUB carrier reach degree 4 at 66,048 points?  [PRE-REGISTERED, no data yet]

**IMPORTANT** — changes a probe's math (`local_sim/carriers.py`). **No seed block,
no network minted, no grader, no vault touch, no `estimator.py` edit.** This is pure
design theory measured against the existing moment harness.

## Strategic question

`docs/CURRENT_STATE.md` closes the design-carrier lane on cost: the verified
strength-5 dual-BCH array is worth a measured **1.4775×** but needs 131,072 points
at `C/B = 1.3509`, over the failure cliff. CP0.3 established that at **66,048**
points the cost line fitted on our own D18 arms gives **`C/B = 0.6932` counted** —
comfortably under the cliff.

So the lane turns on one question, and it needs no networks:

> **Is there a 66,048-point set on 256 dimensions whose degree-4 moment error is
> comparable to the 131,072-point array's `0.0305× iid`?**

If yes, the lane reopens with a measured accuracy prize and an affordable cost.
If no, the K129 premise in `docs/FRONTIER_ARCHITECTURE_PROPOSAL.md` is wrong for
*our* carrier and CP1 dies here, cheaply.

## Why 66,048 is the right number, and why it is not arbitrary

Three independent accountings agree, which is what makes this worth building:

- **K129/MUB:** 256 = 4⁴, so the real-MUB bound `d/2 + 1 = 129` is attainable.
  129 bases × 256 vectors × 2 signs = **66,048** (`test_carriers.py` S5 already
  asserts this arithmetic and nothing more).
- **Kuperberg (arXiv math/0402047):** a degree-5 spherical cubature at `2¹⁶ + 2n`
  points = 65,536 + 512 = **66,048**.
- **Kerdock:** the binary Kerdock code of length 256 has **2¹⁶ = 65,536**
  codewords. It contains the all-ones vector, so its ±1 images form **32,768
  antipodal pairs**; plus the 256 axial directions that is **33,024 distinct
  directions = 129 × 256**, and 66,048 signed.

The three are the same object counted three ways. That is the structure being
built.

## Evidence

- CONFIRMED — the shipped Sobol carrier's degree-4 moment error is **0.510× iid**;
  a Haar frame bank is **exactly** a 3-design and **0.700× iid at k = 4**; the
  dual-BCH strength-5 array is **0.0305× iid at k = 4** and 0.138× at k = 6.
- CONFIRMED — a ±1 cube array is a degree-4 design only *in expectation over
  spread directions*, and the random rotation is what makes that true; balance on
  **all 256** columns is required (`docs/GOTCHAS.md`).
- CONFIRMED — 131,072 is the floor **for linear codes** at width 256, proven by
  exhaustive enumeration of the Sidon-set obstruction. **The Kerdock code is
  NON-LINEAR**, so it is not covered by that proof. This is the stated reopening
  condition, and it is met by construction, not by preference.
- CONFIRMED — `C/B = 0.6932` counted at 66,048 points, from the cost line fitted
  on our own D18 arms A and B (CP0.3; the leg with 48/48 wins and a 2.1% mover).

## Hypothesis

The Kerdock code of length 256, as ±1 directions plus the axial points and under a
random rotation, is a degree-4 design of comparable quality to the dual-BCH array
at **half** the point count.

## Mechanism

Kerdock is the union of 2⁷ = 128 cosets of RM(1,8), whose coset representatives are
quadratic forms whose pairwise differences all have full rank. That is exactly the
"spread" property that makes the 128 bases mutually unbiased, and mutual
unbiasedness at 129 bases is what forces the low-order moments to match the sphere.
It achieves with a *non-linear* code what a linear code provably cannot do below
131,072 rows.

## Intervention

Build the Kerdock code via the **Z₄-linear route** — the Gray image of the
Z₄-linear code over the Galois ring `GR(4, 7)`, which is the construction with an
unambiguous correctness fingerprint — and measure four carriers on the existing
harness:

1. `kerdock` — the 65,536 ±1 vectors alone, randomly rotated;
2. `kerdock+axial` — plus the 512 axial points (Kuperberg's `+2n`);
3. `kerdock unrotated` — to show, as with the BCH array, that the rotation is the mechanism;
4. against the existing `bch OA (rotated)`, `sobol (SHIPPED)`, `haar frames`, `iid`.

## Predictions — builder's, committed before the numbers exist

- **K1 — the correctness fingerprint, and it is unambiguous.** The construction
  yields exactly **65,536 distinct** binary codewords of length 256, closed under
  complement, with the weight distribution supported on exactly
  **{0, 120, 128, 136, 256}** — the Kerdock signature (minimum distance
  `2⁷ − 2³ = 120`). *If K1 fails I have not built Kerdock, and K2/K3 are
  uninterpretable regardless of how good they look.* This is the check that stops
  a pretty number from a wrong object.
- **K2 — the one that decides the lane.** Rotated, the degree-4 moment error is
  **≤ 0.10× iid**, i.e. within ~3× of the BCH array's 0.0305×, at **half** the
  points. **Falsifier: ≥ 0.5× iid** — that is Haar-frame territory (0.700×), it
  would mean the set is no better than a 3-design at degree 4, and the K129
  premise is then wrong for our carrier.
- **K3 — least certain, and the specifically Kuperberg part.** Adding the 512
  axial points **improves** degree 4 over the sign vectors alone. I hold this at
  about 50/50: the axial points are 0.78% of the set and Kuperberg's theorem
  weights them, whereas this build will use them unweighted at first.
- **K4.** Degree 6 stays **well above** degree 4 in relative error — a degree-5
  cubature is exact to 5, not 6 — but lands **below iid**, i.e. < 1.0×.
- **K5.** `preflight -Full` stays **GREEN 12/12**; no seed block; vault counter 1.

## Gate

This slice has no promotion gate — it decides whether CP1 continues.

- **degree-4 ≤ 0.10× iid → GO**, and the next slice prices it end to end on fresh
  networks from a DESIGN block.
- **0.10× < degree-4 < 0.5× iid → PIVOT**, and the successor is named in advance:
  weight the axial points as Kuperberg's theorem prescribes rather than using them
  raw.
- **degree-4 ≥ 0.5× iid → KILL** the K129 carrier route for width 256, and record
  that the published K129 entry's advantage must then come from something other
  than degree-4 carrier strength — which would itself be a finding worth having,
  because it would redirect CP1 at its own cost engineering instead.

## Falsifier

K1 failing falsifies the construction, not the hypothesis. K2 above 0.5× falsifies
the hypothesis.

## Baseline

`bch OA (rotated)` at 131,072 points: degree-4 **0.0305× iid**. `sobol (SHIPPED)`:
**0.510×**. `haar frames`: **0.700×**. All from `test_carriers.py`, 2026-08-19.

## Residual model

Not applicable — no cost claim is made or changed by this slice. The `C/B = 0.6932`
figure is carried in from CP0.3 unmodified.

## Statistic

RMS relative deviation from the exact spherical moment over ≥800 random probe
directions, per degree k = 1..6, and the ratio to iid at the same point count.
Reported as a ratio, never an absolute.

## Confounders

- **Point count is not held equal across carriers**, and cannot be — an orthogonal
  array has one size. Every row is reported with its `n`, and iid is measured at
  the *same* n as the carrier it normalises.
- The probe directions are random; a lucky draw could flatter one carrier. Use the
  same fixed probe seed for every carrier so the comparison is paired.

## Scope

`local_sim/carriers.py` (new constructions + registry), `local_sim/test_carriers.py`
(the fingerprint checks), an artifact under `experiments/2026-08-20_kerdock/`.

## Do not change

`estimator.py`. Any residual model. Any seed block. The vault. The existing
carriers' definitions — `bch_oa` and friends must keep producing byte-identical
point sets so the 2026-08-19 numbers stay reproducible.

## Driver

Claude Opus 5 (1M context), repo-local Builder, session 2026-08-20.

### RESULT — CP1.1, the Kerdock/MUB carrier  [contract above, unmodified, committed b2a92a2]

```
contract commit   b2a92a2   (contract ONLY -- no implementation, no numbers)
commands          uv run python local_sim/test_carriers.py      (27/27 -> 42/42)
                  .\scripts\preflight.ps1 -Full
config / seeds    NO seed block. NO network minted. NO grader. NO vault touch.
                  Pure design theory against the existing moment harness.
```

## The result, in one line

**An EXACT spherical 4-design at 66,048 points on 256 dimensions, at
`C/B = 0.6934` counted — under the failure cliff.**

*(Corrected from “5-design” and “0.6932” after the Skeptic: degrees 1/3/5 vanish for
ANY antipodally closed set, so only 2 and 4 are earned; and the cost figure must
carry the carrier’s own rotation — free only if it is folded into `W₁`.)*

The lane `docs/CURRENT_STATE.md` closed on cost is open again, and the object is
better than the one that was too expensive, not merely cheaper.

## Every prediction, against what was committed before the data existed

| # | prediction | result | verdict |
|---|---|---|---|
| **K1** | 65,536 distinct codewords, length 256, closed under complement, weights supported on exactly {0, 120, 128, 136, 256} | **all four exact.** Counts `{0:1, 120:32512, 128:510, 136:32512, 256:1}` — the textbook Kerdock distribution | **HELD** |
| **K2** | rotated Kerdock degree-4 ≤ 0.10× iid (GO); ≥ 0.5× would KILL the lane | **0.0572× iid** at 65,536 points | **HELD → GO** |
| **K3** | the 512 axial points *improve* degree 4. Held at ~50/50 | **they make it EXACT** — 6.971e-15 absolute, machine precision. Not an improvement, an identity | **HELD, far beyond the prediction** |
| **K4** | degree 6 below iid but not exact | **0.3606× iid** | **HELD** |
| **K5** | preflight GREEN, no seed, vault 1 | see regressions | **HELD** |

## The measured profile — ratio to iid at the same point count

| carrier | n | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 |
|---|---|---|---|---|---|---|---|
| iid | 65,536 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| sobol (SHIPPED) | 65,536 | 0.0149 | 0.1221 | 0.5485 | **0.5286** | 0.7830 | 0.7839 |
| haar frames | 65,536 | 0.0000 | 0.0000 | 0.0000 | 0.6917 | 0.0000 | 1.0641 |
| kerdock (rotated) | 65,536 | 0.0000 | 0.0000 | 0.0000 | **0.0572** | 0.0000 | 0.3792 |
| **kerdock + axial** | **66,048** | 0.0000 | 0.0000 | 0.0000 | **0.0000** | 0.0000 | **0.3606** |
| bch OA (rotated) | 131,072 | 0.0000 | 0.0000 | 0.0000 | **0.0572** | 0.0000 | 0.2520 |

Degrees 1–5 of the axial-corrected design read
`2.5e-16 / 5.6e-15 / 8.2e-16 / 7.0e-15 / 3.5e-15` — machine precision throughout.

## Why it works, and it is an identity rather than a measurement

For a unit direction `a` and a cube point `u = s/√d`, with `S = Σaᵢ⁴`:

```
sum over the 2^16 sign vectors of (a.u)^4  =  2^16 * (3 - 2S) / d^2
sum over the 2d  axial points   of (a.u)^4  =  2 * S
------------------------------------------------------------------
total / (2^16 + 2d)  =  [(3 - 2S) + 2S] / (2^16 + 2d)  =  3 / (d(d+2))
```

**The 512 axial points contribute precisely the `2S` the cube array is short
by — for every `a`, at uniform weights, with nothing to tune.** Verified to
**8.95e-16** over 24 random directions, and the first line verified separately, so
the two halves are checked independently rather than only their sum.

**Kerdock alone reads 7.2321e-04 at degree 4 and the BCH array reads 7.2321e-04 —
identical to five significant figures, at half the points.** That is not a
coincidence and it is not a tie on quality: both sets satisfy `(3−2S)/d²`
*exactly*, so their entire measured degree-4 "error" is the variation of `S` across
the probe directions. Same probe seed, same number. The residual is a property of
the probe, not of either array.

## The finding that reopens a lane closed with a proof

`docs/CURRENT_STATE.md` § *Closed lanes* kills the subspace route:

> *"The random rotation IS the mechanism: it turns every response direction into a
> spread one, which is the only regime where a ±1 cube array is exact — and that
> requires balance on all 256 columns. You cannot have the cheap array and the
> rotation at once."*

That is true of a **bare** cube array, whose degree-4 accuracy needs `S ≈ 3/258`
and which measured **5.15% low** confined to 32 dimensions. The axial correction
removes the `S`-dependence *identically*, so confinement cannot degrade it.
Measured, **unrotated**, at three confinement levels:

| directions confined to | 256 dims | 64 dims | 32 dims |
|---|---|---|---|
| relative degree-4 error | 2.58e-16 | 2.11e-16 | 2.84e-16 |

**Exact at every level.** This does not contradict the closed-lane finding — it
satisfies its stated reopening condition. The 131,072 floor is proven **for linear
codes** via the Sidon-set obstruction; **Kerdock is not linear**, so the proof does
not reach it.

## Deployability — checked, because CP0.2 made the ceilings visible

| constraint | measured | limit |
|---|---|---|
| construction time | **0.11 s** | 5,000 ms setup (package default, not observed) |
| the whole construction as a constant | **32 bytes** — one length-127 Z₄ sequence (254 bits). *The “224 bytes / 108 base64 chars” first written here was three inconsistent numbers: 224 B is the UNREDUCED 7×128 table and 108 chars is neither. Corrected 2026-08-20.*, since `Tr(xʲ·xᵗ) = Tr(x^(j+t))` makes the 7×128 table shifts of one row | the shipped Sobol blob is ~19,000 chars |
| point set memory | 135 MB float64 / **68 MB float32** | RLIMIT_AS 65,536 MB |
| counted cost at 66,048 points | **`C/B = 0.6932`** | failure cliff at 1.0 |

The construction is derived in `carriers.py` from the Hensel lift of `1+x³+x⁷`,
not embedded as a magic blob, and reproduces the independent scratch build
bit-for-bit.

## What this does NOT establish, and it is the important half

- **No end-to-end MSE gain has been measured.** The 1.4775× on record is
  *bchOA vs sobol at matched N = 131,072*. This is a different design at half the
  points. Whether the end-to-end gain is larger, equal or smaller is **unmeasured**
  and needs fresh networks — the next slice, on a **DESIGN** block.
- ~~Degree 6 … the BCH array is actually *better* (0.2520×).~~ **ARTEFACT, withdrawn.**
  0.2520× divided a 131,072-point array by iid at 65,536. At matched n it is
  **0.3429× vs 0.3603× — a 5% gap inside a ±3.5% denominator noise**, and scaling
  the absolute errors to equal n gives **1.016**. The two designs are
  **equivalent** at degree 6. Neither is exact there.
- `C/B = 0.6932` is carried in from CP0.3's cost line, fitted on the Sobol arms.
  **A Kerdock carrier's per-point propagation cost is assumed equal to Sobol's and
  has not been measured.** It should be — the point set is generated differently,
  even though propagation through the network is identical once the points exist.
- The 5-second setup ceiling and the 65,536 MB memory limit are **package
  defaults, never observed on the grader** (CP0.2).

## Gate comparison

Contract gate: degree-4 **≤ 0.10× iid → GO**. Measured **0.0572×** for the sign
set and **0.0000×** with the axial correction. **GO**, by a wide margin, and the
PIVOT and KILL branches are not reached.

## Regressions

`test_carriers.py` **27/27 → 42/42**. Suite time 75 s → ~105 s. No estimator
change, no seed block, no vault touch, D17 and D19 unspent.

---

## 2026-08-20 — CP1.1 REVIEW: independent Skeptic **ACCEPT**, with five corrections and one inherited bug

Fourth Skeptic, fresh context. **The first ACCEPT of this session.** It verified the
pre-registration from git — `b2a92a2` is `experiments/LOG.md` alone, no
implementation and no numbers, and the gate (`degree-4 ≤ 0.10× iid → GO`) is
byte-identical in contract and result.

> *"The mathematics survives. Every counter-hypothesis I built to break it failed,
> and one of them (the 'axial points are the whole mechanism') is decisively
> refuted. But five claims in the durable record are wrong or unsupported, and one
> of them is a pre-existing `docs/CURRENT_STATE.md` error this slice inherits and
> propagates."*

### The inherited bug, and it is the most valuable thing here

**`test_carriers.py` divided every carrier by iid at N = 16,384 — including the
131,072-row BCH array.** iid's own moment error falls as `1/√n`, so any carrier
measured at more points was flattered by `√(n/16384)`. The CP1.1 contract asserted
under § *Confounders* that "iid is measured at the same n as the carrier it
normalises". **It was not.** I asserted a property of the harness without checking
it — the exact failure this repository has now recorded six times.

| figure | as recorded | matched-n truth |
|---|---|---|
| BCH array, degree 4 | **0.0305× iid** | **0.0756–0.0908×** |
| BCH array, degree 6 | **0.138× iid** | **0.343–0.421×** |
| "17× better than the shipped carrier" | 17× | **≈ 5.8×** |
| *(CP1.1's own table)* BCH degree 6 | 0.2520× | 0.3429× |

The ranges are two independent realisations, because **the iid denominator itself
carries ±3.5% per draw** (6 independent draws at each n). **Fixed at source**: an
iid baseline is now built for every distinct point count, in both
`t_design_strength` and `t_kerdock`. `docs/CURRENT_STATE.md` § *Open questions* 2b
is corrected in place with the reason.

**Nothing that was decided changes.** The array's *absolute* degree-4 error
(7.2321e-04) never moved, and the end-to-end **1.4775×** was measured against a
Sobol arm, not against iid.

**And my own concession was an artefact too.** I wrote that "the BCH array is
actually better at degree 6 (0.2520× vs 0.3606×)". At matched n it is 0.3429× vs
0.3603× — a **5%** gap inside a ±3.5% denominator noise. Scaling the absolute
errors to equal n by the `1/√n` law gives **1.016**: at degree 6 the two designs
are **equivalent**. My first replacement check asserted Kerdock *beats* BCH there
and **failed**, correctly — the code was right and the new claim was wrong, for the
second time today.

### The other four corrections

| # | finding | verified | action |
|---|---|---|---|
| 2 | **`C/B = 0.6932` omits the carrier's own rotation.** Rotating 66,048 points by a 256×256 matrix is real counted cost the CP0.3 Sobol line never saw | **yes — and it is TWICE the Skeptic's estimate.** Measured under real flopscope: **1.728e10 FLOPs = +0.0635 of B**, not the +0.0318 estimated | **FIXED, and the fix is free.** `W₁ᵀ(Qu) = (QᵀW₁)ᵀu` — rotate the 256×256 weight, not 66,048 points. **258× cheaper** (6.70e7 FLOPs, +0.00025 of B), verified equivalent to **1.705e-13**. Corrected figures: **0.75669** as the code does it today, **0.69341** as a deployed estimator would. Both under the cliff |
| 3 | **The pre-registered artifact `experiments/2026-08-20_kerdock/` was never written**, and "reproduces an independent scratch build bit-for-bit" had no artifact to check | **yes** | **FIXED.** `local_sim/kerdock_artifact.py` emits it, regenerably, with the fingerprint, the exact Delsarte transform, every arm at matched n, and the cost breakdown |
| 4 | **"224 bytes (108 base64 chars) — one length-127 Z₄ sequence" is three inconsistent numbers.** A length-127 Z₄ sequence is 254 bits = **32 bytes**; 224 bytes is the unreduced 7×128 table; 108 base64 chars is neither | **yes** | **CORRECTED** to 32 bytes for the reduced sequence |
| 5 | **"Reopens a lane closed with a proof" overstates it.** `CURRENT_STATE` § *Next actions* #10 already names *"a non-linear array or spherical 4-design near the 33,152 floor"* as the remaining open route | **yes** | **RE-FRAMED.** CP1.1 **executes the sanctioned route**; it does not overturn a closed verdict. The confinement result is true and buys **zero cost** — the subspace route was about spending *fewer* points, and this design spends 66,048 either way |
| 6 | **"5-design" is degree-4 plus three free odd degrees.** Degrees 1, 3, 5 vanish for *any* antipodally closed set — the suite's own S4 asserts exactly that, and Haar frames read 0.0000 there too | **yes** | **CORRECTED** everywhere to **exact spherical 4-design**; degree 5 is free from the symmetry and is not evidence |
| 7 | Check-count inflation: `2^16 + 2d == 129*256*2 == 66048` is arithmetic on literals and duplicates the pre-existing S5 | **yes** | **REMOVED** rather than counted |

### What the Skeptic proved that I had only argued

Two findings strengthen the result beyond what the slice claimed, and both are now
permanent checks:

1. **Strength 5 is PROVEN, not sampled.** The exact integer Delsarte transform of
   the weight enumerator gives **`B₁ = B₂ = B₃ = B₄ = B₅ = 0`** and
   **`B₆ = 11,606,784`**. Whether some other inequivalent code shares the
   fingerprint is now irrelevant: the fingerprint *implies* the only property the
   identity needs.
2. **The "cheaper sign set would do" hypothesis is dead, by counter-example.**
   Measured at one probe:

   | set | n | degree-4 rel err |
   |---|---|---|
   | **Kerdock + axial** | **66,048** | **3.11e-16** |
   | random antipodal signs + axial | 66,048 | 1.93e-02 |
   | **strength-5 BCH array of the WRONG SIZE + axial** | 131,584 | 3.94e-04 |
   | axial points alone | 512 | 1.01e-01 |

   **Both conditions are load-bearing**: strength ≥ 4 *and* exactly `d² = 65,536`
   points, because the `S`-terms cancel only at that count. I had attributed the
   mechanism to strength alone.

Also noted and accepted: exactness-without-rotation is **degree-4 only** —
unrotated along a coordinate axis the degree-6 moment is ~34× the spherical value,
so the rotation still does work above degree 4.

### The caveat that most affects what happens next

> *"The BCH array measured 1.4775× against the repo's own externally derived
> ceiling of ~1.46×, while carrying a degree-4 error of 0.0756× iid. Kerdock+axial
> removes that residue — but if BCH is already at the ceiling, degree-4 exactness
> has ~no room left, and the entire value of CP1.1 is half the points, not a better
> design."*

**Accepted, and it re-frames the prize.** The commit message's "better AND half the
size" is **half right**. The bankable claim is **the same accuracy at half the
cost**, which is what moves `C/B` from 1.3509 (a failure) to 0.6934 (affordable) —
and that alone is the whole lane. Any *additional* gain from degree-4 exactness is
speculative until measured end to end.

### Terminal state: **PROMOTE_AND_CONTINUE**

Skeptic ACCEPT. All seven findings fixed in-tree and re-verified; the suite went
27/27 → **48/48** including six new checks that pin the mechanism by
counter-example and one that proves the strength exactly. The inherited harness bug
is fixed at source and its consequence corrected in `docs/CURRENT_STATE.md`.

Nothing promoted into `estimator.py`, no seed block drawn, D17 and D19 unspent,
vault 1 of 3.

**Next, in the Skeptic's order and mine:** the end-to-end arm on a **DESIGN** block
— three arms on fresh networks, `sobol@16,384` (the shipped operating point, *not*
sobol@66,048, because the question is whether the whole move beats what we ship),
`kerdock+axial@66,048` rotated, and unrotated. Gated through
`competition_score.decide()` on the arithmetic statistic with a studentised-t
interval, gate pre-registered, GT-noise floor quoted. **The contract must
pre-register the `MSE·N` decomposition rather than discover it**, because
`docs/GOTCHAS.md` already records that law as not resolvable on 48 networks.


---

# 2026-08-21 — **THE PHASE-2 CONFIGURATION LANDED. THE SHAPE MOVED: 1024 x 16.**

*Polled 2026-08-21. This supersedes every shape-dependent plan in this file and in
`docs/CURRENT_STATE.md`. No slice, no seed block, no vault touch -- a read of the
organisers' published dataset.*

## What is now known, from the dataset itself

The HF dataset `aicrowd/arc-whestbench-public-2026` has gained a **`v2-phase2`**
tag (it carried only `v1-phase1` / `v1-warmup` this morning). Its `metadata.json`:

| field | Phase 1 | **Phase 2** |
|---|---|---|
| width | 256 | **1024** |
| depth | 32 | **16** |
| ground-truth samples | 1e9 | 1e9 |
| splits | mini 100 / full 1000 | mini 100 / full 1000 |
| baked with | whestbench 0.14.0 | **whestbench 0.16.0, flopscope 0.12.0** |
| seed protocol | 3.0 | 3.0 (unchanged) |

Round 1429 opens **2026-08-21T23:59Z**, leaderboard 1480 still has 0 entries.

## What this breaks, and it is a great deal

**Samuel's working assumption of 2026-08-20 -- "Phase 1 with different MLPs, maybe
a little harder" -- is FALSIFIED.** The shape moved on both axes, in opposite
directions. It was recorded with its falsifiers, and the first one listed ("a width
that is not 256") is exactly what happened.

1. **The CP1.1 Kerdock/MUB-129 carrier does NOT transfer.** It is a width-256
   object and the whole construction depends on it: `256 = 4^4` is what makes the
   real-MUB bound `d/2 + 1 = 129` attainable, `2^16 = d^2` is what makes the axial
   identity cancel exactly, and the Galois ring is `GR(4,7)` because `2^(7+1) = 256`.
   At **1024 = 4^5** the analogue is `GR(4,9)`, `d/2 + 1 = 513` bases, and
   `d^2 = 2^20 = 1,048,576` sign vectors + 2,048 axial = **1,050,624 points**.
   **The mathematics carries over; the specific object does not.** Whether the
   width-1024 version is affordable is a new and open cost question.
2. **`_PRUNE_TAU = -2.5` and `_BASIS_RANK = 32` are fitted at 256x32 and neither
   transfers.** This was already recorded as the first action when 1429 went live.
3. **The dual-BCH strength-5 array, the 131,072 linear floor, and the Sidon-set
   proof are all width-256 statements.** None of them says anything about 1024.
4. **flopscope moved 0.10.0 -> 0.12.0 and whestbench 0.14.0 -> 0.16.0.** The
   installed toolchain is behind. `docs/CURRENT_STATE.md` flags the price table as
   the cost half's live unknown; two minor versions of flopscope is exactly where
   that would move.

## What this OPENS, and it was pre-registered

`docs/CURRENT_STATE.md` § *Open questions* #1 says, written before any of this was
known:

> *"If `L/n` falls materially -- a wider or shallower Phase 2 -- the mechanistic
> lane reopens wholesale, because ARC's own cumulant propagation breaks even only
> when `L^K < n` (L < 16 for K=2, L < 6.3 for K=3). At 256x32 we sit at L/n = 1/8
> and every variant of it is either over budget or worse than MC."*

**`L/n` fell from 32/256 = 1/8 to 16/1024 = 1/64 -- a factor of eight.** And the
break-even condition is now met at K = 2: `L^2 = 256 < 1024 = n`. At K = 3,
`L^3 = 4096 > 1024`, so it is not met.

**The moonshot / cumulant lane reopens on its own pre-registered condition, by an
external fact rather than by a new idea about old evidence.** That is exactly the
reopening standard `AGENTS.md` demands.

## What is still unknown

**The budget `B`, lambda, and the multiplier floor are NOT in the dataset.** They
live in the round configuration and appear in a graded report's `config_snapshot`.
Until a Phase-2 run exists we do not know `B`, and **every `C/B` in this repository
is a Phase-1 number**. Also unknown: whether the networks still have no biases
(the dataset carries weights, so this is checkable once the toolchain is upgraded).

**Naive scaling, stated as arithmetic and not as a claim:** a dense layer at
1024 wide costs 16x a 256-wide layer per sample, over half the depth, so
**per-sample propagation is ~8x Phase 1**. If `B` did not grow by at least that,
the affordable sample count collapses. This is the first thing to establish.

## Immediate consequences for the plan

- **CP1.2 (the end-to-end Kerdock arm on D18) is SUSPENDED**, not killed. D18's 192
  networks are 256x32 and would measure a carrier for a shape that is no longer
  scored. **Spending compute on it now would be measuring the wrong world.**
- **Nothing that was built is wasted as method.** The instrument work (CP0.1-0.3:
  the 12-step gate, the lifecycle evaluator, the arithmetic gate) is shape-free and
  applies unchanged. The Kerdock derivation generalises to `GR(4,9)`.
- **The first Phase-2 actions are, in order:** upgrade the toolchain to
  whestbench 0.16.0 / flopscope 0.12.0 and re-verify mint provenance at 1024x16;
  establish `B` and the price table; re-fit `_PRUNE_TAU` and `_BASIS_RANK` on a
  DESIGN block at the new shape; only then re-open the carrier question.
- **D17 and D19 remain unspent, and must stay that way** until the new shape's
  constants are re-fitted. A single-use block spent on a stale shape is wasted.

---

# 2026-08-21 — SESSION CLOSE

**Driver:** Claude Opus 5 (1M context), repo-local Builder.
**Everything committed and pushed.** Working tree clean.

## What this session did

| slice | class | outcome |
|---|---|---|
| **CP0 reconciliation** | read-only | ten proposed validation blockers classified against the code with file/line evidence: 2 partially fixed, 8 open |
| **CP0.1** the gate | IMPORTANT | preflight **6 → 12 checks**, two tiers, denominator always 12. Skeptic **PROVISIONAL**, 3 real defects in the repair, all fixed |
| **CP0.2** lifecycle evaluator | IMPORTANT | `local_sim/lifecycle_eval.py`, **37/37**. Skeptic **PROVISIONAL**; its lead finding invalidated the slice's own licence. 12 findings fixed |
| **CP0.3** the arithmetic gate | **STRATEGIC** | Samuel's decision. `decide()` is now the only promotion boolean in the repo. Every recorded verdict re-scored; **none flipped**. Skeptic **PROVISIONAL**, incl. a leak that passed a strictly-worse candidate |
| **CP1.1** the Kerdock carrier | IMPORTANT | an **exact spherical 4-design at 66,048 points**, strength 5 proven by exact Delsarte transform. Skeptic **ACCEPT** — the session's only one |
| **Phase-2 poll** | read-only | **the shape moved to 1024 × 16**, which obsoletes CP1.1 |

## The one thing that changed a belief

**Phase 2 is 1024 wide × 16 deep.** Found by polling the HF dataset tags, where a
`v2-phase2` tag appeared the day the round opened. It falsified Samuel's recorded
working assumption on that assumption's own first-listed falsifier, killed the
CP1.1 carrier as an object, and **reopened the cumulant lane on its own
pre-registered condition** (`L/n` 1/8 → 1/64; `L² = 256 < 1024`).

## State

- **Working tree clean**, all pushed. Commits this session: `44634f9`, `81c01d8`,
  `0eb1279` (contract only), `f8f22d9`, `fd3fdf2`, `b2a92a2` (contract only),
  `2856e7d`, `103ce4c`, `860d249`, and this close.
- **preflight GREEN 10/12** default (2 skipped, named); **12/12 under `-Full`**,
  last full run 789 s. **Portability 174/174.**
- **Vault: 1 of 3 touches**, counter and log agree, `manifest.json` mtime still
  2026-08-06. **D17 and D19 single-use and unspent.** No block drawn all session.
- **`estimator.py` untouched all session** — still v2.6.1, byte-identical to its
  archive.
- **Nothing submitted.**
- **`docs/HUMAN_MODEL.md` is STALE** (12 commits behind) and was never written by
  an agent. `session-start` will open in FIRST-ORIENTATION MODE next time. **That
  is correct behaviour, not a defect** — and the shape change makes it a good
  moment for Samuel to revisit it.

## Open, and what is in the way

- **`B`, λ, the multiplier floor and the bias question are unknown**, and every
  `C/B` in the repository is a Phase-1 number. This blocks all costing.
- **The toolchain is behind**: whestbench 0.14.0 / flopscope 0.10.0 against the
  dataset's 0.16.0 / 0.12.0. Blocks loading the Phase-2 data at all.
- **Five CP0 audit blockers remain open**: 4 (ledger scan gaps), 6 (a vault touch
  is not transactional), 7 (residual model fits aggregate means), 9 (31-bit seed
  masking), 10 (single truth replica). None blocks the next step.
- **CP1.2 is SUSPENDED, not killed.** No block was spent on it.
- **No slice is mid-flight and no slice is finished-but-unreviewed.** All four
  IMPORTANT/STRATEGIC slices completed their `review-slice` with an independent
  Skeptic.

## Next session

`session-start`, then `docs/CURRENT_STATE.md` § *Next actions*, **in order**:
upgrade the toolchain and re-prove mint provenance at 1024 × 16 → establish `B`
and the price table → re-fit `_PRUNE_TAU` and `_BASIS_RANK` on a **new** DESIGN
block at the new shape. Only then re-open the carrier and estimator questions.

**The honest position on the score: it has not moved.** The graded number is still
**2.6314e-7** (submission 327743, 2026-08-10). This session bought a trustworthy
decision layer and one proved design that the shape change then stranded. It did
not buy a better score, and there is nothing today that would rank higher —
Phase 1 is closed and Phase 2 had not opened when this session ended.

---

# 2026-08-21 — P2-CP0 RECONCILIATION: the official Phase-2 state, verified rather than assumed

*Read-only. No slice, no seed block, no vault touch, no estimator edit. Every
line below was fetched from an official source at execution time and hashed into
`experiments/2026-08-21_phase2_manifest/phase2_manifest.json`.*

## What is now VERIFIED from official sources

| question | answer | source |
|---|---|---|
| is round 1429 LIVE? | **NO — `active=false`, state `UPCOMING`**, opens 2026-08-21T23:59:00Z, closes 2026-09-19T23:59:00Z. Leaderboard 1480 exists with **0 entries** | AIcrowd challenge page, polled 2026-08-21 |
| dataset revision | **`v2-phase2`**, commit **`ce926b7d1ced65faa668c01b94a43f8c4f43783d`** (also `main`) | HF refs API |
| shape | **1024 x 16**, no biases, no separate output layer, `h_l = max(0, h_{l-1} @ W_l)`, `W ~ N(0, 2/width)` iid, `x ~ N(0, I)` | `metadata.json` + dataset card |
| truth | **N = 1e9** per MLP, float64 accumulation, downcast float32. **`avg_variance` is a published column**, so `gt_mse = avg_variance / N` is EXACT for the public split | dataset card |
| splits | mini **100**, full **1000**, disjoint seeds, 1100 partials | `metadata.json` |
| budget | **B = 2^41 = 2,199,023,255,552 FLOPs/MLP**; forward pass **33,554,432 FLOPs/sample**; **65,536** naive samples exactly | dataset card, Versions table |
| multiplier floor | **0.1**, stated on the card as `mse_final · max(0.1, C_m / B_m)` | dataset card |
| bake stack | whestbench **0.16.0** / flopscope **0.12.0** (provenance) **and** `*_version_reported` **0.14.0 / 0.10.0** (legacy). Both preserved in the manifest; neither chosen | `metadata.hardware_fingerprints[*]` |
| **whestbench 0.16 installable?** | **NO.** PyPI latest is **0.15.0**; GitHub's newest tag is `v0.15.0` (2026-08-17). 0.16.0 exists only inside the organisers' bake | PyPI JSON API, GitHub tags API |
| flopscope latest | **0.12.0**, released **2026-08-21T05:42Z** — today, hours before the round opens | PyPI + GitHub releases |
| nomination rule | **STILL UNCLARIFIED.** Announcement 18143 says "up to 2 submissions per team"; the Rules tab says one; `inverted_donkey` raised the contradiction on 2026-08-10 and **there is no staff reply in that thread** | discourse thread 18143 |
| new organiser announcement changing the evaluator contract | **NONE.** Newest staff post is 18186 (2026-08-18): *"still ironing out some details ... the deadline for Phase 2 submissions will also be shifted"*. Nothing since | discourse `latest.json`, 60 topics |

## The parity blocker, and it is doubly blocked

**Exact package parity with the Phase-2 bake stack is BLOCKED, for two independent
reasons**, both detected mechanically by `local_sim/phase2_manifest.py`:

1. `whestbench 0.16.0` is not published on PyPI (latest **0.15.0**) and has no
   GitHub tag. It is not installable from any public source.
2. Even setting (1) aside, **public `whestbench 0.15.0` pins
   `flopscope>=0.11.0,<0.12.0`** — which *excludes* the bake's flopscope 0.12.0.
   **No public pair of these two packages reproduces the bake stack at all.**

That second reason is the informative one: 0.15.0 cannot be the Phase-2 scoring
release, because it refuses the flopscope the Phase-2 data was baked with. The
unreleased 0.16.0 is the release that pairs with flopscope 0.12, and flopscope
0.12 shipped this morning. **Expect whestbench 0.16.0 to become public when round
1429 opens; re-run parity then.**

Until it does, every artifact carries `package_parity.limitation_stamp`: *counted
FLOPs measured under a different flopscope release are not the grader's counted
FLOPs.*

## flopscope 0.12.0 changes prices, and one of the changes touches K2 directly

From the 0.12.0 release notes (fetched 2026-08-21): *"It ships as a new version
opening a new phase, so no submission is re-evaluated against it."* Repricings
that matter here:

- **`is_symmetric` / `as_symmetric` now bill `allclose`'s cost formula**, and
  every affected dtype cell moves by exactly **2.00x**. Our covariance path uses
  the symmetry tag; a full-covariance K2 spelling would use it on every layer.
- `nan`-prefixed reductions cost one extra input-sized pass on float input.
- Several narrow-dtype and refused-op paths got *cheaper*.
- **Dense matmul pricing did not move**: the card states the forward-pass count is
  *"exact, and it does not drift between flopscope releases — verified to the unit
  against flopscope 0.4.1 ... as well as 0.11.0 and current main."*

**Nothing here is a measured cost.** These are release notes; the numbers have to
come from running flopscope 0.12.

## Two hard local constraints, recorded because they bound the plan

- **The Phase-2 `full` split is 69.96 GB across 63 parquet shards. This machine
  has 39 GB free.** `full` cannot be held here. CONFIRM-tier work on Phase 2 is
  limited to **mini (7.03 GB, 7 shards)** unless disk is freed. One shard is
  1.119 GB and holds ~16 MLPs — enough for a provenance check.
- The GPU venv has torch 2.13.0+cu130 on an **RTX 5090, 34.2 GB**. Fresh-mint
  truth at 1024x16 is therefore affordable: 2^24 samples x 33,554,432 FLOPs
  is about 5.6e14 FLOPs per network.

## What is still unknown after all of this

`lambda` (the residual price), `setup_timeout_s`, `wall_time_limit_s`,
`predict_timeout_s`, `memory_limit_mb`, and **how many public MLPs are scored**
(Phase 1 scored 50 of 100). None is in the dataset. Only a graded Phase-2 report's
`config_snapshot` + `runtime_environment` + per-MLP telemetry settles them, and
that costs a submission. **Say "the package default is", never "the grader
allows".**

## Correction to a durable claim

The 2026-08-21 entry above ("What is still unknown") and `docs/CURRENT_STATE.md`
both said **"the budget `B`, lambda and the multiplier floor are NOT in the
dataset"**. That is **wrong for two of the three**: the dataset *card* publishes
`B = 2^41` in its Versions table and the floor `0.1` in its score-formula prose,
and it publishes the no-bias / no-output-layer contract explicitly. Only
**lambda** is genuinely unknown. The error came from reading `metadata.json` and
not the card at the same tag.

## Artifacts

- `local_sim/phase2_manifest.py` — builds the manifest; `--offline` replays the
  cached raw sources. Records both stack version pairs, measures package parity
  rather than assuming it, and hashes every source.
- `experiments/2026-08-21_phase2_manifest/phase2_manifest.json`
- `experiments/2026-08-21_phase2_manifest/cache/*.raw` — the fetched bytes, so
  the hashes are checkable.
- `local_sim/phase2_checklist.py` — **repaired**. It printed the nonexistent tag
  `v1-phase2` and recommended the obsolete width-256 Sobol fallback. It now reads
  the manifest, cross-checks the live tag's commit against the one the manifest
  was built from (drift = a re-bake), prints package-parity status, and states the
  1024x16 migration facts including *"do NOT refit `_PRUNE_TAU`/`_BASIS_RANK` as
  the first Phase-2 action"*. ROUTINE (`ai/workflows/routine.md` names
  `phase2_checklist.py` in scope): display text and a new read-only artifact, no
  seed block, no shipped file, no instrument.

---

# 2026-08-21 — P2-CP0.1 [IMPORTANT] the gate must own its children's encoding — PREDICTION, before the fix

*Class: IMPORTANT — `scripts/preflight.ps1` is the enforcement layer
(`AGENTS.md` section Work classification). Written before the change exists.*

## The defect, reproduced

`.\scripts\preflight.ps1` step 3 goes **RED** on a console whose Python child
resolves stdout to cp1252 — which is the default here **even with
`chcp 65001` and `[Console]::OutputEncoding = utf-8`**, because Python 3.10 picks
the ANSI codepage for a *redirected* stdout regardless of the console codepage.

Reproduced verbatim 2026-08-21 with `PYTHONIOENCODING` unset:

```
Validating estimator estimator.py
... SymmetryLossWarning ...
  File ".../whestbench/cli.py", line 2773, in _main_participant
UnicodeEncodeError: 'charmap' codec can't encode characters in position 0-32
During handling of the above exception, another exception occurred:
  File ".../whestbench/cli.py", line 3655, in _print_error
UnicodeEncodeError: 'charmap' codec can't encode characters in position 0-1
EXITCODE=1
```

**The estimator validated.** The crash is in the CLI's own results printer at
`cli.py:2773`, and then again in its *error* printer at `cli.py:3655`. Nothing
about `estimator.py` is wrong; the gate is reporting the console's encoding as an
estimator defect. That is the exact failure class this repository has lost four
checks to — a check reporting a state instead of testing a property.

## Predictions, committed before the change

- **B1.** Setting `PYTHONIOENCODING=utf-8` for preflight's child processes makes
  step 3 exit 0 with no other change to `estimator.py` or the environment.
- **B2.** The child's `sys.stdout.encoding` is currently **not** utf-8 and will
  read `utf-8` after the fix. This is asserted as a property inside the step, so
  a future environment that silently loses it turns the gate RED **at the
  assertion**, not at a downstream Unicode crash sixty lines later.
- **B3.** No other preflight step changes verdict. Steps 1, 2, 4-10 are already
  green and none prints non-ASCII through the failing path.
- **B4.** The variables are restored to their prior values when the script exits,
  so preflight does not silently change the encoding of the shell that called it.
- **B5.** `scripts/verify_portability.py` stays green: the change is inside an
  enforcement script both agents already share, and adds no Claude-only rule.

**Falsifier:** if step 3 still fails after the fix, or if any other step's verdict
moves, the diagnosis is wrong and the fix is reverted.

## P2-CP0.1 — the result, against the predictions committed above

Implemented in `scripts/preflight.ps1`: the gate exports
`PYTHONIOENCODING=utf-8` and `PYTHONUTF8=1` for its child processes before the
first step, **asserts inside step 3 that the child actually reports `utf-8`**,
and restores the caller's prior values on both exit paths through
`Restore-CallerEncoding`.

Run with `PYTHONIOENCODING` and `PYTHONUTF8` both unset in the calling shell:

```
=== 3. whest validate ===
  child stdout encoding: utf-8
  PASS  3. whest validate
...
PREFLIGHT GREEN -- 10/12, 2 SKIPPED
EXIT=0
caller PYTHONIOENCODING after = []
caller PYTHONUTF8 after = []
```

| prediction | outcome |
|---|---|
| **B1** step 3 exits 0 with the export | **CONFIRMED** — `PASS 3. whest validate`, whole gate `GREEN 10/12` |
| **B2** the child reports `utf-8`, asserted as a property | **CONFIRMED** — the step prints `child stdout encoding: utf-8` and throws a readable error if it is ever anything else |
| **B3** no other step changes verdict | **CONFIRMED** — 1, 2, 4-10 all PASS; 11, 12 skipped as always without `-Full` |
| **B4** caller's variables restored | **CONFIRMED** — both read empty after the run, as they were before |
| **B5** portability stays green | **CONFIRMED** — `174 checks, 0 failed` |

**Vault untouched:** counter `1`, `touch_log.jsonl` one line. **No seed block
drawn. `estimator.py` untouched.**

### The knowledge delta

**The console codepage is not the child's encoding, and `chcp 65001` does not
fix this.** The reproduction ran with `chcp` reporting **65001** and
`[Console]::OutputEncoding` reporting **utf-8**, and the Python child still
resolved `sys.stdout.encoding` to **cp1252** — because when stdout is a *pipe*
rather than a console, Python 3.10 takes the locale/ANSI codepage and ignores the
console entirely. Any advice of the form "run it in a UTF-8 console" is therefore
wrong for a redirected child. Only `PYTHONIOENCODING` (or `PYTHONUTF8=1`, or
Python 3.15's default) settles it, and it must be exported by whoever *spawns*
the process.

`AGENTS.md` and `CLAUDE.md` already told humans to set `PYTHONIOENCODING=utf-8`
by hand in a POSIX shell. That instruction was correct and is now also
*mechanical* for the gate: prose that has to be remembered is not enforcement.
