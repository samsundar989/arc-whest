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
