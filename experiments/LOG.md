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

1. **Effective-dimension / better rotation.** The α shift from 0.87 → 1.06 under the Jacobian rotation is the only hint of a *rate* improvement seen all session. Measure how much final-layer variance lives in the top-k input directions; if it is genuinely low-dimensional, a targeted QMC/stratification split (QMC on k dims, MC on the rest) could buy a rate rather than a constant. Test on the **full 100-MLP mini split** only.
2. **Smoothing for QMC.** α = 1.00 is the signature of a non-smooth integrand (ReLU kinks) defeating QMC. Conditional-expectation smoothing of the last layer is the standard remedy and is untested here.
3. `wall_time_limit_s` still unknown (not in the participant report); it has never bound us (0.41 s/MLP used, ~98× headroom), so this is low priority.
4. **Phase 2 opens Aug 8 00:00 UTC** (round 1429). Run `local_sim/phase2_checklist.py`, submit v1.1, then `fetch_graded_report.py` to re-derive shape/budget/λ/floor and record in RESEARCH_NOTES §1.
