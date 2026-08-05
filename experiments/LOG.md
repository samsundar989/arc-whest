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

α becomes bimodal with depth, exactly as the mean-field picture predicts once
activation correlations approach 1 (σ² = wᵀΣw ≈ c·μ², so |α| → 1/√c). By layer 31
**45% of alive neurons are confidently on**. The structure the plan assumed is
there.

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

### Aug-8 runbook (Phase 2 opens 00:00 UTC, round 1429)

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
