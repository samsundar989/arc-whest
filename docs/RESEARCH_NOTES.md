# Research notes & intel (living document)

*Everything we know, with sources. Snapshot dates matter — AIcrowd pages were crawled ~Jul 28–29 and re-checked Aug 3–4, 2026. Update as Phase 2 develops.*

## 1. Confirmed competition spec (Phase 1; Phase 2 assumed same until run report proves otherwise)

| Item | Value | Source |
|---|---|---|
| Network | ReLU MLP, width 256 × depth 32, **no biases**, square layers | starter kit `docs/concepts/problem-setup.md`; HF card |
| Weights | He init, i.i.d. N(0, 2/256), float32; forward `y = ReLU(W.T @ x)`, weights stored (input, output) | problem-setup.md; `local_engine.py` |
| Input dist. | x ~ N(0, I₂₅₆) | overview; problem-setup.md |
| Predict | (depth, width) array of per-neuron post-ReLU means; **final layer scored** | estimator-contract.md; townhall |
| Metric | mean over MLPs of `final_layer_mse × max(0.1, C/B)`; failures → zeros + multiplier 1.0 | scoring-model.md; leaderboard math confirms 0.1 floor (overview page's 0.5 is stale text) |
| Compute | `C = flopscope FLOPs + 1e11 × residual_wall_seconds`; B = 2.72e11 (Phase 1); warm-up was 6.8e10 (forum; overview said 3.4e10 — unresolved discrepancy, moot now) | scoring-model.md; forum 18026 |
| Ground truth | MC, **N = 1e9** samples/MLP → GT noise ≈ 2e-10 MSE | ground-truth.md; HF card |
| Grader | CPU-only, 16 vCPU / 64 GB, no network; only flopscope + whestbench + stdlib importable; arrays immutable; 4 GiB/array cap; hidden per-MLP wall cap (read `admin_config.wall_time_limit_s` in run reports); `setup()` off-budget ≤~5s | forum 17997; stage-5 docs; FAQ |
| Submission | `whest submit --estimator estimator.py`; 50 MiB / 50 files; `.npz` (no pickle); 50/day/team; graded on ~10 held-out MLPs; final = private re-run of designated submission on fresh MLPs | README; FAQ; townhall |
| Seeds | all predict-time randomness from `mlp.seed`; setup-time from `ctx.seed`; violations risk DQ; determinism re-checked on re-grade | 01_random.py; code-patterns.md |
| Dataset | HF `aicrowd/arc-whestbench-public-2026` — tags `v1-warmup`, `v1-phase1` (mini=100 / full=1,000 MLPs); **no v1-phase2 tag as of Aug 4** | HF API refs |
| Timeline | Phase 2: Aug 1 – Sep 19 23:59 UTC; private re-runs Sep 20–30; results Oct 1; NeurIPS ws Dec 11–12; contribution write-up due Sep 26 | overview (re-fetched Aug 3); forum 18041 |
| Prizes | Phase 2: $50k/$20k/$10k + $20k contribution (Phase 1's $50k separate, closed). Winners: OSI open-source; LLM use allowed w/ disclosure | overview; forum 18026, 18041; townhall |

## 2. Baselines & reference numbers (256×32, B=2.72e11, mini split)

| Method | Final-layer MSE | FLOPs | Adjusted |
|---|---|---|---|
| zeros | ~0.91 | ~0 | ~9.1e-2 |
| random U[0,1) | ~0.75 | small | ~7.5e-2 |
| mean propagation | ~9.5e-4 | 11.2M (0.004% B) | ~9.5e-5 |
| covariance propagation | ~8.4e-5 | 1.6e9 (0.6% B) | ~8.4e-6 |
| + published scalar bias fix | ~2.8e-5 | ~same | ~2.8e-6 |
| **ours: + per-layer calibration** | see experiments/LOG.md | ~3.3e9 (f64) | see LOG |
| plain MC at full budget (~65k samples) | ~2.8e-6 | 2.72e11 | ~2.8e-6 |
| RQMC + exact layer 1 (public) | — | C/B=0.42 | 4.10e-7 |
| hybrid structure-aware (public) | 2.18e-7 | — | 1.551e-7 |
| **Phase 1 #1** | 3.12e-8 (col.) | ~40% B | **1.24e-8** |
| GT noise floor | ~2e-10 | — | — |

Phase 1 top-10 adjusted: 1.24e-8, 2.30e-8, 2.32e-8, 4.63e-8, 5.59e-8, 6.38e-8, 9.18e-8, 9.47e-8, 9.76e-8, 1.06e-7. 636 participants / 76 teams / 13.2k submissions (Jul 28 snapshot). Warm-up (256×8) winner: <4e-8 (Manifold market resolution).

## 3. Key forum threads (discourse.aicrowd.com)

- **17997** challenge open — grader specs, 50/day. `/t/arc-white-box-estimation-challenge-is-now-open/17997`
- **18026** Phase 1 launch — 256×32, B=2.72e11, +$50k, `v1-phase1`. `/t/phase-1-launch-deeper-models-and-increased-prizes/18026`
- **18041** contribution-prize criteria — "mechanistic" bias, write-up format, deadlines (P2: Sep 26). `/t/algorithmic-contribution-prize-guidelines-.../18041`
- **18053** RQMC + Rao-Blackwell write-up (evaaaz) — 4.10e-7, unbiasedness proof, rank-1 lattice. `/t/unbiased-randomized-qmc-rao-blackwell-.../18053`
- **18063** Gaussian-closure scale bias (pscamillo) — cov-prop overestimates final means by stable ~0.992×; scalar fix → 3× MSE cut; residual signal decays exp(depth/τ), τ≈5.1 layers; negative results: K=3 cumulants, low-rank corrections, learned correctors all failed to beat the scalar at depth 32. `/t/phase-1-write-up-characterizing-a-systematic-scale-bias-.../18063`
- **18078** townhall summary + recording (youtu.be/pvCgbUJ-nTo) — final-layer-only scoring; OSI; fresh private MLPs; ARC hints: cumulant prop > QMC long-term, low-rank refinements, Edgeworth divergence/Borel resummation. `/t/townhall-summary-recording/18078`
- **18085** variance/bias budget write-up (radiant-allomancer) — RQMC prefix + shrinkage + Edgeworth + offline ridge corrector = 4.47e-7; **≥58% of residual is bias not variance**; control-variate coupling dies by layer 32 (0.86→0.25); third-cumulant transport loses ~98% signal. `/t/phase-1-write-up-a-variance-bias-budget-.../18085`
- **18099** ⚠️ flopscope bypass bug (Kerensa) — raw NumPy-backed arrays escape FLOP counting; billed only wall-time at λ=1e11 < real AVX throughput; jtel: top submissions show instrumented/effective ratios <0.001. No organizer reply as of crawl. `/t/potential-flopscope-accounting-bypass-bug/18099`
- **18106** structure-aware hybrid write-up (natasha_stewart) — dead/on/kink classification, antithetic Sobol to layer 30, linearized "on" neurons last layers → 1.551e-7. `/t/phase-i-submission-structure-aware-estimation-.../18106`
- **18108** mliston (#4) proposal to fix wall-time channel (cap share / raise λ to 1e12 / require instrumented APIs). `/t/recommendation-restoring-the-estimation-framing-.../18108`

## 4. The science (papers that matter, ranked)

**Directly on-task:**
- ARC's method paper: *Estimating the expected output of wide random MLPs more efficiently than sampling* — arXiv:2605.05179 (Wu, Lecomte, Winer, Robinson, Hilton, Christiano). Cumulant propagation w/ Hermite expansions; K-th order → MSE ~O(1/n^K) but cost O(n^{K+1}L); factorized variants O(n^K L²); **known failure: error ~L^K depth compounding; loses to sampling at depth ≥8 for K=4**. Code: github.com/alignment-research-center/mlp_cumulant_propagation
- *Formalizing the Presumption of Independence* — arXiv:2211.06738. The conceptual foundation (heuristic estimators; Wick propagation as worked example).
- ARC blog: *Estimating Tail Risk in Neural Networks* (layer-by-layer activation modeling frame + its known blind spot: cross-layer correlations); *A bird's eye view of ARC's research*; Wu & Hilton arXiv:2410.13211 (low-probability estimation — importance sampling beats extrapolation).

**Toolbox:**
- Gaussian-ReLU closed forms: E[ReLU(X)] = μΦ(μ/σ)+σφ(μ/σ); arc-cosine kernels (Cho & Saul, NIPS 2009) for zero-mean pair covariances; **exact nonzero-mean pair covariance**: Wright et al., AISTATS 2024, arXiv:2403.16163 (bivariate-CDF formulas — the exact alternative to the gain/linearized cross-term; candidate upgrade to our v0).
- Mean-field/NNGP depth theory: Poole et al. 1606.05340; Schoenholz et al. 1611.01232; Lee et al. 1711.00165. He init is ReLU-critical: norms preserved, correlations approach 1 only algebraically (Hayou 1902.06853) → deep layers become near-constant functions (final-layer variance shrinks with depth; observed avg_variance ≈ 0.18 at depth 8).
- Finite-width corrections (the *why* of the bias we calibrate away): Yaida 1910.00019; Roberts/Yaida/Hanin 2106.10165 (expansion parameter L/n = 0.125 here); Hanin 2204.01058 (exactly-our-setting cumulant hierarchies); Hanin & Nica 1909.05989.
- Variance reduction: Owen's MC book (artowen.su.domains/mc); scrambled Sobol; control variates for QMC (Hickernell et al. 2005); MLMC (Giles, Acta Numerica 2015); piecewise-linearity of ReLU nets along lines — Hanin & Rolnick 1901.09021/1906.00904 (few kinks per ray at init → exact radial integration is plausible).
- Weight-space learned models (Phase C): DWSNets 2301.12780; NFN 2302.14040; Universal Neural Functionals 2402.05232; Graph Metanetworks (ICLR 2024); property-from-weights precedent 2002.11448.

## 5. Strategic reading of the field

1. Top of Phase 1 board (~1e-8) is ~100× above GT noise — headroom exists, but part of the top may be wall-time-loophole inflated (thread 18099: instrumented/effective <0.001). A fair re-grade likely compresses the board; **build for the re-grade**.
2. Two independent write-ups agree: at depth 32 the *bias* of Gaussian closure (not sampling variance) is the binding constraint (≥58% of residual; exp-decaying exploitable signal, τ≈5.1). Conclusion: pure sampling plateaus ~1e-7; pure analytics plateaus ~1e-5-ish uncorrected; **the win is bias-corrected analytics + smart sampling of the residual**.
3. ARC's own wishlist (townhall) = Phase D of our plan: depth-robust low-rank cumulants + resummation. They have effectively pre-announced what impresses them.
4. Everyone's negative results (failed learned correctors, dead control variates at depth 32) used *generic* regressors on *final-layer* residuals. Our angle: correct *per-layer* along the way (stop compounding early) rather than patch the end.

## 6. Open questions / to verify next

- [ ] Phase 2 config: same 256×32 + 2.72e11? floor still 0.1? λ still 1e11? → first graded run report (`admin_config`) + forum. No `v1-phase2` HF tag as of Aug 4.
- [ ] Did organizers respond to 18099/18108 (loophole)? Any flopscope ≥0.10 patch notes?
- [ ] Exact flopscope pricing of `stats.norm.cdf/pdf`, `standard_normal` (~16/elt), einsum paths, `as_symmetric` — read from our own run reports.
- [ ] Grader wall cap `admin_config.wall_time_limit_s`; real per-run wall time of our estimator (Python-loop overhead × λ matters: 0.1s ≈ 3.7% budget).
- [ ] Whether `fnp.diagonal`, `fnp.outer`, `flops.stats` names match exactly (validate will tell).
- [ ] Team-formation rules for Phase 2 (Phase 1 merges closed Jul 31).
- [ ] Phase 1 final standings & any released winner write-ups (Aug 7 write-up deadline) — free intel, check ~Aug 8.

## 7. Our experiment provenance

Local cloud sim (2 vCPU): He MLPs self-generated; MC truth 200k samples (measurement floor ~2e-6·(0.4)≈; MC-vs-MC reported in experiments/calibration.json); the exact `estimator.py` runs through numpy shims (`local_sim/`). Per-layer calibration factors fitted at 256×32 (K=5 MLPs), validated on disjoint seeds (validate_estimator.py). Numbers in experiments/LOG.md.
