# September 18 completed-worker evidence review

Scope: independent read of the completed worker evidence, frozen sources, decision adapters, contract and existing Skeptic verdicts. No truth-array access, estimator execution, seed allocation, new accuracy measurement, public/vault read or gate change. Small calculations below use existing JSON rows and three saved final-prediction streams only. They are retrospective DESIGN diagnostics, not preregistered promotion tests. Only this document was written.

## Decision

Keep P13 untouched. The fit improves its own default, but does not become the best measured bundle on the same held-out identities. Corrected fixed-rank storage lost both its recurrent-query condition and its accuracy condition. The strongest supported next preparation is a **matched-cost integration/parity check of existing components**, with saved-prediction error geometry used to reject naive output ensembling. There is no evidence here for multiplying the fit, P3 and compression gains, or for selecting a fourth-core correction from reference traces.

Do not require discovery of a new operator before correcting the implementation comparison: the fitted source still lacks P4's exact-cost transformations. Applying and testing those established transformations is a smaller, clearer intervention than another closure or rank search. It needs source/algebra parity and a complete metered bill before any new truth comparison; this review does not authorize a run or choose new constants.

## What survives the audit

The September17b stage-3 repair correctly distinguishes the official-precision score from the observed score. `local_sim/s12_decide.py:137` uses per-row `mse - gt_noise + gt_noise * (2**25 / 1e9)`, then the Phase-2 counted-FLOP multiplier. Its query gate at line 145 requires E, D3, D21, mean/covariance continuation and final-row displacement. Missing historical E is not filled by an unrelated score result. The corrected-storage reader requires independently saved full-row binding (`local_sim/s16_corrected_storage_decide.py:108`). These repair mechanisms address the recorded defects; no evidence supports retroactively calling the historical qE screen a full pass.

I independently checked current source SHA-256 values against every arm in the stage-1, stage-3 and corrected-storage score files. I also checked all 19 fit source hashes and all 19 row-file hashes, plus byte identity of fit `result.json` and `manifest.json`. Those checks pass. This is a byte/row audit, not an independent truth-accuracy recomputation. The raw fit archive's CRLF preservation and fresh-checkout verification are separately documented in `experiments/2026-09-18_own_fit_archive_skeptic.md`; no archive repair was repeated here.

The selected fit vector remains `(0.625, 1.225, -0.1, 1.0)`. Its 17 training evaluations followed by two fixed last-12 evaluations agree with the deterministic schedule in `local_sim/s15_own_fit_driver.py:76`. The fit's official-precision last-12 scores are 8.314334391e-9 versus 7.262344231e-9. The positive paired difference and all-12 direction support the within-fit result. The corresponding raw errors, 2.944347447e-8 and 2.598872584e-8, have common mean truth-noise floor 2.227113892e-9. These 12 networks were previously exposed to other DESIGN arms; they are not project-wide untouched holdout evidence.

Corrected qEc loses to old qE: official-precision 7.086620919e-9 versus 7.527855182e-9, paired difference -0.441234263e-9 with t95 [-0.574318074,-0.308150451]e-9. Raw errors 3.386033461e-8 and 3.583154632e-8 share floor 2.248070958e-9. Same-state retained energy rises, but complete recurrent E and continuation worsen. This is a useful counterexample to treating immediate retained energy as the objective. It closes this fixed configuration, not every query-aware storage method. Source: `experiments/2026-09-18_corrected_storage_verdict.json`, `query_old.json`, `query_corrected.json` and the separate query/score Skeptics in that directory.

Every relevant accuracy arm has DESIGN caps disabled. Zero saved failures means those relaxed runs completed. It does not establish the residual cliff, wall limit or Linux memory limit. I make no timing transfer from their local measurements.

## The controls reveal a missed comparison

The headline `p3_on_pairq` gain of 1.206999x is **P3 plus P4**, not a clean isolated P3 gain. The actual transformations are explicit in the frozen sources:

| Source | Matrix implementation | P3 |
|---|---|---|
| `estimator_k3_v40_pairq.py` | Strassen depth 4, minimum block 64; no P4 final-variance/block-symmetric toggles | absent |
| `experiments/2026-09-18_own_fit/eval_18_validation.py` | same depth 4/minimum 64 and no P4 toggles | absent |
| `estimator_k3_v42_pairq_p3_uncompressed.py` | P4 final variance and block symmetry, depth 5/minimum 32 | rank 32 with compensation |

Source anchors: v40 lines 659/727; fitted source lines 673/741; v42 header and lines 81/82/685/753. The v42 generator chain explicitly adds closure, then P3, then P4. Its 680.326-billion FLOP bill versus v40's 669.941 billion is the **net** effect of extra P3 work and P4 savings; it is not evidence that P3 alone costs only 1.55%.

The registered 1.171818x stage-2 comparison against `v32_p4d5` is appropriately a **closure-plus-P3** comparison with P4 matched. It is not an isolated P3 effect either. P4 is algebraically exact in real arithmetic, but its floating-point effect and complete cost still need checking on the particular fitted/composed source before transplanting an accuracy claim.

I aligned existing P12 last-12 identities and noise floors, then applied the same official-precision formula and repository `competition_score.paired_compare`. All numbers below are adjusted scores in units of 1e-9; intervals are paired Student-t, and no gate was called:

| Existing control versus fitted source | Control | Fit | Control minus fit, t95 | Fit wins |
|---|---:|---:|---|---:|
| Own PAIR+Q default | 8.314334 | 7.262344 | +1.051990 [+0.803460,+1.300521] | 12/12 |
| v32+P4 | 7.920582 | 7.262344 | +0.658238 [+0.337187,+0.979288] | 11/12 |
| PAIR+Q+P3+P4 | 6.765694 | 7.262344 | -0.496651 [-0.786281,-0.207020] | 2/12 |
| Compressed PAIR+Q/old-qE+P4 | 7.144863 | 7.262344 | -0.117482 [-0.423261,+0.188298] | 4/12 |

The matched v32+P4 gain is only **1.090637x** on these 12 rows, below 1.10 as a point estimate. This retrospective subset is not the registered complete stage-2 test, so it must not be turned into a new gate verdict. It does show why 1.144855x versus the fit's own default cannot stand in for it. Sources: stage-1 `score.json` arm `cand6`, stage-3 `score.json` arms `control`, `cand6`, `cand4`, and fit `eval_18_validation.jsonl`. All share mean local truth-noise floor 2.227113892e-9 on this subset.

The bounded fit is complete, but calibration is not proved exhausted. In the raw manifest, theta=0.625 is `1 - .25 - .125`, the furthest downward point reachable in the two scheduled passes; lambda=1.225 is `1 + .15 + .075`, the furthest upward point. Neither is the parameter's hard bound. Evaluation 9 lowers theta from .75 to .625 at fixed lambda1.15 and improves training official score 7.310379104e-9 to 7.286553327e-9 (0.326%); evaluation 12 raises lambda1.15 to1.225 and improves it to7.252107977e-9 (0.473%). Their penalized objectives improve too. The later tilt step moves to -0.1 and improves to7.232690913e-9; theta/lambda are not revisited at that final point. These training raw errors have common first-12 truth-noise floor2.269028025e-9. There is no evaluated stationary bracket around the final vector. This supports “a finite search stopped as registered,” not “these controls are optimized”; the small final increments also do not support assuming another large gain. Do not extend this finished fit post hoc. If a compressed integration passes parity and warrants calibration, one separately registered bounded fit **directly on that complete source**, with frozen split/budget/stopping rules and saved predictions, is better evidence than transferring this uncompressed optimum or multiplying gains.

## Saved predictions support geometry without another truth read

Let p and q be saved predictions, y the already-scored averaged truth, m_p and m_q their saved MSEs, and d_pq = mean((p-q)^2). Then the exact identity

`mean((p-y)*(q-y)) = (m_p + m_q - d_pq)/2`

recovers the **scalar error inner product per network** without reopening y. For independent unbiased truth noise with expected MSE nu, subtract nu to estimate the true-error inner product. Likewise, for delta = q-p,

`mean((p-y)*delta) = (m_q - m_p - mean(delta^2))/2`.

The cross term needs no explicit noise-floor subtraction because delta contains no truth noise. A fixed affine combination whose weights sum to one has squared error recoverable from these Gram entries; its expected truth-noise contribution is still nu, not nu times the sum of squared weights, because every prediction is compared with the same noisy truth.

This gives noise-unbiased second-moment estimates **when the arms/weights are fixed independently of the truth noise**. It does not remove model-selection bias from reusing P12, make fitted mixture weights unbiased, or provide an unbiased error-correlation ratio. It cannot recover per-neuron signed errors or locate a causal layer. Optimizing a blend on these reconstructed quantities is still label-dependent fitting, even though no truth file is reopened. Such a selection needs its own declared DESIGN procedure and fresh evaluation.

I checked all prediction-row SHA-256 values, source identities, seed identities and saved MSE equality for just three fixed streams: stage-1 `preds_cand6` (PAIR+Q), stage-3 `preds_cand6` (PAIR+Q+P3+P4), and stage-3 `preds_cand4` (compressed PAIR+Q/qE+P4). Each binary is 98,304 bytes (24 x 1024 float32). Calculations use float64 differences of their stored float32 final rows. Results across 24 P12 networks:

- P3-bundle displacement squared norm from PAIR+Q: 7.049103212e-9.
- Compressed-bundle displacement squared norm from PAIR+Q: 4.434006713e-9.
- Their mean displacement inner product: 9.009054072e-11; pooled cosine **0.0161144**. These are bundle displacements, including P4, not pure operator directions.
- P3-bundle versus compressed-bundle final-error inner product after subtracting common noise: 2.110493522e-8; pooled estimated error cosine **0.802101**. The substantial remaining errors align even though the two interventions move predictions in almost orthogonal directions.
- A fixed 50:50 output blend has reconstructed raw MSE **2.617873839e-8**, with floor **2.248070958e-9**. Running both complete arms has summed counted cost about **0.533016 B**, before blending overhead; the resulting official-precision adjusted score is about **1.279563198e-8**. Straight output ensembling is therefore an unattractive cost trade.
- The fixed additive output surrogate `P3_bundle + compressed_bundle - PAIRQ` has reconstructed raw MSE **2.908033728e-8**, with the same floor. This is an algebraic diagnostic only. Actual recurrent composition can change every later state and its cost is unknown; the surrogate is not evidence for a compressed P3 candidate or a way to multiply gains.

The fit cannot enter this geometry from existing artifacts: its driver saves scalar lifecycle rows (`local_sim/s15_own_fit_driver.py:178-183`), not final predictions. No fit `.f32` sidecar exists in its raw directory. Scores alone cannot recover distances between fitted outputs and P3/compressed outputs. Preserve final prediction sidecars in the next already-authorized evaluation path; do not rerun the completed fit just to backfill them.

## The remaining gap has three different scales

P12 has no saved graded-v31 arm in the completed stage-1, stage-3, corrected-storage or fit experiments. Therefore there is **no direct same-panel v31-to-best-P12 release gap** to calculate. The public 4.37601515e-9 half-v31 score and P11 5.18587852e-9 half-v31 score are planning scales, not interchangeable P12 thresholds.

For a fully paired historical calculation, `experiments/2026-09-16_stage1_k4feed/score.json` control is v31 and `experiments/2026-09-17_followupF/score.json` cand3 is the old rank256/P4 bundle, on the same 24 P11 identities:

| Precision convention | v31 adjusted | Bundle adjusted | Paired arithmetic gain | Further factor to half-v31 mean |
|---|---:|---:|---:|---:|
| Observed | 1.037175704e-8 | 8.746249903e-9 | 1.185851897x | 1.686551250x |
| Estimated official precision | 9.678857090e-9 | 8.248344612e-9 | 1.173430251x | 1.704404670x |

P11 averaged truth-noise floor is 2.3817169e-9. Observed paired difference t95 is [1.379270033,1.871744247]e-9; official-precision t95 is [1.186052910,1.674972045]e-9. These reproduce the historical modest gain and demonstrate that correcting noise does not close the gap. They are neither a P12 forecast nor a fresh release result. The P11 raw target near 1.68e-8 further assumes an unchanged additive compression penalty and fixed cost; it is a conditional engineering calculation, not a universal uncompressed requirement.

On P12, the legitimate full-panel comparison is compressed old-qE bundle 7.086620919e-9 versus P3 bundle 6.798844729e-9: only **1.042327x** at official precision, despite very different raw-error/cost tradeoffs. That modest separation and the untested fit/P4 integration are more actionable than extrapolating a public-board gap through multiple panel ratios.

## Bounded next action

Prepare one frozen **fitted PAIR+Q plus P4** source and prove algebraic equivalence of its exact-cost transformations to the fitted source at initial, ordinary and final steps, then verify full recurrent output tolerance and a complete counted bill label-free. Save its operation ledger. Do not add P3, retune parameters or change storage in that same intervention. If the expected exact-cost parity fails, stop at the mismatch. If it passes, the existing fit can be compared at a fair cost setting in a separately registered, source-bound DESIGN evaluation with prediction sidecars; compare against matched unfitted PAIR+Q+P4 and the existing P3 bundle on common identities. That discriminates whether the fit adds value after the known cost improvement, with one causal change and no new search.

Only after that synthesis should a fit/P3/storage composition be considered. The present artifacts justify neither choosing that composition nor spending P13. A fresh decisive release still requires one exact eligible candidate, paired v31, the unchanged 2.0 gate and independent Skeptic review.
