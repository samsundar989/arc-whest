# Frontier research refresh — September 18, 2026

Research assessment only. No estimator, instrument, operative contract, seed ledger, truth artifact, vault, submission or designation was changed. The review read CURRENT_STATE first, LOG from its newest entries, GAME_PLAN, the September 14/16 research, the September 17 response, and the current source generators. The active experiment was still producing artifacts during this review; the parent review owns its final status and the live Chrome competition audit.

## Conclusion

Retain full mean/covariance, persistent factored third-cumulant history, coherent fourth-order feedback, and inexpensive consumed-query storage. This is the best-supported architecture family found. Neither the literature nor available participant code establishes that its present implementation can win. The current accuracy and cost improvements are real but a major jump still needs a new measured mechanism. More nonlinear terms, smaller fixed ranks, generic random polynomial sketches, and a next-weight sketch are not evidence-backed shortcuts to it.

The most useful new external evidence is negative: another team actually tested the simple future-weight and raw TensorSketch proposals that our older notes described as open. These results narrow the next useful experiments. The additional methodological lead is residual-driven enrichment and deterministic cross interpolation of the **consumed aggregate query**, with a strict full-cycle cost test. Neither has demonstrated a challenge gain.

## Source refresh and scope

| Primary source | What it establishes | What it does not establish |
|---|---|---|
| [Wu et al., arXiv 2605.05179v2](https://arxiv.org/html/2605.05179v2) | Direct fixed-network expected-output estimation; cumulant/Hermite recurrences and factorization; width/depth experiments | Its fixed-depth polynomial-activation theory is not a finite-width ReLU accuracy certificate. Appendix D explicitly leaves depth-robust sample-free estimation open. Increasing order can worsen finite-width performance. |
| [ARC code pin](https://github.com/alignment-research-center/mlp_cumulant_propagation/commit/93d091a4c26c042bfffa28f2e76a81bc0aba94bb) | Remote HEAD independently refreshed to this same commit | No newly released winning algorithm was found there. |
| [504aldo code pin](https://github.com/504aldo/whest-p2-cumulant-k3/commit/18c17e2d7a9aeacd399cfc2c6b571e4e16dbfb45) | Remote HEAD independently refreshed to the existing local oracle | It remains a diagnostic reference with fitted tables and an empirical pruning mask; no new upstream source explains the current private leader. |
| [Dunlavy et al., 2508.11139v1](https://arxiv.org/abs/2508.11139v1) | Tensor approximation can optimize quantities of interest instead of reconstruction alone | Its simulation-data optimization does not prove an affordable online basis or rank for our recurrence. Already cited by the repository. |
| [Smith et al., 2603.13532v1](https://arxiv.org/abs/2603.13532v1) | Structured sketches can sum Tucker tensors without expanded intermediate tensors | Does not remove the cost of subsequent nonlinear queries or guarantee sufficient rank. Already in September 16 review. |
| [Ceruti, Kusch, Lubich, 2104.05247v1; BIT 2022](https://arxiv.org/abs/2104.05247v1) | Evolving and previous subspaces can be augmented and truncated adaptively, with robustness properties for the stated differential-equation setting | Our finite nonlinear layer map is not the paper's ODE; its theorem does not transfer automatically. |
| [Hauck and Schnake, 2209.00550v2](https://arxiv.org/abs/2209.00550v2) | Predictor-corrector rank adaptation uses an update residual without materializing its full-rank representation | Its matrix-ODE error estimator is not already an estimator of final ReLU mean error. A challenge-specific residual action is needed. |
| [Ghahremani and Babaee, 2401.04249v1; CMAME 2024](https://arxiv.org/abs/2401.04249v1) | DEIM fiber selection constructs Tucker approximations from selected entries/fibers; authors demonstrate nonlinear PDE applications | Cheap entry access and low ranks must hold here. Dense tensor interpolation can still lose on readout, conditioning or rank. |
| [Celli, 2605.24072v1, May 2026](https://arxiv.org/abs/2605.24072v1) | Nonasymptotic Edgeworth bounds for conditionally Gaussian vectors and random neural-network outputs evaluated at finitely many inputs | A fixed realized weight network under random inputs is a different conditioning problem. This does not prove our truncated ReLU chain or justify arbitrary additional terms. |
| [Adams et al., JMLR 27(33), 2026](https://jmlr.org/papers/v27/24-1199.html) | Finite neural-network distributions at finite input sets can be approximated by GP mixtures with Wasserstein bounds | Not a turnkey method for the competition's fixed weights and Gaussian input integration, nor an affordable 1024-by-16 estimator. |

The last two papers are useful boundary checks, not reasons to restart with neural kernels or Gaussian mixtures. The exact randomness being integrated matters more than a shared keyword such as “finite-width.” Search results from secondary summaries were used only to locate original papers; claims above use original sources.

## Newly recovered participant experiments

The ordinary GitHub issue rendering omitted comment activity. Public GitHub REST responses recovered the following author reports. They have not been independently rerun and do not contain numerical truth-noise floors. Their public-mini development protocol is not this repository's fresh DECIDE evidence.

**E009: one-step future weights failed.** [Issue 17's result comment](https://github.com/tim8es/arc-whitebox/issues/17#issuecomment-5647664240) reports the corrected pulled-back sketch `current_W @ next_W[:, :r]`, tested on one public MLP. It recovered about 20% and 16% of the low-rank degradation at ranks 128 and 192; rank256 worsened. Neither declared promotion condition passed. This supersedes our September 14 evidence pack's “results not public” description. It rejects this cheap sketch formulation, not the actual D21 query objective or every nonlinear sensitivity basis.

**E008: raw TensorSketch products failed badly.** [Issue 16's result comment](https://github.com/tim8es/arc-whitebox/issues/16#issuecomment-5647568257) reports valid old-leg reconstruction, followed by degree-two product relative errors of 15.12 for AP, 8.42 for AA and 9.22 for PP even at sketch dimension512. The author stopped before a full D21 implementation. Thus this is strong adverse evidence for the raw collision-sketch construction, not a mathematical proof that every signed aggregate query sketch must fail. Do not propose the same raw polynomial-feature scheme as the next fresh idea.

**Subsequent public work does not disclose a major gain.** [PR29](https://github.com/tim8es/arc-whitebox/pull/29) describes dead-feedback removal with a small local package-score change and reports approximately 14.2GiB peak RSS. Its “official PASS” wording does not certify an 8GiB server run. [PR30](https://github.com/tim8es/arc-whitebox/pull/30) reports only approximately 0.19% lower error on one public MLP for a final D3-squared correction; its final comment blocks promotion after an unintended rerun. These are useful implementation/evidence lessons, not a new winning baseline.

No leading private estimator source was found in this targeted search. The leader's FLOP utilization is not an identifying signature of its architecture.

## Incorporate the current local result before choosing another experiment

The LOG's late September17 Amendment4 is newer than CURRENT_STATE's batch-A-running header. It records official-precision gains of 1.0761x for the full reference program, 1.0476x for our own pairq program and 1.0150x for b4q, in the specified stage-1 comparison. P12's two-replica-average truth-noise estimate is approximately 2.24805e-9, half the recorded per-replica 4.4961e-9. These are existing development results, not a fresh promotion.

This prevents three misleading claims: that completing the table has already closed the accuracy gap; that adding all higher-order terms must improve it; or that the reference's full program alone explains its final performance. BatchB explicitly contains the lambda interaction cells, P3, and composed storage arms. Finish and synthesize those registered comparisons before creating another table or rank sweep. Different core values on different trajectories do not prove different core formulas; the September17 audit already found matching transport/harmonic formulas.

## Bounded hypotheses after the active slice

These are proposals, not operative contracts, permissions to reopen closed lanes, or forecasts of measured gain. An implementation must first distinguish itself from the corresponding old negative result. Retain the present recurrence and controls unless a separately registered intervention explicitly changes them.

### H1 — residual-driven enrichment of the consumed-query basis

**Mechanism:** the new qE objective improves what is read at a join, but the layer update can create harmful directions absent from that subspace. A fixed rank spends equally when the omitted update is small or large. Predictor-corrector low-rank literature suggests detecting the omitted action and adding only its dominant directions.

**Concrete object:** maintain the current factored source history. Define the residual as the difference between (a) one exact reference update followed by its D3/D21 consumers and (b) the compressed update's consumers, evaluated on selected test directions. Include covariance and new-source feedback in the validation continuation. Do not replace this with factor energy, a scalar mean-gate Jacobian or the already-failed next-weight sketch.

**First discriminator:** on tiny label-free fixtures and authorized saved states, compare fixed qE against qE plus a small residual-enrichment subspace at the same total retained columns and full metered cost. Include signed cancellation, middle-layer birth feedback, and later recovery of a direction initially invisible to D3. Continue the complete recurrence to the final vector.

**Falsifiers:** residual actions cost as much as full queries; enrichment predicts immediate D21 error but misses later mean error; no equal-cost improvement; or the needed rank returns to the current operating point. The external sources justify the residual construction as a method to investigate, not its challenge accuracy. A subsequent layer-dependent rank schedule needs an independently frozen policy and per-network score products, not an adaptive grid selected on DECIDE.

**Potential novelty:** a cheap residual oracle for this particular cumulant recurrence would be a real contribution. “Adaptive rank” itself is established and not novel.

### H2 — deterministic cross approximation of the signed aggregate D21 query

**Mechanism:** current old-source storage reduces transport cost, yet dense reconstruction and polynomial consumers remain expensive. The actual sum consumed as D21 can have different approximation structure from individual A/P legs. DEIM/CUR-style row-and-column interpolation can preserve selected *actual query* values, avoiding noisy collisions in raw polynomial features.

**Representation:** keep persistent source factors and their current update, then approximate only the old-history query as `E ≈ E[:,J] E[I,J]^-1 E[I,:]` or a stable QR/SVD equivalent. The aggregate is formed with signs before selecting/interpolating it. D3 must be handled explicitly. This is not memoryless D21 propagation: future source history remains present. Every selected row/column must be computed from the original factor formulas; treating an exact dense E as a free input would invalidate the claimed saving.

**First discriminator:** derive and meter a complete row/column query oracle, including joins, selection, reconstruction, conditioning checks, D3 and fallback. Compare with the current complete consumer at 1024x16 without labels. Reject before a truth run if useful accuracy requires a rank whose construction/reconstruction costs erase the saving. A full-rank identity and deliberately cancelling sources are mandatory. Do not use a generic dense Tucker core as an unpriced intermediate.

**Risk:** a query's singular spectrum may be too flat; selecting rows can be unstable; repeated approximations may bias covariance and births; retaining all source factors still costs transport. A low D21 approximation error alone is insufficient. This is a higher-risk fallback, appropriate only if profiling confirms query work remains the limiting cost after the active slice.

**Potential novelty:** a stable inexpensive interpolation oracle for this signed cumulant consumer, with full recurrence and cost evidence. The interpolation method is not new.

### H3 — architecture-level discrepancy attribution before a richer closure

**Mechanism:** the reference program is modestly better while its complete chain is better still. Small errors in the regenerated core can accumulate through nonlinear coefficients and births; fitting isolated true moments need not improve a rolled-forward chain. Existing complete-program and lambda interactions narrow this question but do not license a generic core swap.

**Next test only if batchB leaves the gap:** compare the earliest divergence of two complete recurrent chains with compatible full state, identify an independently definable operator difference, and perform a two-way controlled replacement of that operator. The target is an actual recurrence discrepancy with both immediate and final-vector consequences, not a differing norm on unrelated trajectories. If no independently defined difference remains, stop port archaeology and formulate an own closure intervention with a precise new mechanism.

This is the strongest near-term accuracy investigation, but it is already the direction of the active plan, not a new algorithm. Use the existing declared bounded fitting policy if its branch triggers. Do not import empirical tables or post-result pruning masks. Do not add arbitrary sixth/eighth-order terms merely because more moments exist; the current coherent-table results and finite-width theory both caution against that inference.

**A concrete conditional core extension:** if the remaining discrepancy is demonstrably in regeneration rather than merely trajectory inputs, start from an independently specified connected-core operator and split its off-diagonal result as `G_off = lambda*C_off + R`. Preserve the existing exact diagonal, and represent only a signed low-rank residual `R ≈ offdiag(U S U^T)`. Do not force S positive: a cumulant correction is not a covariance. Re-derive every changed D4/D31/D22 consumer and K3 birth/feed with explicit overlap subtraction. A use-only readout correction is not the same intervention.

The first question is whether R has a small **consumer-weighted** residual rank, not whether G is approximately covariance-aligned. Test a tiny independently enumerated connected tensor, full-rank recovery and the complete birth-to-next-query cycle; then price one live-shape layer and all ages. Extra dense core transport alone is affordable-looking, but the historical E14 derivation found 1069 additional square products for its particular complete feed spelling, enough to exceed the budget. That is why a residual transport shortcut must include feeds before anyone launches accuracy work. Public F68 also found that some off-diagonal transport detail did not pay, while F75's adaptive scalar lambda barely helped. This is substantial counterevidence: only a newly isolated residual mechanism can justify revisiting it. [Local complete K4 derivation](2026-09-14_e14_connected_k4.md), [pinned public findings](https://raw.githubusercontent.com/504aldo/whest-p2-cumulant-k3/18c17e2d7a9aeacd399cfc2c6b571e4e16dbfb45/docs/findings_log.md).

This extension is a contingency for a demonstrated missing physical channel, not a recommendation to rebuild the old dense K4 bank or to assume its rank is low. If operator attribution shows the supposedly different cores are already identical, reject that explanation and keep the current recurrence.

## Score economics and practical architecture

For an illustrative adjusted target of 2.0e-9, raw error must be at most 2.0e-8 at the 0.10 floor, 1.33e-8 at 0.15, or 1.0e-8 at 0.20. These are score-identity calculations, not estimates of a candidate. With varying per-network cost, evaluate the mean of each error-times-multiplier product. Do not multiply two separately averaged quantities.

The plausible winning implementation is therefore a coherent K3-plus-K4 chain whose retained-history *update and readout together* cost close to the multiplier floor, with better raw accuracy than the current frozen control. A 0.2B representation with 2e-8 raw is approximately 4e-9: a major advance, not evidence of first place. Current aggressive compression damages accuracy, so reaching the floor by rank reduction alone is not an architecture solution.

The target remains one shape-generic grader-legal export; dense mean/covariance; full-age factored K3 history; explicit births and overlap compensation; a coherent K4 model; exact final-layer pruning and contraction savings; query-aware storage only where the full score pays. The distinction from the current implementation is a **validated accuracy/cost operating point**, not an additional library or a wholesale rewrite.

To shorten iteration, use existing parity fixtures and full-size cost probes before truth, share unchanged controls within the authorized DESIGN panel, stream predictions once, and reserve expensive exact-package/Linux/public checks for a frozen challenger. No literature result excuses candidate-specific accounting, the residual cliff, 8GiB enforcement, or fresh confirmation. Conversely, a new framework or generic tensor package is not needed merely to test either H1 or H2. All substantial probes remain workerq jobs with a recorded optimization review.

## Recommendation

Complete the current slice and record its actual combined outcome. If it misses the major-release gate, retain the best valid artifact and state the remaining accuracy/cost gap. Fund one sharply bounded discriminator: H3 if a concrete accuracy discrepancy survives; otherwise H1. Keep H2 as a more speculative alternative only after proving inexpensive query access. Do not restart sampling, copy the published pruning mask, repeat raw TensorSketch, or present a next-weight sketch as unexplored. There is no evidence-backed promise of a twofold next result; there is a credible architecture and a better-defined frontier experiment.
