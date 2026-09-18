# Deep architecture audit — September 18, 2026

Read-only inspection of CURRENT_STATE, the LOG tail, the active plan, frozen v31/v43 sources, their generators, and existing research/score artifacts. No estimator was executed, truth array opened, seed allocated, or closed lane reopened. The proposals below are unmeasured and are not an operative experiment contract.

## Assessment

The repository has a credible architecture family and a substantial amount of useful evidence. It does **not** have a demonstrated winning architecture/configuration. The latest completed slice is a particularly clear counterexample to describing progress in individual components as a solved package.

The retained complete C10 package has official-precision adjusted DESIGN score **6.486609e-9**, against same-panel v31 **8.860665e-9**. Its **1.366x** advantage is mainly a cost reduction: counted FLOPs fall from **661.962B to 480.851B**, while raw MSE is **3.160759e-8 to 3.183714e-8**, effectively unchanged at this descriptive scale. Both MSEs use the common **2.248071e-9** averaged-truth noise floor. C10 needs another **31.700%** adjusted reduction even to reach the intermediate half-v31 mean target; fresh paired confirmation would still be needed. [Completed packet](../STRATEGIST_PACKET_2026-09-18_COMPLETE_SLICE.md), [factorial decision](../../experiments/2026-09-18_s17_factorial/decision.json).

The new competition observation supplied by the coordinating browser review is a separate, unpaired public score scale: submission #331502 at approximately **2.158e-9**, reported raw MSE **1.645e-8**, and cost multiplier **0.1312**. Its report/row verification belongs to that review. It does not identify the leader's algorithm or give a paired private-performance comparison.

There is an important arithmetic barrier. C10's current score divided by its cost multiplier, about **0.218666**, is approximately **2.96645e-8 effective official-precision MSE**. Even a hypothetical free reduction of its FLOPs all the way to the **0.1** floor would produce approximately **2.96645e-9**, above the new leader scale. Reaching 2.158e-9 therefore requires approximately **27.25% less effective error even at the floor**, or **66.73% less effective error at current cost**. These are planning counterfactuals, not predictions. They use effective error after the existing precision adjustment, not a claim that the local raw label MSE is noiseless.

| Complete cost multiplier | Effective MSE needed for illustrative 2.158e-9 score |
|---:|---:|
| 0.1000 | 2.1580e-8 |
| 0.1312 | 1.6448e-8 |
| 0.1500 | 1.4387e-8 |
| 0.218666 | 0.9869e-8 |

Thus a plan focused only on cheaper old-source readout cannot meet the winning objective at current accuracy. The latest rank32 readout experiment is also decisively negative: **5.514%** FLOP saving bought an increase from **3.183714e-8 to 5.745928e-8** raw MSE at the same noise floor; all 24 pairs worsened. [Query decision](../../experiments/2026-09-18_s17_query_accuracy/decision.json).

## What the current code really represents

The useful scientific backbone is:

1. Full means and covariance, with exact final-layer variance-only pruning.
2. Persistent factored third-cumulant history, including recent exact legs and older shared coordinates.
3. A separable repeated-support birth leg plus rank32 residual. The separable algebra is exact; the residual rank approximation is not.
4. A regenerated fourth-order core consisting of a per-neuron diagonal and a covariance-aligned off-diagonal model. The lambda coefficients are fitted controls, not exact physics.
5. A finite nonlinear closure program and explicit compensated fourth-to-third feeds.
6. A deterministic single-file flopscope export and exact-cost matrix-product identities.

For C10 specifically, shared storage is age4/rank256 with the Gram objective; PAIR+Q is the nonlinear table; P3 is disabled; four own controls are `(0.625, 1.225, -0.1, 1.0)`; P4 and depth5 products are enabled. The current experiment tested this whole configuration. [Frozen C10](../../estimator_k3_v43_complete_c10.py), [composition generator](../../local_sim/s17_build_complete.py), [own-control generator](../../local_sim/s14_own_fit_controls.py).

The code is **not** exact propagation of the true third/fourth cumulants. Its source docstrings sometimes say “EXACT tower,” but exactness is conditional on the represented births, finite closure, and chosen compression. More terms, more exact immediate queries, or a better matrix reconstruction do not automatically reduce final truth error. The corrected qEc experiment and C10/P3 interaction provide direct local evidence for that distinction.

The fitted dense sentinel achieves **2.596107e-8 raw MSE**, common noise floor **2.248071e-9**, but costs **621.352B** and scores worse than C10. Even a hypothetical restoration of that accuracy at unchanged C10 cost would give approximately **5.20e-9 effective adjusted score**, still above the same-panel twofold target **4.43033e-9**. This counterfactual does not establish a compression-independent error floor; it demonstrates that recovering the measured dense sentinel alone would not solve the milestone.

P3 includes two derived kappa3-times-covariance birth routes but excludes the all-distinct gamma route. Its own tiny oracle says the excluded route comprises about 20% of that isolated local response; this is **not 20% of final prediction error**. Gamma has no established economical representation here. Adding the implemented P3 to fitted C10 lowers raw MSE a little but increases complete FLOPs by about 59B and worsens adjusted score. [P3 derivation](../../local_sim/s2_build_p3.py), [completed factorial](../STRATEGIST_PACKET_2026-09-18_COMPLETE_SLICE.md).

## A concrete, very cheap unmeasured hypothesis: constrain the P2 birth diagonal

This is a source-level approximation consequence, **not a newly proved software bug or a promised major gain**.

In the C10 active branch, `resid` is a zero-diagonal repeated-support residual. `_p2_compress` approximates `3*resid.T` by `Zb @ Lb.T`. `_sb = S3` is then stored unchanged. At birth, the carried mode is `M=I` and `diag(A_L)=0`. Therefore the represented newborn third-cumulant diagonal is

```
diag(K3_new_represented) = S3 + diag(Zb @ Lb.T).
```

The low-rank approximation generally has a nonzero diagonal even when the exact residual diagonal is zero. The existing full-rank identity tests are entirely consistent with this: they explicitly test the exact separable rewrite, not diagonal preservation after rank truncation. [Compression and birth sites](../../estimator_k3_v43_complete_c10.py), [identity-test scope](../../local_sim/p2_p2_identity_test.py).

A tiny analytic counterexample requires no network or label. Let `R` be the 3-by-3 matrix with zeros on the diagonal and ones elsewhere. Its unique leading eigenvalue is 2, with unit eigenvector `(1,1,1)/sqrt(3)`. Its best rank1 approximation is `(2/3)*ones(3,3)`. Thus a rank1 approximation to `3R` injects a diagonal of **2** into each newborn third cumulant. The exact residual had zero diagonal.

The constrained approximation would keep the same Z/L factors and change only the existing static birth coefficient:

```
sb = S3 - (Zb * Lb).sum(axis=1)
```

The factor 3 is already in Zb and must not be applied again. In real arithmetic this restores the specified newborn D3 exactly; it does not restore discarded off-diagonal residuals. Subsequent transport carries the correction through the existing `M diag(sb)` representation. This adds no new source, no rank, no new history transport, and no dense matrix construction. For the active rank32/width1024 path, the extra arithmetic is approximately `2*n*r` per birth, under about one million elementary operations across fifteen births before exact flopscope pricing, plus roughly three array calls per birth. That is negligible compared with 480.851 billion counted FLOPs, but must still be metered and checked for floating-point effects.

The correct first discriminator would be a new pre-registered source/approximation question, not a rank sweep:

* Independently reconstruct a tiny symmetric newborn tensor with an arbitrary nonzero-diagonal low-rank residual; verify the old approximation's D3, the constrained D3, the off-diagonal D21, and one arbitrary dense transport. Include the analytic matrix above, nonuniform gates, zero residual, full rank, and signed cancellation.
* On one common incoming live-shape state, record injected diagonal magnitude and its actual next-layer mean/covariance/birth effect. Run the two full endogenous continuations under identical other settings; do not infer truth improvement from the diagonal identity alone.
* Only if the introduced perturbation is material and the complete cost is negligible should the contract allow one fresh paired accuracy comparison. Select no rank and fit no new scalar in that comparison. Keep P13 untouched until an actual release-sized supported case exists.

There is substantial counterevidence to expecting this to supply a major jump. The original P2 identity-test documentation quotes only **+0.9% raw-MSE penalty at rank32** for its earlier oracle chain; the corresponding truth-noise context must be recovered before reusing that as numerical performance evidence. That old-chain attribution is not an upper bound for C10, but it makes a large new gain speculative. Earlier whole-tensor diagonal-protection claims were retracted at matched rank. This proposal differs by imposing the already-known zero diagonal on one residual birth through an existing coefficient at unchanged rank/cost. It does not reopen that old broad lane. A registered scope review should make that distinction explicit.

There is external methodological support for separating known moments from the low-rank remainder. Einkemmer, Ostermann and Scalone construct a conservative low-rank integrator preserving selected physical moments, and Guo/Qiu's LoMaC projects a kinetic approximation to match separately evolved macroscopic observables. These are kinetic-PDE methods; their stability/conservation results do not transfer to cumulant-chain accuracy. The analogy supports testing the constrained birth construction, not predicting a challenge gain. Both primary abstracts were checked during this review. [Conservative low-rank integrator](https://arxiv.org/abs/2206.09374), [LoMaC](https://arxiv.org/abs/2207.00518).

## Where a genuinely larger improvement would have to come from

The remaining serious accuracy question is **which modeled state error drives final truth error**, not whether another approximation has a smaller local norm. The restricted regenerated K4 core, omitted gamma feed, source-residual approximation, and old-source joins are distinct mechanisms. Their errors can compensate. Replacing any one with a more faithful local representation can hurt the rolled-forward prediction.

The best bounded diagnostic already described in the repository is a **single intervention at a fixed join with full continuation**: one common incoming state, one join changed, subsequent uncompressed continuation, and a supplementary frozen-feedback continuation. It can locate compression feedback amplification without selecting on truth. It cannot show that the dense reference is more accurate or justify an expensive online residual oracle. [Existing continuation proposal](2026-09-18_frontier_followup.md).

A signed low-rank residual around `G_off=lambda*C_off` is a possible future K4 extension, but not yet a justified build. Prior full/core transport implementations have adverse complete-cost and numerical evidence. It needs a newly isolated causal residual plus new external evidence sufficient for the applicable reopening rule; its complete birth/feed economics must be derived before a truth job. Merely saying “carry more K4” repeats known work. [Existing K4 hypothesis and counterevidence](2026-09-18_frontier_research.md), [connected K4 derivation](2026-09-14_e14_connected_k4.md).

There is no evidence-supported forecast here of a twofold next result or first place. A strong plan must be allowed to conclude that a diagnostic found no useful mechanism.

The next high-upside question should be a **bounded error-budget attribution**, with the small diagonal repair as a separate cheap hypothesis. Compare the frozen complete chain with a diagnostically uncompressed version of the same nonlinear closure and K4 model, including a full-rank P2 residual; distinguish that from the existing dense-storage sentinel, which still approximates its P2 residual. Use one frozen incoming-state intervention and full continuation to identify whether old joins, newborn residuals, or the modeled K4/feed state supply the dominant recoverable final-vector defect. If the remaining difference lies in the K4 model, require an independently derived same-state core/consumer reference before proposing its replacement. Do not import the reference team's fitted coefficients or assume a stronger augmented chain is truth.

The decision-critical falsifier is practical: if removing storage approximations cannot materially improve truth error, stop allocating the major-jump effort to finer compression objectives. If the more faithful chosen closure still cannot approach the target even when its error is multiplied by the 0.1 cost floor, that closure has no demonstrated accuracy headroom for a compression-only winning plan. Approximation can act as regularization, so this is **not a mathematical lower bound on every member of the family**. Conversely, if a large useful error reduction appears only with an operator whose complete source/feed/update cost cannot approach the target budget, the operator is a diagnostic reference rather than the implementation plan. A claim that the current architecture is sufficient should be rejected until one full candidate survives both tests. An explicit outcome of this diagnostic may be: the missing winning representation has still not been identified.

## Engineering architecture and iteration

Retain frozen archives and the parity fixtures. Gradually replace stacked source-string substitutions with a small explicit internal representation: scientific state, nonlinear term program, compensated birth, storage backend, contraction backend, and single-file exporter. Migrate one interface with source/semantic parity at a time. The current composition generator reads a frozen generated file and applies many asserted string replacements; it works, but every new combination incurs integration review and source-identity repair. A wholesale rewrite would create more risk than it removes.

Use a manifest to name **latest graded v31**, **best DESIGN C10**, and **legacy root estimator v2.6.1** separately. An architecture review or certification runner that silently uses the root file is answering the wrong question. The manifest should bind source hash, generator/dependencies, enabled mechanism configuration, cost evidence, accuracy tier, and certification status.

One narrow exact-cost lead is that `_sb` is labeled “Strassen-Winograd” but spells the standard 18-addition Strassen formula. A true 15-addition Winograd schedule is a distinct engineering hypothesis. Its numerical conditioning and extra live temporaries must be checked; most of the current 480.851B bill is matrix multiplication, so it is unlikely to supply the missing major gain. The label mismatch is not an accuracy defect. Do not bundle that arithmetic change with a new scientific closure.

The intended winning architecture is therefore still a coherent full-mean/full-covariance plus persistent K3/cheap-K4 chain with **a validated substantially better accuracy/cost point**. It is not a new framework. The current family remains worth developing, but the evidence does not justify declaring its present representation sufficient, declaring all alternative families impossible, or treating more uploads as a substitute for identifying the missing error mechanism.
