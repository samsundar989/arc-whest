# Frontier architecture and literature assessment

The evidence supports continuing with a **compact, persistent cross-neuron cumulant architecture**, with a faithful nonlinear recurrence and a separately tested accuracy correction. It does not establish that the current implementation can win. F21 successfully removed redundant arithmetic; its remaining error is too high for cost reduction alone to beat the September 14 leader, even at the minimum score multiplier. The next useful step is the already identified full-rank precision discriminator, followed by one compression intervention. A coherent approximate fourth-order correction is the leading accuracy hypothesis, not an established result.

This is a private research assessment checked September 14, 2026, around 07:27 UTC. Recommendations are proposed experiments, not preregistered contracts, promoted claims, or authorization to use protected data. Repository evidence is taken from CURRENT_STATE, the LOG tail, the September 14 strategist packet, and September 10–12 reviews; no challenge labels or experiment blocks were read for this assessment.

## Directly relevant research and freshness

Wu et al. study the correct **quenched** quantity: input expectation for a fixed set of weights. Their method retains non-Gaussian cross-neuron information through cumulants and Hermite expansions. The asymptotic efficiency theorem assumes fixed depth and polynomial activations; ReLU results are empirical. Their published setup also uses a final linear output, so challenge post-ReLU readout and layer counting must be aligned. These distinctions support the architecture family, not a quantitative guarantee for 1024 by 16. Latest observed paper version remains v2, May 14, 2026. [ARC paper v2](https://arxiv.org/html/2605.05179v2), [version history](https://arxiv.org/abs/2605.05179).

Fresh GitHub API reads returned the same source pins as September 12:

| Source | Current pin and commit date | Implication |
|---|---|---|
| ARC reference | `93d091a4c26c042bfffa28f2e76a81bc0aba94bb`, August 4 | No newly discovered upstream implementation to import |
| 504aldo Phase-2 repository | `18c17e2d7a9aeacd399cfc2c6b571e4e16dbfb45`, September 10 | Source architecture and published ablations unchanged |
| NTK-Unlimited | `22f40289b1aa4fc36fe670abcf03c3c30a141651`, May 28 | An adjacent ensemble-theory implementation, not a challenge estimator |

[ARC pin](https://github.com/alignment-research-center/mlp_cumulant_propagation/commit/93d091a4c26c042bfffa28f2e76a81bc0aba94bb), [504aldo pin](https://github.com/504aldo/whest-p2-cumulant-k3/commit/18c17e2d7a9aeacd399cfc2c6b571e4e16dbfb45), [NTK-Unlimited pin](https://github.com/PhilippMisofCH/ntk-unlimited/commit/22f40289b1aa4fc36fe670abcf03c3c30a141651).

The ARC August 1 code correction separates the factorable augmented terms from the unrestricted augmented method. Consequently, SIMPLE, factored AUGMENT, unfactored AUGMENT, and full K4 are different controls. A comparison to the wrong term set can look like compression failure while actually changing the algorithm. [ARC correction](https://github.com/alignment-research-center/mlp_cumulant_propagation/commit/56f736a7e74c60d9714fa06bd5cb6bf48217a6d8).

## New discussion evidence changes the strength of the negative claims

Topic 18218 now has three posts, superseding the September 12 observation of one post. On September 14 at 03:58:43 UTC, 504aldo reported seeing other entries reduce cost by 7–22% without losing accuracy, and reported sponsor confirmation dated September 11 that Strassen-Winograd through flopscope operations is permitted without a recursion-level limit. The original sponsor communication was not independently verified here. The reply does not disclose how those entrants improved their code. [September 14 reply](https://discourse.aicrowd.com/t/everything-we-tried-a-factorized-k-3-cumulant-propagation-estimator-at-0-25-x-b-where-its-flops-go-and-25-measured-dead-ends-team-504aldo-rank-10/18218/3).

**Inference:** the earlier catalogue is a valuable collection of bounded negative experiments, not a lower bound on all implementations. Similar leaderboard numbers do not prove identical code, common lineage, eligibility, or use of a particular algorithm. James Henry's preceding comment about apparent copies is participant interpretation, not source attribution. No new closure theorem or verified winning implementation was disclosed in these replies. [Discussion](https://discourse.aicrowd.com/t/18218).

The most relevant inspectable empirical benchmark remains 504aldo's own architecture ladder. Its table reports V29 at approximately `5.40e-9` adjusted, `2.13e-8` raw, and `0.2526` budget fraction. Its small-panel ablations favor persistent old-source compression and consistent K4 use plus feed, while K4 use without feed worsens error. These are author reports on reused public development networks, with numerical truth noise floors unavailable; they are not our DECIDE evidence. [Pinned ablations](https://github.com/504aldo/whest-p2-cumulant-k3/blob/18c17e2d7a9aeacd399cfc2c6b571e4e16dbfb45/docs/ablation_table.md).

The existing local audit additionally found that V29 turns its regenerated K4 branch off outside the target shape, and that its variance floor is not a PSD guarantee. Small off-shape smoke success therefore cannot validate the production closure. Its fitted tables and pruned terms are not a clean exact-K3 baseline. [Pinned V29](https://github.com/504aldo/whest-p2-cumulant-k3/blob/18c17e2d7a9aeacd399cfc2c6b571e4e16dbfb45/estimators/estimator_v29.py), [local source audit](2026-09-12_frontier_update.md).

## The score target requires both axes

Repository-confirmed F21 public raw error is `3.779436170248118e-8`, with adjusted score `1.595729544363834e-8`. Public truth noise is unavailable. The September 14 Chrome inspection found leader submission #330876 displayed at `2.540e-9` on its detail page; the board rounds this to `2.50e-9`. Using the more precise displayed detail value only as a dated planning level, the following is elementary score algebra, not an experimental result:

| Candidate cost multiplier | Largest raw error that reaches `2.540e-9` |
|---:|---:|
| 0.10 | `2.540e-8` |
| 0.125 | `2.032e-8` |
| 0.15 | `1.6933e-8` |
| 0.20 | `1.270e-8` |
| 0.25 | `1.016e-8` |

At unchanged reported error, F21 would score `3.7794e-9` even at the floor. It needs approximately 32.79% lower error at that floor, and much more if its multiplier remains higher. Conversely, an error near `2.0e-8` combined with cost at 0.125 is a plausible competitive planning target. Neither number is yet a demonstrated property of our candidate, and the private network distribution and future leaderboard can differ. [Local graded synthesis](../CURRENT_STATE.md), [strategist packet](../STRATEGIST_PACKET_2026-09-14.md).

Faithful compression is a necessary engineering enabler for the selected path that preserves F21's recurrence while reducing its state cost; it is not proved necessary for every possible winning algorithm. It is also not the whole scientific program. A sequence that repeatedly achieves only 10–15% savings can improve rank without reaching the prize frontier. The project needs an explicit remaining-error hypothesis alongside its cost target.

## Adjacent literature: useful transfers and category errors

**Output-sensitive compression.** Dunlavy, Phipps, Kolla, Shadid and Phillips introduce CP/Tucker compression objectives that include quantities of interest and invariants, with examples in combustion and plasma simulation. The inspected version is arXiv v1, August 15, 2025. This supports judging compression by downstream observables rather than Frobenius energy alone. It does not prove a rank bound, cheap online solve, or error guarantee for a nonlinear cumulant recurrence. [Primary paper](https://arxiv.org/html/2508.11139v1).

The idea has older mathematical foundations: Billaud-Friess, Nouy and Zahm derive residual norms aimed at particular solution quantities for high-dimensional equations. Their hypotheses concern a different problem class. The transferable principle is to choose the approximation norm around the quantity consumed by the algorithm; importing its convergence theorem here would be unjustified. [2013 paper](https://arxiv.org/abs/1304.6126).

**Finite-width diagram methods.** Guillen, Misof and Gerken's ICML 2026 paper, latest arXiv v4 dated June 15, develops finite-width corrections to NTK and related statistics. Its observable is averaged over random network parameters. NTK-Unlimited supplies analytic recursions and ensemble sampling. This is useful symbolic-method prior art, but does not estimate a fixed network's input-conditioned neuron means. The absence of some diagonal NTK corrections for ReLU does not imply that our fourth-cumulant correction vanishes. [Paper/version history](https://arxiv.org/abs/2508.11522), [author code](https://github.com/PhilippMisofCH/ntk-unlimited).

**Recent QMC improvements.** Yu and Wang's March 4, 2026 v2 combines importance sampling, active subspaces, and conditional preintegration for financial integrals. It supplies a mechanism for improved sampling when a suitable low-dimensional, smooth conditional integrand exists. It supplies no evidence that such a representation exists cheaply for a fixed 1024-input, depth-16 ReLU network. Learning directions, locating boundaries, and doing conditional integration would all consume the challenge's counted work. [Primary paper](https://arxiv.org/html/2603.01763v2).

**Exact geometric neural integration.** Meister and Harada demonstrate geometric integration of piecewise-linear MLP control variates in two dimensions. Their 2025 result is relevant to tiny correctness oracles, not evidence that region enumeration scales to this challenge. A scalar analytic mean from cumulant propagation is also not automatically a samplewise control variate with a known exact expectation. [Primary paper](https://arxiv.org/abs/2509.15538).

**Theory wording that should be corrected in future synthesis.** RESEARCH_NOTES around line 290 states that a high-dimensional discontinuous-integrand QMC rate means Sobol can only improve constants. A one-sided error guarantee is not an impossibility theorem for faster convergence; ReLU outputs are continuous with derivative kinks. He's primary result states an upper error rate under particular assumptions and describes improved rates for favorable boundaries. Existing negative measurements remain useful; the theoretical universal prohibition does not follow. This correction does not itself reopen a sampling lane. [He, primary paper](https://arxiv.org/abs/1702.03361).

No searched primary source supplied a new, measured, drop-in solver that beats our target at the exact challenge shape and restrictions. This is a limit of the inspected evidence, not a claim that no private or unindexed method exists.

## Architecture recommendation

The intended system should have four explicit components:

1. **Trusted recurrence:** preserve the working Wick and centering equations, covariance feedback, scalar K4, normalization, births, and final readout. Treat this as one numerical operator with a frozen term manifest.
2. **Persistent K3 representation:** preserve every source age, share transported subspaces for sufficiently old sources, and retain a declared residual. Full rank must reproduce the operator before reducing rank. Memoryless regeneration of K3 is a different and previously unsuccessful mechanism.
3. **One coherent accuracy extension:** approximate additional K4 content with one state used consistently by all consumers and induced K3 births. The extension must not silently replace existing scalar K4, change the base recurrence, or combine unrelated pruning changes.
4. **Metered execution:** plan all dense/thin products and temporaries, keep the final layer to required queries, and validate actual target precision, immutable operations, capped process memory, and cold/warm lifecycle behavior.

The repository has identified this destination in GAME_PLAN. Its implementation does not yet reach it: B4 still lacks full-rank recurrence parity; no compressed candidate has passed fresh score and complete cost gates; the integrated signed-source path remains unstable. These are concrete unfinished dependencies, not reasons to restart from covariance-only propagation or to demand an exact finite probability law before testing an approximate closure. [Current plan](../GAME_PLAN.md), [architecture audit](2026-09-12_architecture_review.md).

## Smallest next discriminator

**Recommendation: resolve B4 precision versus recurrence first, with a strict stop after two working days.** Reuse the existing fixed tiny n8/depth6 fixture as an instrument test. It does not require new labels. Compare the frozen recurrence evaluated in float64 to the compact interpreter evaluated in float64 on the same f32-quantized input weights. Keep operation order and every semantic choice explicit. Do not modify frozen archives; use an adapter or separately named diagnostic.

```text
fixture = existing fixed B4 fixture
reference = frozen recurrence evaluated in float64
bridge = full-rank compact representation evaluated in float64
for each layer:
    compare transported sources, D3, D21, scalar K4
    compare each consumed Wick term and centered output
    compare new births and every retained source age
    stop at the first disagreement and save operands
compare final post-ReLU means
```

A float64 pass supports the precision explanation; it does not certify the deployed float32 trajectory. A failure identifies an algebraic or recurrence mismatch and should stop compression work until localized. Do not widen the existing tolerance to convert the failed B4 test into success. After mathematical parity, set an actual-precision gate whose tolerances are tied to the intended final error scale and conditioning. The first over-threshold `wick.(4,1)` checkpoint is the cheapest localization target reported by the existing packet. [B4 record in packet](../STRATEGIST_PACKET_2026-09-14.md).

Once parity passes, one first compression experiment should preserve recent sources, keep every old age, and apply a single shared-subspace schedule. Freeze residual treatment and all other choices. The complete program must show a material counted saving before fresh DECIDE. A proposed planning screen is at least 20% lower total counted work against F21 with no more than 5% added raw error on declared DESIGN data; this corresponds to approximately 16% score improvement above the floor, but is not the final promotion rule. The actual binding comparison must be preregistered through the repository's arithmetic score gate.

## One extension beyond the inspected implementations

**Query-constrained, output-sensitive compression** is the best bounded representation research extension. It is already proposed in the September 10 review; the additional 2025 paper strengthens its methodological basis, not its novelty claim. The concrete hypothesis is that a basis selected using consumed D3/D21 errors and final-output sensitivity can use a smaller rank than a basis selected only by carried-leg energy.

For a source-state perturbation `Delta S`, let `Q_l` denote the exact current slice query and `J_l` a declared approximation to downstream mean sensitivity. A candidate diagnostic objective is

`||Q_l(Delta S)||_scaled^2 + eta ||J_l Delta S||^2`,

with separate weights/scales for D3, D21, covariance feedback, and source births. This is a proposed approximation criterion, not an error theorem. The final mean has 1024 components; a single scalar adjoint cannot certify their MSE. A limited set of deterministic output directions or seed-derived projections may serve as a screen, but full output comparison is required for validation.

Start with fixed tiny states and equal-cost candidate bases: ordinary weighted-leg basis versus query-weighted basis. Introduce signed perturbations with equal state norm but different D3/D21 effects. Reject the idea if it cannot reduce final-output perturbation at fixed rank, or if building/querying the extra weights costs as much as the saved reconstruction. A useful proposed screen is at least 2x smaller squared final-output perturbation at equal rank on these fixtures, followed by a separately frozen rank-reduction test. Passing only proves a local discriminator, not accuracy against truth.

Do not replace this with a full nonlinear optimizer inside `predict`, or with a backward pass that still evaluates every old dense contraction. A winning implementation must lower the retained rank or eliminate a priced reconstruction. The proof-of-concept can use dense tiny objects; the live program must have explicit update, join, rotation, readout, and randomness semantics before a width1024 meter run.

## Accuracy extension and stopping condition

After a competitive compact baseline, test a **coherent approximate K4 core** against that baseline. Define the approximation independently, derive every affected raw moment and K3 birth from the same normalization, and isolate overlap with the existing scalar K4 so it is not counted twice. A covariance-aligned core is externally motivated; its coefficient must be specified theoretically or selected only on permitted fresh DESIGN networks, never copied from public-fitted tables.

The smallest falsifier is a forced-target-branch tiny test: independent dense closure versus compact closure, followed by full-depth covariance and standardized-correction diagnostics. Include the known negative-variance fixture. Algebraic parity is necessary but not sufficient; finite trajectories and meaningful covariance positivity checks must pass before a large run. An arbitrary harmonic K4 core need not itself be PSD, so test assembled raw moments/covariance rather than projecting that core as if it were a covariance matrix.

If necessary damping or PSD repair changes the approximation, register it as an intervention and include its cost. Tiny failures of an explicitly wide-only closure should be interpreted within its declared validity domain; silently disabling the branch on the test is invalid. Failure of the coherent extension should stop that implementation and preserve the compact baseline, not reopen exact signed-law construction by default.

The plausible competitive path is **compression toward the cost floor plus a meaningful error reduction**, not a promise that either one succeeds. If the compact route cannot show substantial full-program savings, or the accuracy extension cannot improve arithmetic score on a fresh decision block, retain F21 and L2 as separately validated choices while preparing reviewed contribution evidence.

This is **not a proposal to rerun the existing failed scalar-core arms unchanged**. Those arms do not become valid merely by renaming them coherent. The new object would be a closure added to the parity-proven working recurrence, with an independent dense definition of all additional moments and births, explicit removal of overlap with existing scalar K4, and a same-state audit of the covariance contribution. The previous synthetic bank was carried linearly while the rest of the state evolved through nonlinearities; the proposed closure must be regenerated or transported according to its declared approximate update instead. No complete, independently accepted inexpensive formula satisfying those requirements has been established in this review. If the Builder cannot make that distinction concrete in a bounded tiny derivation, do not launch its score experiment. The externally reported benefit motivates the derivation; it does not validate a repair.

## Higher-risk alternative: sketch the quadratic query

The specific alternative to reconstructing every old dense leg is a **polynomial feature sketch of the shared basis**, rather than a generic low-rank fit to a dense tensor. Pham and Pagh's TensorSketch approximates polynomial feature inner products using hashing and convolution; the May 18, 2025 v2 explicitly corrects its variance bound. Han, Avron and Shin apply polynomial sketches to elementwise functions of low-rank matrices without forming every matrix entry. These are relevant primitives, not challenge results. [Corrected TensorSketch](https://arxiv.org/abs/2505.08146v2), [ICML 2020 paper](https://proceedings.mlr.press/v119/han20a.html).

Here is a concrete algebraic target, derived for this assessment. Suppose the three legs of one old-source term share basis `Q`, with `A=QF`, `P=QH`, and `M=QK`. Write `q_i` for row `i` of `Q`. For a slice contribution

`D_ic = sum_a (q_i F_a) (q_i H_a) (q_c K_a)`,

define the row feature `Phi2(Q)_i = q_i tensor q_i` and conceptual core

`B = sum_a (F_a tensor H_a) K_a^T`.

Then `D = Phi2(Q) B Q^T`. This exact identity extends by summing the declared symmetrized source terms; it does not eliminate their coefficients or repeated-index corrections. The dense core is `r² x r` and is precisely what must not be materialized at full width. With a sketch `S` of size `m x r²`, the proposed approximation is

`D_tilde = [Phi2(Q) S^T] [sum_a S(F_a tensor H_a) K_a^T] Q^T`.

TensorSketch supplies the two sketched quadratic features. Retain all source-coordinate factors so the `m x r` core can be rebuilt without reconstructing `n x n` legs. In conventional arithmetic, building that core includes a term of order `h n m r`, and readout includes `n m r + n² r`; feature construction and symmetrized terms are additional. This could improve over order `h n² r` reconstruction only when `m` is sufficiently below `n`. The expression is a parameterized hypothesis, not a counted implementation.

Three issues can kill it before labels:

1. **Rotation/update closure:** when QR rotates the shared basis, the quadratic mode transforms by a Kronecker square. A generic fixed sketch cannot update that action from its compressed core alone. Rebuild from retained factors and price it, or prove a different valid update. Do not silently treat the sketch as coordinate invariant.
2. **Signed cancellation:** polynomial-kernel accuracy bounds do not guarantee small relative error for a signed sum whose true slice is much smaller than its constituent products. Test deliberately cancelling sources. More independent sketches reduce variance only by spending compute and memory.
3. **Feedback:** unbiasedness for a fixed linear query does not make the nonlinear sixteen-layer estimator unbiased. Closure coefficients and births depend on earlier sketch outputs. Fresh, seed-derived layer sketches can establish conditional identities for selected queries, but not remove nonlinear bias automatically.

A bounded first test should use fixed tiny states with 2–3 source ages, including repeated-index, diagonal, cancellation, source-birth and basis-rotation fixtures. First verify the unsketched feature identity against a separately materialized tensor oracle at float64 to a preregistered arithmetic tolerance; suggested scale-aware tolerance is `1e-10`. Then, using a fixed synthetic sketch-seed list and two frozen sketch widths, record mean query error, variance, and full-depth perturbation. A proposed screen is normalized D21 RMS at most 1% on non-cancelling fixtures, absolute error normalized by the sum of source magnitudes for cancelling fixtures, and no hidden loss of ages. These are screening choices, not universally justified tolerances; the cancellation threshold and final-output allowance must be frozen before running.

Before challenge labels, require a complete supported-flopscope update/join/read cycle whose estimated old-tier bill is at least 2x cheaper at the tested width. Include hashing/scatter/convolution support, or price the actual substitute. A dense random projection cannot retain the fast TensorSketch complexity claim. Stop if required sketch width, repetitions, or core rebuild eliminates that saving. Do not spend a DECIDE block to discover missing FFT or scatter support.

This mechanism was already proposed conceptually in the September 10 literature review. The equations above make its first oracle more specific; they do not establish priority or a new winning algorithm. A recent approximate-multiplication paper offers output-norm rather than input-norm control, but its stated unbiased squared-error bound scales as `n/r` times output squared norm; that does not justify percent-level replacement of a full signed query. It is at most another residual-only research primitive. [Uffenheimer and Weinstein, February 4, 2026](https://arxiv.org/abs/2602.04386).

## Parameterized cost and resource constraints

Let `n=1024`, `d=16`, `y` be recent full-rank sources, `h` older retained sources, `r` the shared left rank, and `k` the thin residual rank. Conventional FLOP expressions below are structural planning terms; they are not flopscope bills or additive totals without the final program's manifest.

| Component | Representative work | Live storage to inventory |
|---|---|---|
| Mean/covariance backbone | Dense `W C W^T` terms, order `n^3` per applicable layer | Several `n x n` arrays plus weights |
| Recent source legs | Order `y n^3` transport/contraction work | Order `y n^2` per leg family |
| Common basis transport | Order `n^2 r` | Old/new `n x r` basis and temporary |
| Old-source dense reconstruction | Order `h n^2 r` unless an alternative query removes it | Stream one source's reconstructed legs |
| Thin residual transport/read | Order `h n^2 k`, dependent on encoding | Left thin legs, right birth factors, query temporaries |
| Join / compression | Depends on weighted Gram, sketch, QR, and basis rotation | Must include inputs, old/new basis, and co-live workspaces |
| Final readout | Only required variances and diagonal cumulant queries | No unnecessary final full covariance or new unused births |
| Sensitivity extension | Depends on number of directions and closure derivatives | Extra pass/state retention can dominate cheap thin transport |

A tiny CPU-only differential should fit comfortably below 4 GiB and use one CPU thread; that is a planning estimate. A full-width process must prove its own 8-GiB cap behavior. The worker reservation must additionally cover orchestration, dataset/truth loading, and other processes; prefer the machine policy's 16–24 GiB envelope only when conservatively justified. Do not keep duplicated truth tensors or every layer's dense states merely for convenience. Checkpoint per completed network and preserve failed-row diagnostics. Every heavy job requires the recorded pre-submission optimization review and workerq resources.

## Evidence limits and research priorities

The new findings do not overturn the repository's strongest lesson: most errors came from changing more than one mathematical object or validating a different branch from the submitted program. The right response is a shorter sequence of discriminating tests, not another broad framework rewrite.

Prioritize the B4 precision discriminator, then complete costing of one faithful compressed program, then one coherent accuracy extension. Reserve query-sensitive compression as a bounded next representation experiment. Keep generic QMC, region enumeration, ensemble NTK recursions, unconstrained tensor-energy compression, and exact signed-law reconstruction out of the immediate critical path unless new specific evidence establishes an affordable mechanism.

No claim of originality is made for cumulant propagation, shared transport bases, K4 regeneration, output-sensitive reduction, or randomized sketches. Potential contribution lies in a newly tested combination, independent derivation, stronger controlled evidence, or a new verified contraction—not the novelty of the terminology. The private winner's architecture remains unknown.
