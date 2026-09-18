# Deep frontier research — September 18, 2026

Research assessment, not an experiment contract, promotion, contribution-prize claim, or authorization to reopen a closed lane. CURRENT_STATE and the LOG tail were read first, followed by the existing September 10/14/16/18 literature reviews, latest complete-slice packet, and C10 source. No truth arrays, new seeds, estimator execution, candidate edits, submissions or designations were used. The parent review owns the fresh Chrome leaderboard snapshot. Only this research document was written.

## Decision

The retained architecture is the best-supported family found: full means and covariance; persistent factored third-cumulant history; a coherent approximate fourth-order core; economical transport and exact consumed queries where approximations have failed. Neither the research literature nor the inspected public implementations validates the current implementation as a winner. Most obvious literature directions were already researched thoroughly by this repository. A different name for low-rank compression, more Edgeworth terms, or an adjoint is not a new mechanism.

The most concrete next intervention surfaced by the parallel source audit is **preserving the known newborn third-cumulant diagonal through P2 residual compression**, using the already-carried diagonal channel. Conservative low-rank literature supplies a relevant methodological precedent. It does not promise a challenge gain, and old matched-chain evidence suggests this could be an incremental correction. Treat it as a cheap, sharply falsifiable intervention, not an explanation of the entire gap.

If it fails, the useful frontier is a complete-state, one-time compression defect attribution, followed only conditionally by residual enrichment. There is no presently evidence-backed single change that can be promised to move C10 to first place.

## The actual numerical gap

C10's accepted P12 DESIGN result is raw MSE `3.1837142e-8`, with truth-noise floor `2.2480710e-9`; official-precision adjusted estimate `6.4866090e-9`, at `480,850,933,121` counted FLOPs (`~0.21867 B`). The same-panel v31 raw MSE is `3.1607591e-8`: C10's 1.366x adjusted gain is chiefly reduced cost. The complete-slice packet, not a cross-board ratio, supports this statement.

Its effective error under the existing official-precision projection is about `2.97e-8`. Even if exactly the same predictions could be computed at the score multiplier floor, the projected score would still be about `2.97e-9`. Thus **cost reduction alone cannot reach a leader near 2.2e-9**. At the floor, a 2.2e-9 planning target requires effective MSE at most 2.2e-8; at a 0.13 multiplier it requires about 1.69e-8. These are score-identity calculations, not paired promotion evidence, and the current public leader and our DESIGN panel are different networks.

The failed rank32 old-query intervention saved only 5.514% of complete FLOPs and made every paired P12 score worse. The six-arm experiment shows P3's extra accuracy does not repay its complete cost when added to fitted C10. Any next plan must target recurrent error as well as count work; it must not multiply separately selected historical gains.

## What the primary literature actually supports

**Direct problem match.** Wu et al. estimate Gaussian-input expectations with a fixed realized network, exactly the relevant conditioning direction. Their theoretical rates assume fixed depth and polynomial activations; the ReLU experiments do not turn these into a width1024/depth16 guarantee. Their Appendix S.6 treats the non-polynomial diagram expansion as asymptotic, and Appendix D leaves depth robustness unresolved. Increasing order is not monotonically beneficial. This remains the strongest directly applicable primary paper found, already central to this repository. [Wu et al., 2026, v2](https://arxiv.org/html/2605.05179v2).

**Exact Gaussian primitives.** Wright, Nakahira and Moura provide analytic Gaussian covariance propagation. These primitives improve a Gaussian backbone; they do not supply missing fixed-network higher-order dependence. A possible replacement must subtract the Gaussian terms already present before adding a more exact Gaussian expression. Historical Mehler convergence and the public exact-covariance participant result give no basis to expect that alone to close the current gap. It is a possible local error audit, not the recommended architecture replacement. [Wright et al., AISTATS 2024](https://proceedings.mlr.press/v238/wright24a.html).

**Preserve known observables during compression.** Einkemmer, Ostermann and Scalone explain that ordinary low-rank methods can violate physical balances. Their conservative method protects the required moment subspace while compressing the remainder, obtaining specific conservation and robustness results for kinetic equations. This is a methodological basis for separating exact cumulant slices from a compressible residual; it is not a theorem about ReLU recurrence error. [JCP 2023 / arXiv:2206.09374](https://arxiv.org/abs/2206.09374).

**Separate macroscopic moments from a low-rank remainder.** Guo and Qiu's LoMaC method evolves required macroscopic quantities and projects the kinetic low-rank representation to agree with those observables. The relevance is the architecture pattern: keep cheap quantities known accurately outside a lossy representation. Our K3 diagonal is a slice constraint, not a conserved mass or energy, so their equations cannot be transplanted. [Guo and Qiu, 2022/2024](https://arxiv.org/abs/2207.00518).

**Current frontier in that direction.** Cui and Scalone's August 2026 stochastic extension explicitly splits protected and remainder components before truncation; its Section 3.3 keeps the fixed moment modes exactly and truncates the rest. The paper also shows why protecting the right continuous quantities is insufficient unless the discrete update includes the required spaces. That is relevant to checking birth, carry, join and readout separately. Its PDE guarantees and numerical gains do not certify this cumulant chain. [Cui and Scalone, 2026, Section 3.3](https://arxiv.org/html/2608.00397v1#S3.SS3).

**Residual enrichment, conditionally.** Hauck and Schnake's predictor-corrector rank adaptation motivates finding missing update directions rather than choosing a basis only by reconstruction energy. The repo already identified this paper. A useful challenge version needs a cheap residual action for the actual consumed D3/D21 and its covariance/birth feedback. A full exact residual followed by a cheap SVD is not a cheap online method. [Hauck and Schnake](https://arxiv.org/abs/2209.00550v2).

The broad search also found maximum-entropy closures, distributional neural kernels, finite-width ensemble expansions, Stein control variates, and tensor cross approximations. They do not establish an affordable replacement under this problem's conditioning and score. Maximum entropy does not uniquely recover the true distribution from four moments; solving its inverse problem adds work and can have realizability difficulties. Annealed results average over weights, whereas the challenge holds weights fixed. Generic variate methods do not evade the required residual correlation or their own evaluation cost. These are reasons to avoid a new broad rewrite, not universal impossibility claims.

## Next bounded hypothesis: preserve the newborn diagonal

This candidate mechanism was identified by the parallel architecture audit and checked against the current source spelling here. It is an explicit alternative finite-rank approximation, not a claim that every current score was invalid.

The P2 encoding is

```
A_D = M diag(sb) + 3 A_L diag(eb) + Z L^T.
```

At birth `M=I` and the relevant `A_L` diagonal is zero. The intended residual matrix has zero diagonal, but ordinary finite-rank approximation need not preserve it. `_p2_compress` returns factors with the factor 3 already in Z. C10's ordinary branch then keeps `sb=S3`. Consequently the represented newborn diagonal contains the leakage

```
d = diag(Z L^T) = rowsum(Z * L)
represented D3 at birth = S3 + d.
```

The smallest proposed intervention is

```
sb = S3 - rowsum(Z * L).
```

This removes only the unintended diagonal part of the approximated residual using the channel already present. It does not need a dense new diagonal matrix, more rank, a new history bank, a sampled correction, a fit, or a new query approximation. At full residual rank the correction should vanish to arithmetic precision because the intended residual diagonal is zero. The existing protected-feed branch already compensates the diagonal of its added exact factors, but that is not the same as compensating leakage from the compressed residual.

The algebraic distinction from August28's failed generic tensor-diagonal protection matters. That historical scheme reserved tensor approximation capacity and failed its matched-rank comparison. This proposal changes the P2 birth decomposition using the existing diagonal channel, at the same retained factors and rank. It must not be presented as proof that all diagonal protection is good. The architecture audit must establish the full tensor normalization and ensure no companion subtraction already cancels this leakage.

**Cost model before measurement:** one elementwise n-by-r multiplication, one row reduction, one vector subtraction per applicable birth, plus a temporary n-by-r array if not fused. For n=1024/r=32, leading conventional work is about `2nr` operations per birth, about one million over sixteen births, and a float32 temporary is 128 KiB. This is tiny relative to 480.85 billion FLOPs, but complete flopscope and operation-call counts still govern. No server residual or Linux memory pass follows from this structural calculation.

**Focused checks:** independent dense tiny-tensor birth diagonal, both D21 orientations, complete source normalization, zero-residual case, full-rank recovery, permutation relabeling, one linear carry, one nonlinear carry, and existing hub/P3-off compensation. An isolated no-hub fixture is not enough to certify C10. A source-specific before/after test must fail the old finite-rank diagonal assertion and pass the proposed one; a full-rank parity-only suite misses the mechanism.

**Smallest recurrent discriminator:** freeze exact C10 and a compensated clone with all rank, basis, lambda, table, feed and P4 choices unchanged. First compare a single intervention at fixed early/middle/late birth locations, continuing the same complete endogenous recurrence thereafter; then assess the all-birth candidate. Report immediate D3 leakage removal, ensuing D21/covariance changes and final-vector displacement. A lower displacement from an uncompressed reference is a diagnostic, not truth evidence.

**Truth and stopping rule:** a subsequent committed contract must choose the fresh DESIGN or DECIDE tier, exact gate and observation budget before truth, collect Samuel's prediction, and use paired arithmetic score differences with the numerical truth floor. One fixed compensated arm is enough; do not open a rank or damping grid if it fails. P13 remains reserved for a frozen major-release case. An independent Skeptic reviews the algebra, exact bytes and full score evidence before any promotion.

**Expected scale:** unknown, with material adverse prior evidence against a major claim. On the older v31 chain, the LOG's September15 matched 24-network dense-P2/rank32 comparison reports only about a 0.90% raw-error penalty from rank32. The entire approximation penalty there is far below the current roughly31.7% C10 reduction required merely to meet half-v31. A much larger C10 gain would require a demonstrated changed feedback interaction. Do not promise it.

## A larger program if the cheap invariant test is insufficient

The repository's previous one-time compression-defect proposal remains more defensible than another blind rank choice. Separate birth residual truncation, shared-history join, old D21 readout and approximate K4 closure. For one chosen operator, compare exact/no-intervention versus the current approximation on the *same incoming state*, then continue with its own endogenous covariance, gates and births. Frozen-coefficient continuations may attribute amplification, but are artificial diagnostics.

The next candidate should preserve an identified harmful output of that operator at constant or lower complete cost. The external conservative-low-rank papers motivate keeping selected exact constraints outside the lossy residual; they do not tell us which constraints determine final MSE. The failed qEc result already proves that higher immediate query energy is insufficient. A complete n-dimensional final-mean vector and both recurrent consumers matter; a single scalar adjoint or next-weight sketch is inadequate evidence.

No closed path is reopened here. Raw TensorSketch, the published next-weight sketch, radial sampler restart, memoryless D21, recent-only history, unpriced Tucker/CUR readout, generic output correctors and a larger nonlinear table keep their prior scoped negative status. The leading private architecture remains unidentified. The novel contribution worth pursuing would be a cheap verified moment-consistent compression with measured full-recurrence score improvement, not the established phrase “structure-preserving low rank.”

The August2026 paper supplies a sharper external reason for a *multi-step consistency diagnostic*: its conservation proofs require particular functions to remain in the updated approximation space after augmentation and truncation; retaining their current moments alone is insufficient. The repo-specific analogue would explicitly include the image of a proposed protected slice under one next linear carry and nonlinear source-birth map. This is distinct from qE's one-step retained-energy maximization. It is still only an analogy until the relevant image and its online cost are derived; an exact full tensor or free dense Jacobian would defeat it. If a protected set is not closed under the next map without rank growth, that is a decisive cost obstacle, not permission to silently truncate it again.

## Ranked mechanisms and decision value

| Priority | Mechanism | Evidence and upside | First stopping condition |
|---|---|---|---|
| 1 | P2 newborn diagonal compensation through existing `sb` | Concrete source identity plus conservative-truncation precedent; cheap diagnostic and possible incremental gain, no credible major-gain forecast | No actual leakage after full normalization, harmful full score, or no material gain |
| 2 | One-time full-state compression-defect attribution | Directly answers qEc/rank32 recurrence failures; research diagnostic, not a candidate | No stable causal path or effect too small to explain target gap |
| 3 | Moment-constrained multi-step residual compression | Higher potential upside if it preserves the harmful update at substantially smaller rank; no demonstrated affordable representation | Protected next-step image needs dense tensors, full-rank queries or costly online derivatives |
| 4 | Exact Gaussian subexpression replacement with overlap subtraction | Existing exact formulas; low-cost local audit possible, but Gaussian-only public gains and historical series convergence argue against major upside | Correction below recurrent error scale or significant calls/cost for negligible full gain |
| Deferred | Resummed scalar laws, generic maximum entropy, radial rewrite, tensor cross or sketch restart | No new directly applicable favorable challenge evidence; known conditioning, cost or negative-result obstacles | Do not implement without a separately justified operator and external reopening evidence |

This ranking distinguishes work likely to clarify the problem cheaply from work with hypothetical large upside. It does not turn priority3 into a licensed experiment or make priority1 a promised major submission.

## Practical recommendation

Keep C10 as the frozen strongest complete DESIGN package. Make the birth-diagonal hypothesis concrete in a new narrowly scoped contract, then stop or promote on its measured result. If it is merely incremental, record that and undertake one operator-attribution experiment before creating another compressed representation. Retain the current reliable mathematical family; restructure implementation around explicit state and compression invariants only where that removes a demonstrated failure mode. No source found supports claiming that additional submissions, by themselves, will discover a winning architecture faster than a small number of discriminating offline experiments and candidate-specific server checks.
