# Frontier follow-up — September 18, 2026

Research assessment, not a new experiment contract or a contribution-prize claim. Read CURRENT_STATE, the LOG tail, the September 18 Strategist Packet, and the earlier frontier review before this search. No truth arrays, seed blocks, estimator runs, protected files, submissions, or designations were touched. Only this document was written. Public participant measurements below are author reports, not independently replicated evidence; the inspected reports do not supply numerical truth-noise floors.

## Decision

No newly public leading implementation or primary paper was found that supplies the missing twofold gain. The strongest actionable conclusion is narrower: **stop optimizing same-state retained query energy until a one-time compression intervention identifies why the corrected objective hurts the complete recurrence.** The negative qEc result changes the priority of the earlier frontier review. Its original H1 remains a methodological possibility, but “better query energy” is now directly contradicted as a sufficient selector.

Two structural issues deserve explicit treatment: a shared ReLU mean is not evidence that centered cumulants are low rank, and cheap rows of a consumed query do not imply cheap columns. The exact input-radius identity provides a useful independent algebraic audit, but does not justify restarting sampling or replacing the retained architecture.

## What is actually new in public sources

Direct GitHub API refresh during this review returned:

| Repository | Latest default-branch commit | Interpretation |
|---|---|---|
| [504aldo/whest-p2-cumulant-k3](https://github.com/504aldo/whest-p2-cumulant-k3/commit/18c17e2d7a9aeacd399cfc2c6b571e4e16dbfb45) | `18c17e2d7a9aeacd399cfc2c6b571e4e16dbfb45`, September 10 | Same reference already inspected locally |
| [ARC cumulant propagation](https://github.com/alignment-research-center/mlp_cumulant_propagation/commit/93d091a4c26c042bfffa28f2e76a81bc0aba94bb) | `93d091a4c26c042bfffa28f2e76a81bc0aba94bb`, August 4 | Same official implementation |
| [tim8es/arc-whitebox](https://github.com/tim8es/arc-whitebox) | `2af54da98045e36b37330e1f7c9f36f0943f3661`, September 9 | Default branch alone misses research-branch and PR activity; issues/PRs were also inspected |
| [Oishi1029 Phase-1 release](https://github.com/Oishi1029/arc-whestbench-2026/commit/f58d8a9904c57c91a32ff5a5dc0588635aeb8128) | `f58d8a9904c57c91a32ff5a5dc0588635aeb8128`, August 11 | No newly disclosed Phase-2 implementation there |

504aldo's September 14 reply observes other submissions cutting roughly 7–22% of the released chain's cost, but explicitly does not know how. It also reports sponsor confirmation on September 11 that Strassen-Winograd in flopscope operations is permitted without a recursion-level limit. This is a participant's report of the clarification, not independently retrieved organizer correspondence, and introduces no new accuracy mechanism. [Author reply](https://discourse.aicrowd.com/t/everything-we-tried-a-factorized-k-3-cumulant-propagation-estimator-at-0-25-x-b-where-its-flops-go-and-25-measured-dead-ends-team-504aldo-rank-10/18218/3).

The public research repository still has PR29/PR30 open, last updated September 15. The prior review already recovered their small improvements and limitations; this refresh found no subsequent disclosed major gain. E009's future-weight sketch and E008's raw TensorSketch remain negative. [Research issues](https://github.com/tim8es/arc-whitebox/issues), [E009 result](https://github.com/tim8es/arc-whitebox/issues/17#issuecomment-5647664240), [E008 result](https://github.com/tim8es/arc-whitebox/issues/16#issuecomment-5647568257).

Two additional directly relevant negatives were recovered from GitHub comment APIs:

* Exact radial marginalization on a Phase-2 whitened-antithetic sampler was reported **3.25% worse** than its frozen reference, with zero failures; a bounded repeat reproduced the displayed result. [E003 author result](https://github.com/tim8es/arc-whitebox/issues/10#issuecomment-5647375432).
* A Gram-preserving radial fourth-moment rotation gained only about **0.41%**, below its registered 10% threshold, and was dropped. [E006 author result](https://github.com/tim8es/arc-whitebox/issues/14#issuecomment-5666609175).

These reject those sampler constructions. They do not test an analytic separation of the common input radius inside our cumulant chain. Conversely, exactness of the radial identity does not establish a useful new estimator.

The leading private method remains unidentified. Neither public FLOP utilization nor a similar raw error identifies its representation. The current leaderboard numbers in CURRENT_STATE are a separate dated browser observation; this research did not refresh them.

## Fixed weights and the shared-mean trap

Wu et al. explicitly estimate a **quenched** expectation: integrate Gaussian inputs with the weight realization held fixed. Their polynomial-activation, fixed-depth theorem is not a finite-width ReLU guarantee, and their factorization does not itself promise low rank after nonlinear recurrence. This remains the directly applicable primary algorithmic paper. [Wu et al., Sections 1, 4 and 5](https://arxiv.org/html/2605.05179v2).

A recently surfaced primary source proves low stable rank of noncentered random features, including ReLU; its mechanism is a mean-induced spike in the **second moment**. For ReLU the relevant one-dimensional moment ratio is pi. The theorem conditions on a fixed input matrix and averages over Gaussian weights. It does not prove that our centered third-cumulant history, or its repeated-index query, has small numerical rank. [When do spectral gradient updates help in deep learning?, Theorem 2.6 and Sections 2.2/6](https://arxiv.org/html/2512.04299v2).

The elementary counterexample is enough to prevent a misleading architectural leap. If `h = mu + epsilon` with centered isotropic epsilon, then `E[h h^T] = mu mu^T + sigma^2 I`. A large mean makes the uncentered second moment have small effective rank, while the centered covariance retains all dimensions. A third cumulant is unchanged by adding a deterministic mean. Therefore “ReLU outputs share a direction” cannot by itself justify discarding old K3 factors, choosing a covariance principal subspace for them, or treating final-mean error as one scalar.

Independent Phase-1 evidence also cautions against a covariance-eigenspace split: James Henry reports that a sampler's errors concentrated in the same leading directions as the signal, while his analytic chain was worse at every eigen-rank. The experiment concerned a different width/depth and probe-fed chain, so it is not a closure of our Phase-2 storage lane. It does refute the generic argument that low covariance rank automatically creates an easy analytic complement. [Author write-up, Sections 6–7](https://raw.githubusercontent.com/jamesrahenry/arc-whitebox-replication/master/WRITEUP.md).

**Useful discriminator:** when reporting a low-rank finding, name the object: raw second moment, centered covariance, individual source legs, signed sum of sources, immediate D3/D21 query, or complete future-state response. Measure the relevant object on the same incoming state. Do not transfer a rank observation between these objects without the intervening algebra.

## Exact radial separation: a diagnostic, not a proposed rewrite

The following identities are derived here from positive homogeneity and Gaussian polar coordinates; they are not a claim from a paper or a measured challenge result.

Write `X = R U`, with `R ~ chi_n`, `U` uniform on the unit sphere, and R independent of U. Every layer of a bias-free ReLU network is positively homogeneous. Define

```
rbar = E[R]
A = R / rbar
Y_l = h_l(rbar U)
H_l = h_l(X) = A Y_l
a_k = E[A^k] = 2^(k/2) Gamma((n+k)/2) / (Gamma(n/2) rbar^k)
```

Here A is independent of the entire angular trajectory. Thus `E[H_l] = E[Y_l]` at every layer and every raw k-th joint moment differs by the known multiplier a_k. If mu, C, and K3 denote the angular mean, covariance and centered third cumulant,

```
Cov(H) = a_2 C + (a_2 - 1) mu mu^T
K3(H)_ijk = a_3 K3(Y)_ijk
            + (a_3 - a_2)(mu_i C_jk + mu_j C_ik + mu_k C_ij)
            + (a_3 - 3 a_2 + 2) mu_i mu_j mu_k.
```

This identifies a genuine common random component, unlike merely subtracting a deterministic mean. Its third-order repeated-index consumer is inexpensive:

```
K3(H)_iij = a_3 K3(Y)_iij
            + (a_3 - a_2)(2 mu_i C_ij + mu_j C_ii)
            + (a_3 - 3 a_2 + 2) mu_i^2 mu_j.
```

But the angular input is non-Gaussian. With `q = rbar^2/n`, it has covariance `q I` and fourth cumulant

```
K4(Y_0)_ijkl = -2 q^2/(n+2)
               * (delta_ij delta_kl + delta_ik delta_jl + delta_il delta_jk).
```

Consequently, removing a positive radial contribution while keeping a Gaussian initial state is wrong. At input the positive radial mixture and negative angular fourth cumulant cancel exactly to give Gaussian K4=0. Higher even angular cumulants also exist. ARC's existing harmonic/radial cumulants already capture some related structure: a new “radial channel” must first establish a missing term, rather than add a duplicate.

**Bounded use:** on exact tiny moments, verify the raw-moment conversion through order four; test identity weights, a single ReLU, and a two-layer network; prove exact input cancellation and the repeated-index formulas. Audit whether the current core/birth bookkeeping already contains these terms. Stop if this is merely a change of coordinates or an already present harmonic component. A spherical-input chain would require rederived nonlinear and birth rules and is outside the retained next-slice scope. The public radial-sampler negatives and the existing Gaussian-scale-mixture negative give no reason to fund such a rewrite now.

## Query interpolation has an asymmetric access problem

The previous frontier review proposed cross approximation of the signed aggregate D21. DEIM/Tucker cross methods use selected fibers to avoid materializing full tensors, but their savings assume suitably cheap fiber access and useful numerical ranks. [Ghahremani and Babaee](https://arxiv.org/abs/2401.04249).

Here is the concrete missing price, derived from the current source representation. For a representative term

```
E = (A * P) A^T,       A = Q F_A, P = Q F_P,
Q: n by r,            F_A,F_P: r by n,
```

selected **rows** can be cheap. To compute `E[I,:]`, reconstruct A and P only at k selected rows, then multiply `(A[I,:]*P[I,:]) F_A^T Q^T`. Conventional leading arithmetic is `O(k n r)` per source, followed by accumulation. Birth coefficients can be included in the elementwise factor.

Selected **columns** are different. `E[:,J]` requires `(A*P)` at every row. The straightforward implementation first reconstructs full A and P for `O(n^2 r)` per source—the expense that interpolation was supposed to eliminate. A fully reduced alternative forms quadratic matrices `F_A diag(A[j,:]) F_P^T` and evaluates them at all Q rows, costing `O(k n r^2)` per source in the straightforward spelling. Choosing k small does not automatically make this cheaper; r is currently hundreds. These are leading-order algebraic costs, not an official meter result, and clever shared contractions could change them.

Therefore, **do not build CUR on an exact dense E and count only its small inverse**. Its attractive row oracle is real; its column oracle remains the bottleneck. The signed sum must be formed before assessing rank because sources can cancel, but aggregation alone does not make the column quadratic forms disappear. D3, all source ages, selection, reconstruction, basis changes and fallbacks remain payable.

HaTT is an additional primary-source lead: it recompresses a Hadamard product of tensor-train tensors without first materializing the expanded TT product. This shows that rank multiplication need not always be explicitly paid in storage. It does not establish a cheap representation of our three-mode tensor, an affordable CP-to-TT conversion, or low TT ranks of the consumed aggregate. [Sun et al., HaTT](https://arxiv.org/html/2410.04385v1).

**Stop condition before implementation:** write the actual signed-source row/column oracle and count its operations at n=1024 and the frozen r. If full leg reconstruction or `r^2` quadratic forms erase the saving, reject this spelling before truth. A generic tensor library or a binary reshaping of neuron indices is not evidence that the required ranks will shrink.

## One sharper next discriminator: locate the compression-induced feedback path

This is a proposal for a subsequent committed contract, not authorization to measure. It refines the earlier residual-enrichment idea using the completed negative evidence.

**CONFIRMED repository fact:** at fixed age4/rank256, qEc improved its own same-state retained energy but worsened complete E error, D3, D21, continuation and paired truth score. The source repair is correct; usefulness is false at that setting. The own four-parameter fit improved its held-out DESIGN comparison, but changed several mechanisms together and did not identify the harmful storage path. [Current Strategist Packet](../STRATEGIST_PACKET_2026-09-18.md).

**Question:** does the corrected basis lose immediately on the genuinely consumed complete state, or does a smaller immediate defect get amplified by covariance/gate/birth feedback or repeated re-compression?

**Smallest intervention:** at one preregistered join, capture a single complete incoming state from the frozen control. Make three copies: no compression at this join, old qE at this join, corrected qEc at this join. Thereafter continue each with the same complete uncompressed recurrence, preserving its own evolving moments and births. Do not repeatedly overwrite a state, rescale accumulated history, or substitute incoming states from three unrelated chains. Repeat only the predeclared early/middle/late join locations, not a searched layer grid.

**Two supplementary diagnostic continuations:** (a) freeze later nonlinear coefficients and births to the no-compression trajectory, transporting only the perturbed historical tensor; (b) restore endogenous covariance/gate/birth feedback, but suppress later joins in all arms. The first is deliberately an artificial diagnostic, not a candidate estimator. Compare it with the fully endogenous continuation to identify the feedback amplification. A normal repeated-join rollout is then interpretable relative to the isolated one-time intervention.

**Report actual objects:** immediate signed E/D3/D21 errors; next mean and full covariance defects; change in the new-source contribution; two-step defects; final 1024-dimensional mean displacement. Preserve source identities and the same incoming fourth core. Compare final-vector squared norms and alignment, not merely magnitudes of unrelated states. Test a cancelling-source fixture and an exactly linear recurrence first; the latter must reproduce the analytically propagated single perturbation.

**What it can establish:** whether a basis objective misses a particular state-feedback path. If freezing feedback reverses the old-qE/qEc ordering while the complete recurrence restores the failure, a residual-enrichment target becomes concrete: the omitted contribution to that operator, rather than total same-state query energy. If qEc is already worse in the immediate complete-state consumer, fix the claimed objective or its weighting first. If no stable ordering occurs across the predeclared joins, stop this attribution and do not invent a universal feedback story.

**What it cannot establish:** that the uncompressed recurrence is closer to truth, that eliminating compression error improves challenge score, or that an online selector is affordable. The full uncompressed reference is a diagnostic expense. A later shippable basis selector must price its own residual oracle, extra columns, joins, transport, nonlinear queries and operation overhead. An offline Jacobian or exact reference trajectory cannot become free online information.

The predictor-corrector low-rank literature supplies a precedent for enriching a basis using a defect rather than its current reconstruction energy; its matrix-ODE assumptions do not transfer automatically to this discrete nonlinear cumulant chain. [Hauck and Schnake](https://arxiv.org/abs/2209.00550v2). This is a mechanism test, not a theorem-backed promise of lower challenge error.

Keep the old qE source as the comparator; do not promote qEc because its algebra is cleaner. Freeze one subsequent intervention only after the attribution and complete cost model exist. Any new accuracy decision requires a fresh registered block, Samuel's prediction, the paired arithmetic score difference through competition_score, numerical truth-noise reporting and independent Skeptic. No basis/rank sweep, P13 spend or release claim follows from this review.

## Alternatives ranked by the evidence

| Alternative | Status after this follow-up |
|---|---|
| Complete-state, one-time compression-defect attribution | Best bounded diagnostic; distinguishes a concrete feedback mechanism without truth |
| Residual-enriched old-history coordinates | Conditional on attribution and an affordable online oracle; not a proven improvement |
| Exact radius/angular cumulant bookkeeping | Useful independent algebraic audit; a replacement chain is unpriced and may duplicate existing radial terms |
| Signed aggregate cross interpolation | Valid representation idea with a newly explicit column-access obstacle |
| Hadamard-avoiding TT recompression | Real external algorithm, but unsupported rank/conversion economics here |
| Shared-mean/rank-one approximation | Wrong object unless centered cumulant/query evidence is supplied |
| Future-weight sketch, raw TensorSketch, radial sampler restart | Direct participant counterevidence; no new basis to replay them |
| More nonlinear terms or another imported lambda table | Completed local evidence already rejects treating either as the missing explanation |

The plausible next major improvement still requires both a better truth-error mechanism and cheaper complete recurrence. Literature found in this pass narrows false shortcuts and makes a diagnosis more concrete; it does not supply an established winning replacement.
