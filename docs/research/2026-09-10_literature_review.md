# Literature and public-code review — 2026-09-10

This is a research evidence packet, not a promotion, experiment result, or submitted write-up claim. No network was minted, no decision block or vault was read, and no external code was executed for this review. Existing repository files were read only. The central new evidence was found by the parent agent in Samuel's Chrome session and independently checked here against the public source.

## Finding that changes the decision

**There is now public, inspectable Phase-2 code that demonstrates a substantially cheaper cumulant architecture than our current representation.** Team 504aldo published its implementation and findings on September 9–10. Its reported V29 submission is `330093`, adjusted `5.40e-9`, raw `2.13e-8`, cost `0.2526 × 2^41`. These are the author's published/graded reports, not a fresh independent reproduction. A ground-truth noise floor is not supplied alongside those reported rows and must therefore be recorded as unavailable; do not compare tiny differences to our noise-corrected local figures. [Published project](https://github.com/504aldo/whest-p2-cumulant-k3), [forum publication](https://discourse.aicrowd.com/t/everything-we-tried-a-factorized-k-3-cumulant-propagation-estimator-at-0-25-x-b-where-its-flops-go-and-25-measured-dead-ends-team-504aldo-rank-10/18218).

Read alongside our September 10 results, this supports a very specific conclusion: the unsuccessful T1 deletion experiment and the unsuccessful packed-matmul experiment do **not** establish that every affordable higher-cumulant representation has failed. They establish failures of those interventions. This newly available implementation supplies the external evidence required by our closed-lane rule to propose a new, narrowly scoped contract.

The right near-term action is an independent mathematical reconstruction and fresh-data test of this published architecture. A speculative new method should not displace that work. Reproducing it would still not establish a winning solution: the live leaderboard has continued improving, and the parent agent's September 10 Chrome audit reports a leader around `2.7e-9`, with raw about `1.71e-8` at a multiplier about `0.158`. Consult the dated competition audit for the live row and timestamp; the older `0.18–0.21` envelope is no longer the frontier.

## Source provenance and freshness

| Source | Date/version checked | What was actually inspected |
|---|---|---|
| Wu et al., *Estimating the expected output of wide random MLPs more efficiently than sampling* | arXiv v1, May 6, 2026 | Abstract, problem definition, methods, supplement map, theory qualifications |
| ARC reference implementation | `93d091a4c26c042bfffa28f2e76a81bc0aba94bb`, August 4, 2026 | Current main commit history from GitHub API; `factor_k3.py`; README |
| 504aldo public implementation | `18c17e2d7a9aeacd399cfc2c6b571e4e16dbfb45`, September 10, 2026, 03:24:28 UTC | README; V16b/V29 source; V17/V18/V22/V24/V25 generators; derivation map; ablations; claims ledger |
| 504aldo publication | September 9 forum post, September 10 repository state | Discovered in Chrome by parent; source independently read here |
| Adjacent mathematical literature | 2014–2026 original papers listed below | Primary abstracts/theorems and available code descriptions; no experimental reproduction |

ARC's current commit remains the version already pinned by our bridge. There is no newly discovered upstream update to import. The July 31/August 1 correction history matters: old term enumerations and cached figures were invalidated, and factored AUGMENT was redefined as the factorable subset of the augmented diagram set. Consequently, factored AUGMENT, unfactored AUGMENT, and full K4 must remain separate named controls. [ARC commit history](https://github.com/alignment-research-center/mlp_cumulant_propagation/commits/main/), [August 1 implementation change](https://github.com/alignment-research-center/mlp_cumulant_propagation/commit/56f736a7e74c60d9714fa06bd5cb6bf48217a6d8), [current factor implementation](https://github.com/alignment-research-center/mlp_cumulant_propagation/blob/93d091a4c26c042bfffa28f2e76a81bc0aba94bb/src/mlp_kprop/factor_k3.py).

Searches covered cumulant propagation, the exact challenge title, quenched neural-network expectations, uncertainty propagation, streaming tensor compression, unbiased recurrent low-rank approximation, finite-horizon model reduction, and weight-space equivariant learning. This is a targeted review, not a proof that no unpublished or unindexed method exists. Search-index snapshots of GitHub were stale; direct GitHub API/commit reads were used for the two primary implementations.

Two organizer talk links were also supplied by the parent agent from the September 1 discussion: [video 1](https://www.youtube.com/watch?v=_BPsuomAcoI) and [video 2](https://www.youtube.com/watch?v=N0hhsESjKEs). Fetches were throttled and no transcript was verified. They are pointers only; this review does not claim their contents were watched or analyzed.

## What the papers establish, and what they do not

Wu et al. directly study the required conditional expectation for a **fixed weight realization**, then average estimation error over random weights. Their cumulant/Hermite construction is therefore directly relevant. The formal efficiency theorem assumes fixed depth and polynomial activations; ReLU performance is empirical. It is not a theorem that a particular K3 implementation wins at `1024 × 16`, nor that a paper FLOP polynomial equals flopscope billing. [Paper](https://arxiv.org/abs/2605.05179), [full text](https://arxiv.org/html/2605.05179v1).

Wright, Nakahira and Moura give analytic Gaussian covariance propagation through nonlinearities, including ReLU. This supports a reliable Gaussian baseline and exact Gaussian subexpressions. It cannot restore non-Gaussian cross-neuron information discarded by a Gaussian closure. Our K2 versus K3 evidence already identifies that distinction. [AISTATS 2024 paper](https://proceedings.mlr.press/v238/wright24a.html).

Hanin's finite-width cumulant analysis provides tools for tracking non-Gaussian corrections and depth amplification. Such random-network theory must be read with its averaging convention: ensemble cumulants are not automatically the conditional cumulants over inputs that the challenge requests. An annealed formula may be a prior or closure ingredient; it is not a substitute for weight-conditioned propagation. [*Cumulants in Random Networks*, JMLR 2024](https://jmlr.org/papers/v25/23-0643.html).

The Phase-1 higher-moment atlas is useful as historical evidence, not a ready Phase-2 training set. Its card states width 256, depth 32 and 100 million samples per MLP, with both pre- and post-activation moments. It does not match Phase 2's shape, and our public-data policy precludes fitting on public challenge realizations. [Dataset card](https://huggingface.co/datasets/keenanpepper/arc-whestbench-higher-moments-2026).

## The new public architecture, decomposed into separate claims

### 1. Two shared dense legs plus a thin residual

The published source expresses a birth generation using transported matrices `A` and `P`, static column coefficients, and a thin carried residual. The third dense leg is reconstructed algebraically as

```text
M = P diag(s) + 3 A diag(e) + Z Lᵀ.
```

Here the normalization of `Z Lᵀ` includes the residual's coefficient. Substitution combines the dominant `(2,1)` contractions into two products with right operands `Aᵀ` and `Pᵀ`, plus thin terms. Both dense legs share the same subsequent transport `W diag(w1)`. This is a real algebraic change in what is multiplied; it does not rely on persuading the meter that distinct arrays are identical. [V16b source](https://github.com/504aldo/whest-p2-cumulant-k3/blob/18c17e2d7a9aeacd399cfc2c6b571e4e16dbfb45/estimators/estimator_v16b.py).

**Critical limitation:** the whole published chain is not an exact implementation of every upstream SIMPLE term. Its source separately uses finite-rank residuals, removes selected blocks, and prunes a term table. Later versions restore some feedback in thin form. Therefore the experiment must separate algebraic parity from approximation error. Calling the entire V16b/V29 chain “exact K3” would repeat the kind of attribution error the Skeptic just corrected here.

Our packed-spelling negative inspected the operands of our existing representation. This representation changes those operands. It is outside that negative's scope.

### 2. Fourth-order information regenerated consistently with its births

The public method approximates the off-diagonal part of a fourth-cumulant harmonic matrix core by `lambda × C_off`, retaining its separate diagonal `g`. Its transported diagonal is

```text
diag(W G Wᵀ) = (W ⊙ W) g
             + lambda [diag(W C Wᵀ) - (W ⊙ W) diag(C)].
```

The identity is exact for the stated approximate core. Closure occurs in choosing that core and in omitting other transported content. The implementation then uses the **same** coefficient and diagonal in the nonlinear slices and in K4-induced K3 births, reducing those births to existing dense legs plus thin carried vectors. Its coefficient tables were fitted on eight public MLPs. [V17 generator, including birth formulas](https://github.com/504aldo/whest-p2-cumulant-k3/blob/18c17e2d7a9aeacd399cfc2c6b571e4e16dbfb45/generators/make_v17.py).

This is the key distinction from our T1: replace the omitted object with a jointly specified closure **before deriving both its consumers and its births**. Do not delete births from one algorithm while preserving consumers derived for another. Our September 10 result remains valid; it does not test this closure.

The source reports the full regenerated channel nearly halving error, while removing its K3 feed makes error much worse and regenerating the diagonal also fails. Those are author-reported ablations on a small repeatedly used public panel, not our decision evidence. The claimed benefit deserves replication, not automatic promotion. [Ablation table](https://github.com/504aldo/whest-p2-cumulant-k3/blob/18c17e2d7a9aeacd399cfc2c6b571e4e16dbfb45/docs/ablation_table.md).

### 3. Retain every generation, share the left transport space

Old generations are retained as `A_s = Q F_s`, `P_s = Q H_s` with a common transported basis `Q`; the oldest subset uses a nested basis inside it. Birth identities remain in the right-coordinate matrices. Current public choices are age 4/rank 384 and age 7/rank 224. These are empirical choices, not universal constants. Contractions can be reassociated to sum in the smaller coordinate space before a single trailing multiplication by `Qᵀ`. [Derivation and code map](https://github.com/504aldo/whest-p2-cumulant-k3/blob/18c17e2d7a9aeacd399cfc2c6b571e4e16dbfb45/docs/derivation_code_map.md).

Our generation caps delete old information; this method preserves all ages approximately. Our column selectors choose birth coordinates; this method projects the current shared left range. Our `Sigma` selector projects the covariance range; this method constructs a range from the weighted **actual carried legs**. These are different interventions. The existing failures are useful controls, not reasons to refuse the published counterexample.

The source's claims ledger also records failures of source windowing and memoryless K3 slice replacement. That agreement with our negatives is reassuring: preserving K3 history and compressing its transport can both be true. [Claims and evidence](https://github.com/504aldo/whest-p2-cumulant-k3/blob/18c17e2d7a9aeacd399cfc2c6b571e4e16dbfb45/docs/claims_evidence.md).

### 4. Runtime spelling is a separate, constrained layer

V29 batches related product families and uses buffer-backed Strassen operations. It contains extensive `out=` and `copyto` calls. Our current repository contract says arrays are immutable; a public source is not authorization to override that rule. First establish precisely what the installed and official flopscope implementations permit for estimator-owned buffers versus weight arrays. Then obtain the required repository decision or build an immutable version. A public successful submission does not by itself establish compliance or identical behavior on our runner. [V29 source](https://github.com/504aldo/whest-p2-cumulant-k3/blob/18c17e2d7a9aeacd399cfc2c6b571e4e16dbfb45/estimators/estimator_v29.py).

Do not adopt the external README's local-runner recommendation, its source-code “no randomness” wording for weight-derived sketches, its broad package-version range, or its public fitting practice as our protocol. External code is a reference, not a policy.

## Recommended reconstruction sequence

All thresholds below are **proposed design screens**, not accepted promotion gates. Before strategic work, write the exact contract, record Samuel's prediction, claim fresh blocks through the ledger, commit the contract, and use the binding arithmetic-score decision function. This review does not authorize touching any previously held-out artifact.

1. **Construct a clean reference representation.** Pin the public commit and preserve MIT attribution. Implement the shared-leg identities with full residual rank and the complete chosen term set. Compare every relevant slice against the existing trusted chain on tiny hand-constructed arrays before scoring networks. Preserve orientation conventions, repeated-index subtraction, and the distinct SIMPLE/factored-AUGMENT definitions.
2. **Price the exact identity separately.** Produce an operation-family ledger for transport, slice extraction, births, nonlinear terms and final readout. Count conversion, QR, reconstruction, buffers and setup. This can eliminate unaffordable proposals without spending labels. No claimed 7-to-4 ratio should be applied to the entire estimator without that ledger.
3. **Define the coherent K4 closure from equations.** Every use and every K3 birth must consume one consistent `(g, lambda, C)` state. Re-estimate any needed coefficient on newly minted design networks, not the published public-fit table. Start with fixed structural settings; defer the adaptive coefficient, whose reported effect is small.
4. **Measure thin residual approximations on that closure.** Full-rank residual is the control. Screen ranks against end-to-end arithmetic score, not matrix energy alone. Restore/omit blocks as explicit arms. Do not import data-selected term pruning invisibly.
5. **Add one shared old-source tier, then the nested tier.** Preserve all births and exact current read slices. Fit rank/age choices only on design data; confirm the frozen schedule once on fresh decision data. Separate low-rank approximation from algebraic reassociation.
6. **Only then engineer deployment.** Shape-generic `fnp` implementation; exact package pin; read-only f64 grader weights; counted FLOPs; cold/warm subprocess runs; Linux 8-GiB enforcement; residual headroom and whole-process wall safety. No proposed score matters if any MLP fails.

The architectural destination is a full mean/covariance state, compact **persistent** K3 source state, a consistent cheap K4 core closure, and a final diagonal-only readout. It is close to the repository's correct mathematical instinct, but materially different from both the expensive general tower and the memoryless regenerating K3 flow.

## Research beyond the newly published baseline

These are hypotheses, not published challenge results. “Novel” means a concrete proposed application beyond the implementations inspected here, not a claim of priority over all literature or private entrants. Reconstruct the published baseline first. The first candidate below is a representation feasibility question; the second is a modest improvement hypothesis, and the last two are lower-confidence alternatives.

### A. Sketch quadratic products before old-source reconstruction

**External foundation.** Han, Avron and Shin construct approximate low-rank representations of entrywise functions of low-rank matrices without evaluating all dense entries. TensorSketch approximates polynomial feature products with implicit sketches. Its original variance analysis was corrected in the May 2025 version, which must be used for any bound. [ICML 2020 paper](https://proceedings.mlr.press/v119/han20a.html), [Pham and Pagh, corrected TensorSketch paper, May 18, 2025](https://arxiv.org/abs/2505.08146v2).

**Proposed application.** The costly operation in the shared old tier is reconstructing `Q F_s` and `Q H_s` for every source so their Hadamard products can be formed. Algebraically its contribution can instead be written as a contraction of the row-quadratic features `q_i ⊗ q_i`, a merged third-order core in source coordinates, and `q_c`. A dense merged core costs too much; a sketch of its quadratic mode might avoid both dense legs and the full core. This is a different object from selecting fewer CP columns or materializing a Tucker core.

**The first discriminator is a rolling representation proof, not a benchmark.** Specify how a new source enters, how `Q` is transported, how a basis join/rotation updates the sketch, how `(3,)` and `(2,1)` slices are protected, and how randomness remains conditionally valid. A fast static contraction is insufficient: if every join requires rebuilding the full core or replaying all old sources, the saving disappears. Count the complete join/update/read cycle under real flopscope primitives, including scatter/hash/FFT support and all reconstruction. Do not implement unsupported hashing with a dense projection and retain a near-linear cost claim.

**Falsifier.** Kill before seeds if the rolling sketch cannot be updated without the original expensive state, or if its complete operation count cannot cut the old-tier bill by at least half at a fixed candidate sketch dimension. If it passes that algebra/cost screen, use tiny adversarial tensors to measure contraction variance, then a fresh network design panel; stop if accuracy requires sketch width or replicas large enough to restore the old cost. Final promotion would still use paired arithmetic score.

**Why this is not established by either side.** The public catalogue identifies dense Tucker and source-wise Hadamard expansion as expensive. It does not prove that every approximate polynomial sketch fails. Conversely, kernel approximation error bounds do not guarantee percent-level error in signed K3 contractions after sixteen nonlinear layers. This is a concrete high-risk test of a different representation, with no promised win.

### B. Choose shared transport bases by future output sensitivity

**External foundation.** Finite-horizon model reduction explicitly measures error in the input-output map of a time-varying system; it need not minimize state reconstruction error. [Melchior, Van Dooren and Gallivan, *Model reduction of linear time-varying systems over finite horizons*, 2014](https://perso.uclouvain.be/paul.vandooren/publications/MelchiorVG14.pdf). Deep-network Jacobian spectra also show that transport strength is directional, not summarized by its mean squared scale. [Pennington, Schoenholz and Ganguli, NeurIPS 2017](https://arxiv.org/abs/1711.04735).

**Proposed application.** Keep the coherent compact chain intact. At each shared-basis join, use both the carried-leg Gram and a cheap approximation of the remaining output sensitivity to choose the subspace. Preserve the `(3,)` vector exactly and constrain the `(2,1)` error. This is a replacement for the *basis-selection objective*, not a reconstruction of K3 from K2 and not a global first-order expansion around zero K3.

If `T_l = W_{l+1} diag(w1_l)` is the frozen linear transport and `H_L` is a diagonal readout sensitivity proxy, a candidate backward weight is `H_l = T_lᵀ H_{l+1} T_l`. A full dense recursion costs additional cubic work; a rank-q representation is only useful if its construction, transport and join cost remain below the savings. A rough leading transport allowance for q=32 across 16 layers is `2 L n² q / 2^41 ≈ 0.00049` budget units, **excluding** closure differentiation, extra passes, QR and every basis interaction. That exclusion is crucial: no total-cost claim follows from this small number.

**Falsifier.** On synthetic source perturbations and then fresh networks, compare equal-cost bases: published weighted-leg Gram, random range, and output-weighted range. If the output-weighted basis cannot preserve final accuracy at a meaningfully smaller total cost, stop. A plausible initial screen is at least 15% counted-cost reduction with at most 5% arithmetic MSE increase; its final gate must be written on the isolated score difference. Do not read the old WP4 amplitude experiment as a test of this local basis replacement.

**Uncertainty.** Gaussian-proxy sensitivity can point in the wrong direction, covariance feedback may dominate, and nonlinear closure invalidates global balanced-truncation guarantees. The literature motivates a discriminator; it supplies no Phase-2 theorem.

The new public catalogue already rejects a direct source-by-source adjoint as not reducing its dense contraction count. This proposal must therefore show a lower **retained basis rank** at fixed output error; merely reversing those contractions is not new and should not be built. Even success at the suggested screen would be incremental, not the whole remaining factor-of-two score improvement. [Public dead-end catalogue, rows 18–21](https://github.com/504aldo/whest-p2-cumulant-k3/blob/18c17e2d7a9aeacd399cfc2c6b571e4e16dbfb45/docs/community_post.md).

### C. Retain a stochastic tail rather than deleting old lineage

**External foundation.** UORO maintains unbiased randomized approximations of recurrent influence; the OK algorithm proves a minimum-variance unbiased rank-r approximation result for matrices/Kronecker sums. Tensor Network Monte Carlo randomizes truncations to remove contraction bias. [Tallec and Ollivier, ICLR 2018](https://arxiv.org/abs/1702.05043), [Benzing et al., ICML 2019, Theorems 1–2](https://proceedings.mlr.press/v97/benzing19a/benzing19a.pdf), [Ferris, 2015](https://arxiv.org/abs/1507.00767).

**Proposed application.** After deterministic shared-basis compression, carry a small randomized approximation of the omitted *transported tail*, with all ages represented and repeated-index slices protected. The aim is to replace coherent truncation bias by controlled variance at the contracted observable. It is not sampling network inputs and is not selecting a smaller set of unweighted birth columns.

**Essential proof obligation.** Unbiased compression of a tensor is not an unbiased nonlinear estimator. If closure coefficients depend on the sketch, or two occurrences reuse correlated sketches, products are biased. First prove a conditional expectation identity for a frozen-coefficient linear transport or a multilinear contraction. Independent replicas may be necessary and must be priced. UORO's low memory does not imply sufficiently low variance here; variance-reduction work explicitly studies this limitation. [Cooijmans and Martens, 2019](https://arxiv.org/abs/1902.02405).

**Falsifier.** Before any truth labels, test exact contraction recovery on adversarial small tensors, including diagonal and `(2,1)`-dominated cases. Then use a small fresh design panel to measure variance after full depth and estimate its score contribution at fixed total cost. Kill if the number of replicas required to beat the deterministic shared-basis baseline erases the saving. Do not automatically apply Hutch++ trace guarantees to this signed tensor problem.

**Novelty boundary.** The repository already proposed unbiased sketches in general. The new proposal is restricted to the residual of the now-public shared all-age representation, with conditional-unbiasedness and slice protection explicitly specified. It should not be presented as discovering randomized compression.

### D. Learn missing relational closure state rather than another scalar readout

**External foundation.** Neural Functional Networks process weight arrays with neuron-permutation equivariance. Monomial-NFN additionally encodes ReLU scaling symmetries. They demonstrate useful learning over network weights, not challenge-winning activation estimates. [Zhou et al., NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/4e9d8aeeab6120c3c83ccf95d4c211d3-Abstract-Conference.html), [Tran et al., NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/577cd5863ec73be4e6871340be0936ae-Abstract-Conference.html).

**Proposed application.** If a fresh sensitivity audit shows missing in-flight fourth-order state still matters, predict a low-rank correction to the K4 core or its consistent births from pairwise covariance/K3 features and source-coordinate summaries. Use a shared equivariant operator whose output transforms correctly under neuron permutation and scaling. Train against the final error through the actual coherent chain, with every online operation counted.

This differs from T2's smooth scalar correction over its 116 features: it can introduce relational information that those features omitted, before later layers consume it. It also differs from fitting the old K2-state residual. However, the new public closure has already absorbed a major error source, so this may have **less** available signal than the prior head. Do not launch training before a gate-strength positive control and an oracle ceiling show useful recoverable error.

**Falsifier.** On fresh design data only, compare the cheap closure, an exact reachable projection of the missing state, a parameter-matched scalar head, and the relational correction. If the exact reachable projection cannot improve score by a useful margin after cost, stop before training. A suggested continuation screen is 15% end-to-end score improvement over the reconstructed compact baseline; a final gate must account for label precision and be preregistered separately. Additional features may cost O(n²h); channels are not free because training is offline.

## Ideas that remain low priority or closed

| Direction | Reason to deprioritize now |
|---|---|
| More scalar readout calibration | Our T2 and the public source's correction ablation both find little remaining benefit in their tested forms |
| Full K4 or full AUGMENT as the submitted architecture | Useful accuracy oracle, currently unaffordable; a new bounded representation must be priced first |
| Generic CP/Tucker/TT norm compression | Wrong objective unless it preserves the needed slices and transported observables; streaming low-rank algorithms alone do not resolve that |
| Dropping old source generations | Both repositories have direct negative evidence |
| Another symmetric-operand annotation | Does not implement the actual shared-leg identities |
| More primary MC/QMC samples | No new variance-reduction evidence overcomes the Phase-2 cost/accuracy gap |
| Tensor-product cubature or tensor cross integration in all 1024 inputs | Requires unproved low effective dimension/rank; ReLU gates and repeated dense mixing make this a weak first bet |
| Ensemble-average GP/NTK corrections | Do not directly estimate the fixed-network conditional expectation |
| Refit the published public-data coefficient tables here | Conflicts with the repository's validation policy; mint fresh design data |

Randomized tensor-decomposition papers provide useful implementation tools, including sketching contractions without materializing a tensor and streaming Tucker/tree representations. They do not promise the needed diagonal-slice error or score. [Wang et al., 2015](https://arxiv.org/abs/1506.04448), [Hashemi and Nakatsukasa, 2023](https://arxiv.org/abs/2311.14873), [Bucci and Verzella, 2024](https://arxiv.org/abs/2412.06111).

A genuinely recent approximate-matmul paper gives error controlled by the **output** norm, a useful property when products involve cancellation. However, its stated unbiased squared-error bound is proportional to `n/r` times the output squared norm; that is not a near-exact replacement for the whole K3 product. If used at all, restrict it to a small residual with a separately measured variance allowance. [Uffenheimer and Weinstein, February 4, 2026](https://arxiv.org/abs/2602.04386).

## What success would have to mean

Let `b = F / 2^41` and `e` be final MSE. Above the floor, winning depends on `b × e`. A `2.13e-8` raw estimator would need `b < 0.127` to beat a `2.7e-9` score. At `b = 0.18`, raw error must be below `1.50e-8`; at `b = 0.25`, below `1.08e-8`. These are arithmetic illustrations using rounded reported values, not forecasts or confidence bounds.

The published `0.2526` architecture is therefore a credible catch-up reference, not evidence that copying it wins. The rational sequence is: recover its representation-level savings under our stronger validation, determine the new fresh-data Pareto frontier, then spend research effort on one measurable gap. The present literature does not justify a guarantee of winning, an assertion that the leaders' precise method is known, or an assertion that all score-improvement routes are exhausted.

For the contribution prize, this review is prior-art material. Any contribution claim must explicitly distinguish imported identities, independently replicated effects, our negative results, and genuinely new tested extensions. Publication on September 9 predates this review and must be credited.
