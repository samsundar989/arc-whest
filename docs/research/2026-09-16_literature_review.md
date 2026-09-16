# Literature and frontier review — September 16, 2026

Research assessment and proposed discriminators, not measured candidate results or a contribution-prize claim. No challenge labels were read, networks minted, experiments launched, protected files edited, or submissions made. This review read CURRENT_STATE, the LOG tail, the September 16 strategist response/handoff, prior September 10–14 research, and pinned public code. The competition browser audit belongs to the parent review.

## Decision

Keep the working v31 family: fixed-network mean/covariance propagation, the full working third-cumulant recurrence, coherent approximate fourth-order corrections, and explicitly priced compression of persistent history. The literature supports this direction substantially better than a restart with sampling, Gaussian-only closure, an ensemble neural kernel, or an exact signed-law construction.

We have identified a credible family, not a demonstrated winning architecture. The next accuracy investigation D1–D3 is well targeted. The larger cost bet N1 needs two repairs before its result can decide the project: validate what its sensitivity proxy measures, and price the entire merged-core program. A failed Gaussian sensitivity proxy would reject that proxy, not every output-sensitive representation or all chances of winning.

The main new external source is Smith et al. (March 2026), on summing Tucker tensors without large intermediate cores. Its relevance is implementation of a retained-history representation, not evidence that the challenge state has low enough rank. Output-sensitive compression and polynomial query sketches were already in this repository's September 10/14 reviews; rediscovering them is not a new contribution.

## Primary sources and their actual scope

1. **Wu et al., May 2026, v2:** the problem is the expectation over inputs for one fixed weight realization. Their fixed-depth, polynomial-activation theorem and empirical ReLU results support cumulants, power cumulants, and factorization. They do not prove the finite-width ReLU error, stability, rank, or price of our proposed closure. An ensemble NNGP/NTK recurrence averages over the wrong randomness unless an additional fixed-network derivation bridges it. [Paper](https://arxiv.org/html/2605.05179v2).

2. **504aldo's public implementation:** independently refreshed remote HEAD is still `18c17e2d7a9aeacd399cfc2c6b571e4e16dbfb45`, the existing local oracle; ARC upstream HEAD remains `93d091a4c26c042bfffa28f2e76a81bc0aba94bb`. There is no newly published implementation at those two HEADs. [504aldo pin](https://github.com/504aldo/whest-p2-cumulant-k3/commit/18c17e2d7a9aeacd399cfc2c6b571e4e16dbfb45), [ARC pin](https://github.com/alignment-research-center/mlp_cumulant_propagation/commit/93d091a4c26c042bfffa28f2e76a81bc0aba94bb).

3. **Dunlavy et al., August 2025:** CP/Tucker decompositions can optimize errors in quantities of interest as well as tensor reconstruction. Their experiments concern combustion/plasma data; their optimization is not a ready inexpensive online estimator. This supports changing our compression objective, not asserting a rank-halving theorem. [Paper](https://arxiv.org/html/2508.11139v1).

4. **Zahm et al., 2018/2019:** gradient-based dimension reduction for vector-valued functions has Gaussian-measure error bounds using subspace Poincare inequalities; the output norm affects the selected subspace. Our propagated cumulant state is not a Gaussian input parameter covered automatically by that result. Nevertheless, the paper motivates preserving the full output vector rather than one arbitrary scalar adjoint. [Paper](https://arxiv.org/abs/1801.07922).

5. **Willcox and Peraire, balanced POD:** primal and adjoint information permit reduced models that reflect input/output behavior. Standard linear-system guarantees do not transfer to our nonlinear moment recursion without further analysis. It is a methodological precedent for testing downstream sensitivity alongside forward state energy. [Author-hosted paper](https://kiwi.oden.utexas.edu/papers/Balanced-proper-orthogonal-decomposition-Willcox-Peraire.pdf).

6. **Smith et al., March 2026:** structured Khatri–Rao/Kronecker sketches sum Tucker representations through factors and cores without constructing the expanded tensor. Their effective-rank selection includes a heuristic, and their numerical savings are for synthetic/PDE problems. Relevance: avoiding intermediate rank explosion at old-source joins, potentially choosing when to recompress. No challenge-score guarantee follows. [Paper](https://arxiv.org/html/2603.13532v1), [reproduction package referenced by the authors](https://doi.org/10.5281/zenodo.18736760).

7. **Sun et al., streaming Tucker:** linear sketches support compact tensor approximation from streaming/distributed data, with approximation guarantees. This supports streaming the sum of signed source contributions instead of materializing full history. The guarantee controls tensor approximation, not the nonlinear downstream activation error. [Paper](https://arxiv.org/abs/1904.10951).

8. **Han, Avron and Shin; Pham and Pagh:** polynomial tensor sketches can approximate entrywise functions of low-rank matrices and implicit polynomial features. Use the corrected May 2025 TensorSketch variance analysis. A cheap sketch of a quadratic query is plausible; its update under basis rotations and signed cancellation remain unresolved here. [ICML paper](https://proceedings.mlr.press/v119/han20a.html), [corrected TensorSketch](https://arxiv.org/abs/2505.08146v2).

9. **Gilbert, Kuo and Sloan:** preintegration can fail to smooth a kink when a needed monotonicity condition fails. Random signed deep ReLU networks provide no automatic common monotone direction. This weakens a generic claim that analytically integrating one input dimension will make QMC competitive; it does not prove every specialized conditional estimator fails. [Paper](https://arxiv.org/abs/2112.11621).

## Public experimental evidence: use its controls, not its conclusions wholesale

504aldo's F72 reports that compression behavior changed after regeneration/feedback changes: an earlier negative rank result did not transfer unchanged to the newer chain. F88 reports that ordinary merged Tucker becomes expensive at the ranks needed by their forward-energy basis, and that their attempted adjoint spelling still paid source-wise work. Those are relevant scoped negatives. They do not test a cheaper basis selected by actual nonlinear output sensitivity, nor every implicit query sketch. Their same reused public dumps supply development evidence; reported MSEs lack a numerical ground-truth noise floor in the inspected finding and are not our fresh validation. [Pinned findings F72/F88](https://raw.githubusercontent.com/504aldo/whest-p2-cumulant-k3/18c17e2d7a9aeacd399cfc2c6b571e4e16dbfb45/docs/findings_log.md).

Public Gaussian-covariance calibration and covariance/MC blends are useful negative controls, but their disclosed Phase-2 scores remain far above v31. They do not motivate replacing its higher-order state. [Exact covariance participant report](https://discourse.aicrowd.com/t/18214), [blend participant report](https://discourse.aicrowd.com/t/18200). The fixed-network ARC method is the strongest directly applicable primary research found in this targeted search; this is not a claim of exhaustive discovery or access to private leaders' methods.

The strategist's inference that the leaders must use a merged old core is too strong. FLOP totals and a similar layer-error profile are not identifying fingerprints. Different truncations, data-dependent ranks, contractions, or closures can generate similar summaries. Preserve the useful hypothesis while removing the assertion that only one representation is consistent with the board.

## N1 repair 1: test the sensitivity proxy before testing the rank claim

The handoff's `J = product(D W)` is a sensitivity of a frozen mean path. Compression changes third-order slices, covariance, fourth-order consumption, future births, and later nonlinear coefficients. None of these feedback paths is captured merely by multiplying Gaussian activation slopes. Moreover, a state perturbation is a tensor/factor perturbation, not automatically an n-dimensional mean perturbation to which J can be applied. The map between them must be explicit.

Proposed bounded amendment, not an executed experiment:

1. On already authorized DESIGN states, inject one small signed perturbation at one old-source compression point. Keep later recurrence fully enabled. Compare the actual final vector change against the proxy's prediction. Repeat at early/middle/late joins and include a perturbation that primarily changes D21/covariance.
2. Compare three objectives at equal counted construction cost: ordinary forward energy; the cheap J proxy with an explicit slice-to-mean map; a local query objective including D3, D21, covariance feedback and birth changes. Use the actual full recurrence as the diagnostic reference, not MC truth.
3. If the cheap proxy cannot rank harmful discarded directions, stop that proxy. A limited offline tangent/adjoint calculation can locate the omitted path, but its offline price cannot be hidden in a submitted online selector.
4. Only then test the stated rank-384-versus-rank-192 hypothesis against truth. Retain a forward-energy control and the same source ages, closure, thin residuals, and fourth-order coefficients.

**Falsifiers:** no lower final-output perturbation at equal rank; missing covariance/birth sensitivity; required online differentiation erases the saved work; or rank reduction damages paired final score. One scalar adjoint or lower state Frobenius error does not pass this test. This clarifies N1 within its intended mechanism; it is not permission to reopen a closed global-adjoint estimator unchanged.

## N1 repair 2: dense Tucker storage is cubic, but its generic transport is quartic

This is direct algebra, not a measured FLOP result. Let a symmetric old-history tensor be represented by `K_ijk = sum_abc H_abc U_ia U_jb U_kc`, with U of size n by r and dense H of size r cubed.

Transport U by a dense n by n map costs order n squared r. If the basis changes by a dense r by r map T, updating all three modes of H costs approximately `6 r^4` conventional multiply/add FLOPs. Computing the repeated-index slice uses

`S_ic = sum_ab U_ia U_ib H_abc; D21 = S U^T`,

which costs approximately `2 n r^3 + 2 n^2 r`, plus construction and masking. Merely retaining H uses r cubed storage. These are different quantities.

At n=1024 and r=192, one generic three-mode core rotation costs about 8.15 billion conventional FLOPs; the first readout contraction costs about 14.50 billion. At r=384 they are about 130.46 and 115.96 billion. These figures exclude every join, construction, basis update, young source, covariance, thin residual, and runtime overhead. They are neither official counted costs nor additive estimates for a final optimized program. Symmetry, a factored core, retained nonorthogonal bases, infrequent rotations, and actual supported kernels can change the price.

Thus the handoff's `r^3/n^2` readout-unit expression is not a total per-layer tariff. A positive rank test is insufficient. Before adopting a merged core, implement a complete update/join/read cycle under the official meter and show it is cheaper than retaining old factors at the same final error. The March 2026 implicit-sum paper is a plausible way to avoid an expanded join core; it does not remove the generic readout or rotation bill automatically.

## Bounded alternative if a dense merged core is too expensive

The repository already derived `D = Phi2(U) B U^T`, where `Phi2(U)` contains quadratic row features. A compressed quadratic mode could remove per-source dense reconstruction without asking the whole state to have extremely low Tucker rank. This remains higher risk than repairing the fourth-order path.

The useful new implementation question is whether structured sketches of the **sum** can reduce intermediate rank and repeated compression while retaining signed cancellations. Preserve the source factors needed to rebuild after a basis rotation; do not pretend a generic sketch commutes with arbitrary rotations. First test a label-free complete cycle including join, rotation, D3/D21 queries, and nonlinear feedback on tiny saved fixtures. Then meter target-shape primitives. Reject before a truth run if cancellation needs a sketch wider than the affordable rank, or reconstruction/rotation erases the saving. No new sampling or tensor framework is required merely to answer that question.

## Score arithmetic constrains the architecture

Using the current local record, v31 has graded raw MSE 2.9074e-8 at multiplier 0.3010253, adjusted 8.7520e-9. The graded raw truth-noise floor is not supplied here. These are recorded graded values, not a new independent evaluation.

For planning only, the raw MSE required for adjusted score 2.0e-9 is:

| Compute multiplier | Required raw MSE |
|---:|---:|
| 0.30 | 6.67e-9 |
| 0.20 | 1.00e-8 |
| 0.15 | 1.33e-8 |
| 0.10 floor | 2.00e-8 |

This assumes a constant multiplier across networks. With adaptive work, evaluate the mean of each network's product, not the product of mean MSE and mean cost. The target is illustrative and must be refreshed against the live board. Even a perfect cost reduction from v31 to the 0.1 floor alone would leave approximately 2.91e-9. Winning requires accuracy as well as cost. A 0.20B, approximately 2e-8-raw architecture is about 4e-9: valuable progress, not a demonstrated win.

## Recommended implementation order

1. Keep v31 frozen as a real graded fallback. Resolve D1–D3 on P11, with the compared fourth-order objects and consumption conventions made explicit. Do not infer a port defect solely because two distinct trajectories disagree.
2. Complete P3 and P4 as separately attributable changes only after their isolated cost/accuracy checks; their external evidence is stronger than a new architecture promise.
3. Amend N1 with the proxy-validity and complete-cost discriminators above before a broad rank sweep. Preserve a practical shared-history compression fallback independently of whether dense merging wins.
4. Attempt an implicit sum/query sketch only if N1 shows a real old-history bottleneck and a complete inexpensive update can be written. Bound the proof-of-concept; do not replace the near-term release path with months of general tensor machinery.
5. Fit any selected closure/rank policy only on permitted DESIGN data, freeze it, and obtain the required untouched confirmation. Keep all online selection costs in the bill. External author fits are diagnostic references, not evidence for our generalization.

The winning target is an accurate fourth-order-corrected third-cumulant chain with inexpensive retained history and final-only readout, ideally close to the 0.1 compute floor. The repository is following that direction. Its largest unresolved scientific question is whether retained history can be represented and queried cheaply enough without changing the correct feedback. The literature offers tools for that experiment; it does not answer it in advance.
