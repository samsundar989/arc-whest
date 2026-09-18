# Next architecture review — 18 September 2026

Read-only independent review of the completed evidence and source. Only this document is edited. No estimator was executed, no labels or prediction sidecars were opened, and no seed block, evaluator, gate, frozen estimator, vault or submission was changed. Numerical evidence below comes from existing verified summaries and cost ledgers, not a new experiment.

## Recommendation and its limit

**Finish the missing integrated recurrence before selecting another physical operator.** The concrete next implementation is our PAIR+Q program, the frozen own-fit controls, compensated rank-32 P3, age4/rank256 old-qE storage, and the existing P4 depth-5 exact transformations in one generated file. This is an experiment, not a presumed improvement. Match all controls to the same P4 spelling. In particular, v40/own-fit uses the older multiplication spelling while v42 includes P4 depth5: their adjusted-score contrast combines an accuracy change and an exact cost change.

There is a credible engineering gap: `s11_build_candidate.py:84` explicitly rejects P3 together with shared storage. The selected fit source is uncompressed PAIR+Q without P3. Consequently none of the reported comparisons answers whether the best accuracy mechanisms and best storage mechanism coexist. Requiring an isolated new fourth-core defect before measuring this composition would leave that concrete question unanswered.

**Integration alone is unlikely to deliver the requested major jump.** It adds a significant P3 bill; compression damage is not known to disappear; the fitted constants may compensate for the absence of P3 and therefore interact adversely when it is added. A matched 2×2 of P3 off/on and frozen fit off/on, all using the same fixed compressed storage and P4d5, is the smallest useful complete-package accuracy comparison. Uncompressed/full-rank chains supply label-free integration parity; add an uncompressed accuracy sentinel only if its attribution value is preregistered. There is no new fit, closure menu, rank grid, reference lambda or empirical mask in this first comparison.

The next serious cost target remains history transport/reconstruction/readout, not dense covariance. The available evidence does not establish a concrete new representation that reaches 0.16 B with preserved accuracy. Below I distinguish an implementable bounded consumer experiment from proposals whose full cost already looks unattractive. A release should follow only if a resulting complete candidate actually supports the existing twofold gate; there is no honest guaranteed route to that result from the current measurements.

## What the existing implementation actually carries

In `estimator_k3_v41_pairq_a4r256_qE_p4d5.py`:

- Recent lineages carry A and M densely. Old lineages have A=Q FA and M=Q FM, with one shared physical Q and lineage-specific coefficient matrices.
- The repeated-support leg is A_D=M diag(s)+3 A diag(e)+Z L^T. Z is transported separately, and L,s,e are static birth data.
- The two-leg source uses C=A diag(w2_birth).
- Fourth-order hubs retain transported Y and static coefficient vectors.
- Shared joining projects A and M. It does **not** apply one uniform orthogonal projection to every mode of the entire tensor: Z and Y remain outside that shared projection.
- The consumer reconstructs old A/M and the dense Z L^T residual. Its two principal dense terms are E=LA A^T+LP M^T, plus thin residual/hub terms. Sharing Q saves transport, but old source count still appears in reconstruction and contractions.
- Linear transport between joins is exact for the represented factors. Information is lost at joins and at the separately specified birth residual compression.

This distinction is essential for a new objective. The actual error tensor is T minus the tensor represented after the storage operation. Replacing this with T−P_Q tensor-cubed T generally tests a different algorithm.

P3 adds two symmetric three-leg source families, with
A1=F1 R3^T, A2=F2 R4^T,
F1=diag(w1)V, F2=diag(w2)U, and D21≈U V^T.
Its grouped contributions are

```
Gp = A1 * w3_birth + A2 * w2_birth
D3 += 3 sum_columns(Gp * A * M)
LA += Gp * M
LP += Gp * A
D21 += TT F^T
```

The repeated content must be subtracted at birth using the **same compressed** U,V used in these new factors. P3's beta1+beta2 construction omits the gamma family; it is not a complete third-cumulant-times-covariance closure.

The required integration is localized: shared consumers obtain all-age A/M through the existing reconstruction interface; F/R3/R4/w3_birth survive advance, Wick scaling, joining, append, truncation and every fallback; P3's LA/LP additions are present before qE captures its join objective. The thin TT F^T term remains explicitly priced and separately carried. Do not silently project F into Q or drop it on a join to obtain a cheaper file.

## Evidence and magnitude

All complete P12 rows summarized here have averaged truth-noise floor 2.2480709584e-9. The fit's 12 held-out rows have 2.2271138918e-9. These are DESIGN findings, not fresh release evidence.

| Existing complete P12 arm | Observed raw MSE | Official-precision adjusted | F/B |
|---|---:|---:|---:|
| PAIR+Q, uncompressed, older multiplication spelling | 2.91088e-8 | 8.20620e-9 | 0.304654 |
| PAIR+Q+P3+P4 depth5, uncompressed | 2.41486e-8 | 6.79884e-9 | 0.309376 |
| PAIR+Q old-qE age4/rank256+P4 depth5 | 3.38603e-8 | 7.08662e-9 | 0.223640 |

Source: `experiments/2026-09-17b_stage3/synthesis.json`. The frozen fit improves its own 12-row control by 1.144855×, but is neither a P3 comparison nor a compressed comparison.

The PAIR+Q compression raw penalty in these complete-panel summaries is approximately 4.75155e-9; small P4 reassociation differences are included in that contrast. It is not an independent additive noise term and must not be assumed to transfer to P3.

A useful **conditional arithmetic bound**, not a prediction:

1. The existing v32-to-PAIR+Q extra program bill is 5.130118e9 FLOPs. Adding that to the metered v32+P4 bill gives an inferred PAIR+Q+P4 bill of 621.304469e9.
2. Subtracting this inferred matched bill from v42's 680.325773e9 puts P3's cost near 59.021304e9, or 0.026839 B. Exact integrated metering remains required.
3. Adding that cost to compressed PAIR+Q gives about 550.809789e9, F/B≈0.25048, before any integration-specific overhead.
4. The P3 arm's effective official-precision raw error is approximately 2.19760e-8. If this error survived compression **without any damage**, the compressed P3 score would be approximately 5.50e-9.
5. If the separate 1.144855 fit gain also transferred perfectly, that optimistic number would become approximately 4.81e-9, still before compression damage. This multiplication is an illustrative favorable scenario, **not an evidence-supported gain**.
6. Retaining the earlier compression penalty instead raises the no-fit scenario to roughly 6.69e-9.

No universal P12 twofold threshold is known without a matched v31 control. Do not compare these scenarios directly with the public 4.376e-9 milestone as if the panels shared a scale. They do show why a cost-and-accuracy composition must be measured and why it should not be sold as a likely winner.

## Full-cycle cost model for the integrated baseline

Let n=1024, depth=16, r3=32, k_l the all-age incoming lineage count, and B=2^41. Preserve the measured approximately 491.788485e9 whole-file PAIR+Q old-qE bill as a baseline anchor. The P3 additions per layer are explicitly:

| Operation | Ordinary float32 leading cost |
|---|---:|
| Transport F with 2 k_l r3 columns | 4 k_l n² r3 |
| Reconstruct A1,A2 | 4 k_l n² r3 |
| Form the two thin contractions (M*A*c)R | 4 k_l n² r3 |
| Return TT F^T | 4 k_l n² r3 |
| Elementwise Gp, D3, LA/LP and compensation | O(k_l n²) |
| Birth range finder, power pass, QR and projection | O(n² r3+n r3²) |
| Birth D21 approximation and repeated compensation | O(n² r3) |
| Extra immutable concatenation, storage and casts | Count explicitly in emitted ledger |

The source's complete guard model is in `s2_build_p3.py:177–187`; its operation spelling determines discounts and final-layer pruning. The 59.0e9 inferred incremental bill is a more useful anchor than multiplying these ordinary-product leading terms and pretending it is a flopscope total.

At k=15, F plus R3/R4 contains 4 k n r3 float32 scalars, about 7.5 MiB. This is not peak memory: A1,A2,Gp, dense reconstructed old legs, P2 caches, wide copies, and Strassen temporaries dominate. Both A1 and A2 together are about 120 MiB at that layer; caching additional full-wide arrays multiplies this. Use a conservative 16 GiB CPU job declaration for development unless an allocation audit establishes a different peak; the final candidate must still pass its separate enforced Linux 8-GiB cap.

The current compressed file has roughly 20,000 metered operations. Reuse the batched source layout and cached P3 pieces; adding one Python loop per lineage can consume residual headroom without changing FLOPs. Do not infer residual clearance from operation count or local time. The final candidate needs its own subprocess/residual evidence, including worst-MLP headroom 0.4/observed.

Dense covariance is not the main leverage. Even an ordinary full covariance sandwich costs approximately 4 n³ per layer, about 0.031 B across sixteen layers, before the existing symmetry, final-diagonal and P4 savings. It is already cheaper in the current implementation. Removing all of that work would not fix a 0.25 B integrated package; approximating it risks every subsequent source and closure input.

## Why same-state E is the wrong whole-chain guarantee

Let ΔT be the actual symmetric third-order storage error after the nonlinear gate, and τ_i=Σ_j ΔT_ijj. For independent isotropic Gaussian next-weight rows u,v with coordinate variance σ², Wick pairing gives

```
E[ΔT(u,u,u)^2] = σ^6 (6 ||ΔT||_F² + 9 ||τ||²)
E[ΔT(u,u,v)^2] = σ^6 (2 ||ΔT||_F² +   ||τ||²).
```

For n independent next rows, the summed diagonal-plus-off-diagonal repeated-query loss has coefficients (2n²+4n) on ||ΔT||² and (n²+8n) on ||τ||², times σ^6, before consumer normalizations and closure sensitivities. Thus next-layer visibility depends on full tensor error and trace, including directions invisible in the present repeated slices.

This derivation is exact for the stated random-weight expectation, **not** a certificate for the fixed realized next W, later nonlinear gates or final truth error. For the actual W the relevant one-boundary loss is the explicitly summed ΔT(w_i,w_i,w_j) and ΔT(w_i,w_i,w_i), with complete recurrence continuation needed afterward. A mean-only Jacobian omits covariance and birth feedback.

The corrected qEc result is direct counterevidence to treating a better immediate singular-space objective as a useful continuation objective: same-state retention improved while every full recurrent diagnostic and truth score worsened.

Nor does preserving span(τ) preserve τ after projecting the legs. An exact trace correction exists:
H(v)_ijk=(v_i δ_jk+v_j δ_ik+v_k δ_ij)/(n+2),
with trace H(v)=v. But after a dense transport A it becomes a symmetric product of Av and AA^T. Carrying a new full matrix for every such correction reintroduces history cost. Regenerating H from a new trace each layer changes the physical tensor and needs a new closure justification. It is not a free correction.

## Feasibility screen of proposed new storage mechanisms

### Tensor residual probes: scientifically better motivated, not yet a cheap production basis

For a symmetric CP term sym(a⊗b⊗c), T(.,g,g) is the sum of a(b·g)(c·g), b(a·g)(c·g), c(a·g)(b·g), divided by the chosen symmetrization constant. Batched contractions give a residual oracle without an n³ tensor. Independent g,h also give E||ΔT(.,g,h)||²=||ΔT||² for unit Gaussian probes.

However the full history has several source families. A conservative direct spelling with six families, both pre- and post-storage states, and q probes costs about 144 q n² Σk_old FLOPs before reconstruction, QR, selection and recurrence. Age4 has eleven joins and Σk_old=66. With q=16 this alone is about 159e9 FLOPs, 0.0725 B; q=4 still costs around 0.018 B. Shared factors can reduce this bill, but they must be derived and metered before claiming a cheap enrichment. Gaussian probes must follow the existing seed rule; their permutation behavior and fixed-W relevance need explicit checks.

Therefore use full-tensor/trace residuals first as **offline label-free discriminators** of storage errors. Do not open a costly live residual-enrichment experiment just because the oracle avoids dense tensors. No cheap enough rank-selection implementation has yet been established here.

### Merged Tucker N1: readout is not the principal cost barrier

The measured `experiments/2026-09-16_stage3_n1/cycle_cost_v2.md` already includes full joins, transport and queries. At r=128, new-source enrichment s=32:

| Recent age | Symmetry-declared complete N1 cost | Undeclared |
|---|---:|---:|
| 2 | 0.1790 B | 0.1948 B |
| 3 | 0.2035 B | 0.2181 B |

These are v31-based cost-only chains, before the current richer recurrence/P3. At age3, everything except joins costs 0.1349 B; all old queries together cost only about 0.0132 B with symmetry. A join costs 0.00624 B, including approximately 0.0037 B for projecting the incoming n source columns into the cubic core. The measured failure to get below 0.17 is therefore not repaired by a cheaper D3/D21 contraction alone.

Rank96 or compressing source columns before core formation could lower cost, but changes approximation quality; previous tiny fixed/energy-basis accuracy tests were adverse and do not establish viable live-shape ranks. The parked conclusion was overbroad historically, but the actual priced implementation remains economically poor. Do not replay it under a new name.

### Conditional bounded query/join change: lower-rank D21 evaluation with persistent source factors

If a second research arm is needed after integration, a concrete **different** change is to retain the age4/rank256 persistent source factors and approximate their old-history D21 evaluation by a two-pass rank-q query factorization. There is an important integration catch: the current qE join reuses the exact inner query product, so the change must encompass its join input as described below. It cannot honestly be described as a free readout-only substitution.

Write the old contraction E=L F^T Q^T, where L=[LA,LP], F=[FA,FM], and m is the old source-column count. QR the transported Q if needed, preserving its triangular factor; do not assume Q remains orthonormal after a gate. Compute EΩ, form an orthonormal range U, and then compute U^T E through the factorization. Return U(U^T E). The consumer does not overwrite the persistent source factors. D3, recent-lineage D21, and the thin P2/hub/P3 contributions remain exact under the represented state.

The existing join needs Xold=L F^T, stashed as `_qold`. Retaining an exact Xold defeats the dominant saving. A complete proposed spelling instead stores Xtilde=U(U^T L F^T), adding approximately 2 n q r work per join after the thin factor is available, and uses that same Xtilde in qE's basis objective. This explicitly changes the join input as well as D21; future source coefficients therefore change at their normal joins. All existing query-basis products, rotations, coefficient concatenations and joins remain billed. Requiring an unchanged exact join is a cost-based falsifier for this particular proposal, not a reason to quietly recompute the omitted product.

For width n, old basis r and target q, the leading full old-query cost is about 4 n m r+2 n²r. A conservative two-pass reduced query costs approximately
8 n m q+8 m r q+2 n²q+O(n r q+n q²),
plus source reconstruction already required by the baseline, QR of the transported Q when required, and the explicit Xtilde reconstruction above. A fixed q=32 with r=256 costs roughly 31% of the dominating full-query term at n=1024, before lower-order work. Deterministic sketch selection, conditioning, rank-deficient fallback and factorization provenance all require explicit specification; no truth-driven rank choice is allowed. A column sketch chosen by current physical-coordinate leverage scores is one deterministic candidate, but may miss signed cancellations; full-rank and rank-deficient fixtures must exercise a counted dense fallback. This is a proposal for a cost/identity prototype, not a completed production algorithm.

But the magnitude is limited. With Σ old lineages=66, that dominant full-query bill is only about 70.9e9 FLOPs (0.0322 B). Even deleting it entirely cannot move the integrated q≈0.2505 package to 0.16. A rank32 implementation might save approximately 0.02 B, then loses some of that saving to QR and other work. The aggregate query's spectrum and continuation error may make even that uneconomic.

This is implementable and different from raw polynomial TensorSketch, a next-weight sketch, or storing only D21. It is **not** yet an architecture with a credible first-place budget; plan for no more than roughly a ten-percent whole-file saving, subject to an actual complete ledger. Stop it before labels if the complete measured saving is small or the actual continuation error is unfavorable. Approaching 0.16 still requires reducing recent-source or reconstruction work; current age2/3 compression evidence is strongly adverse.

## Bounded next slice and path to a release

1. Commit a new contract with Samuel's predictions. Specify matched P4d5 controls and frozen parameter values; record that integration, not a discovered fourth-core flaw, is the hypothesis. Keep all existing negative lanes closed.
2. Repair only the P3/shared-storage composition interface. Require disabled-P3 identity, full-rank storage identity, arbitrary nonuniform gates and nonsymmetric D21, every mixed age/fallback path, full-tensor tiny source parity, compensation parity, and complete recurrent parity. Include exact lineage birth coefficients after multiple joins. Run preflight and independent Skeptic.
3. Before labels, meter the entire integrated export at live shape. Price source birth, transport, full query, join, QR, projection, immutable copies and final pruning together. Apply the workerq optimization review: one MLP at a time, no duplicated truth/model copies, streamed/checkpointed results, measured conservative RAM.
4. On a newly registered DESIGN panel, compare the matched **complete compressed** 2×2: default/frozen-fit × P3 off/on, each with the fixed age4/rank256 old-qE storage and P4d5. Include v31 as the release-scale anchor. A predeclared selection rule chooses at most one package. This directly tests the final operating point instead of selecting on an uncompressed proxy. If a bounded new fit is subsequently registered, fit the complete compressed package on its designated training subset and hold out its validation subset; do not retune on P13 or infer improvement from the prior uncompressed fit's endpoint parameters. The purpose is an actual package decision, not retrospective attribution of separate gains.
5. If the composition is well short of the paired twofold target, preserve it as a measured fallback and do not spend P13. One label-free low-rank D21 cost/continuation discriminator can determine whether a bounded cost follow-up is worth its expense; it cannot be assumed to bridge the gap. A new major representation proposal needs a complete source-to-query cost model before truth.
6. Only a frozen, DESIGN-supported candidate goes once to P13 under the existing dual twofold gate, then exact-package subprocess, enforced Linux memory, public CONFIRM, preflight, independent Skeptic, and exact-file submission approval. Certification cannot manufacture missing scientific support.

The integrated experiment is justified and bounded. The stronger claim that any presently identified storage change will close the remaining large gap is unsupported. The most useful strategic output is an exact combined baseline and an honestly priced next representation, not another succession of apparently cheap operators whose source history, feedback, or join costs were left out.
