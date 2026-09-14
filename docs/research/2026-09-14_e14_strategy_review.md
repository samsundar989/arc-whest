# E14 strategy review: accepted work, narrower blockers, two executable next steps

September 14, 2026. This review starts from worker commit `9eb1527` and the E14 evidence package. It changes research recommendations and documentation only. It performs no new estimator run, rank experiment, protected-label read, seed claim, vault operation or submission. A fresh reviewer independently reran 21 lightweight reusable-input journal/release tests and checked existing evidence hashes. New algebraic proposals below still require independent implementation tests.

## Verdict on the worker

**Accept the bounded work; revise the dependency structure of the remaining plan.** The worker fixed real infrastructure problems, preserved failed evidence, established the tiny float64 algebraic correspondence, and correctly refused to call a more expensive wrapper an optimization. It was right to stop before spending a fresh decision block on those implementations.

The stopping point does not establish that the architecture cannot be made cheaper. It also does not require another open-ended search for a strategist. Two concrete programs can now be specified: a mixed-precision shared-basis query retaining the full residual, and a scalar-metric accuracy closure with a much simpler persistent feed. Their smallest discriminators can run independently. No winning implementation, useful compression rank, or accuracy gain has yet been established.

## What survived independent inspection

| Worker result | Independent assessment | Limit |
|---|---|---|
| Durable result journal and strict release index | 11 journal and 10 release tests independently pass; actual scorer/lifecycle boundaries are exercised | Process-crash evidence, not permission to resume a spent block or a guarantee against every storage failure |
| Final preflight | All 309 recorded source hashes match; 12 enabled steps passed, four skipped | Root-baseline regression gate; separate E14 tests supply the new-feature evidence |
| Production-truth precision pilot | Recorded arithmetic bounds recompute; producer pins and records IEEE-f32 mode | One common-input fixture, conditional error bounds, no retrospective validation of historical TF32 behavior |
| B4 float64 correspondence | 841/841 checks, worst normalized discrepancy about 9.74e-13 | Fixed tiny mathematical comparison, not target-shape performance |
| Original f32 failures | Three of four output cases fail their committed allowance | Preserve those failures; they do not prove every stable arithmetic policy fails |
| Mixed-f64 wrapper | Four outputs match frozen reference, while tiny structural-zero mismatches remain | Retains all baseline work and adds encoding; not a cheaper candidate |
| Dense cost refusals | Correct for the named implementations | Not lower bounds for the whole representation family |
| Connected-tree K4 | A precisely defined approximate closure and costly literal implementation | No demonstrated accuracy, numerical admissibility or cheap implementation |

The independent evidence and exact checks are recorded in [the worker review](2026-09-14_e14_independent_review.md). The existing F21 graded result remains the competitive anchor, with its separate local refusal and provisional server-byte attribution unchanged. No new release pass follows from E14.

## Three corrections to the old critical path

### 1. Full-rank correctness and compressed cost are different tests

The uncompressed implementation is a reference for the approximation. It need not itself save 20%. A rank-n change of coordinates can be expensive and still make a rank-r implementation scientifically interpretable. Conversely, a cheap rank-r formula without a faithful full-rank operator remains invalid.

Require mathematical and numerical evidence from the reference mode; require a complete cost model and then a counted ledger from the proposed compressed mode. Do not use the cost table to choose a rank's accuracy or use reference-mode correctness to assume useful rank exists. The literal decode-and-run-all-F21-work wrapper remains rejected.

### 2. Reference rounding is not ground truth

On the forced tiny F21 B4 fixture, the stored final field shows maximum absolute reference-f32 versus reference-f64 difference **4.30524**, versus compact-f32 versus reference-f32 difference **4.29549**, at reference scale about **110.981**. This is strong evidence that the tiny reference itself is precision-sensitive. It does not prove f64 is statistically better or that the live 1024x16 method is unstable.

The failed original `2e-10` output-drift gate stays failed. A new arithmetic policy must have its own prospective contract; it cannot widen that old bar. Use the independent f64 recurrence as a mathematical oracle, the frozen f32 program as an operational comparator, and fresh paired accuracy as the eventual statistical authority. Distinguish failures of algebra, rounding, approximation and resource behavior.

There is also a quantitative interpretation problem. For a fixed target and unchanged cost, with baseline prediction error E and prediction-drift energy d,

`|MSE_new-MSE_old| <= 2*sqrt(E*d)+d`.

Thus `d <= 0.01 E` allows a worst-case 21% MSE change, not 1%. The recorded 1% drift-energy rule is not mislabeled as a score guarantee in the strongest evidence, but the distinction must remain explicit in the next contract. A sufficient drift-energy ratio for at most 1% MSE change, conditional on the assumed E, is `(sqrt(1.01)-1)^2 ≈ 2.4876e-5`. This is an explanatory bound, **not a newly adopted gate**. Tie numerical uncertainty to the isolated proposed score gain, account for changed cost, and avoid requiring an unjustifiably precise match to an unstable rounded trajectory.

### 3. Exact transport of one approximate K4 closure is not a prerequisite for all accuracy work

E14's connected-tree approximation already discards a fourth-order residual without a future-error bound. Preserving its particular two-metric transport exactly is a modeling choice, not a theorem that cheaper alternatives lose the useful signal. A different, fully declared scalar-metric approximation has an exact simplification of its induced third-order feed. This can be tested on the existing full-rank F21 tower before shared-basis compression works.

This does not reopen a failed implementation unchanged. The new distinction is a complete scalar-metric response, exact cancellation at the repeated-slice birth boundary, and persistent feed using already transported factors. The proposed core projection and its discarded off-diagonal transport term must be named explicitly.

## Concrete cost path: packed f32 residual, f64 shared basis

The [mathematical review](2026-09-14_e14_math_review.md) specifies all state, transport, admission, rotation, source queries and costs. Its central choice is simple: **keep the entire residual D in F21's existing packed float32 representation; compress only A and P in a float64 common basis**. The newest source remains exact, with symbolic P=I until it ages. All source ages remain.

For old sources write `A=Q F_A`, `P=Q F_P`, and `R=cast64(D)`. After the declared transport, use

```
LA = 2*(A*P)*w
LP = A*A*w + (2/3)*(R*P)
X = LA@F_A.T + LP@F_P.T
Y32 = existing_mm32(cast32(P*P), D.T)
D21_old = X@Q.T + cast64(Y32)/3
D3_old = rowsum(R*P*P) + 3*rowsum(A*P*A*w).
```

Both operands of the residual product are explicitly f32. Merely storing D in f32 while multiplying by f64 P would not obtain that cost. The basis and coordinate products are f64. This is real-arithmetic equivalent to the selected source recurrence, with a new explicit rounding policy. Preserve F21's Wick, centering, scalar projection and birth order; do not deploy the generic independent oracle as a replacement nonlinear evaluator.

The existing batched Strassen primitive has exact counted recurrence

`S_t(a,b,c)=7 S_(t-1)(a/2,b/2,c/2)+3(ab+bc)+4ac`,

including input/output packing and addition charges. At the live square shape, `S4=1,366,126,592` f32 FLOPs. That is why the 196 ordinary-f64-product obstruction does not settle the legal Strassen implementation.

Including all specified products/QR and a conservative explicit pointwise/copy/cast envelope gives **615,258,560,822 modeled FLOPs at hypothetical rank64** and **690,954,856,038 at rank96**, below the exploratory 742,766,109,982.4 ceiling. These are static upper models for a specified implementation, not measured ledgers, recommended accuracy ranks or resource passes. The rank128 upper model exceeds the screen, which does not prove a lower-bound refusal at that rank.

The coordinate-sketch basis proposed in the review is an explicit inexpensive baseline, not a claim of optimal subspace selection or exact permutation equivariance at reduced rank. Measure the declared approximation and its invariance behavior; do not choose rank or basis from the leaderboard. Packed reconstruction and recursive Strassen workspaces can still dominate memory and residual time. The full implementation must respect the stated block allocations or be re-costed.

## Concrete accuracy path: cancel repeated feed, share existing factors

The [accuracy review](2026-09-14_e14_accuracy_review.md) gives the full derivation and all affected consumers. For the separately declared scalar-metric extra fourth cumulant `Q=m L_I(G)`, let `a=w(2,1)`, `b=w(1,1)`. The one-quadratic/two-linear feed is equivalent **after the repeated-slice birth correction** to

`Jred = U((m/12)*a, diag(b) G diag(b))`,

where `U(y,S)_ijk=y_i S_jk+y_j S_ik+y_k S_ij`.

If D denotes the tensor reconstructed from its repeated slices, the identity is `J-D(J)=Jred-D(Jred)`. It is an identity for the whole new K3 tensor, not just the immediate readout. Diagonal G contributes only repeated entries to this feed and cancels there; its effects on the separate moment consumers remain necessary.

For a covariance-aligned off-diagonal G, a source can share F21's transported A and P. Store only a vector y and fixed birth vector r=b. Define `X=A diag(r) P.T`, `S=(X+X.T)/2`, `t=diag(X)`. Then

`D3_feed=3*y*t`,

`D21_feed=diag(y)*(X+X.T)+outer(t,y)`.

The two matrix orientations can fuse into the existing three-product D21 query. This moves the baseline's w2 factor across one product, so active fused/unfused numerical conformance is a real obligation. A disabled-extra control must call the unchanged frozen path, not rely on algebraic zero to conceal reassociation. All old y vectors receive each linear/Wick map once, and each new birth is subtracted once. Tiny dense tensor checks must test these claims before a closure or score experiment uses them.

One complete subsequent closure candidate uses the worker's four-product tree H, projects its off-diagonal part onto output covariance, retains its diagonal g, and at the next preactivation preserves the exact transported diagonal while explicitly discarding the off-diagonal correction from `B diag(g-lambda*diag(Cout)) B.T`. It also replaces the metric `B B.T` by `m I`. These are declared approximations, not exact compression of E14's two-metric closure. Its projection coefficient uses current moments without labels or public-fitted constants. A zero denominator has a defined zero result; nonfinite behavior fails rather than invoking an unregistered repair.

The fused realization adds 60 square tree products rather than 1069 extra squares. With the actual existing square primitive those products cost about **81.97 billion FLOPs**, yielding baseline-plus-square-work **1.01043 trillion** before additional pointwise, vector, copy and consumer costs. It would require **more than 8.11% lower raw error just to break even**, with the true threshold higher after complete counting. This is a promising cost mechanism, not measured accuracy or a complete package bill. The unfused version is an algebra/numerical control, not assumed affordable at the same price.

The simplest next step is the supplied-core tensor identity oracle at n=3/4. It does not require any choice of regeneration coefficient or successful compression. The connected-tree projected closure is a conditional next candidate after synthesis of that result; retain the option to reject it before labels if it is unstable or poorly priced.

## What new literature changes

Primary numerical-analysis work supports explicit attention to matrix scaling and algorithm-dependent error; fast matrix multiplication need not match classical multiplication component by component to be useful. It provides no end-to-end cumulant-accuracy guarantee. [Ballard et al., 2016](https://arxiv.org/abs/1507.00687).

A February 2026 paper proves normwise error bounds for alternative-basis multiplication and analyzes diagonal scaling. Its corrected version explicitly withdraws a preliminary claim of matching Strassen's stability. Such a backend is an optional future primitive, not the next required framework rewrite: it needs its own supported-operation, copy, stability and lifecycle evidence. The existing legal batched primitive already suffices for the new feasibility argument. [Schwartz et al., 2026](https://link.springer.com/article/10.1007/s00211-026-01531-9).

The inspected public covariance-aligned implementation supports the possibility of factor sharing, but includes public-fitted coefficients and other changes. Its results are author evidence on reused data with unavailable truth-noise floors. Our proposed projection and exact birth cancellation need their own tests and fresh decision evidence. Source-authenticity eligibility remains unresolved for the actual candidate lineage. No claim that this is the leader's algorithm or an original invention is made. [Pinned public generator](https://github.com/504aldo/whest-p2-cumulant-k3/blob/18c17e2d7a9aeacd399cfc2c6b571e4e16dbfb45/generators/make_v17.py).

## Winning path and next decision

The architecture family remains credible. E14's results strengthen confidence in parts of its mathematics and instruments, while narrowing the implementation search. They do not establish that a useful low rank exists or that the projected K4 core improves accuracy. Even both hypothetical improvements may leave a gap to the dated 2.540e-9 leader; keep cost and accuracy milestones separate from the winning objective. Do not refresh or tune against public rows merely to choose among these designs.

For scale, adding the rank64 upper model and the accuracy proposal's square-product subtotal gives about0.317 of budget, before the latter's remaining overhead and without proving the two implementations combine unchanged. At that illustrative multiplier, tying2.540e-9 requires raw error about8.0e-9, much lower than the demonstrated F21 level or T1's richer coupled control. This is not a forecast of the combined candidate. Full residual preservation is a first-experiment isolation choice, not a permanent winning-architecture requirement: the named mixed program's all-age residual transport plus older-source residual returns alone is about0.128 of budget, excluding youngest returns and shared-query work. If actual evidence confirms that bottleneck, the next distinct intervention must reduce that residual/query cost or deliver substantially more accuracy. Do not declare these two prerequisites a sufficient route to first place.

Follow [the revised execution plan](../E14_REVISED_PLAN.md): one bounded mixed residual kernel, one independent supplied-core feed identity, then synthesize before either rank selection or a closure candidate. No more general infrastructure work is on the critical path absent a concrete new defect. Fresh promotion and release retain their existing contracts, human predictions, private-data discipline and independent Skeptic. The final [strategy Skeptic](2026-09-14_e14_strategy_skeptic.md) reviews these proposals as research, not as demonstrated candidate performance.
