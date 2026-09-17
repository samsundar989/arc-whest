# September 17 architecture review: the nonlinear closure is not matched

Read-only source and existing-evidence review. No decision data, P11 truth, new networks, estimator edits, or heavy jobs were used. One deterministic five-dimensional algebra fixture was evaluated on CPU. Findings distinguish source facts, algebra, and unmeasured hypotheses.

## Decision

**The next slice should audit and test the nonlinear moment closure before committing to a new old-history representation.** The worker's statement that v17's complete fourth-order recipe failed on our base does not establish that the remaining gap is located in third-order transport. Two specific, still-unmatched mechanisms remain:

1. **Our nonlinear table contains 25 terms; v17 contains 62 plus four (3,1) additions.** In particular, the public file's actual final mean includes two cumulant-product terms absent from our actual final mean. This difference requires no third-order transport explanation.
2. The public source protects certain diagonal/rank-one fourth-order feed terms outside the low-rank residual approximation. Our route-(i) arm leaves their repeated-support equivalents inside the approximated residual. Exact rearrangement and finite-rank approximation do not commute.

Neither fact proves a score gain. Both are inexpensive, concrete discriminators that invalidate the stronger claim that all relevant mechanisms have been exhausted.

## 1. The overlooked nonlinear-table difference

Source inventory was obtained by parsing Python assignments with `ast.literal_eval`, decoding our compressed JSON, and counting their term lists; neither estimator was run.

| output partition | our v32 terms | v17 base terms | v17 extra (3,1) terms |
|---|---:|---:|---:|
| (1,) mean | 3 | 5 | 0 |
| (2,) | 3 | 3 | 0 |
| (3,) | 3 | 3 | 0 |
| (4,) | 3 | 3 | 0 |
| (1,1) | 4 | 17 | 1 |
| (2,1) | 5 | 20 | 2 |
| (2,2) | 4 | 11 | 1 |
| **total** | **25** | **62** | **4** |

Anchors: `estimator_k3_v32.py::_TABLES_B64`, `_wick_program` and `_layer` lines 1069–1113; `external/504aldo/estimators/estimator_v17.py::TERM_SPECS` lines 129–198 and additions lines 246–250.

Our mean is `w(0,1) + D3*w(3,1)/6 + D4*w(4,1)/24`. The public mean additionally contains:

```
D3**2 * w(6,1) / 72
+ D3*D4 * w(7,1) / 144
```

These are **live computation**, not unused reference metadata. v17 `_term_prog` lines 289–318 consumes `D1_IPS`, including `(1,)`; both terms survive modes 1 and 2. Lines 528–538 construct `PK1` and assign `pk1v=PK1[0]`; lines 565–567 append `pk1v` at the last layer. `NO_CORR` defaults true, so the unrelated fitted mean rider is not responsible for these terms.

The coefficients can be independently derived: expanding the univariate cumulant differential operator `exp((D3/3!)*d^3 + (D4/4!)*d^4)` gives `D3^2/[2(3!)^2]=1/72` and `D3*D4/[3!4!]=1/144`. A complete quadratic truncation would also contain `D4^2*w(8,1)/1152`; the public empirical pruning is not a mathematical reason to exclude it. Define our own explicit diagram/degree truncation, derive its table, and test it. The expansion's asymptotic correctness does not guarantee improved finite-network prediction.

The mixed outputs likewise include absent products such as `C*D3`, `C*D4`, `C*D21`, `D21^2`, `D3*D21`, and `D4*D22`. They can change covariance and source births at the same incoming state. Therefore checking only the four `(3,1)` rows does not compare the nonlinear closures.

There is also an API/version distinction behind the shared name SIMPLE. The public `lean/lean_k3.py::build_term_table` lines 40–76 calls `get_all_terms_iso(k_max=3,d_max=4)` and documents 94 SIMPLE terms before its exclusions; the dump comments describe 90 before empirical pruning. Our pinned experiments call the newer `get_all_terms(..., augment=False)` and count 40 before selecting the 25 shipped consumers. Matching a name is not matching a diagram inventory. Generate and compare canonical monomials, including multiplicities and orientations.

**Connected convention:** these are the repository's `pK` objects, not raw joint moments. For example `K11=pK11`, whereas `K21=pK21-2*pK1[:,None]*pK11`; see our `pK_to_K`, v17 `PK2K_TABLE` lines 200–218, and `local_sim/e14_scalar_feed_identity.py` lines 353 onward. Generating ordinary raw moments and feeding them into this centering table would double-count disconnected contributions. A closure generator must establish its conversion independently with polynomial/quadrature fixtures.

**Historical constraint:** earlier A5 tested a raised diagram budget with scalar fourth-order state and lost (approximately 4.3448e-8 versus SIMPLE 4.0047e-8); full matrix fourth-order state was approximately 1.9123e-8. Those historical MSEs are recorded experiment values and require their original truth-noise context before reuse as performance evidence. This review does not reopen a generic "more diagrams always helps" lane. The new hypothesis is the interaction between the existing regenerated per-neuron core and nonlinear terms consuming it, directly motivated by the now-identified public source difference. Record that scoped reopening and its falsifier before measurements.

## 2. The protected-feed representation difference

v17 builds, at birth, `v=w1^2`, `d=v*dG`, `y=(metric/4)*w2`, and `u=y*dG` (lines 585–602). In addition to the off-diagonal lambda hub, it carries:

* A diagonal hub with matrix `diag(d)` and vector `y`.
* A rank-one M-leg contribution `u v^T`.

Their repeated slices are

```
s_E = d*y + u*v
E = offdiag((d y^T + v u^T)/3).
```

At birth their tensor has no all-distinct entries. Thus this tensor is exactly the dense repeated-support completion `D(s_E,E)` represented by M-leg `diag(s_E)+3 E^T`.

v17 removes these slices from `S3c/S21` **before** compressing `Rres` (lines 643–665). It then puts `u` and `y` in additional transported Z columns, `v` in an L column, and `d` into static hub column coefficients (lines 666–680). Consumers use the protected diagonal hub through `P*c2` and its corresponding contractions (lines 735–753).

Our v32 route-(i) implementation only constructs the off-diagonal lambda hub (`estimator_k3_v32.py` lines 1192–1217; `estimator_k3_v32_i.py` sets routes to `('i',)`). Its equivalent diagonal/rank-one content remains in the residual compressed by `_p2_compress`. This remains true of the joint-recipe diagnostic derived from v32_i. Hence copying the lambda and q31 coefficients did not copy the approximation boundary.

Write `P_r` for the matrix rank-r approximation. Protected and unprotected residual spellings differ by

```
P_r(R+E) - P_r(R) - E.
```

This is generally nonzero, even when the two full-rank tensors are identical. The low-rank approximation can also introduce residual diagonal entries, so compare D3 as well as D21 after birth and transport. This is a concrete interaction hypothesis, not evidence that the public allocation is more accurate.

A deterministic n=5 algebra fixture used nonconstant `w1,w2,dG`, a sine-built residual, and a cosine-built dense transport. It verified:

| check | result |
|---|---:|
| protected feed vs repeated completion, maximum absolute difference | 4.34e-19 |
| full-rank birth gauge equality, maximum absolute difference | 8.67e-19 |
| difference after rank-1 SVD approximations, tensor norm | 4.39e-3 |
| after dense transport, D3 difference norm | 2.04e-4 |
| after dense transport, D21 difference norm | 3.72e-4 |

These are synthetic algebra values, not network accuracy, estimated benefit, or an evaluation result. They prove noncommutation, not the practical size or preferred direction.

Fixture recipe for reproduction: `x=arange(1,6)`, `w1=.15+.11*x`, `w2=.3-.025*x`, `dG=.005+.003*x*x`, `metric=2`; define `s_E,E` as above. Set `R=.0004*sin(outer(x,x+1))+.0001*outer(x,x[::-1])`, clear its diagonal, and set total birth diagonal `.002*x`. Compare `D(.002*x,R+E)` to `D(.002*x-s_E,R)+F`; then replace each residual by its rank-one SVD approximation. Transport the difference on all three modes with `Q=cos(outer(x,x+.25))/3`. The reported norms use float64 Euclidean/Frobenius norms. Here `F` is the explicit diagonal hub plus M-rank-one tensor from this section; no estimator implementation is involved.

Other unmatched approximation details: v17 uses residual rank 16 and a layer-weight-column sketch (`R_RES`, line 324; lines 655–665); ours uses rank 32 and residual-norm-selected columns (`_p2_compress`, lines 623–655). The latter improves neuron-permutation behavior, but that invariant does not prove better downstream error. Compare equal ranks and identical residual targets first; any sketch change must preserve the repo's determinism/permutation requirements. Do not copy the reference's first-weight-column sketch into a release solely to reproduce it.

## 3. Same-state comparison needs no 1024-cubed tensor

The packet's proposed dense adaptation at n=128–256 is unnecessarily indirect. Moreover v17 enables regenerated fourth-order state only when `n==1024` and `depth==16` (lines 349 and 378); naive small-shape execution disables the mechanism being studied. A documented test-only adapter can override this, but live-width slice evaluation is simpler and avoids uncertain scaling.

At a nonlinear boundary the local consumer needs only:

```
mu_pre, C_pre, D3, D21, dG, lambda, layer_index
```

These occupy O(n^2) space. The complete incoming third-order tensor is unnecessary for comparing the moment table: its two consumed slices suffice. Carry-forward is the same linear Wick scaling when the input state is held fixed. For new births, compare compact source factors and their D3/D21 after one common arbitrary dense transport using each factorized contraction formula; no cubic tensor is needed at live width.

Use three layers (early, gap onset, mid-depth), from one fixed existing permitted DESIGN-network trajectory, with truth unopened. Capture both directions: reference boundary into both consumers, and our boundary into both consumers. Save named inputs, source SHA, coefficient-table hash and representation convention. Compare, in order:

1. Wick coefficients and every scalar/mixed `pK` output before centering.
2. Centered mean, covariance, K3/K21 and regenerated fourth-order core.
3. Full-rank newborn factors before compression.
4. Identical-rank and identical-sketch newborn factors after compression.
5. One common dense transport followed by D3/D21 readout, both one-source and a small source stack.

Full-rank algebra parity is a test threshold; final prediction accuracy is a separate statistical gate. Hold functions/coefficients fixed when testing representation; hold representation fixed when testing coefficients. This prevents another "single flag changed several mechanisms" interpretation.

Suggested resource shape for the extracted consumer/birth diagnostic: CPU-only, 2 CPUs, 16 GiB conservative RAM declaration, live n=1024 but one boundary at a time; finalize the peak estimate after inspecting all allocated temporaries. Full-trajectory capture requires its own larger measured envelope if retaining stacks. All nontrivial live-width work goes through workerq with the mandated optimization review. No new Monte Carlo truth is required for the first discriminator.

## 4. Recommended bounded next slice

1. **Semantic audit and own table derivation first.** Independently generate the selected cumulant diagrams. Verify connected conventions, factorials, symmetry, scaling and polynomial/quadrature oracles. The public pruning mask was selected from measured `term_rels.json` (`external/504aldo/lean/dump_k3_tables2.py` lines 14–25); it is a diagnostic reference, not our unqualified release table.
2. **Cheap final-readout arm.** Add the predeclared univariate quadratic expansion only at the final mean, holding the trajectory unchanged. Its cost is O(n), and it directly tests whether the omitted mean terms explain any immediate gap. Compare teacher-forced closure deltas before any accuracy screening; do not assume this alone will reach the release target.
3. **Coherent recurrent arm.** Apply the same independently derived closure consistently to all moments needed for covariance and future births, with the current regenerated fourth-order state. Include a mean-only recurrent arm only if predeclared attribution warrants it. A full richer local table costs O(n^2) work per layer and should be metered before the accuracy screen; avoid large all-term stacks when streaming terms suffices.
4. **Protected-feed arm.** At an equal fixed residual rank, protect analytically known diagonal/rank-one pieces outside the compressor. First require uncompressed same-state algebra parity; then measure the effect of finite-rank redistribution. Do not conflate this with lambda fitting or imported constants.
5. **Only after closing the accuracy mechanism, revisit the cost-side v29 family.** Its separate age tiers retain per-source factors, rather than the failed single merged cubic core. A bounded own implementation of that structure is a different intervention from N1, but requires its own meter and error tests. Thin D21 feedback's measured +6.9% cost does not establish failure on adjusted score if coupled to a stronger closure; it establishes failure of the old +1% cost gate. Any new economics must be preregistered, not changed after results.

Keep the full-covariance, source-factor, regenerated-core family. The next bottleneck is **a matched and internally consistent nonlinear closure**, not demonstrated need for wholesale replacement. If these discriminators fail, report their exact scope and reassess; no honest audit can promise a major leaderboard gain in advance. A small supported bundle remains useful as a frozen fallback but is not evidence that the requested major-release target has been met.

## 5. Scope of N1 and P4 conclusions

The continuous N1 accuracy result is evidence against the tested energy/fixed-basis merge policies at the tested small widths and cheap rank ratios. It does not disprove every shared-factor policy at n=1024. Conversely there is no evidence to spend another broad compression sweep before the concrete closure discrepancy above is resolved.

P4 measured 0.2802 B for current feasible spellings, with depth-6 peak above 8 GiB; this does not prove a universal exact-arithmetic lower bound. However the additional measured 1.8% alone cannot meet 0.27 B, and chasing it before the accuracy discrepancy is resolved is unlikely to deliver the desired major jump.

The worker's completed measurements, failed gates, no-submission decision and corrected Skeptic record are useful. The architectural conclusion must be narrowed: **the tested additions fail on our 25-term nonlinear closure and current residual approximation; the public chain's richer closure and protected feed allocation have not been matched or ruled out.**
