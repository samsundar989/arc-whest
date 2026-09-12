# Architecture review — September 12, 2026

## Decision

**Do not make causal compressed signed-law K4 the next submission project. First test a lossless contraction change in the standing L2 program.** There is an explicit three-matmul spelling of its current four-matmul D21 readout. It retains the existing nonlinear closure, scalar K4, all source ages, precision policy and Strassen depth. It needs neither a physical nonlinear K4 source nor a compressed residual. Its correctness is an algebraic identity; its floating-point accuracy, complete counted saving and deployment behavior still require evidence.

If that bounded attempt fails, the next research action is a full-rank differential audit of compact K3 against the standing closure. Simply switching off the new K4 in the existing compact program is not a supported route to a better candidate: that off control already performs poorly on exposed P01, and its nonlinear equations differ from the standing program.

This is an independent read-only architecture review of source and existing unprotected artifacts, following CURRENT_STATE, the LOG tail, GAME_PLAN and STRATEGIST_PACKET_2026-09-12. No protected labels, seed blocks, experiments, estimator edits or heavy runs were used. This document is a proposed design, not a passed experiment or a submission claim. During review Samuel explicitly reopened eligibility review and candidate preparation. Actual submission still needs separate approval; no external messages are authorized and no agent designates.

## 1. The smallest concrete candidate change

The standing `estimator_k3_strassen_L2.py` implements each lineage with three transported matrices `A_D`, `A_L`, `M`, and a fixed birth vector `w2b`. At lines 656–668, define `C_L=A_L*w2b`. The D21 numerator is

```
2 (A_D*M) @ M.T
  + (M*M) @ A_D.T
  + 6 (A_L*M) @ C_L.T
  + 3 (A_L*C_L) @ M.T.
```

Distributivity gives exactly the same real-valued numerator as

```
[2 (A_D*M) + 3 (A_L*C_L)] @ M.T
  + (M*M) @ A_D.T
  + 6 (A_L*M) @ C_L.T.
```

The existing final division by three and diagonal mask stay in place. D3 stays unchanged. Every standing `_layer` Wick-table term, raw-to-centered conversion, birth subtraction, scalar K4 update, source transport, saturation and final mean-only shortcut stays in place. This is one changed contraction, not a new closure.

The earlier symmetric-packing experiment does not test this identity. LOG around lines 38700–38874 and `experiments/2026-09-10_slice5/packed_feasibility.json` concern object-identical operands **within** one multiplication and the meter's symmetry discount. Its site inventory explicitly contains both products with trailing `M.T`, but no proposed cross-product addition before multiplication. This proposal does not seek a discount on an existing multiplication; it eliminates one multiplication. It therefore does not reverse the historical packing verdict.

### Proposed cost and evidence gate

At depth 16 with all generations retained, D21 is read with 1 through 14 lineages: 14 wide calls, or 105 lineage-equivalent products, disappear. The historical standing cost model has 797 total equivalents, so the eliminated products are about 13.2% of that dominant work. This is a conventional arithmetic planning estimate, not an official FLOP measurement or a score prediction. Wider operands, Strassen additions, new input additions and masks must be metered in the complete predict path. Setup is measured separately under the canonical lifecycle and is off the per-MLP FLOP budget.

| Component | Proposed change | Evidence/cost scope |
|---|---|---|
| D21 dense products | Four to three per nonempty readout | Exact displayed identity; 105 fewer lineage equivalents at d16 |
| Source transport, D3 and ordinary births | None | Same arrays and semantics |
| Scalar K4 and nonlinear centered moments | None | Keep the frozen L2 algorithm |
| Matmul backend | Same L2 Strassen backend | Float32 reassociation must be tested |
| Temporary arrays | One combined left operand | Schedule its lifetime; do not keep all input products and old accumulators alive |
| Residual operations | Expected fewer matmul-wrapper operations, extra elementwise input work | Must record full operation count and official subprocess residual headroom |
| Process RAM | Expected similar order, unmeasured | Width1024 lifecycle observation and enforced Linux cap remain necessary |

**Proposed bounded first contract:** at most two working days to obtain a tiny independent contraction oracle, frozen-program trajectory parity and one official-context cost comparison. Stop or synthesize at that checkpoint rather than expanding into a new architecture.

Before tests, preregister: signed, nonconstant, cancellation, zero and off-shape contractions at float64; frozen float32 depth16 trajectories with unchanged scalar K4; both matrix orientations; nonfinite and overflow controls. A practical proposed equivalence gate is `1e-10` normalized error in float64 contraction fixtures and `2e-5` material-relative error for every consumed float32 D3/D21/mean/covariance state, with all finite and the original safety handling unchanged. These are proposed thresholds, not retroactive replacements for any old fixture. The reviewer should independently validate the chosen tolerances before execution.

**Proposed cost screen:** complete official counted FLOPs must fall by at least 8% against the exact same frozen L2 configuration, without increasing operation count or observed memory materially. Below 8%, stop this candidate attempt and document the actual gain rather than spending labels automatically. Passing permits a separate fresh DECIDE contract with paired common random numbers, the repository's arithmetic score gate (proposed `gate=1.05`), noise floor and independent Skeptic. A 1.05 gate is a proposed new isolated candidate gate, not permission to move any existing gate. The exact score multiplier and server offset must be included. No guaranteed 13% score improvement follows before that comparison.

The mechanism has a much shorter path than signed-law K4: an exact identity, then rounding/cost, then the existing candidate validation pipeline. Eligibility/provenance review and candidate preparation can now run under Samuel's reopened scope; neither algebraic independence nor using old candidate code proves eligibility on its own.

## 2. Why the failed K4 path is not ready

The active entry point is `local_sim/p2_k4_integrated_first_source.py:1546`, not the older comparison implementation copied earlier in that file. Its own docstring states that the bank is synthetic, externally supplied and not a later physical source. Lines 1592–1599 map `V` through raw `W`; after birth it does not pass through ReLU or even the old K3 Wick factor. Meanwhile its Gaussian covariance backbone changes at every layer. Thus a linearly carried bank cumulant is consumed against a different evolving covariance scale.

At lines 1614–1629 the code floors variance at `1e-8` and uses `w4=(alpha^2-1)*phi/(var*sqrt(var))`. Flooring avoids division by zero but does not make an indefinite covariance a valid moment matrix, and a fixed unscaled K4 divided by a collapsing variance is not a controlled small correction.

The corrected integrated covariance consumer at lines 1712–1717 is explicitly symmetrized with oriented D31 terms. The n8 failure therefore cannot be explained merely by the old frozen template's asymmetric D31 expression. Its diagonal remains Gaussian-only while off-diagonal K3/K4 corrections are added, which supplies no PSD guarantee even if each tensor contraction is algebraically correct.

Existing raw evidence is decisive about the implemented trajectory:

* `integrated_first_source/n8_result.json`: 186 of 187 checks pass. The failed registered gate is `all_preactivation_variances_positive_and_usable`, minimum `-0.1177162081`. Negative variances recur at later layers; this is not a single harmless roundoff event.
* `integrated_first_source/n8_variance_diagnostic.json`: L2 covariance minimum eigenvalue is `-0.231173159`. At the subsequently failing neuron, Gaussian plus all K3 terms gives `+0.033249802`; the K4 term contributes `-0.150966007`. Their sum explains the failure to reconstruction precision. This is a same-state component decomposition, **not** an end-to-end rerun proving that deleting K4 yields a stable candidate.
* `integrated_first_source/tiny_result.json` and `tiny_diagnostic.json`: the distinct n4 fixture fails 79 of 172 checks, with tiny-variance amplification and overflow. Some early discrepancies also exceed its stringent `1e-8` tolerance at ordinary float32 rounding scale; those should not all be described as overflow. The later unstable trajectory independently blocks the full-width run.
* `integrated_first_source/shortcut_result.json` and its Skeptic verdict: the positive feed shortcut preserves the n8 failure. Its accepted speedup is a cost improvement to the same invalid synthetic trajectory.

The model should therefore not be repaired by clipping its covariance and immediately metering width1024. Symmetrization does not ensure PSD; projecting to PSD or damping K4 changes the represented moment update and could suppress both its instability and any intended signal. Such a new approximate closure requires its own hypothesis and evidence. An exact positive measure gives PSD by construction, but constructing, transporting and compressing that measure is precisely the unresolved problem.

## 3. Finite cumulants do not determine a nonlinear pushforward

The physical n2 signed-law fixture is valuable as an accounting oracle. Its `branches` and `continuous_law` functions in `test_p2_k4_causal_signed_physical.py` resolve activation boundaries for a supplied chain, and its `bank` uses those sectors. An exact tiny, future-boundary-aware bank is not evidence for a causal fixed-rank width1024 bank.

There is also an elementary obstruction to interpreting four moments as an exact sufficient state. Consider two symmetric scalar laws:

* A has mass `2/3` at zero and mass `1/6` at each of `+sqrt(3)` and `-sqrt(3)`.
* B assigns total mass `8/9` equally to `+/-1/sqrt(2)` and total mass `1/9` equally to `+/-sqrt(5)`.

Both have moments `(m0,m1,m2,m3,m4)=(1,0,1,0,3)`. Yet

```
E_A ReLU(X) = sqrt(3)/6,
E_B ReLU(X) = 4/(9 sqrt(2)) + sqrt(5)/18.
```

These are different. Their signed difference is invisible to all moments through degree four and becomes visible immediately after ReLU. This is an analytic example, not a new challenge experiment. The accepted repository mixed-source and hidden-residual counterexamples expose the same class of missing information in more relevant multivariate settings.

A causal approximate state could retain weighted activation atoms and transport each through the actual next ReLU, keeping moments 0–4 and conditional cumulant differences coherent. But fixed-size recompression loses activation-boundary information, and moment matching alone does not bound the lost future means. Representing positive and negative banks separately also does not make their total corrected covariance PSD. No repository result yet supplies a compression-error bound or an adequate rank for that representation.

Consequently the near-term verdict is **NO-GO for building this as the next candidate**, not a theorem that every useful approximate K4 closure is impossible. The 704.165B first-source meter and 818.494B supplied-core meter price different partial programs. They cannot be added or used as an inclusive price for physical source construction, all conditional increments, negative full-rank factors, nonlinear transport and recompression. The accepted 715,844,864-byte named-array model is not a process bound.

## 4. Compact K3 is a second, bounded audit, not the ready fallback

The existing frozen compact `exact` arm is K4-off relative to the **new** four-arm mechanism. Its P01 report records 816,292,940,328 FLOPs and mean raw final MSE `2.11130e-5`, against measured averaged-truth noise `9.31914e-9`. P01 is exposed DESIGN. No paired comparison with standing L2 is in that report, so neither a cross-dataset gain ratio nor a compression-specific accuracy verdict follows.

Source inspection identifies concrete confounders before compression:

* `p2_compact_candidate_template.py:203–217` uses Gaussian-only diagonal variance and Gaussian-only third central moment for births. Standing L2 `:829–928` executes the full Wick `pK` table and `pK_to_K` centering before reading `K[(2,)]`, `K[(3,)]` and `K[(2,1)]`.
* The compact template `:204–205` uses a `0.25` D21 covariance coefficient, while the integrated synthetic instrument `:1707–1710` uses `0.5`. Do not repair this by inspection alone: tensor normalization and the standing table must be independently aligned.
* Standing L2 already carries scalar K4 (`c4`, `D4`, `D22`, and its projection). Calling a new candidate simply “K3-only” must not silently delete that existing behavior. Preserve scalar K4 in the main frozen-L2 comparison. A diagnostic disabling scalar K4 must disable it in both controls and be labeled separately.

There is a clean full-rank bridge for a later compact audit. A standing lineage represents

```
Sym(A_D, M, M) + 3 Sym(A_L, M, A_L diag(w2b)).
```

It fits compact `Source` exactly with `A=A_L`, `P=M`, `w2=w2b`, `s=e=0`, `Z=A_D`, `L=I`, and optional K4-feed fields absent. Its `materialize` equals `A_D`. The D3/D21 identities then reproduce the standing tensor, provided the standing off-diagonal D21 mask is applied and `W` versus `W.T` orientation is aligned. This is an exact full-residual encoding, not a saving.

More generally one may split

```
A_D = P diag(s) + 3 A diag(e) + Z L.T.
```

At birth, standing `A_D=diag(S3)+3 S21.T`. Choosing `s=S3` and any prescribed `e` leaves residual `R=3 S21.T-3 A diag(e)`. Retaining `R` exactly preserves the identity; declaring it rank16 is a new approximation. Full residual rank may erase the proposed two-leg saving. Full shared rank and full residual must first reproduce the frozen nonlinear chain before either approximation is tested.

If the immediate exact contraction candidate does not earn its gate, allow a second bounded two-day layer/state parity audit using this bridge. The falsifier is any mismatch in independent dense K3, D3/D21, centered nonlinear outputs or next births while both ranks are full and all scalar-K4 semantics match. Do not continue with rank sweeps or protected fitting until that mismatch is understood. Only after parity should one separately contract a single shared-basis/residual approximation and a complete cost gate.

## 5. Action order and stopping rule

1. Preregister and independently check the three-product D21 identity against frozen L2, then meter exactly that candidate. Preserve all current candidate bytes and evidence.
2. If the proposed parity/cost gates pass, proceed to a fresh score contract, candidate-specific validation, current-byte Full preflight, Linux memory evidence, official mini and eligibility/provenance preparation. Submission remains a separate user decision.
3. If they fail, synthesize the result and perform only the bounded full-rank compact-closure differential audit described above. Do not make causal signed-law K4 a dependency of this cheaper question.
4. If neither route supplies a conforming, meaningfully cheaper program, retain standing L2 and prepare a contribution narrative with reviewed claims. Preserve the failed source/closure findings as scoped evidence; do not claim a general scientific kill or that exact spelling has exhausted every possible saving.

This ordering spends effort on the smallest concrete route to a better score before committing to a distributional representation whose source, compression, stability and price are all still open.
