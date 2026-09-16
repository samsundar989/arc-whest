# Architecture audit — 16 September 2026

Read-only audit of the repository, including the actual v31 implementation and pinned public competitor source. No estimator was executed, no labels or seed blocks were opened, no experiment was run, and no frozen or operative file was changed. This report records source deductions and proposed experiments; it does not grant a promotion or supersede the active Slice 7 contract. Current competition pages and external literature are being audited separately.

Evidence labels: **CONFIRMED** means directly visible in source or an identified existing artifact; **INFERRED** means a deduction whose assumptions are stated; **SPECULATIVE** means a research hypothesis requiring a new registered experiment. Existing competitor measurements are **REPORTED**, not independently reproduced here.

## Decision

The project has a credible algorithm family and a much improved implementation, but no demonstrated winning architecture. v31 has crossed from an exact but expensive K3 representation to a useful compressed representation. It still pays for every old source at almost full width, and its claim to have the complete cheap κ4-to-κ3 feed is unsupported by the implemented tensor structure. These are the two highest-value places to work.

The existing Slice 7 direction is broadly right: settle the κ4 discrepancy, add useful feedback, reduce mechanical cost, and then investigate old-source compression. The audit adds a specific likely explanation for the κ4 gap and two corrections to N1: test the full carried tensor rather than only repeated slices, and require an economically useful complete implementation rather than a rank-halving proxy alone.

Do not undertake a rewrite before those discriminators. Preserve v31 as a frozen baseline, and use candidate generators to make one identifiable intervention at a time.

## What actually runs

`estimator.py` remains the historical v2.6.1 submission file. The latest submitted algorithm is `estimator_k3_v31.py`, sha256 `6dde5fd43ad41e8ec7199b742dd554d22d506c9cf85a532027e86a590d0a787c`, packaged as non-designated canary #331317. Source: `experiments/2026-09-16_slice7_canary/package_receipt.json`, `docs/CURRENT_STATE.md`, and the LOG entry at 15:43 UTC on September 16. Candidate role must be read from the receipt/current state, never inferred from the root filename.

**CONFIRMED from v31:**

- The state is `(mean, covariance, K3 tower, scalar c4, per-neuron gk4)`; `_layer` begins at line 876.
- Each layer transports covariance and a retained source tower, evaluates a frozen Wick/raw-moment program, centers moments, and appends a new source after subtracting content already represented by the old tower.
- `_Tower` at line 367 stores dense `A_L` and `M` blocks for all retained source ages. At the live shape the generation selector retains them all. `_truncate` at line 748 is a budget fallback, not the submitted compression policy.
- P2 represents the formerly dense `A_D` as `M diag(s) + 3 A_L diag(e) + Z L.T`. The residual is rank 32, selected by `_p2_compress` at line 595. This removes one dense transport and one dense contraction per mature source-layer, while retaining thin work and reconstruction.
- `_dslice21` at line 681 uses two wide dense products plus a thin product. It still builds dense intermediates proportional to the number of old sources. P2 has not solved the old-history scaling problem.
- κ4 transport at line 959 already contains both `-(W.T**2)@(diag(cov)*lambda)` and `+var_pre*lambda`, with previous-layer indexing. A simple missing-subtraction diagnosis is not supported by the source.
- The post-ReLU `g_out` at lines 1141–1147 uses the r=1 harmonic matrix-core coefficients. This corrects the earlier normalization failure. `c4_out` is still computed, but v31's active mean consumers read `dG`; the carried scalar should not be mistaken for an additional independently consumed correction.
- Strassen remains depth 4/minimum block 64. Final-layer D21/birth work is pruned, but the complete covariance sandwich still executes at line 917. A final-layer diagonal-only covariance contraction remains an actual mechanical opportunity.

“Exact K3” here means the selected approximate cumulant program before P2 truncation, not exact Gaussian integration of the network. The class docstring still says “exact” even though rank-32 residual truncation is active. Comments also alternately describe λ as an unfitted initializer and “fitted end to end against score.” These are stale source descriptions; lineage/provenance artifacts, not comments, establish how current constants were obtained. Do not alter the certified file to clean comments in place.

## The immediate accuracy discrepancy: repeated slices are not the full feed

This is the most actionable finding of this audit.

`experiments/2026-09-14_slice6a_p1/feed_already_present.md` correctly establishes that the SIMPLE term table routes fourth-order slices into `K[(3,)]` and `K[(2,1)]`. However, it then concludes that the full feed is present and adding the explicit X3 feed would double count it. Its final paragraph admits that no X3-to-F21 mapping was derived.

The source supports the narrow claim, not the general conclusion:

1. v31's new all-distinct third-order source is the Gaussian two-leg hub, with `fac1 = diag(w1) C11` and its companion `fac1 diag(w2)` (`estimator_k3_v31.py:1036`). κ4 changes the repeated-slice residual `S3,S21`, which becomes the `A_D` leg. No additional κ4-dependent full hub is appended.
2. Public v17 uses the same kind of moment table **and also** carries a κ4-dependent full hub. Its birth is `X3 = diag(w1) G_pre diag(w1)`, `Y3 = (m/4) w2 1.T`, plus a rank-one `M_t1`. It subtracts their repeated slices before assembling the residual. See `external/504aldo/estimators/estimator_v17.py:584–605`, `:639–679`, and `docs/derivation_code_map.md:292–308`.
3. That implementation retains static `c1,c2` coefficients and two extra transported thin columns. Future D3/D21 contractions explicitly consume the feed (`estimator_v17.py:735–755`). v31's `_Tower` has no equivalent `c1/c2/y` fields or contractions.

**INFERRED algebraic distinction:** at birth a change represented only by `Sym(A_D,I,I)` vanishes at every all-distinct index `(i,j,k)`. In contrast, the off-diagonal λ component of `Sym(X3,I,Y3)` is generally nonzero at all-distinct indices: one summand is proportional to `X3[i,j] y[k]`. Therefore knowing and matching D3 and D21 at birth does not determine this part of the tensor. A later dense transport mixes all-distinct entries into the repeated slices that determine means. This is precisely why a memoryless repeated-slice closure can look right locally and fail after transport.

This establishes a structural non-equivalence at fixed incoming state; it does **not** measure how much raw error the missing term causes at 1024×16. It also does not establish that every diagonal component of the competitor feed is missing: some repeated-support content can already be represented by v31's residual. Adding the whole feed without the matching subtraction would indeed double count its repeated part.

**Recommended refinement inside D2/D3:** keep the planned diagnostic order, but explicitly compare the full birth tensor on a tiny forced-active fixture, including all-distinct entries. Hold incoming mean/covariance/Wick coefficients fixed; vary the off-diagonal λ component. Compare current birth, reference feed, and their repeated slices both before and after one arbitrary dense transport. The corrected candidate must add the missing full tensor content and subtract exactly the repeated content already represented by `S3/S21`. Then measure the isolated effect at 1024×16 under the existing contract. A zeroing experiment that removes κ4 from the whole term table is an ablation of existing repeated-slice feedback, not proof that the missing full feed is present or absent.

This is a sharper next test than assuming the raw gap comes from λ scale. Existing λ scaling gives a flat optimum near 0.794 and little further benefit; source already contains the proposed dG transport terms. The depth profile remains testable through D1, but an exact tensor comparison should resolve the representation question without any truth labels.

## How far the current architecture is from winning

**CONFIRMED existing graded evidence:** v31 #331317 reports raw `2.9074e-8`, multiplier `0.3010253`, adjusted `8.75203e-9`, 0/100 failures, worst wall `68.13 s`, worst residual `0.140244 s`. F21 #330950 reports adjusted `1.5957e-8`. Their paired 50-public-network arithmetic difference has t95 `[6.88e-9,7.53e-9]`; the reported gain is 1.8233×. The public truth-noise floor is unavailable. Sources: both JSON graded reports under `experiments/run_reports/` and LOG 16:09 UTC September 16.

The parent audit's live Chrome inspection of leader #331308 gives displayed detail `2.448e-9`, raw `1.678e-8`, multiplier about 0.1459. The descriptive v31 gap is about 3.575×, decomposing into about 1.733× raw error and 2.063× cost. The leaderboard rounds this to `2.4e-9`; use the more precise submission detail. This is not a paired experiment and a new live board can change the target.

For a fixed cost multiplier, the conditional raw error required to tie that dated score is:

| Multiplier | Raw MSE required |
|---:|---:|
| 0.3010 | 8.13e-9 |
| 0.20 | 1.224e-8 |
| 0.15 | 1.632e-8 |
| 0.12 | 2.040e-8 |
| 0.10 floor | 2.448e-8 |

At current raw error, even free computation would score `2.9074e-9`, above that leader. Accuracy improvement is necessary. At a plausible improved raw `1.9e-8`, winning requires multiplier below about 0.1288, so cost improvement is necessary too. A 1.5× raw gain alone would produce approximately `5.83e-9` at current cost. A combination of raw `1.9e-8` and cost 0.18 gives `3.42e-9`, valuable but not first place at today's target.

These are score-identity calculations, not forecasts or proposed promotion thresholds. With adaptive per-network cost, evaluate the mean of each network's error×multiplier; do not multiply mean error and mean cost.

The current handoff is appropriately ambitious about intermediate gains, but the phrase “only route to a prize” for N1 overstates what is known. Failure of one weighted basis does not prove every alternative architecture impossible.

## N1: a worthwhile experiment with two missing conditions

**REPORTED:** public F72/F73 maintain accuracy near ranks 384 after four transports and 224 after seven. One notch lower can be a cliff. F88 prices merged Tucker contractions at roughly `r^3/n^2` source units per old-tier layer, and estimates an advantage only below r≈150. These are competitor reports and pricing arguments, not independent measurements on v31. Sources: `external/504aldo/docs/community_post.md:98–108`, `docs/findings_log.md:2143–2148`, and `docs/claims_evidence.md:59`.

**SPECULATIVE:** output-relevant old-source directions may need a smaller rank than energy-optimal directions. This is a legitimate new hypothesis. However:

**1. A final mean-gate Jacobian is not the sensitivity of the cumulant recurrence.** The proposed `J = product(D_l W_l)` propagates small perturbations of the mean through a mean-gate approximation. K3 errors also enter intermediate D21, covariance, sigma/alpha, Wick coefficients, and new source births. Those routes are explicit in `_layer`. A low-rank final mean map can discard directions that matter through future covariance. Treat J-weighting as a proxy, not the correct observability metric by definition.

A bounded refinement is an offline, tiny-shape comparison of that proxy against directional derivatives of the **complete** remaining cumulant recurrence, with both D3 and D21 perturbations. This can assess whether the proxy orders output errors before investing in the production compressor. A closure-aware weighted POD/balanced-reduction metric is a research direction, not a demonstrated method. Specify the PSD objective or generalized eigenproblem: naively multiplying two PSD Gramians generally produces a nonsymmetric matrix, not an ordinary orthogonal PCA objective. Online Jacobian, Gram, QR, join, rotation and core contraction costs must all be billed.

**2. Rank-halving is not the economic gate.** At n=1024, the reported `r^3/n^2` factor gives 54 units for r=384, 6.75 for r=192, 3.22 for r=150 and 2 for r=128. Across about ten relevant layers, r192 alone costs about 67.5 units before other work, whereas the plan projects a 20–30-unit old tier. A 384→192 accuracy success could pass N1's initial test yet fail its promised cost. Include r128–150 and a complete metered old-tier implementation before claiming that merger can achieve ≤0.20B, much less the ≈0.12B needed with raw ≈1.9e-8.

Retain the existing source-level cost model and a measured complete ledger. Do not repeat the 7→4 versus actual 6→4 error or assume that storage rank directly determines contraction cost.

The literature audit independently raises another omitted cost: a general r×r×r core transformed by a dense r×r restricted propagator in all three modes costs O(r^4), beyond the quoted readout cost. Transporting the physical basis as `Q_next = D W Q` can leave the core unchanged between basis changes, but orthogonalization, changing the retained span, and joining new sources can still require triple core transformations. This must be specified in an executable schedule, not assumed away. Count the additional O(n²r) basis work and O(n r³+n²r) generic readout too; the public r³/n² price is an incomplete asymptotic planning approximation, not a full flopscope quote.

## Target architecture, confidence separated

**Evidence-backed product direction:** dense mean/covariance backbone; complete selected K3 nonlinear recurrence; recent sources retained densely; exact separable identities with a controlled low-rank residual; cheap diagonal-plus-λ κ4 regeneration; coherent full κ4→K3 source feed; thin D21 birth feedback; final-layer mean-only work; float32 and efficient counted matrix products. v31 has only part of this combination, but its 1.823× graded gain supports the direction.

**Unproven frontier component:** older sources must retain the information needed by the nonlinear recurrence at much lower marginal cost. A shared basis is a plausible intermediate. A closure-aware merged tensor, task-specific sketch, or another structured sufficient approximation is research. The top submissions' variable FLOPs and layer-error profiles do not uniquely identify their method, let alone prove they use a merged Tucker core. The public competitor explicitly says its authors do not know the leaders' algorithm.

**Avoid architectural churn:** a new probability-law representation or full AUGMENT implementation is not a prerequisite to fixing v31. The repo already measured dense AUGMENT accuracy headroom but its high-dimensional state is unaffordable. Use that as a reference, not as an implementation plan without an executable memory/cost analysis.

## Practical sequence and architecture hygiene

1. Finish the contracted D1–D3 diagnosis, adding the full-tensor feed discriminator described above. Obtain the required Samuel prediction before new strategic measurements. Use P11 as DESIGN; fitted constants require a separate frozen evaluation block under the operative policy.
2. Freeze a corrected feed candidate only after tiny full-tensor/transport parity and a complete cost ledger. Validate the raw gain against current v31, then prepare a canary if the current contract passes. Do not promise 1.5× from source inspection.
3. P3 and P4 remain sensible bounded steps. For P4, validate pooled `out=` operations against the actual supported backend and immutable-input contract; an old “never mutate grader arrays” sentence must not be treated as blanket permission to overwrite inputs. New scratch allocation reuse and input mutation are different operations. Retain new warning, memory and lifecycle checks for the executed path.
4. Run the N1 discriminator with both closure-sensitive accuracy and full economic accounting. Keep P5 as a measured fallback, not evidence that a winning score has already been identified. A lower-rank residual setting is also a cheap bounded secondary ablation, supported only by the public rank ladder; its effect on our chain is unknown.
5. Only after the main recurrence is correct, consider consumed-correction λ fitting or modest κ4-memory extensions. Their interaction with compression and required fresh evaluation must be explicit. Do not reopen learned readout correction, source deletion, generic QMC or Gaussian-only closure absent new external evidence addressing the recorded failure mechanism.

The architecture itself needs only a small structural cleanup: one candidate-role manifest, one authoritative active plan pointer, one generator/export path, one nonlinear reference oracle, and separate representation/arithmetic backend interfaces. Keep archived estimators frozen. `docs/GAME_PLAN.md` still describes September-12 F21/B4 work while CURRENT_STATE names the September-16 handoff operative; the inherited AGENTS “sole active GAME_PLAN” pointer is stale. Old docs should be marked superseded through authorized routine documentation, not silently interpreted as parallel current obligations.

For iteration speed, reuse existing control predictions, truth artifacts, scoped instrument certificates and invariant results when the inputs/path are unchanged. Run cheap source/shape/one-network diagnostics before a full panel. Preserve failed outputs and source hashes. The Sep-16 4h28m mini loss arose from the wrong wrapper/cap invocation, not a statistical necessity; the repo now has `run_mini_json.ps1 -NoResidualLimit` and one-network `p2_repro_mlp.py` to prevent repeating it. Further validation-system restructuring belongs to the separate infrastructure audit; avoid making it a dependency of the feed experiment unless a concrete defect blocks that result.

## Evidence limits

P1/P2 DESIGN results in `experiments/2026-09-14_slice6a_p1/score_v2.json` and `experiments/2026-09-15_slice6b_p2/score.json` used a repeatedly exposed 24-network P01 subset. Their quoted raw MSE must retain the artifact's truth-noise context; independent replicas support paired/cross estimates but do not make noisy ground truth exact. P11's reported truth noise is `4.76e-9`, a material fraction of the remaining MSE. Public truth floors are unknown.

The small-shape κ4 ablation in `experiments/2026-09-16_exp_q31/README.md` uses forced-open target branches and tuned λ at 64×8/128×8. Its feed contribution differs greatly from the public 1024×16 report. It motivates D2/D3 but cannot establish a target-shape gain or prove which term dominates there.

The v31 forecast of `8.713e-9` versus graded `8.752e-9` is excellent evidence for this paired projection, not a universal half-percent error bound for a different architecture. The latest CURRENT_STATE retracts the old wall factor and records the subprocess/server FLOP-path difference; new architecture claims must use those corrections.

No source inspected here establishes that first place is guaranteed, that every leader uses K3, or that N1 is the only possible route. The concrete next opportunity is narrower and more useful: determine whether the missing all-distinct κ4 birth explains a significant part of our remaining accuracy deficit, while preserving the successful P2 cost improvement.
