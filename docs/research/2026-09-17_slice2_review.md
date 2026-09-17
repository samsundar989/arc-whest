# Independent review record — September 17 next-slice decision

Reviewed worker HEAD `38b240d`. Three parallel bounded reviewers covered architecture, validation and compression/literature. A fourth, fresh-context Skeptic received raw artifact/source locations and questions to falsify, without the other reviewers' reasoning. This file records their findings and final clarification. No reviewer launched an estimator run, heavy workload, truth read or seed-consuming experiment. The validation reviewer ran the lightweight wrapper test suite: 10/10 passed.

## Inputs

- `experiments/2026-09-17_followupF/score.json`, `rows.jsonl`, `decide.json`.
- `experiments/2026-09-17_followup/meter.json`.
- `experiments/2026-09-17_closure/same_state_1024.json`.
- `experiments/2026-09-17_skeptic/verdict.md` and contracts in `experiments/LOG.md`.
- `experiments/2026-09-16_stage1_k4feed/score.json` for the paired v31 control.
- `local_sim/closure_audit/{canon.py,same_state.py,ref_step.py}`, generated closure JSONs, `local_sim/s6_build_closure.py`, `local_sim/s7_build_shared.py`, `local_sim/s8_decide.py`, `local_sim/p2_design_score.py`.
- Frozen own v31/v32/closure/shared candidates; pinned `external/504aldo/estimators/estimator_v17.py` and `estimator_v29.py`.
- Job1642 receipt and contract timing; wrapper source/tests.

## Architecture reviewer

Canonical reference ordinary-mode program: 87 oriented terms, versus Q3q57, PAIR114 and B4 141 after excluding redundant `(1,2)` outputs. Q3q misses33/adds3; PAIR misses6/adds33; B4 misses6/adds60. The three Q3q additions are C-cubed rows. These are different programs, despite matching shared coefficients.

The final same-state prediction displacement is material to task error. At layer14 the transported dG difference is16.631%, and swapping the nonlinear program on a fixed native boundary changes the core8.357%. The core transport/projection identities already agree (`estimator_k3_v32.py:1047,1266`, reference v17`:448,679`). Replacing identical formulas is not a new causal intervention.

The report's O(n-squared) boundary cannot resume the full history. Keep native complete towers and swap operators. v29 additionally has feedback (`:1268,1564`), adaptive lambda (`:1097`) and nested shared-source settings (`:664`); the v17 report cannot attribute v29's extra gain.

Recommendation: four-chain reciprocal full-program swap, native v29 endpoint, at most one conditional interaction experiment, saved prediction vectors. Exact error-change decomposition identifies whether a proposed correction opposes existing error.

## Validation reviewer

All six1642 hashes match, all24 common-network rows complete. Contract3f72a99 at14:34:39Z precedes job start14:34:48Z. Queue snapshot execution is correct;6.81GiB RAM was below10GiB declared. Linked truth/environment identity remains worth recording explicitly.

v31 P11 score1.037175704e-8 vs candidate8.746249903e-9 gives1.185851897x. Naive graded-anchor projection7.38037e-9; noise-aware ratio projection7.45850e-9. Both are arithmetic transfers, not validated official forecasts. Same-panel2x threshold is5.18588e-9; atq.2163113 raw must reach2.39741e-8, another1.68655x gain.

Rank256 official-precision score8.2483446e-9 vs rank3848.2441907e-9. Observed score384-minus-score256 is+9.6919e-11, paired t95[-9.6227e-11,+2.9007e-10]. Label-noise correction already in score.json reverses the tiny point ordering. Both are effectively tied; the registered observed winner remains correctly selected under that historical rule. Its1.14927x observed gain versus v32 becomes1.13428x at estimated official precision. Mean averaged-label noise2.3817169e-9; differential noise accounts for15.69% of the observed score improvement.

The multiplier floor allows a hypothetical4.04336e-9 at unchanged rank256 P11 raw, so mathematical cost-only impossibility is false. Neither feasibility nor a winning algorithm follows from that bound. New candidate deployment evidence remains absent; operation count and Windows commit do not prove residual or Linux memory safety.

## Compression/literature reviewer

The join metric is diagonal-only in consumption coordinates, while actual frozen-query error includes all cross-column and A/M terms. Old importance weights also age, and join-time Wick scaling makes the immediate weights stale. This does not discredit measured outputs, but leaves a specifically different proposal untested.

A query-aligned range finder may reuse the old-plus-joining dense contraction with O(n-squared times rank) sketch work. It must not conceal another wide product. Projecting legs changes left operands too: frozen-query optimality is not whole-step optimality. Measure actual recomputed D3/D21 and downstream effects.

Goal-oriented tensor objectives and output-norm-dependent dimension reduction support this idea as a hypothesis; they do not prove performance here. Sources are linked in the strategist response. Existing q≈.165 machinery is an affordability precedent with failed accuracy, not a claimed viable candidate.

## Fresh Skeptic — verdict and scope

**REJECT the broad claims** that the closure family is closed, that current weighting minimizes query error, that rank256 is clearly best at official precision, or that cost-only major gain is mathematically excluded. **Accept the narrower recorded DESIGN results and registered observed-score selection.**

Independent official-precision difference (384 minus256): -4.15394e-12 with t95[-1.97286e-10,+1.88978e-10],11/24 networks favor256. Correction removes expected noise bias, not realized label-error cross terms or adaptive-selection bias.

The Skeptic's first term-count report included15 redundant `(1,2)` output rows and therefore said Q3q had18 additions. Follow-up explicitly filtered that orientation and confirmed **57 terms,33 missing,3 additions**. This convention correction is preserved here rather than silently dropped.

Strongest cheap discriminator: equal-rank current-query comparison and cost on a label-free live boundary; then fresh DESIGN reference reverse ablation. These can be prepared independently and queued without extra truth minting. No global cost lower bound exists in the supplied artifacts.

The Skeptic reviewed the new handoff as a prospective plan and **ACCEPTED it with one required clarification**: each SWAPPED one-step live program must match its independent reference on the same native boundary in initial, ordinary and final modes. Native endpoint parity can pass while the transplant is wrong. Check pK/covariance/core/birth inputs too. This clarification has been incorporated in the handoff.

Numerical attribution thresholds, resource requests and conditional branch rules still require an actual committed pre-data contract. This review is not a contract, promotion or release certificate.
