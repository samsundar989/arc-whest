# Independent validation and evidence audit — September 12, 2026

## Skeptic verdict: PROVISIONAL

The current evidence supports rejection of the frozen K4 build's promotion, and supports the scoped instrument and memory-inventory acceptances. It supplies no newly competitive submission. The missing next scientific discriminator is agreement with the working full K3/L2 baseline before compression or K4 changes. It is **not** the first evaluation of the compact K4-off arm: that evaluation already exists.

This review read CURRENT_STATE, the LOG tail, AGENTS, VALIDATION and SKEPTIC, then inspected implementation and raw JSON. It recalculated report arithmetic and hashes without opening protected truth archives, minting networks, allocating a seed block, running a candidate or changing an instrument. The report itself is a review artifact. The driver relayed Samuel's new authorization for eligibility review and candidate preparation; the older hold in the standing documents is therefore superseded for this task. Submission still needs individual approval; no external messages or designation are authorized.

## What is established, and what is not

| Evidence | Independently checked here | Valid scope |
|---|---|---|
| `experiments/2026-09-11_wp0e_final_acceptance/mutation_aggregate_linux.json` | PASS, accepted=true, 13 cases, exactly B and M1–M12 covered, no errors; SHA256 `f5e9d11291257bd26029b45db9fd771530acc5e3026ea73eab23665d0c503591` | Historical Stage-0 mutation/provenance coverage. Its positive certification intentionally returns incomplete with named skips. Acceptance of the mutation battery is not a complete candidate certificate. The aggregate explicitly says receipts do not authenticate execution; raw queue logs are the execution authority. |
| `experiments/2026-09-11_stage2_compact_p07/p01_report.json` | SHA256 `a197516ccdd008c8f0083c6c5c781dd7718c5401f2e7af5758e3be29f207f963`; complete; 96 unique rows per arm; every per-row score exactly equals raw MSE times `max(0.1,F/2**41)`; recomputed means match objectives to rounding | Exposed P01 DESIGN, accuracy-only (`residual_cap_override=0`), control exact. A real negative build comparison, not fresh DECIDE, causal rejection of all K4, or deployment evidence. Depth curves use eight networks, not all 96. |
| `integrated_first_source/tiny_result.json` under the same directory | Raw SHA256 `623ec4524c6e526be8037d4a09043cd81314acf97edcbe1eafbc36a087817e28`; 172 checks, 79 fail, hence 93 pass; receipt agrees | Frozen n4 integrated one-source numeric/algebra gate is red. No full-width cost follows. |
| `integrated_first_source/n8_result.json` | Raw SHA256 `15f3b5c5f80942a03219c4babf8812aa5f0440b5bfca878e469e8a9a0dbe2b88`; 186/187 pass; sole failed check is `all_preactivation_variances_positive_and_usable`; receipt agrees | A distinct frozen n8 fixture, not a repair of the n4 gate. |
| `integrated_first_source/n8_variance_diagnostic.json` | L2 covariance minimum eigenvalue -0.23117316; reconstructed next-layer variances include -0.117716205 and -0.057593874. Removing K4 from that fixed decomposition leaves positive values; Gaussian plus ordinary K3 also remains positive there | Localizes this fixed-state failure to the K4 covariance contribution. It is not a rerun of a recursively K4-disabled estimator and cannot prove that such an ablation is stable at every layer. |
| `integrated_first_source/memory_ledger_thin_result.json` | SHA256 `dd3bf40edaedeed3551529d19553d4501a56bbc0d8939164676eaf113eb04697`; 93/93, no failed checks; modeled width1024 peak 715,844,864 bytes versus 910,988,032 padded; `process_cap_gate=NOT_TESTED` | Nine sampled named boundaries at n8 and n16; accepted thin-shape and alias coverage. Existing fresh Skeptic additionally replayed all 172 retained L14 field occurrences. Not a process-memory bound or an 8 GiB test. |

The Stage-0 aggregate's bytes and the later accepted preflight records should remain historical scoped evidence; there is no reason to rerun the entire mutation campaign merely because a new candidate will need its own certificate. Default preflight validates the shipped file and instrument and reports 12/16 with four Full-only skips. Neither its roughly 3 GiB peaks nor Stage-0's positive mutant supplies a future candidate's process-memory or residual measurement.

## The no-K4 control already exists and was measured

`local_sim/p2_build_k4_arms.py:18` defines `exact` as consumers=false and births=false. The generated master switch is false as well. `local_sim/p2_compact_reference.py:513` separately supports `consumers=False, births=False`. The generated candidate therefore has a real K4-off route. Setting lambda to zero on a coherent arm is a different operation and leaves K4 active.

`local_sim/test_p2_compact_candidates.py:242` compares a generated exact arm against coherent bytes with all three switches disabled, including bitwise output and equal FLOPs/operation counts. Its primary oracle is the compact reference, including thin-rank state checks. This verifies implementation of the same compact method; it does not establish equivalence to `estimator_k3_strassen_L2.py` or the full K3 recurrence.

The frozen exact file named in the P01 report has SHA256 `f704f83b2063894c8dcd13120a8baada762229563f6d08bccee39d3aad2e6efc`. Its existing results are:

| quantity | value |
|---|---:|
| raw mean final MSE | 2.111300860481909e-5 |
| measured truth noise floor of averaged replicas | 9.319142001565295e-9 |
| counted FLOPs | 816,292,940,328 |
| mean arithmetic score | 7.837297686454867e-6 |
| lifecycle failures with relaxed residual cap | 0/96 |

The other three mean scores recompute as coherent 1.7583805654537924e11, consumers-only 1.2696697527360155e17, and births-only 6.728865985219634e-4. I found no trivial score/multiplier arithmetic error explaining the compact control's level. Existing `skeptic_verdict_p01_mechanism.md` independently audited the paired outcomes, source bindings, noise and registered t gate. Its warning still matters: the template replaces nonfinite means/output with zero (`p2_compact_candidate_template.py:300` and following), so zero lifecycle failures cannot certify finite internal states.

The exact arm is a compact K3 approximation with rank384 shared history and residual16, not an exact full-distribution truth oracle. No located artifact establishes its output parity with the working full K3/L2 estimator. The P01 comparison contains no standing L2 arm. Quoting an exact numerical performance ratio between P01 and a different public submission would violate paired/tier comparability. Existing scale evidence is sufficient to prioritize diagnosis, not to manufacture that ratio.

The later `integrated_metered_program` in `local_sim/p2_k4_integrated_first_source.py:1546` is another program. Its signature has no consumers/births switches and it unconditionally uses the supplied discrete bank and feeds. Its red n4/n8 gates do not erase the old exact-arm measurement. Conversely, the old exact arm does not certify the integrated program with K4 disabled.

## Actual critical gates

1. **Baseline differential gate.** Independently compare the uncompressed, K4-off recurrence with the working full K3/L2 path on the same tiny deterministic weights. Compare each consumed D3/D21 slice, mean, covariance, birth and transport before output sanitization. Cover more than a final-output equality and identify the first divergence. An exact reassociation should meet a tolerance based on dtype and reduction order. If the two programs intentionally implement different closures, document that difference rather than demanding impossible equality.
2. **One approximation at a time.** Only after identifying the full-recurrence relation, compare the same full source representation with its compact representation. Tiny full-rank parity is appropriate for algebra. Thin-rank discrepancies are approximation diagnostics; requiring thin rank to reproduce every full tensor exactly would rule out the method by definition. A specified approximation is ultimately judged on stable execution and paired score, not a universal exact-distribution theorem.
3. **Candidate-specific stability and cost.** A candidate must avoid hidden nonfinite state, invalid variances and unsafe covariance feedback on relevant fixtures, and run its complete width1024/depth16 path with all source construction, joins, backbone and readout inside the meter. Use workerq with a recorded optimization review and conservative resource request. The current red integrated fixtures block claiming that particular path is ready; they do not require proving a causal K4 source before studying or preparing an independent K3-only candidate.
4. **Evidence to promote.** After a frozen, stable design has plausible cost/accuracy, use the registered canonical lifecycle, counted Phase-2 cost, truth floor, common random numbers and the paired arithmetic gate through `competition_score.decide`. A fresh DECIDE is required for independent promotion; reusable exposed DESIGN is adequate for diagnosis. Public confirmation stays read-only, and the Phase-1 vault is not a Phase-2 validation shortcut.
5. **Submission preparation and deployment.** Review provenance/eligibility against authoritative rules under the newly lifted hold. Bind exact candidate bytes, candidate-specific official subprocess behavior, residual headroom and Linux 8 GiB enforcement; complete required Full checks and clean mini before asking to submit. These are final candidate gates, not prerequisites for every tiny research diagnostic. Do not treat a prediction of server residual as a server measurement.

## Avoid further low-value gates

Do not repeat another symbolic memory-lifetime extension absent a concrete missed binding. The last two independent reviews closed the named tiny boundary defects; their remaining caveat is correctly stated, and only a complete stable process run resolves process memory. Do not rebuild the source-locking/publication framework or replay all Stage-0 mutants without a new affected path or failure. Do not demand exact reconstruction of every four-index cumulant as a prerequisite for an explicitly approximate estimator. Do demand that claims of exactness, causal nonlinear source identities, no double counting and fixed gates withstand their specific counterexamples.

## Classification and cheapest discriminating follow-up

A report-only audit or reading existing score artifacts is ROUTINE. A new truth-free differential probe that changes the mathematical instrument is IMPORTANT: written prediction before implementation, LOG knowledge delta, appropriate default preflight and fresh independent Skeptic. If it merely exercises existing read-only entry points, it is review/verification. A frozen no-K4/full-K3 differential diagnosis does not itself spend a seed block or reopen a closed research lane.

A decision to change the target architecture, reopen a genuinely closed lane, alter a gate or measured cost/residual model, spend DECIDE, change `estimator.py`, or make a submission/write-up claim is STRATEGIC. That needs the committed contract and Samuel's prediction before its new evidence. Label-free does not automatically mean routine; equally, an IMPORTANT instrument check should not be inflated into a request for more protected labels or blanket architecture proof.

**Strongest challenge:** the compact method may faithfully implement a materially different or incomplete K3 recurrence, so K4 repairs and cheaper memory can optimize an already inaccurate baseline.

**Evidence for it:** existing P01 exact-arm MSE and the absence of a located full-K3/L2 differential oracle, together with compact-self-reference tests and explicit rank approximations.

**Evidence for the original work:** accepted source-bound P01 scores, supplied-core and compact-self-reference algebra, genuine cost reductions, and repaired scoped Stage-0/memory coverage are useful and need not be re-earned wholesale.

**Unresolved ambiguity:** whether the compact control's loss originates before rank compression, in the thin history/residual representation, or in another intentionally omitted full-K3 term. Existing results do not isolate these.

**Cheapest discriminating follow-up:** one truth-free tiny full-K3-versus-uncompressed-K4-off layer-state differential, using deterministic synthetic weights and no seed block. Preserve the standing candidate and frozen files. This can identify a recurrence difference before any full-width experiment or new labels.

**Checklist items triggered:** 1, 4, 6–8, 10–14, 17–18. Arithmetic, row counts, relevant raw hashes and JSON scopes were checked directly; no candidate execution, numerical replay of old fixtures, protected read or new independent accuracy claim was performed in this audit.

## Addendum: independent review of the standing D21 fusion proposal

**Scoped verdict: ACCEPT as a distinct algebraic proposal, not as a tested candidate.** After the first audit, the architecture reviewer identified a smaller direct optimization in the standing estimator. This supersedes my recommendation to start with a compact/full-K3 bridge: attempt this isolated standing-program contraction first, then use the bridge as a bounded fallback if needed. I independently read `docs/research/2026-09-12_architecture_review.md`, the standing source and the old packing contract/raw artifact.

In `estimator_k3_strassen_L2.py:656–669`, the first and fourth terms have the same right operand `M.T`. For arbitrary real matrices of the required shapes,

```
2 ((A_D*M) @ M.T) + 3 ((A_L*C_L) @ M.T)
  = (2*(A_D*M) + 3*(A_L*C_L)) @ M.T.
```

The other products `(M*M) @ A_D.T` and `6*(A_L*M) @ C_L.T`, the division by three and the off-diagonal mask remain. This follows by matrix distributivity and needs no symmetry, transpose-orientation change, diagonal assumption or new source representation. It is exact over the reals. Moving scaling/addition before a float32 Strassen product changes rounding and intermediate range; the displayed identity does not prove bitwise or trajectory equivalence.

The frozen best L4 source is available in `submission-k3strassen-L4.tar.gz`, member `estimator.py`; there is **no** current root file named `estimator_k3_strassen_L4.py`. I read that small archive in memory without executing or extracting its code. Member SHA256 is `052dd768e970c4bae831cff26f8ff15b0487f3b95eadf2fd3578e542aec34d58`, 52,380 bytes, `_STRASSEN_DEPTH=4`. Independent AST comparisons against frozen L2 are identical for `_dslice21`, `_layer`, `_advance`, `_dslice3`, `_mm` and `_fit_generations`. The same fusion therefore applies to the best L4 as well as robust L2. A new candidate should be made from the verified appropriate frozen bytes, preserving both archives and the standing L2 file.

The existing `_layer` contains scalar `c4`, `D4`, `D22`, the Wick table, raw-to-centered moment calculations and all transported K3 history. Replacing only this D21 contraction preserves that recurrence algebra. Calling this a new “K3-only” or “no-K4” candidate would be misleading: it keeps the standing scalar K4. It needs none of the unfinished physical signed-source K4 work or compact Source bridge.

**Distinct from the closed lane.** The LOG's September 10 symmetric-packed contract and `experiments/2026-09-10_slice5/packed_feasibility.json` tested object identity/symmetry discounts on existing products, plus covariance congruence via Cholesky. Its raw inventory separately lists all four D21 products, including the common-right-operand sites 657 and 668. It does not combine their left operands. Its 1.006688966 modeled best gain and 1.5 gate reject that unchanged-product symmetry intervention. They do not measure or reject deletion of one product by cross-term distributivity. Proceeding with the latter under its own new contract does not require moving the old gate or declaring the packing lane reopened.

**Cost scope.** At depth16 the nonfinal nonempty history readouts occur at layers 1 through 14. One wide product per call disappears: **14 actual `_mm` calls**, whose inner widths sum to **105 lineage-equivalent products**. The latter is about 13.2% of the historical 797 dominant dense equivalents, not 105 wrapper calls and not an official 13.2% total/score saving. The fused left operand has width `k*n`; scaling and addition now operate on that wide array, whereas some old operations scaled `n*n` outputs. The exact backend bill, blocked contraction corrections, elementwise work, process peak and residual must therefore come from complete matched candidate runs. L4's deeper Strassen recursion can have different numerical and operational consequences from L2.

**Recommended controls and gates.** First compare unfused/fused on identical synthetic tensors and trajectories, separately at fixed L2 and fixed L4 settings; preserve dtype, all scalar K4/full-pK terms, off-shape behavior and the same generation schedule. Keep the existing conservative budget/generation chooser for the first isolated implementation: lowering its estimate could retain extra generations on off-shapes and turn one contraction experiment into two interventions. A later budget-model improvement can be separately scoped if necessary. A complete official cost screen should compare fused L4 to original L4 and fused L2 to original L2. The proposed 8% screen and 1.05 promotion gate are proposals requiring preregistration, not outcomes or existing binding thresholds.

For fresh accuracy decisions, include the robust original L2 **and the best original L4 on the same fresh networks**. The primary incremental comparison for a fused L4 candidate is original L4; original L2 provides the robustness/fallback comparison. Predeclare selection or multiplicity handling if both fused L2 and fused L4 can win. Using compact exact as the sole baseline would answer the wrong submission question. Retain paired arithmetic scores, the noise floor and measured complete counted cost; fixed-shape algebraic similarity does not authorize skipping these or official deployment checks. Candidate-specific Linux memory and subprocess residual/wall evidence remain necessary because the old L4 has little recorded wall headroom.

No candidate code, math instrument, seed ledger or protected data was changed or executed for this addendum. The only additional execution was standard-library tar reading, hashing and AST comparison. Algebraic source review is a proposed route; numerical tolerance, cost, stability and improved submission score are still unmeasured.

## Final execution-document review

**Skeptic verdict: ACCEPT for the revised documentation and proposed execution ordering.** Reviewed September 12: `docs/STRATEGIST_RESPONSE_2026-09-12.md`, `docs/CURRENT_STATE.md`, `docs/GAME_PLAN.md`, `docs/CODEX_EXECUTION_HANDOFF.md`, and `docs/ELIGIBILITY_REVIEW_2026-09-12.md`. This is a document consistency verdict, not a candidate acceptance, approval of proposed gate values, new measurement or legal eligibility certification.

The documents now put the exact standing-program fusion first and the compact/full-recurrence bridge second. The active plan and handoff explicitly preserve the conservative generation chooser and existing scalar K4/full Wick recurrence. They name original L4 as the primary comparator for best-score improvement, include original L2 for robustness on the same fresh networks, and require selection/multiplicity to be resolved before DECIDE. Fourteen wide calls and 105 lineage equivalents are distinguished; roughly 13% remains an unmeasured dominant-work estimate. The 8% engineering screen and 1.05 score gate remain proposals for a new committed contract with Samuel's predictions. No claim of a new measured candidate, guaranteed improvement or winning architecture was found.

Accepted Stage-0 and memory evidence keeps its scope; the red n4/n8 gates and exposed P01 outcome remain visible with archive/LOG pointers. Public data remains CONFIRM, P01/P07 DESIGN, P08 exhausted and the Phase-1 vault protected. The revised scope authorizes eligibility review/preparation while leaving actual submission individually approved and the organizer question unsent. The eligibility note preserves the unresolved rule interpretation rather than asserting that public licensing or previous grading proves eligibility. I did not independently repeat its external rules-page verification in this document-only review.

No material correction is required before committing the revised plan. Two small clarity suggestions were sent to the driver: make the strategist response itself state the 14 actual calls next to 105 equivalents and point to the already verified L4 archive; distinguish setup telemetry, which the canonical lifecycle records separately/off-budget, from the complete per-MLP counted bill when listing metering coverage. These do not change the proposed decision or gates.

The read-only workspace `git diff --check` completed cleanly during this final review. No new experiment, candidate execution, protected-data access or instrument edit occurred.
