# Strategist response — September 18: measure and optimize the complete candidate

## Decision

The worker's no-release decision is correct. The repaired scoring, negative corrected-storage result, bounded own fit, archive repair and preservation of P13 are supported by the reviewed evidence. **The next slice is a complete compressed recurrence experiment: integrate the conditional P3 feed, test the frozen own fit on that same implementation, then pursue one explicitly priced query-cost reduction and, only if warranted, one bounded fit of the complete package.** Do not start another generic architecture search or repeat the completed qEc experiment.

The earlier handoff was too indirect: it let us optimize accuracy without the intended storage, optimize storage without the strongest accuracy feed, and stop before measuring their interaction. This response supplies the new strategic direction requested by Samuel. It supersedes the previous instruction to wait for a strategist; it does not retroactively change any completed experiment's verdict or the 2x release gate. The builder still commits the actual new contract and records Samuel's predictions before new strategic measurements.

Execution: [Claude handoff](CLAUDE_CODE_HANDOFF_2026-09-18.md). Independent evidence: [worker review](research/2026-09-18_worker_evidence_review.md), [architecture review](research/2026-09-18_next_architecture_review.md), [frontier follow-up](research/2026-09-18_frontier_followup.md). This is a research and planning review; no estimator runs, new truth-array reads, seed claims, promotions or submissions occurred here.

## What was accomplished, and what was not

| Finding | Assessment |
|---|---|
| Corrected qEc loses at age4/rank256 | Supported: official-precision score 7.0866e-9 to 7.5279e-9, paired difference -0.4412e-9, t95 [-0.5743,-0.3082]e-9; raw MSE 3.3860e-8 to 3.5832e-8, averaged truth noise 2.2481e-9. Close that tested configuration. |
| The own fit gains 1.144855x | Supported against its own default on the 12 within-fit held-out DESIGN networks; these networks were already exposed elsewhere. It is not a release result. |
| P3 gains 1.207x on PAIR+Q | The adjusted comparison also changes P4/Strassen. Raw improvement is substantial, but this adjusted number is a joint P3-plus-P4 effect. The separate 1.171818x comparison against v32+P4 is a closure-plus-P3 effect with P4 matched. |
| The strongest full-panel artifact is v42 PAIR+Q+P3 | Supported: uncompressed, 6.7988e-9 official-precision adjusted; raw 2.4149e-8 against 2.2481e-9 truth noise. Its compressed composition is absent. |
| Calibration is exhausted | Not established. Theta and lambda both stopped at the furthest points reachable in two scheduled passes, not at their allowed bounds or a bracketed optimum. Last improvements were small, so this does not imply another large gain. |
| We know the exact remaining P12 gap against v31 | False: no same-panel P12 v31 arm was saved. Add that anchor in the next registered batch. Public and P11 thresholds must not substitute for it. |

Source hashes and all 19 fit source/row hashes reproduce. Standard preflight passed its 12 enabled checks; that is neither 16/16 Full preflight nor finalist-specific certification. Disabled DESIGN time caps are legitimate accuracy instrumentation, but zero failures there does not prove deployment safety.

The fitted source still uses depth4/min64 and lacks P4. On the same last-12 P12 rows its official score 7.2623e-9 is only 1.09064x better than v32+P4 and is worse than the P3+P4 bundle's 6.7657e-9. These are retrospective DESIGN comparisons, not new gates. Normalize implementation costs before drawing conclusions about the physical model.

Saved final predictions allow scalar error geometry without reopening truth: <p-y,q-y> = (MSE(p)+MSE(q)-mean((p-q)^2))/2. The P3 and compressed bundles' displacements from PAIR+Q have pooled cosine about 0.016, but their remaining errors have estimated noise-corrected cosine about 0.802. Simple averaging still costs both estimators: its reconstructed official-adjusted score is about 1.280e-8 at summed cost 0.533B. **Do not ensemble the full files.** These retrospective diagnostics cannot predict nonlinear composition or identify a causal layer. Fitted predictions were not saved, so that geometry is unavailable for the fit.

## Competition and architecture assessment

Chrome refreshed September 18: our row is rank **63/171**, 15 Phase-2 entries, displayed 8.80e-9 / raw 2.91e-8; latest graded remains v31. The board displays the leader at about **2.20e-9**, raw 1.65e-8, and still links #331393. Its rounded leaderboard display must not overwrite the prior precise clean-detail measurement 2.239e-9. The previously observed newer submission has a timeout/inconsistent summary; no new failure semantics are inferred. Top ten remains around 3.60e-9; 4.40e-9 is around rank13. [Live leaderboard](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/leaderboards).

The discussion listing still exposes the same leading technical topics; a fresh primary-source search found no newly released winning code. Neither the leader's FLOPs nor similar raw MSE establishes its architecture. The public source repos remain at the previously reviewed commits. [Public technical discussion](https://discourse.aicrowd.com/t/everything-we-tried-a-factorized-k-3-cumulant-propagation-estimator-at-0-25-x-b-where-its-flops-go-and-25-measured-dead-ends-team-504aldo-rank-10/18218/3).

Keep full mean/covariance, persistent factored K3 history, the regenerated approximate K4 core, coherent compensated birth/feed, and legal single-file export. This family is credible; its current operating point is not demonstrated state of the art. Preserve represented history when changing its readout cost. Approximating dense covariance is low priority: even its ordinary unoptimized sixteen-layer sandwich is only about 0.031B, and our current exact implementation costs less.

The central algorithmic lesson is **immediate query energy does not control future recurrence error**. For symmetric tensor error T and trace vector tau_i=sum_j T_ijj, independent Gaussian next rows u,v with coordinate variance sigma² give E[T(u,u,u)²]=sigma^6(6||T||²+9||tau||²) and E[T(u,u,v)²]=sigma^6(2||T||²+||tau||²). These derived identities concern a random next layer, not the known fixed weights or eventual truth error; they explain why current repeated slices can miss consequential tensor directions. Exact formulas and limitations are in the architecture review.

Output-sensitive model reduction has primary support, but does not hand us an affordable challenge algorithm. In particular low-energy directions can affect future outputs in nonlinear dynamics. [Otto, Padovan and Rowley](https://arxiv.org/abs/2207.14387). The direct ARC paper remains the appropriate fixed-weight foundation, not a finite-ReLU optimality guarantee. [Wu et al.](https://arxiv.org/html/2605.05179v2).

Do not adopt another method just because its name sounds more advanced. Full tensor-residual probes can add roughly 0.073B at sixteen probes in a straightforward implementation. The measured merged Tucker N1 joins, not its readout, dominate its cost. Generic CUR has cheap rows but expensive columns here. A shared ReLU mean proves nothing about low-rank centered cumulants. Exact radial identities overlap existing harmonic structure and require angular-cumulant cancellation. These alternatives remain conditional research, not the next implementation.

## The next slice, in order

### A. Localized integration and matched implementation

Compose through maintained generators; leave all frozen archives unchanged. Introduce only the interfaces needed to make P3 coexist with shared storage and own fit. P3's F/R3/R4/w3b fields must survive advance, Wick scaling, join, append, truncation and fallback. Obtain all-age A/M from the shared reconstruction interface. Include P3's LA/LP contributions before any join-statistic capture; price TT F^T separately. Use the same compressed D21 factors in birth and repeated-index compensation. Do not project away P3 factors as an undocumented saving. Preserve s_feed's existing meaning: it scales the hub cA/cB feed, not P3 as well.

Keep P3 rank32, age4/rank256, the existing **gram join** and P4 depth5/min32 fixed. Use `estimator_k3_v41_pairq_a4r256_gram_p4d5.py` as C00. This choice is prospective and deliberate: the composed old-qE improvement over gram is only1.01953x with its paired interval crossing zero, whereas gram avoids the exact old-query dependency described below and costs less. Old-qE remains an existing diagnostic comparator, not a certified post-Wick objective. Do not relabel the failed qEc or historical incomplete S3 gates. Uncompressed, storage-disabled and forced-full-rank versions are correctness references.

Mandatory focused tests: disabled-feature identity; full-rank storage identity; nonsymmetric D21/nonuniform gates; multiple joins with distinct birth coefficients; tiny full represented K3 and D3/D21 equality; birth compensation; fallback/truncation; full recurrent output and cost-transform reassociation tolerance. The fit controls must change the same sites as their frozen source, including lambda depth indexing and compensation. Avoid a framework rewrite. Review the instrument/source diff independently before measurement.

### B. One complete-package factorial, anchored to v31

Preregister four complete compressed arms, all with identical P4/storage settings:

| Arm | Global controls | P3 |
|---|---|---|
| C00 | default (1,1,0,1) | off |
| C10 | frozen fit (0.625,1.225,-0.1,1) | off |
| C01 | default | rank32, compensated |
| C11 | frozen fit | rank32, compensated |

Include graded v31 as a same-panel anchor. Add the fitted uncompressed source with P4 as one matched-cost sentinel; the existing uncompressed P3+P4 artifact is a diagnostic reference if its cache identity verifies. Do not run an extra dense factorial to select a recurrence before testing compression. The scientific question is whether accuracy controls/feed survive and interact constructively **in the actual compressed package**.

Use the existing P12 DESIGN panel with an explicit new registered reuse/receipt, not P11 and not P13. All P12 selection remains DESIGN; neither its last12 nor a re-split is an untouched project holdout. Reuse exact C00/other valid completed outputs only where the full identity contract permits it. Otherwise rerun the specific unmatched control; never rename source identities to fit a cache. Save final predictions for every new arm, including fitted arms.

Before labels, meter complete exports, including source birth, reconstruction, joins, QR, projection, thin feeds, final pruning and fallbacks. Whole-file cost/resource infeasibility is a valid early stop. Keep one MLP/arm in memory at a time. The expected P3 increment is about 59B FLOPs before integration changes; its apparent net 1.55% cost against v40 concealed P4 savings.

Prospective selection: compute all six paired arm comparisons, observed and official precision, through decide(gate=1.0, interval='t'). The four simple effects are C10:C00, C01:C00, C11:C10 and C11:C01; C11:C00 is the direct combined-package check and C01:C10 resolves the alternative single interventions. Report the factorial interaction without pretending these are independent confirmations. These are DESIGN screens, not release gates. Freeze this deterministic selection rule: start incumbent C00, visit C10 then C01 then C11; replace only when the candidate's official mean is lower, its direct official-precision gate passes against BOTH C00 and the current incumbent, and it has no new failed identities against either. Exact ties or unsupported improvements retain the incumbent. Observed comparisons remain visible and the eventual release requires both precision conventions. Record every comparison and uncertainty. The existing release test remains gate2.0 against v31 on fresh P13, supporting both observed and official precision.

### C. One query-only cost discriminator, on the gram-join package

After B is synthesized, test **one fixed rank32, two-pass approximation of the old D21 aggregate**, conditional on a real cost saving. Keep the storage representation, gram-join algorithm, D3 consumer and all thin-term algorithms unchanged. Ordinary nonlinear feedback can change later values; this is not a claim that complete trajectories stay identical. No age/rank grid; no storing D21 in place of the tensor.

**A source dependency rules out the tempting old-qE spelling:** its join reuses the exact Xold=L F^T product. Recomputing Xold would erase the main savings; substituting a low-rank Xold would change both readout and basis selection. The gram join only needs its original LA/LP column-energy statistics, which remain available before the approximate contraction. Preserve these exact statistics. The current gram wrapper still computes/stashes Xold for its exact query; replace that contraction in the new approximate branch and remove its unused stash, rather than running both spellings. Do not quietly switch to qE during this branch or omit its join bill. If inspection of the integrated source finds another consumer requiring exact Xold, reject or rederive the proposal before implementation.

For the existing old contribution E=L F^T Q^T, L=[LA,LP], F=[FA,FM], compute E times a sketch through those factors, QR its range to U, compute U^T E through the same factors, and return U(U^T E). Preserve factors and birth coefficients; Q is generally nonorthogonal after transport. Use ordinary Euclidean operations or an exact QR/rebase with its triangular factor, never assume Q^T Q=I. Freeze a legal deterministic/sketch rule and its rank-deficient fallback before measurement; test determinism and the actual permutation guarantee. Rank-at-least-r recovery requires range coverage, not just a sufficiently wide sketch: use an explicit exact fallback when q>=r and test deficient/zero sketch cases separately. Count all fallback paths and do not assume a seed-fixed random sketch is pathwise permutation-equivariant.

With m=old columns per leg, leading old-query work is about 4nmr+2n²r; a conservative two-pass spelling is about 8nmq+8mrq+2n²q plus QR/lower-order work. Full reconstruction already needed by the baseline remains charged. At r256/q32 this can reduce the dominant contraction, but the entire available ordinary-product saving is only about 0.032B before discounts; roughly 0.02B net is a hypothesis, not a receipt. Thin P3 TT F^T stays exact. This is standard matrix range approximation applied to the consumed signed sum; the potential novelty is a useful challenge-specific operating point, not inventing randomized linear algebra.

Register an engineering cost screen of at least **5% complete counted-FLOP reduction** relative to the selected B package, with no new release-resource obstacle. Tiny exact-rank/operator identities must pass. Perform a paired full-shape label-free continuation audit in the cost job and report its differences; exact-query fidelity cannot be expected to improve when introducing approximation, so do not require a logically impossible zero-error improvement. Stop before labels for numerical instability, invalid state, or failure of the cost screen. Displacement is a diagnostic here, not a surrogate accuracy acceptance gate; the single paired truth comparison below answers whether the approximation is useful. This is a new prospective contract, not a relaxation of the completed S3/qEc contracts.

If feasible, run one paired P12 DESIGN accuracy comparison against the exact-query B package, gate1.05 on official-precision adjusted score and no new failed identities. Keep the original package if this fails. No second rank or sketch retry in this slice. This intervention changes covariance/newborn feedback through D21 even though source transport is retained; full recurrence is essential.

### D. Fit only the complete selected package, if there is a plausible release margin

The previous fit was a transfer initializer, not the optimum of the compressed program. A new bounded fit is permitted prospectively, not as continuation of the old experiment. Use the same four physically defined global controls and hard bounds; no per-network/public fitting, no per-layer grid and no generic learned residual model.

Reuse the reviewed two-pass coordinate schedule with its exact finite evaluation budget, centered on the selected package's frozen initializer and clipped to existing bounds. Before executing, specify step sizes, coordinate order, tie rule, ridge center, duplicate-point handling, training subset and within-fit validation subset in the committed contract. Use P12 first12/last12 with the explicit warning that all are exposed DESIGN. Save source, rows and predictions for every evaluation; matched default/selected validation only after training freeze, no validation-driven refit. If package cost is shape-constant, prove this once rather than unnecessarily rerunning unchanged cost probes, while still using the canonical scorer per arm.

Do not automatically pay for this fit: last uncompressed marginal steps bought less than1% each. Register the conditional trigger before B: continue only if an accepted integrated/query package improves over C00 and needs no more than **25% further adjusted-score reduction** to reach both same-panel half-v31 mean targets. Compute max(0,1-(mean_v31/2)/mean_candidate) separately for observed and official precision; both must be at most0.25 (equivalently candidate mean at most two-thirds of v31 mean). This is an engineering stop, not evidence that25% is achievable. Otherwise the current mechanism is too weak for this release objective; preserve it and stop the finite slice with a quantified gap. A new fitting mechanism then needs new evidence, not more coordinate passes.

After selecting/fixing the fit vector, assemble its complete24-row DESIGN evidence from an exact-source matching first12 evaluation and the frozen last12 evaluation, verifying all source/dependency/configuration/truth/numerical identities. If no matching first12 evaluation exists, allow exactly one preregistered first12 evaluation of the frozen source, with no refit. Final readiness and the25% comparisons use full24 rows; never substitute the better training or last12 subset. Existing old-fit eval13 and eval18 illustrate a source-identical split that can be combined without rerunning, but do not grant compatibility to a new implementation.

### E. Release or a decisive stop

Once a whole file is frozen, use P12 only for a candid major-release case against its actual v31 anchor. If the registered 2x DESIGN case is absent on either observed or official precision, preserve the file and evidence; do not spend P13. If present, pass one exact candidate to untouched P13 once under the unchanged dual release contract. Do not select another candidate on P13 or refit after seeing it.

After passing: candidate-specific official subprocess/certification, enforced Linux8GiB, read-only public CONFIRM, projection with its uncertainty, required preflight, independent Skeptic and exact-SHA submission approval. Required release checks cannot be replaced with a legacy-root check. Use the current provenance-specific server-FLOP offset in predict_public.py: the packet's blanket +2^24 statement is historical; current K3 subprocess projections use +2^25 while grader-twin accounting differs. No local absolute timing is server proof.

If the finite slice misses, report whether the blocker is recurrence accuracy, integration interaction, source/reconstruction cost, query approximation error, or deployment. Supply a next packet containing the full measured package and residual budget. Do not claim that all architecture avenues are exhausted, but do not open a speculative full-K4/radial/Tucker rewrite without a separately supported mechanism and cost model.

## Why this is a better use of the next session

The new information should come from actual complete candidates, not another round of interpreting proxy norms. A same-panel v31 anchor removes cross-panel threshold confusion. Matched P4 removes a cost confound. A four-arm package experiment reveals whether the two existing accuracy improvements cooperate after compression. A single query-only approximation has a defined savings ceiling. One conditional complete-package fit avoids fitting the wrong operating point again.

This is a credible, executable attempt at a major submission, not a guarantee of it. Current evidence does not contain a demonstrated winning configuration, and confidence cannot replace the missing measured accuracy/cost margin.
