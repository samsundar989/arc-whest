# Winning-plan review — September 18, 2026

## Decision

**Keep the mathematical family, change the research priority, and simplify the iteration path. We have not yet identified a demonstrated winning configuration.** The project has strong cost/accounting work and substantial negative evidence. Its best complete DESIGN candidate mostly makes the old accuracy cheaper. The missing major jump requires better accuracy as well as an economical implementation.

This review combines three parallel specialist repo/literature audits and direct read-only exploration through Samuel's Chrome window. A separate fresh-context Skeptic checks the synthesis. It is an implementation/research plan, not a pre-registration, new measurement, candidate promotion, contribution-prize claim, or permission to reopen a closed lane. No new seed, truth-array read, heavy job, estimator modification, upload, or designation occurred.

Samuel explicitly chose **individually approved diagnostic uploads while preserving the 2× major-release target** during this review. This prospective distinction supersedes treating 2× as a requirement for every diagnostic upload; it does not approve any particular file or change an existing experiment's gate. Before the first such upload, record its narrow contract and expected information, certify exact bytes, obtain the existing per-submission approval, and retain the required independent review.

Supporting reports: [competition snapshot](research/2026-09-18_deep_competition_snapshot.md), [architecture](research/2026-09-18_deep_architecture_audit.md), [validation and iteration](research/2026-09-18_deep_validation_audit.md), [primary literature](research/2026-09-18_deep_frontier_research.md). The [completed-slice packet](STRATEGIST_PACKET_2026-09-18_COMPLETE_SLICE.md) remains the raw-evidence map.

## Where we actually stand

| Reference | Adjusted score, lower is better | What it establishes |
|---|---:|---|
| Latest inspected clean leader, J2W #331502 | 2.158e-9 | Current public frontier scale; unknown private algorithm |
| Our latest graded v31 #331317 | 8.752e-9 | Actual public score; rank 64/172 in the inspected board |
| v31 on the same P12 DESIGN panel as C10 | 8.860665e-9 | Correct local comparator |
| C10, exact frozen complete export | 6.486609e-9 | Official-precision forecast on exposed DESIGN, not a submitted score |
| Half-v31 on that DESIGN panel | 4.430333e-9 | Intermediate 2× mean target, not a winning score |

The first two rows come from the [leader detail](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/331502) and [our detail](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/331317). The local rows use `experiments/2026-09-18_s17_factorial/decision.json`. Cross-panel ratios are planning comparisons, not paired evidence. Rank and frontier can change after the snapshot.

C10 improves the paired mean score by about 1.366×, chiefly by cutting counted FLOPs from 661.962 billion to 480.851 billion. Its raw MSE is 3.183714e-8 versus v31's 3.160759e-8, both against the common 2.248071e-9 truth-noise floor. The model of official-precision error is about 2.96645e-8. Even computing identical predictions at the minimum 0.1 score multiplier would yield about 2.96645e-9. **Cost reduction alone cannot catch the inspected 2.158e-9 frontier.** At that floor we still need about 27.25% less effective error; at C10's present cost we need about 66.73% less. These are counterfactual arithmetic, not forecast gains.

C10 still needs 31.700% lower official-precision adjusted score merely to reach the local half-v31 target. The rank 32 old-query trial reduced complete cost 5.514% but worsened all 24 pairs. Adding P3 to the fitted compressed package failed to repay its cost. Neither lane can be restarted with another unmotivated grid. P13 stays untouched until a frozen major-release case actually exists.

## The architecture to retain and the missing part

The supported backbone is full means and covariance, persistent factored third-cumulant history, an economical fourth-order model, internally consistent nonlinear births and feedback, shared storage for old history, and deterministic single-file flopscope execution. This has an appropriate balance between modeling each supplied network and avoiding an impossible full high-order tensor state. The direct research foundation is [Wu et al.](https://arxiv.org/html/2605.05179v2); its theory does not certify the live deep ReLU case or monotonic gains from more terms.

For C10 the concrete configuration is age 4/rank 256 Gram-based shared storage, PAIR+Q closure, P3 off, own controls `(0.625,1.225,-0.1,1.0)`, P4 and depth 5 products. Its old-query contractions are exact for its represented state. The complete model, births and compressed state are not exact cumulant propagation of the true network.

The possible winning extension is **protecting a small set of demonstrably important moments and recurrent effects while compressing only the remainder**. That is a design hypothesis. We have not identified a small protected set, proved affordable closure under subsequent updates, or measured a winning complete configuration. If no such set exists at acceptable cost, the family needs a genuinely different closure/representation; this review does not disguise that unknown as a solved architecture.

Do not replace the scientific family with Monte Carlo, a Gaussian-only recurrence, another raw tensor sketch, a generic adjoint, or a large learned correction on the evidence currently available. Existing scoped failures remain binding. Public K3 code confirms the family is competitive, but does not reveal the leading team's method. The new external low-rank literature supports preserving selected constraints through truncation and update; its PDE guarantees do not transfer to our prediction problem.

## Research program, with explicit exits

### A. One cheap, concrete hypothesis: remove P2 birth-diagonal leakage

CONFIRMED algebraic concern: truncating a zero-diagonal residual generally creates a nonzero diagonal. In C10 the residual is represented as `Z @ L.T`, while the newborn diagonal channel remains `sb=S3`. At birth the represented third-cumulant diagonal is therefore `S3 + rowsum(Z*L)`. The factor 3 is already in Z.

PROPOSED intervention: keep the same factors and set `sb=S3-rowsum(Z*L)`. This preserves the specified newborn diagonal using an existing channel, with no new history or rank. It costs roughly one elementwise product/reduction and vector subtraction per birth, negligible arithmetic relative to 480.851 billion FLOPs but still subject to exact metering and call/materialization checks. The [architecture report](research/2026-09-18_deep_architecture_audit.md) gives a tiny analytic counterexample and normalization checks.

This formula applies to exact C10's active ordinary branch. If later generalized to the inactive protected-feed branch, subtract only compressed-residual leakage from its already-compensated coefficient, or rebuild the coefficient from all returned factors once. Subtracting the combined diagonal again would double-compensate the exact protected factors. The fresh Skeptic independently checked the symmetry normalization and highlighted this implementation trap.

INFERRED rationale: [conservative low-rank integration](https://arxiv.org/abs/2206.09374) and [LoMaC](https://arxiv.org/abs/2207.00518) preserve selected observables separately from a truncated remainder. This is a useful method analogy, not a challenge performance theorem. It differs from the previously failed whole-tensor diagonal-protection scheme, so the eventual contract must explain the exact scope distinction.

SPECULATIVE benefit: a small accuracy improvement. Older P2 evidence suggests the entire rank 32 penalty was small; this is **not a credible standalone forecast of a major jump**. First independently verify tiny tensors, D3 and both D21 orientations, transport, normalization, zero/full-rank cases and active branch compensation. Then, under a new contract, test one fixed clone through the complete recurrence. Do not tune rank, damping or constants after seeing that comparison. Stop if the premise is already canceled elsewhere, if truth score worsens, or if the effect is too small to matter. Record an incremental success as incremental.

### B. The major-jump diagnostic: establish where recoverable error lives

Compare C10 with a diagnostic version of **the same closure** with storage truncations removed, including the P2 residual. The existing dense sentinel still approximates P2; it is not that reference. Use exact structured contractions or bounded streaming, not a materialized width 1024 full K3/K4 tensor. Derive the reference and conservative resource envelope before scheduling. If a feasible same-closure reference cannot be constructed, explicitly limit the claim to the operators that can be checked.

On one identical incoming state, change one birth or join, then let both versions continue with their own endogenous covariance, gates and future births. A supplementary frozen-feedback run can explain amplification but cannot establish actual prediction quality. Separate (i) newborn residual compression, (ii) shared-history joins, (iii) old-query approximation, and (iv) the independently defined K4/feed model. Approximation errors can compensate; a closer tensor or smaller local defect is not automatically better truth error.

The contract must freeze diagnostic locations, arms, cost model, observations, truth tier and paired arithmetic gate before any truth. Start with structural/no-truth checks; use only the smallest defensible truth allocation when the question requires final accuracy. Repeated DESIGN diagnostics must remain labeled exploratory. A promotion claim requires the fresh decision protocol. No fresh P13 touch is justified merely to learn which mechanism to investigate.

**Exit:** if eliminating compression produces no useful truth-error reduction, stop treating compression refinements as the major-jump program. If the more faithful chosen closure remains above the frontier even at the 0.1 multiplier, it supplies no demonstrated headroom for a compression-only winner. This is a practical decision criterion, not a mathematical lower bound: approximation can regularize. If a beneficial reference is too expensive to approximate economically, retain it only as a diagnostic.

### C. Conditional higher-upside implementation

Only after B identifies a harmful update, derive a small protected state and its next-step image. Preserve that effect through birth, linear carry, nonlinear feedback and readout; compress the remaining state. The [August2026 discrete conservation analysis](https://arxiv.org/html/2608.00397v1#S3.SS3) motivates testing whether protected quantities survive the actual update spaces. This is more specific than qE's failed immediate-query energy objective.

A cheap residual direction must be computable without an exact full tensor, dense online Jacobian, or expensive oracle. The complete birth/storage/feed/readout cost must fit the target, not just one contraction. If the protected set needs expanding dense state at each layer, reject the implementation before a truth sweep. A K4 residual or omitted feed remains conditional on independently derived evidence and the applicable closed-lane reopening rule; “carry more K4” is not a new justified mechanism.

There is no evidence-backed promise that A, B or C will win. Their value is a small falsifiable change followed by a diagnostic that can decide whether the retained family has enough accuracy headroom, instead of another succession of locally attractive approximations.

## Make iteration faster without removing useful evidence

The [validation audit](research/2026-09-18_deep_validation_audit.md) supports the core score/mint/precision protocol. It also found avoidable work: 18 completed September 18 preflight transcripts consumed 107.30 aggregate run-minutes; five suites run twice inside one standard preflight; one repeated C10 control evaluation took 18.64 minutes. Those are measured job/check durations, not official grader timing or promised elapsed-time savings.

Implement these bounded IMPORTANT changes before building a new general framework:

1. **Name the actual candidate.** A manifest distinguishes legacy root `estimator.py` v2.6.1, latest graded v31, and best DESIGN C10, binding exact hashes, composition, evidence tier, cost and certificate. Fail a release request that silently certifies the root file. The existing candidate certification is the starting point, not a replacement to discard.
2. **One numerical artifact, reusable across decisions.** Bind predictions to exact source, actual dependency closure, numeric/runtime configuration, ordered network identities, and truth provenance where scores are included. Do not invalidate predictions because a gate changes, an unrelated lab script changes, or a control changes role. Reuse numerical predictions only; timing and memory evidence is environment-specific. A producer bug or meaningful dependency change must invalidate reuse.
3. **One decision reader.** Bind a committed contract and explicit observed/official estimand to a numerical artifact. Keep paired arithmetic differences and the approved confidence intervals. Canonicalize Windows/snapshot paths, hash archived bytes without newline changes, preserve declared row order, and fail closed on source/config/identity/hash mismatches. Historical readers remain frozen evidence.
4. **One execution and release receipt.** Reserve SINGLE_USE access against a shared authoritative ledger before snapshot dispatch; do not rely on copied per-snapshot locks. Isolate temporary files between concurrent preflight runs. Pass structured completed-check receipts to the enforcement-label check so five suites are not reexecuted. Run the still-required full relevant gate once per coherent instrument change; do not simply omit it.
5. **One current runbook with explicit interfaces.** Gradually expose scientific state, birth/closure program, storage backend, contraction backend and single-file export. Replace brittle string substitutions one interface at a time with byte/semantic parity and a frozen fixture. Do not rewrite the working estimator and evaluator simultaneously. Point CURRENT_STATE to the current plan; preserve older plans as history.

For each change: written prediction, focused refusal/parity tests, required preflight, LOG knowledge delta, independent Skeptic. Tests should demonstrate a real prevented error or verified reuse, not mirror an implementation. Acceptance examples: gate-only change reuses identical predictions; source/truth/dependency change refuses reuse; duplicated reservations are rejected across snapshots; concurrent tests use separate temporary paths; all mandatory suites still produce one valid receipt. Measure the new critical path once after the change; no unearned speedup promise.

## What offline validation can and cannot establish

Keep canonical minted networks, grader-style f64 immutable weights, float32 prediction/loss semantics, official-package counting, exact candidate hashes, paired scores and explicit numerical truth floors. The stored v31 public forecast missed its realized public score by about 0.45%, a useful one-case check, not a universal error bar. P12 selection remains exposed DESIGN evidence.

Compute the **mean of per-network products**, including fallback behavior; do not multiply mean error by mean cost for variable-cost candidates. Precision corrections estimate expected label-noise differences, not the actual hidden-label cross-term or fresh-network uncertainty. FLOP offsets are report-path/provenance specific; the old universal offset wording must not override measured K3 evidence.

The in-process lifecycle evaluator does not enforce a real process deadline or Linux address-space limit merely because its configuration names them. The official subprocess path is the authority for local timing; Linux enforcement is needed for an actual 8 GiB process check. Server residual remains candidate-specific, and call count is only a forecast. Certification already measures scalar materializations—retain that safeguard. C10's 23,421 calls and modeled1.2745x residual headroom are not a cap certificate. Offline work can predict score and count work closely; it cannot exactly reproduce hidden networks, fresh truth or server scheduling.

## Submission cadence and release path

[Rules](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/challenge_rules) allow10 Phase 2 uploads per team per UTC day. We should use more of that capacity when each candidate answers a distinct question. A daily quota is capacity, not an optimization objective, and a public-board win is not the final fresh-rerun prize.

* **Diagnostic uploads, now allowed by Samuel:** pin one exact candidate and a server-only question, such as residual safety, Linux/process behavior or prediction/count agreement. C10 is a possible first subject once its local candidate certificate and narrow contract are complete; it has not been certified or approved here. Do not require a 2× accuracy gain from that diagnostic. Record failures and ledger details. Do not fit constants against repeated public results or present a diagnostic as a promoted major release.
* **Major releases:** retain the existing fresh paired dual2× case, clean contract/mini validation, independent review and per-file approval. The2× milestone is intermediate; a competitive final candidate should plausibly approach or beat the roughly 2.16e-9 frontier on independent evidence. Keep a competitive reliable fallback and a mechanistically distinct promising alternative if supported; Samuel alone handles designation.
* **Pipeline:** while one approved upload grades, work on independent derivation, cache/receipt fixes or the next already-registered experiment. Do not wait idly for server output and do not alter a dependent experiment after seeing public results without a new prospective contract. Maintain a submission manifest linking source hash, question, local prediction, measured result and unresolved caps.

Suggested order is operational, not a wall-clock promise: first the manifest/cache/preflight fixes and candidate certification; alongside that the tiny algebra for A and reference design for B; then one contracted mechanism comparison and one independently approved server diagnostic; then the smallest evidence-supported C implementation. No broad grid, automatic optimization loop, preemptive holdout spending or last-minute wholesale rewrite.

All heavy work must use workerq. Before submission, record the required optimization review covering duplicate inputs/loaders, workers, caching, batches/prefetch, unnecessary passes, recovery/checkpoints, device placement and reuse of valid artifacts. Prefer a measured or defensible 16–24 GiB host-RAM envelope; declare conservative RAM/VRAM/CPU needs and request more when the reference requires it. Do not lower precision, remove evidence or weaken recovery to reach that envelope. Saved small jobs above do not establish the full-reference peak.

The contribution-prize write-up remains stale/do-not-submit until its separate STRATEGIC claim review. New constrained-moment results would be valuable only if measured and independently supported; this research note itself supplies no validated novelty claim.

## Review status

The fresh-context [independent Skeptic ACCEPTS this research recommendation](research/2026-09-18_deep_strategy_skeptic.md), after independently recomputing saved scalar scores, checking the candidate hash and checking the P2 symmetry normalization. Its qualifications are binding on execution: establish reference feasibility, avoid protected-feed double compensation, freeze truth-dependent selection prospectively, verify actual cache dependencies and preserve the diagnostic/major-release distinction. It did not independently reproduce the Chrome ledger observation. This is acceptance of a research plan, not a winning candidate, submission or novelty claim. Experiment execution still starts with a new committed contract and Samuel's pre-result predictions.
