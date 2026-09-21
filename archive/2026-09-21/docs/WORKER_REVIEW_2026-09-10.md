# Worker review at bcca354 — September 10, 2026

## Decision and scope

**Do not execute the old pickup instruction to fit theta/lambda and then evaluate P01.** Two independent, fresh-context Skeptics reject the claim that the instrument is complete and the registered coherent-K4 experiment is ready. Retain the bounded repairs that passed; repair the production paths, derive the intended closure, and prove conformance without labels before fitting.

This is a review of worker commits through `bcca354`, following the earlier [competition/research audit](PHASE2_RESEARCH_AUDIT_2026-09-10.md). No estimator, instrument, gate, seed registry or protected corpus was changed; no protected truth or vault labels were read. Source, logs and existing result artifacts were inspected. A small synthetic reproducer and documentation were added. Historical outcomes are not reversed by this review; unsupported completion/cost claims are withdrawn from the active plan.

The worker correctly distinguished **missing memory evidence** from **evidence of failure**. However, its early return also suppresses explicit failures in legacy reports. Passing all 25 memory fixtures did not exercise that case. More negative mutations alone will not solve the problem: the next battery must traverse real producers and consumers with positive, negative, and incomplete evidence.

## What survives independent review

The instrument Skeptic independently passed memory binding **25/25**, truth-read/claim-race fixtures **27/27**, scorer **129/129**, certifier unit tests **26/26**, and the vault lifecycle's **sacrificial-only subset 30/30**. The last number is not a rerun of the worker's full 47-check suite. Identity alignment, ordinary-alpha handling, the unequal-multiplier noise formula, atomic claims, pre-read reservation in the new loader, and the vault's canonical lifecycle call survive those bounded checks.

The mathematical Skeptic accepts Stage 1's shared-basis transport identity, the stated reassociation, and the matrix-core diagonal identity. The stored **28/28** algebra fixtures support those limited claims. Stored three-seed `96 x 5` parity supports the generated exact control at that shape. These do not establish a coherent fourth-order closure or live-shape compact-program cost.

The standing L2 candidate remains untouched: `estimator_k3_strassen_L2.py`, submission **330566**, SHA256 `390bc226bbf517df0110facfc0dc91c78ef670e8a49368a536cabf630057d1db`. No promotion, submission or designation occurred.

## Blocking findings and required discriminators

Line references below describe **bcca354**, and may move after repairs. The [synthetic reproduction](../experiments/2026-09-10_worker_review/reproduce_review_findings.py) and [captured result](../experiments/2026-09-10_worker_review/reproduction.json) use no research labels.

### R1. Production certification does not implement its declared evidence domains

`local_sim/certify_candidate.py:131–147` maps sections `5`, `6`, `7`, `8`, while the real headings include `5-8` at line 694 and `11` at 1166. Both fall back to MATH. The combined cost section does not populate SERVER as intended; local runtime failures can appear to be mathematical failures. Unit fixtures manually supplied correct tags.

**Repair:** assign/check evidence domains at actual check creation, including mixed sections, and integration-test real `head()`/`check()` paths. Do not mechanically relabel an entire mixed section or weaken historical gates. Local observations, extrapolated server risk, and server measurements must remain distinct.

### R2. Memory provenance is incomplete, and the legacy correction discards negative evidence

`_certified_stack()` actually returns only `{'flopscope': '0.12.1+np2.2.6'}`. Both certifier line 327 and Linux worker line 159 rely on a nonexistent `whestbench.__version__`. The test fixture's populated stack masks this. Use installed distribution metadata consistently and require expected fields.

At certifier lines 380–393, version <3 returns before existing status/cap/control/prediction checks. A synthetic v2 report with `status=FAIL`, a `MemoryError`, unapplied cap, and surviving negative control yields **zero failed checks** and DEGRADED. Preserve applicable explicit negative evidence while treating unavailable new bindings as INCOMPLETE. Evidence from an unrelated workload cannot establish that the current candidate fails; record applicability separately.

The instrument identifier check at line 397 accepts any nonempty string; it does not bind the report to a known instrument. Validate provenance against a documented instrument version/hash policy. Replay the **new consumer** against all seven stored reports; recalculating their historical stored check lists, as U10 does, is not a consumer replay. Keep historical artifacts unchanged.

### R3. The memory test may measure its own retained networks

Linux worker lines 181–189 retain the original float32 mint and float64 input; lines 255–257 construct another network without releasing the first. Source accounting implies roughly **192 MiB** of extra network arrays at `1024 x 16`, versus the **64 MiB** growth threshold. This is an allocation analysis, not an observed live-shape peak. A stateless positive control must discriminate instrument retention from candidate retention. Release instrument-owned arrays consistently and define the repeated-worker lifecycle. Equal counted FLOPs across different weight realizations is not a universal rule for value-dependent estimators.

Run substantial memory controls through workerq. A Windows accounting estimate cannot certify an enforced Linux 8192-MiB limit.

### R4. The experiment cannot write evidence or reliably select its arms

`p2_lab.py:246` materializes a dict, but `_write:529` expects `z.files`. After fixing that, line 551 refers to `a`, which is local to `main`. Synthetic calls reproduce **AttributeError**, then **NameError**. The pickup command omits `--arms`, whose default excludes the requested `k3_exact` control. Generator documentation mentions `--fit`, but no fitting command exists.

Validate arms, control, output destination and complete configuration **before** protected truth is opened. Test a tiny synthetic end-to-end report, not just helper functions. Pass configuration explicitly. A fitting interface needs implementation and validation before a copy-ready command can name it.

### R5. The generated intervention is not the registered coherent K4 closure

`p2_build_k4_arms.py:218` reuses a scalar projection coefficient for a centered row-sum vector. The cited matrix-core projection is different. At width 1024, the local coefficient is approximately `2.8554459e-6`, versus the matrix-core coefficient `6/(n+4) = 0.0058365759`. This approximately 2,044-fold difference diagnoses an absent derivation; **it is not an instruction to rescale theta**.

The implementation also lacks the consistent metric conversion, D31 consumer, and explicit fourth-order-to-third-order birth factors. Indirect changes in existing K3 slices do not supply these missing families. The valid diagonal identity alone does not establish the complete closure. Compare the pinned [V17 generator](https://raw.githubusercontent.com/504aldo/whest-p2-cumulant-k3/18c17e2d7a9aeacd399cfc2c6b571e4e16dbfb45/generators/make_v17.py) with the local generator.

Define the represented tensor/polynomial, normalization, harmonic projection, transport, every consumed slice and induced birth. Then use an independently derived dense tiny oracle with signed, nonconstant cores, repeated indices, zero core, permutations and zero-variance limits. Do not test the implementation against a duplicate of its own formulas.

The LOG contract registers exact / coherent / consumers-only / births-only. The generator emits exact / coherent / **no-carry**. Clearing a carried vector is not switching off induced K3 births. The substitute `arm_k3_k4nobirth.py:797` also fails syntax parsing because an inline comment consumes a closing parenthesis. Emit and parse the actual four arms; verify both independent switches and zero-core nesting. Make a counted-cost parity mismatch fail the check rather than merely print it.

The placeholder smoke tested a different intervention. It neither kills nor validates coherent K4. Do not fit around a missing derivation. The asserted exact added cost of `145,977,344` is not recoverable from the stored rounded smoke log; recapture exact metered evidence for the corrected program before claiming that precision.

### R6. P01 is already exposed, and noise-aware reporting is unwired

The truth-read registry records **three completed P01 reads**. P01 is reusable DESIGN, so this is not a SINGLE_USE cap violation. It does violate the stronger contract promise to freeze before inspecting P01. The smoke was used to suggest a fitting range. Preserve that exposure and treat P01 honestly as design data.

The manually supplied read timestamps (`18:00Z`, `18:05Z`) precede contract commit `08f787b` (`19:47:03Z`). They may be inaccurate; they cannot prove preregistration. Reconcile with independent job records if available, record uncertainty otherwise, and never rewrite timestamps to manufacture compliance. Existing predictions concern the registered intervention, not the substitute. Amend the contract transparently and obtain new predictions if its material question changes, before new results.

`noise_corrected_compare` has no production caller. The lab still stores universal-cancellation language and supplies **per-replica** `gt_noise` while scoring against the average of two independent replicas. That average's noise floor is half the per-replica floor. Wire observed score, true-function estimate and official-precision forecast separately, preserving IDs, multipliers, truth precision and noise uncertainty. A helper-only test cannot validate the artifact path.

The declared build, kill and half-arm mechanism conditions also need separate executable outcomes; failure to pass a build gate does not itself establish a kill. Preserve `competition_score.decide(gate=...)` as promotion authority; protocol changes require their own authorization/contract.

The direct load at `p2_corrector.py:357` bypasses the new truth-read registry. Route protected consumers through the shared loader before describing access counting as enforced. P08 remains spent; no relabel/reset. The dormant vault still checks its cap after loading and records a touch after success: repair reservation/crash/concurrency semantics on sacrificial fixtures before any future Phase-2 enablement. The current Phase-1 refusal prevents that live path today.

### R7. The cost screen prices a different algorithm

`p2_compact_reference.py:477` sets new-column count to width and prices an `(r+n)` eigendecomposition: 1408 dimensions at rank 384. Its executable join uses dense SVD. The published program instead uses a weighted Gram sketch, thin QR, factor rotations and a carried Gram core; the local ledger also misses its two shared dense legs. See pinned [V21 generator](https://raw.githubusercontent.com/504aldo/whest-p2-cumulant-k3/18c17e2d7a9aeacd399cfc2c6b571e4e16dbfb45/generators/make_v21.py).

The dense baseline counts generations and final-layer D21 differently from the actual chain. Therefore the published-configuration cost, unsupported-primitive verdict, and claimed calibrated upper bound do not follow. Retain `ledger.json` as a **hypothetical algorithm estimate**, not evidence against compact all-history propagation. Price the algorithm actually implemented, including first/final-layer shortcuts, scratch memory and supported operations, then match its operation ledger to execution.

Additional bounded repair: Student-t inversion silently caps at 10,000. At df=1 and alpha=1e-6 the quantile should be about 636,619.77. Implement valid extreme-tail handling or explicitly reject unsupported inputs; ordinary tested alphas passed.

## Architecture and frontier assessment

**Keep the direction; correct the implementation and evidence.** Weight-conditioned means/covariance with persistent K3 history remains a strong reference. The next research architecture adds a jointly derived cheap K4 core, shared coordinates that retain all old sources, accurate recent sources, and a final mean-only path. Distinguish exact reassociation from low-rank approximation and dropped terms. A small legal submission kernel should be exported from an explicit reference, with conformance tests between them.

The most informative new evidence is executable public structure, not proof of what the leaders do. The [primary literature review](research/2026-09-10_literature_review.md) covers the directly relevant cumulant paper and compression/sketching literature, with limitations and falsifiers. It remains current for the pinned mechanisms inspected here. Further searches and the live Chrome discussion check did not reveal a new verified leader algorithm; this is not proof none exists.

The current Chrome Phase-2 board still shows our best around `1.84e-8` and the leader around `2.70e-9`, approximately a **6.8-fold** descriptive gap. Earlier exact report values remain in the [source ledger](research/2026-09-10_competition_sources.md). Official reference-noise floors are unavailable in those displayed scores. Improving accuracy while retaining the expensive tower will not necessarily close that gap: cost reduction and useful higher-order information must be assessed jointly. Public results do not identify a winning architecture for us.

After a validated compact baseline, the best bounded extension remains **sketching quadratic old-source products before dense reconstruction**, followed by an output-sensitive shared basis. Their evidence is mathematical plausibility from primary literature, not demonstrated challenge gains. Require rolling-update/variance/contraction proofs and complete cost accounting before labels. Do not revive failed scalar readout fits or cap old generations under a new name.

**Samuel's standing decision remains: hold eligibility work off entirely.** Stage 3 is closed by his instruction; do not draft/send a new organizer question or advance a participant-derived reconstruction to submission. Historical clarification text is inactive. This review does not infer that a mathematical rewrite makes a derivative eligible.

## Execution state and completion criteria

The default preflight was queued as **workerq 574**, 8 GiB RAM / 0 VRAM / 2 CPU, no GPU, and remained QUEUED at 20:29 UTC. It is **not `-Full`** and has **no source snapshot**. Record actual executing source before attributing a result; cancel/requeue this owned job with frozen provenance if needed, without duplicating it. Do not bypass resource contention. The worker's full mutation process was present earlier in review and absent by 20:30 UTC, but no terminal result or exit status was captured. Process exit does not establish success. No claim of post-repair preflight or full-battery success is made.

The review is complete when the active plan and handoff reflect these blockers. The **implementation** is complete only after production-path fixtures, independent core conformance, executable full-cost evidence, required preflight and independent Skeptic acceptance exist. Subsequent research decisions require valid contracts and labels under the existing protocol. A pending queue, failed hypothesis, human-owned prerequisite or Stage-3 hold must be reported honestly, never renamed success.

Active continuation: [GAME_PLAN](GAME_PLAN.md) and [Codex execution handoff](CODEX_EXECUTION_HANDOFF.md). The pre-review [CURRENT_STATE](archive/CURRENT_STATE_before_worker_review_2026-09-10.md) and [GAME_PLAN](archive/GAME_PLAN_before_worker_review_2026-09-10.md) are archived verbatim.
